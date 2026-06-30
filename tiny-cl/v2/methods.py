"""
V2 Continual Learning Methods: Naive LoRA, Replay, EWC, Anchor-AVR.
All methods use LoRA fine-tuning on a frozen pretrained base model.
"""

import copy
import random
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from config import MethodConfig, TrainConfig
from anchors import AnchorStore, WeightDeltaAnchorStore


# ══════════════════════════════════════════════
# Replay Buffer (same as V1 but adapted for LoRA)
# ══════════════════════════════════════════════

class ReplayBuffer:
    """Stores a percentage of old training data for replay."""

    def __init__(self, buffer_pct: float = 0.01):
        self.buffer_pct = buffer_pct
        self.buffers = {}  # phase_key -> list of samples
        self.total_items = 0

    def add_phase(self, phase_key: str, train_dataset):
        """Add samples from a completed phase to the buffer."""
        n_available = len(train_dataset)
        n_to_store = max(1, int(n_available * self.buffer_pct))
        n_to_store = min(n_to_store, n_available)

        indices = random.sample(range(n_available), n_to_store)

        samples = []
        for idx in indices:
            sample = train_dataset[idx]
            samples.append({
                "input_ids": sample["input_ids"],
                "labels": sample["labels"],
            })

        self.buffers[phase_key] = samples
        self.total_items += len(samples)

        storage_kb = sum(
            s["input_ids"].numel() + s["labels"].numel()
            for s in samples
        ) * 4 / 1024
        print(f"  Replay buffer: {len(samples)} samples from Phase {phase_key} (~{storage_kb:.1f}KB)")

    def get_replay_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get a mixed batch from all stored phases."""
        if self.total_items == 0:
            return None

        all_samples = []
        for samples in self.buffers.values():
            all_samples.extend(samples)

        batch = random.choices(all_samples, k=batch_size)

        input_ids = torch.stack([s["input_ids"] for s in batch])
        labels = torch.stack([s["labels"] for s in batch])

        return {"input_ids": input_ids, "labels": labels}

    def get_storage_size_kb(self) -> float:
        total = 0
        for samples in self.buffers.values():
            for s in samples:
                total += s["input_ids"].numel() + s["labels"].numel()
        return total * 4 / 1024


# ══════════════════════════════════════════════
# EWC: Elastic Weight Consolidation on LoRA params
# ══════════════════════════════════════════════

class EWCStore:
    """Stores Fisher information matrix for EWC on LoRA parameters."""

    def __init__(self):
        self.fisher = {}       # phase_key -> {param_name: fisher_diag}
        self.opt_params = {}   # phase_key -> {param_name: param_value}

    def compute_fisher(
        self,
        model,
        dataloader: DataLoader,
        phase_key: str,
        n_samples: int = 200,
        device: str = "cuda",
    ):
        """Compute diagonal Fisher information for LoRA parameters."""
        model.eval()

        # Initialize Fisher dict for trainable (LoRA) params
        fisher_dict = {}
        opt_params_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
                opt_params_dict[name] = param.data.clone()

        # Accumulate Fisher information
        n_processed = 0
        for batch in dataloader:
            if n_processed >= n_samples:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2) / n_samples

            n_processed += input_ids.size(0)

        self.fisher[phase_key] = fisher_dict
        self.opt_params[phase_key] = opt_params_dict

        n_params = sum(v.numel() for v in fisher_dict.values())
        print(f"  EWC Fisher computed for Phase {phase_key} "
              f"({len(fisher_dict)} params, ~{n_params * 4 / 1024:.1f}KB)")

        model.train()

    def compute_ewc_loss(
        self,
        model,
        phase_keys: List[str],
        device: str = "cuda",
    ) -> torch.Tensor:
        """Compute EWC penalty: sum of Fisher-weighted squared deviations."""
        total_loss = torch.tensor(0.0, device=device)

        for phase_key in phase_keys:
            if phase_key not in self.fisher:
                continue

            for name, param in model.named_parameters():
                if param.requires_grad and name in self.fisher[phase_key]:
                    fisher = self.fisher[phase_key][name].to(device)
                    opt_param = self.opt_params[phase_key][name].to(device)
                    total_loss = total_loss + (fisher * (param - opt_param).pow(2)).sum()

        return total_loss


# ══════════════════════════════════════════════
# Method Wrappers
# ══════════════════════════════════════════════

class CLMethod:
    """Base class for continual learning methods."""

    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        self.config = config
        self.train_config = train_config
        self.completed_phases = []

    def on_phase_start(self, model, phase_key, phases_data):
        """Called before training on a new phase."""
        pass

    def on_phase_end(self, model, phase_key, train_dataset):
        """Called after training on a phase completes."""
        self.completed_phases.append(phase_key)

    def compute_loss(self, model, batch, device):
        """Compute the training loss for a batch."""
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        return outputs.loss, {}

    def get_storage_kb(self) -> float:
        """Return storage overhead in KB."""
        return 0.0


class NaiveMethod(CLMethod):
    """Naive LoRA — sequential adapter training, no protection."""

    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "naive"


class ReplayMethod(CLMethod):
    """LoRA + Replay — store 1% of old data, mix into training."""

    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "replay"
        self.buffer = ReplayBuffer(buffer_pct=config.replay_buffer_pct)

    def on_phase_end(self, model, phase_key, train_dataset):
        self.buffer.add_phase(phase_key, train_dataset)
        super().on_phase_end(model, phase_key, train_dataset)

    def compute_loss(self, model, batch, device):
        # Current domain loss
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        total_loss = outputs.loss

        # Replay loss
        replay_batch = self.buffer.get_replay_batch(
            int(self.train_config.batch_size * self.config.replay_mix_ratio)
        )
        replay_loss_val = 0.0
        if replay_batch is not None:
            replay_outputs = model(
                input_ids=replay_batch["input_ids"].to(device),
                labels=replay_batch["labels"].to(device),
            )
            total_loss = total_loss + replay_outputs.loss
            replay_loss_val = replay_outputs.loss.item()

        return total_loss, {
            "lm_loss": outputs.loss.item(),
            "replay_loss": replay_loss_val,
        }

    def get_storage_kb(self) -> float:
        return self.buffer.get_storage_size_kb()


class EWCMethod(CLMethod):
    """LoRA + EWC — elastic weight consolidation on LoRA parameters."""

    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "ewc"
        self.ewc_store = EWCStore()

    def on_phase_end(self, model, phase_key, train_dataset):
        """Compute Fisher information after each phase."""
        # Create a temporary dataloader for Fisher computation
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.ewc_store.compute_fisher(
            model, dataloader, phase_key,
            n_samples=self.config.ewc_fisher_n_samples,
            device=self.train_config.device,
        )
        super().on_phase_end(model, phase_key, train_dataset)

    def compute_loss(self, model, batch, device):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        lm_loss = outputs.loss

        ewc_loss = torch.tensor(0.0, device=device)
        if self.completed_phases:
            ewc_loss = self.ewc_store.compute_ewc_loss(
                model, self.completed_phases, device=device
            )

        total_loss = lm_loss + self.config.ewc_lambda * ewc_loss

        return total_loss, {
            "lm_loss": lm_loss.item(),
            "ewc_loss": ewc_loss.item(),
        }


class AnchorAVRContinuous(CLMethod):
    """
    LoRA + Anchor-AVR (Continuous mode).

    The anchor-pull loss is computed every anchor_freq steps during training
    on new phases. The model continuously self-corrects by pulling drifted
    representations back toward the anchor snapshots.
    """

    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "anchor_cont"
        self.anchor_store = AnchorStore()
        self.step_count = 0

    def on_phase_end(self, model, phase_key, train_dataset):
        self.anchor_store.save_anchors(
            model, phase_key, train_dataset,
            n_probes=self.config.n_anchor_probes,
            device=self.train_config.device,
        )
        super().on_phase_end(model, phase_key, train_dataset)

    def compute_loss(self, model, batch, device):
        self.step_count += 1

        need_hidden = self.completed_phases and (self.step_count % self.config.anchor_freq == 0)

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=need_hidden,
        )
        lm_loss = outputs.loss

        anchor_loss = torch.tensor(0.0, device=device)
        drift_report = {}

        if need_hidden and self.completed_phases:
            anchor_loss, drift_report = self.anchor_store.compute_anchor_loss(
                model,
                phase_keys=self.completed_phases,
                device=device,
                anchor_layers=self.config.anchor_layers,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_loss = lm_loss + self.config.anchor_loss_weight * anchor_loss

        metrics = {
            "lm_loss": lm_loss.item(),
            "anchor_loss": anchor_loss.item(),
            "total_loss": total_loss.item(),
        }
        for phase_key, layer_drifts in drift_report.items():
            avg_drift = sum(layer_drifts.values()) / max(len(layer_drifts), 1)
            metrics[f"drift_{phase_key}"] = avg_drift

        return total_loss, metrics

    def get_storage_kb(self) -> float:
        return self.anchor_store.get_storage_size_kb()


class AnchorAVRDiscrete(CLMethod):
    """
    LoRA + Anchor-AVR (Discrete mode).

    Implements the full Absorb-Verify-Repair loop:
    1. ABSORB: Train on new phase normally
    2. VERIFY: Check hidden-state drift (no_grad) — detects WHAT forgot
    3. REPAIR: Pull LoRA weights back toward saved snapshots — fixes the forgetting
    """

    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "anchor_disc"
        self.anchor_store = AnchorStore()
        self.lora_snapshots = {}  # phase_key -> {param_name: tensor}
        self.step_count = 0
        self.total_repairs = 0

    def on_phase_end(self, model, phase_key, train_dataset):
        # Save hidden-state anchors for VERIFICATION
        self.anchor_store.save_anchors(
            model, phase_key, train_dataset,
            n_probes=self.config.n_anchor_probes,
            device=self.train_config.device,
        )
        # Save LoRA weight snapshot for REPAIR (tiny memory — just the adapter weights)
        lora_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_params[name] = param.data.cpu().clone()
        self.lora_snapshots[phase_key] = lora_params
        n_params = sum(v.numel() for v in lora_params.values())
        print(f"  Saved LoRA weight snapshot for Phase {phase_key} ({len(lora_params)} params, ~{n_params * 4 / 1024:.1f}KB)")
        super().on_phase_end(model, phase_key, train_dataset)

    def compute_loss(self, model, batch, device):
        """During training, only use LM loss. Verification and repair happen separately."""
        self.step_count += 1

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )

        return outputs.loss, {"lm_loss": outputs.loss.item()}

    def should_verify(self) -> bool:
        """Check if it's time to verify."""
        return (
            len(self.completed_phases) > 0 and
            self.step_count % self.config.verify_freq == 0
        )

    def verify_and_repair(self, model, device):
        """
        The core Verify -> Repair loop.

        1. VERIFY (no_grad): check hidden-state drift to DETECT what forgot
        2. REPAIR: pull LoRA weights toward saved snapshots — NO forward pass needed
        3. RE-VERIFY (no_grad): confirm repair worked
        """
        # VERIFY — no grad, just checking drift values
        with torch.no_grad():
            drift_report, degraded_phases, degraded_layers = self.anchor_store.verify(
                model,
                phase_keys=self.completed_phases,
                threshold=self.config.drift_threshold,
                device=device,
            )

        if not degraded_phases:
            return drift_report

        unique_degraded = list(set(degraded_phases))
        print(f"  [DRIFT] Detected in {unique_degraded} — running repair ({self.config.repair_steps} steps)")
        self.total_repairs += 1

        # REPAIR using LoRA weight snapshots — NO forward pass, NO OOM
        # Pull LoRA weights toward their values when the degraded phase was learned
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        repair_optimizer = torch.optim.Adam(trainable_params, lr=self.config.repair_lr)

        for repair_step in range(self.config.repair_steps):
            weight_loss = torch.tensor(0.0, device=device)
            for phase_key in unique_degraded:
                if phase_key not in self.lora_snapshots:
                    continue
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.lora_snapshots[phase_key]:
                        target = self.lora_snapshots[phase_key][name].to(device)
                        weight_loss = weight_loss + F.mse_loss(param, target)

            if weight_loss.requires_grad:
                repair_optimizer.zero_grad()
                weight_loss.backward()
                repair_optimizer.step()
            else:
                break  # Nothing to repair

        # RE-VERIFY — no grad
        with torch.no_grad():
            new_drift, new_degraded, _ = self.anchor_store.verify(
                model,
                phase_keys=self.completed_phases,
                threshold=self.config.drift_threshold,
                device=device,
            )

        repair_worked = len(new_degraded) == 0
        if repair_worked:
            print(f"  [REPAIR] Successful — drift back below threshold")
        else:
            print(f"  [REPAIR] Partial — {new_degraded} still degraded")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return new_drift

    def get_storage_kb(self) -> float:
        return self.anchor_store.get_storage_size_kb()


# ══════════════════════════════════════════════
# Method Factory
# ══════════════════════════════════════════════

def create_method(method_name: str, method_config: MethodConfig, train_config: TrainConfig) -> CLMethod:
    """Create a CL method by name."""
    methods = {
        "naive": NaiveMethod,
        "replay": ReplayMethod,
        "ewc": EWCMethod,
        "anchor_cont": AnchorAVRContinuous,
        "anchor_disc": AnchorAVRDiscrete,
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")

    return methods[method_name](method_config, train_config)
