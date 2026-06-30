"""
V4 Methods: Naive, AVR, EWC — adapted for streaming regime.

Key difference from V2: methods must handle variable increment sizes
and AVR must work with very few examples per increment.
"""

import copy
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from config import AVRConfig, EWCConfig, TrainConfig
from anchors import StreamingAnchorStore


class NaiveMethod:
    """Sequential training, no protection. The baseline."""
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.name = "naive"
        self.completed_phases = []
        self.extra_steps = 0  # Track compute overhead

    def on_phase_start(self, model, phase_key):
        pass

    def on_phase_end(self, model, phase_key, streaming_data, device):
        self.completed_phases.append(phase_key)

    def compute_loss(self, model, batch, device):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        return outputs.loss, {"lm_loss": outputs.loss.item()}


class AVRMethod:
    """
    Anchor-based Verify-Repair for streaming regime.
    
    Key adaptations:
    - Anchors built from 20 probes (not 50)
    - Repair uses weight snapshots (same as V2)
    - Verify every N increments (not every N steps)
    - Works with any increment size
    """
    def __init__(self, avr_config: AVRConfig, train_config: TrainConfig):
        self.avr_config = avr_config
        self.train_config = train_config
        self.name = "avr"
        self.anchor_store = StreamingAnchorStore()
        self.weight_snapshots = {}  # phase_key -> {param_name: tensor}
        self.completed_phases = []
        self.extra_steps = 0
        self.total_repairs = 0
        self.total_verifies = 0
        self.increment_count = 0

    def on_phase_start(self, model, phase_key):
        self.increment_count = 0

    def on_increment_end(self, model, phase_key, device):
        """Called after each increment. Check if we should verify."""
        self.increment_count += 1
        if (self.completed_phases and
            self.increment_count % self.avr_config.verify_every_n_increments == 0):
            self._verify_and_repair(model, device)

    def on_phase_end(self, model, phase_key, streaming_data, device):
        """Called after all increments for a phase are done."""
        # Save anchors
        self.anchor_store.save_anchors(
            model, phase_key, streaming_data,
            n_probes=self.avr_config.n_anchor_probes,
            device=device,
        )

        # Save weight snapshot for repair
        snapshot = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                snapshot[name] = param.data.cpu().clone()
        self.weight_snapshots[phase_key] = snapshot
        n_params = sum(v.numel() for v in snapshot.values())
        print(f"  Weight snapshot saved: Phase {phase_key} ({n_params:,} params)")

        # Final verify-repair after phase completes
        self._verify_and_repair(model, device)

        self.completed_phases.append(phase_key)

    def compute_loss(self, model, batch, device):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        return outputs.loss, {"lm_loss": outputs.loss.item()}

    def _verify_and_repair(self, model, device):
        """Core AVR: verify + repair if needed."""
        self.total_verifies += 1
        drift_report, degraded, needs_repair = self.anchor_store.verify(
            model,
            phase_keys=self.completed_phases,
            threshold=self.avr_config.drift_threshold,
            device=device,
        )

        # Log health
        for pk in self.completed_phases:
            health = self.anchor_store.health.get(pk, 1.0)
            status = "HEALTHY" if health > 0.85 else "DEGRADED"
            print(f"  [VERIFY] {pk}: health={health:.3f} [{status}]")

        if not needs_repair:
            return

        # REPAIR: pull weights toward saved snapshot
        self.total_repairs += 1
        unique_degraded = list(set(degraded))
        print(f"  [REPAIR] Drift detected in {unique_degraded} — repairing ({self.avr_config.repair_steps} steps)")

        trainable = [p for p in model.parameters() if p.requires_grad]
        repair_opt = torch.optim.Adam(trainable, lr=self.avr_config.repair_lr)

        for step in range(self.avr_config.repair_steps):
            weight_loss = torch.tensor(0.0, device=device)
            n_targets = 0

            for phase_key in unique_degraded:
                if phase_key not in self.weight_snapshots:
                    continue
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.weight_snapshots[phase_key]:
                        target = self.weight_snapshots[phase_key][name].to(device)
                        weight_loss = weight_loss + F.mse_loss(param, target)
                        n_targets += 1

            if n_targets > 0 and weight_loss.requires_grad:
                weight_loss = weight_loss / n_targets
                repair_opt.zero_grad()
                weight_loss.backward()
                repair_opt.step()
                self.extra_steps += 1
            else:
                break

        # Re-verify
        _, new_degraded, _ = self.anchor_store.verify(
            model,
            phase_keys=self.completed_phases,
            threshold=self.avr_config.drift_threshold,
            device=device,
        )
        if len(new_degraded) == 0:
            print(f"  [REPAIR] Successful")
        else:
            print(f"  [REPAIR] Partial — {new_degraded} still degraded")

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def get_storage_kb(self):
        return self.anchor_store.get_storage_kb()


class EWCMethod:
    """
    EWC for streaming regime.
    
    IMPORTANT: EWC requires a full pass over task data to estimate Fisher.
    At small increment sizes (20-100 examples), Fisher estimation is unstable
    or impossible. This method will FLAG when EWC cannot be computed.
    """
    def __init__(self, ewc_config: EWCConfig, train_config: TrainConfig):
        self.ewc_config = ewc_config
        self.train_config = train_config
        self.name = "ewc"
        self.completed_phases = []
        self.extra_steps = 0
        self.fisher = {}          # phase_key -> {param_name: fisher_diag}
        self.opt_params = {}      # phase_key -> {param_name: param_value}
        self.ewc_computable = {}  # phase_key -> bool (was Fisher stable?)

    def on_phase_start(self, model, phase_key):
        pass

    def on_phase_end(self, model, phase_key, streaming_data, device):
        """Compute Fisher information after phase completes."""
        full_dataset = streaming_data.get_full_dataset()
        n_available = len(full_dataset)
        n_samples = min(self.ewc_config.fisher_n_samples, n_available)

        # Flag if Fisher can't be reliably estimated
        if n_available < 50:
            print(f"  [EWC] WARNING: Only {n_available} examples — Fisher estimate unreliable")
            self.ewc_computable[phase_key] = False
            self.completed_phases.append(phase_key)
            return
        else:
            self.ewc_computable[phase_key] = True

        print(f"  [EWC] Computing Fisher from {n_samples} samples...")

        model.eval()
        fisher_dict = {}
        opt_params_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)
                opt_params_dict[name] = param.data.clone()

        loader = DataLoader(full_dataset, batch_size=16, shuffle=True, drop_last=False)
        n_processed = 0
        for batch in loader:
            if n_processed >= n_samples:
                break
            model.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2) / n_samples
            n_processed += batch["input_ids"].size(0)

        self.fisher[phase_key] = fisher_dict
        self.opt_params[phase_key] = opt_params_dict
        self.completed_phases.append(phase_key)
        model.train()

    def compute_loss(self, model, batch, device):
        # LM loss
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        lm_loss = outputs.loss

        # EWC penalty
        ewc_loss = torch.tensor(0.0, device=device)
        if self.completed_phases:
            for phase_key in self.completed_phases:
                if not self.ewc_computable.get(phase_key, False):
                    continue
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.fisher.get(phase_key, {}):
                        fisher = self.fisher[phase_key][name].to(device)
                        opt = self.opt_params[phase_key][name].to(device)
                        ewc_loss = ewc_loss + (fisher * (param - opt).pow(2)).sum()

        total_loss = lm_loss + self.ewc_config.lambda_ * ewc_loss
        return total_loss, {
            "lm_loss": lm_loss.item(),
            "ewc_loss": ewc_loss.item(),
        }


from torch.utils.data import DataLoader


def create_method(method_name: str, train_config: TrainConfig,
                  avr_config: AVRConfig = None, ewc_config: EWCConfig = None):
    if method_name == "naive":
        return NaiveMethod(train_config)
    elif method_name == "avr":
        return AVRMethod(avr_config or AVRConfig(), train_config)
    elif method_name == "ewc":
        return EWCMethod(ewc_config or EWCConfig(), train_config)
    else:
        raise ValueError(f"Unknown method: {method_name}")
