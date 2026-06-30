"""
Continual learning methods: Naive, Freeze, Replay, Anchor-AVR.
"""

import copy
import random
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset

from config import MethodConfig, TrainConfig


# ══════════════════════════════════════════════
# Anchor Storage & Verification
# ══════════════════════════════════════════════

class AnchorStore:
    """
    Stores compressed representation anchors for a phase.
    
    After training on a phase, we save:
    - probe_input_ids: a small set of input sequences (tiny storage)
    - hidden_states: the model's hidden state at each layer for these probes
    
    These anchors are NOT the training data. They're compressed snapshots
    of the model's internal state — like remembering the shape of what
    you learned, not the words themselves.
    """
    
    def __init__(self):
        self.anchors = {}  # phase_key -> anchor data
    
    def save_anchors(
        self,
        model: GPT2LMHeadModel,
        phase_key: str,
        train_dataset,
        n_probes: int = 200,
        device: str = "cuda",
    ):
        """Save anchor representations after training on a phase."""
        model.eval()
        
        # Select probe sequences from the training data
        n_available = len(train_dataset)
        n_probes = min(n_probes, n_available)
        probe_indices = random.sample(range(n_available), n_probes)
        
        # Collect probe inputs
        probe_input_ids_list = []
        for idx in probe_indices:
            sample = train_dataset[idx]
            probe_input_ids_list.append(sample["input_ids"])
        
        probe_input_ids = torch.stack(probe_input_ids_list).to(device)
        
        # Run through model and save hidden states
        with torch.no_grad():
            outputs = model(input_ids=probe_input_ids, output_hidden_states=True)
        
        # Store compressed representations: mean hidden state per sequence per layer
        hidden_states = {}
        for layer_idx, hs in enumerate(outputs.hidden_states):
            # Compress: average over sequence length → (n_probes, n_embd)
            hidden_states[layer_idx] = hs.mean(dim=1).cpu()
        
        self.anchors[phase_key] = {
            "probe_input_ids": probe_input_ids.cpu(),
            "hidden_states": hidden_states,
            "n_probes": n_probes,
        }
        
        # Calculate storage size
        total_params = sum(
            v.numel() for v in hidden_states.values()
        ) + probe_input_ids.numel()
        storage_kb = total_params * 4 / 1024  # 4 bytes per float32
        
        print(f"  Saved {n_probes} anchors for Phase {phase_key} "
              f"({len(hidden_states)} layers, ~{storage_kb:.1f}KB)")
        
        model.train()
    
    def compute_anchor_loss(
        self,
        model: GPT2LMHeadModel,
        phase_keys: List[str],
        device: str = "cuda",
        anchor_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute anchor-pull loss: MSE between current and anchor hidden states.
        
        This is the core of the self-correction mechanism.
        The model checks: "have my representations drifted from when I 'knew' this?"
        If yes, the gradient from this loss pulls them back.
        
        Args:
            model: The current model
            phase_keys: Which phases' anchors to check
            device: Device
            anchor_layers: Which layers to include (None = all)
        
        Returns:
            (loss, drift_report) where drift_report has per-phase, per-layer drift
        """
        total_loss = torch.tensor(0.0, device=device)
        drift_report = {}
        
        for phase_key in phase_keys:
            if phase_key not in self.anchors:
                continue
            
            anchor_data = self.anchors[phase_key]
            probe_input_ids = anchor_data["probe_input_ids"].to(device)
            
            # Run current model on the same probe inputs
            outputs = model(input_ids=probe_input_ids, output_hidden_states=True)
            
            phase_loss = torch.tensor(0.0, device=device)
            phase_drift = {}
            n_layers = 0
            
            for layer_idx, current_hs in enumerate(outputs.hidden_states):
                if anchor_layers is not None and layer_idx not in anchor_layers:
                    continue
                
                # Compress current hidden state same way
                current_mean = current_hs.mean(dim=1)
                anchor_mean = anchor_data["hidden_states"][layer_idx].to(device)
                
                # MSE drift
                drift = F.mse_loss(current_mean, anchor_mean)
                phase_loss = phase_loss + drift
                phase_drift[layer_idx] = drift.item()
                n_layers += 1
            
            if n_layers > 0:
                phase_loss = phase_loss / n_layers  # Average over layers
            
            total_loss = total_loss + phase_loss
            drift_report[phase_key] = phase_drift
        
        return total_loss, drift_report
    
    def verify(
        self,
        model: GPT2LMHeadModel,
        phase_keys: List[str],
        threshold: float = 0.1,
        device: str = "cuda",
    ) -> Tuple[Dict[str, Dict[int, float]], List[str], List[str]]:
        """
        Verify which phases/layers have drifted beyond threshold.
        
        Returns:
            (drift_report, degraded_phases, degraded_layers)
        """
        _, drift_report = self.compute_anchor_loss(model, phase_keys, device)
        
        degraded_phases = []
        degraded_layers = []
        
        for phase_key, layer_drifts in drift_report.items():
            for layer_idx, drift_val in layer_drifts.items():
                if drift_val > threshold:
                    degraded_phases.append(phase_key)
                    degraded_layers.append(f"{phase_key}_L{layer_idx}")
        
        return drift_report, degraded_phases, degraded_layers
    
    def get_storage_size_kb(self) -> float:
        """Total storage used by all anchors."""
        total = 0
        for phase_data in self.anchors.values():
            total += phase_data["probe_input_ids"].numel()
            for hs in phase_data["hidden_states"].values():
                total += hs.numel()
        return total * 4 / 1024  # float32


# ══════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════

class ReplayBuffer:
    """Stores a percentage of old training data for replay."""
    
    def __init__(self, buffer_pct: float = 0.01):
        self.buffer_pct = buffer_pct
        self.buffers = {}  # phase_key -> list of (input_ids, labels)
        self.total_items = 0
    
    def add_phase(
        self,
        phase_key: str,
        train_dataset,
        n_total: int = None,
    ):
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
        
        # Storage estimate
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
        
        # Sample with replacement
        batch = random.choices(all_samples, k=batch_size)
        
        input_ids = torch.stack([s["input_ids"] for s in batch])
        labels = torch.stack([s["labels"] for s in batch])
        
        return {"input_ids": input_ids, "labels": labels}
    
    def get_storage_size_kb(self) -> float:
        """Total storage used by buffer."""
        total = 0
        for samples in self.buffers.values():
            for s in samples:
                total += s["input_ids"].numel() + s["labels"].numel()
        return total * 4 / 1024


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
        """Compute the training loss for a batch. Override in subclasses."""
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        return outputs.loss, {}


class NaiveMethod(CLMethod):
    """Naive sequential SGD — no protection. The disaster baseline."""
    
    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "naive"


class FreezeMethod(CLMethod):
    """Freeze bottom N layers after Phase A."""
    
    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "freeze"
        self.frozen = False
    
    def on_phase_start(self, model, phase_key, phases_data):
        """Freeze bottom layers after the first phase."""
        if not self.frozen and len(self.completed_phases) > 0:
            from model import freeze_bottom_layers
            freeze_bottom_layers(model, self.config.n_freeze_layers)
            self.frozen = True
    
    def compute_loss(self, model, batch, device):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        return outputs.loss, {}


class ReplayMethod(CLMethod):
    """Blind replay — store 1% of old data, mix into training."""
    
    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "replay"
        self.buffer = ReplayBuffer(buffer_pct=config.replay_buffer_pct)
    
    def on_phase_end(self, model, phase_key, train_dataset):
        """Add completed phase data to replay buffer."""
        self.buffer.add_phase(phase_key, train_dataset)
        super().on_phase_end(model, phase_key, train_dataset)
    
    def compute_loss(self, model, batch, device):
        """Standard LM loss + replay loss."""
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
        if replay_batch is not None:
            replay_outputs = model(
                input_ids=replay_batch["input_ids"].to(device),
                labels=replay_batch["labels"].to(device),
            )
            total_loss = total_loss + replay_outputs.loss
        
        return total_loss, {"replay_loss": replay_outputs.loss.item() if replay_batch is not None else 0}


class AnchorAVRContinuous(CLMethod):
    """
    Anchor-AVR (Continuous mode).
    
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
        """Save anchor snapshots after completing a phase."""
        self.anchor_store.save_anchors(
            model, phase_key, train_dataset,
            n_probes=self.config.n_anchor_probes,
            device=self.train_config.device,
        )
        super().on_phase_end(model, phase_key, train_dataset)
    
    def compute_loss(self, model, batch, device):
        """LM loss + periodic anchor-pull loss."""
        self.step_count += 1
        
        # Standard LM loss (always computed, but only need hidden states
        # when we're also computing anchor loss)
        need_hidden = self.completed_phases and (self.step_count % self.config.anchor_freq == 0)
        
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=need_hidden,
        )
        lm_loss = outputs.loss
        
        # Anchor-pull loss (computed every anchor_freq steps to save compute)
        anchor_loss = torch.tensor(0.0, device=device)
        drift_report = {}
        
        if need_hidden and self.completed_phases:
            anchor_loss, drift_report = self.anchor_store.compute_anchor_loss(
                model,
                phase_keys=self.completed_phases,
                device=device,
                anchor_layers=self.config.anchor_layers,
            )
            # Free GPU memory from anchor computation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_loss = lm_loss + self.config.anchor_loss_weight * anchor_loss
        
        metrics = {
            "lm_loss": lm_loss.item(),
            "anchor_loss": anchor_loss.item(),
            "total_loss": total_loss.item(),
        }
        # Add per-phase drift
        for phase_key, layer_drifts in drift_report.items():
            avg_drift = sum(layer_drifts.values()) / max(len(layer_drifts), 1)
            metrics[f"drift_{phase_key}"] = avg_drift
        
        return total_loss, metrics


class AnchorAVRDiscrete(CLMethod):
    """
    Anchor-AVR (Discrete mode).
    
    Implements the full Absorb-Verify-Repair loop:
    1. ABSORB: Train on new phase normally
    2. VERIFY: Periodically check if old representations have drifted
    3. REPAIR: If drift detected, do targeted anchor-pull on degraded layers only
    """
    
    def __init__(self, config: MethodConfig, train_config: TrainConfig):
        super().__init__(config, train_config)
        self.name = "anchor_disc"
        self.anchor_store = AnchorStore()
        self.step_count = 0
        self.total_repairs = 0
    
    def on_phase_end(self, model, phase_key, train_dataset):
        """Save anchor snapshots after completing a phase."""
        self.anchor_store.save_anchors(
            model, phase_key, train_dataset,
            n_probes=self.config.n_anchor_probes,
            device=self.train_config.device,
        )
        super().on_phase_end(model, phase_key, train_dataset)
    
    def compute_loss(self, model, batch, device):
        """
        During training on new phase, only use LM loss.
        Verification and repair happen separately via verify_and_repair().
        """
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
        The core Verify → Repair loop.
        
        1. Run probe inputs through model
        2. Check which layers have drifted beyond threshold
        3. If any degraded: run targeted anchor-pull repair steps
        4. Re-verify that repair worked
        """
        drift_report, degraded_phases, degraded_layers = self.anchor_store.verify(
            model,
            phase_keys=self.completed_phases,
            threshold=self.config.drift_threshold,
            device=device,
        )
        
        if not degraded_phases:
            return drift_report  # Everything is fine, no repair needed
        
        # REPAIR: targeted anchor-pull on degraded representations
        print(f"  ⚠ Drift detected in {degraded_phases} — running repair ({self.config.repair_steps} steps)")
        self.total_repairs += 1
        
        # Only optimize parameters that affect the drifted layers
        # For simplicity, we optimize all params but with anchor-pull loss only
        repair_optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.repair_lr
        )
        
        for repair_step in range(self.config.repair_steps):
            anchor_loss, _ = self.anchor_store.compute_anchor_loss(
                model,
                phase_keys=list(set(degraded_phases)),  # Unique phases
                device=device,
            )
            
            repair_optimizer.zero_grad()
            anchor_loss.backward()
            repair_optimizer.step()
        
        # RE-VERIFY: check if repair worked
        new_drift, new_degraded, _ = self.anchor_store.verify(
            model,
            phase_keys=self.completed_phases,
            threshold=self.config.drift_threshold,
            device=device,
        )
        
        repair_worked = len(new_degraded) == 0
        if repair_worked:
            print(f"  ✅ Repair successful — drift back below threshold")
        else:
            print(f"  ⚠ Partial repair — {new_degraded} still degraded")
        
        # Free GPU memory after repair
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return new_drift


# ══════════════════════════════════════════════
# Method Factory
# ══════════════════════════════════════════════

def create_method(method_name: str, method_config: MethodConfig, train_config: TrainConfig) -> CLMethod:
    """Create a CL method by name."""
    methods = {
        "naive": NaiveMethod,
        "freeze": FreezeMethod,
        "replay": ReplayMethod,
        "anchor_cont": AnchorAVRContinuous,
        "anchor_disc": AnchorAVRDiscrete,
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")
    
    return methods[method_name](method_config, train_config)
