"""
V2 Anchor Store: Representation anchoring for LoRA-adapted pretrained models.

Key difference from V1: The base model is frozen. Only LoRA adapters change.
Anchors track hidden states with LoRA adapters active, ensuring that adapter
training on new domains doesn't corrupt representations from old domains.
"""

import random
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class AnchorStore:
    """
    Stores compressed representation anchors for each completed phase.

    After training on a phase, we:
    1. Select probe sequences from the training data
    2. Run them through the model (with current LoRA adapters active)
    3. Save mean hidden states per layer as anchor snapshots

    These anchors are NOT training data — they're compressed representations
    of what the model "knows" about each domain.
    """

    def __init__(self):
        self.anchors = {}  # phase_key -> anchor data

    def save_anchors(
        self,
        model,  # PeftModel (base + LoRA)
        phase_key: str,
        train_dataset,
        n_probes: int = 50,
        device: str = "cuda",
    ):
        """Save anchor representations after training on a phase."""
        model.eval()

        # Select probe sequences from training data
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
            # Compress: average over sequence length -> (n_probes, n_embd)
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
        storage_kb = total_params * 4 / 1024

        print(f"  Saved {n_probes} anchors for Phase {phase_key} "
              f"({len(hidden_states)} layers, ~{storage_kb:.1f}KB)")

        model.train()

    def compute_anchor_loss(
        self,
        model,
        phase_keys: List[str],
        device: str = "cuda",
        anchor_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Dict[int, float]]]:
        """
        Compute anchor-pull loss: MSE between current and anchor hidden states.

        This is the core self-correction mechanism. The model checks:
        "have my representations drifted from when I 'knew' this?"
        If yes, the gradient from this loss pulls them back.

        Args:
            model: The current PeftModel (base + LoRA)
            phase_keys: Which phases' anchors to check
            device: Device
            anchor_layers: Which layers to include (None = all)

        Returns:
            (loss, drift_report) where drift_report has per-phase, per-layer drift
        """
        total_loss = None
        drift_report = {}
        PROBE_BATCH_SIZE = 10  # Process probes in mini-batches to avoid OOM

        for phase_key in phase_keys:
            if phase_key not in self.anchors:
                continue

            anchor_data = self.anchors[phase_key]
            probe_input_ids = anchor_data["probe_input_ids"].to(device)

            # Process probes in mini-batches to avoid OOM
            # Each batch forward pass is small enough to fit alongside model weights
            phase_loss = None
            phase_drift = {}
            n_layers = None
            n_probe_batches = (len(probe_input_ids) + PROBE_BATCH_SIZE - 1) // PROBE_BATCH_SIZE

            for batch_idx in range(n_probe_batches):
                start = batch_idx * PROBE_BATCH_SIZE
                end = min(start + PROBE_BATCH_SIZE, len(probe_input_ids))
                batch_probes = probe_input_ids[start:end]

                # Run model on this mini-batch of probes (WITH gradient tracking for repair)
                outputs = model(input_ids=batch_probes, output_hidden_states=True)

                if n_layers is None:
                    n_layers = 0

                for layer_idx, current_hs in enumerate(outputs.hidden_states):
                    if anchor_layers is not None and layer_idx not in anchor_layers:
                        continue

                    # Compress current hidden state: mean over sequence length
                    current_mean = current_hs.mean(dim=1)

                    # Use only the corresponding probe indices for anchor mean
                    anchor_mean_full = anchor_data["hidden_states"][layer_idx].to(device)
                    anchor_mean = anchor_mean_full[start:end]

                    # MSE drift — current_mean carries grad_fn from model forward pass
                    drift = F.mse_loss(current_mean, anchor_mean)
                    if phase_loss is None:
                        phase_loss = drift
                    else:
                        phase_loss = phase_loss + drift
                    
                    if layer_idx not in phase_drift:
                        phase_drift[layer_idx] = 0.0
                    phase_drift[layer_idx] += drift.item()
                    n_layers += 1

            if n_layers is not None and n_layers > 0 and phase_loss is not None:
                phase_loss = phase_loss / n_layers  # Average over layers and probe batches
                # Also average the drift report
                phase_drift = {k: v / n_probe_batches for k, v in phase_drift.items()}

            if phase_loss is not None:
                if total_loss is None:
                    total_loss = phase_loss
                else:
                    total_loss = total_loss + phase_loss
            drift_report[phase_key] = phase_drift

        if total_loss is None:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss, drift_report

    def verify(
        self,
        model,
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


class WeightDeltaAnchorStore:
    """
    Ablation: anchors on LoRA weight deltas instead of hidden states.

    After training on a phase, save the LoRA adapter weights.
    During new phase training, penalize deviation from saved weights.
    This is a simpler but potentially less effective approach.
    """

    def __init__(self):
        self.weight_anchors = {}  # phase_key -> {param_name: param_value}

    def save_anchors(self, model, phase_key: str):
        """Save current LoRA parameters as anchors."""
        lora_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                lora_params[name] = param.data.cpu().clone()

        self.weight_anchors[phase_key] = lora_params
        n_params = sum(v.numel() for v in lora_params.values())
        storage_kb = n_params * 4 / 1024
        print(f"  Saved weight-delta anchors for Phase {phase_key} "
              f"({len(lora_params)} params, ~{storage_kb:.1f}KB)")

    def compute_anchor_loss(
        self,
        model,
        phase_keys: List[str],
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MSE between current LoRA weights and saved anchors."""
        total_loss = torch.tensor(0.0, device=device)
        drift_report = {}

        for phase_key in phase_keys:
            if phase_key not in self.weight_anchors:
                continue

            phase_loss = torch.tensor(0.0, device=device)
            n_params = 0

            for name, param in model.named_parameters():
                if param.requires_grad and "lora" in name.lower():
                    if name in self.weight_anchors[phase_key]:
                        anchor_val = self.weight_anchors[phase_key][name].to(device)
                        drift = F.mse_loss(param, anchor_val)
                        phase_loss = phase_loss + drift
                        n_params += 1

            if n_params > 0:
                phase_loss = phase_loss / n_params

            total_loss = total_loss + phase_loss
            drift_report[phase_key] = phase_loss.item()

        return total_loss, drift_report
