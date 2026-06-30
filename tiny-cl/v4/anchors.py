"""
V4 Anchors: Adapted for streaming regime with small increments.
Key change: anchors work with fewer probes and smaller models.
"""

import random
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class StreamingAnchorStore:
    """
    Lightweight anchor store for streaming experiments.
    Works with both LSTM and GPT models.
    Saves compressed hidden-state fingerprints for drift detection.
    """

    def __init__(self):
        self.anchors = {}       # phase_key -> {probe_input_ids, hidden_states}
        self.health = {}        # phase_key -> float (0-1)

    def save_anchors(
        self,
        model,
        phase_key: str,
        streaming_data,          # StreamingDataset
        n_probes: int = 20,
        device: str = "cpu",
    ):
        """Save anchor fingerprints after learning a phase."""
        model.eval()

        # Get probe sequences
        probe_input_ids = streaming_data.get_probes(n_probes, device)

        # Get hidden states
        with torch.no_grad():
            if hasattr(model, 'get_hidden_states'):
                hiddens = model.get_hidden_states(probe_input_ids)
            else:
                # Fallback: use forward output
                outputs = model(input_ids=probe_input_ids)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    hiddens = outputs.hidden_states
                else:
                    # For LSTM: use the output as single hidden state
                    hiddens = (outputs.logits.detach(),)

        # Compress: mean over sequence length per probe
        compressed = {}
        for idx, hs in enumerate(hiddens):
            compressed[idx] = hs.mean(dim=1).cpu()

        self.anchors[phase_key] = {
            "probe_input_ids": probe_input_ids.cpu(),
            "hidden_states": compressed,
            "n_probes": n_probes,
        }
        self.health[phase_key] = 1.0

        storage_kb = sum(v.numel() for v in compressed.values()) * 4 / 1024
        print(f"  Anchors saved: Phase {phase_key} ({len(compressed)} layers, ~{storage_kb:.1f}KB)")

        model.train()

    def verify(
        self,
        model,
        phase_keys: List[str],
        threshold: float = 0.1,
        device: str = "cpu",
    ) -> Tuple[Dict, List[str], bool]:
        """Check which phases have drifted beyond threshold."""
        if not phase_keys:
            return {}, [], False

        drift_report = {}
        degraded_phases = []

        for phase_key in phase_keys:
            if phase_key not in self.anchors:
                continue

            anchor_data = self.anchors[phase_key]
            probe_input_ids = anchor_data["probe_input_ids"].to(device)

            # Get current hidden states
            with torch.no_grad():
                if hasattr(model, 'get_hidden_states'):
                    hiddens = model.get_hidden_states(probe_input_ids)
                else:
                    outputs = model(input_ids=probe_input_ids)
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        hiddens = outputs.hidden_states
                    else:
                        hiddens = (outputs.logits.detach(),)

            # Compare to anchors
            phase_drift = {}
            for idx, hs in enumerate(hiddens):
                if idx not in anchor_data["hidden_states"]:
                    continue
                current_mean = hs.mean(dim=1)
                anchor_mean = anchor_data["hidden_states"][idx].to(device)
                drift_val = F.mse_loss(current_mean, anchor_mean).item()
                phase_drift[idx] = drift_val

            drift_report[phase_key] = phase_drift

            # Check if any layer exceeds threshold
            max_drift = max(phase_drift.values()) if phase_drift else 0.0
            if max_drift > threshold:
                degraded_phases.append(phase_key)

            # Update health
            avg_drift = sum(phase_drift.values()) / max(len(phase_drift), 1)
            self.health[phase_key] = max(0.0, 1.0 - avg_drift / (threshold * 5))

        needs_repair = len(degraded_phases) > 0
        return drift_report, degraded_phases, needs_repair

    def get_storage_kb(self) -> float:
        total = 0
        for data in self.anchors.values():
            total += data["probe_input_ids"].numel()
            for hs in data["hidden_states"].values():
                total += hs.numel()
        return total * 4 / 1024
