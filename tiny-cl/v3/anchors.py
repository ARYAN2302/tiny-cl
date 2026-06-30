"""
V3 Anchor Store: Representation anchoring with salience tagging.
Extends V2 AnchorStore with per-domain importance weights (amygdala).
"""

import random
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class SalientAnchorStore:
    """
    Stores compressed representation anchors for each completed phase,
    with per-domain salience weights (amygdala-inspired).
    """

    def __init__(self):
        self.anchors = {}       # phase_key -> anchor data
        self.salience = {}      # phase_key -> float (importance weight)
        self.health = {}        # phase_key -> float (latest health score, 0-1)

    def save_anchors(
        self,
        model,
        phase_key: str,
        train_dataset,
        n_probes: int = 50,
        device: str = "cuda",
        salience: float = 1.0,
        anchor_layers: Optional[List[int]] = None,
    ):
        """Save anchor representations after training on a phase."""
        model.eval()

        n_available = len(train_dataset)
        n_probes = min(n_probes, n_available)
        probe_indices = random.sample(range(n_available), n_probes)

        probe_input_ids_list = []
        for idx in probe_indices:
            sample = train_dataset[idx]
            probe_input_ids_list.append(sample["input_ids"])

        probe_input_ids = torch.stack(probe_input_ids_list).to(device)

        with torch.no_grad():
            outputs = model(input_ids=probe_input_ids, output_hidden_states=True)

        hidden_states = {}
        for layer_idx, hs in enumerate(outputs.hidden_states):
            if anchor_layers is not None and layer_idx not in anchor_layers:
                continue
            hidden_states[layer_idx] = hs.mean(dim=1).cpu()

        self.anchors[phase_key] = {
            "probe_input_ids": probe_input_ids.cpu(),
            "hidden_states": hidden_states,
            "n_probes": n_probes,
        }

        self.salience[phase_key] = salience
        self.health[phase_key] = 1.0

        total_params = sum(v.numel() for v in hidden_states.values()) + probe_input_ids.numel()
        storage_kb = total_params * 4 / 1024

        print(f"  Saved {n_probes} anchors for Phase {phase_key} "
              f"({len(hidden_states)} layers, ~{storage_kb:.1f}KB, salience={salience:.1f})")

        model.train()

    def verify(
        self,
        model,
        phase_keys: List[str],
        threshold: float = 0.1,
        device: str = "cuda",
        anchor_layers: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, Dict[int, float]], List[str], List[str]]:
        """
        Verify which phases/layers have drifted beyond threshold.
        Weighted by salience - high-salience domains trigger repair sooner.
        """
        PROBE_BATCH_SIZE = 10
        drift_report = {}
        degraded_phases = []
        degraded_layers = []

        for phase_key in phase_keys:
            if phase_key not in self.anchors:
                continue

            anchor_data = self.anchors[phase_key]
            probe_input_ids = anchor_data["probe_input_ids"].to(device)
            salience = self.salience.get(phase_key, 1.0)

            phase_drift = {}
            n_probe_batches = (len(probe_input_ids) + PROBE_BATCH_SIZE - 1) // PROBE_BATCH_SIZE

            with torch.no_grad():
                for batch_idx in range(n_probe_batches):
                    start = batch_idx * PROBE_BATCH_SIZE
                    end = min(start + PROBE_BATCH_SIZE, len(probe_input_ids))
                    batch_probes = probe_input_ids[start:end]

                    outputs = model(input_ids=batch_probes, output_hidden_states=True)

                    for layer_idx, current_hs in enumerate(outputs.hidden_states):
                        if anchor_layers is not None and layer_idx not in anchor_layers:
                            continue
                        if layer_idx not in anchor_data["hidden_states"]:
                            continue

                        current_mean = current_hs.mean(dim=1)
                        anchor_mean = anchor_data["hidden_states"][layer_idx].to(device)[start:end]

                        drift_val = F.mse_loss(current_mean, anchor_mean).item()

                        if layer_idx not in phase_drift:
                            phase_drift[layer_idx] = 0.0
                        phase_drift[layer_idx] += drift_val

                phase_drift = {k: v / n_probe_batches for k, v in phase_drift.items()}

            drift_report[phase_key] = phase_drift

            effective_threshold = threshold / salience
            for layer_idx, drift_val in phase_drift.items():
                if drift_val > effective_threshold:
                    degraded_phases.append(phase_key)
                    degraded_layers.append(f"{phase_key}_L{layer_idx}")

            if phase_drift:
                avg_drift = sum(phase_drift.values()) / len(phase_drift)
                health = max(0.0, 1.0 - avg_drift / (threshold * 5))
                self.health[phase_key] = health

        return drift_report, degraded_phases, degraded_layers

    def get_salience_weighted_phases(self) -> List[Tuple[str, float]]:
        """Return phases sorted by salience (highest first) for priority repair."""
        return sorted(self.salience.items(), key=lambda x: x[1], reverse=True)

    def get_storage_size_kb(self) -> float:
        """Total storage used by all anchors."""
        total = 0
        for phase_data in self.anchors.values():
            total += phase_data["probe_input_ids"].numel()
            for hs in phase_data["hidden_states"].values():
                total += hs.numel()
        return total * 4 / 1024
