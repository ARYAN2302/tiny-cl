"""
V3 Methods: The Living Model - fast-slow AVR with consolidation.

The AVR loop runs on LFM2.5's natural architectural split:
  ABSORB:    Train conv-layer LoRA (fast path) on new domain
  VERIFY:    Check attention-layer anchors for drift (slow path)
  REPAIR:    Pull conv LoRA toward saved snapshot (MSE on weights)
  CONSOLIDATE: Distill conv knowledge into attention LoRA (slow path)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from config import (
    LivingModelConfig, TrainConfig, FastSlowLoRAConfig,
    CONV_LAYER_IDS, ATTN_LAYER_IDS,
)
from anchors import SalientAnchorStore


class PhaseController:
    """
    State machine: ABSORB -> VERIFY -> REPAIR -> CONSOLIDATE -> ABSORB
    The autonomic nervous system of the Living Model.
    """
    ABSORB = "absorb"
    VERIFY = "verify"
    REPAIR = "repair"
    CONSOLIDATE = "consolidate"

    def __init__(self, config: LivingModelConfig):
        self.config = config
        self.phase = self.ABSORB
        self.step_count = 0
        self.domain_just_changed = False
        self.phase_history = []

    def transition_to(self, new_phase: str, reason: str = ""):
        old = self.phase
        self.phase = new_phase
        self.phase_history.append((self.step_count, old, new_phase, reason))
        print(f"  [PHASE] {old} -> {new_phase}" + (f" ({reason})" if reason else ""))

    def should_verify(self, step_count: int) -> bool:
        return step_count % self.config.verify_every_n_steps == 0 and step_count > 0

    def on_domain_change(self):
        self.domain_just_changed = True


class LivingModelMethod:
    """
    The Living Model: fast-slow architecture + AVR + consolidation.

    LFM2.5 conv layers = FAST (hippocampus) - plastic, absorbs new knowledge
    LFM2.5 attn layers = SLOW (neocortex) - stable, holds consolidated knowledge
    """

    def __init__(
        self,
        living_config: LivingModelConfig,
        lora_config: FastSlowLoRAConfig,
        train_config: TrainConfig,
    ):
        self.living_config = living_config
        self.lora_config = lora_config
        self.train_config = train_config

        self.anchor_store = SalientAnchorStore()
        self.phase_controller = PhaseController(living_config)

        self.conv_lora_snapshots = {}
        self.attn_lora_snapshots = {}

        self.completed_phases = []
        self.step_count = 0
        self.total_repairs = 0
        self.total_consolidations = 0
        self._current_salience = 1.0

    def _is_conv_param(self, name: str) -> bool:
        for idx in CONV_LAYER_IDS:
            if f"layers.{idx}.conv." in name:
                return True
        return False

    def _is_attn_param(self, name: str) -> bool:
        for idx in ATTN_LAYER_IDS:
            if f"layers.{idx}.self_attn." in name:
                return True
        return False

    # ─── Freeze/Unfreeze ───

    def freeze_attn_lora(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and self._is_attn_param(name):
                param.requires_grad = False

    def unfreeze_attn_lora(self, model):
        for name, param in model.named_parameters():
            if "lora_" in name and self._is_attn_param(name):
                param.requires_grad = True

    def freeze_conv_lora(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and self._is_conv_param(name):
                param.requires_grad = False

    def unfreeze_conv_lora(self, model):
        for name, param in model.named_parameters():
            if "lora_" in name and self._is_conv_param(name):
                param.requires_grad = True

    # ─── Phase Handlers ───

    def on_phase_start(self, model, phase_key: str, domain_salience: float = 1.0):
        self.phase_controller.on_domain_change()
        self.unfreeze_conv_lora(model)
        self.freeze_attn_lora(model)
        self._current_salience = domain_salience

    def on_phase_end(self, model, phase_key: str, train_dataset):
        # Anchor on BOTH conv and attn layers — conv is where drift happens during absorb,
        # attn is where we check for consolidation effects.
        # V2 only anchored attn (because LoRA was on attn), but V3 trains conv → must anchor conv too.
        conv_hidden_indices = [idx + 1 for idx in CONV_LAYER_IDS]
        attn_hidden_indices = [idx + 1 for idx in ATTN_LAYER_IDS]
        all_anchor_layers = conv_hidden_indices + attn_hidden_indices

        self.anchor_store.save_anchors(
            model, phase_key, train_dataset,
            n_probes=self.living_config.n_anchor_probes,
            device=self.train_config.device,
            salience=self._current_salience,
            anchor_layers=all_anchor_layers,
        )

        # Save conv LoRA snapshot for REPAIR
        conv_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and self._is_conv_param(name):
                conv_params[name] = param.data.cpu().clone()
        self.conv_lora_snapshots[phase_key] = conv_params

        # Save attn LoRA snapshot for CONSOLIDATION reference
        attn_params = {}
        for name, param in model.named_parameters():
            if "lora_" in name and self._is_attn_param(name):
                attn_params[name] = param.data.cpu().clone()
        self.attn_lora_snapshots[phase_key] = attn_params

        n_conv = sum(v.numel() for v in conv_params.values())
        n_attn = sum(v.numel() for v in attn_params.values())
        print(f"  Saved conv LoRA snapshot: {len(conv_params)} params (~{n_conv * 4 / 1024:.1f}KB)")
        print(f"  Saved attn LoRA snapshot: {len(attn_params)} params (~{n_attn * 4 / 1024:.1f}KB)")

        self.completed_phases.append(phase_key)

    def compute_loss(self, model, batch, device):
        self.step_count += 1
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        return outputs.loss, {"lm_loss": outputs.loss.item(), "phase": "absorb"}

    # ─── VERIFY ───

    def verify(self, model, device) -> Tuple[Dict, List[str], bool]:
        if not self.completed_phases:
            return {}, [], False

        # Check BOTH conv and attn layers — conv drifts during absorb, attn drifts during consolidation
        conv_hidden_indices = [idx + 1 for idx in CONV_LAYER_IDS]
        attn_hidden_indices = [idx + 1 for idx in ATTN_LAYER_IDS]
        all_anchor_layers = conv_hidden_indices + attn_hidden_indices

        drift_report, degraded_phases, degraded_layers = self.anchor_store.verify(
            model,
            phase_keys=self.completed_phases,
            threshold=self.living_config.drift_threshold,
            device=device,
            anchor_layers=all_anchor_layers,
        )

        for pk in self.completed_phases:
            health = self.anchor_store.health.get(pk, 1.0)
            salience = self.anchor_store.salience.get(pk, 1.0)
            status = "HEALTHY" if health > self.living_config.verification_threshold else "DEGRADED"
            print(f"  [HEALTH] {pk}: {health:.3f} (salience={salience:.1f}) [{status}]")

        repair_needed = len(degraded_phases) > 0
        return drift_report, degraded_phases, repair_needed

    # ─── REPAIR ───

    def repair(self, model, degraded_phases: List[str], device) -> bool:
        self.total_repairs += 1

        unique_degraded = list(set(degraded_phases))
        salience_order = self.anchor_store.get_salience_weighted_phases()
        unique_degraded.sort(
            key=lambda pk: dict(salience_order).get(pk, 0.0),
            reverse=True,
        )

        print(f"  [REPAIR] Running {self.living_config.repair_steps} steps for {unique_degraded}")

        conv_params = [
            (name, param) for name, param in model.named_parameters()
            if param.requires_grad and self._is_conv_param(name)
        ]

        if not conv_params:
            print("  [REPAIR] No conv LoRA params to repair")
            return False

        trainable = [p for _, p in conv_params]
        repair_optimizer = torch.optim.Adam(trainable, lr=self.living_config.repair_lr)

        for step in range(self.living_config.repair_steps):
            weight_loss = torch.tensor(0.0, device=device)
            n_targets = 0

            for phase_key in unique_degraded:
                if phase_key not in self.conv_lora_snapshots:
                    continue
                salience = self.anchor_store.salience.get(phase_key, 1.0)

                for name, param in conv_params:
                    if name in self.conv_lora_snapshots[phase_key]:
                        target = self.conv_lora_snapshots[phase_key][name].to(device)
                        drift = salience * F.mse_loss(param, target)
                        weight_loss = weight_loss + drift
                        n_targets += 1

            if n_targets > 0 and weight_loss.requires_grad:
                weight_loss = weight_loss / n_targets
                repair_optimizer.zero_grad()
                weight_loss.backward()
                repair_optimizer.step()
            else:
                break

        # Re-verify on all anchored layers (conv + attn)
        conv_hidden_indices = [idx + 1 for idx in CONV_LAYER_IDS]
        attn_hidden_indices = [idx + 1 for idx in ATTN_LAYER_IDS]
        all_anchor_layers = conv_hidden_indices + attn_hidden_indices
        _, new_degraded, _ = self.anchor_store.verify(
            model,
            phase_keys=self.completed_phases,
            threshold=self.living_config.drift_threshold,
            device=device,
            anchor_layers=all_anchor_layers,
        )

        success = len(new_degraded) == 0
        if success:
            print(f"  [REPAIR] Successful - all domains healthy")
        else:
            print(f"  [REPAIR] Partial - {new_degraded} still degraded")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return success

    # ─── CONSOLIDATE (NEW in V3) ───

    def _bypass_lm_head(self, model):
        """Temporarily replace lm_head with Identity to skip massive logit tensor.

        During consolidation we only need hidden states, not logits.
        The lm_head projection (hidden_dim -> 65536 vocab) allocates ~1.25 GiB
        per batch — the #1 cause of OOM. Bypassing it saves that memory entirely.
        """
        # PeftModelForCausalLM -> base_model.model is the original CausalLM
        base_model = getattr(model, 'base_model', model)
        inner = getattr(base_model, 'model', base_model)

        # Try common attribute paths for lm_head
        for path in ['lm_head', 'model.lm_head']:
            parts = path.split('.')
            obj = inner
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, parts[-1]):
                original = getattr(obj, parts[-1])
                setattr(obj, parts[-1], torch.nn.Identity())
                return original, obj, parts[-1]

        print("  [WARN] Could not find lm_head to bypass — consolidation may OOM")
        return None, None, None

    def _restore_lm_head(self, saved):
        """Restore the original lm_head after consolidation."""
        original, parent, attr_name = saved
        if original is not None and parent is not None:
            setattr(parent, attr_name, original)

    def _zero_conv_lora_B(self, model):
        """Zero out conv LoRA lora_B matrices so conv LoRA has no effect.

        Used during consolidation's student pass: we want attn LoRA to learn
        to produce the same hidden states WITHOUT conv LoRA's help.
        Returns a dict of backup tensors so we can restore after.
        """
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name and self._is_conv_param(name):
                    backup[name] = param.data.clone()
                    param.data.zero_()
        n_zeroed = len(backup)
        print(f"  [CONSOLIDATE] Zeroed {n_zeroed} conv LoRA B matrices for student pass")
        return backup

    def _restore_conv_lora_B(self, model, backup):
        """Restore conv LoRA lora_B matrices from backup."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])
        print(f"  [CONSOLIDATE] Restored {len(backup)} conv LoRA B matrices")

    def consolidate(self, model, phase_key: str, train_dataset, device):
        """
        CONSOLIDATE: Transfer knowledge from conv LoRA (fast) to attn LoRA (slow).

        Distillation strategy:
          1. TEACHER pass: Full model (base + conv LoRA + attn LoRA) → target hidden states
          2. STUDENT pass: Model WITHOUT conv LoRA (base + attn LoRA only) → train attn LoRA
             to reproduce the teacher's hidden states

        The gap between teacher and student is exactly what conv LoRA was contributing.
        Attn LoRA learns to fill this gap → knowledge is consolidated into slow memory.

        Memory optimizations:
          - lm_head replaced with Identity (saves ~1.25 GiB per batch)
          - Small probe batch size (2)
          - Aggressive cache clearing between passes
        """
        self.total_consolidations += 1
        print(f"  [CONSOLIDATE] Distilling Phase {phase_key} from conv -> attn")

        # Bypass lm_head to avoid 1.25 GiB logit allocation
        lm_head_saved = self._bypass_lm_head(model)

        # Setup: freeze conv, unfreeze attn for gradient tracking
        self.freeze_conv_lora(model)
        self.unfreeze_attn_lora(model)

        attn_params = [
            (name, param) for name, param in model.named_parameters()
            if param.requires_grad and self._is_attn_param(name)
        ]

        if not attn_params:
            print("  [CONSOLIDATE] No attn LoRA params - skipping")
            self._restore_lm_head(lm_head_saved)
            self.unfreeze_conv_lora(model)
            self.freeze_attn_lora(model)
            return

        trainable = [p for _, p in attn_params]
        consolid_optimizer = torch.optim.Adam(
            trainable, lr=self.living_config.consolidation_lr
        )

        anchor_data = self.anchor_store.anchors.get(phase_key)
        if anchor_data is None:
            print("  [CONSOLIDATE] No anchors for this phase - skipping")
            self._restore_lm_head(lm_head_saved)
            self.unfreeze_conv_lora(model)
            self.freeze_attn_lora(model)
            return

        probe_input_ids = anchor_data["probe_input_ids"]
        # Use ALL layers for distillation, not just attn layers.
        # Attn LoRA affects attn+ layers, but the loss at ALL layers gives richer signal
        # and captures the downstream propagation of conv LoRA's contribution.
        all_hidden_indices = [idx + 1 for idx in range(16)]  # All 16 layers + embedding
        # Skip embedding (idx 0), keep layers 1-16
        distill_layers = list(range(1, 17))

        PROBE_BATCH_SIZE = 2
        n_probes = len(probe_input_ids)
        n_batches = (n_probes + PROBE_BATCH_SIZE - 1) // PROBE_BATCH_SIZE

        target_hidden = {idx: [] for idx in distill_layers}

        # ═══ TEACHER PASS: Full model (conv LoRA ACTIVE) ═══
        # Conv LoRA is frozen but still APPLIED in forward pass → contributes to hidden states
        print(f"  [CONSOLIDATE] Teacher pass: capturing hidden states with conv LoRA active")
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start = batch_idx * PROBE_BATCH_SIZE
                end = min(start + PROBE_BATCH_SIZE, n_probes)
                batch_probes = probe_input_ids[start:end].to(device)

                outputs = model(input_ids=batch_probes, output_hidden_states=True)

                for layer_idx in distill_layers:
                    target_hidden[layer_idx].append(outputs.hidden_states[layer_idx].cpu().detach())

                del outputs, batch_probes
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Concatenate targets on CPU — move slices to GPU only when needed
        for layer_idx in distill_layers:
            target_hidden[layer_idx] = torch.cat(target_hidden[layer_idx], dim=0)

        # Log teacher hidden state magnitudes for verification (sample a few layers)
        for layer_idx in [1, 3, 6, 12, 16]:
            if layer_idx in target_hidden:
                mag = target_hidden[layer_idx].float().norm().item()
                print(f"    Teacher hidden L{layer_idx} norm: {mag:.4f}")

        # ═══ STUDENT PASS: Conv LoRA ZEROED → attn LoRA must compensate ═══
        conv_B_backup = self._zero_conv_lora_B(model)

        print(f"  [CONSOLIDATE] Student pass: training attn LoRA (conv LoRA disabled)")
        print(f"    Distilling across {len(distill_layers)} layers, {self.living_config.consolidation_steps} steps")
        for step in range(self.living_config.consolidation_steps):
            step_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * PROBE_BATCH_SIZE
                end = min(start + PROBE_BATCH_SIZE, n_probes)
                batch_probes = probe_input_ids[start:end].to(device)

                outputs = model(input_ids=batch_probes, output_hidden_states=True)

                batch_loss = torch.tensor(0.0, device=device)
                n_layers = 0
                # Only compute loss at attn+ layers (where attn LoRA can have effect)
                # attn layers are at indices 2,5,8,10,12,14 → hidden state indices 3,6,9,11,13,15
                # Plus all layers AFTER first attn layer (attn LoRA changes propagate forward)
                attn_plus_layers = [idx for idx in distill_layers if idx >= 3]

                for layer_idx in attn_plus_layers:
                    current_hs = outputs.hidden_states[layer_idx]
                    target_hs = target_hidden[layer_idx][start:end].to(device)
                    batch_loss = batch_loss + F.mse_loss(current_hs, target_hs)
                    n_layers += 1

                if n_layers > 0:
                    batch_loss = batch_loss / n_layers
                    consolid_optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, self.train_config.max_grad_norm)
                    consolid_optimizer.step()
                    step_loss += batch_loss.item()

                del outputs, batch_probes, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if step % 50 == 0:
                avg_loss = step_loss / max(n_batches, 1)
                print(f"    Consolidation step {step}/{self.living_config.consolidation_steps} "
                      f"| Loss: {avg_loss:.6f}")

        # Free target tensors
        del target_hidden

        # ═══ RESTORE conv LoRA before partial reset ═══
        self._restore_conv_lora_B(model, conv_B_backup)
        del conv_B_backup

        print(f"  [CONSOLIDATE] Phase {phase_key} consolidated into attention layers")

        # Update attn LoRA snapshot
        new_attn_params = {}
        for name, param in model.named_parameters():
            if "lora_" in name and self._is_attn_param(name):
                new_attn_params[name] = param.data.cpu().clone()
        self.attn_lora_snapshots[phase_key] = new_attn_params

        # Partially reset conv LoRA to free capacity for next domain
        self._partial_reset_conv_lora(model, device)

        # Restore lm_head before switching back
        self._restore_lm_head(lm_head_saved)

        # Switch back to absorb mode
        self.unfreeze_conv_lora(model)
        self.freeze_attn_lora(model)

        # ═══ POST-CONSOLIDATION VERIFY ═══
        # Consolidation modifies attn LoRA → check if it hurt previous domains
        if self.completed_phases:
            post_drift, post_degraded, _ = self.verify(model, device)
            if post_degraded:
                print(f"  [POST-CONSOLIDATE] Consolidation caused drift on {post_degraded}!")
                self.repair(model, post_degraded, device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _partial_reset_conv_lora(self, model, device):
        """Scale conv LoRA B matrices to free capacity for next domain."""
        factor = self.living_config.fast_reset_factor

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_" in name and self._is_conv_param(name) and "lora_B" in name:
                    param.data *= factor

        print(f"  [RESET] Conv LoRA scaled by {factor} - capacity freed for next domain")

    # ─── Full Cycle ───

    def run_verify_repair_consolidate(self, model, phase_key: str, train_dataset, device):
        """Run the full VERIFY -> REPAIR -> CONSOLIDATE cycle."""
        drift_report, degraded_phases, repair_needed = self.verify(model, device)

        repair_success = True
        if repair_needed:
            repair_success = self.repair(model, degraded_phases, device)

        # Only consolidate if consolidation_steps > 0
        if self.living_config.consolidation_steps > 0:
            self.consolidate(model, phase_key, train_dataset, device)
        else:
            print(f"  [CONSOLIDATE] Skipped (consolidation_steps=0)")

        return {
            "health": dict(self.anchor_store.health),
            "repair_needed": repair_needed,
            "repair_success": repair_success,
            "degraded_phases": degraded_phases,
        }

    def get_storage_kb(self) -> float:
        anchor_kb = self.anchor_store.get_storage_size_kb()
        conv_kb = sum(
            v.numel() * 4 / 1024
            for snap in self.conv_lora_snapshots.values()
            for v in snap.values()
        )
        attn_kb = sum(
            v.numel() * 4 / 1024
            for snap in self.attn_lora_snapshots.values()
            for v in snap.values()
        )
        return anchor_kb + conv_kb + attn_kb


class NaiveMethod:
    """Naive LoRA - sequential adapter training, no protection."""

    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config
        self.name = "naive"
        self.completed_phases = []

    def on_phase_start(self, model, phase_key, domain_salience=1.0):
        pass

    def on_phase_end(self, model, phase_key, train_dataset):
        self.completed_phases.append(phase_key)

    def compute_loss(self, model, batch, device):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
        )
        return outputs.loss, {"lm_loss": outputs.loss.item()}

    def get_storage_kb(self) -> float:
        return 0.0
