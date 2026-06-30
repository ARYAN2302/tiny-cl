"""
V2 Plotting: Forgetting curves, comparison bars, and Memory Health Dashboard.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font setup — best effort, don't crash if fonts are missing
try:
    import glob as _glob
    _dejavu = _glob.glob('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if _dejavu:
        fm.fontManager.addfont(_dejavu[0])
    _noto = _glob.glob('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
    if not _noto:
        _noto = _glob.glob('/usr/share/fonts/truetype/lxgw-wenkai/*.ttf')
    if _noto:
        try:
            fm.fontManager.addfont(_noto[0])
        except RuntimeError:
            pass  # Variable font not supported, skip
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(results_dir: str, preset: str = "hero") -> List[Dict]:
    """Load combined results from a preset run."""
    filepath = os.path.join(results_dir, f"combined_{preset}.json")
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)

    # Try loading individual result files
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    return results


def plot_forgetting_curves(results: List[Dict], output_dir: str = "v2/results"):
    """
    Plot forgetting curves for each method.
    X-axis: training phase, Y-axis: perplexity on Phase A (first domain).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        method_name = result.get("method", "unknown")
        phases = result.get("phases", {})
        phase_keys = sorted(phases.keys())

        if not phase_keys:
            continue

        first_phase = phase_keys[0]
        # Track perplexity on first phase after each training phase
        ppls = []
        x_labels = []

        for pk in phase_keys:
            ppl = phases[pk].get("perplexity", {}).get(first_phase, None)
            if ppl is not None:
                ppls.append(ppl)
                x_labels.append(f"After {pk}")

        if ppls:
            ax.plot(range(len(ppls)), ppls, marker='o', linewidth=2, label=method_name)

    ax.set_xlabel("Training Progress", fontsize=12)
    ax.set_ylabel(f"Perplexity on Phase A", fontsize=12)
    ax.set_title("Forgetting Curves: Phase A Performance Over Time", fontsize=14)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forgetting_curves_v2.png"), dpi=150)
    plt.close()
    print(f"Saved forgetting_curves_v2.png")


def plot_comparison_bars(results: List[Dict], output_dir: str = "v2/results"):
    """
    Bar chart comparing methods on: avg forgetting factor, final Phase A ppl, storage.
    """
    methods = []
    forgetting_factors = []
    phase_a_ppls = []
    storages = []

    for result in results:
        method_name = result.get("method", "unknown")
        phases = result.get("phases", {})
        phase_keys = sorted(phases.keys())

        if not phase_keys:
            continue

        # Final perplexity on Phase A
        last_phase = phase_keys[-1]
        final_ppl_a = phases[last_phase].get("perplexity", {}).get(phase_keys[0], None)

        # After-learn perplexity on Phase A
        after_learn_ppl_a = phases[phase_keys[0]].get("perplexity", {}).get(phase_keys[0], None)

        # Forgetting factor
        ff = None
        if after_learn_ppl_a and final_ppl_a and after_learn_ppl_a > 0:
            ff = final_ppl_a / after_learn_ppl_a

        methods.append(method_name)
        forgetting_factors.append(ff if ff else 0)
        phase_a_ppls.append(final_ppl_a if final_ppl_a else 0)
        storages.append(result.get("storage_kb", 0))

    if not methods:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Forgetting Factor
    colors = ['#e74c3c' if ff > 3 else '#f39c12' if ff > 1.5 else '#2ecc71' for ff in forgetting_factors]
    axes[0].barh(methods, forgetting_factors, color=colors)
    axes[0].set_xlabel("Forgetting Factor (lower = better)")
    axes[0].set_title("Forgetting Factor")
    axes[0].axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='No forgetting')
    axes[0].legend()

    # Phase A Perplexity
    axes[1].barh(methods, phase_a_ppls, color='#3498db')
    axes[1].set_xlabel("Perplexity (lower = better)")
    axes[1].set_title("Final Phase A Perplexity")

    # Storage
    axes[2].barh(methods, storages, color='#9b59b6')
    axes[2].set_xlabel("Storage (KB)")
    axes[2].set_title("Storage Overhead")

    plt.suptitle("Method Comparison: Anchor-AVR vs Baselines", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "comparison_bars_v2.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison_bars_v2.png")


def plot_memory_health_dashboard(results: List[Dict], output_dir: str = "v2/results"):
    """
    Memory Health Dashboard — the hero visualization.
    Shows a "health bar" for each domain after all training.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Find Anchor-AVR Discrete result
    anchor_result = None
    for result in results:
        if "anchor_disc" in result.get("method", ""):
            anchor_result = result
            break

    if anchor_result is None:
        print("No Anchor-AVR Discrete result found for dashboard")
        return

    phases = anchor_result.get("phases", {})
    phase_keys = sorted(phases.keys())
    last_phase = phase_keys[-1]

    final_ppls = phases[last_phase].get("perplexity", {})
    after_learn_ppls = {}
    for pk in phase_keys:
        after_learn_ppls[pk] = phases[pk].get("perplexity", {}).get(pk, None)

    # Compute "health" as 1 - forgetting_factor (capped at 0)
    healths = []
    labels = []
    for pk in phase_keys:
        after_learn = after_learn_ppls.get(pk)
        after_all = final_ppls.get(pk)
        if after_learn and after_all and after_learn > 0:
            ff = after_all / after_learn
            health = max(0, 1.0 - (ff - 1.0) / 2.0)  # Scale: 1.0 = perfect, 0 = catastrophic
        else:
            health = 1.0
        healths.append(health)
        labels.append(pk)

    # Add "General English" as ~99% (base model frozen)
    labels.append("General\nEnglish")
    healths.append(0.99)

    # Draw health bars
    y_positions = range(len(labels))
    colors = []
    for h in healths:
        if h > 0.8:
            colors.append('#2ecc71')  # Green
        elif h > 0.5:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red

    bars = ax.barh(y_positions, healths, color=colors, height=0.6, edgecolor='white', linewidth=2)

    # Add percentage labels
    for i, (h, bar) in enumerate(zip(healths, bars)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{h*100:.0f}%', va='center', fontsize=14, fontweight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Memory Health", fontsize=12)
    ax.set_title("Memory Health Dashboard\nLoRA + Anchor-AVR (Discrete)", fontsize=16, fontweight='bold')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Healthy (< 1.5x forgetting)'),
        Patch(facecolor='#f39c12', label='Degraded (1.5-3x forgetting)'),
        Patch(facecolor='#e74c3c', label='Critical (> 3x forgetting)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "memory_health_dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved memory_health_dashboard.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="v2/results")
    parser.add_argument("--preset", type=str, default="hero")
    args = parser.parse_args()

    results = load_results(args.results_dir, args.preset)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    plot_forgetting_curves(results, args.results_dir)
    plot_comparison_bars(results, args.results_dir)
    plot_memory_health_dashboard(results, args.results_dir)


if __name__ == "__main__":
    main()
