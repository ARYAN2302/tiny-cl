"""
Generate forgetting curves and comparison plots from experiment results.
"""

import os
import json
import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Font setup for compatibility
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PHASE_ORDER = ["A", "B", "C"]
PHASE_COLORS = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800"}

METHOD_COLORS = {
    "naive": "#F44336",
    "freeze": "#9C27B0",
    "replay": "#FF9800",
    "anchor_cont": "#2196F3",
    "anchor_disc": "#4CAF50",
}

METHOD_DISPLAY = {
    "naive": "Naive SGD",
    "freeze": "Freeze Layers",
    "replay": "Blind Replay 1%",
    "anchor_cont": "Anchor-AVR (Cont.)",
    "anchor_disc": "Anchor-AVR (Disc.)",
}


def load_results(results_dir: str) -> dict:
    """Load all experiment result files."""
    all_results = {}
    
    for filepath in glob.glob(os.path.join(results_dir, "results_*.json")):
        filename = os.path.basename(filepath)
        # Parse: results_<method>_<model_size>.json
        parts = filename.replace("results_", "").replace(".json", "").split("_")
        if len(parts) >= 2:
            method_name = "_".join(parts[:-1])
            model_size = parts[-1]
            key = f"{method_name}_{model_size}"
        else:
            key = parts[0]
        
        with open(filepath, "r") as f:
            all_results[key] = json.load(f)
    
    return all_results


def plot_forgetting_curves(all_results: dict, model_size: str = "30M", output_dir: str = "results"):
    """
    Plot forgetting curves: perplexity on each phase over time.
    
    For each method, show how perplexity on Phase A, B, C changes
    as the model learns new phases.
    """
    # Filter to specific model size
    methods = {}
    for key, data in all_results.items():
        if key.endswith(f"_{model_size}") and "error" not in data:
            method_name = key.replace(f"_{model_size}", "")
            methods[method_name] = data
    
    if not methods:
        print(f"No results found for model size {model_size}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Forgetting Curves — {model_size} Model", fontsize=16, fontweight='bold')
    
    for ax_idx, phase_key in enumerate(PHASE_ORDER):
        ax = axes[ax_idx]
        ax.set_title(f"Phase {phase_key} Perplexity", fontsize=13)
        ax.set_xlabel("Training Stage")
        ax.set_ylabel("Perplexity")
        
        for method_name, data in methods.items():
            phase_results = data.get("phase_results", {})
            
            # Track perplexity on this phase after each training stage
            ppl_values = []
            stages = []
            
            for stage in PHASE_ORDER:
                if stage in phase_results and phase_key in phase_results[stage]:
                    ppl_values.append(phase_results[stage][phase_key])
                    stages.append(f"After {stage}")
            
            # Add final eval
            if "final" in phase_results and phase_key in phase_results["final"]:
                ppl_values.append(phase_results["final"][phase_key])
                stages.append("Final")
            
            if ppl_values:
                color = METHOD_COLORS.get(method_name, "#666666")
                label = METHOD_DISPLAY.get(method_name, method_name)
                ax.plot(range(len(ppl_values)), ppl_values, 
                       marker='o', label=label, color=color, linewidth=2, markersize=6)
        
        ax.set_xticks(range(len(stages) if stages else [0]))
        if stages:
            ax.set_xticklabels(stages, rotation=30, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, f"forgetting_curves_{model_size}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_comparison_bar(all_results: dict, model_size: str = "30M", output_dir: str = "results"):
    """
    Bar chart comparing final perplexity on each phase across methods.
    """
    methods = {}
    for key, data in all_results.items():
        if key.endswith(f"_{model_size}") and "error" not in data:
            method_name = key.replace(f"_{model_size}", "")
            methods[method_name] = data
    
    if not methods:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    n_methods = len(methods)
    n_phases = len(PHASE_ORDER)
    bar_width = 0.8 / n_methods
    x = np.arange(n_phases)
    
    for i, (method_name, data) in enumerate(methods.items()):
        phase_results = data.get("phase_results", {})
        final = phase_results.get("final", {})
        
        ppls = []
        for phase in PHASE_ORDER:
            ppl = final.get(phase, float('inf'))
            ppls.append(min(ppl, 500))  # Cap for visualization
        
        color = METHOD_COLORS.get(method_name, "#666666")
        label = METHOD_DISPLAY.get(method_name, method_name)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, ppls, bar_width, label=label, color=color, alpha=0.85)
    
    ax.set_xlabel("Phase", fontsize=12)
    ax.set_ylabel("Final Perplexity (lower = better)", fontsize=12)
    ax.set_title(f"Final Perplexity Comparison — {model_size} Model", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Phase {p}" for p in PHASE_ORDER])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    filepath = os.path.join(output_dir, f"comparison_bar_{model_size}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_bwt_comparison(all_results: dict, model_size: str = "30M", output_dir: str = "results"):
    """
    Bar chart comparing BWT and storage across methods.
    """
    methods = {}
    for key, data in all_results.items():
        if key.endswith(f"_{model_size}") and "error" not in data:
            method_name = key.replace(f"_{model_size}", "")
            methods[method_name] = data
    
    if not methods:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    method_names = []
    bwt_values = []
    storage_values = []
    colors = []
    
    for method_name, data in methods.items():
        method_names.append(METHOD_DISPLAY.get(method_name, method_name))
        bwt_values.append(data.get("bwt", 0))
        storage_values.append(data.get("storage_kb", 0))
        colors.append(METHOD_COLORS.get(method_name, "#666666"))
    
    # BWT comparison
    bars1 = ax1.barh(method_names, bwt_values, color=colors, alpha=0.85)
    ax1.set_xlabel("Backward Transfer (higher = less forgetting)", fontsize=11)
    ax1.set_title("Backward Transfer", fontsize=13, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Storage comparison
    bars2 = ax2.barh(method_names, storage_values, color=colors, alpha=0.85)
    ax2.set_xlabel("Storage (KB)", fontsize=11)
    ax2.set_title("Storage Overhead", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    filepath = os.path.join(output_dir, f"bwt_storage_comparison_{model_size}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_forgetting_heatmap(all_results: dict, model_size: str = "30M", output_dir: str = "results"):
    """
    Heatmap showing forgetting per phase per method.
    """
    methods = {}
    for key, data in all_results.items():
        if key.endswith(f"_{model_size}") and "error" not in data:
            method_name = key.replace(f"_{model_size}", "")
            methods[method_name] = data
    
    if not methods:
        return
    
    method_order = list(methods.keys())
    n_methods = len(method_order)
    n_phases = len(PHASE_ORDER)
    
    # Compute forgetting: ppl increase from right-after-learning to final
    forgetting_matrix = np.zeros((n_methods, n_phases))
    
    for i, method_name in enumerate(method_order):
        data = methods[method_name]
        phase_results = data.get("phase_results", {})
        final = phase_results.get("final", {})
        
        for j, phase in enumerate(PHASE_ORDER):
            # PPL right after learning this phase
            after_learning = phase_results.get(phase, {}).get(phase, float('inf'))
            # PPL after all training
            after_all = final.get(phase, float('inf'))
            
            if after_learning != float('inf') and after_all != float('inf'):
                forgetting_matrix[i, j] = after_all - after_learning
            else:
                forgetting_matrix[i, j] = 0
    
    fig, ax = plt.subplots(figsize=(8, max(4, n_methods * 0.8)), constrained_layout=True)
    
    im = ax.imshow(forgetting_matrix, cmap='Reds', aspect='auto')
    ax.set_xticks(range(n_phases))
    ax.set_xticklabels([f"Phase {p}" for p in PHASE_ORDER])
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in method_order])
    ax.set_title(f"Forgetting (PPL Increase) — {model_size} Model", fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(n_methods):
        for j in range(n_phases):
            val = forgetting_matrix[i, j]
            text_color = 'white' if val > forgetting_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f"{val:.1f}", ha='center', va='center', color=text_color, fontsize=10)
    
    fig.colorbar(im, ax=ax, label='PPL Increase (higher = more forgetting)')
    
    filepath = os.path.join(output_dir, f"forgetting_heatmap_{model_size}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def generate_all_plots(results_dir: str = "results", model_sizes: list = None):
    """Generate all plots from saved results."""
    if model_sizes is None:
        model_sizes = ["30M"]
    
    all_results = load_results(results_dir)
    
    if not all_results:
        print(f"No results found in {results_dir}/")
        return
    
    print(f"Loaded {len(all_results)} experiment results")
    
    for model_size in model_sizes:
        print(f"\nGenerating plots for {model_size} model...")
        plot_forgetting_curves(all_results, model_size, results_dir)
        plot_comparison_bar(all_results, model_size, results_dir)
        plot_bwt_comparison(all_results, model_size, results_dir)
        plot_forgetting_heatmap(all_results, model_size, results_dir)
    
    print(f"\nAll plots saved to {results_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-sizes", nargs="+", default=["30M"])
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.model_sizes)
