"""
V4 Plot: Forgetting factor vs increment size.
One line per method, the key output of the experiment.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font setup
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_grid_results(results_dir="v4/results"):
    """Load all result JSONs from the grid."""
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and fname != "full_grid.json":
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    # Also try full_grid.json
    fg = os.path.join(results_dir, "full_grid.json")
    if os.path.exists(fg):
        with open(fg) as f:
            results = json.load(f)
    return results


def compute_forgetting_from_result(result):
    """Extract forgetting factors from a single run result."""
    phases = result.get("phases", {})
    domain_keys = list(phases.keys())
    if len(domain_keys) < 2:
        return None

    last_key = domain_keys[-1]
    final_ppls = phases[last_key].get("perplexity", {})

    forgetting = {}
    for pk in domain_keys[:-1]:  # Exclude last phase (tautology)
        after_learn = phases[pk].get("perplexity", {}).get(pk)
        after_all = final_ppls.get(pk)
        if after_learn and after_all and after_learn > 0:
            forgetting[pk] = after_all / after_learn

    if not forgetting:
        return None
    return sum(forgetting.values()) / len(forgetting)


def plot_forgetting_vs_increment(results_dir="v4/results", output_path="v4/results/forgetting_vs_increment.png"):
    """Plot the key figure: forgetting factor vs increment size."""
    results = load_grid_results(results_dir)
    if not results:
        print("No results found")
        return

    # Organize: {model: {method: {increment: avg_forgetting}}}
    data = {}
    for r in results:
        model = r.get("model", "unknown")
        method = r.get("method", "unknown")
        inc = r.get("increment_size", 0)
        ff = compute_forgetting_from_result(r)
        if ff is not None:
            data.setdefault(model, {}).setdefault(method, {})[inc] = ff

    # Plot per model
    for model_name, methods_data in data.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for method_name, inc_data in methods_data.items():
            increments = sorted(inc_data.keys())
            forgettings = [inc_data[i] for i in increments]
            labels = ["full-phase" if i == 0 else str(i) for i in increments]

            marker = "o" if method_name == "avr" else ("s" if method_name == "naive" else "^")
            linestyle = "-" if method_name == "avr" else ("--" if method_name == "naive" else ":")
            ax.plot(range(len(increments)), forgettings, 
                    marker=marker, linestyle=linestyle, linewidth=2, 
                    label=method_name.upper())

            # Mark where EWC is undefined
            if method_name == "ewc":
                for i, inc in enumerate(increments):
                    if inc > 0 and inc < 200:
                        ax.annotate("undefined", (i, forgettings[i]),
                                   textcoords="offset points", xytext=(0, 10),
                                   ha='center', fontsize=8, color='red')

        ax.set_xlabel("Increment Size", fontsize=12)
        ax.set_ylabel("Avg Forgetting Factor", fontsize=12)
        ax.set_title(f"Streaming AVR: Forgetting vs Increment Size\n{model_name}", fontsize=13)
        ax.set_xticks(range(len(increments)))
        ax.set_xticklabels(labels)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No forgetting')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = output_path.replace(".png", f"_{model_name}.png")
        plt.savefig(out, dpi=150)
        print(f"Plot saved to {out}")
        plt.close()


def print_summary(results_dir="v4/results"):
    """Print a text summary table."""
    results = load_grid_results(results_dir)
    if not results:
        print("No results found")
        return

    print(f"\n{'='*70}")
    print(f"STREAMING AVR RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Method':<8} {'Increment':<10} {'Avg FF':<10} {'Repairs':<8} {'EWC?':<8}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: (x.get("model",""), x.get("method",""), x.get("increment_size",0))):
        model = r.get("model", "?")
        method = r.get("method", "?")
        inc = r.get("increment_size", 0)
        ff = compute_forgetting_from_result(r)
        inc_str = "full" if inc == 0 else str(inc)

        # Get repairs and EWC status
        last_phase = list(r.get("phases", {}).values())[-1] if r.get("phases") else {}
        repairs = last_phase.get("total_repairs", "-")
        ewc_ok = last_phase.get("ewc_computable", "-")

        ff_str = f"{ff:.2f}x" if ff else "N/A"
        ewc_str = str(ewc_ok) if ewc_ok != "-" else "-"

        print(f"{model:<12} {method:<8} {inc_str:<10} {ff_str:<10} {repairs:<8} {ewc_str:<8}")

    print(f"{'='*70}")


if __name__ == "__main__":
    print_summary()
    plot_forgetting_vs_increment()
