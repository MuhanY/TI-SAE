"""Generate concept knockout figure: PS-ranked vs random knockout curves.

Usage:
    python -m tisae.visualization.concept_knockout_fig \
        --results experiments_ablation/concept_knockout/knockout_results.json \
        --output figures/concept_knockout.pdf
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_knockout(results_path, output_path, light=False):
    with open(results_path) as f:
        data = json.load(f)

    # Handle both standalone knockout and nested OOD format
    if "concept_knockout" in data:
        data = data["concept_knockout"]
    ks = data["knockout_ks"]
    ps = data["ps_ranked"]
    rd = data["random"]
    n_global = data["n_global_pred"]

    ps_prob = [ps[str(k)]["mean_delta_prob"] for k in ks]
    ps_std = [ps[str(k)]["std_delta_prob"] for k in ks]
    rd_prob = [rd[str(k)]["mean_delta_prob"] for k in ks]
    rd_std = [rd[str(k)]["std_delta_prob"] for k in ks]

    # SAE baseline (no knockout)
    sae_prob = ps["sae_full"]["mean_delta_prob"]

    if light:
        plt.style.use("default")
        bg_color = "white"
        text_color = "black"
        ps_color = "#d62728"
        rd_color = "#7f7f7f"
        sae_color = "#2ca02c"
        grid_alpha = 0.3
    else:
        plt.style.use("dark_background")
        bg_color = "black"
        text_color = "white"
        ps_color = "#ff6b6b"
        rd_color = "#888888"
        sae_color = "#69db7c"
        grid_alpha = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=330)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Fill between PS and random
    ax.fill_between(ks, rd_prob, ps_prob, alpha=0.15, color=ps_color)

    # PS-ranked curve (line + markers at select points)
    ax.plot(ks, ps_prob, "-", color=ps_color, linewidth=2, zorder=3,
            label="Importance-ranked")
    marker_ks = [1, 16, 32, 48, 64]
    mk_idx = [ks.index(k) for k in marker_ks if k in ks]
    ax.plot([ks[i] for i in mk_idx], [ps_prob[i] for i in mk_idx],
            "o", color=ps_color, markersize=5, zorder=4)

    # Random curve
    ax.plot(ks, rd_prob, "--", color=rd_color, linewidth=1.5, zorder=3,
            label=f"Random order (avg {data['n_random_trials']} trials)")
    ax.plot([ks[i] for i in mk_idx], [rd_prob[i] for i in mk_idx],
            "s", color=rd_color, markersize=4, zorder=4)

    # SAE baseline
    ax.axhline(sae_prob, color=sae_color, linestyle=":", linewidth=1,
               label="SAE reconstruction", zorder=2)

    ax.set_xlabel("Concepts knocked out ($k$)", fontsize=11)
    ax.set_ylabel("$\\Delta p$ (prediction shift)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    tick_ks = [1, 16, 32, 48, 64]
    ax.set_xticks(tick_ks)
    ax.set_xlim(0, max(ks) + 1)
    ax.grid(True, alpha=grid_alpha)
    ax.set_ylim(0, None)


    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=bg_color)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/concept_knockout.pdf")
    parser.add_argument("--light", action="store_true", help="White background")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_knockout(args.results, args.output, light=args.light)


if __name__ == "__main__":
    main()
