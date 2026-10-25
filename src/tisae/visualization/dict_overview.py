"""Dictionary overview figure for paper appendix.

Two-panel figure:
  (a) Mortality OR distribution histogram with risk-tier shading
  (b) Top 10 high-risk + top 10 protective latent concepts with clinical labels

Usage:
    python -m tisae.visualization.dict_overview \
        --dictionary_path experiments/discrete/mimic3_mortality/latent_dictionary_v1/latent_dictionary.json \
        --output ./outputs/dict_overview.png
"""
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_dict_overview(
    dictionary: List[dict],
    output_path: str,
    dpi: int = 300,
    max_label_len: int = 50,
) -> Path:
    ors = np.array([e["mortality_odds_ratio"] for e in dictionary])
    ors_clipped = np.clip(ors, 0.05, None)
    log_ors = np.log2(ors_clipped)

    fig = plt.figure(figsize=(7.5, 9.0))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[0.9, 1.5], hspace=0.30,
        left=0.32, right=0.95, top=0.96, bottom=0.05,
    )

    # ── Panel (a): OR distribution ──
    ax_a = fig.add_subplot(gs[0])

    xlim = (log_ors.min() - 0.3, log_ors.max() + 0.3)
    bins = np.linspace(xlim[0], xlim[1], 45)
    _, _, patches = ax_a.hist(
        log_ors, bins=bins, edgecolor="white", linewidth=0.4, alpha=0.9, zorder=2,
    )

    for patch, left_edge in zip(patches, bins[:-1]):
        center = left_edge + (bins[1] - bins[0]) / 2
        if center < np.log2(0.5):
            patch.set_facecolor("#27ae60")
        elif center > np.log2(2):
            patch.set_facecolor("#c0392b")
        else:
            patch.set_facecolor("#7f8c8d")

    ax_a.axvspan(xlim[0], np.log2(0.5), alpha=0.06, color="#27ae60", zorder=0)
    ax_a.axvspan(np.log2(2), xlim[1], alpha=0.06, color="#c0392b", zorder=0)
    ax_a.axvline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.3, zorder=1)
    ax_a.axvline(np.log2(0.5), color="#27ae60", linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)
    ax_a.axvline(np.log2(2), color="#c0392b", linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)

    n_prot = sum(1 for o in ors if o < 0.5)
    n_neut = sum(1 for o in ors if 0.5 <= o <= 2)
    n_risk = sum(1 for o in ors if o > 2)
    ymax = ax_a.get_ylim()[1]

    ax_a.text(
        np.log2(0.18), ymax * 0.88,
        f"Protective\nn={n_prot} ({n_prot / len(ors) * 100:.0f}%)",
        ha="center", va="top", fontsize=8, color="#1a7a3a", fontweight="bold",
    )
    ax_a.text(
        np.log2(1), ymax * 0.88,
        f"Neutral\nn={n_neut} ({n_neut / len(ors) * 100:.0f}%)",
        ha="center", va="top", fontsize=8, color="#555", fontweight="bold",
    )
    ax_a.text(
        np.log2(5), ymax * 0.88,
        f"High-risk\nn={n_risk} ({n_risk / len(ors) * 100:.0f}%)",
        ha="center", va="top", fontsize=8, color="#922b21", fontweight="bold",
    )

    tick_ors = [0.1, 0.25, 0.5, 1, 2, 4, 8]
    ax_a.set_xticks([np.log2(t) for t in tick_ors])
    ax_a.set_xticklabels([str(t) for t in tick_ors], fontsize=9)
    ax_a.set_xlabel("Mortality Odds Ratio", fontsize=10)
    ax_a.set_ylabel("Number of Latents", fontsize=10)
    ax_a.set_xlim(xlim)
    for sp in ["top", "right"]:
        ax_a.spines[sp].set_visible(False)
    ax_a.set_title(
        "(a) Mortality Odds Ratio Distribution",
        fontsize=10, fontweight="bold", loc="left",
    )

    # ── Panel (b): Top 10 high-risk + Top 10 protective ──
    ax_b = fig.add_subplot(gs[1])

    prot_pool = [e for e in dictionary if e["mortality_odds_ratio"] > 0]
    top_risk = sorted(dictionary, key=lambda e: e["mortality_odds_ratio"], reverse=True)[:10]
    top_prot = sorted(prot_pool, key=lambda e: e["mortality_odds_ratio"])[:10]
    top_prot = list(reversed(top_prot))

    gap = 1.0
    y_risk = list(range(10))
    y_prot = [i + 10 + gap for i in range(10)]
    y_all = y_risk + y_prot
    entries = top_risk + top_prot

    labels = []
    bar_vals = []
    colors = []
    for e in entries:
        lab = e.get("clinical_label", f"k={e['latent_k']}")
        if len(lab) > max_label_len:
            lab = lab[: max_label_len - 2] + ".."
        labels.append(lab)
        orv = e["mortality_odds_ratio"]
        bar_vals.append(np.log2(max(orv, 0.05)))
        colors.append("#c0392b" if orv > 2 else "#27ae60")

    ax_b.barh(y_all, bar_vals, color=colors, edgecolor="white", height=0.72, alpha=0.85)

    for yp, bv, e in zip(y_all, bar_vals, entries):
        orv = e["mortality_odds_ratio"]
        freq = e["activation_frequency"]
        txt = f"{orv:.2f}  ({freq * 100:.1f}%)"
        if orv >= 1:
            ax_b.text(bv + 0.08, yp, txt, va="center", fontsize=7, color="#444")
        else:
            ax_b.text(0.08, yp, txt, va="center", fontsize=7, color="#444")

    ax_b.set_yticks(y_all)
    ax_b.set_yticklabels(labels, fontsize=7.5)
    ax_b.invert_yaxis()
    ax_b.axvline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.3)
    ax_b.axhline(9.5 + gap / 2, color="#bbb", linestyle="-", linewidth=0.6, alpha=0.6)

    tick_ors_b = [0.1, 0.25, 0.5, 1, 2, 4, 8]
    ax_b.set_xticks([np.log2(t) for t in tick_ors_b])
    ax_b.set_xticklabels([str(t) for t in tick_ors_b], fontsize=9)
    ax_b.set_xlabel(
        "Mortality Odds Ratio (annotation: OR value, activation frequency)", fontsize=9,
    )
    for sp in ["top", "right"]:
        ax_b.spines[sp].set_visible(False)
    ax_b.set_title(
        "(b) Top 10 High-Risk and Protective Latent Concepts",
        fontsize=10, fontweight="bold", loc="left",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Dictionary overview figure")
    parser.add_argument("--dictionary_path", required=True)
    parser.add_argument("--output", default="./outputs/dict_overview.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with open(args.dictionary_path) as f:
        dictionary = json.load(f)

    out = plot_dict_overview(dictionary, args.output, dpi=args.dpi)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
