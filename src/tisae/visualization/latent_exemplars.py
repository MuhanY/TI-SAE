"""
Generate exemplar latent concept figures for paper appendix.

Layout per latent:
  - Island track (top): per-episode Gaussian windows (mu +/- sigma), staggered by episode
  - Variable strips (below): raw values on clinically meaningful y-axes, color-coded by episode

Usage:
    python -m tisae.visualization.latent_exemplars \
        --dictionary_path experiments/.../latent_dictionary.json \
        --output_dir ./outputs/latent_exemplars \
        --latent_keys 268,958,886,466 \
        --episode_ids '{"268":[27612,25928,26855,24229]}' \
        --skip_vars '{"554":["NTproBNP"]}' \
        --title "Exemplar Latent Concepts"
"""

from __future__ import annotations

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Clinically meaningful y-ranges: (lo, hi), unit
VAR_INFO = {
    "BUN": ((5, 80), "mg/dL"),
    "Creatinine Blood": ((0.3, 6), "mg/dL"),
    "Urine": ((0, 600), "mL"),
    "Total CO2": ((10, 38), "mEq/L"),
    "Base Excess": ((-22, 8), "mEq/L"),
    "Insulin NPH": ((0, 50), "units"),
    "CRR": ((-0.1, 1.2), "0/1"),
    "Lactate": ((0, 12), "mmol/L"),
    "Neosynephrine": ((0, 4), "mcg/kg/min"),
    "Intubated": ((-0.1, 1.2), "0/1"),
    "O2 Saturation": ((85, 100), "%"),
    "SBP": ((70, 190), "mmHg"),
    "DBP": ((30, 110), "mmHg"),
    "NTproBNP": ((0, 5000), "pg/mL"),
    "FiO2": ((0.2, 1.0), ""),
    "Temperature": ((96, 102), "°F"),
    "Cefazolin": ((0, 4), "g"),
    "Eoisinophils": ((0, 8), "%"),
    "Magnesium Sulfate (Bolus)": ((0, 6), "g"),
    "HR": ((40, 170), "bpm"),
    "MBP": ((35, 130), "mmHg"),
    "D5W": ((0, 400), "mL/hr"),
    "RR": ((5, 50), "breaths/min"),
    "PCO2": ((20, 80), "mmHg"),
    "pH Blood": ((7.0, 7.6), ""),
    "KCl": ((0, 120), "mEq"),
    "Insulin Regular": ((0, 30), "units"),
    "RDW": ((11, 25), "%"),
    "Hgb": ((4, 17), "g/dL"),
    "Packed RBC": ((0, 6), "units"),
    "Propofol": ((0, 100), "mcg/kg/min"),
    "Lorazepam": ((0, 10), "mg"),
    "Midazolam": ((0, 10), "mg"),
    "GCS_eye": ((0.5, 4.5), ""),
    "Hydromorphone": ((0, 5), "mg"),
    "Morphine Sulfate": ((0, 20), "mg"),
    "Norepinephrine": ((0, 2), "mcg/kg/min"),
    "Epinephrine": ((0, 2), "mcg/kg/min"),
    "Vasopressin": ((0, 5), "units/hr"),
    "Milrinone": ((0, 1), "mcg/kg/min"),
    "Nitroglycerine": ((0, 10), "mcg/kg/min"),
    "Bicarbonate": ((10, 40), "mEq/L"),
    "Chloride": ((90, 120), "mEq/L"),
    "Sodium": ((125, 155), "mEq/L"),
    "Calcium Free": ((0.8, 1.5), "mmol/L"),
    "Albumin": ((1, 5), "g/dL"),
    "Bilirubin (Total)": ((0, 15), "mg/dL"),
    "WBC": ((0, 30), "K/uL"),
    "INR": ((0.8, 6), ""),
    "Troponin I": ((0, 20), "ng/mL"),
    "Fentanyl": ((0, 300), "mcg/hr"),
    "MCV": ((70, 110), "fL"),
    "MCH": ((25, 38), "pg"),
    "MCHC": ((30, 38), "g/dL"),
    "Hct": ((15, 55), "%"),
    "RBC": ((2, 6), "M/uL"),
    "Ammonia": ((0, 200), "umol/L"),
    "GCS_verbal": ((0.5, 5.5), ""),
}

EP_COLORS = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
EP_MARKERS = ["o", "s", "^", "D"]
EP_MARKER_SIZES = [26, 22, 24, 20]


def _get_selected_episodes(
    latent: dict, selected_ids: List[int],
) -> List[dict]:
    ep_map = {ep["episode_idx"]: ep for ep in latent["exemplar_episodes_sae"]}
    return [ep_map[eid] for eid in selected_ids if eid in ep_map]


def _top_islands(islands: List[dict], n: int = 2) -> List[dict]:
    return sorted(islands, key=lambda x: x["scale"], reverse=True)[:n]


def _compute_ylim(
    vname: str, exemplars: List[dict],
) -> Tuple[float, float, str]:
    """Compute y-axis limits: clinical default expanded to fit data."""
    if vname in VAR_INFO:
        (default_lo, default_hi), unit = VAR_INFO[vname]
    else:
        default_lo, default_hi = 0, 1
        unit = ""

    all_vals = []
    for ep in exemplars:
        all_vals.extend(
            [p["val"] for p in ep["variable_traces"].get(vname, [])]
        )

    if all_vals:
        data_min, data_max = min(all_vals), max(all_vals)
        pad = max((data_max - data_min) * 0.12, 0.3)
        y_lo = min(default_lo, data_min - pad)
        y_hi = max(default_hi, data_max + pad)
        y_lo = max(y_lo, data_min - (data_max - data_min) * 0.5 - pad)
        y_hi = min(y_hi, data_max + (data_max - data_min) * 0.5 + pad)
        y_lo = min(y_lo, default_lo)
        y_hi = max(y_hi, default_hi)
    else:
        y_lo, y_hi = default_lo, default_hi

    return y_lo, y_hi, unit


def render_exemplar_page(
    latent_keys: List[Tuple[int, int]],
    dd_map: Dict[Tuple[int, int], dict],
    selected_eps: Dict[int, List[int]],
    output_path: str,
    title: str = "Exemplar Latent Concepts",
    skip_vars: Optional[Dict[int, Set[str]]] = None,
    sigma_mult: float = 1.0,
    n_top_islands: int = 2,
    dpi: int = 300,
) -> Path:
    """Render a page of exemplar latent concept figures.

    Args:
        latent_keys: list of (k, c) tuples
        dd_map: {(k, c): dict} from latent_dictionary.json
        selected_eps: {k: [ep_idx, ...]} hand-picked episodes
        output_path: where to save
        title: figure suptitle
        skip_vars: {k: {"VarName", ...}} variables to skip per latent
        sigma_mult: island width = mu +/- sigma_mult * sigma
        n_top_islands: how many islands to show per episode
        dpi: output resolution
    """
    skip_vars = skip_vars or {}
    n_latents = len(latent_keys)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.5,
    })

    fig = plt.figure(figsize=(10, n_latents * 4.0))
    outer = gridspec.GridSpec(n_latents, 1, hspace=0.42, figure=fig)
    if title:
        fig.suptitle(
            title, fontsize=11, fontweight="bold", y=0.998, color="#1a1a1a",
        )

    for lat_i, (k, c) in enumerate(latent_keys):
        latent = dd_map[(k, c)]
        vars_data = latent["variable_composition"][:3]
        var_names = [v["var_name"] for v in vars_data]
        var_weights = [v["weight"] for v in vars_data]

        sk = skip_vars.get(k, set())
        display_vars = [
            (vn, vw) for vn, vw in zip(var_names, var_weights) if vn not in sk
        ]
        n_vars = len(display_vars)

        inner = gridspec.GridSpecFromSubplotSpec(
            1 + n_vars, 1, subplot_spec=outer[lat_i],
            hspace=0.05, height_ratios=[0.3] + [1.0] * n_vars,
        )

        exemplars = _get_selected_episodes(latent, selected_eps.get(k, []))
        n_eps = len(exemplars)

        # --- Island track ---
        ax_isl = fig.add_subplot(inner[0])

        for ep_i, ep in enumerate(exemplars):
            islands = _top_islands(ep["island_windows"], n_top_islands)
            ep_color = EP_COLORS[ep_i % len(EP_COLORS)]
            y_center = n_eps - 1 - ep_i

            for isl in islands:
                mu, sigma = isl["mu"], isl["sigma"]
                lo_x = max(0, mu - sigma_mult * sigma)
                hi_x = min(48, mu + sigma_mult * sigma)
                ax_isl.fill_between(
                    [lo_x, hi_x], y_center - 0.35, y_center + 0.35,
                    color=ep_color, alpha=0.45, zorder=1,
                    edgecolor=ep_color, linewidth=0.6,
                )
                ax_isl.plot(
                    [mu, mu], [y_center - 0.35, y_center + 0.35],
                    color=ep_color, linewidth=0.6, alpha=0.8, zorder=2,
                )

        ax_isl.set_ylim(-0.5, max(n_eps - 0.5, 0.5))
        ax_isl.set_xlim(-0.5, 48.5)
        ax_isl.set_yticks([])
        ax_isl.tick_params(axis="x", labelbottom=False, length=2)
        ax_isl.set_xticks(range(0, 49, 6))
        ax_isl.grid(axis="x", alpha=0.12, linewidth=0.3)
        ax_isl.spines["top"].set_visible(False)
        ax_isl.spines["right"].set_visible(False)
        ax_isl.spines["left"].set_visible(False)
        ax_isl.text(
            -0.01, 0.5, f"Islands\n(μ±{sigma_mult:.0f}σ)" if sigma_mult == int(sigma_mult)
            else f"Islands\n(μ±{sigma_mult}σ)",
            transform=ax_isl.transAxes, fontsize=5.5,
            color="#888", va="center", ha="right",
        )

        letter = chr(ord("a") + lat_i)
        ax_isl.set_title(
            f"({letter})  k={k}: {latent['clinical_label']}   "
            f"(OR={latent['mortality_odds_ratio']:.2f},  "
            f"freq={latent['activation_frequency']:.1%})",
            fontsize=9, fontweight="bold", loc="left", pad=6,
            color="#1a1a1a",
        )

        for ep_i, ep in enumerate(exemplars):
            mort_char = "D" if ep["mortality"] else "S"
            y_center = n_eps - 1 - ep_i
            y_norm = (y_center + 0.5) / max(n_eps, 1)
            ax_isl.text(
                1.01, y_norm, f"#{ep['episode_idx']} ({mort_char})",
                transform=ax_isl.transAxes, fontsize=6,
                color=EP_COLORS[ep_i % len(EP_COLORS)],
                fontweight="bold", va="center", ha="left",
            )

        # --- Variable strips ---
        for v_i, (vname, vweight) in enumerate(display_vars):
            ax = fig.add_subplot(inner[1 + v_i])

            y_lo, y_hi, unit = _compute_ylim(vname, exemplars)

            for ep_i, ep in enumerate(exemplars):
                traces = ep["variable_traces"].get(vname, [])
                marker = EP_MARKERS[ep_i % len(EP_MARKERS)]
                ep_color = EP_COLORS[ep_i % len(EP_COLORS)]
                ms = EP_MARKER_SIZES[ep_i % len(EP_MARKER_SIZES)]

                if traces:
                    ts = [p["t"] for p in traces]
                    vals = [p["val"] for p in traces]
                    ax.scatter(
                        ts, vals, c=ep_color, s=ms, marker=marker,
                        alpha=0.8, zorder=5, edgecolors="white",
                        linewidths=0.4,
                    )
                    sorted_pts = sorted(zip(ts, vals))
                    if len(sorted_pts) > 1:
                        ax.plot(
                            [p[0] for p in sorted_pts],
                            [p[1] for p in sorted_pts],
                            color=ep_color, alpha=0.25, linewidth=0.6,
                            zorder=3,
                        )

            ax.set_ylim(y_lo, y_hi)
            ax.tick_params(axis="y", labelsize=7, pad=2)
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))

            short_name = vname if len(vname) <= 20 else vname[:18] + ".."
            unit_str = f" ({unit})" if unit else ""
            ax.text(
                1.01, 0.5, f"{short_name}{unit_str}  w={vweight:.2f}",
                transform=ax.transAxes, fontsize=6.5, color="#333",
                fontweight="bold", va="center", ha="left",
            )

            if v_i < n_vars - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.tick_params(axis="x", labelsize=7)
                ax.set_xlabel("Hour", fontsize=7.5, labelpad=2)

            ax.set_xlim(-0.5, 48.5)
            ax.set_xticks(range(0, 49, 6))
            ax.grid(axis="x", alpha=0.12, linewidth=0.3)
            ax.grid(axis="y", alpha=0.08, linewidth=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.subplots_adjust(right=0.80)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate exemplar latent concept figures")
    parser.add_argument("--dictionary_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--latent_keys", required=True,
                        help="Comma-separated k values (c=0 assumed)")
    parser.add_argument("--episode_ids", default=None,
                        help='JSON dict: {"268": [27612, 25928, ...]}')
    parser.add_argument("--skip_vars", default=None,
                        help='JSON dict: {"554": ["NTproBNP"]}')
    parser.add_argument("--title", default="")
    parser.add_argument("--sigma_mult", type=float, default=1.0)
    parser.add_argument("--n_top_islands", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with open(args.dictionary_path) as f:
        dictionary = json.load(f)

    dd_map = {(d["latent_k"], d["channel_c"]): d for d in dictionary}

    latent_keys = [(int(k.strip()), 0) for k in args.latent_keys.split(",")]

    selected_eps = {}
    if args.episode_ids:
        raw = json.loads(args.episode_ids)
        selected_eps = {int(k): v for k, v in raw.items()}
    else:
        for k, c in latent_keys:
            lat = dd_map[(k, c)]
            eps = sorted(
                lat["exemplar_episodes_sae"],
                key=lambda x: x["activation_strength"], reverse=True,
            )
            selected_eps[k] = [ep["episode_idx"] for ep in eps[:4]]

    skip_vars = {}
    if args.skip_vars:
        raw = json.loads(args.skip_vars)
        skip_vars = {int(k): set(v) for k, v in raw.items()}

    render_exemplar_page(
        latent_keys, dd_map, selected_eps, args.output_path,
        title=args.title, skip_vars=skip_vars,
        sigma_mult=args.sigma_mult, n_top_islands=args.n_top_islands,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
