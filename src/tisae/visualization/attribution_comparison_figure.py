"""
Figure 3 generator for TISAE paper.

Combined 4-column figure: Attention | IG | TSR | Decomposed TI-SAE
with variable filtering (top N by attribution mass) and channel-name y-axis.

Usage:
    python -m tisae.visualization.attribution_comparison_figure \
        --experiment_dir experiments/<backbone>/<condition>/tisae \
        --dataset_path data/unified_48h.pkl \
        --episode_pos 20 --output_dir ./outputs/fig3 \
        --top_n_vars 40
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

from tisae.models.ontology import CHANNEL_DEFINITIONS, create_ontology_mapping
from tisae.visualization.attribution_heatmaps import CHANNEL_COLORS, DEFAULT_CMAP


# ── Variable filtering ──────────────────────────────────────────────────────

def get_channel_sorting_filtered(
    var_indices: List[int],
    var_names: List[str],
    n_channels: int = 10,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Sort a filtered subset of variables by clinical channel."""
    ontology = create_ontology_mapping(var_names, n_channels=n_channels)
    ch_assign = ontology.get_channel_assignment().numpy()

    sorted_vars, ch_bounds, ch_labels = [], [], []
    for c in range(len(CHANNEL_DEFINITIONS)):
        ch_vars = [v for v in var_indices if ch_assign[v] == c]
        if ch_vars:
            ch_bounds.append(len(sorted_vars))
            ch_labels.append(CHANNEL_DEFINITIONS[c].name[:12])
            sorted_vars.extend(ch_vars)

    return np.array(sorted_vars), ch_bounds, ch_labels


def select_top_variables(
    attributions: Dict[str, np.ndarray],
    obs_grid: np.ndarray,
    top_n: int,
) -> List[int]:
    """Select top N variables by total attribution mass across all methods."""
    V = obs_grid.shape[1]
    total = np.zeros(V)
    for a in attributions.values():
        total += (np.abs(a) * obs_grid).sum(axis=0)
    obs_per_var = obs_grid.sum(axis=0)
    total[obs_per_var == 0] = 0

    if top_n >= V:
        return list(range(V))
    return sorted(np.argsort(total)[::-1][:top_n].tolist())


# ── Shared styling ───────────────────────────────────────────────────────────

def _set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black",
        "text.color": "white", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
    })


_REDS_BG = np.array(plt.cm.Reds(0.0)[:3])

def _set_light_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
    })


def _reset_style():
    plt.rcParams.update(matplotlib.rcParamsDefault)


def _save_optimized(fig, path, dpi=330, dark=True):
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="black" if dark else "white", edgecolor="none")
    try:
        from PIL import Image
        img = Image.open(path)
        img.save(path, "PNG", optimize=True)
    except ImportError:
        pass


# ── Panel (a): Baselines + Decomposed ───────────────────────────────────────

def plot_fig3_top(
    attributions: Dict[str, np.ndarray],
    obs_grid: np.ndarray,
    decomp_attr: np.ndarray,
    island_data: List[dict],
    latent_ch_map: np.ndarray,
    var_names: List[str],
    output_path: str,
    top_n_vars: int = 40,
    top_k: int = 64,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 330,
    dim_factor: float = 0.08,
    channel_colors: Optional[Dict[int, np.ndarray]] = None,
    dark: bool = True,
) -> Tuple[Path, Dict[int, int]]:
    """4-column figure: baselines + decomposed TI-SAE.

    Args:
        attributions: {method_name: [T, V]} for baseline methods
        obs_grid: [T, V] binary observation mask
        decomp_attr: [T, V] TI-SAE (PS-soft) attribution for decomposition
        island_data: list of island dicts from extract_sae_islands
        latent_ch_map: [K] latent -> channel index
        var_names: full variable name list (length V)
        top_n_vars: show only top N variables by attribution mass
        dark: use dark background (True) or white background (False)
    """
    if channel_colors is None:
        channel_colors = CHANNEL_COLORS

    # Variable filtering + channel sorting
    all_attr = dict(attributions)
    all_attr["_decomp"] = decomp_attr
    top_vars = select_top_variables(all_attr, obs_grid, top_n_vars)
    sorted_vars, ch_bounds, ch_labels = get_channel_sorting_filtered(
        top_vars, var_names)
    n_v = len(sorted_vars)

    # ── Decomposed RGB (same logic as plot_decomposed_heatmap) ──
    T = decomp_attr.shape[0]
    attr_masked = decomp_attr * obs_grid
    attr_sorted = attr_masked[:, sorted_vars]
    global_vmax = max(attr_sorted.max(), 1e-12)
    attr_norm = np.clip(attr_sorted / global_vmax, 0, 1)

    top_islands = island_data[:top_k]

    ch_latent_groups = {}
    for isl in top_islands:
        ch = latent_ch_map[isl["k"]]
        ch_latent_groups.setdefault(ch, {}).setdefault(isl["k"], []).append(isl)

    top1_per_ch = {}
    for ch, lat_groups in ch_latent_groups.items():
        def mass(k_islands):
            mask = np.zeros((T, n_v), dtype=np.float32)
            for isl in k_islands:
                mask = np.maximum(mask, isl["cell_mask"][:, sorted_vars])
            return (mask * attr_sorted).sum()
        best_k = max(lat_groups.items(), key=lambda kv: mass(kv[1]))[0]
        top1_per_ch[ch] = best_k

    if dark:
        heatmap_cmap = cmap
        base = plt.cm.get_cmap(cmap)(attr_norm)[:, :, :3]
        rgb = base * dim_factor
        for ch_idx, k in top1_per_ch.items():
            if ch_idx not in channel_colors:
                continue
            color = channel_colors[ch_idx]
            for isl in top_islands:
                if isl["k"] != k:
                    continue
                active = isl["cell_mask"][:, sorted_vars] > 0.5
                brightness = attr_norm * active
                for c in range(3):
                    rgb[:, :, c] = np.where(
                        active, color[c] * (0.3 + 0.7 * brightness), rgb[:, :, c])
    else:
        heatmap_cmap = "Reds"
        rgb = np.ones((T, n_v, 3), dtype=np.float32)
        for c in range(3):
            rgb[:, :, c] = _REDS_BG[c]
        for ch_idx, k in top1_per_ch.items():
            if ch_idx not in channel_colors:
                continue
            color = channel_colors[ch_idx]
            for isl in top_islands:
                if isl["k"] != k:
                    continue
                active = isl["cell_mask"][:, sorted_vars] > 0.5
                intensity = attr_norm * active
                blend = 0.1 + 0.9 * intensity
                for c in range(3):
                    rgb[:, :, c] = np.where(
                        active,
                        _REDS_BG[c] * (1.0 - blend) + color[c] * blend,
                        rgb[:, :, c])

    decomp_rgb = np.clip(rgb, 0, 1)

    # ── Styling ──
    text_color = "white" if dark else "black"
    label_color = "#cccccc" if dark else "#444444"
    sep_color = "#444444" if dark else "#cccccc"
    sep_color_decomp = "#222222" if dark else "#dddddd"
    bg_color = "black" if dark else "white"

    if dark:
        _set_dark_style()
    else:
        _set_light_style()

    # ── Figure layout ──
    methods = list(attributions.keys())
    n_cols = len(methods) + 1

    col_w = 3.0
    fig_h = max(n_v * 0.09, 3.0)
    fc = bg_color if dark else "white"
    fig = plt.figure(figsize=(col_w * n_cols + 1.0, fig_h), facecolor=fc)
    gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.08,
                           left=0.10, right=0.98, top=0.94, bottom=0.08)

    # Baseline columns
    for col_i, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, col_i])
        a = attributions[method] * obs_grid
        a_sorted = a[:, sorted_vars]
        amax = max(np.abs(a_sorted).max(), 1e-8)
        ax.imshow((a_sorted / amax).T, aspect="auto", origin="lower",
                  cmap=heatmap_cmap, vmin=0, vmax=1, interpolation="nearest")

        for b in ch_bounds[1:]:
            ax.axhline(b - 0.5, color=sep_color, lw=0.5)

        ax.set_yticks([])
        if col_i == 0:
            for bi, (b, lbl) in enumerate(zip(ch_bounds, ch_labels)):
                nxt = ch_bounds[bi + 1] if bi + 1 < len(ch_bounds) else n_v
                ax.text(-1.0, b + (nxt - b) / 2, lbl, fontsize=7,
                        va="center", ha="right", color=label_color,
                        fontweight="bold")

        ax.set_title(method, fontsize=11, color=text_color, fontweight="bold")
        ax.set_xticks([0, 12, 24, 36, 47])
        ax.set_xticklabels(["0", "12", "24", "36", "48"], fontsize=8)
        ax.set_xlabel("Time (hours)", fontsize=9)

    # Decomposed column
    ax_d = fig.add_subplot(gs[0, n_cols - 1])
    display = np.transpose(decomp_rgb, (1, 0, 2))
    ax_d.imshow(display, aspect="auto", origin="lower", interpolation="nearest")

    for b in ch_bounds[1:]:
        ax_d.axhline(b - 0.5, color=sep_color_decomp, lw=0.5)
    ax_d.set_yticks([])
    ax_d.set_title("TI-SAE (decomposed)", fontsize=11, color=text_color, fontweight="bold")
    ax_d.set_xticks([0, 12, 24, 36, 47])
    ax_d.set_xticklabels(["0", "12", "24", "36", "48"], fontsize=8)
    ax_d.set_xlabel("Time (hours)", fontsize=9)

    legend_elements = [
        Line2D([0], [0], marker="s", color=bg_color if dark else tuple(_REDS_BG),
               markerfacecolor=tuple(channel_colors.get(ch, np.array([0.5, 0.5, 0.5]))),
               markersize=5, label=f"k={top1_per_ch[ch]}", linewidth=0)
        for ch in sorted(top1_per_ch.keys())
        if ch in channel_colors
    ]
    legend_fc = "black" if dark else tuple(_REDS_BG)
    legend_tc = "white" if dark else "black"
    ax_d.legend(handles=legend_elements, loc="lower right", fontsize=5,
                facecolor=legend_fc, edgecolor=sep_color, labelcolor=legend_tc,
                framealpha=0.9)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_optimized(fig, out, dpi, dark=dark)
    plt.close(fig)
    _reset_style()
    return out, top1_per_ch


# ── Panel (b): Latent detail ─────────────────────────────────────────────────

def plot_fig3_bottom(
    attr_full: np.ndarray,
    island_data: List[dict],
    W_dec: np.ndarray,
    var_names: List[str],
    selected_latents: List[Tuple[int, int, str, str]],
    output_path: str,
    top_k: int = 64,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 330,
    channel_colors: Optional[Dict[int, np.ndarray]] = None,
    dark: bool = True,
) -> Path:
    """Per-latent detail with Gaussian envelopes + variable bars.

    Args:
        attr_full: [T, V] PS-soft attribution (obs-masked)
        island_data: PredSelector island dicts
        W_dec: [K, V] decoder weight matrix
        var_names: variable name list
        selected_latents: list of (ch_idx, k, channel_name, clinical_label)
        dark: use dark background (True) or white background (False)
    """
    if channel_colors is None:
        channel_colors = CHANNEL_COLORS

    heatmap_cmap = cmap if dark else "Reds"
    text_color = "white" if dark else "black"
    label_color = "#cccccc" if dark else "#444444"
    bg_color = "black" if dark else "white"

    ontology = create_ontology_mapping(var_names)
    ch_assign = ontology.get_channel_assignment().numpy()

    T, V = attr_full.shape
    global_vmax = max(attr_full.max(), 1e-12)
    top_islands = island_data[:top_k]
    t_axis = np.arange(T)

    n_c = len(selected_latents)
    max_ch_vars = max(
        len(np.where(ch_assign == ch_idx)[0])
        for ch_idx, _, _, _ in selected_latents
    )
    heatmap_h = max_ch_vars * 0.12
    gauss_h = 1.4
    bar_h = 1.6

    if dark:
        _set_dark_style()
    else:
        _set_light_style()
    fc = bg_color if dark else "white"
    fig = plt.figure(figsize=(5.0 * n_c, heatmap_h + gauss_h + bar_h + 1.2),
                     facecolor=fc)
    gs = gridspec.GridSpec(3, n_c, figure=fig,
                           height_ratios=[heatmap_h, gauss_h, bar_h],
                           hspace=0.40, wspace=0.30)

    for j, (ch_idx, k, ch_name_short, label) in enumerate(selected_latents):
        color_arr = channel_colors.get(ch_idx, np.array([0.5, 0.5, 0.5]))
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color_arr[0]*255), int(color_arr[1]*255), int(color_arr[2]*255))

        v_indices = np.where(ch_assign == ch_idx)[0]
        v_names_ch = [var_names[vi] for vi in v_indices]
        n_v = len(v_indices)
        ch_attr = attr_full[:, v_indices]
        k_islands = [isl for isl in top_islands if isl["k"] == k]

        lat_mask = np.zeros((T, n_v), dtype=np.float32)
        for isl in k_islands:
            for jj, vi in enumerate(v_indices):
                lat_mask[:, jj] = np.maximum(lat_mask[:, jj], isl["cell_mask"][:, vi])
        lat_attr = lat_mask * ch_attr

        w_row = W_dec[k]
        top_vi = np.argsort(w_row)[::-1][:5]
        top_var_list = [(var_names[vi], w_row[vi]) for vi in top_vi if w_row[vi] > 0.01]

        # Row 1: Heatmap (cropped to channel)
        ax_h = fig.add_subplot(gs[0, j])
        ax_h.imshow((lat_attr / global_vmax).T, aspect="auto", origin="lower",
                     cmap=heatmap_cmap, vmin=0, vmax=1, interpolation="nearest")
        heatmap_label_size = 8.5 if n_v > 16 else 9.5
        ax_h.set_yticks(range(n_v))
        ax_h.set_yticklabels(v_names_ch, fontsize=heatmap_label_size, color=label_color)
        title = f"{ch_name_short} — k={k}"
        if label:
            title += f'\n"{label}"'
        ax_h.set_title(title, fontsize=13, color=color_hex, fontweight="bold")
        ax_h.set_xlabel("Time (hours)", fontsize=11)
        ax_h.set_xticks([0, 12, 24, 36, 47])
        ax_h.set_xticklabels(["0", "12", "24", "36", "48"], fontsize=10)

        # Row 2: Gaussian envelopes
        ax_g = fig.add_subplot(gs[1, j])
        for isl in k_islands:
            g = np.exp(-0.5 * ((t_axis - isl["mu"]) / max(isl["sigma"], 1e-6)) ** 2)
            ax_g.fill_between(t_axis, 0, g, alpha=0.2, color=color_hex)
            ax_g.plot(t_axis, g, color=color_hex, lw=1.5)
            ax_g.axvline(isl["mu"], color=color_hex, ls=":", lw=0.8, alpha=0.5)
            ax_g.text(isl["mu"], 1.05, f'{isl["mu"]:.0f}h',
                      fontsize=9, ha="center", color=color_hex)
        ax_g.set_xlim(0, T - 1)
        ax_g.set_ylim(0, 1.2)
        if j == 0:
            ax_g.set_ylabel("g(t)", fontsize=11)
        ax_g.set_xlabel("Time (hours)", fontsize=11)
        ax_g.set_xticks([0, 12, 24, 36, 47])
        ax_g.set_xticklabels(["0", "12", "24", "36", "48"], fontsize=10)

        # Row 3: Variable composition bars
        ax_bar = fig.add_subplot(gs[2, j])
        if top_var_list:
            names = [n for n, w in top_var_list]
            weights = [w for n, w in top_var_list]
            bars = ax_bar.barh(range(len(names)), weights, color=color_hex,
                               alpha=0.8, height=0.6)
            ax_bar.set_yticks(range(len(names)))
            label_sizes = []
            for n in names:
                if n == "Platelet Count":
                    label_sizes.append(8.0)
                else:
                    label_sizes.append(9.5)
            ylabels = ax_bar.set_yticklabels(names, fontsize=9.5, color=label_color)
            for lbl, sz in zip(ylabels, label_sizes):
                lbl.set_fontsize(sz)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("Decoder weight", fontsize=11)
            for bar, w in zip(bars, weights):
                wt_fontsize = 7 if w >= 0.5 else 8
                ax_bar.text(bar.get_width() + 0.005,
                            bar.get_y() + bar.get_height() / 2,
                            f"{w:.2f}", va="center", fontsize=wt_fontsize, color=label_color)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_optimized(fig, out, dpi, dark=dark)
    plt.close(fig)
    _reset_style()
    return out


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Figure 3 generator")
    parser.add_argument("--experiment_dir", required=True,
                        help="TISAE experiment dir (with config.yaml, best_model.pt)")
    parser.add_argument("--eval_dir", default=None,
                        help="Eval dir with cached attributions (default: sibling eval/)")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--episode_pos", type=int, default=20,
                        help="Position in the 500 test samples")
    parser.add_argument("--output_dir", default="./outputs/fig3")
    parser.add_argument("--top_n_vars", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=330)
    parser.add_argument("--light", action="store_true", help="White background")
    args = parser.parse_args()
    use_dark = not args.light

    from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
    from tisae.models.backbone_adapter import load_backbone_adapter
    from tisae.models.tisae import TISAE, TISAEConfig
    from tisae.evaluation.faithfulness.attribution_methods import extract_pred_selector_softs
    from tisae.evaluation.faithfulness.run_evaluation import load_pred_selector

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.experiment_dir)

    # Resolve sibling dirs
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    else:
        eval_dir = exp_dir.parent / "eval"
    ps_dir = exp_dir.parent / "ps"

    # Load dataset
    ds = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": ds.T, "V": ds.V,
          "var_to_ind_hash": getattr(ds, "var_to_ind_hash", None)}

    # Load cached baseline attributions
    baselines = {}
    for name in ["attention", "ig", "tsr"]:
        data = torch.load(eval_dir / "attributions" / f"{name}.pt",
                          map_location="cpu", weights_only=False)
        baselines[name] = data["attr"][args.episode_pos].numpy()

    # Load PS-soft attribution
    ps_data = torch.load(eval_dir / "attributions" / "pred_selector_soft.pt",
                          map_location="cpu", weights_only=False)
    ps_attr = ps_data["attr"][args.episode_pos].numpy()
    obs_grid = ps_data["obs_grid"][args.episode_pos].numpy()

    # Load backbone + TISAE + PredSelector
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    tisae_cfg_dict = config.get("tisae_config", config.get("sae_config", {}))
    valid_fields = {f.name for f in fields(TISAEConfig)}
    tisae_cfg = TISAEConfig(**{k: v for k, v in tisae_cfg_dict.items() if k in valid_fields})

    bb_path = config.get("args", {}).get("backbone_checkpoint")
    if bb_path is None:
        bb_path = config.get("training", {}).get("backbone_checkpoint")

    adapter = load_backbone_adapter(bb_path, device=str(device), dataset_meta=dm)
    adapter.eval()

    sae = TISAE(tisae_cfg)
    ckpt = torch.load(exp_dir / "best_model.pt", map_location="cpu", weights_only=False)
    sae.load_state_dict(ckpt["sae_state_dict"], strict=False)
    sae = sae.to(device).eval()

    H_grid_mean = ckpt.get("H_grid_mean")
    if H_grid_mean is not None:
        H_grid_mean = H_grid_mean.to(device)

    # Load PredSelector
    ps_result = load_pred_selector(str(ps_dir), device=str(device))
    selector = ps_result["selector"]
    prior_cache = ps_result.get("prior_cache")

    # Prepare batch
    sample_idx = int(ps_data["indices"][args.episode_pos])
    sample = ds.get_item(sample_idx)
    batch = collate_fn([sample])
    batch = {k: v.to(device) for k, v in batch.items()}

    # Extract PredSelector islands (cell_masks are obs-filtered)
    _, all_islands = extract_pred_selector_softs(
        adapter, sae, selector, batch, ds.T, ds.V,
        H_grid_mean=H_grid_mean, prior_cache=prior_cache)
    island_data = all_islands[0]
    for isl in island_data:
        if isinstance(isl["cell_mask"], torch.Tensor):
            isl["cell_mask"] = isl["cell_mask"].cpu().numpy()

    # Channel assignment per latent
    W = sae.get_W_dec().detach().cpu().numpy()
    ontology = create_ontology_mapping(ds.var_names)
    ch_assign = ontology.get_channel_assignment().numpy()
    K = tisae_cfg.K
    latent_ch_map = np.zeros(K, dtype=int)
    for k in range(K):
        ch_mass = np.zeros(len(CHANNEL_DEFINITIONS))
        for v in range(ds.V):
            ch_mass[ch_assign[v]] += abs(W[k, v])
        latent_ch_map[k] = int(np.argmax(ch_mass))

    # W_dec for bottom panel
    W_dec = W

    del adapter, sae, selector
    torch.cuda.empty_cache()

    # Generate
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capitalize method names for display
    display_attr = {"Attention": baselines["attention"],
                    "IG": baselines["ig"],
                    "TI-SAE": ps_attr}

    ps_attr_masked = ps_attr * obs_grid

    out, top1 = plot_fig3_top(
        display_attr, obs_grid, ps_attr,
        island_data, latent_ch_map, ds.var_names,
        str(out_dir / f"fig3_top_n{args.top_n_vars}.png"),
        top_n_vars=args.top_n_vars,
        dpi=args.dpi,
        dark=use_dark,
    )
    print(f"Saved {out}")
    print(f"Top-1 per channel: {top1}")

    # ── Bottom panel: 4 selected latent details ──
    # Select 4 diverse channels from top1_per_ch
    # Default: Infection, Renal, Neurological, Hematologic (matching exp log)
    preferred_channels = [8, 4, 5, 7]  # Infection, Renal, Neurological, Hematologic
    ch_name_map = {c: CHANNEL_DEFINITIONS[c].name for c in range(len(CHANNEL_DEFINITIONS))}

    selected = []
    for ch in preferred_channels:
        if ch in top1:
            k = top1[ch]
            selected.append((ch, k, ch_name_map[ch][:12], ""))
    # Fill remaining slots from other channels
    for ch in sorted(top1.keys()):
        if len(selected) >= 4:
            break
        if ch not in preferred_channels:
            k = top1[ch]
            selected.append((ch, k, ch_name_map[ch][:12], ""))

    out_bottom = plot_fig3_bottom(
        ps_attr_masked, island_data, W_dec, ds.var_names, selected,
        str(out_dir / "fig3_bottom.png"),
        dpi=args.dpi,
        dark=use_dark,
    )
    print(f"Saved {out_bottom}")


if __name__ == "__main__":
    main()
