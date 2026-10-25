"""
Attribution heatmap visualization for TISAE paper figures.

Three main functions matching the Figure 3 panel structure:
  plot_attribution_comparison  — side-by-side method heatmaps (T × V)
  plot_decomposed_heatmap      — TI-SAE attribution with per-channel latent coloring
  plot_latent_detail           — per-latent zoom: heatmap + Gaussian + variable bars

Usage:
    python -m tisae.visualization.attribution_heatmaps \
        --experiment_dir experiments/<backbone>/<condition>/tisae \
        --dataset_path data/unified_48h.pkl \
        --episode_pos 20 --output_dir ./outputs/fig3
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tisae.models.ontology import CHANNEL_DEFINITIONS, create_ontology_mapping


# 8 distinct colors for channel-level latent decomposition
CHANNEL_COLORS = {
    0: np.array([0.12, 0.47, 0.71]),   # Respiratory - blue
    1: np.array([0.84, 0.15, 0.16]),   # Cardiovascular - red
    2: np.array([0.58, 0.40, 0.74]),   # Coagulation - purple
    3: np.array([0.80, 0.52, 0.25]),   # Liver - brown
    4: np.array([1.00, 0.73, 0.00]),   # Renal - amber
    5: np.array([0.55, 0.83, 0.78]),   # Neurological - teal
    6: np.array([0.90, 0.60, 0.80]),   # Metabolic - pink
    7: np.array([0.17, 0.63, 0.17]),   # Hematologic - green
    8: np.array([1.00, 0.50, 0.05]),   # Infection - orange
    9: np.array([0.65, 0.65, 0.65]),   # Other - gray
}

DEFAULT_CMAP = "inferno"


def get_channel_sorting(
    var_names: List[str],
    n_channels: int = 10,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Sort variables by clinical channel.

    Returns:
        sorted_idx: [V] array mapping sorted position -> original var index
        ch_bounds: list of starting positions for each channel in sorted order
        ch_labels: list of channel names
    """
    ontology = create_ontology_mapping(var_names, n_channels=n_channels)
    ch_assign = ontology.get_channel_assignment().numpy()
    V = len(var_names)

    ch_order, ch_bounds, ch_labels = [], [], []
    for c in range(len(CHANNEL_DEFINITIONS)):
        ch_vars = [v for v in range(V) if ch_assign[v] == c]
        if ch_vars:
            ch_bounds.append(len(ch_order))
            ch_labels.append(CHANNEL_DEFINITIONS[c].name[:12])
            ch_order.extend(ch_vars)

    return np.array(ch_order), ch_bounds, ch_labels


def _set_dark_style():
    plt.rcParams.update({
        "figure.facecolor": "black", "axes.facecolor": "black",
        "text.color": "white", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
    })


def _reset_style():
    plt.rcParams.update(matplotlib.rcParamsDefault)


def _draw_channel_labels(ax, ch_bounds, ch_labels, n_sorted):
    for bi, (b, lbl) in enumerate(zip(ch_bounds, ch_labels)):
        nxt = ch_bounds[bi + 1] if bi + 1 < len(ch_bounds) else n_sorted
        ax.text(-1.5, b + (nxt - b) / 2, lbl, fontsize=5,
                va="center", ha="right", color="#cccccc")


def _draw_channel_seps(ax, ch_bounds, color="#444444", lw=0.5):
    for b in ch_bounds[1:]:
        ax.axhline(b - 0.5, color=color, lw=lw)


def _save_optimized(fig, path, dpi=330):
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    try:
        from PIL import Image
        img = Image.open(path)
        img.save(path, "PNG", optimize=True)
    except ImportError:
        pass


def plot_attribution_comparison(
    attributions: Dict[str, np.ndarray],
    obs_grid: np.ndarray,
    var_names: List[str],
    output_path: str,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 330,
    figsize_per_method: Tuple[float, float] = (3.0, 8.0),
) -> Path:
    """Side-by-side attribution heatmaps for multiple methods.

    Args:
        attributions: {method_name: [T, V] array}
        obs_grid: [T, V] binary observation mask
        var_names: variable name list (length V)
        output_path: save path
    """
    sorted_idx, ch_bounds, ch_labels = get_channel_sorting(var_names)
    n_sorted = len(sorted_idx)
    methods = list(attributions.keys())
    n_m = len(methods)

    _set_dark_style()
    fig, axes = plt.subplots(1, n_m,
                              figsize=(figsize_per_method[0] * n_m, figsize_per_method[1]),
                              squeeze=False, facecolor="black")

    for i, method in enumerate(methods):
        ax = axes[0, i]
        a = attributions[method] * obs_grid
        a_sorted = a[:, sorted_idx]
        amax = max(np.abs(a_sorted).max(), 1e-8)
        ax.imshow((a_sorted / amax).T, aspect="auto", origin="lower",
                  cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        _draw_channel_seps(ax, ch_bounds)
        ax.set_yticks([])
        if i == 0:
            _draw_channel_labels(ax, ch_bounds, ch_labels, n_sorted)
        ax.set_title(method, fontsize=9, color="white")
        ax.set_xlabel("Time (hours)", fontsize=7)

    fig.tight_layout()
    out = Path(output_path)
    _save_optimized(fig, out, dpi)
    plt.close(fig)
    _reset_style()
    return out


def plot_decomposed_heatmap(
    attr_full: np.ndarray,
    island_data: List[dict],
    latent_ch_map: np.ndarray,
    var_names: List[str],
    output_path: str,
    top_k: int = 64,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 330,
    figsize: Tuple[float, float] = (4.0, 8.0),
    dim_factor: float = 0.08,
    channel_colors: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[Path, Dict[int, int]]:
    """TI-SAE attribution with per-channel latent coloring.

    Selects top-1 latent per channel from the top-k islands and colors
    each cell by its latent's channel color.

    Args:
        attr_full: [T, V] cell-level attribution
        island_data: list of island dicts (sorted by score descending)
        latent_ch_map: [K] array mapping latent -> channel index
        var_names: variable name list
        output_path: save path
        top_k: number of top islands to consider

    Returns:
        (output_path, top1_per_channel dict)
    """
    if channel_colors is None:
        channel_colors = CHANNEL_COLORS

    sorted_idx, ch_bounds, ch_labels = get_channel_sorting(var_names)
    n_sorted = len(sorted_idx)
    T = attr_full.shape[0]
    global_vmax = max(attr_full.max(), 1e-12)
    attr_sorted = attr_full[:, sorted_idx]
    attr_norm = np.clip(attr_sorted / global_vmax, 0, 1)

    top_islands = island_data[:top_k]

    # Find top-1 latent per channel by attribution mass
    ch_latent_groups = {}
    for isl in top_islands:
        ch = latent_ch_map[isl["k"]]
        ch_latent_groups.setdefault(ch, {}).setdefault(isl["k"], []).append(isl)

    top1_per_ch = {}
    for ch, lat_groups in ch_latent_groups.items():
        def mass(k_islands):
            mask = np.zeros((T, n_sorted), dtype=np.float32)
            for isl in k_islands:
                mask = np.maximum(mask, isl["cell_mask"][:, sorted_idx])
            return (mask * attr_sorted).sum()
        best_k = max(lat_groups.items(), key=lambda kv: mass(kv[1]))[0]
        top1_per_ch[ch] = best_k

    # Build RGB
    base = plt.cm.get_cmap(cmap)(attr_norm)[:, :, :3]
    rgb = base * dim_factor

    for ch_idx, k in top1_per_ch.items():
        if ch_idx not in channel_colors:
            continue
        color = channel_colors[ch_idx]
        for isl in top_islands:
            if isl["k"] != k:
                continue
            active = isl["cell_mask"][:, sorted_idx] > 0.5
            brightness = attr_norm * active
            for c in range(3):
                rgb[:, :, c] = np.where(
                    active, color[c] * (0.3 + 0.7 * brightness), rgb[:, :, c]
                )

    rgb = np.clip(rgb, 0, 1)
    display = np.transpose(rgb, (1, 0, 2))[::-1]

    _set_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="black")
    ax.imshow(display, aspect="auto", origin="upper", interpolation="nearest")
    for b in ch_bounds[1:]:
        ax.axhline(n_sorted - b - 0.5, color="#222222", lw=0.5)
    ax.set_yticks([])
    ax.set_title("Decomposed", fontsize=9, color="white")
    ax.set_xlabel("Time (hours)", fontsize=7)

    legend_elements = [
        Line2D([0], [0], marker="s", color="black",
               markerfacecolor=tuple(channel_colors.get(ch, np.array([0.5, 0.5, 0.5]))),
               markersize=6, label=f"k={top1_per_ch[ch]}", linewidth=0)
        for ch in sorted(top1_per_ch.keys())
        if ch in channel_colors
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=4.5,
              facecolor="black", edgecolor="#444444", labelcolor="white", framealpha=0.9)

    out = Path(output_path)
    _save_optimized(fig, out, dpi)
    plt.close(fig)
    _reset_style()
    return out, top1_per_ch


def plot_latent_detail(
    attr_full: np.ndarray,
    island_data: List[dict],
    latent_ch_map: np.ndarray,
    var_names: List[str],
    selected_latents: List[Tuple[int, int, str, str]],
    output_path: str,
    top_k: int = 64,
    colors: Optional[List[str]] = None,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 330,
) -> Path:
    """Per-latent detail: heatmap + Gaussian envelopes + variable bars.

    Args:
        attr_full: [T, V] cell-level attribution
        island_data: list of island dicts (sorted by score descending)
        latent_ch_map: [K] latent -> channel
        var_names: variable name list
        selected_latents: list of (ch_idx, k, channel_name, clinical_label)
        output_path: save path
        colors: list of hex colors per latent (default: red, teal, yellow, purple)
    """
    from tisae.models.ontology import create_ontology_mapping
    ontology = create_ontology_mapping(var_names)
    ch_assign = ontology.get_channel_assignment().numpy()

    T, V = attr_full.shape
    global_vmax = max(attr_full.max(), 1e-12)
    top_islands = island_data[:top_k]
    t_axis = np.arange(T)

    if colors is None:
        colors = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe",
                  "#ff9f43", "#6c5ce7", "#00b894", "#fd79a8"]

    # Get W for variable bars
    # (caller should pass W_np if available; we'll compute from islands if needed)
    # For now, we need W from the model — but this is a visualization function
    # that shouldn't load models. Use island cell_masks instead.

    n_c = len(selected_latents)
    max_ch_vars = max(
        len(np.where(ch_assign == ch_idx)[0])
        for ch_idx, _, _, _ in selected_latents
    )
    heatmap_h = max_ch_vars * 0.2
    gauss_h = 1.2

    _set_dark_style()
    fig = plt.figure(figsize=(4.5 * n_c, heatmap_h + gauss_h + 1), facecolor="black")
    gs = gridspec.GridSpec(2, n_c, figure=fig,
                            height_ratios=[heatmap_h, gauss_h],
                            hspace=0.35, wspace=0.3)

    for j, (ch_idx, k, ch_name_short, label) in enumerate(selected_latents):
        color = colors[j % len(colors)]
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

        # Heatmap
        ax_h = fig.add_subplot(gs[0, j])
        ax_h.imshow((lat_attr / global_vmax).T, aspect="auto", origin="lower",
                     cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax_h.set_yticks(range(n_v))
        ax_h.set_yticklabels(v_names_ch, fontsize=5, color="#cccccc")
        title = f"{ch_name_short} — k={k}"
        if label:
            title += f'\n"{label}"'
        ax_h.set_title(title, fontsize=7, color=color, fontweight="bold")
        ax_h.set_xlabel("Time (hours)", fontsize=6)

        # Gaussians
        ax_g = fig.add_subplot(gs[1, j])
        for isl in k_islands:
            g = np.exp(-0.5 * ((t_axis - isl["mu"]) / max(isl["sigma"], 1e-6)) ** 2)
            ax_g.fill_between(t_axis, 0, g, alpha=0.2, color=color)
            ax_g.plot(t_axis, g, color=color, lw=1.5)
            ax_g.axvline(isl["mu"], color=color, ls=":", lw=0.8, alpha=0.5)
            ax_g.text(isl["mu"], 1.05, f'{isl["mu"]:.0f}h',
                      fontsize=5, ha="center", color=color)
        ax_g.set_xlim(0, T - 1)
        ax_g.set_ylim(0, 1.2)
        if j == 0:
            ax_g.set_ylabel("g(t)", fontsize=7)
        ax_g.set_xlabel("Time (hours)", fontsize=6)

    fig.tight_layout()
    out = Path(output_path)
    _save_optimized(fig, out, dpi)
    plt.close(fig)
    _reset_style()
    return out


def plot_latent_detail_with_bars(
    attr_full: np.ndarray,
    island_data: List[dict],
    W_dec: np.ndarray,
    var_names: List[str],
    selected_latents: List[Tuple[int, int, str, str]],
    output_path: str,
    top_k: int = 64,
    colors: Optional[List[str]] = None,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 330,
) -> Path:
    """Per-latent detail with variable composition bars (3 rows).

    Same as plot_latent_detail but adds a third row with decoder weight bars.

    Args:
        W_dec: [K, V] absolute decoder weight matrix
        (other args same as plot_latent_detail)
    """
    from tisae.models.ontology import create_ontology_mapping
    ontology = create_ontology_mapping(var_names)
    ch_assign = ontology.get_channel_assignment().numpy()

    T, V = attr_full.shape
    global_vmax = max(attr_full.max(), 1e-12)
    top_islands = island_data[:top_k]
    t_axis = np.arange(T)

    if colors is None:
        colors = ["#ff6b6b", "#4ecdc4", "#ffe66d", "#a29bfe",
                  "#ff9f43", "#6c5ce7", "#00b894", "#fd79a8"]

    n_c = len(selected_latents)
    max_ch_vars = max(
        len(np.where(ch_assign == ch_idx)[0])
        for ch_idx, _, _, _ in selected_latents
    )
    heatmap_h = max_ch_vars * 0.2
    gauss_h = 1.2
    bar_h = 1.5

    _set_dark_style()
    fig = plt.figure(figsize=(4.5 * n_c, heatmap_h + gauss_h + bar_h + 1),
                     facecolor="black")
    gs = gridspec.GridSpec(3, n_c, figure=fig,
                            height_ratios=[heatmap_h, gauss_h, bar_h],
                            hspace=0.35, wspace=0.3)

    for j, (ch_idx, k, ch_name_short, label) in enumerate(selected_latents):
        color = colors[j % len(colors)]
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

        # Heatmap
        ax_h = fig.add_subplot(gs[0, j])
        ax_h.imshow((lat_attr / global_vmax).T, aspect="auto", origin="lower",
                     cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax_h.set_yticks(range(n_v))
        ax_h.set_yticklabels(v_names_ch, fontsize=5, color="#cccccc")
        title = f"{ch_name_short} — k={k}"
        if label:
            title += f'\n"{label}"'
        ax_h.set_title(title, fontsize=7, color=color, fontweight="bold")
        ax_h.set_xlabel("Time (hours)", fontsize=6)

        # Gaussians
        ax_g = fig.add_subplot(gs[1, j])
        for isl in k_islands:
            g = np.exp(-0.5 * ((t_axis - isl["mu"]) / max(isl["sigma"], 1e-6)) ** 2)
            ax_g.fill_between(t_axis, 0, g, alpha=0.2, color=color)
            ax_g.plot(t_axis, g, color=color, lw=1.5)
            ax_g.axvline(isl["mu"], color=color, ls=":", lw=0.8, alpha=0.5)
            ax_g.text(isl["mu"], 1.05, f'{isl["mu"]:.0f}h',
                      fontsize=5, ha="center", color=color)
        ax_g.set_xlim(0, T - 1)
        ax_g.set_ylim(0, 1.2)
        if j == 0:
            ax_g.set_ylabel("g(t)", fontsize=7)
        ax_g.set_xlabel("Time (hours)", fontsize=6)

        # Variable bars
        ax_bar = fig.add_subplot(gs[2, j])
        if top_var_list:
            names = [n for n, w in top_var_list]
            weights = [w for n, w in top_var_list]
            bars = ax_bar.barh(range(len(names)), weights, color=color,
                               alpha=0.8, height=0.6)
            ax_bar.set_yticks(range(len(names)))
            ax_bar.set_yticklabels(names, fontsize=5, color="#cccccc")
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("Decoder weight", fontsize=5)
            for bar, w in zip(bars, weights):
                ax_bar.text(bar.get_width() + 0.005,
                            bar.get_y() + bar.get_height() / 2,
                            f"{w:.3f}", va="center", fontsize=4, color="#cccccc")

    fig.tight_layout()
    out = Path(output_path)
    _save_optimized(fig, out, dpi)
    plt.close(fig)
    _reset_style()
    return out
