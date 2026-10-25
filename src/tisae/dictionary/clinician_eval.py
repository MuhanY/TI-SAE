"""
Generate clinician evaluation materials for dictionary label validation.

Produces a self-contained HTML file with:
  - Latent info (variable composition, exemplar traces, island windows)
  - LLM-generated label + description
  - Overlay mode (default): all episodes on shared axes
  - Separate mode: one panel per episode, shared y-axis limits
  - Toggle buttons to show/hide individual episodes
  - 1-5 Likert scale radio buttons for accuracy rating
  - Export button to download scores as CSV

Usage:
    python -m tisae.dictionary.clinician_eval \
        --dictionary_path experiments/.../latent_dictionary/latent_dictionary.json \
        --labels_path experiments/.../latent_dictionary/llm_labels.json \
        --output_path clinician_eval.html \
        --n_samples 30
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
ISLAND_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


def _render_or_bar(mort_or: float) -> str:
    """Render mortality OR as a colored horizontal bar in HTML/CSS."""
    # Clamp for display: log scale, OR 0.1-10 maps to 0-100%
    import math
    log_or = math.log10(max(mort_or, 0.01))  # -2 to ~1
    # Map log_or: -1 (OR=0.1) -> 0%, 0 (OR=1) -> 50%, 1 (OR=10) -> 100%
    pct = max(0, min(100, (log_or + 1) * 50))

    if mort_or < 0.5:
        color = "#27ae60"  # green — protective
        tier = "Protective"
    elif mort_or < 1.5:
        color = "#95a5a6"  # grey — neutral
        tier = "Neutral"
    elif mort_or < 3.0:
        color = "#e67e22"  # orange — moderate risk
        tier = "Moderate risk"
    else:
        color = "#e74c3c"  # red — high risk
        tier = "High risk"

    return (
        f'<div class="or-bar">'
        f'<div class="or-bar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
        f'<div class="or-bar-center"></div>'
        f'</div>'
        f'<span class="or-tier" style="color:{color}">{tier}</span>'
    )
COLOR_HEX = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
             "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _render_var_comp_bar(var_comp: List[dict]) -> str:
    """Render variable composition as a horizontal bar chart, base64 PNG."""
    if not var_comp:
        return ""
    names = [v["var_name"] for v in var_comp]
    weights = [v["weight"] for v in var_comp]
    colors = [COLOR_HEX[i % len(COLOR_HEX)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(5, max(1.0, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), weights, color=colors, height=0.6, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Decoder Weight", fontsize=9)
    for bar, w in zip(bars, weights):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=8, color="#333")
    ax.set_xlim(0, max(weights) * 1.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


def _compute_ylims(exemplars: List[dict], var_names: List[str]):
    """Compute shared y-axis limits per variable across all exemplars."""
    ylims = {}
    for vn in var_names:
        all_vals = []
        for ex in exemplars:
            trace = ex.get("variable_traces", {}).get(vn, [])
            all_vals.extend([p["val"] for p in trace])
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = max((hi - lo) * 0.1, 0.5)
            ylims[vn] = (lo - margin, hi + margin)
        else:
            ylims[vn] = (0, 1)
    return ylims


def _render_overlay_image(
    exemplars: List[dict],
    var_names: List[str],
    T: int = 48,
) -> str:
    """All episodes overlaid on shared axes, no value annotations."""
    n_vars = len(var_names)
    if n_vars == 0:
        return ""

    ylims = _compute_ylims(exemplars, var_names)
    fig_height = max(2, n_vars * 1.2)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, fig_height), squeeze=False)

    for vi, vn in enumerate(var_names):
        ax = axes[vi, 0]
        for ei, ex in enumerate(exemplars):
            trace = ex.get("variable_traces", {}).get(vn, [])
            if not trace:
                continue
            ts = [p["t"] for p in trace]
            vs = [p["val"] for p in trace]
            is_mort = ex.get("mortality", 0) > 0.5
            marker = "x" if is_mort else "o"
            alpha = 0.7
            color = COLOR_HEX[ei % len(COLOR_HEX)]
            ax.plot(ts, vs, marker=marker, markersize=4, linewidth=1.2,
                    alpha=alpha, color=color)

            # Shade islands (only first 3 episodes to avoid clutter)
            if ei < 3:
                for isl in ex.get("island_windows", []):
                    t_start = max(0, isl["mu"] - 2 * isl["sigma"])
                    t_end = min(T, isl["mu"] + 2 * isl["sigma"])
                    ax.axvspan(t_start, t_end, alpha=0.05, color=color)

        ax.set_xlim(0, T)
        ax.set_ylim(ylims[vn])
        ax.set_ylabel(vn, fontsize=8, rotation=0, labelpad=55, ha="right")
        ax.tick_params(labelsize=7)
        if vi < n_vars - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (hours)", fontsize=9)

    fig.tight_layout(h_pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


def _jitter_annotations(points, fontsize=6, min_gap=8):
    """Compute non-overlapping y-offsets for value annotations.

    points: list of (t, v) tuples
    Returns list of (dx, dy) offset tuples in points.
    """
    if not points:
        return []
    offsets = [(3, 5)] * len(points)
    # Sort by time, alternate above/below for close points
    indexed = sorted(enumerate(points), key=lambda iv: (iv[1][0], iv[1][1]))
    for i in range(1, len(indexed)):
        prev_idx, (pt, pv) = indexed[i - 1]
        curr_idx, (ct, cv) = indexed[i]
        if abs(ct - pt) <= 2 and abs(cv - pv) < (max(abs(pv), 1) * 0.15):
            offsets[curr_idx] = (3, -12)
    return offsets


def _render_episode_image(
    exemplar: dict,
    var_names: List[str],
    ylims: dict,
    color_idx: int,
    T: int = 48,
) -> str:
    """One episode's traces, with shared y-axis limits and jittered annotations."""
    traces = exemplar.get("variable_traces", {})
    islands = exemplar.get("island_windows", [])
    n_vars = len(var_names)
    if n_vars == 0:
        return ""

    fig_height = max(1.5, n_vars * 1.0)
    fig, axes = plt.subplots(n_vars, 1, figsize=(7, fig_height), squeeze=False)
    color = COLOR_HEX[color_idx % len(COLOR_HEX)]

    for vi, vn in enumerate(var_names):
        ax = axes[vi, 0]
        trace = traces.get(vn, [])

        for isl in islands:
            t_start = max(0, isl["mu"] - 2 * isl["sigma"])
            t_end = min(T, isl["mu"] + 2 * isl["sigma"])
            scale = isl.get("scale", 0)
            ax.axvspan(t_start, t_end, alpha=0.12,
                       color=ISLAND_COLORS[scale % len(ISLAND_COLORS)])

        if trace:
            ts = [p["t"] for p in trace]
            vs = [p["val"] for p in trace]
            is_mort = exemplar.get("mortality", 0) > 0.5
            marker = "x" if is_mort else "o"
            ax.plot(ts, vs, marker=marker, markersize=5, linewidth=1.5,
                    color=color, alpha=0.9)
            offsets = _jitter_annotations(list(zip(ts, vs)))
            for (t, v), (dx, dy) in zip(zip(ts, vs), offsets):
                ax.annotate(f"{v:.1f}", (t, v), fontsize=6, color="#555",
                            textcoords="offset points", xytext=(dx, dy))
        else:
            ax.text(0.5, 0.5, "NOT MEASURED", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        ax.set_xlim(0, T)
        ax.set_ylim(ylims.get(vn, (0, 1)))
        ax.set_ylabel(vn, fontsize=8, rotation=0, labelpad=55, ha="right")
        ax.tick_params(labelsize=7)
        if vi < n_vars - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (hours)", fontsize=9)

    fig.tight_layout(h_pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


def _select_stratified(
    dictionary: List[dict],
    n_samples: int = 30,
    seed: int = 42,
) -> List[dict]:
    rng = np.random.RandomState(seed)
    by_or = sorted(dictionary, key=lambda e: e.get("mortality_odds_ratio", 1.0),
                   reverse=True)
    n_high = n_samples // 3
    n_low = n_samples // 3
    n_mid = n_samples - n_high - n_low

    high = by_or[:n_high]
    low = by_or[-n_low:]
    mid_pool = by_or[n_high:-n_low] if len(by_or) > n_high + n_low else []
    if len(mid_pool) > n_mid:
        mid_idx = rng.choice(len(mid_pool), n_mid, replace=False)
        mid = [mid_pool[i] for i in mid_idx]
    else:
        mid = mid_pool

    selected = high + mid + low
    rng.shuffle(selected)
    return selected


def generate_eval_html(
    dictionary: List[dict],
    labels: Dict[str, dict],
    output_path: str,
    n_samples: int = 30,
    seed: int = 42,
    T: int = 48,
    evaluator_name: str = "",
    max_exemplars: int = 10,
) -> Path:
    """Generate self-contained HTML evaluation form."""
    for entry in dictionary:
        key = f"{entry['latent_k']}_{entry['channel_c']}"
        if key in labels:
            entry["clinical_label"] = labels[key].get("clinical_label", "")
            entry["clinical_description"] = labels[key].get("clinical_description", "")

    labeled = [e for e in dictionary if e.get("clinical_label")
               and e["clinical_label"] != "PARSE_ERROR"]
    selected = _select_stratified(labeled, n_samples, seed)
    print(f"Selected {len(selected)} latents for evaluation "
          f"(from {len(labeled)} labeled, {len(dictionary)} total)")

    items_html = []
    for idx, entry in enumerate(selected):
        k = entry["latent_k"]
        c = entry["channel_c"]
        ch_name = entry.get("inferred_channel_name",
                            entry.get("channel_name", "all"))
        label = entry.get("clinical_label", "")
        desc = entry.get("clinical_description", "")
        freq = entry.get("activation_frequency", 0)
        mort_or = entry.get("mortality_odds_ratio", 1.0)
        var_comp = entry.get("variable_composition", [])
        vars_str = ", ".join(f"{v['var_name']} ({v['weight']:.3f})"
                             for v in var_comp[:5])
        var_names = [v["var_name"] for v in var_comp[:5]]
        exemplars = entry.get("exemplar_episodes_sae", [])[:max_exemplars]

        # Shared y-limits
        ylims = _compute_ylims(exemplars, var_names)

        # Overlay image (default view)
        overlay_img = _render_overlay_image(exemplars, var_names, T=T)

        # Per-episode images (separate view)
        toggle_buttons = []
        episode_panels = []
        for ei, ex in enumerate(exemplars):
            ep_id = ex.get("episode_idx", ei)
            is_mort = ex.get("mortality", 0) > 0.5
            strength = ex.get("activation_strength", 0)
            color = COLOR_HEX[ei % len(COLOR_HEX)]

            island_strs = []
            for isl in ex.get("island_windows", []):
                t_start = max(0, isl["mu"] - 2 * isl["sigma"])
                t_end = min(T, isl["mu"] + 2 * isl["sigma"])
                island_strs.append(f"h{t_start:.0f}-{t_end:.0f}")
            island_text = ", ".join(island_strs) if island_strs else "N/A"

            ep_img = _render_episode_image(ex, var_names, ylims, ei, T=T)

            toggle_buttons.append(
                f'<button class="ep-toggle" style="border-color:{color}" '
                f'onclick="toggleEp(this, \'sep-{idx}-{ei}\')">'
                f'Ep {ep_id} ({"D" if is_mort else "S"}, {strength:.3f})</button>'
            )

            episode_panels.append(f"""
            <div class="episode-panel" id="sep-{idx}-{ei}" style="display:none">
                <div class="ep-header">
                    <span style="color:{color};font-weight:bold">Episode {ep_id}</span>
                    <span>Mortality: <b>{"YES" if is_mort else "NO"}</b> |
                          Strength: {strength:.3f} | Island: {island_text}</span>
                </div>
                {"<img src='" + ep_img + "' class='trace-img'/>" if ep_img else ""}
            </div>
            """)

        # Variable composition bar graph
        var_bar_img = _render_var_comp_bar(var_comp)

        # Legend items as toggle buttons (in the overlay header)
        legend_toggles = []
        for ei, ex in enumerate(exemplars):
            ep_id = ex.get("episode_idx", ei)
            is_mort = ex.get("mortality", 0) > 0.5
            strength = ex.get("activation_strength", 0)
            color = COLOR_HEX[ei % len(COLOR_HEX)]
            legend_toggles.append(
                f'<button class="ep-legend-btn" style="color:{color};border-color:{color}" '
                f'onclick="toggleEp(this, \'sep-{idx}-{ei}\', {idx})">'
                f'&#9679; Ep {ep_id} ({"D" if is_mort else "S"}, {strength:.3f})</button>'
            )
        legend_bar = '<div class="legend-bar">' + " ".join(legend_toggles) + '</div>'

        item_html = f"""
        <div class="latent-item" id="item-{idx}">
            <h3>Latent {idx + 1} of {len(selected)} &mdash; k={k}, Channel: {ch_name}</h3>

            <div class="info-grid">
                <div class="info-box">
                    <h4>Variable Composition</h4>
                    {"<img src='" + var_bar_img + "' style='width:100%;max-width:400px'/>" if var_bar_img else "<p>" + vars_str + "</p>"}
                </div>
                <div class="info-box">
                    <h4>Statistics</h4>
                    <p>Active in <b>{freq*100:.1f}%</b> of episodes</p>
                    <div class="or-bar-container">
                        <span class="or-label">Mortality OR: <b>{mort_or:.2f}</b></span>
                        {_render_or_bar(mort_or)}
                    </div>
                    <p class="mort-rates">Mortality when active: <b>{entry.get("mortality_rate_when_active", 0)*100:.1f}%</b> (baseline: {entry.get("mortality_rate_baseline", 0)*100:.1f}%)</p>
                </div>
            </div>

            <div class="traces-section">
                <h4>Exemplar Episodes ({len(exemplars)}) — click episode to expand detail</h4>
                {legend_bar}
                {"<img src='" + overlay_img + "' class='trace-img'/>" if overlay_img else ""}
                {"".join(episode_panels)}
            </div>

            <div class="llm-box">
                <h4>LLM-Generated Label</h4>
                <p class="llm-label">{label}</p>
                <p class="llm-desc">{desc}</p>
            </div>

            <div class="rating-box">
                <h4>How accurately does this label describe the clinical pattern?</h4>
                <div class="radio-group">
                    <label><input type="radio" name="score-{idx}" value="1" data-k="{k}" data-c="{c}"/> 1 — Completely wrong</label>
                    <label><input type="radio" name="score-{idx}" value="2" data-k="{k}" data-c="{c}"/> 2 — Partially related but misleading</label>
                    <label><input type="radio" name="score-{idx}" value="3" data-k="{k}" data-c="{c}"/> 3 — Vaguely correct but imprecise</label>
                    <label><input type="radio" name="score-{idx}" value="4" data-k="{k}" data-c="{c}"/> 4 — Mostly accurate, minor issues</label>
                    <label><input type="radio" name="score-{idx}" value="5" data-k="{k}" data-c="{c}"/> 5 — Accurate and clinically meaningful</label>
                </div>
                <div class="comment-box">
                    <label>Optional comment:</label>
                    <textarea name="comment-{idx}" rows="2" placeholder="Any notes..."></textarea>
                </div>
            </div>
        </div>
        <hr/>
        """
        items_html.append(item_html)
        if (idx + 1) % 5 == 0:
            print(f"  Rendered {idx + 1}/{len(selected)} latents...")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>TI-SAE Dictionary Label Evaluation</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 950px;
           margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
    h3 {{ color: #2c3e50; margin-top: 30px; }}
    h4 {{ color: #555; margin: 8px 0 4px 0; font-size: 14px; }}
    .instructions {{ background: #eaf2f8; border-left: 4px solid #3498db;
                     padding: 15px; margin: 20px 0; border-radius: 4px; }}
    .latent-item {{ background: white; padding: 20px; margin: 15px 0;
                    border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .info-box {{ background: #f8f9fa; padding: 10px; border-radius: 4px; }}
    .info-box p {{ margin: 4px 0; font-size: 13px; }}
    .or-bar-container {{ margin: 6px 0; }}
    .or-label {{ font-size: 13px; }}
    .or-bar {{ width: 100%; height: 12px; background: #eee; border-radius: 6px;
               position: relative; margin: 4px 0; }}
    .or-bar-fill {{ height: 100%; border-radius: 6px; transition: width 0.3s; }}
    .or-bar-center {{ position: absolute; left: 50%; top: 0; bottom: 0;
                      width: 2px; background: #333; }}
    .or-tier {{ font-size: 12px; font-weight: bold; }}
    .mort-rates {{ font-size: 12px; color: #666; }}
    .traces-section {{ margin: 10px 0; }}
    .legend-bar {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }}
    .ep-legend-btn {{ padding: 3px 10px; font-size: 11px; border: 2px solid #aaa;
                      border-radius: 12px; cursor: pointer; background: #f8f8f8;
                      font-weight: bold; transition: background 0.15s; }}
    .ep-legend-btn:hover {{ background: #e8e8e8; }}
    .ep-legend-btn.active {{ background: #e8f4fd; }}
    .episode-panel {{ margin: 6px 0; border: 1px solid #eee; border-radius: 4px;
                      padding: 8px; background: #fafafa; }}
    .ep-header {{ font-size: 12px; color: #555; margin-bottom: 4px; }}
    .trace-img {{ width: 100%; border-radius: 4px; }}
    .llm-box {{ background: #fff3cd; padding: 12px; border-radius: 4px;
                border-left: 4px solid #ffc107; margin: 10px 0; }}
    .llm-label {{ font-size: 18px; font-weight: bold; color: #856404; margin: 4px 0; }}
    .llm-desc {{ font-size: 14px; color: #666; margin: 4px 0; }}
    .rating-box {{ background: #f0f7ff; padding: 12px; border-radius: 4px;
                   border-left: 4px solid #3498db; }}
    .radio-group {{ display: flex; flex-direction: column; gap: 6px; margin: 8px 0; }}
    .radio-group label {{ font-size: 14px; cursor: pointer; padding: 4px 0; }}
    .radio-group input {{ margin-right: 8px; }}
    .comment-box {{ margin-top: 8px; }}
    .comment-box textarea {{ width: 100%; font-size: 13px; padding: 6px;
                             border: 1px solid #ddd; border-radius: 4px; }}
    hr {{ border: none; border-top: 1px solid #eee; margin: 30px 0; }}
    .export-section {{ background: #d4edda; padding: 15px; border-radius: 8px;
                       margin: 20px 0; text-align: center; }}
    .export-btn {{ background: #28a745; color: white; border: none; padding: 12px 30px;
                   font-size: 16px; border-radius: 4px; cursor: pointer; }}
    .export-btn:hover {{ background: #218838; }}
    .progress {{ position: fixed; top: 0; left: 0; right: 0; background: #2c3e50;
                 color: white; padding: 8px 20px; font-size: 14px; z-index: 100;
                 display: flex; justify-content: space-between; }}
    #evaluator-name {{ padding: 6px 10px; font-size: 14px; width: 300px;
                       border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>

<div class="progress">
    <span>TI-SAE Dictionary Label Evaluation</span>
    <span id="progress-text">0 / {len(selected)} scored</span>
</div>

<div style="margin-top: 50px;">
<h1>TI-SAE Dictionary Label Evaluation</h1>

<div class="instructions">
    <h4>Instructions</h4>
    <p>For each latent, you will see:</p>
    <ul>
        <li><strong>Variable composition</strong> — which clinical variables this latent tracks</li>
        <li><strong>Exemplar episodes</strong> — raw clinical values from top-activating episodes.
            <b>Overlay</b> (default) shows all episodes on shared axes for comparison.
            <b>Separate</b> shows one episode at a time with value annotations.
            Shaded regions = learned temporal windows. x = died, o = survived.</li>
        <li><strong>LLM label</strong> — an AI-generated clinical interpretation</li>
    </ul>
    <p>Rate how accurately the label describes the pattern in the data.</p>
    <p><strong>Evaluator:</strong> <input type="text" id="evaluator-name" value="{evaluator_name}" placeholder="Your name"/></p>
</div>

{"".join(items_html)}

<div class="export-section">
    <h3>Export Scores</h3>
    <button class="export-btn" onclick="exportCSV()">Download CSV</button>
</div>
</div>

<script>
function toggleEp(btn, epId, latentIdx) {{
    const section = document.getElementById('item-' + latentIdx);
    const allBtns = section.querySelectorAll('.ep-legend-btn');
    const allPanels = section.querySelectorAll('.episode-panel');
    const panel = document.getElementById(epId);
    const wasOpen = panel.style.display === 'block';
    // Close all
    allBtns.forEach(b => b.classList.remove('active'));
    allPanels.forEach(p => p.style.display = 'none');
    // Toggle: reopen if wasn't open
    if (!wasOpen) {{
        panel.style.display = 'block';
        btn.classList.add('active');
    }}
}}

function updateProgress() {{
    const total = {len(selected)};
    let scored = 0;
    for (let i = 0; i < total; i++) {{
        const radios = document.getElementsByName('score-' + i);
        for (const r of radios) {{
            if (r.checked) {{ scored++; break; }}
        }}
    }}
    document.getElementById('progress-text').textContent = scored + ' / ' + total + ' scored';
}}
document.addEventListener('change', updateProgress);

function exportCSV() {{
    const name = document.getElementById('evaluator-name').value || 'anonymous';
    let csv = 'evaluator,latent_k,channel_c,score,comment\\n';
    const total = {len(selected)};
    for (let i = 0; i < total; i++) {{
        const radios = document.getElementsByName('score-' + i);
        let score = '', k = '', c = '';
        for (const r of radios) {{
            k = r.dataset.k; c = r.dataset.c;
            if (r.checked) {{ score = r.value; break; }}
        }}
        const comment = (document.getElementsByName('comment-' + i)[0]?.value || '').replace(/"/g, '""');
        csv += '"' + name + '",' + k + ',' + c + ',' + score + ',"' + comment + '"\\n';
    }}
    const blob = new Blob([csv], {{type: 'text/csv'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tisae_label_eval_' + name.replace(/\\s/g, '_') + '.csv';
    a.click();
}}
</script>
</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Saved evaluation form: {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clinician evaluation HTML for TI-SAE dictionary labels")
    parser.add_argument("--dictionary_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="clinician_eval.html")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--max_exemplars", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluator_name", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.dictionary_path) as f:
        dictionary = json.load(f)
    with open(args.labels_path) as f:
        labels = json.load(f)
    generate_eval_html(
        dictionary, labels, args.output_path,
        n_samples=args.n_samples, seed=args.seed,
        max_exemplars=args.max_exemplars,
        evaluator_name=args.evaluator_name,
    )
