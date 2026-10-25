"""
Generate clinician evaluation HTML for pairwise verbalized explanation comparison.

Two-phase protocol matching the LLM judge:
  Phase 1 (blinded): Clinician sees patient data + two anonymized explanations (A/B).
      Scores concept reasoning, actionability, faithfulness. Locks answers.
  Phase 2 (unblinded): Source analyses revealed. Scores faithfulness.
      Cannot change Phase 1 answers.

Usage:
    python -m tisae.evaluation.verbalization.clinician_eval \
        --episodes_path .../verb_eval/episodes_data.json \
        --verb_results_path .../verb_eval/v9_results/verb_results.json \
        --output_path clinician_pairwise_eval.html \
        --baseline attention \
        --n_episodes 50
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import numpy as np
from pathlib import Path
from typing import List

from .build_prompts import format_attribution_cells, format_concepts




def _raw_grid_html(raw_values: List[dict], idx: int, T: int = 48) -> str:
    """Render raw values as V x T grid. Rows = variables, columns = hours."""
    # Group by variable
    by_var = {}
    for r in raw_values:
        vn = r["variable"]
        if vn not in by_var:
            by_var[vn] = {}
        t = r["t"]
        by_var[vn][t] = r["value_raw"]

    # Sort variables by observation count (densest first)
    var_order = sorted(by_var.keys(), key=lambda v: -len(by_var[v]))

    # Build time columns (only hours with any observation)
    all_hours = sorted(set(r["t"] for r in raw_values))
    if not all_hours:
        return "<p>No observations</p>"

    lines = ['<div class="grid-wrapper"><table class="vt-grid">']
    # Header row: hours
    lines.append('<tr><th class="var-name-col">Variable</th>')
    for h in all_hours:
        lines.append(f'<th class="hour-col">h{h}</th>')
    lines.append('</tr>')

    for vn in var_order:
        vn_escaped = html_mod.escape(vn)
        vn_js = vn.replace("'", "\\'").replace('"', '\\"')
        lines.append(
            f'<tr class="var-row" onclick="showVarChart({idx},\'{vn_js}\')">'
            f'<td class="var-name-cell">{vn_escaped}</td>'
        )
        obs = by_var[vn]
        for h in all_hours:
            if h in obs:
                val = obs[h]
                lines.append(f'<td class="obs-cell">{val}</td>')
            else:
                lines.append('<td class="empty-cell"></td>')
        lines.append('</tr>')

    lines.append('</table></div>')
    return "\n".join(lines)


def _source_analysis_card_html(episode: dict, method: str, baseline: str) -> str:
    """Build user-friendly HTML card for model analysis."""
    if method == "tisae":
        concepts = episode.get("tisae_top5_latents", [])
        parts = ['<div class="analysis-card concept-card">',
                 '<div class="analysis-title">Concept-Based Analysis</div>']
        for con in concepts:
            vars_str = ", ".join(f"{v[0]} ({v[1]:.2f})" for v in con["top_variables"][:3])
            label = con.get("clinical_label") or ""
            channel = con.get("channel") or ""
            weight = con.get("total_weight", 0)
            # Weight bar
            w_pct = min(weight / 5.0 * 100, 100)
            parts.append(f'''
            <div class="concept-item">
                <div class="concept-header">
                    <span class="concept-rank">#{con["rank"]+1}</span>
                    <span class="concept-channel">{html_mod.escape(channel)}</span>
                    <span class="concept-label">{html_mod.escape(label)}</span>
                </div>
                <div class="concept-weight-bar"><div class="concept-weight-fill" style="width:{w_pct:.0f}%"></div><span class="concept-weight-text">{weight:.1f}</span></div>
                <div class="concept-vars">{html_mod.escape(vars_str)}</div>
            </div>''')
            # Islands
            islands = sorted(
                [isl for isl in con.get("islands", []) if isl.get("weight", 0) >= 0.01],
                key=lambda x: x.get("t_start", 0))
            if islands:
                isl_parts = []
                for isl in islands[:3]:
                    isl_parts.append(f'h{isl["t_start"]}-{isl["t_end"]} (w={isl["weight"]:.2f})')
                parts.append(f'<div class="concept-islands">{", ".join(isl_parts)}</div>')
        parts.append('</div>')
        return "\n".join(parts)
    else:
        cells = episode.get("attribution_cells", {}).get(baseline, [])
        if not cells:
            return '<div class="analysis-card attr-card"><div class="analysis-title">Attribution Analysis</div><p>No cells</p></div>'

        # Build V x T grid colored by importance
        by_var = {}
        max_imp = max((c.get("importance", 0) for c in cells), default=0.001)
        all_hours = set()
        for c in cells:
            vn = c.get("variable", "?")
            t = c.get("t", 0)
            all_hours.add(t)
            if vn not in by_var:
                by_var[vn] = {}
            by_var[vn][t] = {
                "imp": c.get("importance", 0),
                "val": c.get("value_raw"),
            }

        # Sort vars by max importance
        var_order = sorted(by_var.keys(),
                           key=lambda v: -max(d["imp"] for d in by_var[v].values()))
        all_hours = sorted(all_hours)

        parts = ['<div class="analysis-card attr-card">',
                 '<div class="analysis-title">Attribution Analysis (importance-colored)</div>',
                 '<div class="attr-grid-wrap"><table class="attr-grid">']
        # Header
        parts.append('<tr><th class="var-name-col">Variable</th>')
        for h in all_hours:
            parts.append(f'<th class="hour-col-attr">h{h}</th>')
        parts.append('</tr>')

        for vn in var_order:
            vn_escaped = html_mod.escape(vn)
            parts.append(f'<tr><td class="var-name-cell-attr">{vn_escaped}</td>')
            for h in all_hours:
                if h in by_var[vn]:
                    d = by_var[vn][h]
                    imp = d["imp"]
                    val = d["val"]
                    # Blue intensity by importance
                    alpha = min(imp / max_imp, 1.0)
                    r_c = int(230 - 180 * alpha)
                    g_c = int(242 - 130 * alpha)
                    b_c = int(255 - 30 * alpha)
                    bg = f"rgb({r_c},{g_c},{b_c})"
                    val_str = f"{val}" if val is not None else ""
                    parts.append(
                        f'<td class="attr-obs-cell" style="background:{bg}" '
                        f'title="imp={imp:.4f}">{val_str}</td>')
                else:
                    parts.append('<td class="empty-cell"></td>')
            parts.append('</tr>')

        parts.append('</table></div>')
        parts.append('</div>')
        return "\n".join(parts)


def _winner_to_ab(winner: str, tisae_is_a: bool, baseline: str) -> str:
    """Convert method-name winner to A/B/Tie in the HTML frame."""
    if winner == "Tie" or winner == "tie":
        return "Tie"
    if winner == "tisae":
        return "A" if tisae_is_a else "B"
    if winner == baseline:
        return "B" if tisae_is_a else "A"
    return "Tie"


def _winner_to_ab_cross(
    winner: str, src_tisae_is_a: bool, html_tisae_is_a: bool, baseline: str,
) -> str:
    """Translate winner from a different A/B frame to the HTML frame."""
    if winner == "Tie" or winner == "tie":
        return "Tie"
    is_tisae_winner = winner == "tisae"
    if not is_tisae_winner and winner != baseline:
        return "Tie"
    if is_tisae_winner:
        return "A" if html_tisae_is_a else "B"
    else:
        return "B" if html_tisae_is_a else "A"


def _source_analysis_html(episode: dict, method: str, baseline: str) -> str:
    """Build HTML for source analysis (used in faithfulness phase)."""
    if method == "tisae":
        concepts = episode["tisae_top5_latents"]
        lines = ['<div class="source-analysis">',
                 '<strong>Model Analysis (Concept-based):</strong><pre>']
        lines.append(html_mod.escape(format_concepts(concepts)))
        lines.append('</pre></div>')
        return "\n".join(lines)
    else:
        cells = episode["attribution_cells"].get(baseline, [])
        lines = ['<div class="source-analysis">',
                 '<strong>Model Analysis (Attribution cells):</strong><pre>']
        lines.append(html_mod.escape(format_attribution_cells(cells)))
        lines.append('</pre></div>')
        return "\n".join(lines)


def generate_pairwise_eval_html(
    episodes: List[dict],
    verb_results: List[dict],
    output_path: str,
    baseline: str = "attention",
    n_episodes: int = 50,
    seed: int = 42,
    evaluator_name: str = "",
    pairwise_results: List[dict] | None = None,
    faithfulness_results: List[dict] | None = None,
) -> Path:
    rng = np.random.RandomState(seed)

    ep_map = {ep["ep_idx"]: ep for ep in episodes}
    verb_map = {}
    for r in verb_results:
        verb_map[(r["ep_idx"], r["method"])] = r["explanation"]

    # Read A/B assignments and LLM answers from pairwise results
    ab_map = {}
    llm_answers = {}  # ep_idx -> {dim: "A"/"B"/"Tie", dim_rationale: ...}
    if pairwise_results:
        for r in pairwise_results:
            if r.get("attr_method") == baseline:
                ab_map[r["ep_idx"]] = r["tisae_is_a"]
                tisae_is_a = r["tisae_is_a"]
                answers = {}
                for dim in ["concept", "actionability", "faithfulness"]:
                    winner = r.get(dim, "")
                    rationale = r.get(f"{dim}_rationale", "")
                    answers[dim] = _winner_to_ab(winner, tisae_is_a, baseline)
                    answers[f"{dim}_rationale"] = rationale
                llm_answers[r["ep_idx"]] = answers

    # Read faithfulness from separate v15 results if provided
    if faithfulness_results:
        for r in faithfulness_results:
            if r.get("attr_method") == baseline:
                ep = r["ep_idx"]
                v15_tisae_is_a = r["tisae_is_a"]
                winner = r.get("faithfulness", "")
                rationale = r.get("faithfulness_rationale", "")
                # Translate to the v9 A/B frame
                html_tisae_is_a = ab_map.get(ep)
                if html_tisae_is_a is not None:
                    ab_val = _winner_to_ab_cross(
                        winner, v15_tisae_is_a, html_tisae_is_a, baseline)
                    if ep not in llm_answers:
                        llm_answers[ep] = {}
                    llm_answers[ep]["faithfulness"] = ab_val
                    llm_answers[ep]["faithfulness_rationale"] = rationale

    all_ep_idx = sorted(set(
        r["ep_idx"] for r in verb_results if r["method"] == "tisae"
    ))
    if n_episodes < len(all_ep_idx):
        rng.shuffle(all_ep_idx)
        selected_ep = sorted(all_ep_idx[:n_episodes])
    else:
        selected_ep = all_ep_idx

    items = []
    for ep_idx in selected_ep:
        tisae_exp = verb_map.get((ep_idx, "tisae"))
        base_exp = verb_map.get((ep_idx, baseline))
        if not tisae_exp or not base_exp or ep_idx not in ep_map:
            continue
        tisae_is_a = ab_map.get(ep_idx, rng.random() < 0.5)
        items.append({
            "ep_idx": ep_idx,
            "tisae_is_a": tisae_is_a,
            "explanation_a": tisae_exp if tisae_is_a else base_exp,
            "explanation_b": base_exp if tisae_is_a else tisae_exp,
            "episode": ep_map[ep_idx],
            "llm_answers": llm_answers.get(ep_idx, {}),
        })

    print(f"Generated {len(items)} pairwise comparisons ({baseline} baseline)")

    blinded_dims = [
        ("concept", "Concept-Level Reasoning",
         "Which explanation better identifies clinical concepts (co-evolving "
         "variable groups) rather than listing individual abnormalities?"),
        ("actionability", "Clinical Actionability",
         "Which explanation would be more useful for a clinician deciding "
         "on next steps for this patient?"),
    ]
    all_dims = ["concept", "actionability", "faithfulness"]

    # Build per-item HTML
    items_json = []
    items_html_parts = []
    for idx, item in enumerate(items):
        ep = item["episode"]
        raw_grid = _raw_grid_html(ep["raw_values_all"], idx)
        # Analysis cards for Phase 2
        if item["tisae_is_a"]:
            analysis_a_card = _source_analysis_card_html(ep, "tisae", baseline)
            analysis_b_card = _source_analysis_card_html(ep, baseline, baseline)
        else:
            analysis_a_card = _source_analysis_card_html(ep, baseline, baseline)
            analysis_b_card = _source_analysis_card_html(ep, "tisae", baseline)
        risk = ep["risk_level"]
        prob = ep["pred_prob"]

        llm = item["llm_answers"]
        # Compact raw values for JS chart rendering
        raw_by_var = {}
        for r in ep["raw_values_all"]:
            vn = r["variable"]
            if vn not in raw_by_var:
                raw_by_var[vn] = []
            try:
                raw_by_var[vn].append([r["t"], float(r["value_raw"])])
            except (ValueError, TypeError):
                pass

        items_json.append({
            "ep_idx": item["ep_idx"],
            "tisae_is_a": item["tisae_is_a"],
            "llm": {d: llm.get(d, "") for d in all_dims},
            "llm_rationale": {d: llm.get(f"{d}_rationale", "") for d in all_dims},
            "raw_by_var": raw_by_var,
        })

        # Phase 1 scoring (blinded)
        phase1_html = []
        for dim_key, dim_label, dim_desc in blinded_dims:
            radios = []
            for choice, clabel in [("A", "A"), ("B", "B"), ("Tie", "Tie")]:
                radios.append(
                    f'<label class="choice-label"><input type="radio" '
                    f'name="{dim_key}-{idx}" value="{choice}" '
                    f'onchange="onDimSelect({idx},\'{dim_key}\')"/> {clabel}</label>'
                )
            phase1_html.append(f"""
            <div class="dim-row">
                <div class="dim-label">{dim_label}</div>
                <div class="dim-desc">{dim_desc}</div>
                <div class="dim-choices">
                    {" ".join(radios)}
                </div>
            </div>""")

        # Faith radios start disabled — enabled by lockPhase1
        faith_radios_disabled = []
        for choice, clabel in [("A", "A"), ("B", "B"), ("Tie", "Tie")]:
            faith_radios_disabled.append(
                f'<label class="choice-label faith-label-disabled"><input type="radio" '
                f'name="faithfulness-{idx}" value="{choice}" disabled '
                f'onchange="onDimSelect({idx},\'faithfulness\')"/> {clabel}</label>'
            )

        items_html_parts.append(f"""
        <div class="eval-item" id="item-{idx}">
            <h3>Comparison {idx + 1} of {len(items)}</h3>

            <div class="prediction-box">
                <strong>Prediction:</strong> {risk} mortality risk
                (probability: {prob:.3f})
            </div>

            <details class="data-section">
                <summary>Patient Data ({ep["n_observations"]} observations)
                &mdash; click to expand</summary>
                <p class="var-hint">Click any variable row to see its trend chart.</p>
                {raw_grid}
            </details>

            <div class="explanations-row" id="explanations-{idx}">
                <div class="explanation-box exp-a">
                    <h4>Explanation A</h4>
                    <p class="explanation-text">{html_mod.escape(item["explanation_a"])}</p>
                </div>
                <div class="explanation-box exp-b">
                    <h4>Explanation B</h4>
                    <p class="explanation-text">{html_mod.escape(item["explanation_b"])}</p>
                </div>
            </div>

            <!-- Source-analyses cache: hidden until lock; opened in modal window -->
            <div id="source-cache-{idx}" style="display:none;"
                 data-explanation-a="{html_mod.escape(item['explanation_a'])}"
                 data-explanation-b="{html_mod.escape(item['explanation_b'])}">
                <div id="source-a-{idx}">{analysis_a_card}</div>
                <div id="source-b-{idx}">{analysis_b_card}</div>
            </div>

            <!-- Phase 1: Blinded -->
            <div class="phase1-box" id="phase1-{idx}">
                <h4>Phase 1: Blinded Evaluation</h4>
                <p class="phase-desc">Score these 2 dimensions based on the
                explanations above. You will NOT be able to change these after
                proceeding.</p>
                {"".join(phase1_html)}
                <button class="lock-btn" id="lock-btn-{idx}"
                        onclick="lockPhase1({idx})" disabled>
                    Lock Phase 1 &amp; Reveal Source Analyses &rarr;
                </button>
            </div>

            <!-- Phase 2: Faithfulness (always visible, radios disabled until locked) -->
            <div class="phase2-box" id="phase2-{idx}">
                <h4>Phase 2: Faithfulness (Unblinded)</h4>
                <p class="phase-desc" id="phase2-desc-{idx}">Lock Phase 1 to
                enable this question. After locking, the source model analyses
                will appear in a separate window for review.</p>

                <div class="dim-row">
                    <div class="dim-label">Faithfulness</div>
                    <div class="dim-desc">Which explanation more faithfully
                    represents the findings from its source analysis &mdash;
                    accurately conveying the key patterns, priorities, and
                    values identified by the model?</div>
                    <div class="dim-choices">
                        {" ".join(faith_radios_disabled)}
                    </div>
                </div>
                <button class="reopen-source-btn" id="reopen-src-{idx}"
                        onclick="openSourceWindow({idx})" style="display:none;">
                    Re-open source analyses window
                </button>
            </div>

            <div class="comment-box">
                <label>Optional comment:</label>
                <textarea name="comment-{idx}" rows="2"
                          placeholder="Any notes..."></textarea>
            </div>
        </div>
        <hr/>
        """)

    items_json_str = json.dumps(items_json)

    page_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Pairwise Explanation Evaluation</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px;
           margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #2980b9; padding-bottom: 10px; }}
    h3 {{ color: #2c3e50; margin-top: 30px; }}
    h4 {{ color: #555; margin: 8px 0 4px 0; }}
    .instructions {{ background: #eaf2f8; border-left: 4px solid #2980b9;
                     padding: 15px; margin: 20px 0; border-radius: 4px; }}
    .eval-item {{ background: white; padding: 20px; margin: 15px 0;
                  border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .prediction-box {{ background: #f8f9fa; padding: 10px; border-radius: 4px;
                       margin: 8px 0; font-size: 14px; }}
    .data-section {{ margin: 10px 0; }}
    .data-section summary {{ cursor: pointer; color: #3498db; font-size: 14px; }}
    .raw-table {{ font-size: 11px; border-collapse: collapse; width: 100%;
                  max-height: 300px; overflow-y: auto; display: block; }}
    .raw-table th, .raw-table td {{ border: 1px solid #ddd; padding: 3px 8px;
                                    text-align: left; }}
    .raw-table th {{ background: #f0f0f0; position: sticky; top: 0; }}
    .explanations-row {{ display: flex; gap: 15px; margin: 10px 0; }}
    .explanation-box {{ flex: 1; padding: 15px; border-radius: 4px;
                        border: 1px solid #ddd; }}
    .exp-a {{ background: #fef9e7; border-left: 4px solid #f39c12; }}
    .exp-b {{ background: #eaf2f8; border-left: 4px solid #2980b9; }}
    .explanation-text {{ font-size: 14px; line-height: 1.6; color: #2c3e50; }}
    .source-analysis {{ background: #f5f5f5; padding: 10px; border-radius: 4px;
                        margin-top: 10px; font-size: 12px; border: 1px dashed #aaa; }}
    .source-analysis pre {{ white-space: pre-wrap; font-size: 11px;
                            max-height: 250px; overflow-y: auto; }}
    .phase1-box {{ background: #f0f7ff; padding: 15px; border-radius: 4px;
                   border-left: 4px solid #3498db; margin: 10px 0; }}
    .phase2-box {{ background: #fdf2e9; padding: 15px; border-radius: 4px;
                   border-left: 4px solid #e67e22; margin: 10px 0; }}
    .phase-desc {{ font-size: 13px; color: #666; margin: 4px 0 12px 0; }}
    .dim-row {{ margin: 12px 0; padding: 8px 0; border-bottom: 1px solid #eee; }}
    .dim-label {{ font-weight: bold; font-size: 14px; color: #2c3e50; }}
    .dim-desc {{ font-size: 12px; color: #777; margin: 2px 0 6px 0; }}
    .dim-choices {{ display: flex; align-items: center; gap: 20px; }}
    .choice-label {{ font-size: 14px; cursor: pointer; padding: 4px 12px;
                     border: 1px solid #ddd; border-radius: 4px; }}
    .choice-label:hover {{ background: #e8e8e8; }}
    .choice-label input {{ margin-right: 4px; }}
    .lock-btn {{ background: #3498db; color: white; border: none;
                 padding: 10px 24px; font-size: 14px; border-radius: 4px;
                 cursor: pointer; margin-top: 10px; }}
    .lock-btn:hover:not(:disabled) {{ background: #2980b9; }}
    .lock-btn:disabled {{ background: #bdc3c7; cursor: not-allowed; }}
    .locked-row {{ display: none; align-items: center; gap: 12px; margin-top: 8px; }}
    .locked-notice {{ background: #d5f5e3; padding: 6px 12px; border-radius: 4px;
                      color: #27ae60; font-weight: bold; }}
    .comment-box {{ margin-top: 10px; }}
    .comment-box textarea {{ width: 100%; font-size: 13px; padding: 6px;
                             border: 1px solid #ddd; border-radius: 4px; }}
    hr {{ border: none; border-top: 1px solid #eee; margin: 30px 0; }}
    .export-section {{ background: #d4edda; padding: 15px; border-radius: 8px;
                       margin: 20px 0; text-align: center; }}
    .export-btn {{ background: #28a745; color: white; border: none;
                   padding: 12px 30px; font-size: 16px; border-radius: 4px;
                   cursor: pointer; }}
    .export-btn:hover {{ background: #218838; }}
    .progress {{ position: fixed; top: 0; left: 0; right: 0; background: #2c3e50;
                 color: white; padding: 8px 20px; font-size: 14px; z-index: 100;
                 display: flex; justify-content: space-between; }}
    #evaluator-name {{ padding: 6px 10px; font-size: 14px; width: 300px;
                       border: 1px solid #ddd; border-radius: 4px; }}
    .reopen-source-btn {{ background: #e67e22; color: white; border: none;
                          padding: 6px 14px; font-size: 12px; border-radius: 3px;
                          cursor: pointer; margin-top: 6px; }}
    .reopen-source-btn:hover {{ background: #d35400; }}
    .faith-label-disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .source-window-bg {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0;
                         background: rgba(0,0,0,0.4); z-index: 2000; }}
    .source-window {{ position: fixed; top: 5%; left: 5%; right: 5%; bottom: 5%;
                      background: white; border: 2px solid #e67e22; border-radius: 8px;
                      box-shadow: 0 8px 30px rgba(0,0,0,0.3); z-index: 2001;
                      padding: 20px; overflow-y: auto; }}
    .source-window-title {{ font-size: 16px; font-weight: bold; color: #2c3e50;
                            margin-bottom: 6px; }}
    .source-window-subtitle {{ font-size: 12px; color: #777; margin-bottom: 12px; }}
    .source-window-close {{ position: absolute; top: 12px; right: 14px;
                            background: #e74c3c; color: white; border: none;
                            border-radius: 50%; width: 28px; height: 28px;
                            font-size: 14px; cursor: pointer; }}
    .source-window-close:hover {{ background: #c0392b; }}
    .source-pair-row {{ display: flex; gap: 15px; margin-top: 10px; }}
    .source-pair-col {{ flex: 1; min-width: 0; }}
    .source-pair-col h4 {{ margin: 4px 0 8px 0; }}
    .source-explanation {{ background: #fef9e7; padding: 10px; border-radius: 4px;
                           font-size: 13px; line-height: 1.5; margin-bottom: 8px;
                           border-left: 4px solid #f39c12; }}
    .source-pair-col.col-b .source-explanation {{ background: #eaf2f8;
                                                   border-left-color: #2980b9; }}
    .var-hint {{ font-size: 11px; color: #999; margin: 4px 0; }}
    .grid-wrapper {{ overflow-x: auto; max-height: 400px; overflow-y: auto; margin: 6px 0; }}
    .vt-grid {{ border-collapse: collapse; font-size: 10px; white-space: nowrap; }}
    .vt-grid th, .vt-grid td {{ border: 1px solid #e0e0e0; padding: 2px 4px; text-align: center; }}
    .vt-grid th {{ background: #f0f0f0; position: sticky; top: 0; z-index: 2; }}
    .var-name-col {{ position: sticky; left: 0; z-index: 3; background: #f0f0f0; min-width: 120px; text-align: left !important; }}
    .var-name-cell {{ position: sticky; left: 0; z-index: 1; background: #fff; font-weight: bold;
                      color: #2980b9; cursor: pointer; text-align: left !important; min-width: 120px; }}
    .var-name-cell:hover {{ background: #eaf2f8; text-decoration: underline; }}
    .hour-col {{ min-width: 38px; }}
    .obs-cell {{ background: #eaf7ff; font-size: 9px; }}
    .empty-cell {{ background: #fafafa; }}
    .var-row:hover td {{ background: #eaf2f8 !important; }}
    .chart-overlay {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                      background: white; border: 2px solid #2980b9; border-radius: 8px;
                      box-shadow: 0 8px 30px rgba(0,0,0,0.3); z-index: 1000; padding: 10px; }}
    .chart-overlay-bg {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0;
                         background: rgba(0,0,0,0.3); z-index: 999; }}
    .chart-close {{ position: absolute; top: 6px; right: 10px; background: #e74c3c;
                    color: white; border: none; border-radius: 50%; width: 24px; height: 24px;
                    font-size: 14px; cursor: pointer; line-height: 24px; text-align: center; }}
    .chart-close:hover {{ background: #c0392b; }}
    .analysis-card {{ padding: 10px; border-radius: 4px; margin-top: 10px; }}
    .concept-card {{ background: #f9f5ff; border: 1px solid #d5c8f0; }}
    .attr-card {{ background: #f5f9ff; border: 1px solid #c8d8f0; }}
    .analysis-title {{ font-weight: bold; font-size: 13px; color: #555; margin-bottom: 8px; }}
    .concept-item {{ margin: 6px 0; padding: 6px; background: white; border-radius: 4px; border: 1px solid #eee; }}
    .concept-header {{ display: flex; align-items: center; gap: 8px; }}
    .concept-rank {{ font-weight: bold; color: #8e44ad; font-size: 13px; }}
    .concept-channel {{ background: #e8daef; color: #6c3483; padding: 1px 8px; border-radius: 10px; font-size: 10px; font-weight: bold; }}
    .concept-label {{ font-weight: bold; font-size: 13px; color: #2c3e50; }}
    .concept-weight-bar {{ height: 8px; background: #eee; border-radius: 4px; margin: 4px 0; position: relative; }}
    .concept-weight-fill {{ height: 100%; background: #8e44ad; border-radius: 4px; }}
    .concept-weight-text {{ position: absolute; right: 4px; top: -2px; font-size: 9px; color: #666; }}
    .concept-vars {{ font-size: 11px; color: #666; }}
    .concept-islands {{ font-size: 10px; color: #999; margin-top: 2px; }}
    .attr-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; font-size: 11px; }}
    .attr-rank {{ font-weight: bold; color: #2980b9; min-width: 24px; }}
    .attr-var {{ min-width: 200px; color: #333; }}
    .attr-bar {{ width: 80px; height: 8px; background: #eee; border-radius: 4px; }}
    .attr-bar-fill {{ height: 100%; background: #3498db; border-radius: 4px; }}
    .attr-imp {{ font-size: 10px; color: #888; min-width: 50px; }}
    .attr-more {{ font-size: 10px; color: #999; margin-top: 4px; }}
    .attr-grid {{ border-collapse: collapse; font-size: 10px; white-space: nowrap;
                  width: 100%; }}
    .attr-grid th, .attr-grid td {{ border: 1px solid #e0e0e0; padding: 2px 3px; text-align: center; }}
    .attr-grid th {{ background: #f0f0f0; position: sticky; top: 0; z-index: 2; font-size: 9px; }}
    .attr-grid-wrap {{ overflow-x: auto; overflow-y: auto; max-height: 60vh;
                       max-width: 100%; width: 100%; }}
    .var-name-cell-attr {{ position: sticky; left: 0; z-index: 1; background: #fff;
                           font-weight: bold; color: #2980b9; text-align: left !important;
                           min-width: 100px; max-width: 140px; font-size: 9px;
                           overflow: hidden; text-overflow: ellipsis; }}
    .attr-obs-cell {{ font-size: 9px; min-width: 30px; }}
    .hour-col-attr {{ min-width: 30px; font-size: 9px; }}
</style>
</head>
<body>

<div class="progress">
    <span>Pairwise Explanation Evaluation (vs. {baseline.title()})</span>
    <span id="progress-text">0 / {len(items)} complete</span>
</div>

<div style="margin-top: 50px;">
<h1>Pairwise Explanation Evaluation</h1>

<div class="instructions">
    <h4>Instructions</h4>
    <p>You will compare <strong>{len(items)} pairs</strong> of clinical
    explanations for ICU mortality predictions. Each pair explains the same
    patient's prediction using two different methods (A and B, randomly
    assigned).</p>

    <p><strong>Two-phase protocol:</strong></p>
    <ol>
        <li><strong>Phase 1 (Blinded):</strong> Read both explanations and the
        patient data. For each of 2 dimensions, select which explanation is
        better (A, B, or Tie). Then click &ldquo;Lock&rdquo; to proceed.
        <em>You cannot change Phase 1 answers after locking.</em></li>
        <li><strong>Phase 2 (Unblinded):</strong> The source model analyses
        open in a separate window. Judge which explanation more faithfully
        represents its own source analysis. The faithfulness question becomes
        clickable only after locking Phase 1.</li>
    </ol>

    <p>Focus on the quality of the explanation, not whether the prediction is
    correct.</p>

    <p><strong>Evaluator:</strong>
    <input type="text" id="evaluator-name" value="{evaluator_name}"
           placeholder="Your name"/></p>
</div>

{"".join(items_html_parts)}

<div class="export-section">
    <h3>Export Results</h3>
    <p id="export-warning" style="color:#c0392b; display:none;">
        Warning: not all comparisons are complete.</p>
    <button class="export-btn" onclick="exportCSV()">Download CSV</button>
</div>
</div>

<script>
const ITEMS = {items_json_str};
const N = {len(items)};
const BLINDED_DIMS = ["concept", "actionability"];
const ALL_DIMS = ["concept", "actionability", "faithfulness"];

function checkPhase1(idx) {{
    let allScored = true;
    for (const dim of BLINDED_DIMS) {{
        const radios = document.getElementsByName(dim + '-' + idx);
        let scored = false;
        for (const r of radios) {{ if (r.checked) {{ scored = true; break; }} }}
        if (!scored) {{ allScored = false; break; }}
    }}
    document.getElementById('lock-btn-' + idx).disabled = !allScored;
}}

function onDimSelect(idx, dim) {{
    checkPhase1(idx);
    updateProgress();
    saveState();
}}

// ---- localStorage auto-save / restore ----
const STORAGE_KEY = 'clinician_eval_{baseline}_{len(items)}';
let _saveTimer = null;

function saveState() {{
    if (_saveTimer) clearTimeout(_saveTimer);
    _saveTimer = setTimeout(_saveStateNow, 300);
}}

function _saveStateNow() {{
    const state = {{ evaluator: '', items: {{}} }};
    const evInput = document.getElementById('evaluator-name');
    if (evInput) state.evaluator = evInput.value;
    for (let i = 0; i < N; i++) {{
        const itemState = {{}};
        for (const dim of ALL_DIMS) {{
            const radios = document.getElementsByName(dim + '-' + i);
            let val = '';
            for (const r of radios) {{ if (r.checked) {{ val = r.value; break; }} }}
            if (val) itemState[dim] = val;
        }}
        const comment = document.getElementsByName('comment-' + i)[0];
        if (comment && comment.value) itemState.comment = comment.value;
        const lockBtn = document.getElementById('lock-btn-' + i);
        if (lockBtn && lockBtn.style.display === 'none') itemState.locked = true;
        if (Object.keys(itemState).length > 0) state.items[i] = itemState;
    }}
    state.saved_at = new Date().toISOString();
    try {{
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        const status = document.getElementById('save-status');
        if (status) {{
            const t = new Date();
            const hh = String(t.getHours()).padStart(2, '0');
            const mm = String(t.getMinutes()).padStart(2, '0');
            const ss = String(t.getSeconds()).padStart(2, '0');
            status.textContent = 'Auto-saved ' + hh + ':' + mm + ':' + ss;
            status.className = 'save-status saved';
        }}
    }} catch (e) {{
        console.warn('localStorage save failed:', e);
    }}
}}

function restoreState() {{
    let raw = null;
    try {{ raw = localStorage.getItem(STORAGE_KEY); }} catch (e) {{ return; }}
    if (!raw) return;
    let state;
    try {{ state = JSON.parse(raw); }} catch (e) {{ return; }}
    if (state.evaluator) {{
        const ev = document.getElementById('evaluator-name');
        if (ev) ev.value = state.evaluator;
    }}
    const items = state.items || {{}};
    for (const i_str in items) {{
        const i = parseInt(i_str);
        const it = items[i_str];
        for (const dim of ALL_DIMS) {{
            if (!it[dim]) continue;
            const radios = document.getElementsByName(dim + '-' + i);
            for (const r of radios) {{
                if (r.value === it[dim]) {{ r.checked = true; break; }}
            }}
        }}
        if (it.comment) {{
            const c = document.getElementsByName('comment-' + i)[0];
            if (c) c.value = it.comment;
        }}
        checkPhase1(i);
        if (it.locked) lockPhase1Silent(i);
    }}
    updateProgress();
    const status = document.getElementById('save-status');
    if (status && state.saved_at) {{
        status.textContent = 'Restored from ' + state.saved_at.slice(0, 19).replace('T', ' ');
        status.className = 'save-status saved';
    }}
}}

function lockPhase1Silent(idx) {{
    // Same as lockPhase1 but does NOT open the source window (used on restore).
    for (const dim of BLINDED_DIMS) {{
        const radios = document.getElementsByName(dim + '-' + idx);
        for (const r of radios) r.disabled = true;
    }}
    document.getElementById('lock-btn-' + idx).style.display = 'none';
    const phase1Box = document.getElementById('phase1-' + idx);
    let lockedRow = document.getElementById('locked-row-' + idx);
    if (!lockedRow) {{
        lockedRow = document.createElement('div');
        lockedRow.id = 'locked-row-' + idx;
        lockedRow.className = 'locked-row';
        lockedRow.innerHTML =
            '<span class="locked-notice">Phase 1 locked.</span>';
        phase1Box.appendChild(lockedRow);
    }}
    lockedRow.style.display = 'flex';
    const fRadios = document.getElementsByName('faithfulness-' + idx);
    for (const r of fRadios) {{
        r.disabled = false;
        const lab = r.parentElement;
        if (lab) lab.classList.remove('faith-label-disabled');
    }}
    const desc = document.getElementById('phase2-desc-' + idx);
    if (desc) desc.textContent =
        'Source model analyses are shown in a separate window. ' +
        'Use the button below to re-open it as needed.';
    document.getElementById('reopen-src-' + idx).style.display = 'inline-block';
}}

function clearSavedState() {{
    if (!confirm('Clear all auto-saved data for this evaluation? Your current entries on the page will not be erased until you refresh.')) return;
    try {{ localStorage.removeItem(STORAGE_KEY); }} catch (e) {{}}
    const status = document.getElementById('save-status');
    if (status) {{
        status.textContent = 'Saved data cleared (refresh page to reset form)';
        status.className = 'save-status cleared';
    }}
}}

// Save on page unload, comment input, evaluator name change, lock
window.addEventListener('beforeunload', _saveStateNow);
document.addEventListener('input', function(e) {{
    if (e.target && (e.target.tagName === 'TEXTAREA' || e.target.id === 'evaluator-name')) {{
        saveState();
    }}
}});
document.addEventListener('DOMContentLoaded', restoreState);

function showVarChart(idx, varName) {{
    const item = ITEMS[idx];
    const data = item.raw_by_var[varName];
    if (!data || data.length === 0) return;

    // Remove existing overlay
    closeVarChart();

    const W = 650, H = 200, pad = {{l:60, r:25, t:35, b:35}};
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

    const ts = data.map(d => d[0]), vs = data.map(d => d[1]);
    const tMin = 0, tMax = 48;
    let vMin = Math.min(...vs), vMax = Math.max(...vs);
    const vPad = Math.max((vMax - vMin) * 0.1, 0.5);
    vMin -= vPad; vMax += vPad;

    const sx = t => pad.l + (t - tMin) / (tMax - tMin) * pw;
    const sy = v => pad.t + (1 - (v - vMin) / (vMax - vMin)) * ph;

    let svg = `<svg width="${{W}}" height="${{H}}">`;
    svg += `<rect width="${{W}}" height="${{H}}" fill="#fafafa" rx="6"/>`;
    svg += `<text x="${{pad.l}}" y="22" font-size="14" font-weight="bold" fill="#2c3e50">${{varName}}</text>`;
    svg += `<text x="${{W-pad.r}}" y="22" font-size="11" fill="#999" text-anchor="end">${{data.length}} observations</text>`;
    for (let t = 0; t <= 48; t += 6) {{
        const x = sx(t);
        svg += `<line x1="${{x}}" y1="${{pad.t}}" x2="${{x}}" y2="${{H-pad.b}}" stroke="#e0e0e0" stroke-width="1"/>`;
        svg += `<text x="${{x}}" y="${{H-pad.b+16}}" text-anchor="middle" font-size="10" fill="#888">h${{t}}</text>`;
    }}
    const nyt = 4;
    for (let i = 0; i <= nyt; i++) {{
        const v = vMin + (vMax - vMin) * i / nyt;
        const y = sy(v);
        svg += `<line x1="${{pad.l}}" y1="${{y}}" x2="${{W-pad.r}}" y2="${{y}}" stroke="#e8e8e8" stroke-width="1"/>`;
        svg += `<text x="${{pad.l-6}}" y="${{y+4}}" text-anchor="end" font-size="10" fill="#888">${{v.toFixed(1)}}</text>`;
    }}
    const sorted = data.slice().sort((a,b) => a[0]-b[0]);
    if (sorted.length > 1) {{
        let path = `M ${{sx(sorted[0][0])}} ${{sy(sorted[0][1])}}`;
        for (let i = 1; i < sorted.length; i++) path += ` L ${{sx(sorted[i][0])}} ${{sy(sorted[i][1])}}`;
        svg += `<path d="${{path}}" fill="none" stroke="#3498db" stroke-width="2" opacity="0.4"/>`;
    }}
    for (const [t, v] of data) {{
        svg += `<circle cx="${{sx(t)}}" cy="${{sy(v)}}" r="4.5" fill="#3498db" stroke="white" stroke-width="1.2"/>`;
        svg += `<text x="${{sx(t)+6}}" y="${{sy(v)-7}}" font-size="9" fill="#444">${{v.toFixed(1)}}</text>`;
    }}
    svg += '</svg>';

    // Create overlay
    const bg = document.createElement('div');
    bg.className = 'chart-overlay-bg';
    bg.id = 'chart-bg';
    bg.onclick = closeVarChart;
    document.body.appendChild(bg);

    const overlay = document.createElement('div');
    overlay.className = 'chart-overlay';
    overlay.id = 'chart-overlay';
    overlay.innerHTML = `<button class="chart-close" onclick="closeVarChart()">✕</button>${{svg}}`;
    document.body.appendChild(overlay);
}}

function closeVarChart() {{
    const bg = document.getElementById('chart-bg');
    const ov = document.getElementById('chart-overlay');
    if (bg) bg.remove();
    if (ov) ov.remove();
}}



function lockPhase1(idx) {{
    // Disable Phase 1 radios
    for (const dim of BLINDED_DIMS) {{
        const radios = document.getElementsByName(dim + '-' + idx);
        for (const r of radios) r.disabled = true;
    }}
    document.getElementById('lock-btn-' + idx).style.display = 'none';
    // Show locked notice
    const phase1Box = document.getElementById('phase1-' + idx);
    let lockedRow = document.getElementById('locked-row-' + idx);
    if (!lockedRow) {{
        lockedRow = document.createElement('div');
        lockedRow.id = 'locked-row-' + idx;
        lockedRow.className = 'locked-row';
        lockedRow.innerHTML =
            '<span class="locked-notice">Phase 1 locked.</span>';
        phase1Box.appendChild(lockedRow);
    }}
    lockedRow.style.display = 'flex';
    // Enable faithfulness radios
    const fRadios = document.getElementsByName('faithfulness-' + idx);
    for (const r of fRadios) {{
        r.disabled = false;
        const lab = r.parentElement;
        if (lab) lab.classList.remove('faith-label-disabled');
    }}
    // Update phase 2 description
    const desc = document.getElementById('phase2-desc-' + idx);
    if (desc) desc.textContent =
        'Source model analyses are shown in a separate window. ' +
        'Use the button below to re-open it as needed.';
    document.getElementById('reopen-src-' + idx).style.display = 'inline-block';
    // Open source window
    openSourceWindow(idx);
    updateProgress();
    saveState();
}}


function openSourceWindow(idx) {{
    closeSourceWindow();
    const cache = document.getElementById('source-cache-' + idx);
    const expA = cache.dataset.explanationA;
    const expB = cache.dataset.explanationB;
    const srcA = document.getElementById('source-a-' + idx).innerHTML;
    const srcB = document.getElementById('source-b-' + idx).innerHTML;

    const bg = document.createElement('div');
    bg.className = 'source-window-bg';
    bg.id = 'source-window-bg';
    bg.onclick = closeSourceWindow;
    document.body.appendChild(bg);

    const win = document.createElement('div');
    win.className = 'source-window';
    win.id = 'source-window';
    win.innerHTML =
        '<button class="source-window-close" onclick="closeSourceWindow()">&times;</button>' +
        '<div class="source-window-title">Source Analyses (Comparison ' + (idx + 1) + ')</div>' +
        '<div class="source-window-subtitle">Each explanation was derived from the analysis shown beneath it. Use this window to judge faithfulness.</div>' +
        '<div class="source-pair-row">' +
            '<div class="source-pair-col col-a">' +
                '<h4>Explanation A</h4>' +
                '<div class="source-explanation">' + expA + '</div>' +
                srcA +
            '</div>' +
            '<div class="source-pair-col col-b">' +
                '<h4>Explanation B</h4>' +
                '<div class="source-explanation">' + expB + '</div>' +
                srcB +
            '</div>' +
        '</div>';
    document.body.appendChild(win);
}}

function closeSourceWindow() {{
    const bg = document.getElementById('source-window-bg');
    const win = document.getElementById('source-window');
    if (bg) bg.remove();
    if (win) win.remove();
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') closeSourceWindow();
}});

function updateProgress() {{
    let complete = 0;
    for (let i = 0; i < N; i++) {{
        // Item complete = lock-btn hidden (phase 1 locked) AND faith chosen
        const lockBtn = document.getElementById('lock-btn-' + i);
        if (lockBtn && lockBtn.style.display !== 'none') continue;
        const fRadios = document.getElementsByName('faithfulness-' + i);
        let fDone = false;
        for (const r of fRadios) {{ if (r.checked) {{ fDone = true; break; }} }}
        if (fDone) complete++;
    }}
    document.getElementById('progress-text').textContent =
        complete + ' / ' + N + ' complete';
    document.getElementById('export-warning').style.display =
        (complete < N) ? 'block' : 'none';
}}
document.addEventListener('change', updateProgress);

function exportCSV() {{
    const name = document.getElementById('evaluator-name').value || 'anonymous';
    const dims = ALL_DIMS;
    let csv = 'evaluator,ep_idx,tisae_is_a,' + dims.join(',') + ',comment\\n';
    for (let i = 0; i < N; i++) {{
        const item = ITEMS[i];
        let scores = [];
        for (const d of dims) {{
            const radios = document.getElementsByName(d + '-' + i);
            let val = '';
            for (const r of radios) {{ if (r.checked) {{ val = r.value; break; }} }}
            scores.push(val);
        }}
        const c = (document.getElementsByName('comment-' + i)[0]?.value || '')
                    .replace(/"/g, '""');
        csv += '"' + name + '",' + item.ep_idx + ',' + item.tisae_is_a + ',' +
               scores.join(',') + ',"' + c + '"\\n';
    }}
    const blob = new Blob([csv], {{type: 'text/csv'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pairwise_eval_' + name.replace(/\\s/g, '_') + '.csv';
    a.click();
}}
</script>
</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html)
    print(f"Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pairwise clinician evaluation HTML")
    parser.add_argument("--episodes_path", required=True)
    parser.add_argument("--verb_results_path", required=True)
    parser.add_argument("--output_path", default="clinician_pairwise_eval.html")
    parser.add_argument("--baseline", default="attention",
                        choices=["attention", "ig", "tsr"])
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairwise_results_path", default=None,
                        help="Path to LLM judge pairwise_results.json "
                             "(concept/actionability/faithfulness)")
    parser.add_argument("--faithfulness_results_path", default=None,
                        help="Path to v15 pairwise_results.json "
                             "(faithfulness dimension)")
    parser.add_argument("--evaluator_name", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.episodes_path) as f:
        episodes = json.load(f)
    with open(args.verb_results_path) as f:
        verb_results = json.load(f)
    pairwise_results = None
    if args.pairwise_results_path:
        with open(args.pairwise_results_path) as f:
            pairwise_results = json.load(f)
    faithfulness_results = None
    if args.faithfulness_results_path:
        with open(args.faithfulness_results_path) as f:
            faithfulness_results = json.load(f)
    generate_pairwise_eval_html(
        episodes, verb_results, args.output_path,
        baseline=args.baseline, n_episodes=args.n_episodes,
        seed=args.seed, evaluator_name=args.evaluator_name,
        pairwise_results=pairwise_results,
        faithfulness_results=faithfulness_results,
    )
