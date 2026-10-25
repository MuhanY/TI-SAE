"""
Build verbalizer and judge prompts for LLM-as-a-Judge evaluation.

Verbalizer (Claude Sonnet): generates clinical explanations under 3 conditions.
Judge (OpenAI GPT-4o): pairwise comparison on 3 dimensions (concept, actionability blinded; faithfulness unblinded).

Data flow:
  extract_episodes.py  -> episodes_data.json
  build_prompts.py     -> verb_prompts/ + judge_prompts/
  run_verbalization.py -> verb_results.json + judge_results.json
  aggregate_results.py -> latex table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


VERBALIZER_SYSTEM = (
    "You are an ICU physician writing a clinical explanation of a mortality "
    "prediction. In under 150 words, explain why this patient is at this "
    "risk level. Explain exclusively based on the model-provided analysis "
    "below. Plain text, no markdown."
)

JUDGE_SYSTEM = (
    "You are an expert evaluator assessing the quality of clinical "
    "explanations for ICU mortality predictions. Score each explanation "
    "on the provided rubric. Be discriminating — reserve 5 for genuinely "
    "excellent explanations."
)

PAIRWISE_JUDGE_SYSTEM = (
    "You are an expert clinical informaticist comparing two AI-generated "
    "explanations for the same ICU mortality prediction. For each dimension, "
    "select which explanation is better. Be discriminating."
)

PAIRWISE_RUBRIC = """For each dimension, select which explanation (A or B) is better, or Tie.

1. CONCEPT-LEVEL REASONING: Which better identifies clinical concepts (co-evolving variable groups) rather than listing individual abnormalities?

2. CLINICAL ACTIONABILITY: Which would be more useful for a clinician deciding on next steps for this patient?

3. OVERALL: Which explanation is better overall?

Respond ONLY with JSON:
{"concept": "A/B/Tie", "actionability": "A/B/Tie", "overall": "A/B/Tie", "rationale": "one sentence"}"""

PAIRWISE_DIM_RUBRICS = {
    "concept": (
        "You are an expert clinical informaticist comparing two AI-generated "
        "explanations for the same ICU mortality prediction. Evaluate ONLY "
        "concept-level reasoning. Be discriminating.",
        "CONCEPT-LEVEL REASONING: Which explanation better identifies clinical "
        "concepts (co-evolving variable groups) rather than listing individual "
        "abnormalities?\n\n"
        "Respond ONLY with JSON:\n"
        '{"winner": "A/B/Tie", "rationale": "one sentence"}'
    ),
    "actionability": (
        "You are an expert clinical informaticist comparing two AI-generated "
        "explanations for the same ICU mortality prediction. Evaluate ONLY "
        "clinical actionability. Be discriminating.",
        "CLINICAL ACTIONABILITY: Which explanation would be more useful for a "
        "clinician deciding on next steps for this patient?\n\n"
        "Respond ONLY with JSON:\n"
        '{"winner": "A/B/Tie", "rationale": "one sentence"}'
    ),
}

FAITHFULNESS_RUBRIC = (
    "You are an expert clinical informaticist evaluating how faithfully two "
    "AI-generated explanations represent their corresponding model analyses. "
    "Each explanation was derived from a different model analysis of the same "
    "patient. Evaluate ONLY faithfulness. Be discriminating.",
    "FAITHFULNESS: Each explanation was derived from its corresponding model "
    "analysis shown above. Which explanation more faithfully represents the "
    "findings from its source analysis — accurately conveying the key patterns, "
    "priorities, and values identified by the model?\n\n"
    "Respond ONLY with JSON:\n"
    '{"winner": "A/B/Tie", "rationale": "one sentence"}'
)

CONCEPT_FRAMING = (
    "The model identifies clinical concepts — groups of co-evolving variables "
    "active in specific time windows. Each concept is labeled by its dominant "
    "clinical channel (e.g., Renal, Respiratory)."
)

METHOD_LABELS = {
    "raw": "Raw values only",
    "attention": "Attention",
    "ig": "IG",
    "gradient_shap": "GradSHAP",
    "tsr": "TSR",
    "timeshap": "TimeSHAP",
    "dynamask": "Dynamask",
    "tisae": "TI-SAE (ours)",
}

JUDGE_RUBRIC = """=== SCORING RUBRIC (2 dimensions, 1-5 each) ===

1. CONCEPT-LEVEL REASONING: Does the explanation identify clinical concepts (patterns of related variables evolving together, e.g., "acute kidney injury" = rising creatinine + falling urine + elevated BUN), or does it just list individual variable abnormalities? (1=individual variable listing, 3=some grouping, 5=clear clinical concepts with co-evolving variable groups)

2. CLINICAL ACTIONABILITY: Could a clinician use this explanation to understand WHY the model made this prediction and trust/question it? (1=not useful, 3=somewhat helpful, 5=highly informative)

Evaluate the explanation on its own merits. Do not consider whether the prediction was correct.
Be discriminating — reserve 5 for genuinely excellent explanations.

Respond ONLY with JSON:
{"concept": N, "actionability": N, "brief_rationale": "one sentence"}"""

DIMENSIONS = ["concept", "actionability"]


def tabulate_raw(raw_all: List[dict]) -> str:
    header = f"{'Hour':>4} | {'Variable':<30} | {'Value':>10} | {'Decile':>6}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in raw_all:
        lines.append(
            f"{r['t']:>4} | {r['variable']:<30} | {r['value_raw']:>10} | {r['bin']:>6}"
        )
    return "\n".join(lines)


def format_attribution_cells(cells: List[dict]) -> str:
    lines = []
    for c in cells:
        val_str = f", value: {c['value_raw']}" if c.get("value_raw") is not None else ""
        lines.append(
            f"  Hour {c['t']}: {c['variable']} "
            f"(importance: {c['importance']:.4f}{val_str})"
        )
    return "\n".join(lines)


def format_concepts(concepts: List[dict]) -> str:
    lines = []
    for con in concepts:
        vars_str = ", ".join(f"{v[0]} (wt={v[1]})" for v in con["top_variables"][:3])
        label_part = f' "{con["clinical_label"]}"' if con.get("clinical_label") else ""
        lines.append(
            f"  Concept {con['rank'] + 1} [{con['channel']}]{label_part} "
            f"(total weight={con['total_weight']:.1f}): {vars_str}"
        )
        if con.get("clinical_description"):
            first_sentence = con["clinical_description"].split(". ")[0].rstrip(".") + "."
            lines.append(f"    {first_sentence}")
        islands = sorted(
            [isl for isl in con["islands"] if isl["weight"] >= 0.01],
            key=lambda x: x["t_start"],
        )
        for isl in islands:
            lines.append(f"    Island h{isl['t_start']}-{isl['t_end']} (wt={isl['weight']:.2f}):")
            for vn, vals in isl.get("values", {}).items():
                if vals:
                    val_str = ", ".join(f"h{v['t']}={v['value_raw']}" for v in vals[:6])
                    lines.append(f"      {vn}: {val_str}")
    return "\n".join(lines)


class PromptBuilder:
    """Builds verbalizer and judge prompts. Judge is blinded."""

    def build_verbalizer_prompt(self, episode: dict, method: str) -> str:
        tab_str = tabulate_raw(episode["raw_values_all"])
        risk = episode["risk_level"]
        prob = episode["pred_prob"]

        if method == "raw":
            body = self._build_verb_raw(risk, prob, tab_str)
        elif method == "tisae":
            body = self._build_verb_tisae(episode, risk, prob, tab_str)
        else:
            body = self._build_verb_attribution(episode, method, risk, prob, tab_str)

        return body

    def build_judge_prompt(self, episode: dict, explanation: str) -> str:
        tab_str = tabulate_raw(episode["raw_values_all"])
        risk = episode["risk_level"]
        prob = episode["pred_prob"]

        return (
            f"Prediction: {risk} mortality risk (probability: {prob})\n\n"
            f"=== PATIENT DATA (48-hour ICU stay) ===\n"
            f"{tab_str}\n\n"
            f"=== EXPLANATION TO EVALUATE ===\n"
            f"{explanation}\n\n"
            f"{JUDGE_RUBRIC}\n"
        )

    def build_pairwise_judge_prompt(
        self, episode: dict, explanation_a: str, explanation_b: str,
    ) -> str:
        """Build pairwise comparison prompt. Caller randomizes A/B order."""
        tab_str = tabulate_raw(episode["raw_values_all"])
        risk = episode["risk_level"]
        prob = episode["pred_prob"]

        return (
            f"Prediction: {risk} mortality risk (probability: {prob})\n\n"
            f"=== PATIENT DATA (48-hour ICU stay) ===\n"
            f"{tab_str}\n\n"
            f"=== EXPLANATION A ===\n"
            f"{explanation_a}\n\n"
            f"=== EXPLANATION B ===\n"
            f"{explanation_b}\n\n"
            f"{PAIRWISE_RUBRIC}\n"
        )

    def build_pairwise_single_dim_prompt(
        self, episode: dict, explanation_a: str, explanation_b: str, dim: str,
    ) -> tuple:
        """Build pairwise prompt for a single dimension. Returns (prompt, system)."""
        tab_str = tabulate_raw(episode["raw_values_all"])
        risk = episode["risk_level"]
        prob = episode["pred_prob"]

        system, rubric = PAIRWISE_DIM_RUBRICS[dim]
        prompt = (
            f"Prediction: {risk} mortality risk (probability: {prob})\n\n"
            f"=== PATIENT DATA (48-hour ICU stay) ===\n"
            f"{tab_str}\n\n"
            f"=== EXPLANATION A ===\n"
            f"{explanation_a}\n\n"
            f"=== EXPLANATION B ===\n"
            f"{explanation_b}\n\n"
            f"{rubric}\n"
        )
        return prompt, system

    def build_pairwise_faithfulness_prompt(
        self, episode: dict,
        explanation_a: str, analysis_a: str,
        explanation_b: str, analysis_b: str,
    ) -> tuple:
        """Build faithfulness prompt with source analyses visible. Returns (prompt, system)."""
        tab_str = tabulate_raw(episode["raw_values_all"])
        risk = episode["risk_level"]
        prob = episode["pred_prob"]

        system, rubric = FAITHFULNESS_RUBRIC
        prompt = (
            f"Prediction: {risk} mortality risk (probability: {prob})\n\n"
            f"=== PATIENT DATA (48-hour ICU stay) ===\n"
            f"{tab_str}\n\n"
            f"=== MODEL ANALYSIS A ===\n"
            f"{analysis_a}\n\n"
            f"=== EXPLANATION A (derived from Model Analysis A) ===\n"
            f"{explanation_a}\n\n"
            f"=== MODEL ANALYSIS B ===\n"
            f"{analysis_b}\n\n"
            f"=== EXPLANATION B (derived from Model Analysis B) ===\n"
            f"{explanation_b}\n\n"
            f"{rubric}\n"
        )
        return prompt, system

    def build_all_verbalizer_prompts(self, episodes, methods, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for ep in episodes:
            for method in methods:
                key = f"{ep['ep_idx']}_{method}"
                path = out / f"{key}.txt"
                path.write_text(self.build_verbalizer_prompt(ep, method))
                manifest[key] = str(path)
        with open(out / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def build_all_judge_prompts(self, episodes, verb_results, methods, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        exp_map = {(r["ep_idx"], r["method"]): r["explanation"] for r in verb_results}
        manifest = {}
        for ep in episodes:
            for method in methods:
                key = f"{ep['ep_idx']}_{method}"
                explanation = exp_map.get((ep["ep_idx"], method), "NO EXPLANATION")
                path = out / f"{key}.txt"
                path.write_text(self.build_judge_prompt(ep, explanation))
                manifest[key] = str(path)
        with open(out / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    def _build_verb_raw(self, risk, prob, tab_str):
        return (
            f"Patient prediction: {risk} mortality risk (probability: {prob})\n\n"
            f"Clinical values over the 48-hour ICU stay:\n{tab_str}"
        )

    def _build_verb_attribution(self, episode, method, risk, prob, tab_str):
        cells = episode.get("attribution_cells", {}).get(method, [])
        if not cells:
            return self._build_verb_raw(risk, prob, tab_str)
        cells_str = format_attribution_cells(cells)
        return (
            f"Patient prediction: {risk} mortality risk (probability: {prob})\n\n"
            f"The model's attribution analysis identified these observations as "
            f"most important for its prediction:\n{cells_str}\n\n"
            f"Clinical values over the 48-hour ICU stay:\n{tab_str}"
        )

    def _build_verb_tisae(self, episode, risk, prob, tab_str):
        concepts = episode.get("tisae_top5_latents", [])
        concepts_str = format_concepts(concepts)
        return (
            f"Patient prediction: {risk} mortality risk (probability: {prob})\n\n"
            f"The model identified these clinical concepts as most important "
            f"for its prediction:\n{concepts_str}\n\n"
            f"Clinical values over the 48-hour ICU stay:\n{tab_str}"
        )
