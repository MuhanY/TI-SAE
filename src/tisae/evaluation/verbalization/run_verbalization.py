"""
End-to-end verbalization evaluation: extract → verbalize → judge → aggregate.

Verbalizer: Claude Sonnet (default)
Judge: OpenAI GPT-4o (default, different model family to avoid self-preference)

Usage:
    python -m tisae.evaluation.verbalization.run_verbalization \
        --experiment_dir experiments/<backbone>/<condition>/tisae \
        --ps_dir experiments/<backbone>/<condition>/ps \
        --dataset_path data/unified_48h.pkl \
        --attribution_dir experiments/<backbone>/<condition>/eval/attributions \
        --output_dir ./outputs/verb_eval
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from .extract_episodes import EpisodeExtractor
from .build_prompts import (
    PromptBuilder, VERBALIZER_SYSTEM, JUDGE_SYSTEM, PAIRWISE_JUDGE_SYSTEM,
    PAIRWISE_DIM_RUBRICS, FAITHFULNESS_RUBRIC, DIMENSIONS, METHOD_LABELS,
    format_attribution_cells, format_concepts,
)
from .aggregate_results import aggregate, compute_composite, save_all
from tisae.dictionary.llm_api import build_client, LLMCallMetadata
from tisae.dictionary.llm_labeler import parse_llm_json


def run_verbalization(
    episodes: List[dict],
    methods: List[str],
    output_dir: str,
    verb_provider: str = "claude",
    verb_model: Optional[str] = None,
    judge_provider: str = "openai",
    judge_model: Optional[str] = None,
    temperature: float = 0.0,
    dry_run: bool = False,
):
    """Run full verbalization + judging pipeline.

    Args:
        episodes: from EpisodeExtractor.extract_all()
        methods: list of method names (e.g., ["raw", "attention", "ig", "tsr", "tisae"])
        output_dir: where to save results
        verb_provider/model: verbalizer LLM (default: claude/claude-sonnet-4-6)
        judge_provider/model: judge LLM (default: openai/gpt-4o)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    builder = PromptBuilder()
    verb_log = out / "verb_call_log.jsonl"
    judge_log = out / "judge_call_log.jsonl"

    # ── Phase 1: Build verbalizer prompts ──
    print(f"\n{'='*60}")
    print(f"Phase 1: Verbalization ({len(episodes)} episodes × {len(methods)} methods)")
    print(f"{'='*60}")

    verb_prompts_dir = out / "verb_prompts"
    manifest = builder.build_all_verbalizer_prompts(episodes, methods, str(verb_prompts_dir))
    print(f"Built {len(manifest)} verbalizer prompts")

    # ── Phase 2: Call verbalizer LLM ──
    verb_results_path = out / "verb_results.json"
    if verb_results_path.exists():
        print(f"Loading existing verb results from {verb_results_path}")
        with open(verb_results_path) as f:
            verb_results = json.load(f)
        print(f"  {len(verb_results)} existing results")
    else:
        verb_results = []

    existing_keys = {(r["ep_idx"], r["method"]) for r in verb_results}
    pending = []
    for ep in episodes:
        for method in methods:
            if (ep["ep_idx"], method) not in existing_keys:
                pending.append((ep, method))

    if pending and not dry_run:
        verb_client = build_client(verb_provider, verb_model, temperature=temperature)
        print(f"\nVerbalizer: {verb_client.provider}/{verb_client.model}")
        print(f"Pending: {len(pending)} calls")

        for i, (ep, method) in enumerate(pending):
            key = f"{ep['ep_idx']}_{method}"
            prompt = builder.build_verbalizer_prompt(ep, method)
            try:
                response, meta = verb_client.label_one(prompt, VERBALIZER_SYSTEM, key)
                verb_results.append({
                    "ep_idx": ep["ep_idx"], "method": method,
                    "explanation": response,
                })
                with open(verb_log, "a") as f:
                    f.write(json.dumps(asdict(meta)) + "\n")

                print(f"  [{i+1}/{len(pending)}] {key}: {len(response)} chars")
            except Exception as e:
                print(f"  [{i+1}/{len(pending)}] {key}: FAILED ({e})")

            # Checkpoint every 10
            if (i + 1) % 10 == 0:
                with open(verb_results_path, "w") as f:
                    json.dump(verb_results, f, indent=2)

        with open(verb_results_path, "w") as f:
            json.dump(verb_results, f, indent=2)
        print(f"Saved {len(verb_results)} verb results")
    elif dry_run:
        print(f"DRY RUN: would call verbalizer for {len(pending)} prompts")

    # ── Phase 3: Build judge prompts ──
    print(f"\n{'='*60}")
    print(f"Phase 3: Judging ({len(verb_results)} explanations)")
    print(f"{'='*60}")

    judge_prompts_dir = out / "judge_prompts"
    judge_manifest = builder.build_all_judge_prompts(
        episodes, verb_results, methods, str(judge_prompts_dir)
    )
    print(f"Built {len(judge_manifest)} judge prompts")

    # ── Phase 4: Call judge LLM ──
    judge_results_path = out / "judge_results.json"
    if judge_results_path.exists():
        with open(judge_results_path) as f:
            judge_results = json.load(f)
        print(f"Loading existing judge results: {len(judge_results)}")
    else:
        judge_results = []

    existing_judge_keys = {(r["ep_idx"], r["method"]) for r in judge_results}
    judge_pending = []
    for ep in episodes:
        for method in methods:
            if (ep["ep_idx"], method) not in existing_judge_keys:
                judge_pending.append((ep, method))

    if judge_pending and not dry_run:
        judge_client = build_client(judge_provider, judge_model, temperature=temperature)
        print(f"\nJudge: {judge_client.provider}/{judge_client.model}")
        print(f"Pending: {len(judge_pending)} calls")

        exp_map = {(r["ep_idx"], r["method"]): r["explanation"] for r in verb_results}

        for i, (ep, method) in enumerate(judge_pending):
            key = f"{ep['ep_idx']}_{method}"
            explanation = exp_map.get((ep["ep_idx"], method), "NO EXPLANATION")
            prompt = builder.build_judge_prompt(ep, explanation)

            try:
                response, meta = judge_client.label_one(prompt, JUDGE_SYSTEM, key)
                parsed = parse_llm_json(response)
                result = {
                    "ep_idx": ep["ep_idx"], "method": method,
                    **{d: parsed.get(d) for d in DIMENSIONS},
                    "brief_rationale": parsed.get("brief_rationale", ""),
                }
                judge_results.append(result)
                with open(judge_log, "a") as f:
                    f.write(json.dumps(asdict(meta)) + "\n")

                scores = [parsed.get(d, "?") for d in DIMENSIONS]
                print(f"  [{i+1}/{len(judge_pending)}] {key}: {scores}")
            except Exception as e:
                print(f"  [{i+1}/{len(judge_pending)}] {key}: FAILED ({e})")

            if (i + 1) % 10 == 0:
                with open(judge_results_path, "w") as f:
                    json.dump(judge_results, f, indent=2)

        with open(judge_results_path, "w") as f:
            json.dump(judge_results, f, indent=2)
        print(f"Saved {len(judge_results)} judge results")
    elif dry_run:
        print(f"DRY RUN: would call judge for {len(judge_pending)} prompts")

    # ── Phase 5: Aggregate absolute scores ──
    if judge_results:
        print(f"\n{'='*60}")
        print("Phase 5: Aggregation (absolute)")
        print(f"{'='*60}")

        by_method = {}
        for r in judge_results:
            m = r["method"]
            if m not in by_method:
                by_method[m] = {d: [] for d in DIMENSIONS}
            for d in DIMENSIONS:
                v = r.get(d)
                if v is not None:
                    by_method[m][d].append(float(v))

        summary = compute_composite(by_method)
        save_all(summary, str(out / "aggregated"))

    # ── Phase 6: Pairwise comparison (per-dimension, independent calls) ──
    pairwise_path = out / "pairwise_results.json"
    if pairwise_path.exists():
        with open(pairwise_path) as f:
            pairwise_results = json.load(f)
        print(f"Loading existing pairwise results: {len(pairwise_results)}")
    else:
        pairwise_results = []

    attr_methods_for_pw = [m for m in methods if m not in ("raw", "tisae")]
    verb_map = {(r["ep_idx"], r["method"]): r["explanation"] for r in verb_results}

    existing_pw = {(r["ep_idx"], r["attr_method"]) for r in pairwise_results}
    pw_pending = []
    for ep in episodes:
        for attr_m in attr_methods_for_pw:
            if (ep["ep_idx"], attr_m) not in existing_pw:
                tisae_exp = verb_map.get((ep["ep_idx"], "tisae"))
                attr_exp = verb_map.get((ep["ep_idx"], attr_m))
                if tisae_exp and attr_exp:
                    pw_pending.append((ep, attr_m, attr_exp, tisae_exp))

    PAIRWISE_DIMS = list(PAIRWISE_DIM_RUBRICS.keys())

    if pw_pending and not dry_run:
        import random as _rng
        _rng.seed(42)

        judge_client_pw = build_client(judge_provider, judge_model, temperature=temperature)
        pw_log = out / "pairwise_call_log.jsonl"
        n_calls = len(pw_pending) * (len(PAIRWISE_DIMS) + 1)
        print(f"\n{'='*60}")
        print(f"Phase 6: Pairwise comparison ({len(pw_pending)} pairs × "
              f"{len(PAIRWISE_DIMS)} dims = {n_calls} calls)")
        print(f"{'='*60}")
        print(f"Judge: {judge_client_pw.provider}/{judge_client_pw.model}")

        ALL_DIMS = PAIRWISE_DIMS + ["faithfulness"]

        for i, (ep, attr_m, attr_exp, tisae_exp) in enumerate(pw_pending):
            tisae_is_a = _rng.random() < 0.5
            if tisae_is_a:
                exp_a, exp_b = tisae_exp, attr_exp
                a_method, b_method = "tisae", attr_m
            else:
                exp_a, exp_b = attr_exp, tisae_exp
                a_method, b_method = attr_m, "tisae"

            result = {
                "ep_idx": ep["ep_idx"],
                "attr_method": attr_m,
                "tisae_is_a": tisae_is_a,
            }

            for dim in PAIRWISE_DIMS:
                prompt, system = builder.build_pairwise_single_dim_prompt(
                    ep, exp_a, exp_b, dim)
                key = f"pw_{ep['ep_idx']}_{attr_m}_{dim}"

                try:
                    response, meta = judge_client_pw.label_one(prompt, system, key)
                    parsed = parse_llm_json(response)
                    choice = parsed.get("winner", "Tie")
                    if choice == "A":
                        result[dim] = a_method
                    elif choice == "B":
                        result[dim] = b_method
                    else:
                        result[dim] = "Tie"
                    result[f"{dim}_rationale"] = parsed.get("rationale", "")

                    with open(pw_log, "a") as f:
                        f.write(json.dumps(asdict(meta)) + "\n")
                except Exception as e:
                    result[dim] = "Tie"
                    result[f"{dim}_rationale"] = f"FAILED: {e}"

            # Faithfulness: unblinded — include source analyses
            tisae_analysis = format_concepts(ep.get("tisae_top5_latents", []))
            attr_cells = ep.get("attribution_cells", {}).get(attr_m, [])
            attr_analysis = format_attribution_cells(attr_cells)
            if tisae_is_a:
                analysis_a, analysis_b = tisae_analysis, attr_analysis
            else:
                analysis_a, analysis_b = attr_analysis, tisae_analysis

            key = f"pw_{ep['ep_idx']}_{attr_m}_faithfulness"
            try:
                prompt, system = builder.build_pairwise_faithfulness_prompt(
                    ep, exp_a, analysis_a, exp_b, analysis_b)
                response, meta = judge_client_pw.label_one(prompt, system, key)
                parsed = parse_llm_json(response)
                choice = parsed.get("winner", "Tie")
                if choice == "A":
                    result["faithfulness"] = a_method
                elif choice == "B":
                    result["faithfulness"] = b_method
                else:
                    result["faithfulness"] = "Tie"
                result["faithfulness_rationale"] = parsed.get("rationale", "")
                with open(pw_log, "a") as f:
                    f.write(json.dumps(asdict(meta)) + "\n")
            except Exception as e:
                result["faithfulness"] = "Tie"
                result["faithfulness_rationale"] = f"FAILED: {e}"

            pairwise_results.append(result)
            wins = {d: result[d] for d in ALL_DIMS}
            print(f"  [{i+1}/{len(pw_pending)}] ep={ep['ep_idx']} {attr_m}: {wins}")

            if (i + 1) % 10 == 0:
                with open(pairwise_path, "w") as f:
                    json.dump(pairwise_results, f, indent=2)

        with open(pairwise_path, "w") as f:
            json.dump(pairwise_results, f, indent=2)
        print(f"Saved {len(pairwise_results)} pairwise results")

    # ── Phase 7: Aggregate pairwise ���─
    if pairwise_results:
        print(f"\n{'='*60}")
        print("Phase 7: Pairwise aggregation")
        print(f"{'='*60}")

        agg_dims = list(PAIRWISE_DIM_RUBRICS.keys()) + ["faithfulness"]
        for attr_m in attr_methods_for_pw:
            pairs = [r for r in pairwise_results if r["attr_method"] == attr_m]
            if not pairs:
                continue
            print(f"\n  TI-SAE vs {METHOD_LABELS.get(attr_m, attr_m)} (N={len(pairs)}):")
            for d in agg_dims:
                tisae_wins = sum(1 for r in pairs if r.get(d) == "tisae")
                attr_wins = sum(1 for r in pairs if r.get(d) == attr_m)
                ties = sum(1 for r in pairs if r.get(d) == "Tie")
                print(f"    {d:>15}: TI-SAE {tisae_wins} | {METHOD_LABELS.get(attr_m, attr_m)} {attr_wins} | Tie {ties}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end verbalization evaluation")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--ps_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--attribution_dir", default=None)
    parser.add_argument("--dictionary_path", default=None)
    parser.add_argument("--output_dir", default="./outputs/verb_eval")
    parser.add_argument("--methods", nargs="+",
                        default=["attention", "ig", "tsr", "tisae"])
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--top_n_concepts", type=int, default=5)
    parser.add_argument("--verb_provider", default="claude")
    parser.add_argument("--verb_model", default=None)
    parser.add_argument("--judge_provider", default="openai")
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    from tisae.utils.seeding import set_seed

    args = parse_args()
    set_seed(args.seed)

    # Extract episodes
    extractor = EpisodeExtractor(
        experiment_dir=args.experiment_dir, ps_dir=args.ps_dir,
        dataset_path=args.dataset_path, device=args.device,
        attribution_dir=args.attribution_dir,
        dictionary_path=args.dictionary_path,
        top_n_concepts=args.top_n_concepts,
    )
    extractor.load_models()

    attr_methods = [m for m in args.methods if m not in ("raw", "tisae")]
    attributions = extractor.load_attributions(attr_methods)

    # Restrict episode selection to indices with saved attributions
    import numpy as np
    restrict_to = None
    if attributions:
        common_idx = None
        for m, lookup in attributions.items():
            idx_set = set(lookup.keys())
            common_idx = idx_set if common_idx is None else common_idx & idx_set
        if common_idx:
            restrict_to = np.array(sorted(common_idx))
            print(f"Restricting to {len(restrict_to)} episodes with all attributions")

    indices = extractor.select_episodes(args.n_episodes, args.seed, restrict_to=restrict_to)
    episodes = extractor.extract_all(indices, attr_methods, attributions)

    ep_path = Path(args.output_dir) / "episodes_data.json"
    extractor.save(episodes, str(ep_path))

    # Run verbalization + judging
    run_verbalization(
        episodes, args.methods, args.output_dir,
        verb_provider=args.verb_provider, verb_model=args.verb_model,
        judge_provider=args.judge_provider, judge_model=args.judge_model,
        temperature=args.temperature, dry_run=args.dry_run,
    )
