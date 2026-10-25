"""
Plausibility evaluation: clinical prior agreement with saved attribution maps.

Evaluates each method against SOFA-derived clinical prior (6 organ systems).

Requires attributions saved by run_evaluation --save_attributions.

Usage:
    python -m tisae.evaluation.plausibility.run_plausibility \
        --eval_dir experiments/discrete/mimic3_mortality/eval \
        --dataset_path data/unified_mimic3_48h.pkl
"""

from __future__ import annotations

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

from tisae.evaluation.plausibility.clinical_prior import (
    compute_clinical_prior,
    compute_plausibility_metrics,
)
from tisae.evaluation.utils import build_obs_grid, build_value_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Plausibility evaluation")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Directory with attributions/ from run_evaluation --save_attributions")
    parser.add_argument("--dataset_path", type=str,
                        default="data/unified_mimic3_48h.pkl")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to evaluate (default: all found in attributions/)")
    parser.add_argument("--modes", nargs="+", default=["sofa"],
                        choices=["sofa"],
                        help="Prior modes to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--topk_fractions", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--deterioration_window", type=int, default=6,
                        help="Max time bins to look back for deterioration")
    return parser.parse_args()


def load_saved_attributions(eval_dir: Path, methods=None):
    """Load saved .pt attribution files from eval_dir/attributions/."""
    attr_dir = eval_dir / "attributions"
    if not attr_dir.exists():
        raise FileNotFoundError(f"No attributions/ directory in {eval_dir}")

    results = {}
    for pt_file in sorted(attr_dir.glob("*.pt")):
        method_name = pt_file.stem
        if methods and method_name not in methods:
            continue
        if pt_file.stat().st_size == 0:
            print(f"  SKIP {method_name}: 0-byte file (corrupted write)")
            continue
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError) as e:
            print(f"  SKIP {method_name}: {type(e).__name__}: {e}")
            continue
        results[method_name] = data  # keys: attr, indices, obs_grid
    return results


def format_results_table(all_results: Dict[str, Dict[str, float]]) -> str:
    """Format results as markdown table."""
    if not all_results:
        return "No results."

    keys = list(all_results.keys())
    all_metrics = set()
    for v in all_results.values():
        all_metrics.update(v.keys())
    metrics = sorted(all_metrics)

    header = "| Method | " + " | ".join(metrics) + " |"
    sep = "|---" + "|---" * len(metrics) + "|"
    rows = [header, sep]

    for key in keys:
        vals = all_results[key]
        row = f"| {key} | " + " | ".join(
            f"{vals.get(m, float('nan')):.4f}" for m in metrics
        ) + " |"
        rows.append(row)

    return "\n".join(rows)


def main():
    args = parse_args()

    eval_dir = Path(args.eval_dir)
    saved = load_saved_attributions(eval_dir, args.methods)
    if not saved:
        print("No attribution files found in", eval_dir / "attributions")
        return

    # Load dataset for value grids
    from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    T, V = dataset.T, dataset.V
    var_names = dataset.var_names

    print(f"Dataset: T={T}, V={V}, {len(var_names)} variables")
    print(f"Methods: {list(saved.keys())}")

    # Precompute value grids (shared across methods — same sample indices)
    first_data = list(saved.values())[0]
    indices = first_data["indices"]
    obs_all = first_data["obs_grid"]
    N = indices.shape[0]

    val_grids = []
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        idx = indices[start:end]
        batch = dataset.get_batch(idx)
        batch = {k: v.to("cpu") for k, v in batch.items()}
        val_grids.append(build_value_grid(batch, T, V).cpu())
    val_all = torch.cat(val_grids, dim=0)

    # Precompute priors for each mode
    priors = {}
    for mode in args.modes:
        print(f"\nComputing {mode} prior...")
        mode_priors = []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            mode_priors.append(compute_clinical_prior(
                val_all[start:end], obs_all[start:end], var_names,
                mode=mode, deterioration_window=args.deterioration_window,
            ))
        priors[mode] = torch.cat(mode_priors, dim=0)

    # Evaluate each method against each prior mode
    all_results = {}

    for method_name, data in saved.items():
        attr_all = data["attr"]
        if attr_all.isnan().any():
            n_nan = attr_all.isnan().sum().item()
            attr_all = attr_all.nan_to_num(0.0)
            print(f"  NOTE: {n_nan} NaN in {method_name} attributions (empty-mask edge case), set to 0")

        for mode in args.modes:
            prior_all = priors[mode]
            metrics_accum = {}

            for start in range(0, N, args.batch_size):
                end = min(start + args.batch_size, N)
                m = compute_plausibility_metrics(
                    attr_all[start:end],
                    prior_all[start:end],
                    obs_all[start:end],
                    topk_fractions=tuple(args.topk_fractions),
                )
                for k, v in m.items():
                    metrics_accum.setdefault(k, []).append(v)

            avg = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
            key = f"{method_name}/{mode}"
            all_results[key] = avg

    # Print summary
    print("\n" + "=" * 80)
    print("PLAUSIBILITY RESULTS")
    print("=" * 80)
    print(format_results_table(all_results))

    # Save
    results_path = eval_dir / "plausibility_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_path}")

    table_path = eval_dir / "plausibility_table.md"
    with open(table_path, "w") as f:
        f.write("# Plausibility Evaluation (Clinical Prior)\n\n")
        f.write(format_results_table(all_results))
    print(f"Table saved to {table_path}")


if __name__ == "__main__":
    main()
