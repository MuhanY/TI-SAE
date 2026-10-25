"""
Aggregate LLM-as-a-Judge results into summary tables.

Reads judge result JSON files, groups by method, computes mean +/- std,
and outputs LaTeX table + summary.

Usage:
    python -m tisae.evaluation.verbalization.aggregate_results \
        --results judge_results.json --output_dir ./outputs/verb_output
"""

from __future__ import annotations

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .build_prompts import DIMENSIONS, METHOD_LABELS


def aggregate(result_files: List[str]) -> Dict[str, Dict[str, list]]:
    """Load and group judge results by method."""
    all_results = []
    for f in result_files:
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict):
                all_results.append(data)

    by_method = defaultdict(lambda: {d: [] for d in DIMENSIONS})
    for r in all_results:
        method = r.get("method", "unknown")
        for dim in DIMENSIONS:
            val = r.get(dim)
            if val is not None:
                by_method[method][dim].append(float(val))

    return dict(by_method)


def compute_composite(by_method: dict) -> Dict[str, dict]:
    """Compute per-method summary: mean, std, composite."""
    summary = {}
    for method, scores in by_method.items():
        dims = {}
        all_means = []
        for dim in DIMENSIONS:
            vals = scores.get(dim, [])
            if vals:
                dims[dim] = {"mean": np.mean(vals), "std": np.std(vals), "n": len(vals)}
                all_means.append(np.mean(vals))
            else:
                dims[dim] = {"mean": 0, "std": 0, "n": 0}
        composite = np.mean(all_means) if all_means else 0
        summary[method] = {"dimensions": dims, "composite": round(float(composite), 2)}
    return summary


def format_table(summary: dict) -> str:
    """Pretty-print results table."""
    header = f"{'Method':<25}" + "".join(f"{d[:8]:>10}" for d in DIMENSIONS) + f"{'Composite':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for method in sorted(summary.keys()):
        s = summary[method]
        row = f"{METHOD_LABELS.get(method, method):<25}"
        for dim in DIMENSIONS:
            m = s["dimensions"][dim]["mean"]
            row += f"{m:>10.1f}"
        row += f"{s['composite']:>12.1f}"
        lines.append(row)
    return "\n".join(lines)


def format_latex(summary: dict) -> str:
    """LaTeX table matching paper format. Best per column in bold."""
    # Find best per dimension
    best = {}
    for dim in DIMENSIONS + ["composite"]:
        if dim == "composite":
            vals = {m: s["composite"] for m, s in summary.items()}
        else:
            vals = {m: s["dimensions"][dim]["mean"] for m, s in summary.items()}
        best[dim] = max(vals, key=vals.get) if vals else None

    lines = [
        r"\begin{tabular}{l " + "c" * len(DIMENSIONS) + " c}",
        r"\toprule",
        "Condition & " + " & ".join(d.capitalize() for d in DIMENSIONS) + r" & Composite \\",
        r"\midrule",
    ]

    method_order = ["raw", "attention", "ig", "tsr", "tisae"]
    for method in method_order:
        if method not in summary:
            continue
        s = summary[method]
        label = METHOD_LABELS.get(method, method)
        parts = [label]
        for dim in DIMENSIONS:
            val = s["dimensions"][dim]["mean"]
            val_str = f"{val:.1f}"
            if best.get(dim) == method:
                val_str = r"\textbf{" + val_str + "}"
            parts.append(val_str)
        comp = s["composite"]
        comp_str = f"{comp:.1f}"
        if best.get("composite") == method:
            comp_str = r"\textbf{" + comp_str + "}"
        parts.append(comp_str)
        lines.append(" & ".join(parts) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def save_all(summary: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "aggregated.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    with open(out / "summary.txt", "w") as f:
        f.write(format_table(summary))

    with open(out / "latex_table.tex", "w") as f:
        f.write(format_latex(summary))

    print(f"Saved aggregated results to {out}/")
    print(format_table(summary))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output_dir", default="./outputs/verb_output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    by_method = aggregate(args.results)
    summary = compute_composite(by_method)
    save_all(summary, args.output_dir)
