"""Fixed-dictionary prediction-selector stability (Stage B).

Given a frozen TISAE dictionary, compare prediction selectors trained with
different random seeds (and optionally the original PS).
Because the dictionary is identical across runs, latent indices align directly.
-> are the concepts selected for the same patient stable across independent selector training runs?

Metrics (per patient, averaged over the test set and over all run pairs):
  - top-k concept Jaccard for k in {5,10,64}
  - PS-score Spearman rank correlation over active latents
  - random-selection null for each Jaccard k (interpretation baseline)

Usage:
    python -m tisae.ablation.robustness.selector_stability \
        --sae_dir experiments/discrete/mimic3_mortality/tisae \
        --backbone experiments/discrete/mimic3_mortality/backbone/best_model.pt \
        --dataset_path data/unified_mimic3_48h.pkl \
        --ps original=experiments/discrete/mimic3_mortality/ps \
             42=experiments/ablation/selector_stability/seed_42/ps \
             43=experiments/ablation/selector_stability/seed_43/ps \
             44=experiments/ablation/selector_stability/seed_44/ps \
             45=experiments/ablation/selector_stability/seed_45/ps \
             46=experiments/ablation/selector_stability/seed_46/ps \
        --n_samples 500 \
        --output experiments/ablation/selector_stability/stability_results.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.ablation.robustness.seed_reproducibility import (
    load_sae,
    load_ps,
    collect_seed_data,
)


def topk_sets(scores: np.ndarray, k: int) -> list[set]:
    """Per-patient set of top-k latent indices."""
    idx = np.argsort(-scores, axis=1)[:, :k]
    return [set(row.tolist()) for row in idx]


def mean_topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    sa, sb = topk_sets(a, k), topk_sets(b, k)
    js = [len(x & y) / len(x | y) for x, y in zip(sa, sb) if (x | y)]
    return float(np.mean(js)) if js else 0.0


def mean_spearman(a: np.ndarray, b: np.ndarray, active: np.ndarray) -> float:
    """Mean per-patient Spearman rank corr over active latents (no scipy dep)."""
    def rank(x):
        order = np.argsort(x, axis=1)
        r = np.empty_like(order)
        rows = np.arange(x.shape[0])[:, None]
        r[rows, order] = np.arange(x.shape[1])[None, :]
        return r.astype(float)
    aa, bb = a[:, active], b[:, active]
    ra, rb = rank(aa), rank(bb)
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra**2).sum(1) * (rb**2).sum(1)) + 1e-12
    rho = num / den
    rho = rho[np.isfinite(rho)]
    return float(np.mean(rho)) if len(rho) else 0.0


def random_null_jaccard(K: int, k: int, n_active: int, trials: int = 3000) -> float:
    """Expected Jaccard of two random top-k drawn from the n_active active latents."""
    rng = np.random.default_rng(0)
    pool = min(K, n_active)
    js = []
    for _ in range(trials):
        sa = set(rng.choice(pool, k, replace=False).tolist())
        sb = set(rng.choice(pool, k, replace=False).tolist())
        js.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(js))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", required=True, help="frozen canonical TISAE dir")
    ap.add_argument("--backbone", required=True, help="backbone best_model.pt")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--ps", nargs="+", required=True,
                    help="label=ps_dir entries (e.g. 42=/path/seed_42/ps)")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    runs = {}
    for entry in args.ps:
        label, path = entry.split("=", 1)
        runs[label] = path

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V,
          "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    backbone = load_backbone_adapter(args.backbone, device=str(device), dataset_meta=dm)
    backbone.eval()
    sae, hgm = load_sae(args.sae_dir)
    sae = sae.to(device).eval()
    if hgm is not None:
        hgm = hgm.to(device)
    test_idx = np.array(dataset.splits["test"][:args.n_samples])

    # Collect PS scores [N, K] for each run (same latent basis; no matching).
    scores = {}
    for label, ps_dir in runs.items():
        print(f"[collect] {label} <- {ps_dir}", flush=True)
        ps = load_ps(ps_dir).to(device).eval()
        data = collect_seed_data(backbone, sae, ps, hgm, dataset, test_idx,
                                 device, batch_size=args.batch_size)
        scores[label] = data["ps_scores"]  # [N, K]
        del ps
        torch.cuda.empty_cache()

    K = next(iter(scores.values())).shape[1]
    # active latents = nonzero variance across patients in any run
    active_mask = np.zeros(K, dtype=bool)
    for s in scores.values():
        active_mask |= s.std(axis=0) > 1e-8
    active = np.where(active_mask)[0]
    n_active = int(active.size)

    labels = list(scores.keys())
    seed_labels = [l for l in labels if l != "original"]

    ks = [5, 10, 64]
    nulls = {k: random_null_jaccard(K, k, n_active) for k in ks}

    def pair_metrics(la, lb):
        a, b = scores[la], scores[lb]
        m = {f"jaccard_top{k}": mean_topk_jaccard(a, b, k) for k in ks}
        m["spearman"] = mean_spearman(a, b, active)
        return m

    # Cross-seed pairs (exclude original) — the headline stability number.
    cross_seed_pairs = list(itertools.combinations(seed_labels, 2))
    cs = {f"jaccard_top{k}": [] for k in ks}
    cs["spearman"] = []
    for la, lb in cross_seed_pairs:
        m = pair_metrics(la, lb)
        for key in cs:
            cs[key].append(m[key])
    cross_seed = {key: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                  for key, v in cs.items()}

    # seed-42-rerun vs original (reproduction / within-harness check)
    repro = pair_metrics("42", "original") if "42" in scores and "original" in scores else None

    result = {
        "n_samples": int(len(test_idx)),
        "K": int(K),
        "n_active_latents": n_active,
        "runs": labels,
        "random_null_jaccard": nulls,
        "cross_seed": cross_seed,
        "cross_seed_pairs": [f"{a}-{b}" for a, b in cross_seed_pairs],
        "reproduction_seed42_vs_original": repro,
        "all_pairs": {f"{a}-{b}": pair_metrics(a, b)
                      for a, b in itertools.combinations(labels, 2)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print("\n==== 1c selector stability ====")
    print(f"n_active latents = {n_active} / {K}")
    print(f"random null Jaccard: " + ", ".join(f"top{k}={nulls[k]:.3f}" for k in ks))
    print("CROSS-SEED (mean +/- std over %d pairs):" % len(cross_seed_pairs))
    for k in ks:
        c = cross_seed[f"jaccard_top{k}"]
        print(f"  Jaccard@{k}: {c['mean']:.3f} +/- {c['std']:.3f}  (null {nulls[k]:.3f})")
    print(f"  Spearman:  {cross_seed['spearman']['mean']:.3f} +/- {cross_seed['spearman']['std']:.3f}")
    if repro:
        print("REPRODUCTION seed42-rerun vs original: " +
              ", ".join(f"J@{k}={repro[f'jaccard_top{k}']:.3f}" for k in ks) +
              f", Spearman={repro['spearman']:.3f}")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
