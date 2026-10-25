#!/usr/bin/env python3
"""
Island identity stability: Jaccard similarity of selected island (k,m)
identities under input event dropout.

Usage:
    python -m tisae.evaluation.dictionary.stability \
        --experiment_dir experiments/discrete/mimic3_mortality/tisae
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from tisae.utils.seeding import set_seed
from tisae.evaluation.utils import load_evaluation_models, run_tisae_forward


# =============================================================================
# Core metric
# =============================================================================

def _drop_random_events(
    batch: Dict[str, torch.Tensor],
    drop_frac: float,
) -> Dict[str, torch.Tensor]:
    """Create a copy of batch with drop_frac of valid events randomly zeroed."""
    new_batch = {k: v.clone() for k, v in batch.items()}
    obs_mask = new_batch["obs_mask"]
    B, L = obs_mask.shape

    for b in range(B):
        valid_idx = (obs_mask[b] > 0).nonzero(as_tuple=True)[0]
        n_drop = max(1, int(len(valid_idx) * drop_frac))
        perm = torch.randperm(len(valid_idx))[:n_drop]
        drop_idx = valid_idx[perm]
        obs_mask[b, drop_idx] = 0.0

    new_batch["obs_mask"] = obs_mask
    return new_batch


@torch.no_grad()
def island_identity_jaccard(
    sae_forward_fn,
    batch: Dict[str, torch.Tensor],
    n_perturbations: int = 10,
    drop_frac: float = 0.05,
) -> float:
    """Island identity stability: Jaccard similarity of selected island (k,m)
    identities under input event dropout.

    For each perturbation trial:
      1. Run SAE on original input -> sel[B, K_eff, C_eff, M] -> set of selected (k,m) pairs
      2. Run SAE on perturbed input (5% events dropped) -> set of selected (k,m) pairs'
      3. Jaccard(set, set') per sample

    Args:
        sae_forward_fn: callable(batch) -> sel[B, K_eff, C_eff, M] binary selection tensor
        batch: input batch dict
        n_perturbations: number of perturbation trials
        drop_frac: fraction of events to randomly drop

    Returns:
        Mean Jaccard similarity (higher = more stable). Range [0, 1].
    """
    base_sel = sae_forward_fn(batch)  # [B, K_eff, C_eff, M]
    base_binary = (base_sel > 0.5)
    B = base_sel.shape[0]

    jaccards = []
    for _ in range(n_perturbations):
        perturbed = _drop_random_events(batch, drop_frac)
        pert_sel = sae_forward_fn(perturbed)  # [B, K_eff, C_eff, M]
        pert_binary = (pert_sel > 0.5)

        for b in range(B):
            base_b = base_binary[b].flatten()
            pert_b = pert_binary[b].flatten()
            intersection = (base_b & pert_b).sum().item()
            union = (base_b | pert_b).sum().item()
            if union > 0:
                jaccards.append(intersection / union)

    return float(np.mean(jaccards)) if jaccards else 0.0


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Island identity stability (Jaccard under event dropout)")
    p.add_argument("--experiment_dir", type=str, required=True)
    p.add_argument("--dataset_path", type=str,
                   default="data/unified_mimic3_48h.pkl")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_perturbations", type=int, default=10)
    p.add_argument("--drop_frac", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading models from {args.experiment_dir}...")
    models = load_evaluation_models(
        args.experiment_dir, args.dataset_path, device=args.device)
    backbone = models["backbone"]
    sae = models["sae"]
    dataset = models["dataset"]
    H_grid_mean = models["H_grid_mean"]
    T, V = models["T"], models["V"]

    print(f"Architecture: K_eff={models['K_eff']}, C_eff={models['C_eff']}, "
          f"M={models['M']}, n_global={sae.cfg.n_global}")

    def sae_forward_fn(batch):
        return run_tisae_forward(backbone, sae, batch, H_grid_mean, T, V)["sel"]

    test_indices = np.array(dataset.splits["test"])
    n_use = min(len(test_indices), args.n_samples)
    test_indices = test_indices[:n_use]

    all_jaccards = []
    for start in range(0, n_use, args.batch_size):
        end = min(start + args.batch_size, n_use)
        idx = test_indices[start:end]
        batch = dataset.get_batch(idx)
        batch = {k: v.to(device) for k, v in batch.items()}

        j = island_identity_jaccard(
            sae_forward_fn, batch,
            n_perturbations=args.n_perturbations,
            drop_frac=args.drop_frac,
        )
        all_jaccards.append(j)

        done = min(end, n_use)
        if (start // args.batch_size) % 10 == 0:
            print(f"  {done}/{n_use} samples...", flush=True)

    mean_j = float(np.mean(all_jaccards))

    # Random baseline: select n_global random islands from K_eff*C_eff*M slots
    n_slots = models['K_eff'] * models['C_eff'] * models['M']
    n_global = sae.cfg.n_global
    rand_jaccards = []
    for _ in range(1000):
        a = set(np.random.choice(n_slots, n_global, replace=False))
        b = set(np.random.choice(n_slots, n_global, replace=False))
        inter = len(a & b)
        union = len(a | b)
        rand_jaccards.append(inter / union if union > 0 else 0.0)
    rand_j = float(np.mean(rand_jaccards))

    results = dict(
        n_samples=n_use,
        n_perturbations=args.n_perturbations,
        drop_frac=args.drop_frac,
        sae_jaccard=mean_j,
        random_jaccard=rand_j,
    )

    print(f"\nIsland Identity Stability (Jaccard)")
    print(f"  n_samples={n_use}, n_perturbations={args.n_perturbations}, "
          f"drop_frac={args.drop_frac}")
    print(f"  SAE (n_global={sae.cfg.n_global}):  {mean_j:.3f}")
    print(f"  Random baseline:  {rand_j:.3f}")

    out_path = Path(args.experiment_dir) / "stability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
