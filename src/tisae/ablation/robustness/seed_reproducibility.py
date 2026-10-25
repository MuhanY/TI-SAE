"""Concept seed reproducibility:
measure whether TI-SAE learns the same concepts across different random seeds.

Trains additional TISAE + PredSelector models with different seeds,
then compares the learned dictionaries via two-stage latent matching.

Usage:
    # Step 1: Train seed variants (skip if already trained)
    python -m tisae.ablation.seed_reproducibility train \
        --reference_dir experiments/<backbone>/<condition>/tisae \
        --ps_reference_dir experiments/<backbone>/<condition>/ps \
        --backbone_checkpoint experiments/<backbone>/<condition>/backbone/best_model.pt \
        --dataset_path data/unified_48h.pkl \
        --output_dir experiments_ablation/seed_stability \
        --seeds 43 44

    # Step 2: Compare all seed pairs
    python -m tisae.ablation.seed_reproducibility compare \
        --reference_dir experiments/<backbone>/<condition>/tisae \
        --ps_reference_dir experiments/<backbone>/<condition>/ps \
        --backbone_checkpoint experiments/<backbone>/<condition>/backbone/best_model.pt \
        --dataset_path data/unified_48h.pkl \
        --output_dir experiments_ablation/seed_stability \
        --seeds 43 44 \
        --cos_threshold 0.7 \
        --n_samples 500
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.tisae import TISAE, TISAEConfig
from tisae.models.pred_selector import (
    PredSelector,
    PredSelectorConfig,
    assemble_island_features,
)


# ============================================================
# Model loading
# ============================================================

def load_sae(tisae_dir: str | Path) -> Tuple[TISAE, Optional[torch.Tensor]]:
    tisae_dir = Path(tisae_dir)
    with open(tisae_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    tisae_cfg_dict = config.get("tisae_config", config.get("sae_config", {}))
    valid = {f.name for f in fields(TISAEConfig)}
    tisae_cfg = TISAEConfig(**{k: v for k, v in tisae_cfg_dict.items() if k in valid})
    sae = TISAE(tisae_cfg)
    ckpt = torch.load(tisae_dir / "best_model.pt", map_location="cpu", weights_only=False)
    sae.load_state_dict(ckpt["sae_state_dict"], strict=False)
    H_grid_mean = ckpt.get("H_grid_mean")
    return sae, H_grid_mean


def load_ps(ps_dir: str | Path) -> PredSelector:
    ps_dir = Path(ps_dir)
    with open(ps_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    ps_cfg_dict = config.get("pred_selector_config", {})
    valid = {f.name for f in fields(PredSelectorConfig)}
    ps_cfg = PredSelectorConfig(**{k: v for k, v in ps_cfg_dict.items() if k in valid})
    ps = PredSelector(
        ps_cfg,
        n_island_features=config.get("n_island_features", 276),
        D=config.get("D", 128),
        C_eff=config.get("C_eff", 1),
    )
    ckpt = torch.load(ps_dir / "best_model.pt", map_location="cpu", weights_only=False)
    ps.load_state_dict(ckpt["selector_state_dict"])
    return ps


def get_reference_seed(tisae_dir: str | Path) -> int:
    with open(Path(tisae_dir) / "config.yaml") as f:
        config = yaml.safe_load(f)
    return config.get("args", {}).get("seed", 42)


def get_tisae_args(tisae_dir: str | Path) -> Dict:
    with open(Path(tisae_dir) / "config.yaml") as f:
        return yaml.safe_load(f).get("args", {})


def get_ps_args(ps_dir: str | Path) -> Dict:
    with open(Path(ps_dir) / "config.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# Data collection
# ============================================================

@torch.no_grad()
def collect_seed_data(
    backbone,
    sae: TISAE,
    ps: PredSelector,
    H_grid_mean: Optional[torch.Tensor],
    dataset: UnifiedTimeSeriesDataset,
    test_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Collect traces, energy, and PS scores for one seed model."""
    N = len(test_indices)
    traces_list, energy_list, ps_scores_list = [], [], []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        samples = [dataset.get_item(int(i)) for i in test_indices[start:end]]
        batch = collate_fn(samples)
        batch = {k: v.to(device) for k, v in batch.items()}

        H_grid, obs_grid = backbone.extract_hidden(batch)
        if H_grid_mean is not None:
            H_grid = (H_grid - H_grid_mean.unsqueeze(0)) * obs_grid.unsqueeze(-1)

        out = sae(H_grid, obs_grid)
        traces_list.append(out["A_hat"].cpu())

        A0 = sae.encoder(H_grid, obs_grid)
        A0_4d = A0.unsqueeze(-1)
        mu, sigma, gate = sae.island_params(A0_4d)
        g = sae.gaussian_mask.build_gaussians(mu, sigma.detach())
        A0_perm = A0_4d.permute(0, 1, 3, 2)
        e = (A0_perm.unsqueeze(3) * g).sum(-1).sum(dim=(2, 3))
        energy_list.append(e.cpu())

        W_dec = sae.decoder.get_W().detach()
        features = assemble_island_features(
            A0_4d, g, mu, sigma, gate, obs_grid, W_dec,
            None, sae.K_eff, sae.C_eff, include_obs_profile=True,
        )
        obs_mask = batch["obs_mask"]
        if backbone.extract_layer != -1:
            h_pool = backbone.forward(batch, return_pooled=True)["pooled"]
        else:
            H_ev = backbone.extract_hidden(batch, return_event_h=True)[2]
            h_pool, _ = backbone.pool_events(H_ev, obs_mask)
        s_pred = ps(features, h_pool).squeeze(2).max(dim=-1).values
        ps_scores_list.append(s_pred.cpu())

        if (start // batch_size) % 20 == 0:
            print(f"  {min(end, N)}/{N}", flush=True)

    W = sae.get_W_dec().detach().cpu().numpy()
    return {
        "traces": torch.cat(traces_list, dim=0).numpy(),
        "energy": torch.cat(energy_list, dim=0).numpy(),
        "ps_scores": torch.cat(ps_scores_list, dim=0).numpy(),
        "W_raw": W,
        "W_normed": W / np.linalg.norm(W, axis=1, keepdims=True).clip(1e-8),
    }


# ============================================================
# Matching
# ============================================================

def two_stage_match(
    data_a: Dict[str, np.ndarray],
    data_b: Dict[str, np.ndarray],
    cos_threshold: float = 0.7,
) -> Tuple[np.ndarray, List[int]]:
    """Two-stage matching: Hungarian on decoder cosine,
    refined by temporal trace correlation among structurally similar candidates.

    Returns:
        match_idx: array where match_idx[i] maps latent i in A to latent in B
        n_cands_per_latent: number of structural candidates per latent
    """
    Wa = data_a["W_normed"]
    Wb = data_b["W_normed"]
    cos_mat = Wa @ Wb.T
    K = cos_mat.shape[0]
    n = min(cos_mat.shape)

    ri, ci_h = linear_sum_assignment(-cos_mat[:n, :n])
    ci = np.copy(ci_h)
    n_cands_list = []

    for idx in range(n):
        cands = np.where(cos_mat[idx] > cos_threshold)[0]
        n_cands_list.append(len(cands))
        if len(cands) <= 1:
            continue
        ta = data_a["traces"][:, idx, :].flatten()
        if ta.std() < 1e-8:
            continue
        best_corr, best_c = -2.0, ci_h[idx]
        for c in cands:
            tb = data_b["traces"][:, c, :].flatten()
            if tb.std() > 1e-8:
                corr = float(np.corrcoef(ta, tb)[0, 1])
                if corr > best_corr:
                    best_corr, best_c = corr, c
        ci[idx] = best_c

    return ci, n_cands_list


def random_two_stage_match(
    data_a: Dict[str, np.ndarray],
    data_b: Dict[str, np.ndarray],
    n_cands_list: List[int],
) -> np.ndarray:
    """Random baseline: same per-latent candidate count, random candidates,
    best temporal match among them."""
    K = data_a["traces"].shape[1]
    Kb = data_b["traces"].shape[1]
    n = min(K, Kb, len(n_cands_list))
    match = np.zeros(n, dtype=int)

    for idx in range(n):
        n_c = max(1, n_cands_list[idx])
        n_c = min(n_c, Kb)
        candidates = np.random.choice(Kb, n_c, replace=False)
        ta = data_a["traces"][:, idx, :].flatten()
        if ta.std() < 1e-8:
            match[idx] = candidates[0]
            continue
        best_corr, best_c = -2.0, candidates[0]
        for c in candidates:
            tb = data_b["traces"][:, c, :].flatten()
            if tb.std() > 1e-8:
                corr = float(np.corrcoef(ta, tb)[0, 1])
                if corr > best_corr:
                    best_corr, best_c = corr, c
        match[idx] = best_c

    return match


# ============================================================
# Metrics
# ============================================================

def _binary_ndcg(ranking: np.ndarray, relevant_set: set, k_eval: int) -> float:
    dcg = sum(
        1.0 / np.log2(r + 2)
        for r in range(min(k_eval, len(ranking)))
        if ranking[r] in relevant_set
    )
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant_set), k_eval)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    data_a: Dict[str, np.ndarray],
    data_b: Dict[str, np.ndarray],
    match_idx: np.ndarray,
    n_samples: int,
    trace_subsample: int = 5,
) -> Dict[str, float]:
    """Compute all five reproducibility metrics on matched pairs."""
    n = len(match_idx)

    # 1. Decoder cosine
    cos_vals = [
        float(np.dot(data_a["W_normed"][i], data_b["W_normed"][match_idx[i]]))
        for i in range(n)
    ]

    # 2. Variable overlap >= 2/3
    t3_vals = []
    for i in range(n):
        va = set(np.abs(data_a["W_raw"][i]).argsort()[-3:])
        vb = set(np.abs(data_b["W_raw"][match_idx[i]]).argsort()[-3:])
        t3_vals.append(1 if len(va & vb) >= 2 else 0)

    # 3. Latent energy correlation
    ec_vals = []
    for i in range(n):
        ea = data_a["energy"][:, i]
        eb = data_b["energy"][:, match_idx[i]]
        if ea.std() > 1e-8 and eb.std() > 1e-8:
            ec_vals.append(float(np.corrcoef(ea, eb)[0, 1]))

    # 4. Per-sample temporal trace cosine
    psc_vals = []
    for i in range(n):
        cosines = []
        for ns in range(0, n_samples, trace_subsample):
            ta = data_a["traces"][ns, i, :]
            tb = data_b["traces"][ns, match_idx[i], :]
            na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
            if na > 1e-8 and nb > 1e-8:
                cosines.append(float(np.dot(ta, tb) / (na * nb)))
        if cosines:
            psc_vals.append(np.mean(cosines))

    # 5. Prediction selector NDCG@64 (binary)
    ndcg_vals = []
    for ns in range(n_samples):
        scores_a = data_a["ps_scores"][ns, :n]
        scores_b_m = np.array([data_b["ps_scores"][ns, match_idx[i]] for i in range(n)])
        relevant = set(scores_a.argsort()[-64:])
        ndcg_vals.append(_binary_ndcg(scores_b_m.argsort()[::-1], relevant, 64))

    return {
        "decoder_cosine": float(np.mean(cos_vals)),
        "variable_overlap": float(np.mean(t3_vals)),
        "energy_correlation": float(np.mean(ec_vals)) if ec_vals else 0.0,
        "temporal_trace_cosine": float(np.mean(psc_vals)) if psc_vals else 0.0,
        "prediction_selector_ndcg": float(np.mean(ndcg_vals)),
        "n_matched": n,
    }


# ============================================================
# Train command
# ============================================================

def cmd_train(args):
    ref_args = get_tisae_args(args.reference_dir)
    ref_seed = ref_args.get("seed", 42)
    print(f"Reference seed: {ref_seed}")

    ps_config = get_ps_args(args.ps_reference_dir)
    ps_cfg = ps_config.get("pred_selector_config", {})

    out = Path(args.output_dir)

    for seed in args.seeds:
        tisae_dir = out / f"seed_{seed}" / "tisae"
        ps_dir = out / f"seed_{seed}" / "ps"

        # Train TISAE
        if (tisae_dir / "best_model.pt").exists():
            print(f"SKIP seed {seed} TISAE: already trained")
        else:
            print(f"Training TISAE seed {seed}...")
            tisae_cmd = [
                sys.executable, "-m", "tisae.training.train_tisae",
                "--backbone_checkpoint", ref_args["backbone_checkpoint"],
                "--dataset_path", args.dataset_path,
                "--output_dir", str(tisae_dir),
                "--seed", str(seed),
            ]
            copy_keys = [
                "K", "M", "n_global", "W_sparsity_k", "encoder_sparsity_k",
                "encoder_chunk_size", "island_hidden", "island_kernel_size",
                "score_mode", "lambda_recon", "lambda_pred", "lambda_auxk",
                "auxk_k", "lambda_ch_purity", "lambda_div_W",
                "lambda_mask_coverage", "lambda_mu_repulsion",
                "dead_boost_scale", "dead_threshold_batches",
                "gumbel_topk_noise", "max_sigma", "min_sigma", "n_sigma_iter",
                "batch_size", "lr", "weight_decay", "epochs", "patience",
                "diag_every", "extract_layer",
            ]
            for k in copy_keys:
                if k in ref_args:
                    tisae_cmd.extend([f"--{k}", str(ref_args[k])])
            if ref_args.get("normalize_atoms"):
                tisae_cmd.append("--normalize_atoms")
            subprocess.run(tisae_cmd, check=True)

        # Train PredSelector
        if (ps_dir / "best_model.pt").exists():
            print(f"SKIP seed {seed} PS: already trained")
        else:
            print(f"Training PredSelector seed {seed}...")
            ps_cmd = [
                sys.executable, "-m", "tisae.training.train_pred_selector",
                "--sae_experiment_dir", str(tisae_dir),
                "--dataset_path", args.dataset_path,
                "--output_dir", str(ps_dir),
                "--seed", str(seed),
            ]
            ps_copy_keys = [
                "batch_size", "lr", "weight_decay", "epochs", "patience",
                "d_sel", "mlp_hidden", "n_global_pred",
                "lambda_logit", "lambda_comp", "delta_comp",
            ]
            for k in ps_copy_keys:
                if k in ps_cfg:
                    ps_cmd.extend([f"--{k}", str(ps_cfg[k])])
            if ps_cfg.get("soft_selection", True):
                ps_cmd.append("--soft_selection")
            if "soft_tau" in ps_cfg:
                ps_cmd.extend(["--soft_tau", str(ps_cfg["soft_tau"])])
            subprocess.run(ps_cmd, check=True)

    print("Training complete.")


# ============================================================
# Compare command
# ============================================================

def cmd_compare(args):
    device = torch.device(args.device)
    out = Path(args.output_dir)

    ref_seed = get_reference_seed(args.reference_dir)
    all_seeds = [ref_seed] + args.seeds
    print(f"Seeds: {all_seeds}")

    # Seed directories
    seed_dirs = {ref_seed: (args.reference_dir, args.ps_reference_dir)}
    for s in args.seeds:
        seed_dirs[s] = (str(out / f"seed_{s}" / "tisae"), str(out / f"seed_{s}" / "ps"))

    # Load dataset and backbone
    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V,
          "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    backbone = load_backbone_adapter(args.backbone_checkpoint, device=str(device), dataset_meta=dm)
    backbone.eval()
    test_indices = np.array(dataset.splits["test"][:args.n_samples])

    # Load models and collect data
    seed_data: Dict[int, Dict[str, np.ndarray]] = {}
    for seed, (tisae_dir, ps_dir) in seed_dirs.items():
        print(f"\nLoading seed {seed}...")
        sae, hgm = load_sae(tisae_dir)
        sae = sae.to(device).eval()
        if hgm is not None:
            hgm = hgm.to(device)
        ps = load_ps(ps_dir)
        ps = ps.to(device).eval()

        print(f"Collecting data for seed {seed}...")
        seed_data[seed] = collect_seed_data(
            backbone, sae, ps, hgm, dataset, test_indices, device,
            batch_size=args.batch_size,
        )
        del sae, ps
        torch.cuda.empty_cache()

    # Compare all pairs
    print(f"\n{'='*60}")
    print(f"Comparing (cos_threshold={args.cos_threshold})")
    print(f"{'='*60}")

    actual_results = []
    all_n_cands = []
    pairs = []
    for i in range(len(all_seeds)):
        for j in range(i + 1, len(all_seeds)):
            sa, sb = all_seeds[i], all_seeds[j]
            pairs.append((sa, sb))
            match, n_cands = two_stage_match(
                seed_data[sa], seed_data[sb], cos_threshold=args.cos_threshold)
            all_n_cands.append(n_cands)
            r = compute_metrics(seed_data[sa], seed_data[sb], match, args.n_samples)
            actual_results.append(r)
            print(f"\n  {sa} vs {sb} (n={r['n_matched']}, avg_cands={np.mean(n_cands):.1f}):")
            print(f"    Decoder cosine:        {r['decoder_cosine']:.3f}")
            print(f"    Variable overlap ≥2/3: {r['variable_overlap']:.1%}")
            print(f"    Energy correlation:     {r['energy_correlation']:.3f}")
            print(f"    Temporal trace cosine:  {r['temporal_trace_cosine']:.3f}")
            print(f"    Pred. selector NDCG:    {r['prediction_selector_ndcg']:.3f}")

    # Fair random baseline
    print(f"\nComputing fair random baseline ({args.n_random_trials} trials)...")
    rand_results = []
    sa, sb = pairs[0]
    n_cands = all_n_cands[0]
    for _ in range(args.n_random_trials):
        match = random_two_stage_match(seed_data[sa], seed_data[sb], n_cands)
        r = compute_metrics(seed_data[sa], seed_data[sb], match, args.n_samples)
        rand_results.append(r)

    # Summary
    metric_keys = [
        ("decoder_cosine", "Decoder cosine"),
        ("variable_overlap", "Variable overlap ≥2/3"),
        ("energy_correlation", "Latent energy correlation"),
        ("temporal_trace_cosine", "Latent temporal cosine"),
        ("prediction_selector_ndcg", "Prediction selector NDCG"),
    ]

    print(f"\n{'='*65}")
    print(f"RESULTS (cos>{args.cos_threshold}, n_samples={args.n_samples})")
    print(f"{'='*65}")
    print(f"{'Metric':<30s} {'Actual (mean±std)':>18s} {'Fair Random':>12s}")
    print(f"{'-'*65}")
    for key, label in metric_keys:
        act_vals = [r[key] for r in actual_results]
        rand_vals = [r[key] for r in rand_results]
        act_m, act_s = np.mean(act_vals), np.std(act_vals)
        rand_m = np.mean(rand_vals)
        if key == "variable_overlap":
            print(f"  {label:<30s} {act_m:.0%} ± {act_s:.0%}          {rand_m:.1%}")
        else:
            print(f"  {label:<30s} {act_m:.3f} ± {act_s:.3f}      {rand_m:.3f}")

    # Save
    save_data = {
        "config": {
            "cos_threshold": args.cos_threshold,
            "n_samples": args.n_samples,
            "seeds": all_seeds,
            "n_random_trials": args.n_random_trials,
        },
        "actual": [
            {"pair": f"{pairs[i][0]}_{pairs[i][1]}", **actual_results[i]}
            for i in range(len(pairs))
        ],
        "fair_random": {
            "mean": {k: float(np.mean([r[k] for r in rand_results])) for k, _ in metric_keys},
            "std": {k: float(np.std([r[k] for r in rand_results])) for k, _ in metric_keys},
        },
    }
    out_path = out / "seed_stability_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Concept seed reproducibility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train seed variants")
    p_train.add_argument("--reference_dir", required=True, help="Reference TISAE experiment dir")
    p_train.add_argument("--ps_reference_dir", required=True, help="Reference PredSelector dir")
    p_train.add_argument("--backbone_checkpoint", required=True)
    p_train.add_argument("--dataset_path", required=True)
    p_train.add_argument("--output_dir", required=True)
    p_train.add_argument("--seeds", nargs="+", type=int, default=[43, 44])

    # Compare
    p_cmp = subparsers.add_parser("compare", help="Compare seed pairs")
    p_cmp.add_argument("--reference_dir", required=True, help="Reference TISAE experiment dir")
    p_cmp.add_argument("--ps_reference_dir", required=True, help="Reference PredSelector dir")
    p_cmp.add_argument("--backbone_checkpoint", required=True)
    p_cmp.add_argument("--dataset_path", required=True)
    p_cmp.add_argument("--output_dir", required=True)
    p_cmp.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    p_cmp.add_argument("--cos_threshold", type=float, default=0.7)
    p_cmp.add_argument("--n_samples", type=int, default=500)
    p_cmp.add_argument("--n_random_trials", type=int, default=10)
    p_cmp.add_argument("--batch_size", type=int, default=8)
    p_cmp.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
