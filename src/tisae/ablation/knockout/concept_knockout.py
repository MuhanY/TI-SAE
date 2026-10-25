"""Concept knockout experiment:
zero-ablate top-k PredSelector concepts, measure prediction change.

Removing top-ranked concepts should monotonically degrade prediction.

Usage:
    python -m tisae.ablation.concept_knockout \
        --experiment_dir experiments/discrete/mimic3_mortality/tisae \
        --pred_selector_dir experiments/discrete/mimic3_mortality/ps \
        --dataset_path data/unified_48h.pkl \
        --n_samples 500 --seed 42 \
        --output_dir experiments_ablation/concept_knockout
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

from tisae.evaluation.utils import load_evaluation_models
from tisae.evaluation.faithfulness.run_evaluation import load_pred_selector
from tisae.models.pred_selector import assemble_island_features


KNOCKOUT_KS = list(range(1, 65))


@torch.no_grad()
def concept_knockout(
    backbone, sae, selector, dataset,
    n_samples=500, knockout_ks=None, batch_size=32,
    n_random_trials=10,
    device="cuda", H_grid_mean=None, seed=42,
):
    """Run concept knockout experiment with random control.

    For each sample:
    1. Get full backbone prediction (logit_full)
    2. Run SAE encoder + PredSelector to rank islands
    3. For each k in knockout_ks:
       a. PS-ranked: remove top-k islands by PS score
       b. Random: remove k random islands (n_random_trials times)
    4. Record delta_logit and delta_prob

    Returns dict with per-k statistics for both PS-ranked and random.
    """
    if knockout_ks is None:
        knockout_ks = KNOCKOUT_KS

    cfg = sae.cfg
    K, M = cfg.K, cfg.M
    K_eff, C_eff = sae.K_eff, sae.C_eff
    V = cfg.V
    n_global_pred = selector.config.n_global_pred

    test_indices = np.array(dataset.splits["test"])
    rng = np.random.RandomState(seed)
    if len(test_indices) > n_samples:
        test_indices = rng.choice(test_indices, n_samples, replace=False)
    n_samples = len(test_indices)

    # Collect per-sample results
    results = {k: {"delta_logit": [], "delta_prob": []} for k in knockout_ks}
    results_random = {k: {"delta_logit": [], "delta_prob": []} for k in knockout_ks}
    results["sae_full"] = {"delta_logit": [], "delta_prob": []}
    logits_full_all = []
    rng_ko = np.random.RandomState(seed + 1)

    for start in tqdm(range(0, n_samples, batch_size), desc="Knockout"):
        end = min(start + batch_size, n_samples)
        idx = test_indices[start:end]
        batch = dataset.get_batch(idx)
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["obs_mask"].shape[0]

        # 1. Full backbone prediction
        logit_full = backbone.forward(batch)["logits"]  # [B]
        prob_full = torch.sigmoid(logit_full)
        logits_full_all.append(logit_full.cpu())

        # 2. SAE encode
        H_grid, obs_grid, H = backbone.extract_hidden(batch, return_event_h=True)

        if H_grid_mean is not None:
            if H_grid_mean.dim() == 2:
                H_grid_input = H_grid - H_grid_mean.unsqueeze(0).unsqueeze(1)
            else:
                H_grid_input = H_grid - H_grid_mean.unsqueeze(0)
            H_grid_input = H_grid_input * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

        A0 = sae.encoder(H_grid_input, obs_grid)  # [B, K, T]
        T_len = A0.size(2)

        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T_len).permute(0, 2, 3, 1)
        else:
            A0_4d = A0.unsqueeze(-1)  # [B, K_eff, T, 1]

        mu, sigma, gate = sae.island_params(A0_4d)
        g = sae.gaussian_mask.build_gaussians(mu, sigma.detach())

        # 3. PredSelector scores
        if backbone.extract_layer != -1:
            h_pool = backbone.forward(batch, return_pooled=True)["pooled"]
        else:
            h_pool, _ = backbone.pool_events(H, batch["obs_mask"])

        W_dec = sae.decoder.get_W().detach()
        include_obs = (selector.n_island_features != 6 + V)
        features = assemble_island_features(
            A0_4d, g, mu, sigma, gate, obs_grid, W_dec,
            None, K_eff, C_eff, include_obs_profile=include_obs)

        s_pred = selector(features, h_pool)  # [B, K_eff, C_eff, M]
        s_flat = s_pred.view(B, -1)  # [B, K_eff*C_eff*M]

        # Full PS selection (top n_global_pred)
        _, top_idx = s_flat.topk(n_global_pred, dim=-1)
        sel_full = torch.zeros_like(s_flat).scatter_(-1, top_idx, 1.0)

        # SAE-full prediction (all selected, no knockout)
        sel_full_4d = sel_full.view(B, K_eff, C_eff, M)
        mask_full = sae.gaussian_mask.forward_mask(g, sel_full_4d)
        A_hat_full = A0_4d * mask_full
        if C_eff > 1:
            A_hat_3d = A_hat_full.permute(0, 3, 1, 2).reshape(B, K, T_len)
        else:
            A_hat_3d = A_hat_full.squeeze(-1)
        H_hat_full = sae.decoder(A_hat_3d)
        logit_sae = backbone.predict_from_grid(H_hat_full, batch, obs_grid)
        prob_sae = torch.sigmoid(logit_sae)
        results["sae_full"]["delta_logit"].append(
            (logit_full - logit_sae).abs().cpu())
        results["sae_full"]["delta_prob"].append(
            (prob_full - prob_sae).abs().cpu())

        # 4. Knockout experiments
        # Helper to decode and predict from a selection mask
        def _knockout_predict(sel_mask_flat):
            sel_4d = sel_mask_flat.view(B, K_eff, C_eff, M)
            mask = sae.gaussian_mask.forward_mask(g, sel_4d)
            a_hat = A0_4d * mask
            if C_eff > 1:
                a3d = a_hat.permute(0, 3, 1, 2).reshape(B, K, T_len)
            else:
                a3d = a_hat.squeeze(-1)
            h_hat = sae.decoder(a3d)
            return backbone.predict_from_grid(h_hat, batch, obs_grid)

        # Find indices of selected islands (for random sampling)
        selected_idx = [torch.where(sel_full[b] > 0)[0] for b in range(B)]

        for k in knockout_ks:
            if k > s_flat.shape[1]:
                continue

            # 4a. PS-ranked knockout: remove top-k by PS score
            _, ko_idx = s_flat.topk(k, dim=-1)
            sel_ko = sel_full.clone().scatter_(-1, ko_idx, 0.0)
            logit_ko = _knockout_predict(sel_ko)
            prob_ko = torch.sigmoid(logit_ko)
            results[k]["delta_logit"].append(
                (logit_full - logit_ko).abs().cpu())
            results[k]["delta_prob"].append(
                (prob_full - prob_ko).abs().cpu())

            # 4b. Random knockout: remove k random selected islands
            rand_deltas_logit = []
            rand_deltas_prob = []
            for _ in range(n_random_trials):
                sel_rand = sel_full.clone()
                for b in range(B):
                    n_sel = len(selected_idx[b])
                    if k >= n_sel:
                        sel_rand[b] = 0.0
                    else:
                        perm = rng_ko.permutation(n_sel)[:k]
                        rand_idx = selected_idx[b][perm]
                        sel_rand[b, rand_idx] = 0.0
                logit_rand = _knockout_predict(sel_rand)
                prob_rand = torch.sigmoid(logit_rand)
                rand_deltas_logit.append(
                    (logit_full - logit_rand).abs().cpu())
                rand_deltas_prob.append(
                    (prob_full - prob_rand).abs().cpu())
            # Average across trials
            results_random[k]["delta_logit"].append(
                torch.stack(rand_deltas_logit).mean(dim=0))
            results_random[k]["delta_prob"].append(
                torch.stack(rand_deltas_prob).mean(dim=0))

    # Aggregate
    def _aggregate(res_dict):
        summary = {}
        for key in list(res_dict.keys()):
            dl = torch.cat(res_dict[key]["delta_logit"])
            dp = torch.cat(res_dict[key]["delta_prob"])
            summary[str(key)] = {
                "mean_delta_logit": dl.mean().item(),
                "std_delta_logit": dl.std().item(),
                "median_delta_logit": dl.median().item(),
                "mean_delta_prob": dp.mean().item(),
                "std_delta_prob": dp.std().item(),
                "median_delta_prob": dp.median().item(),
            }
        return summary

    logits_full_cat = torch.cat(logits_full_all)
    output = {
        "ps_ranked": _aggregate(results),
        "random": _aggregate(results_random),
        "baseline": {
            "mean_logit": logits_full_cat.mean().item(),
            "mean_prob": torch.sigmoid(logits_full_cat).mean().item(),
        },
        "n_samples": n_samples,
        "knockout_ks": knockout_ks,
        "n_global_pred": n_global_pred,
        "n_random_trials": n_random_trials,
    }
    return output


def main():
    parser = argparse.ArgumentParser(description="Concept knockout experiment")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--pred_selector_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    models = load_evaluation_models(
        args.experiment_dir, args.dataset_path, device=device)

    backbone = models["backbone"]
    sae = models["sae"]
    dataset = models["dataset"]
    H_grid_mean = models.get("H_grid_mean")

    ps_data = load_pred_selector(args.pred_selector_dir, device=device)
    selector = ps_data["selector"]

    results = concept_knockout(
        backbone, sae, selector, dataset,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
        H_grid_mean=H_grid_mean,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "knockout_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n=== Concept Knockout Results ===")
    print(f"{'k':>6s}  {'PS Δlogit':>10s}  {'PS Δprob':>10s}  "
          f"{'Rand Δlogit':>11s}  {'Rand Δprob':>10s}")
    print("-" * 55)
    ps = results["ps_ranked"]
    rd = results["random"]
    print(f"{'sae':>6s}  {ps['sae_full']['mean_delta_logit']:10.4f}  "
          f"{ps['sae_full']['mean_delta_prob']:10.4f}  "
          f"{'--':>11s}  {'--':>10s}")
    for k in results["knockout_ks"]:
        sk = str(k)
        if sk in ps and sk in rd:
            print(f"{k:>6d}  {ps[sk]['mean_delta_logit']:10.4f}  "
                  f"{ps[sk]['mean_delta_prob']:10.4f}  "
                  f"{rd[sk]['mean_delta_logit']:11.4f}  "
                  f"{rd[sk]['mean_delta_prob']:10.4f}")
    print(f"\nBaseline: mean_logit={results['baseline']['mean_logit']:.4f}, "
          f"mean_prob={results['baseline']['mean_prob']:.4f}")
    print(f"n_samples={results['n_samples']}, n_global_pred={results['n_global_pred']}, "
          f"n_random_trials={results['n_random_trials']}")
    print(f"\nSaved to {out_dir / 'knockout_results.json'}")


if __name__ == "__main__":
    main()
