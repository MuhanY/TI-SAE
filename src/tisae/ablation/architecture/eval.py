"""Full evaluation for ablation model variants (no_temporal, simple_sae).

Computes faithfulness (AOPC with removal/marginal/conditional),
structural metrics, and saves attributions for plausibility.

Usage:
    python -m tisae.ablation.architecture.eval \\
        --experiment_dir experiments_ablation/structural/no_temporal/tisae \\
        --ps_dir experiments_ablation/structural/no_temporal/ps \\
        --dataset_path data/unified_mimic3_48h.pkl \\
        --output_dir experiments_ablation/eval/no_temporal \\
        --n_samples 500 --save_attributions

    # Then plausibility on saved attributions:
    python -m tisae.evaluation.plausibility.run_plausibility \\
        --eval_dir experiments_ablation/eval/no_temporal/faithfulness \\
        --dataset_path data/unified_mimic3_48h.pkl
"""

import argparse
import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import fields, asdict
from tqdm import tqdm

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.tisae import TISAEConfig
from tisae.models.pred_selector import (
    PredSelector, PredSelectorConfig, assemble_island_features,
)
from tisae.models.ontology import create_ontology_mapping
from tisae.evaluation.faithfulness.attribution_methods import (
    compute_sae_attribution,
    compute_pred_selector_attribution,
    extract_pred_selector_softs,
    marginal_masked_predict,
    conditional_masked_predict,
)
from tisae.evaluation.faithfulness.faithfulness import (
    comprehensiveness, sufficiency,
)
from tisae.evaluation.utils import build_obs_grid

from tisae.ablation.architecture.tisae_no_temporal import TISAENoTemporal
from tisae.ablation.architecture.simple_sae import SimpleSAE, SimpleSAEConfig


FRACTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def load_ablation_model(experiment_dir, dataset_path, device='cuda'):
    """Load ablation model from experiment directory."""
    exp_dir = Path(experiment_dir)
    device = torch.device(device)

    with open(exp_dir / 'config.yaml') as f:
        config = yaml.safe_load(f)

    model_type = config.get('model_type', 'tisae_no_temporal')
    dataset = UnifiedTimeSeriesDataset.load(dataset_path)
    dm = {"T": dataset.T, "V": dataset.V,
          "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    T, V = dataset.T, dataset.V

    bb_path = config.get('args', {}).get('backbone_checkpoint')
    adapter = load_backbone_adapter(bb_path, device=str(device), dataset_meta=dm)
    extract_layer = config.get('args', {}).get('extract_layer', -1)
    adapter.set_extract_layer(extract_layer)

    ckpt = torch.load(exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)

    if model_type == 'tisae_no_temporal':
        cfg_dict = config.get('tisae_config', {})
        valid = {f.name for f in fields(TISAEConfig)}
        cfg = TISAEConfig(**{k: v for k, v in cfg_dict.items() if k in valid})

        channel_var_mask = None
        if cfg.lambda_ch_purity > 0:
            ontology = create_ontology_mapping(dataset.var_names, n_channels=cfg.C)
            channel_var_mask = ontology.get_hard_mask()

        model = TISAENoTemporal(cfg, channel_var_mask=channel_var_mask)
        model.load_state_dict(ckpt['sae_state_dict'])
        model = model.to(device).eval()

        H_grid_mean = ckpt.get('H_grid_mean')
        if H_grid_mean is not None:
            H_grid_mean = H_grid_mean.to(device)
            adapter.set_residualization_mean(H_grid_mean)

        return {
            'model': model, 'adapter': adapter, 'dataset': dataset,
            'config': cfg, 'H_grid_mean': H_grid_mean,
            'model_type': model_type, 'T': T, 'V': V,
        }

    elif model_type == 'simple_sae':
        cfg_dict = config.get('simple_sae_config', {})
        cfg = SimpleSAEConfig(**cfg_dict)
        model = SimpleSAE(cfg)
        model.load_state_dict(ckpt['sae_state_dict'])
        model = model.to(device).eval()

        # No residualization for simple_sae
        return {
            'model': model, 'adapter': adapter, 'dataset': dataset,
            'config': cfg, 'H_mean': None,
            'model_type': model_type, 'T': T, 'V': V,
        }

    raise ValueError(f"Unknown model_type: {model_type}")


def load_pred_selector(ps_dir, device):
    """Load trained PredSelector from directory."""
    ps_dir = Path(ps_dir)
    with open(ps_dir / 'config.yaml') as f:
        config = yaml.safe_load(f)
    ps_cfg_dict = config.get('pred_selector_config', {})
    valid = {f.name for f in fields(PredSelectorConfig)}
    ps_cfg = PredSelectorConfig(**{k: v for k, v in ps_cfg_dict.items() if k in valid})
    ps = PredSelector(
        ps_cfg,
        n_island_features=config.get('n_island_features', 276),
        D=config.get('D', 128),
        C_eff=config.get('C_eff', 1),
    )
    ckpt = torch.load(ps_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    ps.load_state_dict(ckpt['selector_state_dict'])
    return ps.to(device).eval()


@torch.no_grad()
def compute_attribution_simple_sae(model, adapter, batch, T, V, H_mean=None):
    """Compute [B, T, V] attribution from SimpleSAE (energy-based).
    No residualization — operates on raw H[B,L,D]."""
    _, _, H = adapter.extract_hidden(batch, return_event_h=True)
    obs_mask = batch["obs_mask"]
    tau = batch["tau"]
    varis = batch["varis"]
    H_input = H * obs_mask.unsqueeze(-1)
    return model.compute_attribution(H_input, obs_mask, tau, varis, T, V)


@torch.no_grad()
def compute_attribution_simple_sae_ps(model, adapter, ps, batch, T, V, H_mean=None):
    """Compute [B, T, V] PS-weighted attribution from SimpleSAE.

    PS scores each latent k → weight per-event activations by PS score.
    importance[b, l] = sum_k z[b,l,k] * ps_score[b,k]
    """
    from tisae.ablation.architecture.train_ps import assemble_latent_features_simple

    _, _, H = adapter.extract_hidden(batch, return_event_h=True)
    obs_mask = batch["obs_mask"]
    tau = batch["tau"]
    varis = batch["varis"]
    B = obs_mask.shape[0]

    # No residualization for simple_sae
    H_input = H * obs_mask.unsqueeze(-1)

    z = model.encode(H_input)  # [B, L, K]

    # Get h_pool for PS
    if adapter.extract_layer != -1:
        out_full = adapter.forward(batch, return_pooled=True)
        h_pool = out_full['pooled']
    else:
        h_pool, _ = adapter.pool_events(H, obs_mask)

    # Assemble features and get PS scores
    features = assemble_latent_features_simple(z, obs_mask, varis, V)
    s_pred = ps(features, h_pool)  # [B, K, 1, 1]
    scores = s_pred.view(B, -1)    # [B, K]
    weights = torch.softmax(scores, dim=-1)  # [B, K]

    # Weighted per-event importance
    importance = (z * weights.unsqueeze(1)).abs().sum(dim=-1)  # [B, L]
    importance = importance * obs_mask

    # Scatter to (T, V) grid
    attr = torch.zeros(B, T, V, device=H.device)
    for b in range(B):
        valid = obs_mask[b].bool()
        t_idx = tau[b, valid].long().clamp(0, T - 1)
        v_idx = varis[b, valid].long().clamp(0, V - 1)
        attr[b].index_put_((t_idx, v_idx), importance[b, valid], accumulate=True)

    return attr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--ps_dir", default=None)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--methods", nargs="+",
                        default=["sae", "pred_selector", "pred_selector_soft"])
    parser.add_argument("--masking_strategies", nargs="+",
                        default=["removal", "marginal", "conditional"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_attributions", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    faith_dir = Path(args.output_dir) / "faithfulness"
    faith_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    loaded = load_ablation_model(args.experiment_dir, args.dataset_path, str(device))
    model = loaded['model']
    adapter = loaded['adapter']
    dataset = loaded['dataset']
    model_type = loaded['model_type']
    T, V = loaded['T'], loaded['V']
    D = adapter.D

    # Load PredSelector if needed
    ps = None
    ps_methods = {"pred_selector", "pred_selector_soft"}
    if ps_methods & set(args.methods):
        ps_dir = args.ps_dir
        if ps_dir is None:
            ps_dir = str(Path(args.experiment_dir).parent / "ps")
        if Path(ps_dir).exists():
            ps = load_pred_selector(ps_dir, device)
            print(f"PredSelector loaded from {ps_dir}")
        else:
            print(f"WARNING: PS dir {ps_dir} not found, skipping PS methods")
            args.methods = [m for m in args.methods if m not in ps_methods]

    # Marginal distribution for marginal/conditional masking
    marginal_dist = None
    if "marginal" in args.masking_strategies or "conditional" in args.masking_strategies:
        print("Computing per-variable marginal distributions from training data...")
        n_bins = 10
        train_indices = np.array(dataset.splits["train"])
        counts = torch.zeros(V, n_bins)
        n_marg = min(5000, len(train_indices))
        for i in range(0, n_marg, 100):
            end_i = min(i + 100, n_marg)
            batch_m = dataset.get_batch(train_indices[i:end_i])
            for b in range(batch_m['obs_mask'].shape[0]):
                om = batch_m['obs_mask'][b]
                vb_b = batch_m['value_bins'][b][om > 0.5]
                va_b = batch_m['varis'][b][om > 0.5]
                for v_idx in range(V):
                    mask_v = va_b == v_idx
                    if mask_v.any():
                        bc = torch.bincount(vb_b[mask_v].long(), minlength=n_bins)
                        counts[v_idx] += bc[:n_bins].float()
        counts_sum = counts.sum(dim=1, keepdim=True).clamp_min(1)
        marginal_dist = (counts / counts_sum).clamp_min(1e-6).to(device)
        print(f"  Marginal dist computed from {n_marg} training samples")

    test_indices = dataset.splits["test"][:args.n_samples]
    H_grid_mean = loaded.get('H_grid_mean')
    H_mean = loaded.get('H_mean')

    # Build attribution functions
    def get_attr_fn(method):
        if model_type == 'tisae_no_temporal':
            # TISAENoTemporal has TISAE-compatible shims (island_params, gaussian_mask)
            if method == "sae":
                return lambda batch: compute_sae_attribution(
                    adapter, model, batch, T, V, H_grid_mean=H_grid_mean)
            elif method == "pred_selector":
                return lambda batch: compute_pred_selector_attribution(
                    adapter, model, ps, batch, T, V, H_grid_mean=H_grid_mean)
            elif method == "pred_selector_soft":
                return lambda batch: extract_pred_selector_softs(
                    adapter, model, ps, batch, T, V, H_grid_mean=H_grid_mean)[0]
        elif model_type == 'simple_sae':
            if method == "sae":
                return lambda batch: compute_attribution_simple_sae(
                    model, adapter, batch, T, V, H_mean=H_mean)
            elif method in ps_methods:
                return lambda batch: compute_attribution_simple_sae_ps(
                    model, adapter, ps, batch, T, V, H_mean=H_mean)
        raise ValueError(f"Unknown method {method} for {model_type}")

    # Build predict functions for masking strategies
    def get_predict_fn(strategy):
        if strategy == "removal":
            return None  # default: zero-out
        elif strategy == "marginal":
            return lambda backbone, batch, mask_tv: marginal_masked_predict(
                backbone, batch, mask_tv, marginal_dist=marginal_dist)
        elif strategy == "conditional":
            return lambda backbone, batch, mask_tv: conditional_masked_predict(
                backbone, batch, mask_tv, marginal_dist=marginal_dist)

    # Main evaluation loop
    all_results = {}
    for method in args.methods:
        attr_fn = get_attr_fn(method)
        print(f"\n{'='*60}")
        print(f"Evaluating: {method}")
        print(f"{'='*60}")

        method_results = defaultdict(list)
        saved_attrs = []
        saved_obs = []
        saved_indices = []

        for i in tqdm(range(len(test_indices)), desc=method):
            idx = test_indices[i]
            sample = dataset.get_item(idx)
            batch = collate_fn([sample])
            batch = {k: v.to(device) for k, v in batch.items()}

            attr = attr_fn(batch)  # [1, T, V]
            obs_grid = build_obs_grid(batch, T, V)

            # Faithfulness per masking strategy
            for strat in args.masking_strategies:
                predict_fn = get_predict_fn(strat)
                comp = comprehensiveness(adapter, batch, attr,
                                         fractions=FRACTIONS, predict_fn=predict_fn)
                suff_res = sufficiency(adapter, batch, attr,
                                       fractions=FRACTIONS, predict_fn=predict_fn)
                for k, v in comp.items():
                    method_results[f"{strat}/comp/{k}"].append(v)
                for k, v in suff_res.items():
                    method_results[f"{strat}/suff/{k}"].append(v)

            if args.save_attributions:
                saved_attrs.append(attr.cpu())
                saved_obs.append(obs_grid.cpu())
                saved_indices.append(idx)

        # Aggregate
        method_summary = {}
        for k, vals in method_results.items():
            method_summary[k] = float(np.mean(vals))

        # AOPC per strategy
        for strat in args.masking_strategies:
            aopc_vals = [v for k, v in method_summary.items()
                         if k.startswith(f"{strat}/comp/comp@")]
            if aopc_vals:
                method_summary[f"{strat}/aopc"] = float(np.mean(aopc_vals))

        all_results[method] = method_summary

        # Save attributions
        if args.save_attributions:
            attr_dir = faith_dir / "attributions"
            attr_dir.mkdir(exist_ok=True)
            torch.save({
                'attr': torch.cat(saved_attrs, dim=0),
                'obs_grid': torch.cat(saved_obs, dim=0),
                'indices': np.array(saved_indices),
            }, attr_dir / f"{method}.pt")

    # Save all results
    final = {
        'model_type': model_type,
        'n_samples': len(test_indices),
        'methods': args.methods,
        'masking_strategies': args.masking_strategies,
        'results': all_results,
    }
    with open(faith_dir / 'evaluation_results.json', 'w') as f:
        json.dump(final, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary: {model_type}")
    print(f"{'='*60}")
    for method, summary in all_results.items():
        print(f"\n  {method}:")
        for strat in args.masking_strategies:
            aopc_key = f"{strat}/aopc"
            if aopc_key in summary:
                print(f"    AOPC ({strat}): {summary[aopc_key]:.4f}")
        for k in sorted(summary.keys()):
            if k.startswith("struct/"):
                print(f"    {k}: {summary[k]:.4f}")

    print(f"\nSaved to {faith_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
