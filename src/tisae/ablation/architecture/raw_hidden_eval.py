"""Faithfulness eval for the raw-hidden prediction selector.

Attribution = the PS's per-cell selection weight scattered to the [T,V] grid.
Reuses the same comprehensiveness/sufficiency machinery as the ablation eval,
and saves attributions so run_plausibility.py can score them.

Usage:
    python -m tisae.ablation.architecture.raw_hidden_eval \
        --backbone experiments/discrete/mimic3_mortality/backbone/best_model.pt \
        --ps_dir experiments/ablation/structural/raw_hidden/ps \
        --dataset_path data/unified_mimic3_48h.pkl \
        --output_dir experiments/ablation/structural/raw_hidden/eval \
        --n_samples 500 --save_attributions
    # then:
    python -m tisae.evaluation.plausibility.run_plausibility \
        --eval_dir experiments/ablation/structural/raw_hidden/eval/faithfulness \
        --dataset_path data/unified_mimic3_48h.pkl
"""
import argparse, json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.pred_selector import PredSelector, PredSelectorConfig
from tisae.models.ontology import create_ontology_mapping
from tisae.evaluation.faithfulness.attribution_methods import (
    marginal_masked_predict, conditional_masked_predict,
)
from tisae.evaluation.faithfulness.faithfulness import comprehensiveness, sufficiency
from tisae.evaluation.utils import build_obs_grid
from tisae.ablation.architecture.raw_hidden_ps import assemble_cell_features

FRACTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def load_ps(ps_dir, device):
    cfg = yaml.safe_load(open(Path(ps_dir) / 'config.yaml'))
    psc = cfg['pred_selector_config']
    valid = {f.name for f in fields(PredSelectorConfig)}
    ps_cfg = PredSelectorConfig(**{k: v for k, v in psc.items() if k in valid})
    ps = PredSelector(ps_cfg, cfg['n_island_features'], cfg['D'], C_eff=1)
    ck = torch.load(Path(ps_dir) / 'best_model.pt', map_location=device, weights_only=False)
    ps.load_state_dict(ck['selector_state_dict'])
    return ps.to(device).eval()


@torch.no_grad()
def compute_attribution_raw_hidden(adapter, ps, batch, T, V):
    """[B,T,V] attribution = PS per-cell selection weight scattered to grid."""
    _, _, H = adapter.extract_hidden(batch, return_event_h=True)
    obs_mask = batch['obs_mask']; tau = batch['tau']; varis = batch['varis']
    B, L = obs_mask.shape
    if adapter.extract_layer != -1:
        h_pool = adapter.forward(batch, return_pooled=True)['pooled']
    else:
        h_pool, _ = adapter.pool_events(H, obs_mask)
    features = assemble_cell_features(H, obs_mask, varis, tau, T, V)
    s = ps(features, h_pool).view(B, -1)                     # [B,L]
    s = s.masked_fill(obs_mask < 0.5, float('-inf'))
    w = torch.softmax(s, dim=-1).nan_to_num(0.0) * obs_mask  # per-cell importance
    attr = torch.zeros(B, T, V, device=H.device)
    for b in range(B):
        valid = obs_mask[b].bool()
        t_idx = tau[b, valid].long().clamp(0, T - 1)
        v_idx = varis[b, valid].long().clamp(0, V - 1)
        attr[b].index_put_((t_idx, v_idx), w[b, valid], accumulate=True)
    return attr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--ps_dir", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--masking_strategies", nargs="+",
                    default=["removal", "marginal", "conditional"])
    ap.add_argument("--save_attributions", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    faith_dir = Path(args.output_dir) / "faithfulness"
    faith_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V, "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    adapter = load_backbone_adapter(args.backbone, device=str(device), dataset_meta=dm)
    adapter.eval()
    T, V = dataset.T, dataset.V
    ps = load_ps(args.ps_dir, device)

    marginal_dist = None
    if "marginal" in args.masking_strategies or "conditional" in args.masking_strategies:
        n_bins = 10; tr = np.array(dataset.splits["train"]); counts = torch.zeros(V, n_bins)
        n_marg = min(5000, len(tr))
        for i in range(0, n_marg, 100):
            bm = dataset.get_batch(tr[i:min(i+100, n_marg)])
            for b in range(bm['obs_mask'].shape[0]):
                om = bm['obs_mask'][b]; vb = bm['value_bins'][b][om > 0.5]; va = bm['varis'][b][om > 0.5]
                for vi in range(V):
                    mk = va == vi
                    if mk.any(): counts[vi] += torch.bincount(vb[mk].long(), minlength=n_bins)[:n_bins].float()
        marginal_dist = (counts / counts.sum(1, keepdim=True).clamp_min(1)).clamp_min(1e-6).to(device)

    def predict_fn_for(strat):
        if strat == "removal": return None
        if strat == "marginal": return lambda bb, ba, m: marginal_masked_predict(bb, ba, m, marginal_dist=marginal_dist)
        if strat == "conditional": return lambda bb, ba, m: conditional_masked_predict(bb, ba, m, marginal_dist=marginal_dist)

    test_idx = dataset.splits["test"][:args.n_samples]
    res = defaultdict(list); saved_attrs = []; saved_obs = []; saved_indices = []
    for idx in tqdm(test_idx, desc="raw_hidden"):
        batch = collate_fn([dataset.get_item(idx)])
        batch = {k: v.to(device) for k, v in batch.items()}
        attr = compute_attribution_raw_hidden(adapter, ps, batch, T, V)
        for strat in args.masking_strategies:
            pf = predict_fn_for(strat)
            for k, v in comprehensiveness(adapter, batch, attr, fractions=FRACTIONS, predict_fn=pf).items():
                res[f"{strat}/comp/{k}"].append(v)
            for k, v in sufficiency(adapter, batch, attr, fractions=FRACTIONS, predict_fn=pf).items():
                res[f"{strat}/suff/{k}"].append(v)
        if args.save_attributions:
            saved_attrs.append(attr.cpu()); saved_obs.append(build_obs_grid(batch, T, V).cpu()); saved_indices.append(idx)

    summary = {k: float(np.mean(v)) for k, v in res.items()}
    for strat in args.masking_strategies:
        aopc = [v for k, v in summary.items() if k.startswith(f"{strat}/comp/comp@")]
        if aopc: summary[f"{strat}/aopc"] = float(np.mean(aopc))
    out = {'model_type': 'raw_hidden', 'n_samples': len(test_idx),
           'masking_strategies': args.masking_strategies, 'results': {'raw_hidden': summary}}
    json.dump(out, open(faith_dir / 'evaluation_results.json', 'w'), indent=2)
    if args.save_attributions:
        ad = faith_dir / "attributions"; ad.mkdir(exist_ok=True)
        torch.save({'attr': torch.cat(saved_attrs), 'obs_grid': torch.cat(saved_obs),
                    'indices': torch.tensor([int(i) for i in saved_indices])},
                   ad / "raw_hidden.pt")
    print("\n=== raw_hidden faithfulness ===")
    for strat in args.masking_strategies:
        if f"{strat}/aopc" in summary: print(f"  AOPC ({strat}): {summary[f'{strat}/aopc']:.4f}")
    print(f"saved -> {faith_dir/'evaluation_results.json'}")


if __name__ == "__main__":
    main()
