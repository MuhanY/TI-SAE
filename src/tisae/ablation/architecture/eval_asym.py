"""Faithfulness + attribution export for the asymmetric TI-SAE ablation.

Thin wrapper (copy-modify of ablation eval.py): builds AsymTISAE + its PS and runs the canonical attribution.
Saves pred_selector_soft attributions for plausibility.

Usage:
  python -m tisae.ablation.architecture.eval_asym \
    --sae_experiment_dir experiments/ablation/island_shape/learned/tisae \
    --ps_dir experiments/ablation/island_shape/learned/ps \
    --dataset_path data/unified_mimic3_48h.pkl \
    --output_dir experiments/ablation/island_shape/learned/eval \
    --n_samples 500
"""
import argparse
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from dataclasses import fields
from tqdm import tqdm

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.tisae import TISAEConfig
from tisae.models.pred_selector import PredSelector, PredSelectorConfig
from tisae.models.ontology import create_ontology_mapping
from tisae.evaluation.utils import build_obs_grid
from tisae.evaluation.faithfulness.attribution_methods import (
    extract_pred_selector_softs, compute_pred_selector_attribution,
    marginal_masked_predict, conditional_masked_predict,
)
from tisae.evaluation.faithfulness.faithfulness import (
    comprehensiveness, sufficiency,
)
from tisae.ablation.architecture.asym_tisae import AsymTISAE

# Same fraction grid as the canonical faithfulness eval (0.05..0.5).
FRACTIONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def build_asym_sae(tisae_dir, dataset, device, fix_a_override=None):
    cfgd = yaml.safe_load(open(Path(tisae_dir) / "config.yaml"))
    a_max = cfgd.get("asym_a_max", 0.7)
    fix_a = fix_a_override if fix_a_override is not None else cfgd.get("fix_a", None)
    valid = {f.name for f in fields(TISAEConfig)}
    cfg = TISAEConfig(**{k: v for k, v in cfgd["tisae_config"].items() if k in valid})
    cvm = None
    if cfg.lambda_ch_purity > 0:
        cvm = create_ontology_mapping(dataset.var_names, n_channels=cfg.C).get_hard_mask()
    sae = AsymTISAE(cfg, channel_var_mask=cvm, asym_a_max=a_max, fix_a=fix_a)
    ck = torch.load(Path(tisae_dir) / "best_model.pt", map_location="cpu", weights_only=False)
    sae.load_state_dict(ck["sae_state_dict"], strict=False)
    return sae.to(device).eval(), ck.get("H_grid_mean"), cfg, cfgd


def load_ps(ps_dir, device):
    cfgd = yaml.safe_load(open(Path(ps_dir) / "config.yaml"))
    valid = {f.name for f in fields(PredSelectorConfig)}
    pcfg = PredSelectorConfig(**{k: v for k, v in cfgd["pred_selector_config"].items() if k in valid})
    ps = PredSelector(pcfg, cfgd.get("n_island_features", 276),
                      cfgd.get("D", 128), cfgd.get("C_eff", 1))
    ck = torch.load(Path(ps_dir) / "best_model.pt", map_location="cpu", weights_only=False)
    ps.load_state_dict(ck["selector_state_dict"])
    return ps.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_experiment_dir", required=True)
    ap.add_argument("--ps_dir", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--fix_a", type=float, default=None,
                    help="override config fix_a (e.g. 0.0 to eval a known model as exactly symmetric)")
    ap.add_argument("--masking_strategies", nargs="+",
                    default=["removal", "marginal", "conditional"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V, "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    sae, H_grid_mean, cfg, asym_cfgd = build_asym_sae(
        args.sae_experiment_dir, dataset, device, fix_a_override=args.fix_a)
    bb = asym_cfgd["args"]["backbone_checkpoint"]
    extract_layer = asym_cfgd.get("args", {}).get("extract_layer", -1)
    adapter = load_backbone_adapter(bb, device=str(device), dataset_meta=dm)
    adapter.set_extract_layer(extract_layer)
    if H_grid_mean is not None:
        H_grid_mean = H_grid_mean.to(device)
        adapter.set_residualization_mean(H_grid_mean)
    ps = load_ps(args.ps_dir, device)
    T, V = cfg.T, cfg.V
    print(f"AsymTISAE eval: a_max={asym_cfgd.get('asym_a_max')}, n_samples={args.n_samples}")

    # marginal distribution for marginal/conditional masking
    marginal_dist = None
    if {"marginal", "conditional"} & set(args.masking_strategies):
        n_bins = 10
        tr = np.array(dataset.splits["train"])
        counts = torch.zeros(V, n_bins)
        n_marg = min(5000, len(tr))
        for i in range(0, n_marg, 100):
            bm = dataset.get_batch(tr[i:min(i + 100, n_marg)])
            for b in range(bm['obs_mask'].shape[0]):
                om = bm['obs_mask'][b]
                vb = bm['value_bins'][b][om > 0.5]
                va = bm['varis'][b][om > 0.5]
                for v_idx in range(V):
                    mv = va == v_idx
                    if mv.any():
                        counts[v_idx] += torch.bincount(vb[mv].long(), minlength=n_bins)[:n_bins].float()
        marginal_dist = (counts / counts.sum(1, keepdim=True).clamp_min(1)).clamp_min(1e-6).to(device)

    def predict_fn_for(strat):
        if strat == "removal":
            return None
        if strat == "marginal":
            return lambda bb_, ba, m: marginal_masked_predict(bb_, ba, m, marginal_dist=marginal_dist)
        if strat == "conditional":
            return lambda bb_, ba, m: conditional_masked_predict(bb_, ba, m, marginal_dist=marginal_dist)

    test_idx = dataset.splits["test"][:args.n_samples]
    method_results = defaultdict(list)
    saved_attrs, saved_obs, saved_indices = [], [], []

    for i in tqdm(range(len(test_idx)), desc="pred_selector_soft"):
        idx = test_idx[i]
        batch = collate_fn([dataset.get_item(idx)])
        batch = {k: v.to(device) for k, v in batch.items()}
        attr = extract_pred_selector_softs(adapter, sae, ps, batch, T, V, H_grid_mean=H_grid_mean)[0]
        obs_grid = build_obs_grid(batch, T, V)
        for strat in args.masking_strategies:
            pf = predict_fn_for(strat)
            comp = comprehensiveness(adapter, batch, attr, fractions=FRACTIONS, predict_fn=pf)
            suff = sufficiency(adapter, batch, attr, fractions=FRACTIONS, predict_fn=pf)
            for k, v in comp.items():
                method_results[f"{strat}/comp/{k}"].append(v)
            for k, v in suff.items():
                method_results[f"{strat}/suff/{k}"].append(v)
        saved_attrs.append(attr.cpu())
        saved_obs.append(obs_grid.cpu())
        saved_indices.append(int(idx))

    # aggregate: AOPC = mean over fractions of comprehensiveness (aopc_comp key)
    summary = {}
    for k, vals in method_results.items():
        summary[k] = float(np.mean(vals))
    # headline AOPC per strategy = mean of the per-fraction comprehensiveness
    for strat in args.masking_strategies:
        frac_keys = [k for k in summary
                     if k.startswith(f"{strat}/comp/comp@") and not k.endswith("_samples")]
        if frac_keys:
            summary[f"{strat}/AOPC_comp_mean"] = float(np.mean([summary[k] for k in frac_keys]))

    json.dump(summary, open(out / "faithfulness_summary.json", "w"), indent=2)
    # save in the layout run_plausibility expects: <eval_dir>/attributions/<method>.pt
    # with keys attr / indices / obs_grid.
    attr_dir = out / "attributions"
    attr_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "attr": torch.cat(saved_attrs, dim=0),       # [N,T,V]
        "indices": torch.tensor(saved_indices),      # tensor
        "obs_grid": torch.cat(saved_obs, dim=0),     # [N,T,V]
    }, attr_dir / "pred_selector_soft.pt")

    print("\n=== AsymTISAE faithfulness (compare CONDITIONAL to canonical .149) ===")
    for strat in args.masking_strategies:
        key = f"{strat}/AOPC_comp_mean"
        if key in summary:
            print(f"  {strat:12s} AOPC(comp mean over fractions) = {summary[key]:.4f}")
    print(f"saved -> {out}/faithfulness_summary.json + attributions/pred_selector_soft.pt")


if __name__ == "__main__":
    main()
