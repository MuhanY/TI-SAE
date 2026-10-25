"""Stage-B PredSelector trainer for the asymmetric TI-SAE ablation.

Builds the frozen AsymTISAE and reuses the canonical PS training loop (train_pred_selector.py).
PS hyperparameters are read from the canonical ps config for equivalence.

Usage:
  python -m tisae.ablation.architecture.ps_asym \
    --sae_experiment_dir experiments/ablation/island_shape/learned/tisae \
    --dataset_path data/unified_mimic3_48h.pkl \
    --ps_config experiments/discrete/mimic3_mortality/ps/config.yaml \
    --output_dir experiments/ablation/island_shape/learned/ps
"""
import argparse
import yaml
import torch
from pathlib import Path
from dataclasses import fields, asdict
from torch.utils.data import DataLoader

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.tisae import TISAEConfig
from tisae.models.pred_selector import PredSelector, PredSelectorConfig
from tisae.models.ontology import create_ontology_mapping
from tisae.training.train_pred_selector import (
    Logger, PSDataset, ps_collate_fn, worker_init_fn, set_seed,
    train_epoch, evaluate,
)
from tisae.ablation.architecture.asym_tisae import AsymTISAE


def build_asym_sae(tisae_dir, dataset, device):
    cfgd = yaml.safe_load(open(Path(tisae_dir) / "config.yaml"))
    a_max = cfgd.get("asym_a_max", 0.7)
    fix_a = cfgd.get("fix_a", None)
    valid = {f.name for f in fields(TISAEConfig)}
    cfg = TISAEConfig(**{k: v for k, v in cfgd["tisae_config"].items() if k in valid})
    cvm = None
    if cfg.lambda_ch_purity > 0:
        cvm = create_ontology_mapping(dataset.var_names, n_channels=cfg.C).get_hard_mask()
    sae = AsymTISAE(cfg, channel_var_mask=cvm, asym_a_max=a_max, fix_a=fix_a)
    ck = torch.load(Path(tisae_dir) / "best_model.pt", map_location="cpu", weights_only=False)
    sae.load_state_dict(ck["sae_state_dict"], strict=False)
    return sae.to(device), ck.get("H_grid_mean"), cfg, cfgd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_experiment_dir", required=True, help="asym tisae dir")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--ps_config", required=True, help="canonical ps config.yaml (hyperparam equivalence)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = Logger(out)

    pc = yaml.safe_load(open(args.ps_config))
    psc = pc["pred_selector_config"]
    n_prefilter = pc.get("n_prefilter", 2560)
    n_pool_full = pc.get("n_pool_full", 256)
    max_grad_norm = pc.get("max_grad_norm", 1.0)
    include_obs_profile = pc.get("include_obs_profile", True)
    soft_selection = pc.get("soft_selection", True)
    soft_tau = pc.get("soft_tau", 1.0)

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V, "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}

    sae, H_grid_mean, cfg, asym_cfgd = build_asym_sae(args.sae_experiment_dir, dataset, device)
    for p in sae.parameters():
        p.requires_grad = False
    sae.eval()

    bb = asym_cfgd["args"]["backbone_checkpoint"]
    extract_layer = asym_cfgd.get("args", {}).get("extract_layer", -1)
    adapter = load_backbone_adapter(bb, device=str(device), dataset_meta=dm)
    adapter.set_extract_layer(extract_layer)
    if H_grid_mean is not None:
        H_grid_mean = H_grid_mean.to(device)
        adapter.set_residualization_mean(H_grid_mean)

    T, V, D = cfg.T, cfg.V, adapter.D
    K_eff, C_eff = sae.K_eff, sae.C_eff
    W_dec = sae.decoder.get_W().detach()

    config = PredSelectorConfig(
        d_sel=psc["d_sel"], mlp_hidden=psc["mlp_hidden"], n_global_pred=psc["n_global_pred"],
        n_per_channel=psc.get("n_per_channel"), gumbel_topk_noise=psc.get("gumbel_topk_noise", 0.0),
        separated_topk=psc.get("separated_topk", False),
        lambda_logit=psc["lambda_logit"], lambda_comp=psc["lambda_comp"], delta_comp=psc["delta_comp"],
        lambda_prior=psc.get("lambda_prior", 0.0), prior_tau=psc.get("prior_tau", 1.0),
        lr=psc["lr"], weight_decay=psc["weight_decay"],
        epochs=args.epochs if args.epochs is not None else psc["epochs"],
        patience=psc["patience"], batch_size=psc["batch_size"])

    n_island_features = 6 + 2 * V if include_obs_profile else 6 + V
    selector = PredSelector(config, n_island_features, D, C_eff).to(device)
    logger.write(f"AsymTISAE PS: {sum(p.numel() for p in selector.parameters()):,} params, "
                 f"n_features={n_island_features}, C_eff={C_eff}, a_max={asym_cfgd.get('asym_a_max')}")

    train_loader = DataLoader(PSDataset(dataset, 'train'), batch_size=config.batch_size, shuffle=True,
                              collate_fn=ps_collate_fn, num_workers=2, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(PSDataset(dataset, 'val'), batch_size=config.batch_size, shuffle=False,
                            collate_fn=ps_collate_fn, num_workers=2, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    full_config = {
        'pred_selector_config': asdict(config), 'sae_experiment_dir': args.sae_experiment_dir,
        'tisae_config': asdict(cfg), 'n_island_features': n_island_features,
        'D': D, 'K_eff': K_eff, 'C_eff': C_eff, 'M': cfg.M, 'T': T, 'V': V,
        'n_prefilter': n_prefilter, 'n_pool_full': n_pool_full, 'max_grad_norm': max_grad_norm,
        'include_obs_profile': include_obs_profile, 'soft_selection': soft_selection, 'soft_tau': soft_tau,
        'model_type': 'asym_tisae',
    }
    with open(out / 'config.yaml', 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False)

    prior_cache = None  # canonical lambda_prior=0 -> grad-act prior inert -> equivalent
    kw = dict(n_prefilter=n_prefilter, n_pool_full=n_pool_full,
              include_obs_profile=include_obs_profile, soft_selection=soft_selection, soft_tau=soft_tau)
    best_val = float('inf')
    patience_counter = 0
    for epoch in range(1, config.epochs + 1):
        logger.write(f"\n--- PS Epoch {epoch}/{config.epochs} ---")
        tm = train_epoch(adapter, sae, selector, train_loader, optimizer, T, V,
                         H_grid_mean, W_dec, prior_cache, config, device,
                         max_grad_norm=max_grad_norm, **kw)
        vm = evaluate(adapter, sae, selector, val_loader, T, V,
                      H_grid_mean, W_dec, prior_cache, config, device, **kw)
        logger.write(f"Train L_sel={tm['L_sel']:.4f} L_logit={tm.get('L_logit', 0):.4f} | "
                     f"Val L_sel={vm['L_sel']:.4f}")
        vl = vm['L_sel']
        if vl < best_val:
            best_val = vl
            patience_counter = 0
            torch.save({'selector_state_dict': selector.state_dict(),
                        'config': asdict(config), 'epoch': epoch, 'val_L_sel': vl},
                       out / 'best_model.pt')
            logger.write(f"  * New best val L_sel={vl:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.write(f"Early stopping at epoch {epoch}")
                break
    logger.write(f"\nBest val L_sel: {best_val:.6f}\nOutput: {out}")


if __name__ == "__main__":
    main()
