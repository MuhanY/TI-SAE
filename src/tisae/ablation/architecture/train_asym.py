"""Stage-A trainer for the asymmetric TI-SAE ablation.

Builds AsymTISAE from the canonical config so every hyperparameter is identical
except the island shape, then reuses the canonical training loop.

Usage:
  python -m tisae.ablation.architecture.train_asym \
    --backbone_checkpoint experiments/discrete/mimic3_mortality/backbone/best_model.pt \
    --dataset_path data/unified_mimic3_48h.pkl \
    --canonical_config experiments/discrete/mimic3_mortality/tisae/config.yaml \
    --output_dir experiments/ablation/island_shape/learned/tisae \
    --asym_a_max 0.7
"""
import argparse
import yaml
import torch
from pathlib import Path
from dataclasses import fields, asdict

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.tisae import TISAEConfig
from tisae.models.ontology import create_ontology_mapping
from tisae.training.train_tisae import (
    TISAETrainer, TISAEDataset, precompute_H_grid_mean, Logger,
    train_epoch, evaluate, save_checkpoint,
)
from tisae.ablation.architecture.asym_tisae import AsymTISAE


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_checkpoint", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--canonical_config", required=True,
                    help="seed-42 tisae config.yaml for hyperparameter equivalence")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--asym_a_max", type=float, default=0.7)
    ap.add_argument("--fix_a", type=float, default=None,
                    help="impose constant skew a (fixed-shape control); None => learned per-island")
    ap.add_argument("--epochs", type=int, default=None, help="override canonical epochs")
    ap.add_argument("--seed", type=int, default=None, help="override canonical seed")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    canon = yaml.safe_load(open(args.canonical_config))
    ca = canon.get("args", {})
    ct = canon.get("training", {})
    tcfg = canon.get("tisae_config", {})

    seed = args.seed if args.seed is not None else ca.get("seed", 42)
    epochs = args.epochs if args.epochs is not None else ca.get("epochs", 100)
    lr = ct.get("lr", ca.get("lr", 1e-4))
    batch_size = ct.get("batch_size", ca.get("batch_size", 32))
    weight_decay = ct.get("weight_decay", ca.get("weight_decay", 0.01))
    patience = ct.get("patience", ca.get("patience", 15))
    extract_layer = ca.get("extract_layer", -1)
    lambda_pred = ca.get("lambda_pred", 0.0)
    gradient_checkpointing = ca.get("gradient_checkpointing", False)

    set_seed(seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(out))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.write(f"AsymTISAE (split-normal). seed={seed} epochs={epochs} "
                 f"lr={lr} bs={batch_size} wd={weight_decay} a_max={args.asym_a_max}")

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V,
          "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    T, V = dataset.T, dataset.V
    train_loader = torch.utils.data.DataLoader(
        TISAEDataset(dataset, "train"), batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        worker_init_fn=lambda w: set_seed(seed + w))
    val_loader = torch.utils.data.DataLoader(
        TISAEDataset(dataset, "val"), batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers)

    adapter = load_backbone_adapter(args.backbone_checkpoint, device=str(device), dataset_meta=dm)
    adapter.set_extract_layer(extract_layer)
    D = adapter.D
    logger.write(f"Backbone: D={D}, T={T}, V={V}, extract_layer={extract_layer}")

    valid = {f.name for f in fields(TISAEConfig)}
    cfg = TISAEConfig(**{k: v for k, v in tcfg.items() if k in valid})
    cfg.D, cfg.T, cfg.V = D, T, V  # ensure match with data/adapter

    channel_var_mask = None
    if cfg.lambda_ch_purity > 0:
        ontology = create_ontology_mapping(dataset.var_names, n_channels=cfg.C)
        channel_var_mask = ontology.get_hard_mask()

    sae = AsymTISAE(cfg, channel_var_mask=channel_var_mask,
                    asym_a_max=args.asym_a_max, fix_a=args.fix_a).to(device)
    model = TISAETrainer(adapter, sae, T, V, extract_layer=extract_layer,
                         lambda_pred=lambda_pred,
                         gradient_checkpointing=gradient_checkpointing).to(device)

    H_grid_mean = precompute_H_grid_mean(adapter, train_loader, device, T, V,
                                         extract_layer=extract_layer)
    model.register_buffer('H_grid_mean', H_grid_mean)
    adapter.set_residualization_mean(H_grid_mean)
    logger.write(f"AsymTISAE params: {sum(p.numel() for p in sae.parameters()):,} "
                 f"(fc_asym adds {sum(p.numel() for p in sae.island_params.fc_asym.parameters()):,})")

    config_to_save = {
        'args': vars(args),
        'tisae_config': asdict(cfg),
        'model_type': 'asym_tisae',
        'asym_a_max': args.asym_a_max,
        'fix_a': args.fix_a,
        'training': {'residualize': True, 'residualize_mode': 'v',
                     'lr': lr, 'batch_size': batch_size, 'weight_decay': weight_decay,
                     'epochs': epochs, 'patience': patience, 'lambda_pred': lambda_pred},
    }
    with open(out / 'config.yaml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False)

    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float('inf')
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        logger.write(f"\n--- Epoch {epoch}/{epochs} ---")
        tm = train_epoch(model, train_loader, optimizer, device, args, compute_grads=(epoch == 1))
        vm = evaluate(model, val_loader, device, compute_fidelity=False)
        logger.write(f"Train: loss={tm['loss']:.4f} recon={tm['L_recon']:.6f} "
                     f"n_sel={tm.get('n_selected', 0):.1f}")
        logger.write(f"Val:   loss={vm['loss']:.4f} recon={vm['L_recon']:.6f}")
        vl = vm['loss']
        if vl < best_val:
            best_val = vl
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, {'val_loss': vl, **vm}, out / 'best_model.pt')
            logger.write(f"  * New best val_loss={vl:.6f}")
        else:
            patience_counter += 1
            logger.write(f"  No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            logger.write(f"Early stopping at epoch {epoch}")
            break

    logger.write(f"\nBest val_loss: {best_val:.6f}\nOutput: {out}")


if __name__ == "__main__":
    main()
