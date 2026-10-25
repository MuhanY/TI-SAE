"""Training script for ablation model variants.

Handles two model types:
  - tisae_no_temporal: TISAENoTemporal (grid-level, flat temporal mask)
  - simple_sae: SimpleSAE (event-level, no grid)

For tisae_no_temporal, reuses TISAETrainer from train_tisae.py.
For simple_sae, uses a lightweight trainer.

Usage:
    # No temporal decomposition
    python -m tisae.ablation.architecture.train --model tisae_no_temporal \\
        --backbone_checkpoint <path> --dataset_path <path> \\
        --output_dir experiments_ablation/structural/no_temporal/tisae

    # Standard event-level SAE
    python -m tisae.ablation.architecture.train --model simple_sae \\
        --backbone_checkpoint <path> --dataset_path <path> \\
        --output_dir experiments_ablation/structural/simple_sae/model
"""

import argparse
import math
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from dataclasses import fields, asdict
from tqdm import tqdm

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.tisae import TISAEConfig
from tisae.models.ontology import create_ontology_mapping
from tisae.training.train_tisae import (
    TISAETrainer, TISAEDataset, precompute_H_grid_mean, Logger,
    train_epoch, evaluate, save_checkpoint,
)

from tisae.ablation.architecture.tisae_no_temporal import TISAENoTemporal
from tisae.ablation.architecture.simple_sae import SimpleSAE, SimpleSAEConfig


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# SimpleSAE Trainer
# =========================================================================

class SimpleSAETrainer(nn.Module):
    """Training wrapper for SimpleSAE: frozen backbone + trainable SAE."""

    def __init__(self, adapter, sae: SimpleSAE, T: int, V: int):
        super().__init__()
        self.adapter = adapter
        self.sae = sae
        self.T = T
        self.V = V
        self.adapter.freeze()

    def forward(self, batch, compute_fidelity=False):
        batch_dev = batch
        H_grid, obs_grid, H = self.adapter.extract_hidden(batch_dev, return_event_h=True)
        obs_mask = batch_dev["obs_mask"]

        # No residualization for simple_sae — raw H[B,L,D]
        H_input = H * obs_mask.unsqueeze(-1)

        sae_out = self.sae(H_input, obs_mask)

        H_hat = sae_out['H_hat']
        H_hat_full = H_hat

        cos_H = torch.tensor(0.0, device=H.device)
        if compute_fidelity:
            with torch.no_grad():
                cos_all = F.cosine_similarity(H_hat_full, H, dim=-1)
                cos_H = (cos_all * obs_mask).sum() / obs_mask.sum().clamp_min(1.0)

        cfg = self.sae.cfg
        loss = sae_out['loss_total'] + cfg.lambda_auxk * sae_out['L_auxk']

        return {
            'loss': loss,
            'L_recon': sae_out['L_recon'],
            'L_recon_H': sae_out['L_recon'],
            'L_auxk': sae_out['L_auxk'],
            'L_auxk_combined': sae_out['L_auxk'],
            'cos_H': cos_H,
            'n_dead': sae_out['n_dead'],
            'n_selected': torch.tensor(float(sae_out.get('n_active', 0))),
            # Compat keys (unused but expected by train_epoch)
            'loss_Z': torch.tensor(0.0),
            'L_pred': torch.tensor(0.0),
            'KL_pred': torch.tensor(0.0),
            'L_auxk_pred': torch.tensor(0.0),
            'KL_auxk_pred': torch.tensor(0.0),
            'L_repulsion': torch.tensor(0.0),
            'L_div_W': torch.tensor(0.0),
            'L_a0_contrast': torch.tensor(0.0),
            'L_mask_coverage': torch.tensor(0.0),
            'L_ch_purity': torch.tensor(0.0),
            'mu_var_batch': torch.tensor(0.0),
            'mu_std_median': torch.tensor(0.0),
            'mu_std_pct_low': torch.tensor(0.0),
            'mu_std_sorted_median': torch.tensor(0.0),
            'mu_std_sorted_pct_low': torch.tensor(0.0),
        }


def precompute_H_event_mean(adapter, dataloader, device, max_batches=100):
    """Compute per-variable population mean from event-level H."""
    adapter.eval()
    V = adapter.V
    D = adapter.D
    H_sum = torch.zeros(V, D, device=device)
    obs_count = torch.zeros(V, device=device)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _, _, H = adapter.extract_hidden(batch, return_event_h=True)
            varis = batch["varis"]
            obs_mask = batch["obs_mask"]
            for b in range(H.shape[0]):
                valid = obs_mask[b].bool()
                v_idx = varis[b, valid].long()
                H_sum.index_add_(0, v_idx, H[b, valid])
                obs_count.index_add_(0, v_idx, torch.ones(valid.sum(), device=device))

    return H_sum / obs_count.unsqueeze(-1).clamp_min(1)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["tisae_no_temporal", "simple_sae"])
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)

    # Shared hyperparams
    parser.add_argument("--K", type=int, default=1280)
    parser.add_argument("--n_global", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # TISAE-specific (for tisae_no_temporal)
    parser.add_argument("--W_sparsity_k", type=int, default=3)
    parser.add_argument("--encoder_sparsity_k", type=int, default=5)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_auxk", type=float, default=0.3)
    parser.add_argument("--auxk_k", type=int, default=256)
    parser.add_argument("--lambda_div_W", type=float, default=0.01)
    parser.add_argument("--lambda_ch_purity", type=float, default=0.1)
    parser.add_argument("--C", type=int, default=10)
    parser.add_argument("--normalize_atoms", action="store_true", default=True)
    parser.add_argument("--dead_boost_scale", type=float, default=0.5)
    parser.add_argument("--dead_threshold_batches", type=int, default=5)
    parser.add_argument("--gumbel_topk_noise", type=float, default=0.05)
    parser.add_argument("--extract_layer", type=int, default=-1)

    # SimpleSAE-specific
    parser.add_argument("--top_k", type=int, default=32)

    args = parser.parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.write(f"Model: {args.model}, Device: {device}")

    # Load dataset
    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V, "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    T, V = dataset.T, dataset.V

    train_loader = torch.utils.data.DataLoader(
        TISAEDataset(dataset, "train"),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        worker_init_fn=lambda w: set_seed(args.seed + w),
    )
    val_loader = torch.utils.data.DataLoader(
        TISAEDataset(dataset, "val"),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # Load backbone
    adapter = load_backbone_adapter(args.backbone_checkpoint, device=str(device), dataset_meta=dm)
    adapter.set_extract_layer(args.extract_layer)
    D = adapter.D
    logger.write(f"Backbone loaded: D={D}, T={T}, V={V}")

    if args.model == "tisae_no_temporal":
        cfg = TISAEConfig(
            D=D, K=args.K, T=T, V=V, M=1,
            W_sparsity_k=args.W_sparsity_k,
            encoder_sparsity_k=args.encoder_sparsity_k,
            encoder_chunk_size=32,
            n_global=args.n_global,
            separated_topk=False,
            restrict_to_channels=False,
            C=args.C,
            normalize_atoms=args.normalize_atoms,
            lambda_recon=args.lambda_recon,
            lambda_pred=0.0,
            lambda_auxk=args.lambda_auxk,
            auxk_k=args.auxk_k,
            lambda_mu_repulsion=0.0,
            lambda_div_W=args.lambda_div_W,
            lambda_mask_coverage=0.0,
            lambda_ch_purity=args.lambda_ch_purity,
            score_mode="energy_gated",
            gumbel_topk_noise=args.gumbel_topk_noise,
            dead_boost_scale=args.dead_boost_scale,
            dead_threshold_batches=args.dead_threshold_batches,
        )

        channel_var_mask = None
        if cfg.lambda_ch_purity > 0:
            ontology = create_ontology_mapping(dataset.var_names, n_channels=cfg.C)
            channel_var_mask = ontology.get_hard_mask()

        sae = TISAENoTemporal(cfg, channel_var_mask=channel_var_mask).to(device)
        model = TISAETrainer(adapter, sae, T, V, extract_layer=args.extract_layer)

        H_grid_mean = precompute_H_grid_mean(adapter, train_loader, device, T, V)
        model.register_buffer('H_grid_mean', H_grid_mean)
        adapter.set_residualization_mean(H_grid_mean)
        logger.write(f"TISAENoTemporal: K={cfg.K}, n_global={cfg.n_global}, W_k={cfg.W_sparsity_k}")
        logger.write(f"  Params: {sum(p.numel() for p in sae.parameters()):,}")

        config_to_save = {
            'args': vars(args),
            'tisae_config': asdict(cfg),
            'model_type': 'tisae_no_temporal',
            'training': {
                'residualize': True, 'residualize_mode': 'v',
                'lr': args.lr, 'batch_size': args.batch_size,
                'weight_decay': args.weight_decay, 'epochs': args.epochs,
                'patience': args.patience,
            },
        }

    elif args.model == "simple_sae":
        cfg = SimpleSAEConfig(
            D=D, K=args.K, top_k=args.top_k,
            lambda_recon=args.lambda_recon,
            lambda_auxk=args.lambda_auxk,
            auxk_k=args.auxk_k,
            dead_threshold_batches=args.dead_threshold_batches,
        )
        sae = SimpleSAE(cfg).to(device)
        model = SimpleSAETrainer(adapter, sae, T, V)
        # No residualization for simple_sae — operates on raw H[B,L,D]
        logger.write(f"SimpleSAE: K={cfg.K}, mode={cfg.sparsity_mode}, "
                      f"top_k={cfg.top_k} (no residualization)")
        logger.write(f"  Params: {sum(p.numel() for p in sae.parameters()):,}")

        config_to_save = {
            'args': vars(args),
            'simple_sae_config': asdict(cfg),
            'model_type': 'simple_sae',
            'training': {
                'lr': args.lr, 'batch_size': args.batch_size,
                'weight_decay': args.weight_decay, 'epochs': args.epochs,
                'patience': args.patience,
            },
        }

    model = model.to(device)

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False)

    optimizer = torch.optim.AdamW(
        sae.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Post-step decoder unit-norm constraint (standard in SAE literature:
    # Gao et al. 2024, Bricken et al. 2023, Rajamanoharan et al. 2024)
    if args.model == 'simple_sae' and hasattr(sae, 'W_dec'):
        def _normalize_decoder(opt, args=None, kwargs=None):
            with torch.no_grad():
                sae.W_dec.data = F.normalize(sae.W_dec.data, p=2, dim=-1)
        optimizer.register_step_post_hook(_normalize_decoder)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        logger.write(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_metrics = train_epoch(model, train_loader, optimizer, device, args,
                                     compute_grads=(epoch == 1))
        val_metrics = evaluate(model, val_loader, device, compute_fidelity=False)

        logger.write(f"Train: loss={train_metrics['loss']:.4f} recon={train_metrics['L_recon']:.6f} "
                      f"n_sel={train_metrics.get('n_selected', 0):.1f}")
        logger.write(f"Val:   loss={val_metrics['loss']:.4f} recon={val_metrics['L_recon']:.6f}")

        if epoch % 5 == 0:
            fid_metrics = evaluate(model, val_loader, device, compute_fidelity=True)
            logger.write(f"  cos_H={fid_metrics.get('cos_H', 0):.4f}")

        val_loss = val_metrics['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch,
                           {'val_loss': val_loss, **val_metrics},
                           output_dir / 'best_model.pt')
            logger.write(f"  * New best val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            logger.write(f"  No improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            logger.write(f"Early stopping at epoch {epoch}")
            break

    logger.write(f"\nBest val_loss: {best_val_loss:.6f}")
    logger.write(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
