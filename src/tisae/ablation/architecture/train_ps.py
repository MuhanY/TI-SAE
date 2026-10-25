"""PredSelector training for ablation model variants.

Handles:
  - tisae_no_temporal: uses synthetic flat island features → standard PS
  - simple_sae: uses per-latent aggregate features → standard PS architecture

Usage:
    python -m tisae.ablation.architecture.train_ps \\
        --sae_experiment_dir experiments_ablation/structural/no_temporal/tisae \\
        --dataset_path data/unified_mimic3_48h.pkl \\
        --output_dir experiments_ablation/structural/no_temporal/ps
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from tqdm import tqdm

from tisae.models.pred_selector import (
    PredSelector, PredSelectorConfig,
    assemble_island_features, pred_selector_losses,
)
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.training.train_tisae import Logger, TISAEDataset
from tisae.utils.seeding import set_seed, worker_init_fn

from tisae.ablation.architecture.eval import load_ablation_model


# =========================================================================
# Feature assembly for SimpleSAE
# =========================================================================

@torch.no_grad()
def assemble_latent_features_simple(
    z: torch.Tensor,       # [B, L, K]
    obs_mask: torch.Tensor, # [B, L]
    varis: torch.Tensor,    # [B, L]
    V: int,
    include_obs_profile: bool = True,
) -> torch.Tensor:
    """Assemble per-latent features from SimpleSAE activations.

    Produces features compatible with PredSelector (same n_features = 6 + 2*V).
    Shape: [B, K, 1, 1, 6 + 2*V]

    Scalar features [6]:
        0: mean activation (across valid tokens)
        1: max activation
        2: activation std
        3: frequency (fraction of valid tokens where z>0)
        4: 0 (gate placeholder)
        5: 0 (prior placeholder)

    Variable profile [V]:
        For each variable v: mean z across tokens of variable v.

    Observation profile [V]:
        For each variable v: fraction of total tokens that are variable v.
    """
    B, L, K = z.shape
    device = z.device

    valid = obs_mask.unsqueeze(-1)  # [B, L, 1]
    z_valid = z * valid  # [B, L, K]
    n_valid = obs_mask.sum(dim=1, keepdim=True).clamp_min(1)  # [B, 1]

    mean_act = z_valid.sum(dim=1) / n_valid  # [B, K]
    max_act = z_valid.max(dim=1).values      # [B, K]
    z_sq = (z_valid ** 2).sum(dim=1) / n_valid
    std_act = (z_sq - mean_act ** 2).clamp_min(0).sqrt()  # [B, K]
    freq = ((z_valid > 0).float().sum(dim=1)) / n_valid  # [B, K]
    zeros = torch.zeros(B, K, device=device)

    def _batch_norm(x):
        bmax = x.detach().amax(dim=1, keepdim=True).clamp_min(1e-8)
        return x / bmax

    scalars = torch.stack([
        _batch_norm(mean_act),
        _batch_norm(max_act),
        _batch_norm(std_act),
        freq,
        zeros,
        zeros,
    ], dim=-1)  # [B, K, 6]

    # Variable profile: for each (b, k, v), mean z across tokens of variable v
    # Vectorized with scatter_add (replaces slow nested Python loop)
    var_profile = torch.zeros(B, K, V, device=device)
    var_count = torch.zeros(B, V, device=device)
    for b in range(B):
        m = obs_mask[b].bool()
        v_idx = varis[b, m].long().clamp(0, V - 1)  # [n_valid]
        z_b = z_valid[b, m]  # [n_valid, K]
        n_b = m.sum()
        var_count[b].index_add_(0, v_idx, torch.ones(n_b, device=device))
        # scatter_add z values by variable index, then divide by count
        v_idx_exp = v_idx.unsqueeze(1).expand(n_b, K)  # [n_valid, K]
        var_sum = torch.zeros(V, K, device=device)
        var_sum.scatter_add_(0, v_idx_exp, z_b)  # [V, K]
        vc = var_count[b].clamp_min(1)  # [V]
        var_profile[b] = (var_sum / vc.unsqueeze(1)).t()  # [K, V]
    var_profile = var_profile / var_profile.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    parts = [scalars, var_profile]

    if include_obs_profile:
        obs_profile = var_count / n_valid  # [B, V]
        obs_profile = obs_profile.unsqueeze(1).expand(B, K, V)
        parts.append(obs_profile)

    features = torch.cat(parts, dim=-1)  # [B, K, 6 + V (+ V)]
    return features.unsqueeze(2).unsqueeze(2)  # [B, K, 1, 1, n_features]


# =========================================================================
# Training step functions
# =========================================================================

def ps_step_no_temporal(
    adapter, sae, selector, batch, T, V, H_grid_mean, W_dec,
    config, device, include_obs_profile=True, soft_tau=1.0,
):
    """PS training step for TISAENoTemporal — synthetic flat island features."""
    K_eff, C_eff = sae.K_eff, sae.C_eff
    cfg = sae.cfg

    batch_dev = {k: v.to(device) for k, v in batch.items()}
    obs_mask = batch_dev['obs_mask']
    B = obs_mask.size(0)

    with torch.no_grad():
        H_grid, obs_grid, H = adapter.extract_hidden(batch_dev, return_event_h=True)
        if adapter.extract_layer != -1:
            out_full = adapter.forward(batch_dev, return_pooled=True)
            h_pool = out_full['pooled']
            z_teacher = out_full['logits']
        else:
            h_pool, _ = adapter.pool_events(H, obs_mask)
            z_teacher = adapter.predict_from_pooled(h_pool)

        if H_grid_mean is not None:
            H_grid_input = (H_grid - H_grid_mean.unsqueeze(0).unsqueeze(1)) * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

        out = sae(H_grid_input, obs_grid)
        A0_4d = out['A0'].unsqueeze(-1) if C_eff == 1 else out['A0'].view(B, C_eff, K_eff, T).permute(0, 2, 3, 1)
        g = out['g']
        mu = out['mu']
        sigma = out['sigma']
        gate = out['gate']

        features = assemble_island_features(
            A0_4d, g, mu, sigma, gate, obs_grid, W_dec,
            None, K_eff, C_eff, include_obs_profile=include_obs_profile)

        # Full reconstruction prediction
        H_hat_grid = out['H_hat_grid']
        z_full = adapter.predict_from_grid(H_hat_grid, batch_dev, obs_grid)

    # PredSelector forward (trainable)
    M = 1
    s_pred = selector(features, h_pool.detach())  # [B, K_eff, C_eff, M]

    s_flat = s_pred.view(B, -1)
    weights = F.softmax(s_flat / soft_tau, dim=-1) * config.n_global_pred
    sel_pred = weights.view(B, K_eff, C_eff, M)

    # Flat mask (no Gaussian modulation)
    mask_pred = sel_pred.permute(0, 1, 3, 2).expand(B, K_eff, T, C_eff)
    A_hat_keep_4d = A0_4d * mask_pred
    if C_eff > 1:
        A_hat_keep = A_hat_keep_4d.permute(0, 3, 1, 2).reshape(B, cfg.K, T)
    else:
        A_hat_keep = A_hat_keep_4d.squeeze(-1)
    H_hat_keep = sae.decoder(A_hat_keep)
    z_keep = adapter.predict_from_grid(H_hat_keep, batch_dev, obs_grid)

    # Drop complement
    A_hat_drop_4d = A0_4d * (1 - mask_pred).clamp_min(0)
    if C_eff > 1:
        A_hat_drop = A_hat_drop_4d.permute(0, 3, 1, 2).reshape(B, cfg.K, T)
    else:
        A_hat_drop = A_hat_drop_4d.squeeze(-1)
    H_hat_drop = sae.decoder(A_hat_drop)
    z_drop = adapter.predict_from_grid(H_hat_drop, batch_dev, obs_grid)

    losses = pred_selector_losses(
        z_keep, z_drop, z_full, z_teacher,
        s_pred, None, config,
    )
    return losses


def ps_step_simple_sae(
    adapter, sae, selector, batch, T, V, H_mean,
    config, device, include_obs_profile=True, soft_tau=1.0,
):
    """PS training step for SimpleSAE — per-latent aggregate features.

    No residualization — operates on raw H[B,L,D].
    """
    K = sae.cfg.K

    batch_dev = {k: v.to(device) for k, v in batch.items()}
    obs_mask = batch_dev['obs_mask']
    varis = batch_dev['varis']
    tau = batch_dev['tau']
    B = obs_mask.size(0)

    with torch.no_grad():
        _, _, H = adapter.extract_hidden(batch_dev, return_event_h=True)
        if adapter.extract_layer != -1:
            out_full = adapter.forward(batch_dev, return_pooled=True)
            h_pool = out_full['pooled']
            z_teacher = out_full['logits']
        else:
            h_pool, _ = adapter.pool_events(H, obs_mask)
            z_teacher = adapter.predict_from_pooled(h_pool)

        H_input = H * obs_mask.unsqueeze(-1)
        z = sae.encode(H_input)  # [B, L, K]

        # Full reconstruction prediction (no residualization)
        H_hat_full = sae.decode(z)
        pooled_full, _ = adapter.pool_events(H_hat_full, obs_mask)
        z_full = adapter.predict_from_pooled(pooled_full)

        features = assemble_latent_features_simple(
            z, obs_mask, varis, V, include_obs_profile=include_obs_profile)

    # PredSelector forward (trainable)
    s_pred = selector(features, h_pool.detach())  # [B, K, 1, 1]

    s_flat = s_pred.view(B, -1)  # [B, K]
    weights = F.softmax(s_flat / soft_tau, dim=-1) * config.n_global_pred  # [B, K]

    # Weighted reconstruction: keep (no residualization)
    z_weighted = z * weights.unsqueeze(1)  # [B, L, K]
    H_hat_keep = sae.decode(z_weighted)
    pooled_keep, _ = adapter.pool_events(H_hat_keep, obs_mask)
    z_keep = adapter.predict_from_pooled(pooled_keep)

    # Drop complement (no residualization)
    z_drop = z * (1 - weights).clamp_min(0).unsqueeze(1)
    H_hat_drop = sae.decode(z_drop)
    pooled_drop, _ = adapter.pool_events(H_hat_drop, obs_mask)
    z_drop_logit = adapter.predict_from_pooled(pooled_drop)

    losses = pred_selector_losses(
        z_keep, z_drop_logit, z_full, z_teacher,
        s_pred, None, config,
    )
    return losses


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_experiment_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--d_sel", type=int, default=64)
    parser.add_argument("--mlp_hidden", type=int, default=64)
    parser.add_argument("--n_global_pred", type=int, default=64)
    parser.add_argument("--lambda_logit", type=float, default=1.0)
    parser.add_argument("--lambda_comp", type=float, default=0.1)
    parser.add_argument("--delta_comp", type=float, default=0.5)
    parser.add_argument("--lambda_prior", type=float, default=0.0)
    parser.add_argument("--soft_tau", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=52)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir)

    # Load ablation model
    loaded = load_ablation_model(args.sae_experiment_dir, args.dataset_path, str(device))
    model_type = loaded['model_type']
    sae = loaded['model']
    adapter = loaded['adapter']
    dataset = loaded['dataset']
    T, V = loaded['T'], loaded['V']
    D = adapter.D

    for p in sae.parameters():
        p.requires_grad = False
    sae.eval()

    logger.write(f"PS training for ablation model: {model_type}")

    # Determine feature dimensions
    if model_type == 'tisae_no_temporal':
        K_eff, C_eff = sae.K_eff, sae.C_eff
        W_dec = sae.decoder.get_W().detach()
        H_grid_mean = loaded.get('H_grid_mean')
        n_island_features = 6 + 2 * V
        step_fn = lambda batch: ps_step_no_temporal(
            adapter, sae, selector, batch, T, V, H_grid_mean, W_dec,
            ps_config, device, soft_tau=args.soft_tau)
    elif model_type == 'simple_sae':
        K_eff = sae.cfg.K
        C_eff = 1
        H_mean = loaded.get('H_mean')
        n_island_features = 6 + 2 * V
        step_fn = lambda batch: ps_step_simple_sae(
            adapter, sae, selector, batch, T, V, H_mean,
            ps_config, device, soft_tau=args.soft_tau)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Build PredSelector
    ps_config = PredSelectorConfig(
        d_sel=args.d_sel, mlp_hidden=args.mlp_hidden,
        n_global_pred=args.n_global_pred,
        lambda_logit=args.lambda_logit, lambda_comp=args.lambda_comp,
        delta_comp=args.delta_comp, lambda_prior=args.lambda_prior,
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size,
    )
    selector = PredSelector(ps_config, n_island_features, D, C_eff).to(device)
    n_params = sum(p.numel() for p in selector.parameters() if p.requires_grad)
    logger.write(f"PredSelector: {n_params:,} params, n_features={n_island_features}, D={D}, C_eff={C_eff}")

    # Re-bind step_fn now that selector and ps_config exist
    if model_type == 'tisae_no_temporal':
        step_fn = lambda batch: ps_step_no_temporal(
            adapter, sae, selector, batch, T, V, H_grid_mean, W_dec,
            ps_config, device, soft_tau=args.soft_tau)
    elif model_type == 'simple_sae':
        step_fn = lambda batch: ps_step_simple_sae(
            adapter, sae, selector, batch, T, V, H_mean,
            ps_config, device, soft_tau=args.soft_tau)

    # Data loaders
    train_ds = TISAEDataset(dataset, "train")
    val_ds = TISAEDataset(dataset, "val")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
        worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
        worker_init_fn=worker_init_fn)

    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Save config
    full_config = {
        'pred_selector_config': asdict(ps_config),
        'sae_experiment_dir': args.sae_experiment_dir,
        'model_type': model_type,
        'n_island_features': n_island_features,
        'D': D, 'K_eff': K_eff, 'C_eff': C_eff, 'T': T, 'V': V,
    }
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        selector.train()
        train_metrics = defaultdict(float)
        n_train = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            losses = step_fn(batch)
            loss = losses['L_sel']

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), max_norm=1.0)
            optimizer.step()

            B = batch['value_bins'].size(0)
            n_train += B
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    train_metrics[k] += v.item() * B

        # Validate
        selector.eval()
        val_metrics = defaultdict(float)
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                losses = step_fn(batch)
                B = batch['value_bins'].size(0)
                n_val += B
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        val_metrics[k] += v.item() * B

        train_str = " | ".join(f"{k}={v/n_train:.4f}" for k, v in sorted(train_metrics.items()))
        val_str = " | ".join(f"{k}={v/n_val:.4f}" for k, v in sorted(val_metrics.items()))
        logger.write(f"\n--- Epoch {epoch}/{args.epochs} ---")
        logger.write(f"Train: {train_str}")
        logger.write(f"Val:   {val_str}")

        val_loss = val_metrics.get('L_sel', float('inf')) / max(n_val, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'selector_state_dict': selector.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pt')
            logger.write(f"  * New best val L_sel={val_loss:.6f}")
        else:
            patience_counter += 1
            logger.write(f"  No improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            logger.write(f"Early stopping at epoch {epoch}")
            break

    logger.write(f"\nBest val L_sel: {best_val_loss:.6f}")
    logger.write(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
