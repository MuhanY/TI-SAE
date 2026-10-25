# tisae/training/train_pred_selector.py
"""
Train PredSelector on a frozen TISAE + frozen backbone.

Usage:
    python -m tisae.training.train_pred_selector \
        --sae_experiment_dir experiments/discrete/mimic3_mortality/tisae \
        --dataset_path data/unified_mimic3_48h.pkl \
        --output_dir experiments/discrete/mimic3_mortality/ps \
        --n_global_pred 64 --epochs 100 --patience 20 --lr 1e-3
"""

import sys
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tisae.models.pred_selector import (
    PredSelector, PredSelectorConfig,
    assemble_island_features, pred_selector_losses,
)
from tisae.models.tisae import TISAE, TISAEConfig
from tisae.models.backbone_adapter import (
    BackboneAdapter,
    load_backbone_adapter,
)
from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.training.train_tisae import TISAEDataset
from tisae.utils.seeding import set_seed, worker_init_fn


# =============================================================================
# Logger
# =============================================================================

class Logger:
    def __init__(self, log_dir, filename='log.txt'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename

    def write(self, msg: str):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')


# =============================================================================
# Dataset wrapper that returns original indices
# =============================================================================

class PSDataset(TISAEDataset):
    """TISAEDataset that also returns the original dataset index."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['orig_idx'] = self.indices[idx]
        return item


def ps_collate_fn(batch):
    """Collate that handles orig_idx."""
    orig_indices = [b.pop('orig_idx') for b in batch]
    collated = collate_fn(batch)
    collated['orig_idx'] = torch.tensor(orig_indices, dtype=torch.long)
    return collated


# =============================================================================
# Model loading
# =============================================================================

def load_tisae_models(experiment_dir, dataset_path, device='cuda'):
    """Load frozen BackboneAdapter + TISAE from experiment directory.

    Returns dict with:
        adapter, backbone, sae, dataset, tisae_config, H_grid_mean
    """
    exp_dir = Path(experiment_dir)
    device = torch.device(device)

    # Load config
    with open(exp_dir / 'config.yaml') as f:
        config = yaml.safe_load(f)

    tisae_cfg_dict = config.get('tisae_config', config.get('sae_config', {}))
    valid_fields = {f.name for f in fields(TISAEConfig)}
    tisae_cfg = TISAEConfig(**{k: v for k, v in tisae_cfg_dict.items() if k in valid_fields})

    dataset = UnifiedTimeSeriesDataset.load(dataset_path)

    # Load backbone via the adapter factory
    bb_path = config.get('args', {}).get('backbone_checkpoint')
    if bb_path is None:
        bb_path = config.get('training', {}).get('backbone_checkpoint')
    if bb_path is None:
        raise ValueError("Cannot find backbone_checkpoint in config.yaml")

    dataset_meta = {
        'T': dataset.T,
        'V': dataset.V,
        'var_to_ind_hash': getattr(dataset, 'var_to_ind_hash', None),
    }
    adapter = load_backbone_adapter(bb_path, device=str(device), dataset_meta=dataset_meta)

    # Build channel mask if needed
    channel_var_mask = None
    if tisae_cfg.restrict_to_channels or tisae_cfg.lambda_ch_purity > 0:
        from tisae.models.ontology import create_ontology_mapping
        ontology = create_ontology_mapping(dataset.var_names, n_channels=tisae_cfg.C)
        channel_var_mask = ontology.get_hard_mask()

    # Create TISAE
    sae = TISAE(tisae_cfg, channel_var_mask=channel_var_mask)

    # Load checkpoint
    ckpt_path = exp_dir / 'best_model.pt'
    if not ckpt_path.exists():
        ckpt_path = exp_dir / 'final_model.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sae_sd = ckpt['sae_state_dict']
    # Strip FSDP .module. prefix if present
    sae_sd = {k.replace('.module.', '.'): v for k, v in sae_sd.items()}
    sae.load_state_dict(sae_sd)
    sae = sae.to(device).eval()

    H_grid_mean = ckpt.get('H_grid_mean')
    if H_grid_mean is not None:
        H_grid_mean = H_grid_mean.to(device)
        # Wire the residualization mean into the adapter so any caller of
        # adapter.predict_from_grid(H_hat_grid_res, ...) gets de-residualization
        # for free (matches train_tisae.py).
        adapter.set_residualization_mean(H_grid_mean)

    # Wire extract_layer from TISAE config into the adapter so extract_hidden
    # returns intermediate-layer embeddings (P0.2: locality-preserving).
    extract_layer = config.get('args', {}).get('extract_layer', -1)
    adapter.set_extract_layer(extract_layer)

    # Read prediction task from backbone checkpoint
    _bb_ckpt = torch.load(bb_path, map_location='cpu', weights_only=False)
    _task = _bb_ckpt.get('training_args', {}).get('task', 'mortality')
    del _bb_ckpt

    # Filter readmission splits (exclude in-hospital deaths with label -1)
    if _task == 'readmission' and hasattr(dataset, 'y_readmission'):
        for split in dataset.splits:
            inds = np.array(dataset.splits[split])
            valid = dataset.y_readmission[inds] >= 0
            dataset.splits[split] = inds[valid].tolist()

    return {
        'adapter': adapter,
        'backbone': adapter,  # backwards-compat alias for legacy code
        'sae': sae,
        'dataset': dataset,
        'tisae_config': tisae_cfg,
        'H_grid_mean': H_grid_mean,
        'task': _task,
    }


# =============================================================================
# Energy prefilter
# =============================================================================

def _energy_prefilter(sae, A0_4d, g, gate, cfg, n_pool, device,
                      alive_mask=None):
    """Pre-filter to top-n_pool islands by energy score. Returns binary mask."""
    A0_perm = A0_4d.permute(0, 1, 3, 2)  # [B, K_eff, C_eff, T]
    eff_len = g.sum(dim=-1).clamp_min(1e-6)  # [B, K_eff, C_eff, M]
    energy = (A0_perm.unsqueeze(3) * g).sum(dim=-1)  # [B, K_eff, C_eff, M]
    if cfg.score_mode == "energy_gated" and gate is not None:
        scores = energy / eff_len.sqrt() * gate
    else:
        scores = energy / eff_len

    if alive_mask is not None:
        dead = ~alive_mask
        scores = scores.masked_fill(dead.unsqueeze(0).unsqueeze(-1), -1e9)

    B = scores.size(0)
    scores_flat = scores.view(B, -1)

    if n_pool is None:
        if alive_mask is not None:
            alive_flat = alive_mask.unsqueeze(-1).expand_as(
                scores[:1]).reshape(1, -1).expand(B, -1)
            return alive_flat.float().view_as(scores)
        else:
            return torch.ones_like(scores)

    n_sel = min(n_pool, scores_flat.size(1))
    _, idx = scores_flat.topk(n_sel, dim=-1)
    pool_mask = torch.zeros_like(scores_flat).scatter_(-1, idx, 1.0)
    return pool_mask.view_as(scores)


# =============================================================================
# Training step
# =============================================================================

def train_step(
    adapter, sae, selector, batch, T, V, H_grid_mean, W_dec,
    prior_cache, config, device, n_prefilter=2560, n_pool_full=256,
    include_obs_profile=True, soft_selection=True, soft_tau=1.0,
    alive_mask=None,
):
    """Single training step. Returns loss dict.

    `adapter` is a frozen :class:`BackboneAdapter`. All backbone
    operations route through it; this function never touches the underlying
    nn.Module directly so any backbone is supported.
    """
    K_eff, C_eff = sae.K_eff, sae.C_eff
    cfg = sae.cfg
    K, M = cfg.K, cfg.M

    # Move ALL keys to device — STraTS/HyMaTE adapters need values_zscore
    # and times_norm in addition to the binned (value_bins, tau) pair, and
    # a hand-curated 4-key subset would silently miss them and KeyError
    # inside StratsBackboneAdapter.extract_hidden.
    batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    vb = batch_dev['value_bins']
    tau = batch_dev['tau']
    varis = batch_dev['varis']
    obs_mask = batch_dev['obs_mask']
    orig_idx = batch_dev['orig_idx']
    B = vb.size(0)

    # Frozen backbone — extract per-event H and the [B,T,V,D] grid via adapter.
    # When extract_layer != -1, H is at the intermediate layer (for the TISAE
    # grid), but h_pool and z_teacher MUST come from the full backbone.
    with torch.no_grad():
        H_grid, obs_grid, H = adapter.extract_hidden(batch_dev, return_event_h=True)
        if adapter.extract_layer != -1:
            out_full = adapter.forward(batch_dev, return_pooled=True)
            h_pool = out_full['pooled']
            z_teacher = out_full['logits']
        else:
            h_pool, _ = adapter.pool_events(H, obs_mask)
            z_teacher = adapter.predict_from_pooled(h_pool)

    # Scatter to grid, subtract per-variable population mean
    with torch.no_grad():
        if H_grid_mean is not None:
            H_grid_input = H_grid - H_grid_mean.unsqueeze(0).unsqueeze(1)
            H_grid_input = H_grid_input * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

    # Frozen SAE sub-components
    with torch.no_grad():
        A0 = sae.encoder(H_grid_input, obs_grid)
        T_len = A0.size(2)

        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T_len).permute(0, 2, 3, 1)
        else:
            A0_4d = A0.unsqueeze(-1)

        mu, sigma, gate = sae.island_params(A0_4d)
        sigma_main = sigma.detach()
        g = sae.gaussian_mask.build_gaussians(mu, sigma_main)

        pool_mask = _energy_prefilter(sae, A0_4d, g, gate, cfg, n_prefilter, device,
                                      alive_mask=alive_mask)

        _n_full = n_pool_full if n_pool_full is not None else n_prefilter
        if _n_full is not None and n_prefilter is not None and _n_full < n_prefilter:
            full_mask = _energy_prefilter(sae, A0_4d, g, gate, cfg, _n_full, device,
                                          alive_mask=alive_mask)
        else:
            full_mask = pool_mask
        mask_full = sae.gaussian_mask.forward_mask(g, full_mask)
        A_hat_full_4d = A0_4d * mask_full
        if C_eff > 1:
            A_hat_full = A_hat_full_4d.permute(0, 3, 1, 2).reshape(B, K, T_len)
        else:
            A_hat_full = A_hat_full_4d.squeeze(-1)
        H_hat_grid_full = sae.decoder(A_hat_full)

        # z_full: predict from the full-pool SAE reconstruction.
        # When extract_layer != -1, predict_from_grid runs remaining layers.
        # When extract_layer == -1, predict_from_grid does direct pool+predict.
        # predict_from_grid handles de-residualization internally.
        z_full = adapter.predict_from_grid(H_hat_grid_full, batch_dev, obs_grid)

    # Prior scores
    prior_scores = None
    if prior_cache is not None:
        prior_scores = prior_cache[orig_idx].to(device).float()

    # Assemble features (no grad)
    with torch.no_grad():
        features = assemble_island_features(
            A0_4d, g, mu, sigma, gate, obs_grid, W_dec,
            prior_scores, K_eff, C_eff,
            include_obs_profile=include_obs_profile)

    # PredSelector (TRAINABLE)
    s_pred = selector(features, h_pool.detach())
    s_pred = s_pred + (pool_mask - 1) * 1e9

    if soft_selection:
        s_flat = s_pred.view(B, -1)
        weights = F.softmax(s_flat / soft_tau, dim=-1) * config.n_global_pred
        sel_pred = weights.view(B, K_eff, C_eff, M)
        mask_bkct = (sel_pred.unsqueeze(-1) * g).sum(dim=-2).clamp(0, 1)
        mask_pred = mask_bkct.permute(0, 1, 3, 2)
    else:
        sel_pred = selector.select(s_pred)
        mask_pred = sae.gaussian_mask.forward_mask(g, sel_pred)

    # A_hat_keep
    A_hat_keep_4d = A0_4d * mask_pred
    if C_eff > 1:
        A_hat_keep = A_hat_keep_4d.permute(0, 3, 1, 2).reshape(B, K, T_len)
    else:
        A_hat_keep = A_hat_keep_4d.squeeze(-1)
    H_hat_grid_keep = sae.decoder(A_hat_keep)

    # z_keep: predict from kept-island reconstruction.
    # predict_from_grid handles de-residualization + remaining layers.
    z_keep = adapter.predict_from_grid(H_hat_grid_keep, batch_dev, obs_grid)

    # Drop (complement within pool)
    mask_drop = (mask_full - mask_pred).clamp(0, 1)
    A_hat_drop_4d = A0_4d * mask_drop
    if C_eff > 1:
        A_hat_drop = A_hat_drop_4d.permute(0, 3, 1, 2).reshape(B, K, T_len)
    else:
        A_hat_drop = A_hat_drop_4d.squeeze(-1)
    H_hat_grid_drop = sae.decoder(A_hat_drop)

    # z_drop: predict from dropped-island reconstruction.
    # predict_from_grid handles de-residualization + remaining layers.
    z_drop = adapter.predict_from_grid(H_hat_grid_drop, batch_dev, obs_grid)

    # Losses
    losses = pred_selector_losses(
        z_keep.unsqueeze(-1),
        z_drop.unsqueeze(-1),
        z_full.unsqueeze(-1),
        z_teacher.unsqueeze(-1),
        s_pred=s_pred,
        s_prior=prior_scores,
        config=config,
    )

    # Extra metrics
    with torch.no_grad():
        n_selected = (sel_pred > 0.5).float().sum(dim=(1, 2, 3)).mean()
        losses['n_selected'] = n_selected
        losses['z_teacher_mean'] = z_teacher.mean()
        losses['z_keep_mean'] = z_keep.mean()
        losses['z_drop_mean'] = z_drop.mean()
        losses['z_full_mean'] = z_full.mean()
        logit_mae = (z_keep - z_teacher).abs().mean()
        losses['logit_mae'] = logit_mae

        err = (z_keep - z_teacher).abs()
        losses['err_p25'] = err.quantile(0.25)
        losses['err_p50'] = err.quantile(0.50)
        losses['err_p75'] = err.quantile(0.75)
        losses['err_p95'] = err.quantile(0.95)

        # Score stats
        s_pool = s_pred + (pool_mask - 1) * 1e9
        s_valid = s_pool[pool_mask > 0.5]
        if s_valid.numel() > 0:
            losses['score_mean'] = s_valid.mean()
            losses['score_std'] = s_valid.std()

        # Overlap with energy heuristic
        _A0_perm = A0_4d.permute(0, 1, 3, 2)
        _eff_len = g.sum(dim=-1).clamp_min(1e-6)
        energy_scores = (_A0_perm.unsqueeze(3) * g).sum(dim=-1) / _eff_len.sqrt()
        if gate is not None:
            energy_scores = energy_scores * gate
        energy_pooled = energy_scores * pool_mask + (pool_mask - 1) * 1e9
        _, eidx = energy_pooled.view(B, -1).topk(config.n_global_pred, dim=-1)
        energy_sel = torch.zeros(B, K_eff * C_eff * M, device=device).scatter_(-1, eidx, 1.0)
        pred_sel_flat = (sel_pred > 0.5).float().view(B, -1)
        overlap = (energy_sel * pred_sel_flat).sum(-1) / config.n_global_pred
        losses['energy_overlap'] = overlap.mean()

    return losses


# =============================================================================
# Training loop
# =============================================================================

def train_epoch(adapter, sae, selector, dataloader, optimizer, T, V,
                H_grid_mean, W_dec, prior_cache, config, device,
                n_prefilter=2560, n_pool_full=256, max_grad_norm=1.0,
                include_obs_profile=True, soft_selection=True, soft_tau=1.0,
                alive_mask=None):
    selector.train()
    metrics = defaultdict(float)
    n_samples = 0

    pbar = tqdm(dataloader, desc="Training", file=sys.stderr)
    for batch in pbar:
        losses = train_step(
            adapter, sae, selector, batch, T, V,
            H_grid_mean, W_dec, prior_cache, config, device,
            n_prefilter=n_prefilter, n_pool_full=n_pool_full,
            include_obs_profile=include_obs_profile,
            soft_selection=soft_selection, soft_tau=soft_tau,
            alive_mask=alive_mask)

        loss = losses['L_sel']
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(selector.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        B = batch['value_bins'].size(0)
        n_samples += B
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                metrics[k] += v.item() * B

        pbar.set_postfix({
            'L_sel': f"{metrics['L_sel']/n_samples:.4f}",
            'L_logit': f"{metrics['L_logit']/n_samples:.4f}",
            'mae': f"{metrics['logit_mae']/n_samples:.4f}",
        })

    for k in metrics:
        metrics[k] /= max(1, n_samples)
    return dict(metrics)


@torch.no_grad()
def evaluate(adapter, sae, selector, dataloader, T, V,
             H_grid_mean, W_dec, prior_cache, config, device,
             n_prefilter=2560, n_pool_full=256, include_obs_profile=True,
             soft_selection=True, soft_tau=1.0, alive_mask=None):
    selector.eval()
    metrics = defaultdict(float)
    n_samples = 0

    for batch in dataloader:
        losses = train_step(
            adapter, sae, selector, batch, T, V,
            H_grid_mean, W_dec, prior_cache, config, device,
            n_prefilter=n_prefilter, n_pool_full=n_pool_full,
            include_obs_profile=include_obs_profile,
            soft_selection=soft_selection, soft_tau=soft_tau,
            alive_mask=alive_mask)

        B = batch['value_bins'].size(0)
        n_samples += B
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                metrics[k] += v.item() * B

    for k in metrics:
        metrics[k] /= max(1, n_samples)
    return dict(metrics)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train PredSelector on frozen TISAE")
    parser.add_argument('--sae_experiment_dir', type=str, required=True,
                        help='Path to frozen TISAE experiment')
    parser.add_argument('--dataset_path', type=str,
                        default='data/unified_mimic3_48h.pkl')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prior_cache_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')

    # PredSelector config
    parser.add_argument('--d_sel', type=int, default=64)
    parser.add_argument('--mlp_hidden', type=int, default=64)
    parser.add_argument('--n_global_pred', type=int, default=64)
    parser.add_argument('--n_per_channel', type=int, default=None)
    parser.add_argument('--gumbel_topk_noise', type=float, default=0.0)
    parser.add_argument('--separated_topk', action='store_true', default=None)
    parser.add_argument('--no_separated_topk', dest='separated_topk',
                        action='store_false')

    # Losses
    parser.add_argument('--lambda_logit', type=float, default=1.0)
    parser.add_argument('--lambda_comp', type=float, default=0.1)
    parser.add_argument('--delta_comp', type=float, default=0.5)
    parser.add_argument('--lambda_prior', type=float, default=0.0)
    parser.add_argument('--prior_tau', type=float, default=1.0)

    # Training
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=52)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--early_stop_metric', type=str, default='L_sel')

    # Pre-filtering
    parser.add_argument('--n_prefilter', type=int, default=2560)
    parser.add_argument('--n_pool_full', type=int, default=256)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--alive_mask', action='store_true', default=False)
    parser.add_argument('--no_obs_profile', action='store_true', default=False)
    parser.add_argument('--soft_selection', action='store_true', default=True)
    parser.add_argument('--soft_tau', type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    logger = Logger(output_dir)

    # Load frozen models
    logger.write(f"Loading frozen TISAE from {args.sae_experiment_dir}...")
    models = load_tisae_models(args.sae_experiment_dir, args.dataset_path, device=str(device))

    adapter: BackboneAdapter = models['adapter']
    sae = models['sae']
    dataset = models['dataset']
    tisae_config = models['tisae_config']
    H_grid_mean = models.get('H_grid_mean')

    T, V, D = tisae_config.T, tisae_config.V, tisae_config.D
    K_eff, C_eff = sae.K_eff, sae.C_eff
    M = tisae_config.M

    # Freeze everything. The adapter is already frozen by load_backbone_adapter;
    # we only need to freeze the SAE here.
    for p in sae.parameters():
        p.requires_grad = False
    sae.eval()

    W_dec = sae.decoder.get_W().detach()

    # Load prior cache
    prior_cache = None
    prior_path = args.prior_cache_path
    if prior_path is None:
        auto_path = Path(args.sae_experiment_dir) / 'grad_act_prior.pt'
        if auto_path.exists():
            prior_path = str(auto_path)
            logger.write(f"Auto-detected prior cache: {prior_path}")

    if prior_path:
        logger.write(f"Loading prior cache from {prior_path}...")
        prior_data = torch.load(prior_path, map_location='cpu')
        prior_cache = prior_data['prior_cache'].float()
        logger.write(f"  Prior cache shape: {prior_cache.shape}")

    # Alive mask
    alive_mask = None
    if args.alive_mask and hasattr(sae, 'batches_since_fired'):
        dead_thresh = tisae_config.dead_threshold_batches
        alive_mask = sae.batches_since_fired <= dead_thresh
        n_alive = alive_mask.sum().item()
        n_total = alive_mask.numel()
        logger.write(f"Alive latents: {n_alive}/{n_total} ({100*n_alive/n_total:.1f}%)")
    elif args.alive_mask:
        logger.write("WARNING: --alive_mask requested but SAE has no batches_since_fired tracker")

    # Resolve separated_topk
    separated_topk = args.separated_topk
    if separated_topk is None:
        separated_topk = tisae_config.separated_topk
        logger.write(f"Inheriting separated_topk={separated_topk} from TISAE config")

    # Build config
    config = PredSelectorConfig(
        d_sel=args.d_sel,
        mlp_hidden=args.mlp_hidden,
        n_global_pred=args.n_global_pred,
        n_per_channel=args.n_per_channel,
        gumbel_topk_noise=args.gumbel_topk_noise,
        separated_topk=separated_topk,
        lambda_logit=args.lambda_logit,
        lambda_comp=args.lambda_comp,
        delta_comp=args.delta_comp,
        lambda_prior=args.lambda_prior,
        prior_tau=args.prior_tau,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    # Build PredSelector
    include_obs_profile = not args.no_obs_profile
    n_island_features = 6 + 2 * V if include_obs_profile else 6 + V
    selector = PredSelector(config, n_island_features, D, C_eff).to(device)
    n_params = sum(p.numel() for p in selector.parameters() if p.requires_grad)
    logger.write(f"PredSelector: {n_params:,} trainable params")
    logger.write(f"  n_island_features={n_island_features}, D={D}, C_eff={C_eff}")
    logger.write(f"  n_global_pred={config.n_global_pred}")

    # Data
    train_ds = PSDataset(dataset, 'train')
    val_ds = PSDataset(dataset, 'val')
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, collate_fn=ps_collate_fn,
                              num_workers=2, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, collate_fn=ps_collate_fn,
                            num_workers=2, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Save config
    n_prefilter = args.n_prefilter
    n_pool_full = args.n_pool_full
    max_grad_norm = args.max_grad_norm
    full_config = {
        'pred_selector_config': asdict(config),
        'sae_experiment_dir': args.sae_experiment_dir,
        'tisae_config': asdict(tisae_config),
        'n_island_features': n_island_features,
        'D': D, 'K_eff': K_eff, 'C_eff': C_eff, 'M': M, 'T': T, 'V': V,
        'n_trainable_params': n_params,
        'n_prefilter': n_prefilter,
        'n_pool_full': n_pool_full,
        'max_grad_norm': max_grad_norm,
        'include_obs_profile': include_obs_profile,
        'soft_selection': args.soft_selection,
        'soft_tau': args.soft_tau,
        'n_alive_latents': int(alive_mask.sum().item()) if alive_mask is not None else None,
        'n_total_latents': int(alive_mask.numel()) if alive_mask is not None else K_eff * C_eff,
    }
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False)
    logger.write(f"  n_prefilter={n_prefilter}, n_pool_full={n_pool_full}, max_grad_norm={max_grad_norm}")

    # Training
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        logger.write(f"\n--- Epoch {epoch}/{config.epochs} ---")

        train_metrics = train_epoch(
            adapter, sae, selector, train_loader, optimizer,
            T, V, H_grid_mean, W_dec, prior_cache, config, device,
            n_prefilter=n_prefilter, n_pool_full=n_pool_full,
            max_grad_norm=max_grad_norm,
            include_obs_profile=include_obs_profile,
            soft_selection=args.soft_selection, soft_tau=args.soft_tau,
            alive_mask=alive_mask)

        val_metrics = evaluate(
            adapter, sae, selector, val_loader,
            T, V, H_grid_mean, W_dec, prior_cache, config, device,
            n_prefilter=n_prefilter, n_pool_full=n_pool_full,
            include_obs_profile=include_obs_profile,
            soft_selection=args.soft_selection, soft_tau=args.soft_tau,
            alive_mask=alive_mask)

        # Hard eval: if training with soft selection, also eval with hard top-k
        if args.soft_selection:
            val_hard = evaluate(
                adapter, sae, selector, val_loader,
                T, V, H_grid_mean, W_dec, prior_cache, config, device,
                n_prefilter=n_prefilter, n_pool_full=n_pool_full,
                include_obs_profile=include_obs_profile,
                soft_selection=False, soft_tau=args.soft_tau,
                alive_mask=alive_mask)

        _ps_keys = ['L_sel', 'L_logit', 'L_comp', 'n_selected', 'logit_mae']
        _fmt = lambda m: " | ".join(f"{k}={m[k]:.4f}" for k in _ps_keys if k in m)
        logger.write(f"Train: " + _fmt(train_metrics))
        logger.write(f"Val:   " + _fmt(val_metrics))
        if args.soft_selection:
            logger.write(f"Hard:  " + _fmt(val_hard))

        val_loss = val_metrics.get(args.early_stop_metric, val_metrics.get('L_sel', float('inf')))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'selector_state_dict': selector.state_dict(),
                'config': asdict(config),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')
            logger.write(f"  * New best val {args.early_stop_metric}={val_loss:.4f}")
        else:
            patience_counter += 1
            logger.write(f"  No improvement ({patience_counter}/{config.patience})")
            if patience_counter >= config.patience:
                logger.write(f"Early stopping at epoch {epoch}")
                break

    logger.write(f"\nBest val {args.early_stop_metric}: {best_val_loss:.4f}")
    logger.write(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
