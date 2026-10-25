"""
TimeX, TimeX++, and ExtremalMask attribution baselines.

Separated from main attribution_methods.py for code isolation.
All methods produce attr[B, T, V] — same contract as other methods.

References:
  - TimeX: Queen et al., NeurIPS 2023 (txai package)
  - TimeX++: Liu et al., ICML 2024 (txai_pp package)
  - ExtremalMask: Enguehard, ICML 2023 (tint package)
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from typing import Dict, Tuple, Optional, Callable, List

from tisae.models.backbone_adapter import BackboneAdapter
from tisae.evaluation.faithfulness.attribution_methods import (
    _build_obs_grid,
    _build_value_grid,
    _precompute_replacement_embeddings,
    soft_masked_predict,
    soft_replacement_predict,
    masked_predict,
    marginal_masked_predict,
    conditional_masked_predict,
)

# ── Conditional imports ──────────────────────────────────────────────────────

HAS_TXAI = False
try:
    from txai.models.mask_generators.maskgen import MaskGenerator
    from txai.utils.predictors.loss import GSATLoss, ConnectLoss
    HAS_TXAI = True
except ImportError:
    pass

HAS_TXAI_PP = False
try:
    from txai_pp.models.mask_generators.maskgen import MaskGenerator as MaskGenerator_PP
    from txai_pp.utils.predictors.loss import GSATLoss as GSATLoss_PP, ConnectLoss as ConnectLoss_PP
    from txai_pp.models.bc_model import DistributionParams
    from txai_pp.utils.functional import js_divergence as js_divergence_pp
    HAS_TXAI_PP = True
except ImportError:
    pass

HAS_EXTREMAL = False
try:
    from tint.attr import ExtremalMask as _TintExtremalMask
    from tint.attr.models import ExtremalMaskNet as _TintExtremalMaskNet
    import pytorch_lightning as _pl
    _PLTrainer = _pl.Trainer
    HAS_EXTREMAL = True
except ImportError:
    pass



# =============================================================================
# Shared helpers
# =============================================================================

def _pad_and_cat(tensors, pad_val=0):
    """Pad variable-length tensors along dim=1 to max length, then cat along dim=0."""
    max_L = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        pad_size = max_L - t.shape[1]
        if pad_size > 0:
            pad_shape = list(t.shape)
            pad_shape[1] = pad_size
            t = torch.cat([t, torch.full(pad_shape, pad_val, dtype=t.dtype)], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0), max_L

def _build_timex_input(batch, T, V, device, train_mean, train_std):
    """Build TimeX input: [B, T, 2V] = [z-normed values, obs mask]."""
    value_grid = _build_value_grid(batch, T, V, device)     # [B, T, V]
    obs_grid = _build_obs_grid(batch, T, V, device)          # [B, T, V]
    normed = (value_grid - train_mean) / (train_std + 1e-8)
    normed = normed * obs_grid  # zero out unobserved
    return torch.cat([normed, obs_grid], dim=-1)              # [B, T, 2V]


def precompute_timex_stats(dataset, train_indices, T, V, device, batch_size=64):
    """Compute per-variable mean/std from training set for z-score normalization."""
    all_grids, all_masks = [], []
    for start in range(0, len(train_indices), batch_size):
        end = min(start + batch_size, len(train_indices))
        batch = dataset.get_batch(train_indices[start:end])
        batch = {k: v.to(device) for k, v in batch.items()}
        all_grids.append(_build_value_grid(batch, T, V, device))
        all_masks.append(_build_obs_grid(batch, T, V, device))
    grids = torch.cat(all_grids)   # [N, T, V]
    masks = torch.cat(all_masks)   # [N, T, V]
    obs_sum = (grids * masks).sum(dim=(0, 1))             # [V]
    obs_count = masks.sum(dim=(0, 1)).clamp_min(1)        # [V]
    train_mean = (obs_sum / obs_count)[None, None, :]     # [1, 1, V]
    obs_sq = ((grids - train_mean) ** 2 * masks).sum(dim=(0, 1))
    train_std = (obs_sq / obs_count).sqrt()[None, None, :]  # [1, 1, V]
    return train_mean, train_std


@torch.no_grad()
def _backbone_to_grid(backbone: BackboneAdapter, batch, T, V, D, device):
    """Frozen backbone H → scatter to grid → pool D → cat obs_grid → [B, T, 2V].

    Uses BackboneAdapter.extract_hidden() for correct, vectorized scatter.
    Returns backbone-informed grid: first V channels = H_grid.mean(dim=D),
    last V channels = obs_grid.
    """
    H_grid, obs_grid = backbone.extract_hidden(batch)  # [B,T,V,D], [B,T,V]
    H_mean = H_grid.mean(dim=-1)                       # [B, T, V]
    return torch.cat([H_mean, obs_grid], dim=-1)        # [B, T, 2V]


def _precompute_timex_features(
    backbone: BackboneAdapter, dataset, train_indices, T, V, device,
    batch_size, train_mean, train_std, cache_path=None,
):
    """Pre-compute all frozen backbone outputs for TimeX/TimeX++ training.

    Stores variable-length sequences as per-sample lists on GPU.
    If cache_path is given, saves/loads from disk to skip re-computation.

    Returns: (inputs, backbone_grids, logits_all, pooled_all,
              emb_list, tau_list, varis_list, om_list)
    """
    D = backbone.D

    # Try loading from cache
    if cache_path is not None and os.path.exists(cache_path):
        print(f"  Loading pre-computed features from {cache_path}...", flush=True)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        inputs = cache["inputs"]
        backbone_grids = cache["backbone_grids"]
        logits_all = cache["logits_all"]
        pooled_all = cache["pooled_all"]
        emb_list = [e.to(device) for e in cache["emb_list"]]
        tau_list = [t.to(device) for t in cache["tau_list"]]
        varis_list = [v.to(device) for v in cache["varis_list"]]
        om_list = [o.to(device) for o in cache["om_list"]]
        avg_L = sum(e.shape[0] for e in emb_list) / len(emb_list)
        gpu_mb = sum(e.nelement() * 4 for e in emb_list) / 1e6
        print(f"  Loaded: N={inputs.shape[0]}, avg_L={avg_L:.0f} "
              f"(emb cache: {gpu_mb:.0f} MB GPU)", flush=True)
        return inputs, backbone_grids, logits_all, pooled_all, \
               emb_list, tau_list, varis_list, om_list

    # Compute from scratch
    inputs = []
    backbone_grids = []
    logits_all = []
    pooled_all = []
    emb_list = []
    tau_list = []
    varis_list = []
    om_list = []

    print(f"  Pre-computing backbone features for {len(train_indices)} samples...", flush=True)

    for start in range(0, len(train_indices), batch_size):
        end = min(start + batch_size, len(train_indices))
        idx = train_indices[start:end]
        batch = dataset.get_batch(idx)
        batch_dev = {k: v.to(device) for k, v in batch.items()}

        inp = _build_timex_input(batch_dev, T, V, device, train_mean, train_std)
        inputs.append(inp.cpu())

        bg = _backbone_to_grid(backbone, batch_dev, T, V, D, device)
        backbone_grids.append(bg.cpu())

        with torch.no_grad():
            out = backbone.forward(batch_dev, return_pooled=True)
            logits_all.append(out["logits"].cpu())
            pooled_all.append(out["pooled"].cpu())
            emb = backbone.get_event_embeddings(batch_dev)

        om = batch_dev["obs_mask"]
        for b in range(emb.shape[0]):
            n_obs = int(om[b].sum().item())
            emb_list.append(emb[b, :n_obs].detach())
            tau_list.append(batch_dev["tau"][b, :n_obs])
            varis_list.append(batch_dev["varis"][b, :n_obs])
            om_list.append(om[b, :n_obs])

    inputs = torch.cat(inputs, dim=0)
    backbone_grids = torch.cat(backbone_grids, dim=0)
    logits_all = torch.cat(logits_all, dim=0)
    pooled_all = torch.cat(pooled_all, dim=0)

    avg_L = sum(e.shape[0] for e in emb_list) / len(emb_list)
    gpu_mb = sum(e.nelement() * 4 for e in emb_list) / 1e6
    print(f"  Pre-computation done: N={inputs.shape[0]}, avg_L={avg_L:.0f} "
          f"(emb cache: {gpu_mb:.0f} MB GPU)", flush=True)

    # Save to cache
    if cache_path is not None:
        print(f"  Saving cache to {cache_path}...", flush=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "inputs": inputs,
            "backbone_grids": backbone_grids,
            "logits_all": logits_all,
            "pooled_all": pooled_all,
            "emb_list": [e.cpu() for e in emb_list],
            "tau_list": [t.cpu() for t in tau_list],
            "varis_list": [v.cpu() for v in varis_list],
            "om_list": [o.cpu() for o in om_list],
        }, cache_path)
        print(f"  Cache saved ({os.path.getsize(cache_path)/1e9:.1f} GB)", flush=True)

    return inputs, backbone_grids, logits_all, pooled_all, \
           emb_list, tau_list, varis_list, om_list


# =============================================================================
# 1. TimeX — Queen et al., NeurIPS 2023
# =============================================================================

def train_timex_explainer(
    backbone: BackboneAdapter,
    dataset,
    train_indices,
    T: int,
    V: int,
    device: str = "cuda",
    n_epochs: int = 50,
    batch_size: int = 128,
    gsat_r: float = 0.15,
    lr: float = 1e-3,
    beta_sim: float = 1.0,
    beta_exp: float = 0.5,
    cache_path: str = None,
) -> Tuple:
    """Train TimeX mask generator to explain backbone predictions.

    Uses backbone features + MaskGenerator from txai to produce masks.
    The backbone serves as the feature source and target model.

    Returns: (mask_gen, train_mean, train_std)
    """
    if not HAS_TXAI:
        raise ImportError("txai package required: pip install git+https://github.com/mims-harvard/TimeX.git")

    backbone.eval()

    # Step 1: Compute normalization stats
    train_mean, train_std = precompute_timex_stats(
        dataset, train_indices, T, V, device, batch_size)

    # Step 2: Pre-compute (or load cached) frozen backbone outputs
    inputs, backbone_grids, logits_all, pooled_all, \
        emb_list, tau_list, varis_list, om_list = _precompute_timex_features(
            backbone, dataset, train_indices, T, V, device,
            batch_size, train_mean, train_std, cache_path)
    n_train = inputs.shape[0]

    # Step 3: Create mask generator
    d_inp = 2 * V
    d_pe = 16
    d_z = d_inp + d_pe

    mask_gen = MaskGenerator(
        d_z=d_z, max_len=T, d_pe=d_pe, use_ste=True,
    ).to(device)

    gsat_loss_fn = GSATLoss(r=gsat_r)
    connect_loss_fn = ConnectLoss()

    # Step 4: Training loop
    params = list(mask_gen.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    bce = nn.BCEWithLogitsLoss()
    cos_sim = nn.CosineSimilarity(dim=-1)

    for epoch in range(n_epochs):
        mask_gen.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            bs = len(idx)

            logits_full = logits_all[idx].to(device)
            pooled_full = pooled_all[idx].to(device)

            # z_seq from backbone grid + pos_encoder
            bg = backbone_grids[idx].to(device).permute(1, 0, 2)  # [T, B, 2V]
            times = torch.arange(T, device=device).float().unsqueeze(1).expand(T, bs)
            z_seq = torch.cat([bg, mask_gen.pos_encoder(times)], dim=-1)  # [T, B, d_z]
            src = inputs[idx].to(device).permute(1, 0, 2)  # [T, B, 2V]

            # MaskGenerator returns (ste_mask [T,B,d_inp], mask_logits [B,T,d_inp])
            ste_mask_tf, mask_logits = mask_gen(z_seq, src, times)
            ste_mask = ste_mask_tf.permute(1, 0, 2)  # [B, T, d_inp]
            mask_tv = ste_mask[..., :V]               # [B, T, V]

            # Pad per-batch from variable-length GPU lists (no global padding)
            emb = torch.nn.utils.rnn.pad_sequence(
                [emb_list[i] for i in idx], batch_first=True)    # [B, L_batch, D]
            tau_b = torch.nn.utils.rnn.pad_sequence(
                [tau_list[i] for i in idx], batch_first=True)    # [B, L_batch]
            varis_b = torch.nn.utils.rnn.pad_sequence(
                [varis_list[i] for i in idx], batch_first=True)  # [B, L_batch]
            om_b = torch.nn.utils.rnn.pad_sequence(
                [om_list[i] for i in idx], batch_first=True)     # [B, L_batch]

            # Apply mask in event space
            batch_idx = torch.arange(bs, device=device).unsqueeze(1).expand_as(tau_b)
            event_mask = mask_tv[batch_idx, tau_b, varis_b]  # [B, L]
            masked_emb = emb * event_mask.unsqueeze(-1)

            # Forward from masked embeddings (need pooled for sim_loss)
            pooled_mask, _ = backbone.pool_events(masked_emb, om_b)
            logits_mask = backbone.predict_from_pooled(pooled_mask)

            clf_loss = bce(logits_mask, torch.sigmoid(logits_full).detach())
            sim_loss = beta_sim * (1.0 - cos_sim(pooled_mask, pooled_full.detach())).mean()
            exp_loss = beta_exp * (gsat_loss_fn(mask_logits) + connect_loss_fn(mask_logits))

            loss = clf_loss + sim_loss + exp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        print(f"  TimeX epoch {epoch+1}/{n_epochs}: loss={avg:.4f}", flush=True)

    mask_gen.eval()
    return mask_gen, train_mean, train_std


@torch.no_grad()
def compute_timex_attribution(
    backbone: BackboneAdapter,
    timex_model,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
) -> torch.Tensor:
    """Extract TimeX masks as attribution. Returns [B, T, V].

    ste_mask ~ 1 = important (keep), ~ 0 = unimportant.
    """
    mask_gen = timex_model
    device = batch["obs_mask"].device
    D = backbone.D

    inp = _build_timex_input(batch, T, V, device, train_mean, train_std)  # [B,T,2V]
    B = inp.shape[0]

    bg = _backbone_to_grid(backbone, batch, T, V, D, device)  # [B, T, 2V]
    bg_tf = bg.permute(1, 0, 2)  # [T, B, 2V]
    times = torch.arange(T, device=device).float().unsqueeze(1).expand(T, B)
    z_seq = torch.cat([bg_tf, mask_gen.pos_encoder(times)], dim=-1)  # [T, B, d_z]

    src = inp.permute(1, 0, 2)  # [T, B, 2V]

    ste_mask_tf, _ = mask_gen(z_seq, src, times)  # [T, B, 2V]
    ste_mask = ste_mask_tf.permute(1, 0, 2)       # [B, T, 2V]
    mask = ste_mask[..., :V]                       # [B, T, V]

    obs_grid = _build_obs_grid(batch, T, V, device)
    return mask * obs_grid


# =============================================================================
# 2. TimeX++ — Liu et al., ICML 2024
# =============================================================================

def _kl_divergence_gaussian(mean1, logvar1, mean2, logvar2):
    """KL(N(mean1,var1) || N(mean2,var2)) for TimeX++ information bottleneck."""
    return 0.5 * torch.mean(
        logvar2 - logvar1 - 1
        + (logvar1.exp() + (mean1 - mean2).pow(2)) / logvar2.exp()
    )


def train_timexpp_explainer(
    backbone: BackboneAdapter,
    dataset,
    train_indices,
    T: int,
    V: int,
    device: str = "cuda",
    n_epochs: int = 50,
    batch_size: int = 128,
    gsat_r: float = 0.15,
    lr: float = 1e-3,
    beta_exp: float = 0.5,
    beta_sim: float = 1.0,
    bias_weight: float = 0.1,
    cache_path: str = None,
) -> Tuple:
    """Train TimeX++ mask generator with information bottleneck.

    Same backbone-as-encoder approach as TimeX, with additional TimeX++
    KL divergence between source/masked distributions + MSE reference matching.

    Returns: (components, train_mean, train_std)
        where components = (mask_gen, src_dist, mask_dist, mask_conn)
    """
    if not HAS_TXAI_PP:
        raise ImportError(
            "txai_pp package required: clone TimeXplusplus, rename to txai_pp, pip install -e"
        )

    backbone.eval()

    # Step 1: Compute normalization stats
    train_mean, train_std = precompute_timex_stats(
        dataset, train_indices, T, V, device, batch_size)

    # Step 2: Pre-compute (or load cached) frozen backbone outputs
    inputs, backbone_grids, logits_all, pooled_all, \
        emb_list, tau_list, varis_list, om_list = _precompute_timex_features(
            backbone, dataset, train_indices, T, V, device,
            batch_size, train_mean, train_std, cache_path)
    n_train = inputs.shape[0]

    # Step 3: Create mask generator + distribution components
    d_inp = 2 * V
    d_pe = 16
    d_z = d_inp + d_pe
    flat_dim = T * d_inp

    mask_gen = MaskGenerator_PP(
        d_z=d_z, max_len=T, d_pe=d_pe, use_ste=True,
    ).to(device)

    src_dist = DistributionParams(flat_dim, flat_dim).to(device)
    mask_dist = DistributionParams(flat_dim, flat_dim).to(device)
    mask_conn = nn.Sequential(
        nn.Linear(2, 32), nn.ELU(), nn.Linear(32, 1)
    ).to(device)

    gsat_loss_fn = GSATLoss_PP(r=gsat_r)
    connect_loss_fn = ConnectLoss_PP()

    # Step 4: Training loop
    params = (
        list(mask_gen.parameters())
        + list(src_dist.parameters()) + list(mask_dist.parameters())
        + list(mask_conn.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.01)
    cos_sim = nn.CosineSimilarity(dim=-1)

    for epoch in range(n_epochs):
        mask_gen.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            bs = len(idx)

            logits_full = logits_all[idx].to(device)
            pooled_full = pooled_all[idx].to(device)

            bg = backbone_grids[idx].to(device).permute(1, 0, 2)  # [T, B, 2V]
            times = torch.arange(T, device=device).float().unsqueeze(1).expand(T, bs)
            z_seq = torch.cat([bg, mask_gen.pos_encoder(times)], dim=-1)
            src = inputs[idx].to(device).permute(1, 0, 2)

            # txai_pp MaskGenerator_PP returns batch-first
            ste_mask, mask_logits = mask_gen(z_seq, src, times)  # [B, T, d_inp]
            mask_tv = ste_mask[..., :V]

            # Pad per-batch from variable-length GPU lists (no global padding)
            emb = torch.nn.utils.rnn.pad_sequence(
                [emb_list[i] for i in idx], batch_first=True)
            tau_b = torch.nn.utils.rnn.pad_sequence(
                [tau_list[i] for i in idx], batch_first=True)
            varis_b = torch.nn.utils.rnn.pad_sequence(
                [varis_list[i] for i in idx], batch_first=True)
            om_b = torch.nn.utils.rnn.pad_sequence(
                [om_list[i] for i in idx], batch_first=True)

            batch_idx = torch.arange(bs, device=device).unsqueeze(1).expand_as(tau_b)
            event_mask = mask_tv[batch_idx, tau_b, varis_b]
            masked_emb = emb * event_mask.unsqueeze(-1)

            pooled_mask, _ = backbone.pool_events(masked_emb, om_b)
            logits_mask = backbone.predict_from_pooled(pooled_mask)

            # Loss 1: clf_loss — JS divergence
            p_mask = torch.sigmoid(logits_mask).unsqueeze(-1)
            p_full = torch.sigmoid(logits_full.detach()).unsqueeze(-1)
            p_mask_2 = torch.cat([p_mask, 1 - p_mask], dim=-1).clamp(1e-8, 1.0)
            p_full_2 = torch.cat([p_full, 1 - p_full], dim=-1).clamp(1e-8, 1.0)
            clf_loss = js_divergence_pp(p_mask_2, p_full_2)

            # Loss 2: sim_loss — distribution matching (KL + MSE)
            src_flat = inputs[idx].to(device).reshape(bs, -1)
            masked_flat = (inputs[idx].to(device) * ste_mask).reshape(bs, -1)

            org_mean, org_logvar = src_dist(src_flat)
            emb_mean, emb_logvar = mask_dist(masked_flat)

            kl_loss = _kl_divergence_gaussian(
                emb_mean, emb_logvar,
                org_mean.mean(0, keepdim=True).detach(),
                org_logvar.mean(0, keepdim=True).detach(),
            )
            ref_loss = F.mse_loss(masked_flat, src_flat.detach())
            sim_loss = beta_sim * (bias_weight * kl_loss + ref_loss)

            # Loss 3: exp_loss — mask regularization
            exp_loss = beta_exp * (gsat_loss_fn(mask_logits) + connect_loss_fn(mask_logits))

            loss = clf_loss + sim_loss + exp_loss

            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        print(f"  TimeX++ epoch {epoch+1}/{n_epochs}: loss={avg:.4f}", flush=True)

    mask_gen.eval()
    components = (mask_gen, src_dist, mask_dist, mask_conn)
    return components, train_mean, train_std


@torch.no_grad()
def compute_timexpp_attribution(
    backbone: BackboneAdapter,
    timexpp_model,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
) -> torch.Tensor:
    """Extract TimeX++ masks as attribution. Returns [B, T, V]."""
    mask_gen = timexpp_model[0]
    device = batch["obs_mask"].device
    D = backbone.D

    inp = _build_timex_input(batch, T, V, device, train_mean, train_std)
    B = inp.shape[0]

    bg = _backbone_to_grid(backbone, batch, T, V, D, device)
    bg_tf = bg.permute(1, 0, 2)
    times = torch.arange(T, device=device).float().unsqueeze(1).expand(T, B)
    z_seq = torch.cat([bg_tf, mask_gen.pos_encoder(times)], dim=-1)
    src = inp.permute(1, 0, 2)

    # txai_pp MaskGenerator_PP returns batch-first
    ste_mask, _ = mask_gen(z_seq, src, times)  # [B, T, d_inp]
    mask = ste_mask[..., :V]

    obs_grid = _build_obs_grid(batch, T, V, device)
    return mask * obs_grid


# =============================================================================
# 3. ExtremalMask — Enguehard, ICML 2023
# =============================================================================

class _GridToLogitModel(nn.Module):
    """Wraps backbone to accept [B, T, V] observation grids -> [B, 2] logits.

    Modes:
      - Soft embedding masking (no replacement embs): grid values multiply
        event embeddings. Differentiable. Used for removal strategy.
      - Soft interpolation (with replacement embs): grid values interpolate
        orig/replacement embeddings. Differentiable. Used for marginal/conditional.
    """

    def __init__(self, backbone: BackboneAdapter, batch_template, T, V,
                 orig_emb=None, replacement_emb=None):
        super().__init__()
        self.backbone = backbone
        self.T = T
        self.V = V
        self.batch_template = {k: v[0:1].detach() for k, v in batch_template.items()}
        self._orig_emb = orig_emb
        self._replacement_emb = replacement_emb

    def forward(self, grid):
        """grid: [B, T, V] -> logits [B, 2]."""
        B = grid.shape[0]
        logits = []

        for b in range(B):
            if self._orig_emb is not None and self._replacement_emb is not None:
                logit = soft_replacement_predict(
                    self.backbone, self.batch_template, grid[b],
                    self._orig_emb, self._replacement_emb)
            else:
                logit = soft_masked_predict(
                    self.backbone, self.batch_template, grid[b])
            logits.append(logit)

        logit_cat = torch.cat(logits)
        return torch.stack([-logit_cat, logit_cat], dim=-1)


def compute_extremal_mask_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    n_epoch: int = 500,
    masking_strategy: str = "removal",
    marginal_dist: torch.Tensor = None,
) -> torch.Tensor:
    """ExtremalMask attribution via tint package.

    Learns a differentiable perturbation mask that preserves the model's
    prediction using extremal perturbation theory.

    Args:
        masking_strategy: "removal", "marginal", or "conditional"
        marginal_dist: [V, n_bins] per-variable bin probabilities
        n_epoch: per-sample optimization epochs (default 500)

    Returns: attr_tv [B, T, V]
    """
    if not HAS_EXTREMAL:
        raise ImportError("tint package required for ExtremalMask: pip install time-interpret")

    B = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device
    backbone.eval()

    # Precompute replacement embeddings for marginal/conditional
    orig_emb = None
    replacement_emb = None
    if masking_strategy in ("marginal", "conditional"):
        orig_emb, replacement_emb = _precompute_replacement_embeddings(
            backbone, batch, marginal_dist, masking_strategy)

    all_attrs = []

    for b in range(B):
        single_batch = {k: v[b:b+1] for k, v in batch.items()}
        obs_grid = _build_obs_grid(single_batch, T, V, device)  # [1, T, V]

        if masking_strategy in ("marginal", "conditional"):
            model = _GridToLogitModel(
                backbone, single_batch, T, V,
                orig_emb=orig_emb[b:b+1],
                replacement_emb=replacement_emb[b:b+1])
        else:
            model = _GridToLogitModel(backbone, single_batch, T, V)

        explainer = _TintExtremalMask(model)

        # Tuned hyperparams: lambda_1=0.01, lr=0.01 prevent flat masks.
        # Default tint settings (lambda_1=1.0, lr=1e-3) produce flat masks
        # because L1 gradient (~8e-5) overwhelms fidelity gradient (~1e-6).
        mask_net = _TintExtremalMaskNet(
            forward_func=model,
            preservation_mode=True,
            lambda_1=0.01,
            loss="mse",
            optim="adam",
            lr=0.01,
        )
        trainer = _PLTrainer(
            max_epochs=n_epoch,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="gpu" if str(device).startswith("cuda") else "cpu",
            devices=1,
        )

        try:
            attr = explainer.attribute(
                obs_grid,
                trainer=trainer,
                mask_net=mask_net,
            )
            all_attrs.append(attr.abs().to(device))
        except Exception as e:
            print(f"  ExtremalMask failed for sample {b}: {e}")
            all_attrs.append(torch.zeros(1, T, V, device=device))

    return torch.cat(all_attrs, dim=0)

