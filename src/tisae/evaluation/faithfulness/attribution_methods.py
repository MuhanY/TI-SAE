"""
Attribution methods for evaluating SAE vs baselines.

All methods produce attr[B, T, V]
(importance of variable v at time bin t for the backbone's mortality prediction).

The backbone takes irregular event sequences (value_bins, tau, varis, obs_mask),
not a regular [T, V] grid. Methods that need grid-level masking use the masked_predict() adapter.

Supports TISAE models via compute_sae_attribution().

Methods implemented:
  - Random / Uniform baselines
  - Attention (from AttentionPooling weights)
  - Captum: IG, GradientSHAP, Grad x Input, DeepLIFT (captum package)
  - TISAE (amplitude mode)
  - DynaMask (dynamask package, Crabbe & Van Der Schaar ICML 2021)
  - TimeSHAP (timeshap package, Bento et al. KDD 2021)
  - FIT (tint package, Tonekaboni et al. NeurIPS 2020)
  - WinIT (tint package, Leung et al. ICLR 2023)
  - TSR (TSInterpret package, Ismail et al. NeurIPS 2020)
  - Concept/ACE (K-means clustering + TCAV with statistical test)
"""

from __future__ import annotations

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from tisae.models.backbone_adapter import BackboneAdapter

# Conditional imports for optional packages
try:
    from captum.attr import IntegratedGradients, GradientShap, InputXGradient, DeepLift
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

try:
    from dynamask.attribution.mask import Mask
    from dynamask.attribution.perturbation import FadeReference
    from dynamask.utils.losses import mse as dynamask_mse
    HAS_DYNAMASK = True
except ImportError:
    HAS_DYNAMASK = False

try:
    from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level
    HAS_TIMESHAP = True
except ImportError:
    HAS_TIMESHAP = False


try:
    from tint.attr import Fit as _TintFit
    from tint.attr.models import JointFeatureGeneratorNet
    from captum.attr._utils.attribution import PerturbationAttribution
    HAS_FIT = True
except ImportError:
    HAS_FIT = False


try:
    from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR as _TSR
    HAS_TSR = True
except ImportError:
    HAS_TSR = False


# =============================================================================
# Core adapter: (T,V)-mask wrapper for perturbation methods
# =============================================================================

def masked_predict(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    mask_tv: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass with (T,V) mask applied to event sequences.

    Args:
        backbone: BinnedTimeTransformer model
        batch: dict with value_bins[B,L], tau[B,L], varis[B,L], obs_mask[B,L]
        mask_tv: [T, V] or [B, T, V] binary tensor. 1 = keep, 0 = remove.
                 If [B, T, V], each sample gets its own mask (batched eval).

    Returns:
        logits: [B] mortality logits
    """
    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]

    # Lookup per-event mask
    if mask_tv.dim() == 3:
        # Batched: mask_tv[b, tau[b,l], varis[b,l]] -> [B, L]
        B = tau.shape[0]
        event_mask = mask_tv[torch.arange(B, device=tau.device)[:, None], tau, varis]
    else:
        # Single mask broadcast to all samples
        event_mask = mask_tv[tau, varis]  # [B, L]
    masked_om = obs_mask * event_mask.float()

    return backbone.predict_with_obs_mask(batch, masked_om)  # [B]


def _mask_tv_to_replace_event_mask(
    batch: Dict[str, torch.Tensor],
    mask_tv: torch.Tensor,
) -> torch.Tensor:
    """Convert a (T,V) cell mask to a per-event boolean replace_mask[B, L].

    The cell mask convention is: 1 = keep, 0 = replace. An event at (t, v) is
    REPLACED iff the cell is 0 AND the event is observed (obs_mask > 0.5).
    Padded / unobserved events are never replaced.

    Returns:
        replace_mask[B, L] (bool): 1 = replace this event, 0 = keep it.
    """
    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]
    device = tau.device

    if mask_tv.dim() == 3:
        B = tau.shape[0]
        event_mask = mask_tv[torch.arange(B, device=device)[:, None], tau, varis]
    else:
        event_mask = mask_tv[tau, varis]  # [B, L], 1=keep, 0=replace
    return (obs_mask > 0.5) & (event_mask < 0.5)


def marginal_masked_predict(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    mask_tv: torch.Tensor,
    *,
    marginal_dist: Optional[torch.Tensor] = None,             # binned backbones
    population_values_zscore: Optional[Dict[int, "np.ndarray"]] = None,  # CVE backbones
) -> torch.Tensor:
    """Forward pass replacing masked events with marginally-sampled values.

    Backbone-uniform via `adapter.replace_event_value(strategy='marginal')`:
      - Binned backbones: replaces `value_bins` from `marginal_dist`
        (`[V, n_bins]` per-variable bin probabilities).
      - CVE backbones: replaces `values_zscore` from
        `population_values_zscore` (per-variable training pool of z-scored
        floats — exposed by the unified dataset).

    The adapter ignores the irrelevant kwarg per backbone type, so callers can
    safely bind both via `partial(...)`.

    Args:
        backbone: BackboneAdapter (parameter name kept as `backbone` for backwards compat
                  with the existing call sites; rename to `adapter`).
        batch: unified-collate batch dict
        mask_tv: [T, V] or [B, T, V] binary tensor. 1 = keep, 0 = replace.
        marginal_dist: per-variable bin distribution for binned backbones.
        population_values_zscore: per-variable z-score pool for CVE backbones.

    Returns:
        logits: [B] mortality logits
    """
    replace_mask = _mask_tv_to_replace_event_mask(batch, mask_tv)
    new_batch = backbone.replace_event_value(
        batch, replace_mask, strategy="marginal",
        marginal_dist=marginal_dist,
        population_values_zscore=population_values_zscore,
    )
    return backbone.forward(new_batch)["logits"]


def conditional_masked_predict(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    mask_tv: torch.Tensor,
    *,
    marginal_dist: Optional[torch.Tensor] = None,             # binned backbones
    population_values_zscore: Optional[Dict[int, "np.ndarray"]] = None,  # CVE backbones
) -> torch.Tensor:
    """Forward pass replacing masked events with conditionally-sampled
    (nearest-same-variable-in-patient-history) values, falling back to
    marginal sampling for variables with no observed history.

    Backbone-uniform via `adapter.replace_event_value(strategy='conditional')`.
    Both `marginal_dist` (binned) and `population_values_zscore` (CVE) are
    accepted; the adapter picks the right one for its backbone type.

    Args:
        backbone: BackboneAdapter
        batch: unified-collate batch dict
        mask_tv: [T, V] or [B, T, V] binary tensor. 1 = keep, 0 = replace.
        marginal_dist: per-variable bin distribution (binned fallback)
        population_values_zscore: per-variable z-score pool (CVE fallback)

    Returns:
        logits: [B] mortality logits
    """
    replace_mask = _mask_tv_to_replace_event_mask(batch, mask_tv)
    new_batch = backbone.replace_event_value(
        batch, replace_mask, strategy="conditional",
        marginal_dist=marginal_dist,
        population_values_zscore=population_values_zscore,
    )
    return backbone.forward(new_batch)["logits"]


def soft_masked_predict(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    mask_tv: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable forward pass with soft (T,V) mask applied to embeddings.

    Unlike masked_predict which modifies obs_mask (non-differentiable due to
    boolean masking in the backbone), this applies the mask to the continuous
    event embeddings, preserving gradient flow for optimization-based methods
    like DynaMask.

    Args:
        backbone: BinnedTimeTransformer model
        batch: dict with value_bins[B,L], tau[B,L], varis[B,L], obs_mask[B,L]
        mask_tv: [T, V] soft mask in (0, 1). 1 = keep, 0 = suppress.

    Returns:
        logits: [B] mortality logits
    """
    tau = batch["tau"]
    varis = batch["varis"]

    # Get event embeddings via the adapter dict API (backbone-uniform).
    # Detached because we don't need grad w.r.t. the embeddings themselves;
    # the gradient flows through the soft `mask_tv` argument.
    with torch.no_grad():
        embeddings = backbone.get_event_embeddings(batch)  # [B, L, D]

    # Apply soft mask to embeddings (gradient flows through mask_tv)
    event_mask = mask_tv[tau, varis]  # [B, L], soft values in (0, 1)
    masked_emb = embeddings * event_mask.unsqueeze(-1)  # [B, L, D]

    # Forward from masked embeddings (obs_mask stays original for attention structure)
    return _forward_from_embeddings(backbone, masked_emb, batch)


# =============================================================================
# Helper: forward from embedding space (adapter delegation)
# =============================================================================

def _forward_from_embeddings(
    adapter: nn.Module,
    embeddings: torch.Tensor,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Captum-friendly forward from a [B, L, D] event-embedding tensor.

    Delegates to `adapter.forward_from_event_embeddings(embeddings, batch)`,
    which walks the post-embedding pipeline (dropout → transformer / mamba →
    fusion attention → head) for the appropriate backbone.

    embeddings: [B, L, D]
    batch: full unified-collate dict (must contain at least `obs_mask`)
    Returns: logits [B]
    """
    return adapter.forward_from_event_embeddings(embeddings, batch)


def soft_replacement_predict(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    mask_tv: torch.Tensor,
    orig_emb: torch.Tensor,
    replacement_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable forward with soft interpolation between original and replacement embeddings.

    mixed_emb = mask * orig_emb + (1-mask) * replacement_emb

    At hard boundaries (mask=0 or 1), equivalent to discrete replacement.
    Differentiable w.r.t. mask_tv for optimization-based methods (DynaMask).

    Args:
        backbone: BinnedTimeTransformer model
        batch: dict with tau[B,L], varis[B,L], obs_mask[B,L]
        mask_tv: [T, V] soft mask in (0, 1). 1 = keep original, 0 = use replacement.
        orig_emb: [B, L, D] original event embeddings (detached)
        replacement_emb: [B, L, D] replacement event embeddings (detached)

    Returns:
        logits: [B] mortality logits
    """
    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]

    event_mask = mask_tv[tau, varis]  # [B, L], soft values in (0, 1)
    m = event_mask.unsqueeze(-1)  # [B, L, 1]
    mixed_emb = m * orig_emb + (1.0 - m) * replacement_emb  # [B, L, D]

    return _forward_from_embeddings(backbone, mixed_emb, batch)


@torch.no_grad()
def _precompute_replacement_embeddings(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    marginal_dist: torch.Tensor,
    strategy: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute original and replacement embeddings for soft interpolation.

    For marginal: sample value_bins from per-variable population distribution.
    For conditional: use patient's nearest same-variable observed value.
    Falls back to marginal for variables with only one observation.

    Args:
        backbone: BinnedTimeTransformer model
        batch: dict with value_bins[B,L], tau[B,L], varis[B,L], obs_mask[B,L]
        marginal_dist: [V, n_bins] per-variable bin probabilities
        strategy: "marginal" or "conditional"

    Returns:
        orig_emb: [B, L, D] original event embeddings (detached)
        replacement_emb: [B, L, D] replacement event embeddings (detached)
    """
    vb = batch["value_bins"]
    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]
    device = vb.device
    B, L = vb.shape

    orig_emb = backbone.get_event_embeddings(batch).detach()  # dict API

    replacement_vb = vb.clone()

    if strategy == "marginal":
        for b in range(B):
            valid = obs_mask[b] > 0.5
            if valid.any():
                event_vars = varis[b, valid]
                probs = marginal_dist[event_vars]
                sampled = torch.multinomial(probs, 1).squeeze(-1)
                replacement_vb[b, valid] = sampled

    elif strategy == "conditional":
        for b in range(B):
            valid_idx = (obs_mask[b] > 0.5).nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue

            valid_vars = varis[b, valid_idx]
            valid_times = tau[b, valid_idx]
            valid_vals = vb[b, valid_idx]
            replaced = torch.zeros(len(valid_idx), dtype=torch.bool, device=device)

            for v in valid_vars.unique():
                v_mask = valid_vars == v
                v_indices = valid_idx[v_mask]
                v_times = valid_times[v_mask]
                v_vals = valid_vals[v_mask]

                if len(v_indices) <= 1:
                    # Only one observation — use marginal fallback
                    if marginal_dist is not None:
                        probs = marginal_dist[v.unsqueeze(0)]
                        replacement_vb[b, v_indices] = torch.multinomial(
                            probs, 1).squeeze(-1)
                    replaced[v_mask] = True
                    continue

                # Nearest OTHER event of same variable
                dists = (v_times.unsqueeze(1) - v_times.unsqueeze(0)).abs().float()
                dists.fill_diagonal_(float('inf'))
                nearest = dists.argmin(dim=1)
                replacement_vb[b, v_indices] = v_vals[nearest]
                replaced[v_mask] = True

            # Marginal fallback for unreplaced events
            if not replaced.all():
                unreplaced = valid_idx[~replaced]
                if marginal_dist is not None:
                    fb_vars = varis[b, unreplaced]
                    replacement_vb[b, unreplaced] = torch.multinomial(
                        marginal_dist[fb_vars], 1).squeeze(-1)

    replacement_emb = backbone.get_event_embeddings(
        {**batch, "value_bins": replacement_vb}
    ).detach()
    return orig_emb, replacement_emb


# =============================================================================
# Helper: aggregate event-level [B, L] attributions to [B, T, V] grid
# =============================================================================

def aggregate_to_grid(
    attr_event: torch.Tensor,
    tau: torch.Tensor,
    varis: torch.Tensor,
    obs_mask: torch.Tensor,
    T: int,
    V: int,
) -> torch.Tensor:
    """
    Scatter-add event-level attributions into a [B, T, V] grid.

    Args:
        attr_event: [B, L] per-event attribution (non-negative recommended)
        tau: [B, L] time bin indices
        varis: [B, L] variable indices
        obs_mask: [B, L] observation mask (1=valid)
        T: number of time bins
        V: number of variables

    Returns:
        attr_tv: [B, T, V] grid attribution
    """
    B, L = attr_event.shape
    device = attr_event.device

    attr_event = attr_event * obs_mask  # zero out padding

    # Flatten to 1D index: cell = tau * V + varis
    cell_idx = tau * V + varis  # [B, L]

    attr_tv = torch.zeros(B, T * V, device=device, dtype=attr_event.dtype)
    attr_tv.scatter_add_(1, cell_idx, attr_event)

    return attr_tv.view(B, T, V)


# =============================================================================
# Helper: build observation and value grids from irregular event sequences
# =============================================================================

def _get_sae_W(sae):
    """Get decoder W [K, V] from SAE."""
    W = sae.decoder.get_W()
    if W is not None:
        return W.detach()
    raise ValueError("Cannot get decoder W from SAE")


def _build_obs_grid(
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert irregular event sequences to (B, T, V) binary observation grid."""
    tau = batch["tau"]        # [B, L]
    varis = batch["varis"]    # [B, L]
    om = batch["obs_mask"]    # [B, L]
    B, L = tau.shape
    grid = torch.zeros(B, T, V, device=device)
    for b in range(B):
        valid = om[b].bool()
        t_idx = tau[b, valid].long()
        v_idx = varis[b, valid].long()
        grid[b, t_idx, v_idx] = 1.0
    return grid


def _build_value_grid(
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert irregular events to (B, T, V) grid with values.

    Uses values_raw if available, otherwise falls back to value_bins (float).
    """
    tau = batch["tau"]
    varis = batch["varis"]
    om = batch["obs_mask"]
    vals = batch.get("values_raw", batch["value_bins"])
    B, L = tau.shape
    grid = torch.zeros(B, T, V, device=device)
    for b in range(B):
        valid = om[b].bool()
        t_idx = tau[b, valid].long()
        v_idx = varis[b, valid].long()
        grid[b, t_idx, v_idx] = vals[b, valid].float()
    return grid


# =============================================================================
# Helper: compute per-variable median value bins from training data
# =============================================================================

def compute_median_bins(
    dataset,
    V: int,
    max_samples: int = 10000,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute per-variable median value bin from training split.

    Returns:
        median_bins: [V] int64 tensor of median value bin per variable.
    """
    train_indices = np.array(dataset.splits["train"])
    n_use = min(len(train_indices), max_samples)
    train_indices = train_indices[:n_use]

    # Collect all (variable, value_bin) pairs
    per_var_vals: Dict[int, List[int]] = {v: [] for v in range(V)}
    for start in range(0, n_use, batch_size):
        end = min(start + batch_size, n_use)
        batch = dataset.get_batch(train_indices[start:end])
        varis = batch["varis"]      # [B, L]
        vb = batch["value_bins"]    # [B, L]
        om = batch["obs_mask"]      # [B, L]
        for b in range(varis.shape[0]):
            valid = om[b].bool()
            vs = varis[b, valid].numpy()
            bins = vb[b, valid].numpy()
            for vi, bi in zip(vs, bins):
                per_var_vals[int(vi)].append(int(bi))

    median_bins = torch.zeros(V, dtype=torch.long)
    for v in range(V):
        if per_var_vals[v]:
            median_bins[v] = int(np.median(per_var_vals[v]))
        else:
            median_bins[v] = 5  # fallback to middle bin
    return median_bins


# =============================================================================
# 1. Random baseline
# =============================================================================

def compute_random_attribution(
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
) -> torch.Tensor:
    """Random attribution: shuffled uniform noise, masked to observed cells. [B, T, V]"""
    B = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device
    attr = torch.rand(B, T, V, device=device)
    obs_grid = _build_obs_grid(batch, T, V, device)
    return attr * obs_grid


# =============================================================================
# 2. Uniform baseline
# =============================================================================

def compute_uniform_attribution(
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
) -> torch.Tensor:
    """Uniform attribution: equal importance at observed cells. [B, T, V]"""
    B = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device
    obs_grid = _build_obs_grid(batch, T, V, device)
    return obs_grid  # 1 at observed, 0 elsewhere


# =============================================================================
# 3. Attention-based attribution
# =============================================================================

@torch.no_grad()
def compute_attention_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
) -> torch.Tensor:
    """
    Attribution from AttentionPooling weights.

    The backbone's forward returns attention[B, L] — per-event importance
    for mortality prediction. Delegated to `adapter.get_attention_grid` which
    runs the forward, reads `out['attention']`, and scatters to the [B, T, V]
    cell grid (with safe_tau / safe_varis routing for padded positions).
    """
    return backbone.get_attention_grid(batch, T, V)


# =============================================================================
# 4. Captum: IG, GradientSHAP, Grad x Input
# =============================================================================

class _EmbToLogit(nn.Module):
    """Captum wrapper: `[B,L,D]` event embeddings → `[B]` logits.

    Used by `_emb`-variant captum methods (IG / GradientSHAP / DeepLIFT /
    InputXGradient) on all backbones. Differentiates w.r.t. the post-embedding
    triplet tensor — works on binned and CVE backbones uniformly because both produce a `[B, L, D]` event embedding.
    """

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, embeddings, batch):
        return _forward_from_embeddings(self.adapter, embeddings, batch)


class _ValueToLogit(nn.Module):
    """Captum wrapper for CVE backbones: `[B, L]` `values_zscore` → `[B]` logits.

    Used by `_value`-variant captum methods on STraTS/HyMaTE only.
    Differentiates w.r.t. the raw `values_zscore` input. Per-event attribution
    comes naturally as a single scalar (no D-axis aggregation needed).
    """

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, values_zscore, batch):
        new_batch = {**batch, "values_zscore": values_zscore}
        return self.adapter.forward(new_batch)["logits"]


def _slice_batch(batch: Dict[str, torch.Tensor], b: int) -> Dict[str, torch.Tensor]:
    """Slice every tensor in `batch` to `[b:b+1]`. Used by the per-sample
    captum loop so each iteration sees a single-sample batch dict that the
    `_EmbToLogit` / `_ValueToLogit` wrappers can route through `adapter.forward`.
    """
    return {k: (v[b:b+1] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def compute_captum_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    method: str = "ig",
    n_steps: int = 50,
    baseline_mode: str = "zero",
    median_bins: Optional[torch.Tensor] = None,
    variant: str = "emb",  # "emb" | "value"
) -> torch.Tensor:
    """Attribution via captum package (Meta).

    Args:
        method: "ig" (Integrated Gradients), "gradient_shap", "grad_x_input", or "deeplift"
        n_steps: interpolation steps for IG, samples for GradientShap
        baseline_mode: "zero" (zero embedding / zero z-score) or "median"
                       (per-variable median value bin for binned, zero z-score for CVE)
        median_bins: [V] int tensor of per-variable median value bins.
                     Required when `baseline_mode='median'` and `variant='emb'`
                     on a binned backbone.
        variant: "emb" → differentiate w.r.t. post-embedding `[B, L, D]` tensor.
                 "value" → differentiate w.r.t. raw `values_zscore[B, L]` (CVE backbones only).

    The `variant` arg implements the "best of all viable adaptations" protocol
    from the migration plan: callers run BOTH `_emb` and `_value` on CVE
    backbones and the aggregation step picks the highest-AOPC variant per
    (family, backbone) cell. Binned backbones only support `_emb`.

    Returns:
        attr_tv: [B, T, V]
    """
    if not HAS_CAPTUM:
        raise ImportError("captum package required: pip install captum>=0.7.0")

    if variant not in ("emb", "value"):
        raise ValueError(f"Unknown captum variant: {variant!r}; expected 'emb' or 'value'")

    if variant == "value" and not backbone.has_continuous_values:
        raise ValueError(
            "captum variant='value' requires a CVE backbone (StratsBackboneAdapter "
            f"or HyMaTEBackboneAdapter); got {type(backbone).__name__} which uses "
            "discrete value bins. Use variant='emb' for binned backbones."
        )

    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]

    was_training = backbone.training
    backbone.eval()

    if variant == "emb":
        # Embedding-level: differentiate through [B, L, D] event embeddings.
        # Uses the dict API on `get_event_embeddings` so this path is
        # backbone-uniform (TISAE / STraTS / HyMaTE all accept the dict form).
        with torch.no_grad():
            emb = backbone.get_event_embeddings(batch)  # [B, L, D]
        if baseline_mode == "median":
            # If the caller passed `median_bins` (binned-only path) and the
            # adapter doesn't have its own median_bins installed yet, install them on the fly.
            if (
                not backbone.has_continuous_values
                and getattr(backbone, "median_bins", None) is None
            ):
                if median_bins is None:
                    raise ValueError(
                        "median_bins required when baseline_mode='median' on a binned backbone"
                    )
                backbone.set_median_bins(median_bins)
            with torch.no_grad():
                baseline = backbone.get_baseline_event_embeddings(batch, mode="median")
        else:
            baseline = torch.zeros_like(emb)
        wrapper = _EmbToLogit(backbone)
        inputs = emb  # [B, L, D]
    else:
        # Value-level: differentiate through [B, L] values_zscore
        # Baseline is z-score 0 (population mean) for both 'zero' and 'median'
        # modes — z-scoring already centers the data.
        inputs = batch["values_zscore"]  # [B, L]
        baseline = torch.zeros_like(inputs)
        wrapper = _ValueToLogit(backbone)

    # Process per-sample to control memory
    B = obs_mask.shape[0]
    all_attr_events = []

    for b in range(B):
        x_b = inputs[b:b+1].detach().requires_grad_(True)
        base_b = baseline[b:b+1]
        batch_b = _slice_batch(batch, b)

        if method == "gradient_shap":
            attr_method = GradientShap(wrapper)
            attr = attr_method.attribute(
                x_b, baselines=base_b,
                additional_forward_args=(batch_b,),
                n_samples=n_steps,
            )
        elif method == "grad_x_input":
            attr_method = InputXGradient(wrapper)
            attr = attr_method.attribute(
                x_b,
                additional_forward_args=(batch_b,),
            )
        elif method == "deeplift":
            attr_method = DeepLift(wrapper)
            attr = attr_method.attribute(
                x_b, baselines=base_b,
                additional_forward_args=(batch_b,),
            )
        else:  # "ig"
            attr_method = IntegratedGradients(wrapper)
            attr = attr_method.attribute(
                x_b, baselines=base_b,
                additional_forward_args=(batch_b,),
                n_steps=n_steps,
            )

        if variant == "emb":
            # Reduce D dimension: sum then abs (preserves IG completeness axiom)
            attr_event_b = attr.sum(dim=-1).abs().detach()  # [1, L]
        else:
            # Already per-event ([1, L])
            attr_event_b = attr.abs().detach()
        all_attr_events.append(attr_event_b)

    attr_event = torch.cat(all_attr_events, dim=0)  # [B, L]

    if was_training:
        backbone.train()

    return aggregate_to_grid(attr_event, tau, varis, obs_mask, T, V)


# =============================================================================
# 5. SAE attribution
# =============================================================================

def _build_H_grid(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Delegates to `adapter.extract_hidden(batch)` which scatters backbone
    event embeddings into a [B, T, V, D] grid via `tisae.utils.grid_ops.build_H_grid`.

    """
    return backbone.extract_hidden(batch)


@torch.no_grad()
def _run_grid_sae_forward(
    backbone: BackboneAdapter,
    sae: nn.Module,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    H_grid_mean: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Run backbone -> H_grid -> TISAE and return all intermediates.

    Returns:
        out: SAE output dict
        H_grid: [B, T, V, D]
        obs_grid: [B, T, V]
    """
    H_grid, obs_grid = _build_H_grid(backbone, batch, T, V)

    # Residualize
    if H_grid_mean is not None:
        H_grid = H_grid - H_grid_mean.unsqueeze(0)
        H_grid = H_grid * obs_grid.unsqueeze(-1)

    out = sae(H_grid, obs_grid)
    return out, H_grid, obs_grid


@torch.no_grad()
def compute_grid_sae_attribution(
    backbone: BackboneAdapter,
    sae: nn.Module,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    H_grid_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SAE amplitude attribution.

    attr[t,v] = sum_k |A_hat[k,t]| * W[k,v]
    """
    out, H_grid, obs_grid = _run_grid_sae_forward(
        backbone, sae, batch, T, V, H_grid_mean)

    activation = out["A_hat"].abs()  # [B, K, T]

    # W: [K, V] from decoder
    sae_mod = sae
    W = sae_mod.decoder.get_W().detach()  # [K, V]

    # attr[b,t,v] = sum_k activation[b,k,t] * W[k,v]
    attr = torch.einsum("bkt,kv->btv", activation, W)

    return attr * obs_grid  # [B, T, V]


def compute_sae_attribution(
    backbone: BackboneAdapter,
    sae: nn.Module,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    H_grid_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SAE attribution via latent activations + decoder variable composition.

    Uses |A_hat| (amplitude mode): attr[b,t,v] = sum_k |A_hat| * W.

    Args:
        H_grid_mean: [T, V, D] population mean for residualization

    Returns:
        attr_tv: [B, T, V]
    """
    return compute_grid_sae_attribution(
        backbone, sae, batch, T, V, H_grid_mean=H_grid_mean)


# =============================================================================
# 5b. Island-wise attribution + island metadata
# =============================================================================

def extract_sae_islands(
    backbone: BackboneAdapter,
    sae: nn.Module,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    g_threshold: float = 0.607,
    H_grid_mean: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[List[Dict]]]:
    """Extract SAE attribution + structured island metadata for island-wise eval.

    Each island is a (k, c, m) triple with:
      - temporal footprint: time bins where g(t|mu,sigma) > g_threshold
      - variable footprint: variables with nonzero W[c,k,v]
      - importance: prediction-weighted contribution (d(pred)/d(A_hat) x envelope)
      - cell set: explicit list of (t, v) cells covered

    Args:
        g_threshold: Gaussian cutoff (0.607 ~ 1 sigma, total width 2sigma)
        H_grid_mean: [T, V, D] population mean for residualization

    Returns:
        attr_tv: [B, T, V] grid attribution (island-structured prediction attribution)
        islands: List[List[Dict]] — per-sample list of island dicts
    """
    return _extract_grid_sae_islands(
        backbone, sae, batch, T, V, g_threshold, H_grid_mean)


def _extract_grid_sae_islands(
    backbone: BackboneAdapter,
    sae: nn.Module,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    g_threshold: float = 0.607,
    H_grid_mean: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[List[Dict]]]:
    """TISAE island extraction with prediction-gradient importance.

    """
    was_training_bb = backbone.training
    was_training_sae = sae.training
    backbone.eval()
    sae.eval()

    sae_mod = sae

    with torch.no_grad():
        H_grid, obs_grid = _build_H_grid(backbone, batch, T, V)
        if H_grid_mean is not None:
            H_grid_input = H_grid - H_grid_mean.unsqueeze(0)
            H_grid_input = H_grid_input * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

    W = sae_mod.decoder.get_W().detach()  # [K, V]
    W_norm = W / W.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    W_mask = (W > 0).float()

    B = batch["obs_mask"].shape[0]
    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask_batch = batch["obs_mask"]
    all_attr = []
    all_islands = []

    for b in range(B):
        H_grid_b = H_grid_input[b:b+1]
        obs_grid_b_full = obs_grid[b]  # [T, V]

        sae_out = sae(H_grid_b, obs_grid[b:b+1])
        A_hat_b = sae_out["A_hat"]  # [1, K, T]
        A_hat_b.retain_grad()
        H_hat_grid_b = sae_out["H_hat_grid"]  # [1, T, V, D]

        mu_b = sae_out["mu"].detach()     # [1, K_eff, C_eff, M]
        sigma_b = sae_out["sigma"].detach()
        sel_b = sae_out["sel"].detach()

        # De-residualize and predict (with grad through A_hat)
        if H_grid_mean is not None:
            H_hat_grid_b = H_hat_grid_b + H_grid_mean.unsqueeze(0)

        # Scatter to event level
        H_hat_events = H_hat_grid_b[0, tau[b], varis[b]].unsqueeze(0)  # [1, L, D]
        # Pattern E1 → adapter dict API: pool then predict
        pooled_b, _ = backbone.pool_events(H_hat_events, obs_mask_batch[b:b+1])
        logit_b = backbone.predict_from_pooled(pooled_b)  # [1]

        logit_b.backward()
        pred_sens = A_hat_b.grad.abs().detach()  # [1, K, T]

        # Prediction-weighted attribution
        attr_b = torch.einsum('kt,kv->tv', pred_sens[0], W_norm)  # [T, V]
        all_attr.append(attr_b.unsqueeze(0))

        # Extract islands
        K_eff = mu_b.shape[1]
        C_eff = mu_b.shape[2]
        M_dim = mu_b.shape[3]
        device = A_hat_b.device

        t_axis = torch.arange(T, device=device, dtype=torch.float32)

        sample_islands = []
        for k_eff in range(K_eff):
            for c_eff in range(C_eff):
                for m in range(M_dim):
                    if sel_b[0, k_eff, c_eff, m] < 0.5:
                        continue

                    mu_val = mu_b[0, k_eff, c_eff, m]
                    sig_val = sigma_b[0, k_eff, c_eff, m].clamp_min(1e-6)
                    g_trace = torch.exp(-0.5 * ((t_axis - mu_val) / sig_val) ** 2)
                    t_mask = (g_trace > g_threshold).float()
                    if t_mask.sum() < 1:
                        continue

                    # Map k_eff, c_eff to full latent index
                    if C_eff > 1:
                        k_full = k_eff + c_eff * K_eff
                    else:
                        k_full = k_eff

                    if k_full >= W.shape[0]:
                        continue

                    v_mask = W_mask[k_full]
                    if v_mask.sum() < 1:
                        continue

                    cell_mask = t_mask.unsqueeze(-1) * v_mask.unsqueeze(0)
                    n_cells = int(cell_mask.sum().item())
                    obs_in_island = int((cell_mask * obs_grid_b_full).sum().item())

                    # Prediction-gradient importance
                    ps_trace = pred_sens[0, k_full, :] if k_full < pred_sens.shape[1] else torch.zeros(T, device=device)
                    importance = (ps_trace * g_trace * t_mask).sum().item()

                    # Also store activation-based importance and raw pred_sens trace
                    a_hat_trace = A_hat_b[0, k_full, :].detach() if k_full < A_hat_b.shape[1] else torch.zeros(T, device=device)
                    recon_importance = (a_hat_trace.abs() * g_trace * t_mask).sum().item()

                    sample_islands.append({
                        "k": k_full,
                        "c": c_eff,
                        "m": m,
                        "importance": importance,
                        "recon_importance": recon_importance,
                        "pred_sens_trace": ps_trace.cpu(),  # [T]
                        "a_hat_trace": a_hat_trace.abs().cpu(),  # [T]
                        "cell_mask": cell_mask,
                        "n_cells": n_cells,
                        "n_obs_cells": obs_in_island,
                        "mu": mu_val.item(),
                        "sigma": sig_val.item(),
                    })

        sample_islands.sort(key=lambda x: x["importance"], reverse=True)
        all_islands.append(sample_islands)

        sae.zero_grad()
        backbone.zero_grad()

    if was_training_bb:
        backbone.train()
    if was_training_sae:
        sae.train()

    attr = torch.cat(all_attr, dim=0)
    obs_grid_full = _build_obs_grid(batch, T, V, attr.device)
    return attr * obs_grid_full, all_islands


# =============================================================================
# 6. DynaMask — official package (Crabbe & Van Der Schaar, ICML 2021)
# =============================================================================

def _value_level_soft_masked_predict(
    adapter: nn.Module,
    batch: Dict[str, torch.Tensor],
    mask_tv: torch.Tensor,
) -> torch.Tensor:
    """Value-level differentiable forward for CVE backbones.

    Mirrors `soft_masked_predict` but masks at the **value** level instead of
    the embedding level: `values_zscore * event_mask`. The intervention is
    "fade each measured value smoothly toward the population mean (z-score 0)
    while keeping variable identity, time, and obs_mask unchanged".

    Used by `dynamask_value`. Only valid on CVE adapters
    (`adapter.has_continuous_values=True`).
    """
    tau = batch["tau"]
    varis = batch["varis"]
    if mask_tv.dim() == 3:
        B = tau.shape[0]
        event_mask = mask_tv[torch.arange(B, device=tau.device)[:, None], tau, varis]
    else:
        event_mask = mask_tv[tau, varis]  # [B, L]
    return adapter.forward_with_value_mask(batch, event_mask)


def compute_dynamask_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    n_epoch: int = 500,
    areas: List[float] = (0.1, 0.2, 0.3, 0.5),
    masking_strategy: str = "removal",
    marginal_dist: torch.Tensor = None,
    variant: str = "emb",  # "emb" | "value"
) -> torch.Tensor:
    """
    DynaMask attribution using official dynamask package.

    Uses FadeReference perturbation with X=ones and baseline=zeros.
    The perturbation reduces to: X_pert = mask * 1 + (1-mask) * 0 = mask.

    Two variants:
      - **emb** (default, all backbones): inner forward is `soft_masked_predict`,
        which multiplies the post-embedding `[B, L, D]` triplet by the per-event
        soft mask. Uses `_forward_from_embeddings → adapter.forward_from_event_embeddings`.
      - **value** (CVE backbones only): inner forward is
        `_value_level_soft_masked_predict`, which multiplies `values_zscore` by
        the per-event soft mask via `adapter.forward_with_value_mask`. Smoothly
        fades each measurement value toward the population mean (z-score 0).

    Both variants share the same Adam optimization loop, sparsity penalty, and
    multi-area selection — only the inner masked-forward callable changes.

    For removal: f(mask) = (variant-specific soft predict).
    For marginal/conditional: f(mask) = soft_replacement_predict(...) (emb only).

    Args:
        masking_strategy: "removal" (default), "marginal", or "conditional"
        marginal_dist: [V, n_bins] per-variable bin probabilities (for marginal/conditional)
        variant: "emb" (embedding-level, all backbones) or "value" (value-level, CVE only)

    Returns:
        attr_tv: [B, T, V] where higher = more important
    """
    if not HAS_DYNAMASK:
        raise ImportError("dynamask package required: pip install dynamask")

    if variant not in ("emb", "value"):
        raise ValueError(f"Unknown dynamask variant: {variant!r}; expected 'emb' or 'value'")

    if variant == "value":
        if not backbone.has_continuous_values:
            raise ValueError(
                "dynamask variant='value' requires a CVE backbone "
                f"(StratsBackboneAdapter or HyMaTEBackboneAdapter); got "
                f"{type(backbone).__name__}. Use variant='emb' for binned backbones."
            )
        if masking_strategy != "removal":
            # Marginal / conditional perturbations operate via _precompute_replacement_embeddings
            # which is binned-only. The value-level variant only supports the
            # removal strategy (smooth fade toward population mean).
            raise ValueError(
                f"dynamask variant='value' only supports masking_strategy='removal'; "
                f"got {masking_strategy!r}. Use variant='emb' for marginal/conditional."
            )

    B = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device

    was_training = backbone.training
    backbone.eval()

    # Freeze backbone parameters so they don't participate in DynaMask's
    # backward pass (avoids "backward through graph a second time" error)
    orig_requires_grad = {}
    for name, p in backbone.named_parameters():
        orig_requires_grad[name] = p.requires_grad
        p.requires_grad_(False)

    # Precompute replacement embeddings for marginal/conditional (emb only)
    orig_emb = None
    replacement_emb = None
    if variant == "emb" and masking_strategy in ("marginal", "conditional"):
        orig_emb, replacement_emb = _precompute_replacement_embeddings(
            backbone, batch, marginal_dist, masking_strategy)

    all_attrs = []

    for b in range(B):
        single_batch = {k: v[b:b+1] for k, v in batch.items()}

        X = torch.ones(T, V, device=device)
        X_ref = torch.zeros(T, V, device=device)

        if variant == "value":
            # CVE value-level: smooth fade of values_zscore toward 0
            def f(x_pert, _sb=single_batch):
                logit = _value_level_soft_masked_predict(backbone, _sb, x_pert)
                return logit.squeeze()
        elif masking_strategy == "removal":
            def f(x_pert, _sb=single_batch):
                logit = soft_masked_predict(backbone, _sb, x_pert)
                return logit.squeeze()
        else:
            # emb + marginal/conditional: soft interpolation between original
            # and replacement embeddings
            o_emb = orig_emb[b:b+1]   # [1, L, D]
            r_emb = replacement_emb[b:b+1]  # [1, L, D]

            def f(x_pert, _sb=single_batch, _o=o_emb, _r=r_emb):
                logit = soft_replacement_predict(backbone, _sb, x_pert, _o, _r)
                return logit.squeeze()

        # Try each area level, pick best (lowest error)
        best_mask = None
        best_error = float("inf")

        for area in areas:
            try:
                pert = FadeReference(device=device, X_ref=X_ref)
                mask = Mask(
                    perturbation=pert,
                    device=device,
                    task="regression",
                    verbose=False,
                )
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    mask.fit(
                        X=X, f=f, loss_function=dynamask_mse,
                        keep_ratio=area,
                        n_epoch=n_epoch,
                        initial_mask_coeff=0.5,
                        size_reg_factor_init=0.5,
                        size_reg_factor_dilation=100,
                        learning_rate=0.1,
                        momentum=0.9,
                        time_reg_factor=0.01,
                    )
                error = mask.get_error().item()
                if error < best_error:
                    best_error = error
                    best_mask = mask.mask_tensor.detach().clone()
            except Exception as e:
                print(f"  DynaMask area={area} failed for sample {b}: {e}")
                continue

        if best_mask is None:
            # Fallback: uniform attribution
            best_mask = torch.ones(T, V, device=device) * 0.5

        # mask ~ 1 = important (correct polarity natively)
        all_attrs.append(best_mask.unsqueeze(0))

    # Restore backbone parameter grad flags
    for name, p in backbone.named_parameters():
        p.requires_grad_(orig_requires_grad[name])
    if was_training:
        backbone.train()

    # Mask to observed cells — DynaMask assigns values to the full (T,v) grid
    # but unobserved cells cannot affect the prediction
    attr = torch.cat(all_attrs, dim=0)  # [B, T, V]
    obs_grid = _build_obs_grid(batch, T, V, device)
    return attr * obs_grid


# =============================================================================
# 7. TimeSHAP — official package (Bento et al., KDD 2021)
# =============================================================================

def _timeshap_df_to_grid(cell_data, T, V):
    """Convert TimeSHAP cell-level DataFrame to (T, V) attribution grid.

    cell_data has columns: 'Event', 'Feature', 'Shapley Value'.
    Event format: "Event -N" (N steps from end, 1-indexed) or "Pruned Events"/"Other Events".
    Feature format: "Feature N" (0-indexed) or custom name or "Other Features"/"Pruned Events".
    """
    grid = np.zeros((T, V), dtype=np.float32)

    for _, row in cell_data.iterrows():
        event_str = str(row["Event"])
        feat_str = str(row["Feature"])
        shap_val = float(row["Shapley Value"])

        # Parse event index: "Event -N" -> t = T - N
        event_match = re.match(r"Event\s*-(\d+)", event_str)
        if event_match is None:
            continue  # Skip "Pruned Events", "Other Events"

        n = int(event_match.group(1))
        t = T - n
        if t < 0 or t >= T:
            continue

        # Parse feature index: "Feature N" -> v = N
        feat_match = re.match(r"Feature\s*(\d+)", feat_str)
        if feat_match is None:
            continue  # Skip "Other Features", "Pruned Events"

        v = int(feat_match.group(1))
        if v < 0 or v >= V:
            continue

        grid[t, v] = shap_val

    return grid


def compute_timeshap_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    nsamples: int = 2000,
    predict_fn=None,
) -> torch.Tensor:
    """
    TimeSHAP attribution using official timeshap package.

    Per-sample cell-level KernelSHAP: computes event-level, feature-level,
    then cell-level SHAP values via the local_* API.

    The model function maps perturbed observation grids to predictions:
    non-zero cells are treated as observed events, zero cells as removed
    (or replaced, depending on predict_fn).

    Args:
        predict_fn: callable(backbone, batch, mask_tv) -> logits[B].
                    Defaults to masked_predict (removal masking).
                    Pass marginal_masked_predict or conditional_masked_predict
                    for alternative masking strategies.

    Returns:
        attr_tv: [B, T, V]
    """
    if not HAS_TIMESHAP:
        raise ImportError("timeshap package required: pip install timeshap")

    if predict_fn is None:
        predict_fn = masked_predict

    B = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device
    backbone.eval()

    all_attrs = []

    for b in range(B):
        single_batch = {k: v[b:b+1] for k, v in batch.items()}

        # Model callable: (n_seq, T, V) numpy -> (n_seq, 1) numpy
        # Batched: replicate single sample n_seq times, apply different masks
        _pf = predict_fn  # capture for closure
        _BATCH_CHUNK = 64  # process coalitions in chunks to limit memory

        def model_fn(X, _sb=single_batch, _pf=_pf):
            n_seq = X.shape[0]
            preds = np.zeros((n_seq, 1))
            with torch.no_grad():
                for start in range(0, n_seq, _BATCH_CHUNK):
                    end = min(start + _BATCH_CHUNK, n_seq)
                    chunk = end - start
                    # Replicate single sample chunk times
                    batch_rep = {k: v.expand(chunk, *v.shape[1:])
                                 for k, v in _sb.items()}
                    # [chunk, T, V] masks from coalition grids
                    masks = torch.tensor(
                        (X[start:end] != 0).astype(np.float32), device=device)
                    logits = _pf(backbone, batch_rep, masks)  # [chunk]
                    preds[start:end, 0] = logits.sigmoid().cpu().numpy()
            return preds

        # Build observation grid: (1, T, V)
        obs_grid = _build_obs_grid(
            single_batch, T, V, device).cpu().numpy()  # (1, T, V)
        baseline = np.zeros((1, V))  # zero baseline = no events

        try:
            # Step 1: Pruning (find relevant time window)
            pruning_data, prune_idx = local_pruning(
                model_fn, obs_grid,
                {"tol": 0.025},
                baseline,
                entity_uuid=f"s{b}", entity_col="id",
            )

            # Step 2: Event-level SHAP
            event_data = local_event(
                model_fn, obs_grid,
                {"rs": 42, "nsamples": nsamples},
                entity_uuid=f"s{b}", entity_col="id",
                baseline=baseline, pruned_idx=prune_idx,
            )

            # Step 3: Feature-level SHAP
            feat_data = local_feat(
                model_fn, obs_grid,
                {"rs": 42, "nsamples": nsamples},
                entity_uuid=f"s{b}", entity_col="id",
                baseline=baseline, pruned_idx=prune_idx,
            )

            # Step 4: Cell-level SHAP (event x feature)
            # Limit top_x to avoid KernelSHAP regression failure
            # (n_features = top_x_feats * top_x_events must be << nsamples)
            top_x_events = min(10, T)
            top_x_feats = min(20, V)
            cell_data = local_cell_level(
                model_fn, obs_grid,
                {"rs": 42, "nsamples": max(nsamples, top_x_events * top_x_feats + 50),
                 "top_x_feats": top_x_feats, "top_x_events": top_x_events},
                event_data=event_data, feat_data=feat_data,
                entity_uuid=f"s{b}", entity_col="id",
                baseline=baseline, pruned_idx=prune_idx,
            )

            # Convert cell_data DataFrame -> (T, V) grid
            attr_b = _timeshap_df_to_grid(cell_data, T, V)
            all_attrs.append(
                torch.tensor(attr_b, device=device).unsqueeze(0))

        except Exception as e:
            # Fallback for samples where TimeSHAP fails
            print(f"  TimeSHAP failed for sample {b}: {e}")
            all_attrs.append(torch.zeros(1, T, V, device=device))

    # Mask to observed cells — TimeSHAP can assign values to unobserved grid
    # positions via the (event, feature) cross-product
    attr = torch.cat(all_attrs, dim=0).abs()  # [B, T, V]
    obs_grid = _build_obs_grid(batch, T, V, device)
    return attr * obs_grid


# =============================================================================
# 8. FIT — Feature Importance in Time (Tonekaboni et al., NeurIPS 2020)
# =============================================================================

if HAS_FIT:
    class _PretrainedFit(_TintFit):
        """FIT explainer with pre-trained generator.

        tint's Fit constructor always re-initializes and re-trains the generator.
        This subclass bypasses that by calling the grandparent __init__ directly
        and setting the pre-trained generator manually.

        Overrides representation() to only iterate over observed (t,v) cells,
        giving ~10x speedup for sparse irregular time-series data.
        """

        def __init__(self, forward_func, trained_generator, obs_mask_tv=None):
            PerturbationAttribution.__init__(self, forward_func=forward_func)
            self.generator = trained_generator
            self._obs_mask_tv = obs_mask_tv  # [B, T, V] bool, optional

        def representation(
            self, inputs, additional_forward_args=None,
            n_samples=10, distance_metric="kl", multilabel=False,
            temporal_additional_forward_args=None, show_progress=False,
        ):
            """Sparse-optimized FIT: only iterate over observed (t,v) cells."""
            if self._obs_mask_tv is None:
                # Fall back to original dense implementation
                return super().representation(
                    inputs, additional_forward_args, n_samples,
                    distance_metric, multilabel,
                    temporal_additional_forward_args, show_progress)

            from captum._utils.common import _run_forward
            from tint.utils import get_progress_bars, _slice_to_time

            _, t_len, n_features = inputs.shape
            score = torch.zeros(inputs.shape, device=inputs.device)
            activation = (lambda x: torch.sigmoid(x)) if multilabel \
                else (lambda x: F.softmax(x, -1))

            # Collect observed (t, v) pairs per sample
            obs = self._obs_mask_tv  # [B, T, V]

            t_range = range(1, t_len)
            if show_progress:
                t_range = get_progress_bars()(t_range, desc="FIT (sparse)")

            all_features = set(range(n_features))

            for t in t_range:
                # Which features are observed at time t? (any sample)
                obs_at_t = obs[:, t, :].any(dim=0)  # [V]
                observed_features = torch.where(obs_at_t)[0].tolist()
                if not observed_features:
                    continue
                unobserved_features = list(all_features - set(observed_features))

                partial_inputs, kwargs_copy = _slice_to_time(
                    inputs=inputs, time=t + 1,
                    additional_forward_args=additional_forward_args,
                    temporal_additional_forward_args=temporal_additional_forward_args)
                p_y_t = activation(_run_forward(
                    forward_func=self.forward_func, inputs=partial_inputs,
                    additional_forward_args=kwargs_copy.get("additional_forward_args")))

                partial_inputs, kwargs_copy = _slice_to_time(
                    inputs=inputs, time=t,
                    additional_forward_args=additional_forward_args,
                    temporal_additional_forward_args=temporal_additional_forward_args)
                p_tm1 = activation(_run_forward(
                    forward_func=self.forward_func, inputs=partial_inputs,
                    additional_forward_args=kwargs_copy.get("additional_forward_args")))

                for i in observed_features:
                    x_hat, kwargs_copy = _slice_to_time(
                        inputs=inputs, time=t + 1,
                        additional_forward_args=additional_forward_args,
                        temporal_additional_forward_args=temporal_additional_forward_args)
                    div_all = []

                    # Keep feature i + all unobserved; only replace other observed
                    sig_inds = [i] + unobserved_features

                    for _ in range(n_samples):
                        x_hat_t, _ = self.generator.forward_conditional(
                            inputs[:, :t, :], inputs[:, t, :], sig_inds)
                        x_hat[:, t, :] = x_hat_t
                        y_hat_t = activation(_run_forward(
                            forward_func=self.forward_func, inputs=x_hat,
                            additional_forward_args=kwargs_copy.get("additional_forward_args")))

                        if distance_metric == "kl":
                            div = torch.sum(
                                F.kl_div(torch.log(p_tm1), p_y_t, reduction="none"), -1
                            ) - torch.sum(
                                F.kl_div(torch.log(y_hat_t), p_y_t, reduction="none"), -1
                            )
                            div_all.append(div.cpu().detach())
                        else:
                            raise NotImplementedError(
                                f"Only 'kl' supported, got '{distance_metric}'")

                    e_div = torch.stack(div_all).mean(0)
                    score[:, t, i] = 2.0 / (1 + torch.exp(-5 * e_div)) - 1

            return score


def _build_value_grid_fit(batch, T, V, n_bins, device):
    """Build value grid [B, T, V] with normalized bin values for FIT.

    Observed cells get (bin_index + 1) / (n_bins + 1) → range (0.09, 0.91) for n_bins=10.
    Unobserved cells get 0. This encoding lets us distinguish unobserved (0) from bin 0 (0.09).
    """
    vb = batch["value_bins"]
    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]
    B = vb.shape[0]

    grid = torch.zeros(B, T, V, device=device)
    for b in range(B):
        valid = obs_mask[b].bool()
        t_idx = tau[b, valid].long()
        v_idx = varis[b, valid].long()
        grid[b, t_idx, v_idx] = (vb[b, valid].float() + 1) / (n_bins + 1)
    return grid


def _grid_to_bins(grid_values, n_bins):
    """Convert normalized grid values back to bin indices. Inverse of _build_value_grid."""
    return (grid_values * (n_bins + 1) - 1).round().long().clamp(0, n_bins - 1)


class _MaskAwareJointFeatureGenerator(nn.Module):
    """JointFeatureGenerator that concatenates observation mask to RNN input.

    The RNN receives [values, mask] (2V dims) so it can distinguish
    "variable measured at 0" from "variable not measured" in sparse data.
    Output distribution is still V-dimensional.
    """

    def __init__(self, rnn_hidden_size=100, dist_hidden_size=10, latent_size=100):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.dist_hidden_size = dist_hidden_size
        self.latent_size = latent_size
        self.feature_size = None

    def init(self, feature_size):
        self.feature_size = feature_size
        # RNN takes 2V input (values + obs mask)
        self.rnn = nn.GRU(feature_size * 2, self.rnn_hidden_size, batch_first=True)
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if "weight" in p:
                    nn.init.normal_(self.rnn.__getattr__(p), 0.0, 0.02)

        self.dist_predictor = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.dist_hidden_size),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=self.dist_hidden_size),
            nn.Linear(self.dist_hidden_size, self.latent_size * 2),
            nn.Tanh(),
        )
        self.cov_generator = nn.Sequential(
            nn.Linear(self.latent_size, self.dist_hidden_size),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=self.dist_hidden_size),
            nn.Linear(self.dist_hidden_size, feature_size ** 2),
            nn.ReLU(),
        )
        self.mean_generator = nn.Sequential(
            nn.Linear(self.latent_size, self.dist_hidden_size),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=self.dist_hidden_size),
            nn.Linear(self.dist_hidden_size, feature_size),
        )
        # Mask head: predicts P(observed) per feature from RNN hidden state
        self.mask_head = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, self.dist_hidden_size),
            nn.Tanh(),
            nn.Linear(self.dist_hidden_size, feature_size),
        )

    def _encode(self, past):
        """Run RNN on past, return hidden state h."""
        mask = (past > 0).float()
        rnn_input = torch.cat([past, mask], dim=-1)     # [B, T, 2V]
        _, encoding = self.rnn(rnn_input)
        return encoding.view(encoding.size(1), -1)

    def predict_mask(self, past):
        """Predict P(feature observed at next step | past). Returns [B, V]."""
        h = self._encode(past)
        return torch.sigmoid(self.mask_head(h))

    def likelihood_distribution(self, past):
        # Concatenate observation mask so RNN sees [values, mask]
        h = self._encode(past)

        mu_std = self.dist_predictor(h)
        mu = mu_std[:, :mu_std.shape[1] // 2]
        std = mu_std[:, mu_std.shape[1] // 2:]
        z = mu + std * torch.randn_like(mu)

        mean = self.mean_generator(z)
        V = self.feature_size
        cov_noise = (torch.eye(V, device=z.device).unsqueeze(0) * 1e-5).expand(len(z), -1, -1)
        a = self.cov_generator(z).view(-1, V, V)
        covariance = torch.bmm(a, a.transpose(1, 2)) + cov_noise
        return mean, covariance

    def forward(self, past):
        mean, covariance = self.likelihood_distribution(past)
        dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance)
        return dist.rsample()

    def forward_conditional(self, past, current, sig_inds):
        if current.shape[-1] == len(sig_inds):
            return current, current
        if len(current.shape) == 1:
            current = current.unsqueeze(0)

        mean, covariance = self.likelihood_distribution(past)
        V = past.shape[-1]
        sig_inds_comp = list(set(range(V)) - set(sig_inds))
        ind_len = len(sig_inds)
        ind_len_not = len(sig_inds_comp)

        x_ind = current[:, sig_inds].view(-1, ind_len)
        mean_1 = mean[:, sig_inds_comp].view(-1, ind_len_not)
        cov_1_2 = covariance[:, sig_inds_comp, :][:, :, sig_inds].view(-1, ind_len_not, ind_len)
        cov_2_2 = covariance[:, sig_inds, :][:, :, sig_inds].view(-1, ind_len, ind_len)
        cov_1_1 = covariance[:, sig_inds_comp, :][:, :, sig_inds_comp].view(-1, ind_len_not, ind_len_not)

        mean_cond = mean_1 + torch.bmm(
            torch.bmm(cov_1_2, torch.inverse(cov_2_2)),
            (x_ind - mean[:, sig_inds]).view(-1, ind_len, 1),
        ).squeeze(-1)
        covariance_cond = cov_1_1 - torch.bmm(
            torch.bmm(cov_1_2, torch.inverse(cov_2_2)), cov_1_2.transpose(1, 2)
        )

        dist = torch.distributions.MultivariateNormal(loc=mean_cond, covariance_matrix=covariance_cond)
        sample = dist.rsample()
        full_sample = current.clone()
        full_sample[:, sig_inds_comp] = sample
        return full_sample, mean[:, sig_inds_comp]


if HAS_FIT:
    class _MaskAwareFitGeneratorNet(JointFeatureGeneratorNet):
        """FIT generator with mask-aware RNN input and training loss.

        - RNN sees [values, obs_mask] (2V dims) to distinguish missing from zero
        - Loss computed only on observed cells (value > 0)
        """

        def __init__(self, **kwargs):
            # Skip JointFeatureGeneratorNet.__init__ which creates the wrong generator
            from tint.models import Net
            generator = _MaskAwareJointFeatureGenerator(
                rnn_hidden_size=kwargs.get('rnn_hidden_size', 100),
                dist_hidden_size=kwargs.get('dist_hidden_size', 10),
                latent_size=kwargs.get('latent_size', 100),
            )
            Net.__init__(
                self, layers=generator, loss=None,
                optim=kwargs.get('optim', 'adam'),
                lr=kwargs.get('lr', 0.001),
                l2=kwargs.get('l2', 0.0),
            )

        def step(self, batch, batch_idx, stage, t):
            x = batch[0]                                    # [B, T, V]
            obs = (x > 0)                                   # observed mask
            mean, cov = self.net.likelihood_distribution(x[:, :t, ...])
            var = torch.diagonal(cov, dim1=-2, dim2=-1)     # [B, V]
            target = x[:, t, :]                              # [B, V]
            mask = obs[:, t, :].float()                      # [B, V]
            # Per-variable Gaussian NLL, masked to observed cells
            nll = 0.5 * ((target - mean) ** 2 / var.clamp(min=1e-6)
                          + torch.log(var.clamp(min=1e-6)))  # [B, V]
            n_obs = mask.sum().clamp(min=1)
            loss = (nll * mask).sum() / n_obs
            self.log(f"{stage}_loss", loss)
            return loss

        def training_step(self, batch, batch_idx):
            import torch as th
            t = th.randint(low=4, high=batch[0].shape[1], size=(1,)).item()
            return self.step(batch, batch_idx, "train", t)

        def validation_step(self, batch, batch_idx):
            import torch as th
            t = th.randint(low=4, high=batch[0].shape[1], size=(1,)).item()
            self.step(batch, batch_idx, "val", t)


def train_fit_generator(
    dataset,
    train_indices,
    T: int,
    V: int,
    device: str = "cuda",
    batch_size: int = 32,
    max_epochs: int = 300,
) -> Tuple:
    """
    Train FIT's conditional generator on training value grids.

    Uses mask-aware training: loss is computed only on observed (nonzero)
    cells so the generator learns real measurement distributions rather
    than predicting zeros for unobserved cells.

    Returns:
        (trained_generator, n_bins) tuple
    """
    if not HAS_FIT:
        raise ImportError("tint package required: pip install time-interpret")

    from pytorch_lightning import Trainer
    from torch.utils.data import DataLoader, TensorDataset

    n_bins = dataset.n_value_bins

    # Build training value grids [N, T, V]
    print(f"  Building value grids for {len(train_indices)} training samples...", flush=True)
    train_grids = []
    for start in range(0, len(train_indices), batch_size):
        end = min(start + batch_size, len(train_indices))
        batch = dataset.get_batch(train_indices[start:end])
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        train_grids.append(_build_value_grid_fit(batch_dev, T, V, n_bins, device).cpu())
    train_grids = torch.cat(train_grids)  # [N, T, V]
    print(f"  Value grids: shape={train_grids.shape}, "
          f"nonzero={(train_grids > 0).float().mean():.3f}", flush=True)

    # Train mask-aware generator
    gen_net = _MaskAwareFitGeneratorNet()
    gen_net.net.init(feature_size=V)

    # Numerical stability: bump diagonal of covariance from 1e-5 to 1e-3
    _orig_likelihood = gen_net.net.likelihood_distribution
    def _stable_likelihood(past):
        mean, cov = _orig_likelihood(past)
        cov = cov + 1e-3 * torch.eye(V, device=cov.device).unsqueeze(0)
        return mean, cov
    gen_net.net.likelihood_distribution = _stable_likelihood

    # Patch forward_conditional: conditional covariance can lose pos-def
    _orig_forward_conditional = gen_net.net.forward_conditional
    def _stable_forward_conditional(past, current, sig_inds):
        try:
            return _orig_forward_conditional(past, current, sig_inds)
        except ValueError:
            # Fallback: unconditional sample + keep sig_inds from current
            sample = gen_net.net(past)  # [B, V]
            full_sample = current.clone()
            sig_inds_comp = list(set(range(V)) - set(sig_inds))
            full_sample[:, sig_inds_comp] = sample[:, sig_inds_comp]
            return full_sample, sample[:, sig_inds_comp]
    gen_net.net.forward_conditional = _stable_forward_conditional

    trainer = Trainer(max_epochs=max_epochs, enable_progress_bar=True,
                      enable_checkpointing=False, logger=False)
    loader = DataLoader(TensorDataset(train_grids), batch_size=batch_size)
    print(f"  Training FIT generator ({max_epochs} epochs, mask-aware)...", flush=True)
    trainer.fit(gen_net, train_dataloaders=loader)
    gen_net.eval()
    gen_net.to(device)

    return gen_net.net, n_bins


def compute_fit_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    fit_generator,
    n_bins: int = 10,
    n_samples: int = 10,
) -> torch.Tensor:
    """
    FIT attribution (Tonekaboni et al., NeurIPS 2020), adapted for event-sequence backbone.

    For each observed (t, v) cell, measures how much feature v's actual value
    at time t uniquely contributes to the prediction change, using counterfactual
    samples from the generator.

    Score = sigmoid(KL(p_t || p_{t-1}) - KL(p_t || y_hat_counterfactual))
    where the counterfactual keeps feature v's value but replaces all others
    at time t with conditional samples from the generator.

    Optimized: only iterates over observed (t, v) cells, not full T×V grid.

    Args:
        fit_generator: Pre-trained JointFeatureGenerator
        n_bins: Number of value bins in the dataset
        n_samples: Monte Carlo samples per feature

    Returns:
        attr_tv: [B, T, V]
    """
    if not HAS_FIT:
        raise ImportError("tint package required: pip install time-interpret")

    B = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device
    backbone.eval()

    # Build value grids for this batch
    value_grids = _build_value_grid_fit(batch, T, V, n_bins, device)  # [B, T, V]

    # Pre-build per-sample event lookup: for each (b, t, v), which event index l?
    # This lets us efficiently replace value_bins for specific (t,v) cells.
    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]

    # event_at[b][t][v] = event index l (or -1 if unobserved)
    event_at = torch.full((B, T, V), -1, dtype=torch.long, device=device)
    for b in range(B):
        valid = obs_mask[b].bool()
        indices = torch.where(valid)[0]
        t_idx = tau[b, valid].long()
        v_idx = varis[b, valid].long()
        event_at[b, t_idx, v_idx] = indices

    def _predict_with_grid(grid_batchwise):
        """grid: [B, T', V] normalized values → [B, 2] logits.
        Replaces value_bins in the original batch based on grid values,
        masks out events beyond temporal slice.
        """
        B_in, t_len = grid_batchwise.shape[0], grid_batchwise.shape[1]

        new_vb = batch["value_bins"].clone()
        new_om = obs_mask.clone().float()

        for b in range(B_in):
            valid_events = obs_mask[b].bool()
            t_ev = tau[b, valid_events].long()
            v_ev = varis[b, valid_events].long()
            ev_indices = torch.where(valid_events)[0]

            # Mask out events beyond temporal slice
            if t_len < T:
                beyond = t_ev >= t_len
                new_om[b, ev_indices[beyond]] = 0.0

            # Replace value_bins from grid for all in-range events.
            # Generator may produce negative values — clamp before converting.
            in_range = t_ev < t_len
            t_in = t_ev[in_range]
            v_in = v_ev[in_range]
            ev_in = ev_indices[in_range]
            grid_vals = grid_batchwise[b, t_in, v_in].clamp(min=1e-6)
            new_bins = _grid_to_bins(grid_vals, n_bins)
            new_vb[b, ev_in] = new_bins

        with torch.no_grad():
            out = backbone.forward({**batch, "value_bins": new_vb, "obs_mask": new_om})
        logits = out["logits"]
        return torch.stack([torch.zeros_like(logits), logits], dim=-1)  # [B, 2]

    # Create FIT explainer with pre-trained generator
    # Pass obs_mask_tv for sparse optimization (only iterate observed cells)
    obs_mask_tv = (value_grids > 0)  # [B, T, V] bool
    explainer = _PretrainedFit(_predict_with_grid, fit_generator, obs_mask_tv=obs_mask_tv)

    with torch.no_grad():
        attr = explainer.attribute(
            value_grids, n_samples=n_samples, show_progress=True)

    return attr.abs().to(device)  # [B, T, V]


@torch.no_grad()
def compute_fit_removal_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
) -> torch.Tensor:
    """
    FIT-style attribution using event removal as counterfactual.

    Instead of replacing other features' values with generator samples,
    removes other events at time t (sets obs_mask → 0). No generator needed.
    Natural counterfactual for sparse event-sequence data.

    Score(i, t) = sigmoid(KL(p_t || p_{t-1}) - KL(p_t || y_hat))
    where y_hat keeps only feature i's event at time t.

    Returns:
        attr_tv: [B, T, V]
    """
    B = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device
    backbone.eval()

    tau = batch["tau"]        # [B, L]
    varis = batch["varis"]    # [B, L]
    obs_mask = batch["obs_mask"]  # [B, L]

    score = torch.zeros(B, T, V, device=device)

    def _predict_masked(om_override):
        logits = backbone.predict_with_obs_mask(batch, om_override)
        return F.softmax(torch.stack([torch.zeros_like(logits), logits], dim=-1), dim=-1)

    # obs_at_t[b, l] is True if event l is observed AND has tau == t
    # Used to quickly build per-feature masks below.
    obs_bool = obs_mask > 0.5  # [B, L]

    for t in range(1, T):
        # Which features are observed at this timestep? (union across batch)
        at_t = obs_bool & (tau == t)  # [B, L]
        feats_at_t = varis[at_t]      # 1-D flat
        if feats_at_t.numel() == 0:
            continue
        all_obs_features = feats_at_t.unique().tolist()

        # om_t: keep events at time <= t, zero events at time > t
        om_t = obs_mask * (tau <= t).float()  # [B, L]
        p_y_t = _predict_masked(om_t)

        # om_tm1: keep events at time < t (strictly before t)
        om_tm1 = obs_mask * (tau < t).float()  # [B, L]
        p_y_tm1 = _predict_masked(om_tm1)

        kl_base = torch.sum(
            F.kl_div(torch.log(p_y_tm1.clamp(min=1e-8)),
                     p_y_t, reduction="none"), -1)  # [B]

        for i in all_obs_features:
            # om_hat = events before t, plus ONLY feature i's event at t
            feat_i_at_t = at_t & (varis == i)  # [B, L]
            om_hat = om_tm1 + feat_i_at_t.float()  # [B, L], values 0 or 1

            y_hat = _predict_masked(om_hat)

            kl_hat = torch.sum(
                F.kl_div(torch.log(y_hat.clamp(min=1e-8)),
                         p_y_t, reduction="none"), -1)  # [B]
            div = kl_base - kl_hat
            s = 2.0 / (1 + torch.exp(-5 * div)) - 1

            # Assign to samples where feature i is observed at t
            has_feat = feat_i_at_t.any(dim=1)  # [B]
            score[has_feat, t, i] = s[has_feat]

    return score.abs()


def compute_winit_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    fit_generator,
    n_bins: int = 10,
    n_samples: int = 3,
    window_size: int = 10,
    metric: str = "kl",
    eval_batch_size: int = 4,
) -> torch.Tensor:
    """
    WinIT: Windowed Feature Importance in Time (Leung et al., ICLR 2023).

    Extends FIT by accounting for temporal dependencies: measures how much
    feature v's values in a window [t-N+1, t] uniquely contribute to the
    prediction at time t, using finite differences across window positions.

    Batched: for each (t, n), all features × MC samples are batched into a
    single forward pass. Processes eval_batch_size samples at a time to
    control memory.

    Args:
        fit_generator: Pre-trained JointFeatureGenerator (same as FIT)
        n_bins: Number of value bins
        n_samples: Monte Carlo samples per counterfactual
        window_size: Window size N for temporal aggregation
        metric: Distance metric ('kl', 'pd')
        eval_batch_size: Samples per mini-batch (controls GPU memory)

    Returns:
        attr_tv: [B, T, V]
    """
    B_total = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device
    backbone.eval()

    all_scores = []

    for b_start in range(0, B_total, eval_batch_size):
        b_end = min(b_start + eval_batch_size, B_total)
        mb = {k: v[b_start:b_end] for k, v in batch.items()}
        B = mb["obs_mask"].shape[0]

        value_grids = _build_value_grid_fit(mb, T, V, n_bins, device)
        obs_mask_tv = (value_grids > 0)

        mb_tau = mb["tau"]
        mb_varis = mb["varis"]
        mb_obs = mb["obs_mask"]
        L = mb_tau.shape[1]

        def _predict_batch(grid, base_vb, base_tau, base_varis, base_om):
            B_in, t_len = grid.shape[0], grid.shape[1]
            new_vb = base_vb.clone()
            new_om = base_om.clone().float()

            b_idx = torch.arange(B_in, device=device).unsqueeze(1).expand(-1, base_tau.shape[1])
            t_ev = base_tau.long().clamp(0, t_len - 1)
            v_ev = base_varis.long()

            if t_len < T:
                new_om[base_tau >= t_len] = 0.0

            grid_vals = grid[b_idx, t_ev, v_ev]
            replace_mask = (grid_vals > 0) & base_om.bool()
            new_vb[replace_mask] = _grid_to_bins(grid_vals[replace_mask], n_bins)

            with torch.no_grad():
                out = backbone.forward({**mb, "value_bins": new_vb, "obs_mask": new_om})
            logits = out["logits"]
            return F.softmax(torch.stack([-logits, logits], dim=-1), dim=-1)

        score = torch.zeros(B, T, V, device=device)

        for t in range(1, T):
            t_start_w = max(0, t - window_size + 1)
            obs_in_window = obs_mask_tv[:, t_start_w:t+1, :].any(dim=0).any(dim=0)
            observed_features = torch.where(obs_in_window)[0].tolist()
            if not observed_features:
                continue

            n_feat = len(observed_features)
            win = min(t, window_size)

            p_y = _predict_batch(
                value_grids[:, :t+1, :],
                mb["value_bins"], mb_tau, mb_varis, mb_obs)

            iS_array = torch.zeros(n_feat, win, B, device=device)

            for n_idx in range(win):
                time_past = t - n_idx
                time_forward = n_idx + 1

                past = value_grids[:, :time_past, :]
                if past.shape[1] < 4:
                    past_padded = torch.zeros(B, 4, V, device=device)
                    past_padded[:, -past.shape[1]:, :] = past
                    past = past_padded

                # Generate counterfactuals: [n_feat, n_samples, B, t+1, V]
                x_hat_all = value_grids[:, :t+1, :].unsqueeze(0).unsqueeze(0).expand(
                    n_feat, n_samples, -1, -1, -1).clone()

                for s in range(n_samples):
                    for tf in range(time_forward):
                        gen_sample = fit_generator(past)
                        t_abs = time_past + tf
                        for fi, f in enumerate(observed_features):
                            x_hat_all[fi, s, :, t_abs, f] = gen_sample[:, f]

                # Batched prediction with chunking to control memory.
                # Max ~64 backbone forward passes per chunk.
                NF = n_feat * n_samples
                x_hat_flat = x_hat_all.reshape(NF * B, t + 1, V)

                vb_exp = mb["value_bins"].unsqueeze(0).expand(NF, -1, -1).reshape(NF * B, L)
                tau_exp = mb_tau.unsqueeze(0).expand(NF, -1, -1).reshape(NF * B, L)
                var_exp = mb_varis.unsqueeze(0).expand(NF, -1, -1).reshape(NF * B, L)
                om_exp = mb_obs.unsqueeze(0).expand(NF, -1, -1).reshape(NF * B, L)

                fwd_chunk = 64  # max samples per backbone forward
                p_y_hat_parts = []
                for ci in range(0, NF * B, fwd_chunk):
                    ce = min(ci + fwd_chunk, NF * B)
                    p_y_hat_parts.append(_predict_batch(
                        x_hat_flat[ci:ce], vb_exp[ci:ce],
                        tau_exp[ci:ce], var_exp[ci:ce], om_exp[ci:ce]))
                p_y_hat_flat = torch.cat(p_y_hat_parts, dim=0)
                p_y_hat = p_y_hat_flat.reshape(n_feat, n_samples, B, 2)

                p_y_exp = p_y.unsqueeze(0).unsqueeze(0).expand_as(p_y_hat)
                if metric == "kl":
                    divs = torch.sum(F.kl_div(
                        torch.log(p_y_hat.clamp(min=1e-8)),
                        p_y_exp, reduction="none"), -1)
                else:
                    divs = torch.sum(torch.abs(p_y_hat - p_y_exp), -1)

                iS_array[:, n_idx, :] = divs.mean(dim=1).clamp(-1e6, 1e6)

            if win > 1:
                diff = iS_array[:, 1:, :] - iS_array[:, :-1, :]
                iS_array[:, 1:, :] = diff

            for fi, f in enumerate(observed_features):
                score[:, t, f] = iS_array[fi, :, :].abs().sum(dim=0)

        all_scores.append(score)

    return torch.cat(all_scores, dim=0).to(device)


@torch.no_grad()
def compute_winit_removal_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    window_size: int = 10,
) -> torch.Tensor:
    """
    WinIT-style attribution using event removal as counterfactual.

    For each observed feature f at time t, measures the importance by
    removing f's events in windows of increasing depth [t-n, t] and
    decomposing via finite differences.

    iS(t, n) = KL(P || Q_n) where Q_n removes feature f's events in [t-n, t].
    delta_iS(t, n) = iS(t, n) - iS(t, n-1) decomposes per-timestep contribution.
    Final score = sum of |delta_iS| across window depths.

    No generator needed — uses event removal as counterfactual.

    Returns:
        attr_tv: [B, T, V]
    """
    B = batch["obs_mask"].shape[0]
    device = batch["obs_mask"].device
    backbone.eval()

    tau = batch["tau"]        # [B, L]
    varis = batch["varis"]    # [B, L]
    obs_mask = batch["obs_mask"]  # [B, L]

    score = torch.zeros(B, T, V, device=device)
    obs_bool = obs_mask > 0.5  # [B, L]

    def _predict(om_override):
        logits = backbone.predict_with_obs_mask(batch, om_override)
        return F.softmax(torch.stack([torch.zeros_like(logits), logits], dim=-1), dim=-1)

    for t in range(1, T):
        at_t = obs_bool & (tau == t)  # [B, L]
        feats_at_t = varis[at_t]
        if feats_at_t.numel() == 0:
            continue

        observed_features = feats_at_t.unique().tolist()
        n_feat = len(observed_features)
        win = min(t, window_size)

        # om_t: keep events at time <= t
        om_t = obs_mask * (tau <= t).float()  # [B, L]
        p_y = _predict(om_t)

        # iS_array[fi, n, b] = KL(P || Q_n)
        iS_array = torch.zeros(n_feat, win, B, device=device)

        for n_idx in range(win):
            t_start = t - n_idx
            for fi, f in enumerate(observed_features):
                # Remove feature f's events in [t_start, t]
                remove = obs_bool & (varis == f) & (tau >= t_start) & (tau <= t)
                om_cf = om_t * (~remove).float()  # [B, L]

                q_n = _predict(om_cf)

                kl = torch.sum(F.kl_div(
                    torch.log(q_n.clamp(min=1e-8)),
                    p_y, reduction="none"), -1)  # [B]
                iS_array[fi, n_idx, :] = kl

        # Finite differences
        if win > 1:
            diff = iS_array[:, 1:, :] - iS_array[:, :-1, :]
            iS_array[:, 1:, :] = diff

        # Aggregate
        for fi, f in enumerate(observed_features):
            feat_score = iS_array[fi, :, :].abs().sum(dim=0)  # [B]
            has_feat = (at_t & (varis == f)).any(dim=1)  # [B]
            score[has_feat, t, f] = feat_score[has_feat]

    return score


# =============================================================================
# 10. TSR — Temporal Saliency Rescaling (Ismail et al., NeurIPS 2020)
# =============================================================================

class _TSRModelWrapperBase(nn.Module):
    """Common base for the per-sample TSR captum wrappers.

    Stores the entire single-sample batch dict (not just buffers) so that
    CVE backbones can read fields like `values_zscore` and `times_norm`.

    Captum may batch perturbations (B > 1 inside `forward`), so we expand
    each tensor along the first dim. The `obs_grid → event_w` lookup uses
    the same `tau_idx`/`varis_idx` for every expanded copy (same events).
    """

    def __init__(self, adapter, single_batch, T, V):
        super().__init__()
        self.adapter = adapter
        self.T = T
        self.V = V
        # Cache references to the single-sample batch tensors. Not registered
        # as nn.Module buffers because captum doesn't need to track them as
        # parameters; the gradient flows through the soft `obs_grid` input.
        self._batch = {k: v for k, v in single_batch.items() if isinstance(v, torch.Tensor)}
        # Cached event coordinates for the obs_grid lookup (same for every
        # expanded copy)
        self._tau_idx = single_batch["tau"][0]      # [L]
        self._varis_idx = single_batch["varis"][0]  # [L]

    def _expand_batch(self, B: int) -> Dict[str, torch.Tensor]:
        """Tile every tensor in self._batch to leading dim B."""
        return {k: v.expand(B, *v.shape[1:]) for k, v in self._batch.items()}

    def _event_w(self, obs_grid: torch.Tensor) -> torch.Tensor:
        """Lookup per-event soft weights from a [B, T, V] grid via cached
        tau/varis indices. Differentiable in `obs_grid`.
        """
        B = obs_grid.shape[0]
        dev = self._tau_idx.device
        obs_grid = obs_grid.to(dev)
        if obs_grid.shape[1] < self.T:
            full = torch.zeros(B, self.T, self.V, device=dev)
            full[:, :obs_grid.shape[1]] = obs_grid
            obs_grid = full
        return obs_grid[:, self._tau_idx, self._varis_idx]  # [B, L]


class _TSRModelWrapperEmb(_TSRModelWrapperBase):
    """Embedding-level TSR wrapper. Multiplies the post-embedding triplet
    tensor by the soft event weight, then runs `forward_from_event_embeddings`.
    Backbone-uniform.
    """

    def forward(self, obs_grid):
        event_w = self._event_w(obs_grid)  # [B, L]
        B = event_w.shape[0]
        expanded = self._expand_batch(B)
        with torch.no_grad():
            emb = self.adapter.get_event_embeddings(expanded)  # [B, L, D]
        masked_emb = emb * (event_w * expanded["obs_mask"]).unsqueeze(-1)
        logit = self.adapter.forward_from_event_embeddings(masked_emb, expanded)  # [B]
        return torch.stack([-logit, logit], dim=-1)


class _TSRModelWrapperValue(_TSRModelWrapperBase):
    """Value-level TSR wrapper for CVE backbones. Multiplies `values_zscore`
    by the soft event weight (so masked events fade toward population mean),
    then runs the standard backbone forward via `forward_with_value_mask`.
    Only valid on `adapter.has_continuous_values=True` backbones.
    """

    def forward(self, obs_grid):
        event_w = self._event_w(obs_grid)  # [B, L]
        B = event_w.shape[0]
        expanded = self._expand_batch(B)
        # Soft mask combines the event_w from obs_grid with the original obs_mask
        soft_mask = event_w * expanded["obs_mask"]  # [B, L]
        logit = self.adapter.forward_with_value_mask(expanded, soft_mask)  # [B]
        return torch.stack([-logit, logit], dim=-1)


# Backwards-compat alias: defaults to the embedding-level variant.
_TSRModelWrapper = _TSRModelWrapperEmb


def compute_tsr_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    method: str = "IG",
    n_steps: int = 50,
    baseline_mode: str = "zero",
    median_bins: Optional[torch.Tensor] = None,
    variant: str = "emb",  # "emb" | "value"
) -> torch.Tensor:
    """
    TSR attribution (Ismail et al., NeurIPS 2020) — native implementation.

    Temporal Saliency Rescaling: compute base saliency S(t,v) via IG/Grad,
    then rescale by temporal and feature marginals:
        TSR(t,v) = |S(t,v)| × temporal_score(t) × feature_score(v)

    where temporal_score(t) = sum_v |S(t,v)| and feature_score(v) = sum_t |S(t,v)|.

    This amplifies cells that are important both temporally and feature-wise.

    Args:
        method: "IG" (Integrated Gradients), "GRAD" (simple gradients),
                "GS" (GradientShap)
        n_steps: IG interpolation steps
        baseline_mode: "zero" or "median" — passed to compute_captum_attribution
        median_bins: [V] per-variable median value bins (for baseline_mode="median")
        variant: "emb" (embedding-level, default; works on all backbones) or
                 "value" (value-level, CVE backbones only). Forwarded to
                 `compute_captum_attribution` so the base saliency is computed
                 in the requested space; the temporal/feature rescaling is
                 unchanged.

    Returns:
        attr_tv: [B, T, V]
    """
    if not HAS_CAPTUM:
        raise ImportError("captum required for TSR base saliency: pip install captum")

    # Step 1: Compute base saliency via Captum (same as compute_captum_attribution)
    base_method = {"IG": "ig", "GRAD": "grad_x_input", "GS": "gradient_shap"}[method]
    base_attr = compute_captum_attribution(
        backbone, batch, T, V, method=base_method, n_steps=n_steps,
        baseline_mode=baseline_mode, median_bins=median_bins,
        variant=variant,
    )  # [B, T, V]

    # Step 2: Temporal Saliency Rescaling
    abs_attr = base_attr.abs()  # [B, T, V]

    # Temporal importance: how important is each time step (sum over features)
    temporal_score = abs_attr.sum(dim=2)  # [B, T]
    temporal_score = temporal_score / (temporal_score.sum(dim=1, keepdim=True) + 1e-12)

    # Feature importance: how important is each feature (sum over time)
    feature_score = abs_attr.sum(dim=1)  # [B, V]
    feature_score = feature_score / (feature_score.sum(dim=1, keepdim=True) + 1e-12)

    # TSR = base_saliency × temporal_score × feature_score
    tsr = abs_attr * temporal_score.unsqueeze(2) * feature_score.unsqueeze(1)

    return tsr  # [B, T, V]


# =============================================================================
# 11. Concept/ACE — K-means clustering + TCAV with statistical test
# =============================================================================

@torch.no_grad()
def compute_concept_attribution(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    concept_vectors: torch.Tensor,
    concept_importances: torch.Tensor,
    T: int,
    V: int,
) -> torch.Tensor:
    """
    Concept-based attribution using precomputed concept vectors and importances.

    Each event is assigned to its nearest concept, and gets that concept's
    TCAV importance score.

    Args:
        concept_vectors: [N_concepts, D] cluster centers
        concept_importances: [N_concepts] TCAV scores per concept

    Returns:
        attr_tv: [B, T, V]
    """
    tau = batch["tau"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]

    # Pattern D → adapter dict API: extract per-event H via the adapter.
    _, _, H = backbone.extract_hidden(batch, return_event_h=True)  # [B, L, D]

    # Nearest concept per event: [B, L]
    dists = torch.cdist(H, concept_vectors.unsqueeze(0).expand(H.shape[0], -1, -1))
    nearest = dists.argmin(dim=-1)  # [B, L]

    # Attribution per event = importance of nearest concept
    attr_event = concept_importances[nearest]  # [B, L]

    return aggregate_to_grid(attr_event.abs(), tau, varis, obs_mask, T, V)


def build_concept_vectors(
    backbone: BackboneAdapter,
    batches: List[Dict[str, torch.Tensor]],
    n_concepts: int = 50,
    n_random: int = 20,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Phase 1+2: Extract concepts via K-means on H embeddings, then compute
    TCAV importances via directional derivatives with statistical significance test.

    Returns:
        concept_vectors: [n_concepts, D]
        concept_importances: [n_concepts]
    """
    from sklearn.cluster import MiniBatchKMeans
    from scipy import stats

    # Phase 1: Collect embeddings
    all_H = []
    backbone.eval()
    with torch.no_grad():
        for batch in tqdm(batches, desc="Collecting embeddings for concepts"):
            batch = {k: v.to(device) for k, v in batch.items()}
            obs_mask = batch["obs_mask"]

            # Pattern D → dict API
            _, _, H = backbone.extract_hidden(batch, return_event_h=True)  # [B, L, D]
            mask = obs_mask.bool()
            for b in range(H.shape[0]):
                valid_H = H[b, mask[b]]  # [n_valid, D]
                all_H.append(valid_H.cpu())

    H_cat = torch.cat(all_H, dim=0).numpy()  # [N_events, D]
    D = H_cat.shape[1]

    # K-means clustering
    km = MiniBatchKMeans(n_clusters=n_concepts, batch_size=min(10000, len(H_cat)))
    km.fit(H_cat)
    concept_vectors = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)

    # Phase 2: TCAV — directional derivative of logit w.r.t. each concept
    concept_importances = torch.zeros(n_concepts, device=device)
    n_samples = 0

    for batch in tqdm(batches, desc="Computing TCAV scores"):
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        obs_mask = batch_dev["obs_mask"]

        # Dict API: backbone-uniform across TISAE / STraTS / HyMaTE.
        embeddings = backbone.get_event_embeddings(batch_dev).detach().requires_grad_(True)
        logits = _forward_from_embeddings(backbone, embeddings, batch_dev)
        logits.sum().backward()

        grad = embeddings.grad.detach()  # [B, L, D]

        mask = obs_mask.bool()
        for b in range(grad.shape[0]):
            g = grad[b, mask[b]].mean(dim=0)  # [D]
            for c_idx in range(n_concepts):
                direction = F.normalize(concept_vectors[c_idx], dim=0)
                concept_importances[c_idx] += (g * direction).sum()
            n_samples += 1

    concept_importances /= max(n_samples, 1)

    # Phase 3: Random CAV significance test
    random_scores = []
    for _ in range(n_random):
        random_dir = F.normalize(torch.randn(D, device=device), dim=0)
        random_score = torch.tensor(0.0, device=device)
        n_rand = 0
        for batch in batches:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            obs_mask = batch_dev["obs_mask"]

            # Dict API: backbone-uniform.
            embeddings = backbone.get_event_embeddings(
                batch_dev).detach().requires_grad_(True)
            logits = _forward_from_embeddings(backbone, embeddings, batch_dev)
            logits.sum().backward()

            grad = embeddings.grad.detach()
            mask_b = obs_mask.bool()
            for b_idx in range(grad.shape[0]):
                g = grad[b_idx, mask_b[b_idx]].mean(dim=0)
                random_score += (g * random_dir).sum()
                n_rand += 1

        random_scores.append((random_score / max(n_rand, 1)).cpu().item())

    random_arr = np.array(random_scores)
    for c in range(n_concepts):
        _, p_value = stats.ttest_1samp(random_arr, concept_importances[c].cpu().item())
        if p_value >= 0.05:
            concept_importances[c] = 0.0  # not statistically significant

    return concept_vectors, concept_importances


# =============================================================================
# Registry: method name -> compute function
# =============================================================================

# =============================================================================
# PredSelector attribution
# =============================================================================

def compute_pred_selector_attribution(
    backbone: BackboneAdapter,
    sae: nn.Module,
    selector,  # PredSelector module
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    H_grid_mean: Optional[torch.Tensor] = None,
    prior_cache: Optional[torch.Tensor] = None,
    n_select: Optional[int] = None,
) -> torch.Tensor:
    """PredSelector attribution: attr[t,v] = sum_k |A_hat_pred[k,t]| * W[k,v].

    Uses hard top-k selection by PredSelector scores (n_select islands).
    n_select defaults to selector.config.n_global_pred (typically 64).
    """
    from tisae.models.pred_selector import assemble_island_features

    sae_mod = sae
    K_eff, C_eff = sae_mod.K_eff, sae_mod.C_eff
    cfg = sae_mod.cfg
    K, M = cfg.K, cfg.M

    obs_mask = batch["obs_mask"]
    device = obs_mask.device

    with torch.no_grad():
        # Pattern D → adapter dict API: extract H_grid + obs_grid + per-event H
        # in one call. Backbone-uniform across TISAE / STraTS / HyMaTE.
        H_grid, obs_grid, H = backbone.extract_hidden(batch, return_event_h=True)
        # h_pool MUST come from the full backbone (all layers), not from the
        # extracted-layer H. When extract_layer != -1, H is at the intermediate
        # layer, so we use adapter.forward() instead.
        if backbone.extract_layer != -1:
            h_pool = backbone.forward(batch, return_pooled=True)['pooled']
        else:
            h_pool, _ = backbone.pool_events(H, obs_mask)

        if H_grid_mean is not None:
            if H_grid_mean.dim() == 2:
                H_grid_input = H_grid - H_grid_mean.unsqueeze(0).unsqueeze(1)
            else:
                H_grid_input = H_grid - H_grid_mean.unsqueeze(0)
            H_grid_input = H_grid_input * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

        # Encoder
        A0 = sae_mod.encoder(H_grid_input, obs_grid)

        B = obs_mask.size(0)
        T_len = A0.size(2)

        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T_len).permute(0, 2, 3, 1)
        else:
            A0_4d = A0.unsqueeze(-1)

        # Island params
        mu, sigma, gate = sae_mod.island_params(A0_4d)
        g = sae_mod.gaussian_mask.build_gaussians(mu, sigma.detach())

        # Decoder W
        W_dec = sae_mod.decoder.get_W().detach()

        # Prior scores
        prior_scores = None
        if prior_cache is not None and 'orig_idx' in batch:
            prior_scores = prior_cache[batch['orig_idx']].to(device).float()

        # Assemble features — infer obs_profile from selector's input dim
        include_obs = (selector.n_island_features != 6 + V)
        features = assemble_island_features(
            A0_4d, g, mu, sigma, gate, obs_grid, W_dec,
            prior_scores, K_eff, C_eff,
            include_obs_profile=include_obs)

        # Hard top-k selection by PredSelector scores
        s_pred = selector(features, h_pool)  # [B, K_eff, C_eff, M]
        s_flat = s_pred.view(B, -1)
        _n_select = n_select if n_select is not None else selector.config.n_global_pred
        n_sel = min(_n_select, s_flat.size(1))
        _, top_idx = s_flat.topk(n_sel, dim=-1)
        sel_mask = torch.zeros_like(s_flat).scatter_(-1, top_idx, 1.0)
        sel_pred = sel_mask.view(B, K_eff, C_eff, M)

        # Build mask and decode
        mask_pred = sae_mod.gaussian_mask.forward_mask(g, sel_pred)
        A_hat_4d = A0_4d * mask_pred
        if C_eff > 1:
            A_hat = A_hat_4d.permute(0, 3, 1, 2).reshape(B, K, T_len)
        else:
            A_hat = A_hat_4d.squeeze(-1)

        # attr[b,t,v] = sum_k |A_hat[b,k,t]| * W[k,v]
        attr = torch.einsum("bkt,kv->btv", A_hat.abs(), W_dec)

    return attr * obs_grid


def compute_grad_act_attribution(
    backbone: BackboneAdapter,
    sae: nn.Module,
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    H_grid_mean: Optional[torch.Tensor] = None,
    n_select: int = 64,
) -> torch.Tensor:
    """Grad×Act attribution: score islands by |∂z/∂energy| × |energy|, select top-k.

    A non-learned heuristic that identifies prediction-relevant islands via
    one backward pass, without requiring a trained PredSelector.
    """
    sae_mod = sae
    K_eff, C_eff = sae_mod.K_eff, sae_mod.C_eff
    cfg = sae_mod.cfg
    K, M = cfg.K, cfg.M

    obs_mask = batch["obs_mask"]
    device = obs_mask.device

    with torch.no_grad():
        H_grid, obs_grid = backbone.extract_hidden(batch)
        if H_grid_mean is not None:
            H_grid = H_grid - H_grid_mean.unsqueeze(0)
            H_grid = H_grid * obs_grid.unsqueeze(-1)

        A0 = sae_mod.encoder(H_grid, obs_grid)
        B, _, T_len = A0.shape

        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T_len).permute(0, 2, 3, 1)
        else:
            A0_4d = A0.unsqueeze(-1)

        mu, sigma, gate = sae_mod.island_params(A0_4d)
        g = sae_mod.gaussian_mask.build_gaussians(mu, sigma.detach())
        W_dec = sae_mod.decoder.get_W().detach()

    A0_perm = A0_4d.detach().permute(0, 1, 3, 2)  # [B, K_eff, C_eff, T]
    energy = (A0_perm.unsqueeze(3) * g.detach()).sum(-1)  # [B, K_eff, C_eff, M]
    energy_param = energy.clone().requires_grad_(True)

    eff_len = g.detach().sum(-1).clamp_min(1e-6)  # [B, K_eff, C_eff, M]
    A_hat_from_energy = energy_param.unsqueeze(-1) * g.detach() / eff_len.unsqueeze(-1)
    # [B, K_eff, C_eff, M, T]

    A_hat_combined = A_hat_from_energy.sum(dim=3)  # sum over M: [B, K_eff, C_eff, T]
    A_hat_combined = A_hat_combined.permute(0, 1, 3, 2)  # [B, K_eff, T, C_eff]
    if C_eff > 1:
        A_hat = A_hat_combined.permute(0, 3, 1, 2).reshape(B, K, T_len)
    else:
        A_hat = A_hat_combined.squeeze(-1)

    H_hat_grid = sae_mod.decoder(A_hat)
    z = backbone.predict_from_grid(H_hat_grid, batch, obs_grid)
    z.sum().backward()

    grad_energy = energy_param.grad  # [B, K_eff, C_eff, M]
    score = (grad_energy.abs() * energy.abs()).detach()

    with torch.no_grad():
        s_flat = score.view(B, -1)
        n_sel = min(n_select, s_flat.size(1))
        _, top_idx = s_flat.topk(n_sel, dim=-1)
        sel_mask = torch.zeros_like(s_flat).scatter_(-1, top_idx, 1.0)
        sel = sel_mask.view(B, K_eff, C_eff, M)

        mask = sae_mod.gaussian_mask.forward_mask(g, sel)
        A_hat_sel = A0_4d * mask
        if C_eff > 1:
            A_hat_sel = A_hat_sel.permute(0, 3, 1, 2).reshape(B, K, T_len)
        else:
            A_hat_sel = A_hat_sel.squeeze(-1)

        attr = torch.einsum("bkt,kv->btv", A_hat_sel.abs(), W_dec)

    return attr * obs_grid


def extract_pred_selector_softs(
    backbone: BackboneAdapter,
    sae: nn.Module,
    selector,  # PredSelector module
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
    H_grid_mean: Optional[torch.Tensor] = None,
    prior_cache: Optional[torch.Tensor] = None,
    g_threshold: float = 0.607,
) -> Tuple[torch.Tensor, List[List[Dict]]]:
    """Extract PredSelector-scored islands for island-level faithfulness eval.

    Returns:
        attr_tv: [B, T, V] soft attribution (same as compute_pred_selector_attribution)
        islands: List[List[Dict]] — per-sample list of island dicts, sorted by
                 PredSelector score (descending). Each dict has:
                   'k', 'c', 'm': island identity
                   'importance': PredSelector score
                   'cell_mask': [T, V] binary mask of cells this island covers
                   'n_cells': number of cells covered
                   'mu', 'sigma': island center and width
    """
    from tisae.models.pred_selector import assemble_island_features

    sae_mod = sae
    K_eff, C_eff = sae_mod.K_eff, sae_mod.C_eff
    cfg = sae_mod.cfg
    K, M = cfg.K, cfg.M

    obs_mask = batch["obs_mask"]
    device = obs_mask.device

    with torch.no_grad():
        # Pattern D → adapter dict API
        H_grid, obs_grid, H = backbone.extract_hidden(batch, return_event_h=True)
        # h_pool MUST come from the full backbone (all layers).
        if backbone.extract_layer != -1:
            h_pool = backbone.forward(batch, return_pooled=True)['pooled']
        else:
            h_pool, _ = backbone.pool_events(H, obs_mask)

        if H_grid_mean is not None:
            if H_grid_mean.dim() == 2:
                H_grid_input = H_grid - H_grid_mean.unsqueeze(0).unsqueeze(1)
            else:
                H_grid_input = H_grid - H_grid_mean.unsqueeze(0)
            H_grid_input = H_grid_input * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

        A0 = sae_mod.encoder(H_grid_input, obs_grid)

        B = obs_mask.size(0)
        T_len = A0.size(2)

        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T_len).permute(0, 2, 3, 1)
        else:
            A0_4d = A0.unsqueeze(-1)

        mu, sigma, gate = sae_mod.island_params(A0_4d)
        g = sae_mod.gaussian_mask.build_gaussians(mu, sigma.detach())
        W_dec = sae_mod.decoder.get_W().detach()
        W_mask = (W_dec > 0).float()

        prior_scores = None
        if prior_cache is not None and 'orig_idx' in batch:
            prior_scores = prior_cache[batch['orig_idx']].to(device).float()

        include_obs = (selector.n_island_features != 6 + V)
        features = assemble_island_features(
            A0_4d, g, mu, sigma, gate, obs_grid, W_dec,
            prior_scores, K_eff, C_eff,
            include_obs_profile=include_obs)

        # Get raw scores
        s_pred = selector(features, h_pool)  # [B, K_eff, C_eff, M]

        # Soft attribution (same as compute_pred_selector_attribution)
        n_global_pred = selector.config.n_global_pred
        s_flat = s_pred.view(B, -1)
        weights = F.softmax(s_flat, dim=-1) * n_global_pred
        sel_soft = weights.view(B, K_eff, C_eff, M)
        mask_bkct = (sel_soft.unsqueeze(-1) * g).sum(dim=-2).clamp(0, 1)
        mask_pred = mask_bkct.permute(0, 1, 3, 2)
        A_hat_4d = A0_4d * mask_pred
        if C_eff > 1:
            A_hat = A_hat_4d.permute(0, 3, 1, 2).reshape(B, K, T_len)
        else:
            A_hat = A_hat_4d.squeeze(-1)
        attr = torch.einsum("bkt,kv->btv", A_hat.abs(), W_dec) * obs_grid

        # Build per-sample island lists sorted by score.
        # Vectorized: threshold g to get temporal footprints, batch-build
        # all cell_masks per sample in one tensor op.
        g_binary = (g > g_threshold).float()  # [B, K_eff, C_eff, M, T]

        # n_cells per island via einsum (avoids materializing [K,C,M,T,V]):
        # var_count[k_flat, b, t] = sum_v W_mask[k_flat,v] * obs_grid[b,t,v]
        K_total = W_mask.shape[0]
        var_count = torch.einsum("kv,btv->kbt", W_mask, obs_grid)  # [K_total, B, T]

        all_islands = []
        for b in range(B):
            obs_grid_b = obs_grid[b]  # [T, V]

            # Flatten (K, C, M) → N and compute n_cells
            g_flat = g_binary[b].reshape(-1, T)            # [K*C*M, T]
            N_total = g_flat.shape[0]
            k_idx = torch.arange(N_total, device=device) // (C_eff * M)
            c_idx = (torch.arange(N_total, device=device) % (C_eff * M)) // M
            m_idx = torch.arange(N_total, device=device) % M
            kf_idx = c_idx * K_eff + k_idx if C_eff > 1 else k_idx

            # n_cells[i] = sum_t g_flat[i,t] * var_count[kf,b,t]
            vc_b = var_count[:, b, :]                       # [K_total, T]
            n_cells = (g_flat * vc_b[kf_idx]).sum(dim=1)    # [N_total]

            # Filter valid + sort by score
            scores_flat = s_pred[b].reshape(-1)
            valid = n_cells > 0
            valid_idx = torch.where(valid)[0]
            if valid_idx.numel() == 0:
                all_islands.append([])
                continue
            sorted_order = scores_flat[valid_idx].argsort(descending=True)
            sel = valid_idx[sorted_order]                    # [N_valid]

            # Batch-build all cell_masks: [N_valid, T, V]
            t_masks = g_flat[sel]                            # [N_valid, T]
            w_rows = W_mask[kf_idx[sel]]                     # [N_valid, V]
            cell_masks = t_masks.unsqueeze(2) * w_rows.unsqueeze(1) * obs_grid_b

            # Pack into dicts (same interface as before)
            k_np = k_idx[sel].cpu().numpy()
            c_np = c_idx[sel].cpu().numpy()
            m_np = m_idx[sel].cpu().numpy()
            sc_np = scores_flat[sel].cpu().numpy()
            nc_np = n_cells[sel].cpu().numpy()
            mu_flat = mu[b].reshape(-1)
            sig_flat = sigma[b].reshape(-1)
            mu_np_b = mu_flat[sel].cpu().numpy()
            sig_np_b = sig_flat[sel].cpu().numpy()

            sample_islands = [
                {'k': int(k_np[i]), 'c': int(c_np[i]), 'm': int(m_np[i]),
                 'importance': float(sc_np[i]),
                 'cell_mask': cell_masks[i],
                 'n_cells': int(nc_np[i]),
                 'mu': float(mu_np_b[i]), 'sigma': float(sig_np_b[i])}
                for i in range(len(sel))
            ]
            all_islands.append(sample_islands)

    return attr, all_islands


METHODS = {
    "random": "compute_random_attribution",
    "uniform": "compute_uniform_attribution",
    "attention": "compute_attention_attribution",
    "ig": "compute_captum_attribution",
    "gradient_shap": "compute_captum_attribution",
    "grad_x_input": "compute_captum_attribution",
    "deeplift": "compute_captum_attribution",
    "sae": "compute_sae_attribution",
    "grad_act": "compute_grad_act_attribution",
    "pred_selector": "compute_pred_selector_attribution",
    "dynamask": "compute_dynamask_attribution",
    "timeshap": "compute_timeshap_attribution",
    "fit": "compute_fit_attribution",
    "fit_removal": "compute_fit_removal_attribution",
    "winit_removal": "compute_winit_removal_attribution",
    "winit": "compute_winit_attribution",
    "tsr": "compute_tsr_attribution",
    "concept": "compute_concept_attribution",
}
