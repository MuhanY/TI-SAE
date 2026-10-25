"""
Faithfulness evaluation metrics for attribution methods.

Metrics:
  - Comprehensiveness: prediction drop when removing top-f important features
  - Sufficiency: prediction preservation when keeping only top-f features
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from tisae.evaluation.faithfulness.attribution_methods import masked_predict, soft_masked_predict
from tisae.models.backbone_adapter import BackboneAdapter


# =============================================================================
# Masking helpers
# =============================================================================

def _build_obs_grid(batch: Dict[str, torch.Tensor], T: int, V: int) -> torch.Tensor:
    """Build [B, T, V] binary grid of observed cells from event sequences."""
    tau = batch["tau"]       # [B, L]
    varis = batch["varis"]   # [B, L]
    om = batch["obs_mask"]   # [B, L]
    B = tau.shape[0]
    device = tau.device
    grid = torch.zeros(B, T, V, device=device)
    for b in range(B):
        valid = om[b].bool()
        t_idx = tau[b, valid].long()
        v_idx = varis[b, valid].long()
        grid[b, t_idx, v_idx] = 1.0
    return grid


def _threshold_mask(
    attr: torch.Tensor,
    fraction: float,
    mode: str,
    obs_grid: torch.Tensor = None,
) -> torch.Tensor:
    """
    Create binary mask from attribution.

    Args:
        attr: [B, T, V] attribution (higher = more important)
        fraction: fraction of **observed** cells to select (not full grid)
        mode: "remove_top" (zero out top-f) or "keep_top" (keep only top-f)
        obs_grid: [B, T, V] binary grid of observed cells. If provided,
                  fraction is relative to observed cells per sample.
                  If None, fraction is relative to full T*V grid.

    Returns:
        mask: [B, T, V] binary. 1 = keep, 0 = remove.
    """
    B, T, V = attr.shape
    n_cells = T * V

    masks = []
    for b in range(B):
        flat = attr[b].reshape(-1)  # [T*V]

        if obs_grid is not None:
            n_observed = int(obs_grid[b].sum().item())
            k = max(1, int(fraction * n_observed))
            # Mask unobserved cells to -inf so topk only selects observed cells.
            # Without this, methods that produce full-grid attributions (DynaMask,
            # TimeSHAP, etc.) waste budget on unobserved cells that have zero
            # effect when removed.
            obs_flat = obs_grid[b].reshape(-1)
            flat = flat.clone()
            flat[obs_flat < 0.5] = float("-inf")
        else:
            k = max(1, int(fraction * n_cells))

        # Clamp k to valid range (can't exceed observed cells if obs_grid given)
        if obs_grid is not None:
            k = min(k, max(1, n_observed))
        else:
            k = min(k, n_cells)

        _, top_idx = flat.topk(k)

        if mode == "remove_top":
            mask = torch.ones(n_cells, device=attr.device)
            mask[top_idx] = 0.0
        else:  # keep_top
            mask = torch.zeros(n_cells, device=attr.device)
            mask[top_idx] = 1.0

        masks.append(mask.reshape(T, V))

    return torch.stack(masks, dim=0)  # [B, T, V]


# =============================================================================
# Comprehensiveness
# =============================================================================

@torch.no_grad()
def comprehensiveness(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    attr: torch.Tensor,
    fractions: List[float] = (0.1, 0.2, 0.3, 0.5),
    predict_fn=None,
) -> Dict[str, float]:
    """
    Comprehensiveness: how much prediction drops when removing top-f features.

    Higher = better attribution (removing important features hurts more).

    Uses sign-aligned probability change: for predicted-positive samples,
    comp = p_orig - p_masked (removing important features should decrease P(y=1)).
    For predicted-negative, comp = p_masked - p_orig.

    Args:
        predict_fn: callable(backbone, batch, mask_tv) -> logits[B].
                    Defaults to masked_predict (hard event removal).
                    Use soft_masked_predict for embedding-level masking.

    Returns:
        dict mapping f"comp@{f}" -> mean comprehensiveness for each fraction,
        plus "p_orig_mean" -> mean |p_orig| (for NAOPC normalization).
    """
    if predict_fn is None:
        predict_fn = masked_predict

    B, T, V = attr.shape
    device = attr.device

    # Build obs grid for fraction normalization (fraction of observed cells)
    obs_grid = _build_obs_grid(batch, T, V)

    # Original prediction (always use hard masked_predict for baseline — no masking)
    full_mask = torch.ones(T, V, device=device)
    logit_orig = masked_predict(backbone, batch, full_mask)  # [B]
    p_orig = logit_orig.sigmoid()
    predicted_pos = (logit_orig > 0).float()

    sign = 2 * predicted_pos - 1  # [B], +1 for pos, -1 for neg

    results = {"p_orig_mean": float(p_orig.abs().mean().item())}
    for f in fractions:
        masks = _threshold_mask(attr, f, mode="remove_top", obs_grid=obs_grid)  # [B, T, V]
        # Batched forward: pass [B, T, V] masks, get [B] logits
        logit_m = predict_fn(backbone, batch, masks)  # [B]
        p_m = logit_m.sigmoid()
        comp_vals = (sign * (p_orig - p_m)).cpu().numpy()
        results[f"comp@{f}"] = float(np.mean(comp_vals))
        results[f"comp@{f}_samples"] = comp_vals.tolist()

    return results


# =============================================================================
# Sufficiency
# =============================================================================

@torch.no_grad()
def sufficiency(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    attr: torch.Tensor,
    fractions: List[float] = (0.1, 0.2, 0.3, 0.5),
    predict_fn=None,
) -> Dict[str, float]:
    """
    Sufficiency: how well prediction is preserved when keeping ONLY top-f features.

    Lower = better attribution (sufficient features preserve prediction).

    Args:
        predict_fn: callable(backbone, batch, mask_tv) -> logits[B].
                    Defaults to masked_predict (hard event removal).

    Returns:
        dict mapping f"suff@{f}" -> mean sufficiency for each fraction
    """
    if predict_fn is None:
        predict_fn = masked_predict

    B, T, V = attr.shape
    device = attr.device

    obs_grid = _build_obs_grid(batch, T, V)

    full_mask = torch.ones(T, V, device=device)
    logit_orig = masked_predict(backbone, batch, full_mask)
    p_orig = logit_orig.sigmoid()
    predicted_pos = (logit_orig > 0).float()

    sign = 2 * predicted_pos - 1  # [B]

    results = {}
    for f in fractions:
        masks = _threshold_mask(attr, f, mode="keep_top", obs_grid=obs_grid)  # [B, T, V]
        logit_k = predict_fn(backbone, batch, masks)  # [B]
        p_k = logit_k.sigmoid()
        suff_vals = (sign * (p_orig - p_k)).cpu().numpy()
        results[f"suff@{f}"] = float(np.mean(suff_vals))
        results[f"suff@{f}_samples"] = suff_vals.tolist()

    return results


# =============================================================================
# Island-wise masking
# =============================================================================

def _island_mask(
    islands: List[Dict],
    fraction: float,
    T: int,
    V: int,
    mode: str,
    device: torch.device,
    obs_grid: torch.Tensor = None,
) -> Tuple[torch.Tensor, float]:
    """
    Create binary mask by greedily adding/removing whole islands.

    Args:
        islands: list of island dicts (sorted by importance descending),
                 each with 'cell_mask' [T,V] and 'importance'
        fraction: target fraction of cells to select. If obs_grid is provided,
                  fraction is relative to observed cells (consistent with
                  cell-level evaluation). Otherwise relative to T*V.
        T, V: grid dimensions
        mode: "remove_top" (remove most important islands first)
              "keep_top" (keep only most important islands)
        device: torch device
        obs_grid: [T, V] binary grid of observed cells for this sample.
                  If provided, only count observed cells within islands.

    Returns:
        mask: [T, V] binary (1 = keep, 0 = remove)
        actual_frac: actual fraction of observed cells affected
    """
    if obs_grid is not None:
        n_cells = int(obs_grid.sum().item())
    else:
        n_cells = T * V
    n_cells = max(n_cells, 1)
    target = max(1, int(fraction * n_cells))

    if mode == "remove_top":
        # Start with all cells, greedily remove most important islands
        mask = torch.ones(T, V, device=device)
        removed = torch.zeros(T, V, device=device)
        n_removed = 0

        for island in islands:  # already sorted descending
            cell_mask = island["cell_mask"]
            if cell_mask.device != device:
                cell_mask = cell_mask.to(device)
            # Only count observed cells within this island
            if obs_grid is not None:
                effective_mask = cell_mask * obs_grid
            else:
                effective_mask = cell_mask
            new_cells = effective_mask * (1.0 - removed)
            n_new = int(new_cells.sum().item())

            if n_new == 0:
                continue

            if n_removed + n_new > target * 1.5:
                # Would overshoot by >50% — skip this island
                # (try smaller islands next)
                continue

            # Remove ALL cells in the island footprint (including unobserved,
            # which has no effect but keeps the mask clean)
            removed = (removed + cell_mask).clamp(max=1.0)
            n_removed += n_new

            if n_removed >= target:
                break

        mask = 1.0 - removed
        actual_frac = n_removed / n_cells

    else:  # keep_top
        # Start with nothing, greedily add most important islands
        mask = torch.zeros(T, V, device=device)
        n_kept = 0

        for island in islands:
            cell_mask = island["cell_mask"]
            if cell_mask.device != device:
                cell_mask = cell_mask.to(device)
            if obs_grid is not None:
                effective_mask = cell_mask * obs_grid
            else:
                effective_mask = cell_mask
            new_cells = effective_mask * (1.0 - mask)
            n_new = int(new_cells.sum().item())

            if n_new == 0:
                continue

            if n_kept + n_new > target * 1.5:
                continue

            mask = (mask + cell_mask).clamp(max=1.0)
            n_kept += n_new

            if n_kept >= target:
                break

        actual_frac = n_kept / n_cells

    return mask, actual_frac


@torch.no_grad()
def comprehensiveness_island(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    islands_per_sample: List[List[Dict]],
    T: int,
    V: int,
    fractions: List[float] = (0.1, 0.2, 0.3, 0.5),
    predict_fn=None,
) -> Dict[str, float]:
    """
    Island-wise comprehensiveness: remove whole islands instead of individual cells.

    Fractions are relative to observed cells (consistent with cell-level eval).
    Returns dict with icomp@{f} and icomp_frac@{f} (actual fraction of observed cells removed).
    """
    if predict_fn is None:
        predict_fn = masked_predict
    B = len(islands_per_sample)
    device = batch["obs_mask"].device

    # Build per-sample obs grids for fraction counting
    obs_grids = _build_obs_grid(batch, T, V)  # [B, T, V]

    full_mask = torch.ones(T, V, device=device)
    logit_orig = masked_predict(backbone, batch, full_mask)
    p_orig = logit_orig.sigmoid()
    predicted_pos = (logit_orig > 0).float()

    sign = 2 * predicted_pos - 1  # [B]

    results = {}
    for f in fractions:
        masks_list = []
        frac_vals = []
        for b in range(B):
            mask, actual_f = _island_mask(
                islands_per_sample[b], f, T, V, "remove_top", device,
                obs_grid=obs_grids[b],
            )
            masks_list.append(mask)
            frac_vals.append(actual_f)
        masks = torch.stack(masks_list)  # [B, T, V]
        logit_m = predict_fn(backbone, batch, masks)  # [B]
        p_m = logit_m.sigmoid()
        comp_vals = (sign * (p_orig - p_m)).cpu().numpy()

        results[f"icomp@{f}"] = float(np.mean(comp_vals))
        results[f"icomp@{f}_samples"] = comp_vals.tolist()
        results[f"icomp_frac@{f}"] = float(np.mean(frac_vals))

    return results


@torch.no_grad()
def sufficiency_island(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    islands_per_sample: List[List[Dict]],
    T: int,
    V: int,
    fractions: List[float] = (0.1, 0.2, 0.3, 0.5),
    predict_fn=None,
) -> Dict[str, float]:
    """
    Island-wise sufficiency: keep only whole islands.

    Fractions are relative to observed cells (consistent with cell-level eval).
    Returns dict with isuff@{f} and isuff_frac@{f} (actual fraction of observed cells kept).
    """
    if predict_fn is None:
        predict_fn = masked_predict
    B = len(islands_per_sample)
    device = batch["obs_mask"].device

    obs_grids = _build_obs_grid(batch, T, V)  # [B, T, V]

    full_mask = torch.ones(T, V, device=device)
    logit_orig = masked_predict(backbone, batch, full_mask)
    p_orig = logit_orig.sigmoid()
    predicted_pos = (logit_orig > 0).float()

    sign = 2 * predicted_pos - 1  # [B]

    results = {}
    for f in fractions:
        masks_list = []
        frac_vals = []
        for b in range(B):
            mask, actual_f = _island_mask(
                islands_per_sample[b], f, T, V, "keep_top", device,
                obs_grid=obs_grids[b],
            )
            masks_list.append(mask)
            frac_vals.append(actual_f)
        masks = torch.stack(masks_list)  # [B, T, V]
        logit_k = predict_fn(backbone, batch, masks)  # [B]
        p_k = logit_k.sigmoid()
        suff_vals = (sign * (p_orig - p_k)).cpu().numpy()

        results[f"isuff@{f}"] = float(np.mean(suff_vals))
        results[f"isuff@{f}_samples"] = suff_vals.tolist()
        results[f"isuff_frac@{f}"] = float(np.mean(frac_vals))

    return results


@torch.no_grad()
def evaluate_faithfulness_island(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    islands_per_sample: List[List[Dict]],
    T: int,
    V: int,
    fractions: List[float] = (0.1, 0.2, 0.3, 0.5),
    predict_fn=None,
) -> Dict[str, float]:
    """
    Compute island-wise faithfulness metrics for SAE.

    Returns dict with icomp@f, icomp_frac@f, isuff@f, isuff_frac@f.
    """
    results = {}
    results.update(comprehensiveness_island(backbone, batch, islands_per_sample, T, V, fractions, predict_fn=predict_fn))
    results.update(sufficiency_island(backbone, batch, islands_per_sample, T, V, fractions, predict_fn=predict_fn))
    return results


# =============================================================================
# Combined evaluation
# =============================================================================

@torch.no_grad()
def evaluate_faithfulness(
    backbone: BackboneAdapter,
    batch: Dict[str, torch.Tensor],
    attr: torch.Tensor,
    fractions: List[float] = (0.1, 0.2, 0.3, 0.5),
    predict_fn=None,
) -> Dict[str, float]:
    """
    Compute all fast proxy faithfulness metrics.

    Args:
        backbone: BinnedTimeTransformer model
        batch: input batch dict
        attr: [B, T, V] attribution from any method
        predict_fn: callable(backbone, batch, mask_tv) -> logits[B].
                    Defaults to masked_predict (hard event removal).
                    Use soft_masked_predict for methods trained with soft masking.

    Returns:
        dict with comp@f, suff@f
    """
    results = {}
    results.update(comprehensiveness(backbone, batch, attr, fractions, predict_fn=predict_fn))
    results.update(sufficiency(backbone, batch, attr, fractions, predict_fn=predict_fn))
    return results
