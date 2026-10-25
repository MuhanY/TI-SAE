# tisae/utils/grid_ops.py
"""Grid conversion utilities for scatter/gather between event and grid representations."""

import torch
from typing import Tuple


def build_H_grid(
    H: torch.Tensor,
    tau: torch.Tensor,
    varis: torch.Tensor,
    obs_mask: torch.Tensor,
    T: int,
    V: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter H[B,L,D] into H_grid[B,T,V,D] by (tau, varis).
    Average duplicate (t,v) events.

    Args:
        H: [B, L, D] backbone embeddings
        tau: [B, L] time bin indices
        varis: [B, L] variable indices
        obs_mask: [B, L] observation mask (1 = real, 0 = padding)
        T: number of time bins
        V: number of variables

    Returns:
        H_grid: [B, T, V, D]
        obs_grid: [B, T, V] binary mask
    """
    B, L, D = H.shape
    device = H.device

    batch_idx = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
    flat_idx = (batch_idx * T * V + tau * V + varis).view(-1)  # [B*L]

    H_flat = H.reshape(-1, D)          # [B*L, D]
    mask_flat = obs_mask.reshape(-1)    # [B*L]

    H_grid_flat = torch.zeros(B * T * V, D, device=device, dtype=H.dtype)
    counts = torch.zeros(B * T * V, device=device)

    H_grid_flat.scatter_add_(
        0, flat_idx.unsqueeze(1).expand(-1, D), H_flat * mask_flat.unsqueeze(1)
    )
    counts.scatter_add_(0, flat_idx, mask_flat)

    H_grid = H_grid_flat.view(B, T, V, D) / counts.view(B, T, V).unsqueeze(-1).clamp_min(1)
    obs_grid = (counts.view(B, T, V) > 0).float()

    return H_grid, obs_grid


def grid_to_H(
    H_hat_grid: torch.Tensor,
    tau: torch.Tensor,
    varis: torch.Tensor,
) -> torch.Tensor:
    """
    Index lookup: H_hat_grid[B,T,V,D] → H_hat[B,L,D]

    Args:
        H_hat_grid: [B, T, V, D]
        tau: [B, L] time bin indices
        varis: [B, L] variable indices

    Returns:
        H_hat: [B, L, D]
    """
    B = H_hat_grid.size(0)
    batch_idx = torch.arange(B, device=H_hat_grid.device).unsqueeze(1)
    return H_hat_grid[batch_idx, tau, varis]  # [B, L, D]
