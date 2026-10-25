"""Shared utilities for TISAE evaluation layers.

Functions:
    build_obs_grid     — irregular events → binary (B, T, V) observation grid
    build_value_grid   — irregular events → (B, T, V) clinical value grid
    run_tisae_forward  — backbone → H_grid → residualize → TISAE forward
    load_evaluation_models — load frozen backbone + TISAE + metadata
    compute_population_stats — two-pass mean/std from training split
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tisae.models.ontology import CHANNEL_DEFINITIONS
from tisae.training.train_pred_selector import load_tisae_models
from tisae.utils.grid_ops import build_H_grid


# =============================================================================
# Grid builders
# =============================================================================

def build_obs_grid(
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
) -> torch.Tensor:
    """Convert irregular event sequences to (B, T, V) binary observation grid."""
    tau = batch["tau"]        # [B, L]
    varis = batch["varis"]    # [B, L]
    om = batch["obs_mask"]    # [B, L]
    B, L = tau.shape
    device = tau.device
    grid = torch.zeros(B, T, V, device=device)
    for b in range(B):
        valid = om[b].bool()
        t_idx = tau[b, valid].long()
        v_idx = varis[b, valid].long()
        grid[b, t_idx, v_idx] = 1.0
    return grid


def build_value_grid(
    batch: Dict[str, torch.Tensor],
    T: int,
    V: int,
) -> torch.Tensor:
    """Convert irregular events to (B, T, V) grid with clinical values.

    Uses values_raw if available, otherwise falls back to value_bins (float).
    """
    tau = batch["tau"]
    varis = batch["varis"]
    om = batch["obs_mask"]
    vals = batch.get("values_raw", batch["value_bins"])
    B, L = tau.shape
    device = tau.device
    grid = torch.zeros(B, T, V, device=device)
    for b in range(B):
        valid = om[b].bool()
        t_idx = tau[b, valid].long()
        v_idx = varis[b, valid].long()
        grid[b, t_idx, v_idx] = vals[b, valid].float()
    return grid


# =============================================================================
# TISAE forward pass
# =============================================================================

@torch.no_grad()
def run_tisae_forward(
    adapter,
    sae,
    batch: Dict[str, torch.Tensor],
    H_grid_mean: Optional[torch.Tensor],
    T: int,
    V: int,
) -> Dict[str, torch.Tensor]:
    """Adapter → build_H_grid → residualize → TISAE forward.

    Returns the TISAE output dict (A_hat, mu, sigma, sel, etc.).

    `adapter` is a frozen :class:`tisae.models.backbone_adapter.BackboneAdapter`.
    """
    H_grid, obs_grid = adapter.extract_hidden(batch)

    if H_grid_mean is not None:
        H_grid_input = (H_grid - H_grid_mean.unsqueeze(0)) * obs_grid.unsqueeze(-1)
    else:
        H_grid_input = H_grid

    return sae(H_grid_input, obs_grid)


# =============================================================================
# Model loading
# =============================================================================

def load_evaluation_models(
    experiment_dir: str,
    dataset_path: str,
    device: str = "cuda",
) -> Dict:
    """Load frozen BackboneAdapter + TISAE and enrich with evaluation metadata.

    Returns dict with keys:
        adapter, backbone, sae, dataset, tisae_config, H_grid_mean,
        var_names, channel_names, T, V, K_eff, C_eff, M

    `adapter` is the canonical :class:`BackboneAdapter`. `backbone`
    is a backwards-compat alias to the same object so existing eval scripts
    that have not yet been migrated keep working — they should be migrated
    to use `adapter.<method>` rather than poking the underlying nn.Module.
    """
    models = load_tisae_models(experiment_dir, dataset_path, device=device)

    cfg = models["tisae_config"]
    sae = models["sae"]
    dataset = models["dataset"]
    adapter = models["adapter"]

    # Fail fast if a CVE backbone (STraTS/HyMaTE) is paired with a dataset
    # that lacks values_zscore / times_norm.
    if adapter.has_continuous_values and not hasattr(dataset, "values_zscore"):
        raise ValueError(
            f"CVE backbone ({adapter.name}) requires a UnifiedTimeSeriesDataset "
            f"(with values_zscore / times_norm fields), but the loaded dataset "
            f"is missing these fields."
        )

    K_eff = sae.K_eff
    C_eff = sae.C_eff
    M = cfg.M

    var_names = dataset.var_names
    if C_eff > 1:
        channel_names = [CHANNEL_DEFINITIONS[i].name for i in range(C_eff)]
    else:
        channel_names = ["all"]

    models.update(
        var_names=var_names,
        channel_names=channel_names,
        T=cfg.T,
        V=cfg.V,
        K_eff=K_eff,
        C_eff=C_eff,
        M=M,
    )
    return models


# =============================================================================
# Population statistics
# =============================================================================

@torch.no_grad()
def compute_population_stats(
    dataset,
    T: int,
    V: int,
    device: torch.device,
    max_samples: int = 5000,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-variable population mean and std from training split.

    Two-pass (mean then variance) for numerical stability.
    Variables with <10 observations get std clamped to 1.0.

    Returns:
        pop_mean: [V], pop_std: [V]
    """
    train_indices = np.array(dataset.splits["train"])
    n_use = min(len(train_indices), max_samples)
    train_indices = train_indices[:n_use]

    val_sum = torch.zeros(V, dtype=torch.float64)
    obs_count = torch.zeros(V, dtype=torch.float64)

    for start in range(0, n_use, batch_size):
        end = min(start + batch_size, n_use)
        idx = train_indices[start:end]
        batch = dataset.get_batch(idx)
        batch = {k: v.to(device) for k, v in batch.items()}

        og = build_obs_grid(batch, T, V).cpu().double()
        vg = build_value_grid(batch, T, V).cpu().double()

        obs_count += og.sum(dim=(0, 1))
        val_sum += (vg * og).sum(dim=(0, 1))

    pop_mean = (val_sum / obs_count.clamp(min=1)).float()

    # Second pass for variance
    val_sq_sum = torch.zeros(V, dtype=torch.float64)
    for start in range(0, n_use, batch_size):
        end = min(start + batch_size, n_use)
        idx = train_indices[start:end]
        batch = dataset.get_batch(idx)
        batch = {k: v.to(device) for k, v in batch.items()}

        og = build_obs_grid(batch, T, V).cpu().double()
        vg = build_value_grid(batch, T, V).cpu().double()

        val_sq_sum += ((vg - pop_mean.double()) ** 2 * og).sum(dim=(0, 1))

    pop_std = (val_sq_sum / obs_count.clamp(min=1)).sqrt().float()

    rare_mask = obs_count < 10
    pop_std[rare_mask] = 1.0
    pop_std = pop_std.clamp(min=0.01)

    return pop_mean, pop_std
