"""BATBackboneAdapter — adapter for the BAT (Bi-Axial Transformer) backbone.

BAT is grid-native: it operates on [B, V, T] grids internally, which
naturally aligns with TISAE's H_grid[B, T, V, D]. No scatter is needed
for extract_hidden — just concat the two tracks and permute.

Key differences from STraTS/HyMaTE adapters:
- ``extract_hidden`` obtains H_grid directly (no event-level scatter)
- ``predict_from_grid`` splits tracks and pools at grid level
- ``pool_events`` uses masked mean (no FusionAtt — BAT's native
  pooling is grid-level masked mean; adding untrained FusionAtt would
  be meaningless since backbone training doesn't use it)
- ``forward_from_event_embeddings`` scatters events to grid via
  build_H_grid, then runs cross-parallel attention
- "attention" eval method is skipped for BAT

Module-level ``register_adapter('bat', ...)`` call at the bottom
auto-registers when this module is imported.
"""

from __future__ import annotations

import warnings
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from tisae.models.backbone_adapter import BackboneAdapter, register_adapter
from tisae.utils.grid_ops import build_H_grid, grid_to_H


class BATBackboneAdapter(BackboneAdapter):
    """BackboneAdapter for the BAT model.

    Continuous-value backbone: consumes ``values_zscore`` and
    ``times_norm`` from the unified-collate batch dict. Grid
    construction uses ``tau`` and ``varis`` for cell coordinates.
    """

    name = "bat"

    def __init__(
        self,
        module,  # tisae.models.external.bat.BAT
        T: int,
        V: int,
        var_to_ind_hash: Optional[str] = None,
        is_finetune: bool = False,
    ):
        super().__init__(T=T, V=V, D=module.bat_cfg.D, n_value_bins=0)
        self.module = module
        self.var_to_ind_hash = var_to_ind_hash
        self.is_finetune = is_finetune

    # ------------------------------------------------------------------
    # extract_hidden
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_hidden(
        self,
        batch: Dict[str, torch.Tensor],
        return_event_h: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        values = batch["values_zscore"]
        tau = batch["tau"]
        varis = batch["varis"]
        obs_mask = batch["obs_mask"]

        value_grid, obs_grid_vt = self.module.build_grid(values, tau, varis, obs_mask)
        x_embedded = self.module.embed(value_grid, obs_grid_vt)

        x_sensor, x_time = self.module.cross_parallel_attention(
            x_embedded, obs_grid_vt,
            extract_layer=self.extract_layer if self.extract_layer != -1 else -1,
        )

        # concat tracks -> [B, V, T, D] -> permute -> [B, T, V, D]
        H_grid = torch.cat([x_sensor, x_time], dim=-1).permute(0, 2, 1, 3)
        obs_grid = obs_grid_vt.permute(0, 2, 1)  # [B, T, V]

        if return_event_h:
            H_event = grid_to_H(H_grid, tau, varis)
            return H_grid, obs_grid, H_event
        return H_grid, obs_grid

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_pooled: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        values = batch["values_zscore"]
        times = batch["times_norm"]
        varis = batch["varis"]
        obs_mask = batch["obs_mask"]
        tau = batch["tau"]

        value_grid, obs_grid_vt = self.module.build_grid(values, tau, varis, obs_mask)
        x_embedded = self.module.embed(value_grid, obs_grid_vt)
        x_sensor, x_time = self.module.cross_parallel_attention(x_embedded, obs_grid_vt)

        # Pool + merge + predict
        pooled_raw = self.module.pool_grid(x_sensor, x_time, obs_grid_vt)  # [B, D]
        ts_emb = self.module.nonlinear_merger(pooled_raw).relu()  # [B, D]

        if self.is_finetune:
            logits = self.module.binary_head(self.module.forecast_head(ts_emb))[:, 0]
        else:
            logits = self.module.binary_head(ts_emb)[:, 0]

        # Contextual embeddings for "H" field
        H_grid = torch.cat([x_sensor, x_time], dim=-1).permute(0, 2, 1, 3)
        H_event = grid_to_H(H_grid, tau, varis)

        result: Dict[str, torch.Tensor] = {"logits": logits, "H": H_event}
        if return_pooled:
            # Return pre-merger pooled (lesson #6: avoid double-merger)
            result["pooled"] = pooled_raw
        if return_attention:
            total = obs_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            result["attention"] = obs_mask / total
        return result

    # ------------------------------------------------------------------
    # predict_from_grid
    # ------------------------------------------------------------------

    def predict_from_grid(
        self,
        H_hat_grid: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        obs_grid: torch.Tensor,
    ) -> torch.Tensor:
        # H_hat_grid: [B, T, V, D], obs_grid: [B, T, V]
        if self.H_grid_mean is not None:
            H_hat_grid = H_hat_grid + self.H_grid_mean[None, None, :, :]

        E = self.D // 2
        grid_vt = H_hat_grid.permute(0, 2, 1, 3)  # [B, V, T, D]
        x_sensor = grid_vt[:, :, :, :E]  # [B, V, T, E]
        x_time = grid_vt[:, :, :, E:]     # [B, V, T, E]
        obs_grid_vt = obs_grid.permute(0, 2, 1)  # [B, V, T]

        if self.extract_layer != -1:
            return self.module.forward_from_grid_state(
                x_sensor, x_time, obs_grid_vt, self.extract_layer,
            )

        pooled = self.module.pool_grid(x_sensor, x_time, obs_grid_vt)
        ts_emb = self.module.nonlinear_merger(pooled).relu()
        if self.is_finetune:
            return self.module.binary_head(self.module.forecast_head(ts_emb))[:, 0]
        return self.module.binary_head(ts_emb)[:, 0]

    # ------------------------------------------------------------------
    # pool_events — masked mean
    # ------------------------------------------------------------------

    def pool_events(
        self,
        H_event: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = obs_mask.unsqueeze(-1)  # [B, L, 1]
        pooled = (H_event * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        attn = obs_mask / obs_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled, attn

    def predict_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled is pre-merger [B, D] (from pool_events or forward's "pooled")
        ts_emb = self.module.nonlinear_merger(pooled).relu()
        if self.is_finetune:
            return self.module.binary_head(self.module.forecast_head(ts_emb))[:, 0]
        return self.module.binary_head(ts_emb)[:, 0]

    # ------------------------------------------------------------------
    # Captum gradient path
    # ------------------------------------------------------------------

    def forward_from_event_embeddings(
        self,
        emb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Captum path: scatter [B,L,D] to grid, run full model.

        ``emb`` is from ``get_event_embeddings`` which duplicates the
        E-dim embedding to D=2E. Both halves remain equal through
        Captum interpolation, so we take the first half as input to
        cross_parallel_attention.
        """
        obs_mask = batch["obs_mask"]
        tau = batch["tau"]
        varis = batch["varis"]

        # Captum IG expands embeddings to [n_steps, L, D] but batch stays [1, L]
        B_emb = emb.shape[0]
        if tau.shape[0] != B_emb:
            tau = tau.expand(B_emb, -1)
            varis = varis.expand(B_emb, -1)
            obs_mask = obs_mask.expand(B_emb, -1)

        # Scatter to grid (gradient flows through build_H_grid)
        H_grid, _ = build_H_grid(emb, tau, varis, obs_mask, self.T, self.V)

        # Take first E dims as embedded grid input
        E = self.D // 2
        x_embedded = H_grid[:, :, :, :E].permute(0, 2, 1, 3)  # [B, V, T, E]

        # obs_grid for masking (no gradient needed)
        with torch.no_grad():
            vals_exp = batch["values_zscore"]
            if vals_exp.shape[0] != B_emb:
                vals_exp = vals_exp.expand(B_emb, -1)
            _, obs_grid_vt = self.module.build_grid(
                vals_exp, tau, varis, obs_mask,
            )

        x_sensor, x_time = self.module.cross_parallel_attention(x_embedded, obs_grid_vt)

        pooled = self.module.pool_grid(x_sensor, x_time, obs_grid_vt)
        ts_emb = self.module.nonlinear_merger(pooled).relu()
        if self.is_finetune:
            return self.module.binary_head(self.module.forecast_head(ts_emb))[:, 0]
        return self.module.binary_head(ts_emb)[:, 0]

    def get_event_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.module.get_event_embeddings(
            batch["values_zscore"],
            batch["times_norm"],
            batch["varis"],
            tau=batch["tau"],
            obs_mask=batch["obs_mask"],
        )

    def get_baseline_event_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "median",
    ) -> torch.Tensor:
        if mode not in ("median", "zero"):
            raise ValueError(f"Unknown baseline mode: {mode!r}")
        zero_zscore = torch.zeros_like(batch["values_zscore"])
        return self.module.get_event_embeddings(
            zero_zscore,
            batch["times_norm"],
            batch["varis"],
            tau=batch["tau"],
            obs_mask=batch["obs_mask"],
        )

    # ------------------------------------------------------------------
    # Attention grid
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_attention_grid(
        self,
        batch: Dict[str, torch.Tensor],
        T: int,
        V: int,
    ) -> torch.Tensor:
        """Uniform attention over observed cells."""
        _, obs_grid_vt = self.module.build_grid(
            batch["values_zscore"], batch["tau"], batch["varis"], batch["obs_mask"],
        )
        obs_grid = obs_grid_vt.permute(0, 2, 1)  # [B, T, V]
        total = obs_grid.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        return obs_grid / total

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_continuous_values(self) -> bool:
        return True

    def forward_with_value_mask(
        self,
        batch: Dict[str, torch.Tensor],
        soft_mask: torch.Tensor,
    ) -> torch.Tensor:
        new_batch = {**batch, "values_zscore": batch["values_zscore"] * soft_mask}
        return self.forward(new_batch)["logits"]

    # ------------------------------------------------------------------
    # extract_layer override (BAT depth = 2 * layers)
    # ------------------------------------------------------------------

    def set_extract_layer(self, layer: int) -> None:
        if layer < -1:
            n = 2 * len(self.module.attn_sensor.blocks)
            layer = n + layer
            if layer < 0:
                raise ValueError(
                    f"extract_layer resolves to {layer} (BAT has "
                    f"{n} total layers across 2 rounds)"
                )
        self._extract_layer = layer

    # ------------------------------------------------------------------
    # Value replacement (D10: same pattern as StratsBackboneAdapter)
    # ------------------------------------------------------------------

    def _sample_population_zscore(
        self,
        var_ids: torch.Tensor,
        population_values_zscore: Dict[int, np.ndarray],
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = id(population_values_zscore)
        if getattr(self, "_median_zscore_cache_key", None) != cache_key:
            V = max(population_values_zscore.keys()) + 1
            lut = torch.zeros(V, dtype=torch.float32)
            for v, pool in population_values_zscore.items():
                if pool is not None and len(pool) > 0:
                    lut[int(v)] = float(np.median(pool))
            self._median_zscore_cache_key = cache_key
            self._median_zscore_lut = lut

        lut = self._median_zscore_lut.to(device)  # type: ignore[attr-defined]
        return lut[var_ids.long()]

    def replace_event_value(
        self,
        batch: Dict[str, torch.Tensor],
        replace_mask: torch.Tensor,
        strategy: str,
        marginal_dist: Optional[torch.Tensor] = None,
        population_values_zscore: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, torch.Tensor]:
        new_batch = {k: v for k, v in batch.items()}
        replace_mask = replace_mask.bool()
        device = batch["values_zscore"].device

        if strategy == "removal":
            new_batch["obs_mask"] = batch["obs_mask"] * (~replace_mask).float()
            return new_batch

        if population_values_zscore is None:
            raise ValueError(
                f"BATBackboneAdapter.replace_event_value strategy={strategy!r} "
                f"requires `population_values_zscore` from the unified dataset"
            )

        vz = batch["values_zscore"].clone()
        varis = batch["varis"]

        if strategy == "marginal":
            if replace_mask.any():
                rep_vars = varis[replace_mask]
                vz[replace_mask] = self._sample_population_zscore(
                    rep_vars, population_values_zscore, device,
                )
            new_batch["values_zscore"] = vz
            return new_batch

        if strategy == "conditional":
            tau = batch["tau"]
            obs_mask = batch["obs_mask"]
            keep_mask = (obs_mask > 0.5) & (~replace_mask)

            same_var = varis.unsqueeze(2) == varis.unsqueeze(1)
            valid = same_var & keep_mask.unsqueeze(1)
            dists = (tau.unsqueeze(2) - tau.unsqueeze(1)).abs().float()
            dists = dists.masked_fill(~valid, float("inf"))

            nearest_j = dists.argmin(dim=2)
            nearest_vz = vz.gather(1, nearest_j)
            nearest_dist = dists.gather(2, nearest_j.unsqueeze(2)).squeeze(2)
            has_match = nearest_dist < float("inf")

            matched_positions = replace_mask & has_match
            vz[matched_positions] = nearest_vz[matched_positions]

            fallback_positions = replace_mask & ~has_match
            if fallback_positions.any():
                fb_vars = varis[fallback_positions]
                vz[fallback_positions] = self._sample_population_zscore(
                    fb_vars, population_values_zscore, device,
                )
            new_batch["values_zscore"] = vz
            return new_batch

        raise ValueError(f"Unknown replacement strategy: {strategy!r}")


# =============================================================================
# Factory
# =============================================================================


def _make_bat_adapter(
    ckpt: Dict[str, Any],
    device: torch.device,
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> BATBackboneAdapter:
    from tisae.models.external.bat import BAT, BATConfig

    bb_cfg_dict = ckpt.get("backbone_config", {})
    valid_fields = {f.name for f in fields(BATConfig)}
    cfg = BATConfig(**{k: v for k, v in bb_cfg_dict.items() if k in valid_fields})

    module = BAT(cfg)
    missing, unexpected = module.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        msg = (
            f"BAT state_dict load mismatch:\n"
            f"  missing keys ({len(missing)}): {missing[:8]}\n"
            f"  unexpected keys ({len(unexpected)}): {unexpected[:8]}"
        )
        if missing:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
    module = module.to(device).eval()

    md = ckpt.get("unified_dataset_metadata") or {}
    dm = dataset_meta or {}
    T = md.get("T") or dm.get("T") or cfg.T
    V = md.get("V") or dm.get("V") or cfg.V
    h = md.get("var_to_ind_hash") or dm.get("var_to_ind_hash")

    adapter = BATBackboneAdapter(
        module,
        T=int(T),
        V=int(V),
        var_to_ind_hash=h,
        is_finetune=bool(ckpt.get("is_finetune", False)),
    )
    return adapter.freeze()


# Self-register when this module is imported.
register_adapter("bat", _make_bat_adapter)
