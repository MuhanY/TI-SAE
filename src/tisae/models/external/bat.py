"""BAT (Bi-Axial Transformer) backbone port — Gupta et al., 2024.

Source: https://github.com/rvandewater/YAIB (DeVries et al., arXiv 2508.12418)
Original class: `EncoderClassifierCrossParallel`.

PreNormTransformerEncoder is a pure-PyTorch replacement for
x_transformers.Encoder v1.16.23 as used by YAIB's BAT.

Key specs (verified empirically against x_transformers v1.16.23):
- dim_head=64 (x_transformers DEFAULT_DIM_HEAD), inner_dim = heads * 64
- Q/K/V/Out projections: bias=False
- LayerNorm: affine=True (standard nn.LayerNorm)
- FFN: Linear(bias=True) → GELU → Dropout → Linear(bias=True), no trailing dropout
- Pre-norm residual pattern
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from tisae.models.external.heads import TimeSeriesModelBase, TimeSeriesModelBaseConfig
from tisae.utils.grid_ops import grid_to_H


DEFAULT_DIM_HEAD = 64


class MultiHeadAttention(nn.Module):
    """Multi-head attention with fixed dim_head=64 (matching x_transformers).

    Unlike nn.MultiheadAttention which uses head_dim = embed_dim // num_heads,
    this uses a fixed dim_head so inner_dim = heads * 64 can exceed input dim.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = DEFAULT_DIM_HEAD,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq, dim]
            key_padding_mask: [B, seq] True=IGNORE (PyTorch MHA convention)
        """
        B, L, _ = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, L, h, self.dim_head).transpose(1, 2)  # [B, h, L, dh]
        k = self.to_k(x).reshape(B, L, h, self.dim_head).transpose(1, 2)
        v = self.to_v(x).reshape(B, L, h, self.dim_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, L, L]

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf"),
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, h * self.dim_head)  # [B, L, inner]
        return self.to_out(out)  # [B, L, dim]


class PreNormTransformerBlock(nn.Module):
    """Pre-norm transformer block matching x_transformers v1.16.23 defaults.

    LN(affine=True) → MHA(dim_head=64, bias=False) → residual
    LN(affine=True) → FFN(bias=True, GELU, no trailing dropout) → residual
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = DEFAULT_DIM_HEAD,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout=attn_dropout)
        self.norm_ff = nn.LayerNorm(dim)
        ff_dim = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.norm_ff(x))
        return x


class PreNormTransformerEncoder(nn.Module):
    """Stack of PreNormTransformerBlocks with a final LayerNorm.

    Supports partial execution via ``stop_after_layer`` (for extract_layer)
    and ``start_from_layer`` (for continuing from an intermediate state).
    When stopped early, the final LayerNorm is NOT applied.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = DEFAULT_DIM_HEAD,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PreNormTransformerBlock(
                dim, heads, dim_head, ff_mult, attn_dropout, ff_dropout,
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        stop_after_layer: Optional[int] = None,
        start_from_layer: int = 0,
    ) -> torch.Tensor:
        """Forward through blocks ``[start_from_layer, ...]``.

        Args:
            x: [B, seq_len, dim]
            mask: [B, seq_len] True=valid, False=padding
                  (x_transformers convention; inverted for MHA).
            stop_after_layer: return after this block index (no final_norm).
            start_from_layer: skip blocks before this index.
        """
        kpm = ~mask if mask is not None else None
        for i in range(start_from_layer, len(self.blocks)):
            x = self.blocks[i](x, key_padding_mask=kpm)
            if stop_after_layer is not None and i == stop_after_layer:
                return x
        return self.final_norm(x)


# =============================================================================
# PositionalEncodingTF — byte-for-byte from YAIB (sinusoidal)
# =============================================================================


class PositionalEncodingTF(nn.Module):
    """Sinusoidal positional encoding from SEFT / BAT.

    Byte-for-byte port of YAIB BAT EncoderClassifierCrossParallel.
    Only changes: type annotations, super() modernized, dead variable
    ``B`` removed, ``getPE`` inlined into ``forward``.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self._num_timescales = d_model // 2

    def forward(self, P_time: torch.Tensor) -> torch.Tensor:
        """Args:
            P_time: [*, T] float tensor of time indices.
        Returns:
            [*, T, d_model] positional encoding.
        """
        P_time = P_time.float()
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)
        times = P_time.unsqueeze(-1)  # [*, T, 1]
        scaled_time = times / torch.tensor(
            timescales, device=P_time.device, dtype=torch.float32,
        )  # [*, T, num_timescales]
        pe = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1,
        )  # [*, T, d_model]
        return pe


# =============================================================================
# BATConfig
# =============================================================================


def _auto_sensor_encoding_out(V: int) -> int:
    """BAT's original formula for sensor encoding dimension."""
    out = min(600, round(1.6 * V ** 0.5))
    if out % 2 != 0:
        out += 1
    return out


@dataclass
class BATConfig:
    """Configuration for the BAT backbone.

    ``embed_out`` (value_embed_size) is the primary tunable parameter.
    ``sensor_encoding_out`` is auto-computed from V by default (BAT's
    original formula: ``min(600, round(1.6 * V^0.5))``).
    ``E = embed_out + sensor_encoding_out``, ``D = 2 * E``.
    """

    V: int
    T: int = 48
    embed_out: int = 16
    sensor_encoding_out: int = -1  # -1 = auto-compute from V
    layers: int = 1
    heads: int = 1
    dropout: float = 0.2
    attn_dropout: float = 0.2
    use_mask: bool = False
    pooling: str = "mean"
    max_len: int = 500
    pretrain: bool = False
    finetune: bool = False
    pos_class_weight: float = 1.0

    def __post_init__(self):
        if self.sensor_encoding_out < 0:
            self.sensor_encoding_out = _auto_sensor_encoding_out(self.V)

    @property
    def E(self) -> int:
        return self.embed_out + self.sensor_encoding_out

    @property
    def D(self) -> int:
        return 2 * self.E


# =============================================================================
# BAT model
# =============================================================================


class BAT(TimeSeriesModelBase):
    """Bi-Axial Transformer (Gupta et al., 2024) — clean port for TISAE.

    Architecture::

        build_grid -> embed -> 2-round cross-parallel attention
            -> grid-level masked mean pool -> merger -> prediction head

    Internal convention: grids are ``[B, V, T, E]`` (BAT-native).
    TISAE convention: ``H_grid[B, T, V, D]``. Conversion at adapter boundary.
    """

    def __init__(self, cfg: BATConfig):
        base_cfg = TimeSeriesModelBaseConfig(
            V=cfg.V,
            hid_dim=cfg.D,
            pretrain=cfg.pretrain,
            finetune=cfg.finetune,
            pos_class_weight=cfg.pos_class_weight,
        )
        super().__init__(base_cfg)
        self.bat_cfg = cfg

        # D11: [value, mask_indicator] -> Linear(2, embed_out)
        self.time_embedding = nn.Linear(2, cfg.embed_out)
        self.sensor_encoding = nn.Embedding(cfg.V, cfg.sensor_encoding_out)
        self.pos_encoder = PositionalEncodingTF(cfg.E, max_len=cfg.max_len)

        # Cross-parallel attention: 2 shared modules, 4 calls (2 rounds)
        self.attn_sensor = PreNormTransformerEncoder(
            dim=cfg.E, depth=cfg.layers, heads=cfg.heads,
            attn_dropout=cfg.attn_dropout, ff_dropout=cfg.dropout,
        )
        self.attn_time = PreNormTransformerEncoder(
            dim=cfg.E, depth=cfg.layers, heads=cfg.heads,
            attn_dropout=cfg.attn_dropout, ff_dropout=cfg.dropout,
        )

        # Merger: [B, D] -> [B, D] (applied after pooling, before head)
        self.nonlinear_merger = nn.Linear(cfg.D, cfg.D)

        self.dropout_p = cfg.dropout

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def build_grid(
        self,
        values_zscore: torch.Tensor,
        tau: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatter sparse events into dense grids.

        Returns:
            value_grid ``[B, V, T]``: mean-aggregated z-scored values
            obs_grid_vt ``[B, V, T]``: binary observation mask
        """
        B, L = values_zscore.shape
        V, T = self.bat_cfg.V, self.bat_cfg.T
        device = values_zscore.device

        batch_idx = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
        tau_long = tau.long()
        varis_long = varis.long()
        flat_idx = (batch_idx * V * T + varis_long * T + tau_long).reshape(-1)

        vals_flat = values_zscore.reshape(-1)
        mask_flat = obs_mask.reshape(-1).float()

        val_sum = torch.zeros(B * V * T, device=device, dtype=values_zscore.dtype)
        counts = torch.zeros(B * V * T, device=device)
        val_sum.scatter_add_(0, flat_idx, vals_flat * mask_flat)
        counts.scatter_add_(0, flat_idx, mask_flat)

        value_grid = (val_sum / counts.clamp_min(1)).view(B, V, T)
        obs_grid_vt = (counts.view(B, V, T) > 0).float()
        return value_grid, obs_grid_vt

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        value_grid: torch.Tensor,
        obs_grid_vt: torch.Tensor,
    ) -> torch.Tensor:
        """Embed the dense grid into ``[B, V, T, E]``."""
        B, V, T = value_grid.shape
        device = value_grid.device

        # Value + mask -> embed_out  (D11)
        x = torch.stack([value_grid, obs_grid_vt], dim=-1)  # [B, V, T, 2]
        x = self.time_embedding(x)  # [B, V, T, embed_out]

        # Sensor encoding
        s_idx = torch.arange(V, device=device)
        s_emb = self.sensor_encoding(s_idx)  # [V, sensor_encoding_out]
        x = torch.cat(
            [x, s_emb[None, :, None, :].expand(B, -1, T, -1)], dim=-1,
        )  # [B, V, T, E]

        # Sinusoidal PE (D13: integer bin indices)
        with torch.no_grad():
            t_idx = torch.arange(T, device=device).float()
            pe = self.pos_encoder(t_idx.unsqueeze(0).expand(B, -1))  # [B, T, E]
        x = x + pe[:, None, :, :]  # broadcast over V

        return x  # [B, V, T, E]

    # ------------------------------------------------------------------
    # Cross-parallel attention
    # ------------------------------------------------------------------

    def cross_parallel_attention(
        self,
        x_embedded: torch.Tensor,
        obs_grid_vt: torch.Tensor,
        extract_layer: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run 2-round cross-parallel attention.

        Round 1: attn_sensor on ``[B*T, V, E]``, attn_time on ``[B*V, T, E]``
        Round 2 (cross): swap tracks through the other module.

        Returns ``(x_sensor, x_time)`` each ``[B, V, T, E]``.
        """
        B, V, T, E = x_embedded.shape
        lpr = len(self.attn_sensor.blocks)  # layers per round

        # D12: timepoint mask from obs_grid
        time_mask: Optional[torch.Tensor] = None
        if self.bat_cfg.use_mask:
            tp_mask = obs_grid_vt.any(dim=1)  # [B, T]
            time_mask = tp_mask[:, None, :].expand(-1, V, -1).reshape(B * V, T)

        # Split into tracks
        x_sensor = x_embedded.permute(0, 2, 1, 3).reshape(B * T, V, E)  # [B*T, V, E]
        x_time = x_embedded.reshape(B * V, T, E)                         # [B*V, T, E]

        # --- Round 1 ---
        # stop_after_layer=None means "run all blocks + final_norm".
        # When extracting at the last block of a round, we must NOT use
        # stop_after_layer (which skips final_norm); instead run fully
        # and control early return via a separate flag.
        extract_in_r1 = 0 <= extract_layer < lpr
        r1_stop = extract_layer if (extract_in_r1 and extract_layer < lpr - 1) else None
        x_sensor = self.attn_sensor(x_sensor, stop_after_layer=r1_stop)  # [B*T, V, E]
        x_time = self.attn_time(x_time, mask=time_mask, stop_after_layer=r1_stop)  # [B*V, T, E]

        x_sensor = x_sensor.reshape(B, T, V, E)  # [B, T, V, E]
        x_time = x_time.reshape(B, V, T, E)      # [B, V, T, E]

        if extract_in_r1:
            # Normalize x_sensor to [B, V, T, E] before returning
            return x_sensor.permute(0, 2, 1, 3), x_time

        # --- Round 2: cross ---
        # Normalize x_sensor to [B, V, T, E] for consistent round-2 entry
        x_sensor = x_sensor.permute(0, 2, 1, 3)  # [B, T, V, E] -> [B, V, T, E]

        r2_stop = None
        if extract_layer >= lpr:
            r2_layer = extract_layer - lpr
            # Only use stop_after_layer for non-final blocks; final block
            # needs full encoder pass (including final_norm).
            if r2_layer < lpr - 1:
                r2_stop = r2_layer

        x_sensor, x_time = self._run_round2(
            x_sensor, x_time, time_mask, B, V, T, E, r2_stop,
        )
        return x_sensor, x_time

    def _run_round2(
        self,
        x_sensor: torch.Tensor,
        x_time: torch.Tensor,
        time_mask: Optional[torch.Tensor],
        B: int, V: int, T: int, E: int,
        stop: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Round 2 of cross-parallel attention.

        Both inputs MUST be ``[B, V, T, E]`` (canonical shape).
        Returns ``(x_sensor, x_time)`` both ``[B, V, T, E]``.
        """
        # x_time -> sensor attention: need [B*T, V, E]
        x_time_r2 = x_time.permute(0, 2, 1, 3).reshape(B * T, V, E)  # [B,V,T,E]->[B,T,V,E]->[B*T,V,E]
        x_time_r2 = self.attn_sensor(x_time_r2, stop_after_layer=stop)
        x_time = x_time_r2.reshape(B, T, V, E).permute(0, 2, 1, 3)   # -> [B, V, T, E]

        # x_sensor -> time attention: need [B*V, T, E]
        x_sensor_r2 = x_sensor.reshape(B * V, T, E)  # [B,V,T,E]->[B*V,T,E] (contiguous, no permute)
        x_sensor_r2 = self.attn_time(x_sensor_r2, mask=time_mask, stop_after_layer=stop)
        x_sensor = x_sensor_r2.reshape(B, V, T, E)   # -> [B, V, T, E]

        return x_sensor, x_time

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def pool_grid(
        self,
        x_sensor: torch.Tensor,
        x_time: torch.Tensor,
        obs_grid_vt: torch.Tensor,
    ) -> torch.Tensor:
        """Pool both tracks from ``[B, V, T, E]`` -> ``[B, D]``."""
        pooling = self.bat_cfg.pooling
        if pooling == "max":
            if self.bat_cfg.use_mask:
                mask = obs_grid_vt.unsqueeze(-1)  # [B, V, T, 1]
                neg_inf = torch.finfo(x_sensor.dtype).min
                x_sensor = x_sensor.masked_fill(mask == 0, neg_inf)
                x_time = x_time.masked_fill(mask == 0, neg_inf)
            s_pooled = x_sensor.amax(dim=(1, 2))  # [B, E]
            t_pooled = x_time.amax(dim=(1, 2))
        elif pooling == "mean":
            if self.bat_cfg.use_mask:
                mask = obs_grid_vt.unsqueeze(-1)
                total = mask.sum(dim=(1, 2)).clamp_min(1.0)
                s_pooled = (x_sensor * mask).sum(dim=(1, 2)) / total
                t_pooled = (x_time * mask).sum(dim=(1, 2)) / total
            else:
                s_pooled = x_sensor.mean(dim=(1, 2))
                t_pooled = x_time.mean(dim=(1, 2))
        else:
            raise ValueError(f"Unknown pooling: {pooling!r}")
        return torch.cat([s_pooled, t_pooled], dim=-1)  # [B, D]

    # ------------------------------------------------------------------
    # Continue attention from intermediate state (for predict_from_grid)
    # ------------------------------------------------------------------

    def forward_from_grid_state(
        self,
        x_sensor: torch.Tensor,
        x_time: torch.Tensor,
        obs_grid_vt: torch.Tensor,
        from_extract_layer: int,
    ) -> torch.Tensor:
        """Continue from an extracted grid state to logits.

        Both tracks enter as ``[B, V, T, E]`` (canonical output of
        cross_parallel_attention).
        """
        lpr = len(self.attn_sensor.blocks)
        next_layer = from_extract_layer + 1
        B, V, T, E = x_sensor.shape

        time_mask: Optional[torch.Tensor] = None
        if self.bat_cfg.use_mask:
            tp_mask = obs_grid_vt.any(dim=1)
            time_mask = tp_mask[:, None, :].expand(-1, V, -1).reshape(B * V, T)

        round_num = next_layer // lpr
        internal_start = next_layer % lpr

        # Finish round 1 if still in it
        if round_num == 0:
            # x_sensor[B,V,T,E] -> [B*T,V,E] for sensor attention
            x_s = x_sensor.permute(0, 2, 1, 3).reshape(B * T, V, E)
            # x_time[B,V,T,E] -> [B*V,T,E] for time attention
            x_t = x_time.reshape(B * V, T, E)
            x_s = self.attn_sensor(x_s, start_from_layer=internal_start)
            x_t = self.attn_time(x_t, mask=time_mask, start_from_layer=internal_start)
            # Normalize both to [B, V, T, E] for round 2
            x_sensor = x_s.reshape(B, T, V, E).permute(0, 2, 1, 3)  # [B, V, T, E]
            x_time = x_t.reshape(B, V, T, E)                          # [B, V, T, E]
            round_num = 1
            internal_start = 0

        # Run (remaining) round 2
        if round_num == 1:
            r2_start = internal_start
            # x_time -> sensor attention: [B,V,T,E]->[B*T,V,E]
            x_time_r2 = x_time.permute(0, 2, 1, 3).reshape(B * T, V, E)
            x_time_r2 = self.attn_sensor(x_time_r2, start_from_layer=r2_start)
            x_time = x_time_r2.reshape(B, T, V, E).permute(0, 2, 1, 3)  # [B, V, T, E]

            # x_sensor -> time attention: [B,V,T,E]->[B*V,T,E]
            x_sensor_r2 = x_sensor.reshape(B * V, T, E)
            x_sensor_r2 = self.attn_time(x_sensor_r2, mask=time_mask, start_from_layer=r2_start)
            x_sensor = x_sensor_r2.reshape(B, V, T, E)  # [B, V, T, E]

        # Pool + predict
        pooled = self.pool_grid(x_sensor, x_time, obs_grid_vt)
        ts_emb = self.nonlinear_merger(pooled).relu()
        if self.finetune:
            return self.binary_head(self.forecast_head(ts_emb))[:, 0]
        return self.binary_head(ts_emb)[:, 0]

    # ------------------------------------------------------------------
    # Contextual embeddings (for adapter)
    # ------------------------------------------------------------------

    def _resolve_tau(
        self,
        times_norm: torch.Tensor,
        tau: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if tau is not None:
            return tau
        # times_norm is in [-1, 1] (unified dataset convention)
        t01 = (times_norm + 1.0) * 0.5
        return (t01 * (self.bat_cfg.T - 1)).round().long().clamp(0, self.bat_cfg.T - 1)

    def get_contextual_embeddings(
        self,
        values_zscore: torch.Tensor,
        times_norm: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
        *,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-event contextual embeddings ``[B, L, D]`` after full attention."""
        tau = self._resolve_tau(times_norm, tau)
        value_grid, obs_grid_vt = self.build_grid(values_zscore, tau, varis, obs_mask)
        x_embedded = self.embed(value_grid, obs_grid_vt)
        x_sensor, x_time = self.cross_parallel_attention(x_embedded, obs_grid_vt)
        # [B, V, T, E] x2 -> cat -> [B, V, T, D] -> permute -> [B, T, V, D]
        H_grid = torch.cat([x_sensor, x_time], dim=-1).permute(0, 2, 1, 3)
        return grid_to_H(H_grid, tau, varis)

    def get_contextual_embeddings_at_layer(
        self,
        values_zscore: torch.Tensor,
        times_norm: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
        layer: int,
        *,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-event embeddings at an intermediate attention layer."""
        tau = self._resolve_tau(times_norm, tau)
        value_grid, obs_grid_vt = self.build_grid(values_zscore, tau, varis, obs_mask)
        x_embedded = self.embed(value_grid, obs_grid_vt)
        x_sensor, x_time = self.cross_parallel_attention(
            x_embedded, obs_grid_vt, extract_layer=layer,
        )
        H_grid = torch.cat([x_sensor, x_time], dim=-1).permute(0, 2, 1, 3)
        return grid_to_H(H_grid, tau, varis)

    def get_event_embeddings(
        self,
        values_zscore: torch.Tensor,
        times_norm: torch.Tensor,
        varis: torch.Tensor,
        *,
        tau: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-attention event embeddings ``[B, L, D]``.

        Pre-mask convention: NOT multiplied by obs_mask.
        Both tracks start from the same embedding, so we duplicate to D=2E.
        """
        tau = self._resolve_tau(times_norm, tau)
        if obs_mask is None:
            obs_mask = torch.ones_like(values_zscore)
        value_grid, obs_grid_vt = self.build_grid(values_zscore, tau, varis, obs_mask)
        x_embedded = self.embed(value_grid, obs_grid_vt)  # [B, V, T, E]
        # Duplicate to D = 2E (both tracks start from same embedding)
        x_pre = torch.cat([x_embedded, x_embedded], dim=-1)  # [B, V, T, D]
        x_pre = x_pre.permute(0, 2, 1, 3)  # [B, T, V, D]
        return grid_to_H(x_pre, tau, varis)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        values_zscore: torch.Tensor,
        times_norm: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        forecast_values: Optional[torch.Tensor] = None,
        forecast_mask: Optional[torch.Tensor] = None,
        *,
        tau: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Full forward pass.

        Returns:
            pretrain: forecast MSE loss (scalar)
            supervised: (loss_or_probs, ts_emb)
        """
        tau = self._resolve_tau(times_norm, tau)

        value_grid, obs_grid_vt = self.build_grid(values_zscore, tau, varis, obs_mask)
        x_embedded = self.embed(value_grid, obs_grid_vt)
        x_sensor, x_time = self.cross_parallel_attention(x_embedded, obs_grid_vt)

        pooled = self.pool_grid(x_sensor, x_time, obs_grid_vt)  # [B, D]
        ts_emb = self.nonlinear_merger(pooled).relu()  # [B, D]

        if self.pretrain:
            return self.forecast_final(ts_emb, forecast_values, forecast_mask)

        if self.finetune:
            logits = self.binary_head(self.forecast_head(ts_emb))[:, 0]
        else:
            logits = self.binary_head(ts_emb)[:, 0]

        return self.binary_cls_final(logits, labels), ts_emb
