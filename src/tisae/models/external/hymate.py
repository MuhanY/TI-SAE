"""HyMaTE port — Mottalib et al., ACM BCB 2025.

Source: https://github.com/mdmottalib/HyMaTE (Mottalib et al., ACM BCB 2025)

This file is a **byte-for-byte port** with the following surgical edits:

1. **Imports**:
     - `from models import TimeSeriesModel` → `from tisae.models.external.heads
       import TimeSeriesModelBase, TimeSeriesModelBaseConfig`
     - `from timm.models.layers import DropPath, to_2tuple, trunc_normal_` →
       `from torch.nn.init import trunc_normal_` (built into PyTorch >= 1.12)
       AND a vendored `DropPath` defined in this file (~10 lines).
       `to_2tuple` is imported but never used in the upstream — dropped.

2. **Upstream bug fix**:
     - Upstream `modeling_hymate.py:312` is:
           `self.att_weights = self.self_att(ts_emb, obs_mask)`
       where `ts_emb` is `[B, hid_dim]` (a 2D pooled embedding) but
       `SelfAttentionBlock.forward` expects `[B, L, D]` (3D). This crashes
       any supervised forward.
     - The intended behavior is on line 307 (commented out):
           `# self.att_weights = attention_weights`
     - Per the upstream bug fix: delete the buggy line, uncomment the intended line.
     - Net effect: `self.att_weights` stores the fusion-attention weights
       (1D per-event saliency, used for diagnostic introspection only —
       not consumed by the loss path or by the BackboneAdapter), and the
       erroneous `self_att` call is removed.

3. **No demographics** — surgical edits to `HyMaTE.forward`:
     - `demo_emb = self.demo_emb(demo) if ... else demo` line REMOVED
     - `ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)` line REPLACED
       with `ts_demo_emb = ts_emb`
     - `forward(...)` no longer takes a `demo` argument
     - `binary_head` input dim is `hid_dim` (not `hid_dim + demo_emb_dim`),
       handled inside `TimeSeriesModelBase` per the no-demographics design decision.

4. **`HyMaTEConfig` dataclass** replaces the `argparse.Namespace` glue.

5. **`get_contextual_embeddings(values, times, varis, obs_mask)`** method
   added — returns the per-event `[B, L, hid_dim]` mamba output BEFORE
   fusion-attention pool. Used by `HyMaTEBackboneAdapter` to obtain the
   per-event hidden states for the (T, V) grid scatter.

6. **`get_event_embeddings(values, times, varis)`** method added — returns
   the pre-context triplet embedding (no mamba). Used by Captum gradient
   methods. Pre-mask convention.

7. **`istrats` branch removed** — upstream supports an "istrats" mode that
   pools `triplet_emb` (raw embeddings) instead of `contextual_emb`. We
   don't use it. Removed for clarity.

8. **Train-time `var_mask` Bernoulli dropout removed**, same rationale as
   the STraTS port.

9. **`SelfAttentionBlock` is REMOVED** from the model (only the bugfix line
   307 referenced it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
# Built-in trunc_normal_ (replaces timm import)
from torch.nn.init import trunc_normal_

from tisae.models.external.heads import TimeSeriesModelBase, TimeSeriesModelBaseConfig


@dataclass
class HyMaTEConfig:
    """Hyperparameters for the HyMaTE backbone, mirroring the subset of
    `argparse.Namespace` fields that the upstream `modeling_hymate.py` reads.

    Fields:
        V                : number of variables
        hid_dim          : embedding dim (default 32 in upstream main.py)
        num_layers       : (vestigial; was passed to the upstream
                           SelfAttentionBlock which we removed). Kept in the
                           config for round-trip compatibility with saved
                           checkpoints.
        num_heads        : (vestigial)
        num_blocks       : depth of the Block_mamba stack (default 3 in upstream)
        dropout          : dropout rate (default 0.2)
        attention_dropout: (vestigial)
        pretrain         : forecast-only pretraining
        finetune         : finetune-from-pretrain (binary_head ∘ forecast_head)
        pos_class_weight : positive-class weight for BCE
    """

    V: int
    hid_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    num_blocks: int = 3
    dropout: float = 0.2
    attention_dropout: float = 0.2
    pretrain: bool = False
    finetune: bool = False
    pos_class_weight: float = 1.0


# =============================================================================
# Vendored DropPath (replaces timm.models.layers.DropPath)
# =============================================================================


class DropPath(nn.Module):
    """Stochastic depth for residual paths.

    Drops the entire residual contribution with probability `drop_prob`
    during training. Identity at eval time. Vendored from timm to avoid
    pulling in the full timm dependency for ~10 lines of code.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # Per-sample dropout: shape [B, 1, 1, ...]
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        mask = mask.div(keep)
        return x * mask


# =============================================================================
# CVE — byte-for-byte from upstream HyMaTE (matches the STraTS port)
# =============================================================================


class CVE(nn.Module):
    """Continuous Value Embedding — same as the STraTS port."""

    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = int(np.sqrt(hid_dim))
        self.W1 = nn.Parameter(torch.empty(1, int_dim), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.W2 = nn.Parameter(torch.empty(int_dim, hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]
        x = self.activation(x)
        x = torch.matmul(x, self.W2)
        return x


# =============================================================================
# FusionAtt — byte-for-byte from upstream HyMaTE (matches STraTS port)
# =============================================================================


class FusionAtt(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        int_dim = hid_dim
        self.W = nn.Parameter(torch.empty(hid_dim, int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim), requires_grad=True)
        self.u = nn.Parameter(torch.empty(int_dim, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        att = torch.matmul(x, self.W) + self.b[None, None, :]
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]
        att = att + (1 - mask) * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)
        return att


# =============================================================================
# EinFFT — Fourier-transform channel-mixing MLP (byte-for-byte from upstream)
# =============================================================================


class EinFFT(nn.Module):
    """Frequency-domain MLP block used as the channel-mixing component in
    Block_mamba. Performs FFT2 → complex linear → relu → complex linear →
    soft shrinkage → IFFT2. Vendored verbatim from upstream HyMaTE
    `modeling_hymate.py:57-107`.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size,
                        dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size,
                        dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size,
                        dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size,
                        dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum("...bd,bdk->...bk", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0])
            - self.multiply(x.imag, self.complex_weight_1[1])
            + self.complex_bias_1[0])
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1])
            + self.multiply(x.imag, self.complex_weight_1[0])
            + self.complex_bias_1[1])
        x_real_2 = (
            self.multiply(x_real_1, self.complex_weight_2[0])
            - self.multiply(x_imag_1, self.complex_weight_2[1])
            + self.complex_bias_2[0])
        x_imag_2 = (
            self.multiply(x_real_1, self.complex_weight_2[1])
            + self.multiply(x_imag_1, self.complex_weight_2[0])
            + self.complex_bias_2[1])

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # Upstream comment: "RuntimeError: 'fused_dropout' not implemented for 'ComplexFloat'"
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x


# =============================================================================
# MambaLayer — wraps the vendored Mamba with a LayerNorm
# =============================================================================


class MambaLayer(nn.Module):
    def __init__(self, dim: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        return x_mamba


# =============================================================================
# Block_mamba — Mamba + EinFFT residual block
# =============================================================================


class Block_mamba(nn.Module):
    """Stack of `blocks` residual mamba+EinFFT blocks. Byte-for-byte from
    upstream HyMaTE `modeling_hymate.py:224-266`.

    Note: `self.attn` is a SINGLE `MambaLayer` instance reused across all
    `blocks` iterations in `forward` (NOT a `nn.ModuleList`). Same for
    `self.norm2` and `self.mlp`. This means `num_blocks=3` applies the
    SAME mamba layer three times with residuals — not three independent
    mamba layers. This matches upstream HyMaTE byte-for-byte; do not
    "fix" it without re-running the upstream training to confirm.
    """

    def __init__(
        self,
        dim: int,
        blocks: int,
        mlp_ratio: float = 1.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio: int = 1,
        cm_type: str = "EinFFT",
    ):
        super().__init__()
        self.blocks = blocks
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        if cm_type == "EinFFT":
            self.mlp = EinFFT(dim)
        else:
            raise ValueError(
                f"HyMaTE port only supports cm_type='EinFFT'; got {cm_type!r}"
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_blocks(
        self,
        x: torch.Tensor,
        start_block: int = 0,
        stop_block: int | None = None,
    ) -> torch.Tensor:
        """Run blocks [start_block, stop_block). Same shared-weight iteration as forward.

        forward_blocks(x, 0, None) == forward(x)
        forward_blocks(x, 0, 1)  # first block only
        forward_blocks(x, 1, 3)  # blocks 1 and 2 (continues from block 0 output)
        """
        if stop_block is None:
            stop_block = self.blocks
        for _ in range(start_block, stop_block):
            x = x + self.drop_path(self.attn(x))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_blocks(x, 0, None)


# =============================================================================
# HyMaTE — top-level model
# =============================================================================


class HyMaTE(TimeSeriesModelBase):
    """HyMaTE backbone (Mottalib et al., ACM BCB 2025) — clean port for TISAE.

    Forward signature:
        forward(values, times, varis, obs_mask, labels=None,
                forecast_values=None, forecast_mask=None)

    Architecture (per the upstream paper):
        triplet_emb = cve_time + cve_value + variable_emb
        contextual_emb = Block_mamba(triplet_emb)
        att = fusion_att(contextual_emb, obs_mask)
        ts_emb = (contextual_emb * att).sum(dim=1)        # pooled
        logits = binary_head(ts_emb)                       # supervised
                or  forecast_head(ts_emb)                  # pretrain
    """

    def __init__(self, cfg: HyMaTEConfig):
        base_cfg = TimeSeriesModelBaseConfig(
            V=cfg.V,
            hid_dim=cfg.hid_dim,
            pretrain=cfg.pretrain,
            finetune=cfg.finetune,
            pos_class_weight=cfg.pos_class_weight,
        )
        super().__init__(base_cfg)
        self.hymate_cfg = cfg

        self.cve_time = CVE(cfg.hid_dim)
        self.cve_value = CVE(cfg.hid_dim)
        self.variable_emb = nn.Embedding(cfg.V, cfg.hid_dim)
        self.mamba = Block_mamba(cfg.hid_dim, cfg.num_blocks)
        # NOTE: upstream constructed `self.self_att = SelfAttentionBlock(...)`
        # here. Per Upstream bug fix it was vestigial / buggy and we drop it.
        self.fusion_att = FusionAtt(cfg.hid_dim)
        self.dropout_p = cfg.dropout
        self.V = cfg.V

    # ------------------------------------------------------------------
    # Helpers used by HyMaTEBackboneAdapter
    # ------------------------------------------------------------------

    def get_contextual_embeddings(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-event mamba contextual embeddings `[B, L, hid_dim]`
        AFTER the Block_mamba stack but BEFORE the fusion attention pool.
        Used by the BackboneAdapter to obtain the per-event hidden states
        that get scattered into the (T, V) grid TISAE expects.

        Note: `obs_mask` is taken as a parameter for signature parity with
        the STraTS adapter, even though the Mamba forward path doesn't use
        it (Mamba's recurrence is causal and processes all tokens; padding
        positions just get junk values which fusion_att masks out later).
        """
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        vari_emb = self.variable_emb(varis)
        triplet_emb = time_emb + value_emb + vari_emb
        triplet_emb = F.dropout(triplet_emb, self.dropout_p, self.training)
        contextual_emb = self.mamba(triplet_emb)
        return contextual_emb

    def get_contextual_embeddings_at_layer(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Return contextual embeddings after `layer` mamba blocks (0-indexed).

        layer=0  → output after 1st mamba block
        layer=-1 → all blocks (same as get_contextual_embeddings)
        layer>=num_blocks → all blocks (clamped)
        """
        triplet_emb = self.get_event_embeddings(values, times, varis)
        triplet_emb = F.dropout(triplet_emb, self.dropout_p, self.training)
        if layer == -1 or layer >= self.mamba.blocks:
            return self.mamba(triplet_emb)
        return self.mamba.forward_blocks(triplet_emb, 0, layer + 1)

    def forward_from_layer(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        start_layer: int,
    ) -> torch.Tensor:
        """Continue the mamba blocks from `start_layer`, then pool and predict.

        Used by predict_from_grid when extract_layer != -1: the SAE's
        reconstructed H_hat is at the extract layer, so we run it through
        the remaining blocks before pooling.

        Note: `obs_mask` is taken for signature parity with the STraTS
        adapter but is only used by fusion_att (not by the mamba blocks).
        """
        x = self.mamba.forward_blocks(x, start_layer, None)
        attn = self.fusion_att(x, obs_mask)[:, :, None]
        ts_emb = (x * attn).sum(dim=1)
        if self.finetune:
            return self.binary_head(self.forecast_head(ts_emb))[:, 0]
        return self.binary_head(ts_emb)[:, 0]

    def get_event_embeddings(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        varis: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-context triplet embedding `[B, L, hid_dim]` (no mamba).

        Pre-mask convention: returned tensor is NOT multiplied by obs_mask.
        """
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        vari_emb = self.variable_emb(varis)
        return time_emb + value_emb + vari_emb

    # ------------------------------------------------------------------
    # Forward — supervised + pretrain in a single signature.
    # ------------------------------------------------------------------

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
        labels: torch.Tensor = None,
        forecast_values: torch.Tensor = None,
        forecast_mask: torch.Tensor = None,
    ):
        """Full forward pass.
        """
        contextual_emb = self.get_contextual_embeddings(values, times, varis, obs_mask)

        # fusion attention
        attention_weights = self.fusion_att(contextual_emb, obs_mask)[:, :, None]
        ts_emb = (contextual_emb * attention_weights).sum(dim=1)

        self.att_weights = attention_weights

        ts_demo_emb = ts_emb

        # prediction / loss
        if self.pretrain:
            return self.forecast_final(ts_demo_emb, forecast_values, forecast_mask)

        if self.finetune:
            logits = self.binary_head(self.forecast_head(ts_demo_emb))[:, 0]
        else:
            logits = self.binary_head(ts_demo_emb)[:, 0]

        return self.binary_cls_final(logits, labels), ts_demo_emb
