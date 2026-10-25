"""STraTS port — Tipirneni & Reddy, TKDD 2022.

Source: https://github.com/sindhura97/STraTS (Tipirneni & Reddy, TKDD 2022).

This file is a **byte-for-byte port** with the following surgical edits:

1. **Imports**:
     - `from models import TimeSeriesModel`
       → `from tisae.models.external.heads import TimeSeriesModelBase, TimeSeriesModelBaseConfig`
     - `from argparse import Namespace` removed; `Strats.__init__` accepts a
       :class:`StratsConfig` dataclass instead of an `argparse.Namespace`.

2. **No demographics** — surgical edits to `Strats.forward`:
     - `demo_emb = self.demo_emb(demo) if ... else demo` line REMOVED
     - `ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)` line REPLACED with
       `ts_demo_emb = ts_emb`
     - `forward(...)` no longer takes a `demo` argument
     - `binary_head` input dim is `hid_dim` (not `hid_dim + demo_emb_dim`),
       handled inside `TimeSeriesModelBase` per the no-demographics design decision.

3. **Finetune dispatch** — kept unchanged from upstream: when
   `self.finetune` is True (set via `StratsConfig.finetune`), the prediction
   path stacks `binary_head ∘ forecast_head`.

4. **`get_contextual_embeddings` method added** — returns the per-event
   `[B, L, hid_dim]` tensor that comes out of the transformer BEFORE
   `fusion_att` pools it. The :class:`StratsBackboneAdapter` (in
   `tisae.models.backbone_adapter`) calls this and scatters into the (T, V)
   grid that TISAE expects.

5. **`istrats` branch removed** — the upstream supports an "istrats" mode that
   pools `triplet_emb` (raw embeddings) instead of `contextual_emb` (transformer
   output). We don't use it, and keeping it would require carrying a `model_type`
   string through the adapter. Removed for clarity.

6. **`var_mask` train-time dropout removed**. Upstream randomly masks events at
   training time via a per-(batch, var) Bernoulli draw inside `forward`. This
   is a regularization that the TISAE training pipeline does not use (it
   conflicts with the bit-equivalence contract: the mask state changes
   per-batch even with a fixed seed). Surgical removal documented here so any
   reproduction of upstream STraTS numbers must add this back. The TISAE-side
   regularization (dropout in `binary_head` and `forecast_head` is provided by
   the heads themselves) is preserved.

The non-edited code (CVE, FusionAtt, Transformer) is byte-for-byte from the
upstream file. Do not "improve" the math without re-running validation on
both pretraining and finetuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tisae.models.external.heads import TimeSeriesModelBase, TimeSeriesModelBaseConfig


@dataclass
class StratsConfig:
    """Hyperparameters for the STraTS backbone, mirroring the subset of
    `argparse.Namespace` fields that the upstream `modeling_strats.py` reads.

    Fields:
        V                : number of variables (used for variable embedding +
                           forecast_head output dim)
        hid_dim          : transformer hidden dim (default 32 in upstream main.py)
        num_layers       : transformer depth (default 2)
        num_heads        : transformer attention heads (default 4)
        dropout          : dropout rate everywhere (default 0.2)
        attention_dropout: attention-matrix dropout (default 0.2)
        pretrain         : True for forecast-only pretraining
        finetune         : True for finetune-from-pretrain (binary_head ∘ forecast_head)
        pos_class_weight : positive-class weight for BCEWithLogitsLoss
    """

    V: int
    hid_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    attention_dropout: float = 0.2
    pretrain: bool = False
    finetune: bool = False
    pos_class_weight: float = 1.0
    residual: str = "average"  # "average": (x+h)/2, "layernorm": LayerNorm(x+h)


# =============================================================================
# CVE — Continuous Value Embedding (byte-for-byte from upstream)
# =============================================================================


class CVE(nn.Module):
    """Continuous Value Embedding from STraTS paper.

    A scalar value is mapped through a tanh-MLP into the hidden-dim space.
    """

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
        # x: bsz, max_len
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]  # bsz,max_len,int_dim
        x = self.activation(x)
        x = torch.matmul(x, self.W2)  # bsz,max_len,hid_dim
        return x


# =============================================================================
# FusionAtt — additive attention pool (byte-for-byte from upstream)
# =============================================================================


class FusionAtt(nn.Module):
    """Additive attention used to pool the contextualized event sequence into
    a single per-sample vector. Mask is `obs_mask[B, max_len]`.
    """

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
        # x: bsz, max_len, hid_dim
        # mask: bsz, max_len  (1 = real event, 0 = padding)
        att = torch.matmul(x, self.W) + self.b[None, None, :]  # bsz,max_len,int_dim
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]  # bsz,max_len
        att = att + (1 - mask) * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)  # bsz,max_len
        return att


# =============================================================================
# Transformer — multi-layer self-attention (byte-for-byte from upstream)
# =============================================================================


class Transformer(nn.Module):
    """Multi-layer self-attention transformer using ParameterList instead of
    nn.MultiheadAttention. Implements the same math as the original STraTS
    repo (and the HyMaTE port). Layer normalization is REPLACED by `(x + h) / 2`
    averaging — this is the upstream design choice and we preserve it.
    """

    def __init__(
        self,
        num_layers: int,
        hid_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        residual: str = "average",
    ):
        super().__init__()
        self.N = num_layers
        self.d = hid_dim
        self.dff = self.d * 2
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.h = num_heads
        self.dk = self.d // self.h
        self.all_head_size = self.dk * self.h
        self.residual = residual

        self.Wq = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wk = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wv = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wo = nn.Parameter(self.init_proj((self.N, self.all_head_size, self.d)), requires_grad=True)
        self.W1 = nn.Parameter(self.init_proj((self.N, self.d, self.dff)), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((self.N, 1, 1, self.dff)), requires_grad=True)
        self.W2 = nn.Parameter(self.init_proj((self.N, self.dff, self.d)), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros((self.N, 1, 1, self.d)), requires_grad=True)

        if residual == "layernorm":
            self.norm_attn = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layers)])
            self.norm_ffn = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layers)])

    @staticmethod
    def init_proj(shape, gain: float = 1.0):
        x = torch.rand(shape)
        fan_in_out = shape[-1] + shape[-2]
        scale = gain * np.sqrt(6 / fan_in_out)
        x = x * 2 * scale - scale
        return x

    def forward_layers(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        start_layer: int = 0,
        stop_layer: int | None = None,
    ) -> torch.Tensor:
        """Run layers [start_layer, stop_layer) on x. Returns x after those layers.

        forward_layers(x, mask, 0, N) == forward(x, mask)  # full forward
        forward_layers(x, mask, 0, 1)  # layer 0 only
        forward_layers(x, mask, 1, 2)  # layer 1 only (continues from layer 0 output)
        """
        if stop_layer is None:
            stop_layer = self.N
        bsz, max_len, _ = x.size()
        mask_2d = mask[:, :, None] * mask[:, None, :]
        attention_mask = (1 - mask_2d)[:, None, :, :] * torch.finfo(x.dtype).min
        layer_mask = attention_mask
        for i in range(start_layer, stop_layer):
            # MHA
            q = torch.einsum("bld,hde->bhle", x, self.Wq[i])  # bsz,h,max_len,dk
            k = torch.einsum("bld,hde->bhle", x, self.Wk[i])  # bsz,h,max_len,dk
            v = torch.einsum("bld,hde->bhle", x, self.Wv[i])  # bsz,h,max_len,dk
            A = torch.einsum("bhle,bhke->bhlk", q, k)  # bsz,h,max_len,max_len
            if self.training:
                dropout_mask = (
                    torch.rand_like(A) < self.attention_dropout
                ).float() * torch.finfo(x.dtype).min
                layer_mask = attention_mask + dropout_mask
            A = A + layer_mask
            A = torch.softmax(A, dim=-1)
            v = torch.einsum("bhkl,bhle->bkhe", A, v)  # bsz,max_len,h,dk
            all_head_op = v.reshape((bsz, max_len, -1))
            all_head_op = torch.matmul(all_head_op, self.Wo[i])
            all_head_op = F.dropout(all_head_op, self.dropout, self.training)
            # Residual connection
            if self.residual == "layernorm":
                x = self.norm_attn[i](x + all_head_op)
            else:
                x = (all_head_op + x) / 2
            # FFN
            ffn_op = torch.matmul(x, self.W1[i]) + self.b1[i]
            ffn_op = F.gelu(ffn_op)
            ffn_op = torch.matmul(ffn_op, self.W2[i]) + self.b2[i]
            ffn_op = F.dropout(ffn_op, self.dropout, self.training)
            if self.residual == "layernorm":
                x = self.norm_ffn[i](x + ffn_op)
            else:
                x = (ffn_op + x) / 2
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_layers(x, mask, 0, None)


# =============================================================================
# Strats — top-level model (surgical edits per file header)
# =============================================================================


class Strats(TimeSeriesModelBase):
    """STraTS backbone (Tipirneni & Reddy, TKDD 2022) — clean port for TISAE.

    Forward signature:
        forward(values, times, varis, obs_mask, labels=None, forecast_values=None, forecast_mask=None)

    Returns the same loss/output convention as upstream: a (loss, ts_emb) tuple
    in the supervised path, or the forecast loss alone in the pretrain path.
    """

    def __init__(self, cfg: StratsConfig):
        base_cfg = TimeSeriesModelBaseConfig(
            V=cfg.V,
            hid_dim=cfg.hid_dim,
            pretrain=cfg.pretrain,
            finetune=cfg.finetune,
            pos_class_weight=cfg.pos_class_weight,
        )
        super().__init__(base_cfg)
        self.strats_cfg = cfg

        self.cve_time = CVE(cfg.hid_dim)
        self.cve_value = CVE(cfg.hid_dim)
        self.variable_emb = nn.Embedding(cfg.V, cfg.hid_dim)
        self.transformer = Transformer(
            num_layers=cfg.num_layers,
            hid_dim=cfg.hid_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            residual=cfg.residual,
        )
        self.fusion_att = FusionAtt(cfg.hid_dim)
        self.dropout_p = cfg.dropout
        self.V = cfg.V

    # ------------------------------------------------------------------
    # Helper called by the BackboneAdapter to scatter into the TISAE grid.
    # ------------------------------------------------------------------

    def get_contextual_embeddings(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-event contextual embeddings `[B, L, hid_dim]` AFTER the
        transformer but BEFORE the fusion attention pool.

        Used by :class:`tisae.models.backbone_adapter.StratsBackboneAdapter`
        to obtain the per-event hidden states that get scattered into the
        (T, V) grid TISAE expects. The signature is the unified
        STraTS/HyMaTE convention (no `demo`, per the no-demographics design decision).
        """
        time_emb = self.cve_time(times)
        value_emb = self.cve_value(values)
        vari_emb = self.variable_emb(varis)
        triplet_emb = time_emb + value_emb + vari_emb
        triplet_emb = F.dropout(triplet_emb, self.dropout_p, self.training)
        contextual_emb = self.transformer(triplet_emb, obs_mask)
        return contextual_emb

    def get_contextual_embeddings_at_layer(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        varis: torch.Tensor,
        obs_mask: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Return contextual embeddings after `layer` transformer layers (0-indexed).

        layer=0  → output after 1st transformer layer
        layer=-1 → all layers (same as get_contextual_embeddings)
        layer>=N → all layers (clamped)
        """
        triplet_emb = self.get_event_embeddings(values, times, varis)
        triplet_emb = F.dropout(triplet_emb, self.dropout_p, self.training)
        if layer == -1 or layer >= self.transformer.N:
            return self.transformer(triplet_emb, obs_mask)
        return self.transformer.forward_layers(triplet_emb, obs_mask, 0, layer + 1)

    def forward_from_layer(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        start_layer: int,
    ) -> torch.Tensor:
        """Continue the transformer from `start_layer`, then pool and predict.

        Used by predict_from_grid when extract_layer != -1: the SAE's
        reconstructed H_hat is at the extract layer, so we run it through
        the remaining layers before pooling.
        """
        x = self.transformer.forward_layers(x, obs_mask, start_layer, None)
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
        """Pre-context triplet embedding `[B, L, hid_dim]` (no transformer).

        Mirrors the same name on the binned-time backbone but takes
        STraTS-style continuous z-scored `values` and normalized `times`.
        Per the BackboneAdapter pre-mask convention, this MUST NOT multiply
        by `obs_mask`.
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

        ts_demo_emb = ts_emb

        # prediction / loss
        if self.pretrain:
            return self.forecast_final(ts_demo_emb, forecast_values, forecast_mask)

        if self.finetune:
            logits = self.binary_head(self.forecast_head(ts_demo_emb))[:, 0]
        else:
            logits = self.binary_head(ts_demo_emb)[:, 0]

        return self.binary_cls_final(logits, labels), ts_demo_emb
