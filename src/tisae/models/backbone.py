"""
Binned-Time Transformer Backbone

Transformer encoder for irregular EHR event sequences with discrete embeddings.
Uses Flash Attention (via F.scaled_dot_product_attention) to avoid materializing
the full L x L attention matrix.

Event embedding = Variable_emb + Value_emb + Time_emb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math


@dataclass
class BackboneConfig:
    V: int                          # Number of variables
    T: int                          # Number of time bins
    n_value_bins: int = 10          # Number of value quantile bins
    hid_dim: int = 256              # Hidden dimension
    num_layers: int = 5             # Number of transformer layers
    num_heads: int = 4              # Number of attention heads
    dropout: float = 0.3            # Dropout rate
    attention_dropout: float = 0.2
    num_classes: int = 1            # Binary classification
    use_flash_attention: bool = True
    pretrain: bool = False          # 2h value-bin forecast pretraining
    pos_class_weight: float = 1.0   # Positive class weight for BCE


class TransformerLayer(nn.Module):
    """Transformer layer with Flash Attention support."""
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        d = config.hid_dim
        self.num_heads = config.num_heads
        self.head_dim = d // config.num_heads
        self.use_flash_attention = config.use_flash_attention

        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * 4, d),
            nn.Dropout(config.dropout)
        )

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        self.dropout = nn.Dropout(config.dropout)
        self.attention_dropout_p = config.attention_dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]
        mask: [B, L] observation mask (1 = valid, 0 = padding)
        """
        B, L, D = x.shape

        Q = self.Wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_flash_attention:
            attn_mask = mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            dropout_p = self.attention_dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False
            )
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attn_mask == 0, torch.finfo(torch.float).min)
            attn = F.softmax(scores, dim=-1)
            if self.training:
                attn = F.dropout(attn, p=self.attention_dropout_p)
            out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.Wo(out)

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.ff(x))

        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence aggregation."""
    def __init__(self, hid_dim: int):
        super().__init__()
        self.W = nn.Linear(hid_dim, hid_dim)
        self.u = nn.Parameter(torch.randn(hid_dim) * 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D]
        mask: [B, L]
        Returns: pooled [B, D], attention weights [B, L]
        """
        att = torch.tanh(self.W(x))
        att = torch.matmul(att, self.u)
        att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)

        pooled = torch.einsum('bl,bld->bd', att, x)
        return pooled, att


class BinnedTimeTransformer(nn.Module):
    """Transformer backbone with fully discrete embeddings.

    Embedding scheme:
    - Variable: Learnable embedding [V, D]
    - Value: Learnable embedding [n_value_bins, D] (shared across variables)
    - Time: Learnable embedding [T, D] (time bin embedding)

    Event embedding = Variable_emb + Value_emb + Time_emb
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        d = config.hid_dim

        self.variable_emb = nn.Embedding(config.V, d)
        self.value_emb = nn.Embedding(config.n_value_bins, d)
        self.time_emb = nn.Embedding(config.T, d)

        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])

        self.attn_pool = AttentionPooling(d)

        self.pred_head = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, config.num_classes)
        )

        if config.pretrain:
            self.forecast_head = nn.Linear(d, config.V * config.n_value_bins)

        self.dropout = nn.Dropout(config.dropout)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.variable_emb.weight, std=0.02)
        nn.init.normal_(self.value_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb.weight, std=0.02)

    def get_event_embeddings(self, value_bins: torch.Tensor, tau: torch.Tensor,
                             varis: torch.Tensor) -> torch.Tensor:
        """Create event embeddings from discrete inputs."""
        var_emb = self.variable_emb(varis)
        val_emb = self.value_emb(value_bins)
        time_emb = self.time_emb(tau)

        return var_emb + val_emb + time_emb

    def get_embeddings(self, value_bins: torch.Tensor, tau: torch.Tensor,
                       varis: torch.Tensor, obs_mask: torch.Tensor,
                       return_layer: int = -1) -> Dict[str, torch.Tensor]:
        """Get embeddings from specified layer."""
        x = self.get_event_embeddings(value_bins, tau, varis)
        x = self.dropout(x)

        layer_outputs = [x]

        for layer in self.layers:
            x = layer(x, obs_mask)
            layer_outputs.append(x)

        if return_layer == -1:
            H = layer_outputs[-1]
        else:
            H = layer_outputs[return_layer]

        return {
            'H': H,
            'all_layers': layer_outputs
        }

    def forward_from_layer(self, x: torch.Tensor, obs_mask: torch.Tensor,
                           start_layer: int) -> torch.Tensor:
        """Continue forward from intermediate layer output through remaining
        layers, attention pool, and prediction head. Returns [B] logits."""
        for layer in self.layers[start_layer:]:
            x = layer(x, obs_mask)
        pooled, _ = self.attn_pool(x, obs_mask)
        return self.pred_head(pooled).squeeze(-1)

    def forecast_final_bins(
        self,
        pooled: torch.Tensor,
        forecast_value_bins: torch.Tensor,
        forecast_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked cross-entropy forecast loss over value bins.

        pooled            : [B, D]  attention-pooled representation
        forecast_value_bins : [B, V]  target bin index per variable (0..n_value_bins-1)
        forecast_mask     : [B, V]  1 = variable observed in forecast window
        """
        B = pooled.size(0)
        pred = self.forecast_head(pooled).view(B, self.config.V, self.config.n_value_bins)
        valid = forecast_mask.bool()
        if not valid.any():
            return torch.tensor(0.0, device=pooled.device)
        return F.cross_entropy(pred[valid], forecast_value_bins[valid].long(), reduction="mean")

    def forward(self, value_bins: torch.Tensor, tau: torch.Tensor,
                varis: torch.Tensor, obs_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                forecast_value_bins: Optional[torch.Tensor] = None,
                forecast_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Supervised path (default): returns logits + optional BCE loss.
        Pretrain path (config.pretrain=True, forecast_value_bins provided):
            returns forecast cross-entropy loss over value bins.
        """
        emb_out = self.get_embeddings(value_bins, tau, varis, obs_mask)
        H = emb_out['H']

        pooled, attention = self.attn_pool(H, obs_mask)

        outputs: Dict[str, torch.Tensor] = {
            'H': H,
            'pooled': pooled,
            'attention': attention,
            'all_layers': emb_out['all_layers'],
        }

        # Pretrain path: 2h value-bin forecast
        if self.config.pretrain and forecast_value_bins is not None:
            loss = self.forecast_final_bins(pooled, forecast_value_bins, forecast_mask)
            outputs['loss'] = loss
            return outputs

        # Supervised path
        logits = self.pred_head(pooled).squeeze(-1)
        outputs['logits'] = logits

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels,
                pos_weight=torch.tensor(self.config.pos_class_weight, device=logits.device),
            )
            outputs['loss'] = loss

        return outputs

    def get_embedding_weights(self) -> Dict[str, torch.Tensor]:
        """Return embedding weights for analysis."""
        return {
            'variable_emb': self.variable_emb.weight.detach(),
            'value_emb': self.value_emb.weight.detach(),
            'time_emb': self.time_emb.weight.detach()
        }
