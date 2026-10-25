# tisae/models/islands.py
"""
Temporal island components for TI-SAE.

Components:
  IslandParamNet   — predicts (mu, sigma, gate) for M Gaussian islands per latent
  IslandSelector   — STE top-k selection with optional per-channel cap
  GaussianMask     — constructs temporal masks from island parameters

Regularizers:
  island_mu_repulsion — push apart island centers within each latent
  mask_coverage_loss  — penalize total Gaussian mass budget
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# Helpers
# =============================================================================

def _pos(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.softplus(x) + eps



# =============================================================================
# IslandParamNet
# =============================================================================

class IslandParamNet(nn.Module):
    """
    Predict (mu, sigma) for M Gaussian islands per latent,
    plus optional gate for energy-gated scoring.

    Architecture: two 1D conv layers over time, then:
      - mu: soft-attention centers over time
      - sigma: EM-style iterative estimation from activation responsibilities
      - gate: optional per-island budget gate (sigmoid, 0-1)
    """

    def __init__(self, T: int, M: int = 4, hidden: int = 64,
                 min_sigma: float = 0.5, max_sigma: float = 10.0,
                 kernel_size: int = 48,
                 predict_gate: bool = False,
                 n_sigma_iter: int = 3):
        super().__init__()
        self.T = T
        self.M = M
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.predict_gate = predict_gate
        self.n_sigma_iter = n_sigma_iter
        ks = int(kernel_size)

        # Temporal feature extraction (operates on BKC as batch dim)
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=ks, padding='same')
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=ks, padding='same')

        # Center: soft attention logits over time → M centers
        self.to_center_logits = nn.Conv1d(hidden, M, kernel_size=1)

        if predict_gate:
            self.fc_gate = nn.Linear(hidden, 1)

    def forward(self, A0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        A0: [B, K, T, C] non-negative encoder activations

        Returns:
            mu:    [B, K, C, M]  island centers
            sigma: [B, K, C, M]  island widths
            gate:  [B, K, C, M]  learned budget gate (sigmoid, 0-1), or None
        """
        B, K, T, C = A0.shape
        device = A0.device

        # Reshape: [B,K,T,C] → [BKC, 1, T]
        x = A0.permute(0, 1, 3, 2).reshape(B * K * C, 1, T)

        # Feature extraction
        h = F.gelu(self.conv1(x))   # [BKC, H, T]
        h = F.gelu(self.conv2(h))   # [BKC, H, T]

        # === mu: soft attention over time ===
        center_logits = self.to_center_logits(h)   # [BKC, M, T]
        w = F.softmax(center_logits, dim=-1)        # [BKC, M, T]

        t_axis = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T)
        mu = (w * t_axis).sum(dim=-1)               # [BKC, M]

        # === sigma: EM-style from A0 responsibilities ===
        A0_trace = x.squeeze(1)                              # [BKC, T]
        mu_e = mu.unsqueeze(-1)                              # [BKC, M, 1]
        init_sigma = min(T / (2.0 * self.M), 3.0)
        sigma = torch.full_like(mu, init_sigma)              # [BKC, M]
        for _ in range(self.n_sigma_iter):
            sigma_e = sigma.unsqueeze(-1).clamp_min(1e-6)                   # [BKC, M, 1]
            g_em = torch.exp(-0.5 * ((t_axis - mu_e) / sigma_e) ** 2)      # [BKC, M, T]
            r = g_em / g_em.sum(dim=1, keepdim=True).clamp_min(1e-6)        # [BKC, M, T]
            weight = A0_trace.unsqueeze(1) * r                               # [BKC, M, T]
            denom = weight.sum(dim=-1).clamp_min(1e-6)                       # [BKC, M]
            sigma = torch.sqrt((weight * (t_axis - mu_e) ** 2).sum(dim=-1) / denom + 1e-6)
            sigma = sigma.clamp(self.min_sigma, self.max_sigma)

        # gate: learned budget gate
        gate = None
        if self.predict_gate:
            # Detach h and w so mcov gradient only reaches fc_gate weights, not A0/mu/sigma
            h_exp = h.detach().unsqueeze(1)    # [BKC, 1, H, T]
            w_exp = w.detach().unsqueeze(2)    # [BKC, M, 1, T]
            feat_gate = (h_exp * w_exp).sum(dim=-1)  # [BKC, M, H]
            gate = torch.sigmoid(self.fc_gate(feat_gate).squeeze(-1))  # [BKC, M]

        # Reshape to [B, K, C, M]
        mu    = mu.view(B, K, C, self.M)
        sigma = sigma.view(B, K, C, self.M)
        if gate is not None:
            gate = gate.view(B, K, C, self.M)

        return mu, sigma, gate


# =============================================================================
# IslandSelector — STE top-k
# =============================================================================

class IslandSelector(nn.Module):
    """
    Island selection with straight-through estimator (STE):

    1) Optional per-channel top-k: keep n_per_channel per (b,c) across K*M
    2) Global top-k: keep n_global per sample across K*C*M

    Receives pre-computed scores [B,K,C,M] from the caller.
    """

    def __init__(self, n_global: int = 32,
                 n_per_channel: Optional[int] = None,
                 gumbel_topk_noise: float = 0.0,
                 separated_topk: bool = False,
                 cross_channel_norm: bool = False):
        super().__init__()
        self.n_global = int(n_global)
        self.n_per_channel = n_per_channel
        self.gumbel_topk_noise = gumbel_topk_noise
        self.separated_topk = separated_topk
        self.cross_channel_norm = cross_channel_norm

        self.temperature = nn.Parameter(torch.tensor([1.0]))

    @staticmethod
    def _gumbel_noise(shape, device):
        """Standard Gumbel(0,1) noise."""
        u = torch.rand(shape, device=device).clamp(1e-10, 1 - 1e-10)
        return -torch.log(-torch.log(u))

    def forward(self, scores: torch.Tensor,
                batches_since_fired: Optional[torch.Tensor] = None,
                dead_boost_scale: float = 0.0) -> torch.Tensor:
        """
        scores: [B, K, C, M] pre-computed island scores
        batches_since_fired: [K, C] or None — dead latent tracking buffer
        dead_boost_scale: float — additive UCB bonus scale (0 = disabled)

        Returns: sel [B, K, C, M] binary selection mask (STE)
        """
        B, K, C, M = scores.shape

        # Compute score scale once
        score_scale = scores.detach().abs().mean().clamp_min(1e-6)

        # Dead latent score boost (Log-UCB, scale-relative, training only)
        if dead_boost_scale > 0 and batches_since_fired is not None and self.training:
            boost = dead_boost_scale * score_scale * torch.log1p(batches_since_fired)  # [K, C]
            scores = scores + boost.unsqueeze(0).unsqueeze(3)  # [1,K,C,1] broadcast

        # Gumbel noise for stochastic selection (training only)
        if self.gumbel_topk_noise > 0 and self.training:
            gumbel = self._gumbel_noise(scores.shape, scores.device)
            scores_noisy = scores + self.gumbel_topk_noise * score_scale * gumbel
        else:
            scores_noisy = scores

        sel = torch.ones_like(scores)

        # Per-channel top-k (over K*M per (b,c))
        if self.n_per_channel is not None:
            if self.separated_topk:
                npc_per_scale = max(1, int(self.n_per_channel) // M)
                s_mc = (scores_noisy * sel).permute(0, 3, 2, 1)  # [B, M, C, K]
                s_mc_flat = s_mc.reshape(B * M, C, K)
                npc_k = min(npc_per_scale, K)
                _, idx = s_mc_flat.topk(npc_k, dim=-1)
                chan_mask = torch.zeros_like(s_mc_flat).scatter_(-1, idx, 1.0)
                chan_mask = chan_mask.reshape(B, M, C, K).permute(0, 3, 2, 1)  # [B,K,C,M]
                sel = sel * chan_mask
            else:
                s_flat = (scores_noisy * sel).permute(0, 2, 1, 3).reshape(B, C, K * M)
                npc = min(int(self.n_per_channel), K * M)
                _, idx = s_flat.topk(npc, dim=-1)
                chan_mask = torch.zeros_like(s_flat).scatter_(-1, idx, 1.0)
                chan_mask = chan_mask.view(B, C, K, M).permute(0, 2, 1, 3)
                sel = sel * chan_mask

        # Cross-channel score normalization (before global top-k)
        if self.cross_channel_norm:
            if self.separated_topk:
                masked_scores = scores_noisy * sel
                chan_sum = masked_scores.sum(dim=1, keepdim=True)
                chan_count = (sel > 0.5).float().sum(dim=1, keepdim=True)
                chan_mean = chan_sum / chan_count.clamp_min(1.0)
                global_mean = chan_sum.sum(dim=2, keepdim=True) / chan_count.sum(dim=2, keepdim=True).clamp_min(1.0)
                scale = global_mean / chan_mean.clamp_min(1e-8)
                scale = scale.clamp(0.2, 5.0)
                scores_noisy = scores_noisy * scale
                scores = scores * scale
            else:
                masked_scores = scores_noisy * sel
                chan_sum = masked_scores.sum(dim=(1, 3), keepdim=True)
                chan_count = (sel > 0.5).float().sum(dim=(1, 3), keepdim=True)
                chan_mean = chan_sum / chan_count.clamp_min(1.0)
                global_mean = chan_sum.sum(dim=2, keepdim=True) / chan_count.sum(dim=2, keepdim=True).clamp_min(1.0)
                scale = global_mean / chan_mean.clamp_min(1e-8)
                scale = scale.clamp(0.2, 5.0)
                scores_noisy = scores_noisy * scale
                scores = scores * scale

        # Global top-k
        if self.separated_topk:
            n_per_scale_base = self.n_global // M
            n_remainder = self.n_global % M

            global_mask = torch.zeros(B, K, C, M, device=scores.device)
            for m in range(M):
                n_m = n_per_scale_base + (1 if m < n_remainder else 0)
                s_m = (scores_noisy * sel)[:, :, :, m].reshape(B, K * C)
                n_m = min(n_m, s_m.size(-1))
                _, idx = s_m.topk(n_m, dim=-1)
                mask_m = torch.zeros_like(s_m).scatter_(-1, idx, 1.0)
                global_mask[:, :, :, m] = mask_m.reshape(B, K, C)

            s_full = (scores * sel).view(B, -1)
            global_soft = torch.sigmoid(s_full * _pos(self.temperature)).view(B, K, C, M)
            sel = sel * (global_mask - global_soft.detach() + global_soft)
        else:
            s_flat = (scores_noisy * sel).view(B, -1)
            ng = min(self.n_global, s_flat.size(-1))
            _, idx = s_flat.topk(ng, dim=-1)

            global_mask = torch.zeros_like(s_flat).scatter_(-1, idx, 1.0)
            global_mask = global_mask.view(B, K, C, M)

            s_full = (scores * sel).view(B, -1)
            global_soft = torch.sigmoid(s_full * _pos(self.temperature)).view(B, K, C, M)
            sel = sel * (global_mask - global_soft.detach() + global_soft)

        return sel


# =============================================================================
# GaussianMask
# =============================================================================

class GaussianMask(nn.Module):
    """Constructs temporal masks from island parameters."""

    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.register_buffer(
            't_axis',
            torch.arange(T, dtype=torch.float32).view(1, 1, 1, 1, -1)  # [1,1,1,1,T]
        )

    def build_gaussians(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian envelopes.

        mu, sigma: [B, K, C, M]
        Returns: g [B, K, C, M, T]
        """
        mu_e    = mu.unsqueeze(-1)                        # [B,K,C,M,1]
        sigma_e = sigma.unsqueeze(-1).clamp_min(1e-6)     # [B,K,C,M,1]
        return torch.exp(-0.5 * ((self.t_axis - mu_e) / sigma_e) ** 2)

    def forward_mask(self, g: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
        """
        Build temporal gating mask from selected Gaussians.

        g:   [B, K, C, M, T]  Gaussian envelopes
        sel: [B, K, C, M]     island selection mask

        Returns: mask [B, K, T, C] in [0, 1]
        """
        mask_bkct = (sel.unsqueeze(-1) * g).sum(dim=-2).clamp(0.0, 1.0)  # [B,K,C,T]
        mask = mask_bkct.permute(0, 1, 3, 2)  # [B, K, T, C]
        return mask


# =============================================================================
# Regularizers
# =============================================================================

def island_mu_repulsion(mu: torch.Tensor, delta: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Push apart island centers within each latent to prevent collapse.

    mu: [B, K, C, M]
    """
    mu_i = mu.unsqueeze(-1)  # [B,K,C,M,1]
    mu_j = mu.unsqueeze(-2)  # [B,K,C,1,M]
    d2 = (mu_i - mu_j) ** 2
    overlap = torch.exp(-d2 / max(eps, 2.0 * delta ** 2))
    M = mu.size(-1)
    eye = torch.eye(M, device=mu.device, dtype=mu.dtype).view(1, 1, 1, M, M)
    return (overlap * (1.0 - eye)).mean()


def mask_coverage_loss(g: torch.Tensor, sel: torch.Tensor, n_global: int) -> torch.Tensor:
    """Total Gaussian mass budget of selected islands, normalized by budget.

    g: [B,K,C,M,T], sel: [B,K,C,M].
    Uses unclamped Gaussian sum so overlapping islands pay full cost.
    0.0 = no coverage, 1.0 = each budget slot covers all of T.
    """
    mass = (sel.unsqueeze(-1) * g).sum(dim=3)               # [B,K,C,T] unclamped
    coverage_per_sample = mass.mean(dim=-1).sum(dim=(1, 2))  # [B]
    return (coverage_per_sample / n_global).mean()
