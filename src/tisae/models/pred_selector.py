# tisae/models/pred_selector.py
"""
PredSelector — Learned prediction-aware island selector for frozen TISAE.

Stage B of the decoupled pipeline:
  Frozen TISAE dictionary + PredSelector → prediction-critical island subset

The PredSelector scores each (k, m) island based on:
  - Island features: mu, sigma, energy, peak, gate, Grad×Act prior
  - Variable features: decoder W[k,:] row
  - Observation profile: temporal obs coverage within island window
  - Patient query: h_pool from backbone's attention pooling

Score computation:
  e_tilda = W_e(island_features)
  q_tilda = W_q(patient_query)
  s = MLP([e_tilda, q_tilda, e_tilda ⊙ q_tilda]) → scalar

Selection via IslandSelector (STE top-k).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

from tisae.models.islands import IslandSelector


# =============================================================================
# Config
# =============================================================================

@dataclass
class PredSelectorConfig:
    d_sel: int = 64                  # projection dim
    mlp_hidden: int = 64             # MLP hidden dim
    n_global_pred: int = 64          # prediction selection budget
    n_per_channel: Optional[int] = None
    gumbel_topk_noise: float = 0.0
    separated_topk: bool = False
    lambda_logit: float = 1.0
    lambda_comp: float = 0.1
    delta_comp: float = 0.5          # hinge margin for L_comp
    lambda_prior: float = 0.0        # KL regularizer
    prior_tau: float = 1.0           # temperature for KL softmax
    use_grad_act_prior: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    patience: int = 20
    batch_size: int = 52


# =============================================================================
# PredSelector Module
# =============================================================================

class PredSelector(nn.Module):
    """Learned prediction-aware island selector.

    Scores each (k, m) island using island features + patient query,
    then selects via STE top-k.
    """

    def __init__(self, config: PredSelectorConfig,
                 n_island_features: int,   # 6 + 2*V
                 D: int,                   # backbone hidden dim
                 C_eff: int = 1):
        super().__init__()
        self.config = config
        self.n_island_features = n_island_features
        self.D = D
        self.C_eff = C_eff

        d = config.d_sel

        self.W_e = nn.Linear(n_island_features, d)
        self.W_q = nn.Linear(D, d)

        self.score_mlp = nn.Sequential(
            nn.Linear(3 * d, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, 1),
        )

        self.selector = IslandSelector(
            n_global=config.n_global_pred,
            n_per_channel=config.n_per_channel,
            gumbel_topk_noise=config.gumbel_topk_noise,
            separated_topk=config.separated_topk,
        )

        self._init_weights()

    def _init_weights(self):
        """Small init so initial scores are near zero (prior-aligned)."""
        nn.init.normal_(self.W_e.weight, std=0.01)
        nn.init.zeros_(self.W_e.bias)
        nn.init.normal_(self.W_q.weight, std=0.01)
        nn.init.zeros_(self.W_q.bias)
        nn.init.normal_(self.score_mlp[0].weight, std=0.01)
        nn.init.zeros_(self.score_mlp[0].bias)
        nn.init.normal_(self.score_mlp[2].weight, std=0.001)
        nn.init.zeros_(self.score_mlp[2].bias)

    def forward(self, e: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Compute per-island prediction relevance scores.

        Args:
            e: [B, K_eff, C_eff, M, n_island_features]  per-island embeddings
            q: [B, D]                                     patient query (h_pool)

        Returns:
            s_pred: [B, K_eff, C_eff, M]  raw scores
        """
        e_tilda = self.W_e(e)                              # [B, K_eff, C_eff, M, d_sel]
        q_tilda = self.W_q(q)[:, None, None, None, :]     # [B, 1, 1, 1, d_sel]
        q_tilda = q_tilda.expand_as(e_tilda)               # broadcast

        interaction = torch.cat([
            e_tilda, q_tilda, e_tilda * q_tilda
        ], dim=-1)                                         # [B, K_eff, C_eff, M, 3*d_sel]

        return self.score_mlp(interaction).squeeze(-1)     # [B, K_eff, C_eff, M]

    def select(self, s_pred: torch.Tensor) -> torch.Tensor:
        """Apply STE top-k selection."""
        return self.selector(s_pred)


# =============================================================================
# Feature Assembly
# =============================================================================

def assemble_island_features(
    A0_4d: torch.Tensor,
    g: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    gate: Optional[torch.Tensor],
    obs_grid: torch.Tensor,
    W_dec: torch.Tensor,
    prior_scores: Optional[torch.Tensor],
    K_eff: int,
    C_eff: int,
    include_obs_profile: bool = True,
) -> torch.Tensor:
    """
    Assemble per-island feature vectors.

    Args:
        A0_4d: [B, K_eff, T, C_eff]    encoder activations
        g: [B, K_eff, C_eff, M, T]     Gaussian envelopes
        mu: [B, K_eff, C_eff, M]       island centers
        sigma: [B, K_eff, C_eff, M]    island widths
        gate: [B, K_eff, C_eff, M] or None
        obs_grid: [B, T, V]            observation mask
        W_dec: [K, V]                  decoder weights (detached)
        prior_scores: [B, K_eff, C_eff, M] or None  precomputed Grad×Act
        K_eff: int
        C_eff: int

    Returns:
        features: [B, K_eff, C_eff, M, n_features]
            n_features = 6 + V if include_obs_profile is False, else 6 + 2*V
    """
    B, _, T, _ = A0_4d.shape
    M = mu.shape[-1]
    V = obs_grid.shape[-1]

    # 1) Scalar island features [B, K_eff, C_eff, M, 6]
    A0_perm = A0_4d.permute(0, 1, 3, 2)          # [B, K_eff, C_eff, T]
    eff_len = g.sum(dim=-1).clamp_min(1e-6)       # [B, K_eff, C_eff, M]
    energy = (A0_perm.unsqueeze(3) * g).sum(-1)   # [B, K_eff, C_eff, M]
    energy_norm = energy / eff_len.sqrt()

    mu_idx = mu.round().long().clamp(0, T - 1)
    peak = A0_perm.gather(-1, mu_idx)

    if gate is None:
        gate_feat = torch.zeros_like(mu)
    else:
        gate_feat = gate

    if prior_scores is None:
        prior_feat = torch.zeros_like(mu)
    else:
        prior_feat = prior_scores

    def _batch_norm_nonneg(x):
        bmax = x.detach().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        return x / bmax

    scalars = torch.stack([
        mu / T,
        sigma / T,
        _batch_norm_nonneg(energy_norm),
        _batch_norm_nonneg(peak),
        gate_feat,
        _batch_norm_nonneg(prior_feat),
    ], dim=-1)                            # [B, K_eff, C_eff, M, 6]

    # 2) Variable features: w_k [K_eff, C_eff, V] → [B, K_eff, C_eff, M, V]
    if C_eff > 1:
        w_k = W_dec.view(C_eff, K_eff, V).permute(1, 0, 2)
    else:
        w_k = W_dec.unsqueeze(1)
    w_k = w_k[None, :, :, None, :].expand(B, -1, -1, M, -1)

    parts = [scalars, w_k]

    # 3) Observation profile (optional)
    if include_obs_profile:
        obs_profile = torch.einsum('bkcmt,btv->bkcmv', g, obs_grid)
        obs_profile = obs_profile / eff_len.unsqueeze(-1).clamp_min(1e-6)
        parts.append(obs_profile)

    return torch.cat(parts, dim=-1)


# =============================================================================
# Loss Functions
# =============================================================================

def pred_selector_losses(
    z_keep: torch.Tensor,
    z_drop: torch.Tensor,
    z_full: torch.Tensor,
    z_teacher: torch.Tensor,
    s_pred: Optional[torch.Tensor] = None,
    s_prior: Optional[torch.Tensor] = None,
    config: Optional[PredSelectorConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute PredSelector training losses.

    Args:
        z_keep: [B, 1] logit from PredSelector's selected islands
        z_drop: [B, 1] logit from complement islands
        z_full: [B, 1] logit from ALL islands
        z_teacher: [B, 1] teacher logit (frozen backbone)
        s_pred: [B, K_eff, C_eff, M] selector scores (for KL prior)
        s_prior: [B, K_eff, C_eff, M] Grad×Act prior scores (for KL prior)
        config: PredSelectorConfig

    Returns:
        dict with L_logit, L_comp, L_prior, L_sel (total)
    """
    if config is None:
        config = PredSelectorConfig()

    L_logit = (z_keep - z_teacher).pow(2).mean()

    c = z_teacher.sign()
    m_full = c * z_full
    m_drop = c * z_drop
    L_comp = F.relu(config.delta_comp - (m_full - m_drop)).mean()

    L_prior = torch.tensor(0.0, device=z_keep.device)
    if config.lambda_prior > 0 and s_pred is not None and s_prior is not None:
        tau = config.prior_tau
        log_p = F.log_softmax(s_pred.view(s_pred.size(0), -1) / tau, dim=-1)
        q = F.softmax(s_prior.view(s_prior.size(0), -1) / tau, dim=-1)
        L_prior = F.kl_div(log_p, q, reduction='batchmean')

    L_sel = (config.lambda_logit * L_logit +
             config.lambda_comp * L_comp +
             config.lambda_prior * L_prior)

    return {
        'L_logit': L_logit,
        'L_comp': L_comp,
        'L_prior': L_prior,
        'L_sel': L_sel,
    }


# =============================================================================
# Budget Matching: effective cell coverage
# =============================================================================

@torch.no_grad()
def compute_effective_cell_budget(
    sel_pred: torch.Tensor,
    g: torch.Tensor,
    obs_grid: torch.Tensor,
    W_dec: torch.Tensor,
    K_eff: int,
    C_eff: int,
    threshold: float = 0.1,
) -> float:
    """Compute effective number of observed (t,v) cells covered by selected islands."""
    B, _, _, M = sel_pred.shape
    T = g.shape[-1]
    V = obs_grid.shape[-1]
    K = K_eff * C_eff

    mask_bkct = (sel_pred.unsqueeze(-1) * g).sum(dim=-2).clamp(0.0, 1.0)
    if C_eff > 1:
        mask_kt = mask_bkct.permute(0, 2, 1, 3).reshape(B, K, T)
    else:
        mask_kt = mask_bkct.squeeze(2)

    W_bin = (W_dec > 0).float()
    active_kt = (mask_kt > threshold).float()
    coverage = torch.einsum('bkt,kv->btv', active_kt, W_bin)
    covered = (coverage > 0).float() * obs_grid

    return covered.sum(dim=(1, 2)).mean().item()
