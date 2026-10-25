# tisae/models/tisae.py
"""
TI-SAE — Temporal Island Sparse Autoencoder

Pipeline:
    H_grid[B,T,V,D]
    → Encoder              → A0[B,K,T]
    → IslandParamNet       → (mu, sigma)[B,K_eff,C_eff,M]
    → IslandSelector       → sel[B,K_eff,C_eff,M]
    → GaussianMask         → mask[B,K_eff,T,C_eff]
    → A_hat = A0 * mask
    → Decoder              → H_hat_grid[B,T,V,D]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, Dict, Tuple, List

from tisae.models.islands import (
    IslandParamNet, IslandSelector, GaussianMask,
    island_mu_repulsion, mask_coverage_loss,
)


# =============================================================================
# Config
# =============================================================================

@dataclass
class TISAEConfig:
    """Configuration for TI-SAE."""
    # Core
    D: int                              # embedding dimension (from backbone)
    K: int = 1280                       # number of latents
    T: int = 48                         # time bins
    V: int = 135                        # variables
    M: int = 4                          # islands per latent

    # Channel organization
    C: int = 10                         # clinical channels
    restrict_to_channels: bool = False  # hard channel assignment (ablation)
    lambda_ch_purity: float = 0.1       # soft self-organizing purity

    # Encoder / decoder (W is shared via tying, both use top-k sparsity)
    encoder_sparsity_k: int = 5         # top-k per row in W
    encoder_chunk_size: int = 32        # chunk over K for memory efficiency
    W_sparsity_k: int = 3              # top-k per row in decoder's view of W
    normalize_atoms: bool = True

    # Island parameters
    min_sigma: float = 0.5
    max_sigma: float = 12.0
    island_hidden: int = 32
    island_kernel_size: int = 25
    n_sigma_iter: int = 3               # EM iterations

    # Selector
    n_global: int = 128                 # global island budget per sample
    n_per_channel: Optional[int] = None # per-channel cap (for hard-channel ablation)
    score_mode: str = "energy_gated"    # "energy_gated" or "energy" (ablation)
    separated_topk: bool = False
    gumbel_topk_noise: float = 0.05
    dead_boost_scale: float = 0.5
    dead_threshold_batches: int = 5

    # Loss weights
    lambda_recon: float = 1.0
    lambda_pred: float = 0.0           # >0 for joint training ablation
    lambda_auxk: float = 0.3
    auxk_k: int = 256
    lambda_mu_repulsion: float = 0.01
    lambda_div_W: float = 0.01
    lambda_mask_coverage: float = 0.005


# =============================================================================
# Encoder
# =============================================================================

class Encoder(nn.Module):
    """
    Encoder: H_grid[B,T,V,D] → A0[B,K,T]

    Each latent k has a sparse routing vector w_k and projection direction d_k:
      A0[b,k,t] = ReLU(Σ_v O[b,t,v] · W[k,v] · (H_grid[b,t,v,:] · d_k) + b_k)

    W is sparse (top-k per row, row-normalized, non-negative) and shared
    with the decoder via weight tying.
    """

    def __init__(self, K: int, V: int, D: int,
                 sparsity_k: int = 3,
                 chunk_size: int = 32,
                 latent_var_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.K = K
        self.V = V
        self.D = D
        self.sparsity_k = sparsity_k
        self.chunk_size = chunk_size

        self.W_raw = nn.Parameter(torch.randn(K, V) * 0.5)
        self.d_enc = nn.Parameter(torch.randn(K, D) * (1.0 / math.sqrt(D)))
        self.b_enc = nn.Parameter(torch.zeros(K))
        self.softplus = nn.Softplus()

        if latent_var_mask is not None:
            self.register_buffer('latent_var_mask', latent_var_mask)  # [K, V]
        else:
            self.latent_var_mask = None

    def get_W(self) -> torch.Tensor:
        """Returns sparse, normalized W_enc [K, V]."""
        W = self.softplus(self.W_raw)  # [K, V]
        if self.latent_var_mask is not None:
            W = W * self.latent_var_mask
        kk = min(max(1, self.sparsity_k), self.V)
        if kk < self.V:
            _, idx = torch.topk(W, kk, dim=-1)
            mask = torch.zeros_like(W).scatter_(-1, idx, 1.0)
            W = W * mask
        return W / W.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, H_grid: torch.Tensor, obs_grid: torch.Tensor) -> torch.Tensor:
        """
        H_grid: [B, T, V, D]
        obs_grid: [B, T, V]   (1 = observed, 0 = unobserved)
        Returns: A0 [B, K, T]
        """
        B, T, V, D = H_grid.shape
        W = self.get_W()  # [K, V]

        masked_H = H_grid * obs_grid.unsqueeze(-1)  # [B, T, V, D]

        chunks = []
        for k0 in range(0, self.K, self.chunk_size):
            k1 = min(k0 + self.chunk_size, self.K)
            w_chunk = W[k0:k1]          # [chunk, V]
            d_chunk = self.d_enc[k0:k1]  # [chunk, D]

            H_pooled = torch.einsum('btvd,cv->btcd', masked_H, w_chunk)
            score = torch.einsum('btcd,cd->btc', H_pooled, d_chunk)
            chunks.append(score)

        A0 = torch.cat(chunks, dim=-1)  # [B, T, K]
        A0 = A0 + self.b_enc  # [B, T, K]
        A0 = F.relu(A0)
        return A0.permute(0, 2, 1)  # [B, K, T]


# =============================================================================
# Decoder
# =============================================================================

class Decoder(nn.Module):
    """
    Decoder: A_hat[B,K,T] → H_hat_grid[B,T,V,D]

    Each latent k has a learnable atom P[k,:] and reuses the encoder's
    routing weights W (shared parameter, independent top-k sparsification):
      H_hat[b,t,v,d] = Σ_k A_hat[b,k,t] · W[k,v] · P[k,d] + bias[d]
    """

    def __init__(self, K: int, V: int, D: int,
                 W_raw: nn.Parameter,
                 sparsity_k: int = 3,
                 normalize_atoms: bool = True,
                 latent_var_mask: Optional[torch.Tensor] = None,
                 C: int = 1):
        super().__init__()
        self.K, self.V, self.D = K, V, D
        self.sparsity_k = sparsity_k
        self.normalize_atoms = normalize_atoms

        self.P = nn.Parameter(torch.randn(K, D) * (1.0 / math.sqrt(D)))

        # Shared W from encoder (not registered as decoder parameter)
        object.__setattr__(self, '_W_raw', W_raw)
        self.softplus = nn.Softplus()

        if latent_var_mask is not None:
            self.register_buffer('latent_var_mask', latent_var_mask)  # [K, V]
        else:
            self.latent_var_mask = None

        # Per-channel diversity (when restrict_to_channels is active)
        if latent_var_mask is not None and C > 1 and K % C == 0:
            self.K_per_c = K // C
        else:
            self.K_per_c = None

        self.bias = nn.Parameter(torch.zeros(D))

    def get_W(self) -> torch.Tensor:
        """Returns sparse, normalized W [K, V]."""
        W = self.softplus(self._W_raw)
        if self.latent_var_mask is not None:
            W = W * self.latent_var_mask
        kk = min(max(1, self.sparsity_k), self.V)
        if kk < self.V:
            _, idx = torch.topk(W, kk, dim=-1)
            mask = torch.zeros_like(W).scatter_(-1, idx, 1.0)
            W = W * mask
        return W / W.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def get_P(self) -> torch.Tensor:
        """Returns [K, D] atom directions."""
        P = self.P
        if self.normalize_atoms:
            P = F.normalize(P, p=2, dim=-1)
        return P

    def forward(self, A_hat: torch.Tensor) -> torch.Tensor:
        """
        A_hat: [B, K, T]
        Returns: H_hat_grid [B, T, V, D]
        """
        W = self.get_W()  # [K, V]
        P = self.get_P()  # [K, D]

        latent_contrib = torch.einsum('bkt,kd->btkd', A_hat, P)
        return torch.einsum('btkd,kv->btvd', latent_contrib, W) + self.bias

    def diversity_loss(self) -> torch.Tensor:
        """W row diversity: penalize variable selection overlap across latents."""
        W = self.get_W()  # [K, V]
        if self.K_per_c is not None:
            C = self.K // self.K_per_c
            total = 0.0
            for c in range(C):
                W_c = W[c * self.K_per_c : (c + 1) * self.K_per_c]
                W_norm = F.normalize(W_c, p=2, dim=1)
                sim = W_norm @ W_norm.T
                eye = torch.eye(self.K_per_c, device=W.device)
                total = total + ((sim - eye) ** 2).mean()
            return total / C
        else:
            W_norm = F.normalize(W, p=2, dim=1)
            sim = W_norm @ W_norm.T
            eye = torch.eye(self.K, device=W.device)
            return ((sim - eye) ** 2).mean()



# =============================================================================
# TISAE — Main Module
# =============================================================================

class TISAE(nn.Module):
    """Temporal Island Sparse Autoencoder.

    When restrict_to_channels=False (default), all K latents share one
    effective channel (C_eff=1). When True, latents are partitioned into
    C channels with K/C latents each (C_eff=C).
    """

    def __init__(self, config: TISAEConfig | Dict,
                 channel_var_mask: Optional[torch.Tensor] = None):
        """
        Args:
            config: TISAEConfig or dict
            channel_var_mask: [C, V] binary mask (for restrict_to_channels or ch_purity)
        """
        super().__init__()
        if isinstance(config, dict):
            valid_fields = {f.name for f in fields(TISAEConfig)}
            config = TISAEConfig(**{k: v for k, v in config.items() if k in valid_fields})
        self.cfg = config

        K, D, T, V = config.K, config.D, config.T, config.V

        # Channel reshaping
        if config.restrict_to_channels:
            self.C_eff = config.C
            assert K % self.C_eff == 0, f"K={K} must be divisible by C={self.C_eff}"
            self.K_eff = K // self.C_eff
        else:
            self.C_eff = 1
            self.K_eff = K

        # Build latent-variable mask for channel restriction
        latent_var_mask = None
        if config.restrict_to_channels and channel_var_mask is not None:
            C = config.C
            K_per_c = K // C
            channel_assignment = torch.arange(K) // K_per_c  # [K]
            latent_var_mask = channel_var_mask[channel_assignment]  # [K, V]

        # Encoder
        self.encoder = Encoder(
            K, V, D,
            sparsity_k=config.encoder_sparsity_k,
            chunk_size=config.encoder_chunk_size,
            latent_var_mask=latent_var_mask,
        )

        # Island parameter network
        self.island_params = IslandParamNet(
            T=T, M=config.M, hidden=config.island_hidden,
            min_sigma=config.min_sigma, max_sigma=config.max_sigma,
            kernel_size=config.island_kernel_size,
            predict_gate=(config.score_mode == "energy_gated"),
            n_sigma_iter=config.n_sigma_iter,
        )

        # Selector
        self.selector = IslandSelector(
            n_global=config.n_global,
            n_per_channel=config.n_per_channel,
            gumbel_topk_noise=config.gumbel_topk_noise,
            separated_topk=config.separated_topk,
            cross_channel_norm=(self.C_eff > 1),
        )

        # Gaussian mask builder
        self.gaussian_mask = GaussianMask(T)

        # Decoder (W shared with encoder)
        self.decoder = Decoder(
            K, V, D,
            W_raw=self.encoder.W_raw,
            sparsity_k=config.W_sparsity_k,
            normalize_atoms=config.normalize_atoms,
            latent_var_mask=latent_var_mask,
            C=self.C_eff,
        )

        # Channel purity mask
        if channel_var_mask is not None and config.lambda_ch_purity > 0:
            self.register_buffer('channel_var_mask_full', channel_var_mask)  # [C, V]
        else:
            self.channel_var_mask_full = None

        # Dead latent tracking — [K_eff, C_eff]
        self.register_buffer('batches_since_fired', torch.zeros(self.K_eff, self.C_eff))

    @staticmethod
    def per_cell_centered_nmse(H_hat_grid, H_grid, obs_grid, eps=1e-8, max_nmse=10.0):
        """Per-token centered NMSE on observed grid cells."""
        Hc = H_grid - H_grid.mean(dim=-1, keepdim=True)
        Hhatc = H_hat_grid - H_hat_grid.mean(dim=-1, keepdim=True)

        err = (Hhatc - Hc).pow(2).sum(dim=-1)
        denom = Hc.pow(2).sum(dim=-1).clamp_min(eps)
        nmse = (err / denom).clamp_max(max_nmse)

        nmse = nmse * obs_grid
        return nmse.sum() / obs_grid.sum().clamp_min(1)

    def forward(self, H_grid: torch.Tensor, obs_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        H_grid: [B, T, V, D]   — residualized grid embeddings
        obs_grid: [B, T, V]    — observation mask

        Returns dict with all outputs and losses.
        """
        cfg = self.cfg
        B, T, V, D = H_grid.shape
        K, M = cfg.K, cfg.M
        K_eff, C_eff = self.K_eff, self.C_eff
        device = H_grid.device

        # === 1) Encode ===
        A0 = self.encoder(H_grid, obs_grid)  # [B, K, T]

        # === 2) Channel reshape ===
        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T).permute(0, 2, 3, 1)  # [B, K_eff, T, C_eff]
        else:
            A0_4d = A0.unsqueeze(-1)  # [B, K, T, 1]

        # === 3) Island params ===
        mu, sigma, gate = self.island_params(A0_4d)  # [B, K_eff, C_eff, M]

        # === 4) Build Gaussians, scores, select ===
        sigma_main = sigma.detach()  # sigma does not receive gradient through reconstruction
        g = self.gaussian_mask.build_gaussians(mu, sigma_main)  # [B,K_eff,C_eff,M,T]

        A0_score = A0_4d.detach()  # scores do not backprop through encoder
        A0_perm = A0_score.permute(0, 1, 3, 2)  # [B, K_eff, C_eff, T]
        eff_len = g.sum(dim=-1).clamp_min(1e-6)  # [B, K_eff, C_eff, M]

        if cfg.score_mode == "energy_gated":
            energy = (A0_perm.unsqueeze(3) * g).sum(dim=-1)  # [B, K_eff, C_eff, M]
            scores = energy / eff_len.sqrt() * gate
        else:  # "energy"
            energy = (A0_perm.unsqueeze(3) * g).sum(dim=-1)
            scores = energy / eff_len

        sel = self.selector(
            scores,
            batches_since_fired=self.batches_since_fired if cfg.dead_boost_scale > 0 else None,
            dead_boost_scale=cfg.dead_boost_scale,
        )  # [B, K_eff, C_eff, M]

        # === 5) Mask + decode ===
        mask = self.gaussian_mask.forward_mask(g, sel)  # [B, K_eff, T, C_eff]
        A_hat_4d = A0_4d * mask  # [B, K_eff, T, C_eff]

        if C_eff > 1:
            A_hat = A_hat_4d.permute(0, 3, 1, 2).reshape(B, K, T)
        else:
            A_hat = A_hat_4d.squeeze(-1)

        # === 6) Decode ===
        H_hat_grid = self.decoder(A_hat)  # [B,T,V,D]

        # === 7) Losses ===
        L_recon = self.per_cell_centered_nmse(H_hat_grid, H_grid, obs_grid)

        # AuxK loss for dead latents
        L_auxk = torch.tensor(0.0, device=device)
        H_hat_grid_aux = None
        n_dead = 0
        if cfg.lambda_auxk > 0 and self.training:
            fired_kc = (A_hat_4d.abs().sum(dim=(0, 2)) > 1e-6)  # [K_eff, C_eff]

            with torch.no_grad():
                self.batches_since_fired += 1
                self.batches_since_fired[fired_kc] = 0

            dead_mask = self.batches_since_fired > cfg.dead_threshold_batches
            if cfg.score_mode == "energy_gated" and gate is not None:
                gate_mean = gate.detach().mean(dim=(0, 3))
                budget_paused = gate_mean < 0.1
                dead_mask = dead_mask & ~budget_paused
            n_dead = dead_mask.sum().item()

            if n_dead > 0:
                residual = (H_grid - H_hat_grid).detach()
                residual = residual * obs_grid.unsqueeze(-1)

                A0_dead_4d = A0_4d * dead_mask.unsqueeze(0).unsqueeze(2).float()

                auxk_k = min(cfg.auxk_k, n_dead)
                if auxk_k > 0:
                    mu_d, sigma_d, _ = self.island_params(A0_dead_4d)
                    g_d = self.gaussian_mask.build_gaussians(mu_d, sigma_d)

                    A0d_perm = A0_dead_4d.detach().permute(0, 1, 3, 2)
                    energy_d = (A0d_perm.unsqueeze(3) * g_d).sum(dim=-1)
                    eff_d = g_d.sum(dim=-1).clamp_min(1e-6)
                    if cfg.score_mode == "energy_gated":
                        scores_d = energy_d / eff_d.sqrt()
                    else:
                        scores_d = energy_d / eff_d

                    scores_kc = scores_d.sum(dim=3)
                    _, top_idx = scores_kc.view(B, -1).topk(auxk_k, dim=-1)
                    aux_sel_flat = torch.zeros(B, K_eff * C_eff, device=device)
                    aux_sel_flat.scatter_(-1, top_idx, 1.0)
                    aux_sel_4d = aux_sel_flat.view(B, K_eff, C_eff).unsqueeze(3).expand(B, K_eff, C_eff, M)

                    mask_d = self.gaussian_mask.forward_mask(g_d, aux_sel_4d)
                    A_aux_4d = A0_dead_4d * mask_d

                    if C_eff > 1:
                        A_aux = A_aux_4d.permute(0, 3, 1, 2).reshape(B, K, T)
                    else:
                        A_aux = A_aux_4d.squeeze(-1)

                    H_aux = self.decoder(A_aux)
                    H_hat_grid_aux = H_aux

                    residual_var = (residual.pow(2).sum() / (obs_grid.sum() * D).clamp_min(1)).clamp_min(1e-8)
                    L_auxk = ((H_aux - residual).pow(2) * obs_grid.unsqueeze(-1)).sum() / (obs_grid.sum() * D).clamp_min(1) / residual_var

        # Island regularizers
        L_repulsion = island_mu_repulsion(mu)

        # Mask coverage
        L_mask_coverage = torch.tensor(0.0, device=device)
        if cfg.lambda_mask_coverage > 0:
            g_budget = self.gaussian_mask.build_gaussians(mu.detach(), sigma)
            L_mask_coverage = mask_coverage_loss(g_budget, sel, cfg.n_global)

        # Decoder regularizers
        L_div_W = self.decoder.diversity_loss() if cfg.lambda_div_W > 0 else torch.tensor(0.0, device=device)

        # A0 contrast (diagnostics only, no lambda)
        g_contrast = g.detach()
        mask_union = g_contrast.sum(dim=3).clamp(0, 1)
        A0_abs = A0_4d.permute(0, 1, 3, 2).abs()
        inside_mean = (A0_abs * mask_union).sum(-1) / mask_union.sum(-1).clamp_min(1.0)
        antimask = 1 - mask_union
        outside_mean = (A0_abs * antimask).sum(-1) / antimask.sum(-1).clamp_min(1.0)
        contrast = inside_mean / (inside_mean + outside_mean + 1e-8)
        L_a0_contrast = (2 * (1 - contrast)).mean()

        # Self-organizing channel purity loss
        L_ch_purity = torch.tensor(0.0, device=device)
        if cfg.lambda_ch_purity > 0 and self.channel_var_mask_full is not None:
            W_pur = self.encoder.get_W()
            ch_mass = W_pur @ self.channel_var_mask_full.T  # [K, C]
            total_mass_pur = ch_mass.sum(dim=-1).clamp_min(1e-8)
            purity = ch_mass.max(dim=-1).values / total_mass_pur
            L_ch_purity = (1.0 - purity).mean()

        # Total internal loss
        loss_Z = (
            cfg.lambda_mu_repulsion * L_repulsion
            + cfg.lambda_div_W * L_div_W
            + cfg.lambda_mask_coverage * L_mask_coverage
            + cfg.lambda_ch_purity * L_ch_purity
        )

        # NOTE: L_auxk is NOT added here. The training wrapper combines it.
        loss_total = cfg.lambda_recon * L_recon + loss_Z

        # Selection diagnostics
        sel_binary = (sel > 0.5).float()
        n_selected = sel_binary.sum(dim=(1, 2, 3))  # [B]

        n_selected_per_scale = None
        if cfg.separated_topk:
            n_selected_per_scale = sel_binary.sum(dim=(0, 1, 2))  # [M]

        n_selected_per_channel = None
        if C_eff > 1:
            n_selected_per_channel = sel_binary.sum(dim=(0, 1, 3))  # [C_eff]

        # Mu diversity diagnostics
        with torch.no_grad():
            mu_var_batch = mu.var(dim=0, correction=0).mean()
            mu_std_per = mu.var(dim=0, correction=0).sqrt()
            mu_std_median = mu_std_per.median()
            mu_std_pct_low = (mu_std_per < 0.5).float().mean()
            mu_sorted = mu.sort(dim=-1).values
            mu_std_sorted_per = mu_sorted.var(dim=0, correction=0).sqrt()
            mu_std_sorted_median = mu_std_sorted_per.median()
            mu_std_sorted_pct_low = (mu_std_sorted_per < 0.5).float().mean()

        return {
            # Reconstructions
            'H_hat_grid': H_hat_grid,
            'H_hat_grid_aux': H_hat_grid_aux,
            'A0': A0,
            'A_hat': A_hat,
            'mask': mask,

            # Island params
            'mu': mu,
            'sigma': sigma,
            'sel': sel,
            'gate': gate,

            # Losses
            'loss_total': loss_total,
            'loss_Z': loss_Z,
            'L_recon': L_recon,
            'L_auxk': L_auxk,
            'L_repulsion': L_repulsion,
            'L_div_W': L_div_W,
            'L_a0_contrast': L_a0_contrast,
            'L_mask_coverage': L_mask_coverage,
            'L_ch_purity': L_ch_purity,

            # Diagnostics
            'n_dead': n_dead,
            'n_selected': n_selected,
            'n_selected_per_scale': n_selected_per_scale,
            'n_selected_per_channel': n_selected_per_channel,
            'mu_var_batch': mu_var_batch,
            'mu_std_median': mu_std_median,
            'mu_std_pct_low': mu_std_pct_low,
            'mu_std_sorted_median': mu_std_sorted_median,
            'mu_std_sorted_pct_low': mu_std_sorted_pct_low,
        }

    # === Convenience accessors ===

    def get_atoms(self) -> torch.Tensor:
        return self.decoder.get_P()

    def get_W_dec(self) -> torch.Tensor:
        return self.decoder.get_W()

    def get_W_enc(self) -> torch.Tensor:
        return self.encoder.get_W()
