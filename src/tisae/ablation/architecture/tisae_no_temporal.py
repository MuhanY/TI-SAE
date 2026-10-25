"""TI-SAE variant with no temporal decomposition (flat mask ablation).

Reuses the canonical TISAE but replaces Gaussian island masking with a flat (uniform) mask.
Selected latents activate at all time steps.

Key differences from canonical TISAE:
  - No IslandParamNet, no GaussianMask
  - Selection: top-n_global latents by mean|A0|
  - Mask = 1 for selected latents at all T, 0 otherwise
  - No mu_repulsion or mask_coverage losses
  - M is forced to 1 (multi-island concept is meaningless)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, fields
from typing import Optional, Dict

from tisae.models.tisae import TISAEConfig, Encoder, Decoder


class _FlatGaussianMask:
    """Shim that mimics GaussianMask but returns ones (flat mask).

    Allows standard eval attribution functions to work on TISAENoTemporal without modification.
    """

    def __init__(self, T: int):
        self.T = T

    def build_gaussians(self, mu, sigma):
        B, K, C, M = mu.shape
        return torch.ones(B, K, C, M, self.T, device=mu.device)

    def forward_mask(self, g, sel):
        B, K, C, M, T = g.shape
        mask_bkct = sel.unsqueeze(-1).expand(B, K, C, M, T).sum(dim=3).clamp(0, 1)
        return mask_bkct.permute(0, 1, 3, 2)


class _FlatIslandParams(nn.Module):
    """Shim that mimics IslandParamNet but returns flat params (mu=T/2, sigma=T)."""

    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def forward(self, A0_4d):
        B, K, T_len, C = A0_4d.shape
        device = A0_4d.device
        mu = torch.full((B, K, C, 1), self.T / 2.0, device=device)
        sigma = torch.full((B, K, C, 1), float(self.T), device=device)
        gate = torch.ones(B, K, C, 1, device=device)
        return mu, sigma, gate


class TISAENoTemporal(nn.Module):
    """TISAE with flat temporal mask."""

    def __init__(self, config: TISAEConfig,
                 channel_var_mask: Optional[torch.Tensor] = None):
        super().__init__()
        if isinstance(config, dict):
            valid_fields = {f.name for f in fields(TISAEConfig)}
            config = TISAEConfig(**{k: v for k, v in config.items() if k in valid_fields})

        assert config.M == 1, "no_temporal requires M=1"
        self.cfg = config

        K, D, T, V = config.K, config.D, config.T, config.V

        if config.restrict_to_channels:
            self.C_eff = config.C
            assert K % self.C_eff == 0
            self.K_eff = K // self.C_eff
        else:
            self.C_eff = 1
            self.K_eff = K

        latent_var_mask = None
        if config.restrict_to_channels and channel_var_mask is not None:
            C = config.C
            K_per_c = K // C
            channel_assignment = torch.arange(K) // K_per_c
            latent_var_mask = channel_var_mask[channel_assignment]

        self.encoder = Encoder(
            K, V, D,
            sparsity_k=config.encoder_sparsity_k,
            chunk_size=config.encoder_chunk_size,
            latent_var_mask=latent_var_mask,
        )

        self.decoder = Decoder(
            K, V, D,
            W_raw=self.encoder.W_raw,
            sparsity_k=config.W_sparsity_k,
            normalize_atoms=config.normalize_atoms,
            latent_var_mask=latent_var_mask,
            C=self.C_eff,
        )

        # TISAE-compatible shims for eval attribution functions
        self.island_params = _FlatIslandParams(T)
        self.gaussian_mask = _FlatGaussianMask(T)

        if channel_var_mask is not None and config.lambda_ch_purity > 0:
            self.register_buffer('channel_var_mask_full', channel_var_mask)
        else:
            self.channel_var_mask_full = None

        self.register_buffer('batches_since_fired', torch.zeros(self.K_eff, self.C_eff))

    def forward(self, H_grid: torch.Tensor, obs_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        B, T, V, D = H_grid.shape
        K = cfg.K
        K_eff, C_eff = self.K_eff, self.C_eff
        device = H_grid.device

        # 1) Encode
        A0 = self.encoder(H_grid, obs_grid)  # [B, K, T]

        # 2) Channel reshape
        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T).permute(0, 2, 3, 1)
        else:
            A0_4d = A0.unsqueeze(-1)  # [B, K, T, 1]

        # 3) Selection: top-n_global latents by mean|A0| (no islands)
        with torch.no_grad():
            latent_scores = A0_4d.detach().abs().mean(dim=2)  # [B, K_eff, C_eff]
            scores_flat = latent_scores.view(B, -1)  # [B, K_eff * C_eff]
            n_sel = min(cfg.n_global, scores_flat.shape[1])
            _, top_idx = scores_flat.topk(n_sel, dim=-1)
            sel_flat = torch.zeros_like(scores_flat)
            sel_flat.scatter_(-1, top_idx, 1.0)
            sel = sel_flat.view(B, K_eff, C_eff)  # [B, K_eff, C_eff]

        # 4) Flat mask: selected latents active at all T
        mask = sel.unsqueeze(2).expand(B, K_eff, T, C_eff)  # [B, K_eff, T, C_eff]
        A_hat_4d = A0_4d * mask

        if C_eff > 1:
            A_hat = A_hat_4d.permute(0, 3, 1, 2).reshape(B, K, T)
        else:
            A_hat = A_hat_4d.squeeze(-1)

        # 5) Decode
        H_hat_grid = self.decoder(A_hat)

        # 6) Losses
        L_recon = self.per_cell_centered_nmse(H_hat_grid, H_grid, obs_grid)

        # AuxK for dead latents (simplified: no Gaussians)
        L_auxk = torch.tensor(0.0, device=device)
        n_dead = 0
        if cfg.lambda_auxk > 0 and self.training:
            fired_kc = (A_hat_4d.abs().sum(dim=(0, 2)) > 1e-6)  # [K_eff, C_eff]
            with torch.no_grad():
                self.batches_since_fired += 1
                self.batches_since_fired[fired_kc] = 0

            dead_mask = self.batches_since_fired > cfg.dead_threshold_batches
            n_dead = dead_mask.sum().item()

            if n_dead > 0:
                residual = (H_grid - H_hat_grid).detach() * obs_grid.unsqueeze(-1)
                A0_dead_4d = A0_4d * dead_mask.unsqueeze(0).unsqueeze(2).float()
                auxk_k = min(cfg.auxk_k, n_dead)
                if auxk_k > 0:
                    dead_scores = A0_dead_4d.abs().mean(dim=2).view(B, -1)
                    _, top_dead = dead_scores.topk(auxk_k, dim=-1)
                    aux_sel = torch.zeros_like(dead_scores)
                    aux_sel.scatter_(-1, top_dead, 1.0)
                    aux_mask = aux_sel.view(B, K_eff, C_eff).unsqueeze(2).expand(B, K_eff, T, C_eff)
                    A_aux_4d = A0_dead_4d * aux_mask
                    if C_eff > 1:
                        A_aux = A_aux_4d.permute(0, 3, 1, 2).reshape(B, K, T)
                    else:
                        A_aux = A_aux_4d.squeeze(-1)
                    H_aux = self.decoder(A_aux)
                    res_var = (residual.pow(2).sum() / (obs_grid.sum() * D).clamp_min(1)).clamp_min(1e-8)
                    L_auxk = ((H_aux - residual).pow(2) * obs_grid.unsqueeze(-1)).sum() / (obs_grid.sum() * D).clamp_min(1) / res_var

        L_div_W = self.decoder.diversity_loss() if cfg.lambda_div_W > 0 else torch.tensor(0.0, device=device)

        L_ch_purity = torch.tensor(0.0, device=device)
        if cfg.lambda_ch_purity > 0 and self.channel_var_mask_full is not None:
            W_pur = self.encoder.get_W()
            ch_mass = W_pur @ self.channel_var_mask_full.T
            total_mass = ch_mass.sum(dim=-1).clamp_min(1e-8)
            purity = ch_mass.max(dim=-1).values / total_mass
            L_ch_purity = (1.0 - purity).mean()

        loss_Z = cfg.lambda_div_W * L_div_W + cfg.lambda_ch_purity * L_ch_purity
        loss_total = cfg.lambda_recon * L_recon + loss_Z

        n_selected = sel.sum(dim=(1, 2))  # [B]

        # Synthetic island params for PredSelector compatibility.
        mu_synth = torch.full((B, K_eff, C_eff, 1), T / 2.0, device=device)
        sigma_synth = torch.full((B, K_eff, C_eff, 1), float(T), device=device)
        gate_synth = torch.ones(B, K_eff, C_eff, 1, device=device)
        g_synth = torch.ones(B, K_eff, C_eff, 1, T, device=device)

        return {
            'H_hat_grid': H_hat_grid,
            'H_hat_grid_aux': None,
            'A0': A0,
            'A_hat': A_hat,
            'mask': mask,
            'mu': mu_synth,
            'sigma': sigma_synth,
            'sel': sel.unsqueeze(-1),  # [B, K_eff, C_eff, 1] for compat
            'gate': gate_synth,
            'g': g_synth,
            'loss_total': loss_total,
            'loss_Z': loss_Z,
            'L_recon': L_recon,
            'L_auxk': L_auxk,
            'L_repulsion': torch.tensor(0.0, device=device),
            'L_div_W': L_div_W,
            'L_a0_contrast': torch.tensor(0.0, device=device),
            'L_mask_coverage': torch.tensor(0.0, device=device),
            'L_ch_purity': L_ch_purity,
            'n_dead': n_dead,
            'n_selected': n_selected,
            'n_selected_per_scale': None,
            'n_selected_per_channel': None,
            'mu_var_batch': torch.tensor(0.0),
            'mu_std_median': torch.tensor(0.0),
            'mu_std_pct_low': torch.tensor(0.0),
            'mu_std_sorted_median': torch.tensor(0.0),
            'mu_std_sorted_pct_low': torch.tensor(0.0),
        }

    @staticmethod
    def per_cell_centered_nmse(H_hat_grid, H_grid, obs_grid, eps=1e-8, max_nmse=10.0):
        Hc = H_grid - H_grid.mean(dim=-1, keepdim=True)
        Hhatc = H_hat_grid - H_hat_grid.mean(dim=-1, keepdim=True)
        err = (Hhatc - Hc).pow(2).sum(dim=-1)
        denom = Hc.pow(2).sum(dim=-1).clamp_min(eps)
        nmse = (err / denom).clamp_max(max_nmse)
        nmse = nmse * obs_grid
        return nmse.sum() / obs_grid.sum().clamp_min(1)

    def get_atoms(self):
        return self.decoder.get_P()

    def get_W_dec(self):
        return self.decoder.get_W()

    def get_W_enc(self):
        return self.encoder.get_W()
