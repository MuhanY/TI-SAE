"""TI-SAE variant with asymmetric temporal islands.

Tests whether symmetric Gaussian windows are restrictive.
Each island becomes a split-normal Gaussian at matched mass:

    sigma_L = sigma * (1 - a),   sigma_R = sigma * (1 + a),   a in (-a_max, a_max)
    g(t) = exp(-1/2 ((t-mu)/sigma_L)^2)   for t < mu   (left half-Gaussian)
    g(t) = exp(-1/2 ((t-mu)/sigma_R)^2)   for t >= mu  (right half-Gaussian)

a = 0 recovers the exact symmetric Gaussian (verified).
Total Gaussian mass is the same so mask_coverage_loss and lambda_mask_coverage need NO change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from tisae.models.tisae import TISAE, TISAEConfig
from tisae.models.islands import (
    IslandParamNet, GaussianMask, island_mu_repulsion, mask_coverage_loss,
)


# =============================================================================
# Asymmetric island parameter net (copy of IslandParamNet.forward + skew head)
# =============================================================================

class AsymIslandParamNet(IslandParamNet):
    """IslandParamNet + a per-island skew head `a` in (-a_max, a_max).

    forward returns a 4-tuple (mu, sigma, gate, a).
    The skew head reads detached conv features and is initialised to output ~0,
    so training starts symmetric (natural warm-up) and the mu/conv path is unchanged.
    """

    def __init__(self, *args, asym_a_max: float = 0.7,
                 fix_a: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.asym_a_max = float(asym_a_max)
        self.fix_a = fix_a   # if set, impose a CONSTANT skew (bypass the learned head)
        self.fc_asym = nn.Linear(self.conv2.out_channels, 1)
        nn.init.zeros_(self.fc_asym.bias)
        nn.init.normal_(self.fc_asym.weight, std=1e-3)  # start ~symmetric

    def forward(self, A0: torch.Tensor, return_asym: bool = False):
        # --- copy of IslandParamNet.forward (mu, sigma, gate) ---
        B, K, T, C = A0.shape
        device = A0.device

        x = A0.permute(0, 1, 3, 2).reshape(B * K * C, 1, T)
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))

        center_logits = self.to_center_logits(h)
        w = F.softmax(center_logits, dim=-1)

        t_axis = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T)
        mu = (w * t_axis).sum(dim=-1)

        A0_trace = x.squeeze(1)
        mu_e = mu.unsqueeze(-1)
        init_sigma = min(T / (2.0 * self.M), 3.0)
        sigma = torch.full_like(mu, init_sigma)
        for _ in range(self.n_sigma_iter):
            sigma_e = sigma.unsqueeze(-1).clamp_min(1e-6)
            g_em = torch.exp(-0.5 * ((t_axis - mu_e) / sigma_e) ** 2)
            r = g_em / g_em.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weight = A0_trace.unsqueeze(1) * r
            denom = weight.sum(dim=-1).clamp_min(1e-6)
            sigma = torch.sqrt((weight * (t_axis - mu_e) ** 2).sum(dim=-1) / denom + 1e-6)
            sigma = sigma.clamp(self.min_sigma, self.max_sigma)

        gate = None
        if self.predict_gate:
            h_exp = h.detach().unsqueeze(1)
            w_exp = w.detach().unsqueeze(2)
            feat_gate = (h_exp * w_exp).sum(dim=-1)
            gate = torch.sigmoid(self.fc_gate(feat_gate).squeeze(-1))

        # --- per-island skew a in (-a_max, a_max) ---
        if self.fix_a is not None:
            # imposed constant skew (fixed-shape control): bypass the learned head
            a = torch.full((B * K * C, self.M), float(self.fix_a),
                           device=device, dtype=mu.dtype)
        else:
            h_exp_a = h.detach().unsqueeze(1)          # [BKC, 1, H, T]
            w_exp_a = w.detach().unsqueeze(2)          # [BKC, M, 1, T]
            feat_asym = (h_exp_a * w_exp_a).sum(dim=-1)  # [BKC, M, H]
            a = self.asym_a_max * torch.tanh(self.fc_asym(feat_asym).squeeze(-1))  # [BKC, M]

        mu = mu.view(B, K, C, self.M)
        sigma = sigma.view(B, K, C, self.M)
        if gate is not None:
            gate = gate.view(B, K, C, self.M)
        a = a.view(B, K, C, self.M)
        self._last_a = a  # stash
        if return_asym:
            return mu, sigma, gate, a
        return mu, sigma, gate


# =============================================================================
# Asymmetric (split-normal) Gaussian mask
# =============================================================================

class AsymGaussianMask(GaussianMask):
    """GaussianMask whose build_gaussians supports a per-island skew `a`.

    a=None -> symmetric Gaussian.
    """

    def __init__(self, T: int, min_sigma: float = 0.5, max_sigma: float = 12.0):
        super().__init__(T)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self._island_net = None

    def build_gaussians(self, mu: torch.Tensor, sigma: torch.Tensor,
                        a: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Canonical PS/eval callers pass no `a`; pull the stashed skew so their
        # (frozen-SAE, single-call) build produces the ASYM envelope.
        if a is None and self._island_net is not None:
            a = getattr(self._island_net, "_last_a", None)
        mu_e = mu.unsqueeze(-1)                                # [B,K,C,M,1]
        if a is None:
            sigma_e = sigma.unsqueeze(-1).clamp_min(1e-6)
            return torch.exp(-0.5 * ((self.t_axis - mu_e) / sigma_e) ** 2)
        # split-normal: matched mass (sigma_L + sigma_R = 2 sigma)
        a_e = a.unsqueeze(-1)                                  # [B,K,C,M,1]
        sig = sigma.unsqueeze(-1)                              # [B,K,C,M,1]
        sigma_L = (sig * (1.0 - a_e)).clamp(self.min_sigma, self.max_sigma)
        sigma_R = (sig * (1.0 + a_e)).clamp(self.min_sigma, self.max_sigma)
        d = self.t_axis - mu_e                                 # [B,K,C,M,T]
        sigma_where = torch.where(d < 0, sigma_L, sigma_R).clamp_min(1e-6)
        return torch.exp(-0.5 * (d / sigma_where) ** 2)


# =============================================================================
# AsymTISAE — canonical TISAE with split-normal islands
# =============================================================================

class AsymTISAE(TISAE):
    """Canonical TISAE with asymmetric (split-normal) temporal islands.

    Inherits __init__/helpers from TISAE; swaps in the asym island net + mask;
    copies `forward` to thread the per-island skew `a`.
    With asym_a_max=0 (and thus a==0) the forward is bit-identical to canonical TISAE.
    """

    def __init__(self, config: TISAEConfig, channel_var_mask=None,
                 asym_a_max: float = 0.7, fix_a: Optional[float] = None):
        super().__init__(config, channel_var_mask)
        cfg = self.cfg
        self.asym_a_max = float(asym_a_max)
        self.fix_a = fix_a
        # swap symmetric island machinery for the asymmetric versions
        self.island_params = AsymIslandParamNet(
            T=cfg.T, M=cfg.M, hidden=cfg.island_hidden,
            min_sigma=cfg.min_sigma, max_sigma=cfg.max_sigma,
            kernel_size=cfg.island_kernel_size,
            predict_gate=(cfg.score_mode == "energy_gated"),
            n_sigma_iter=cfg.n_sigma_iter,
            asym_a_max=asym_a_max, fix_a=fix_a,
        )
        self.gaussian_mask = AsymGaussianMask(cfg.T, cfg.min_sigma, cfg.max_sigma)
        self.gaussian_mask._island_net = self.island_params  # enable stash path for PS/eval

    def forward(self, H_grid: torch.Tensor, obs_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        # === COPY of TISAE.forward ===
        cfg = self.cfg
        B, T, V, D = H_grid.shape
        K, M = cfg.K, cfg.M
        K_eff, C_eff = self.K_eff, self.C_eff
        device = H_grid.device

        # 1) Encode
        A0 = self.encoder(H_grid, obs_grid)

        # 2) Channel reshape
        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T).permute(0, 2, 3, 1)
        else:
            A0_4d = A0.unsqueeze(-1)

        # 3) Island params (+ skew a).
        mu, sigma, gate, a = self.island_params(A0_4d, return_asym=True)

        # 4) Build split-normal Gaussians, scores, select
        sigma_main = sigma.detach()          # sigma: no grad through recon (canonical)
        g = self.gaussian_mask.build_gaussians(mu, sigma_main, a)   # a NOT detached -> learns

        A0_score = A0_4d.detach()
        A0_perm = A0_score.permute(0, 1, 3, 2)
        eff_len = g.sum(dim=-1).clamp_min(1e-6)

        if cfg.score_mode == "energy_gated":
            energy = (A0_perm.unsqueeze(3) * g).sum(dim=-1)
            scores = energy / eff_len.sqrt() * gate
        else:
            energy = (A0_perm.unsqueeze(3) * g).sum(dim=-1)
            scores = energy / eff_len

        sel = self.selector(
            scores,
            batches_since_fired=self.batches_since_fired if cfg.dead_boost_scale > 0 else None,
            dead_boost_scale=cfg.dead_boost_scale,
        )

        # 5) Mask + decode
        mask = self.gaussian_mask.forward_mask(g, sel)
        A_hat_4d = A0_4d * mask
        if C_eff > 1:
            A_hat = A_hat_4d.permute(0, 3, 1, 2).reshape(B, K, T)
        else:
            A_hat = A_hat_4d.squeeze(-1)

        # 6) Decode
        H_hat_grid = self.decoder(A_hat)

        # 7) Losses
        L_recon = self.per_cell_centered_nmse(H_hat_grid, H_grid, obs_grid)

        L_auxk = torch.tensor(0.0, device=device)
        H_hat_grid_aux = None
        n_dead = 0
        if cfg.lambda_auxk > 0 and self.training:
            fired_kc = (A_hat_4d.abs().sum(dim=(0, 2)) > 1e-6)
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
                    mu_d, sigma_d, _, a_d = self.island_params(A0_dead_4d, return_asym=True)
                    g_d = self.gaussian_mask.build_gaussians(mu_d, sigma_d, a_d)

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

        L_repulsion = island_mu_repulsion(mu)

        L_mask_coverage = torch.tensor(0.0, device=device)
        if cfg.lambda_mask_coverage > 0:
            # a detached: coverage must not drive the skew (at matched mass d(mass)/d(a) ~= 0)
            g_budget = self.gaussian_mask.build_gaussians(mu.detach(), sigma, a.detach())
            L_mask_coverage = mask_coverage_loss(g_budget, sel, cfg.n_global)

        L_div_W = self.decoder.diversity_loss() if cfg.lambda_div_W > 0 else torch.tensor(0.0, device=device)

        g_contrast = g.detach()
        mask_union = g_contrast.sum(dim=3).clamp(0, 1)
        A0_abs = A0_4d.permute(0, 1, 3, 2).abs()
        inside_mean = (A0_abs * mask_union).sum(-1) / mask_union.sum(-1).clamp_min(1.0)
        antimask = 1 - mask_union
        outside_mean = (A0_abs * antimask).sum(-1) / antimask.sum(-1).clamp_min(1.0)
        contrast = inside_mean / (inside_mean + outside_mean + 1e-8)
        L_a0_contrast = (2 * (1 - contrast)).mean()

        L_ch_purity = torch.tensor(0.0, device=device)
        if cfg.lambda_ch_purity > 0 and self.channel_var_mask_full is not None:
            W_pur = self.encoder.get_W()
            ch_mass = W_pur @ self.channel_var_mask_full.T
            total_mass_pur = ch_mass.sum(dim=-1).clamp_min(1e-8)
            purity = ch_mass.max(dim=-1).values / total_mass_pur
            L_ch_purity = (1.0 - purity).mean()

        loss_Z = (
            cfg.lambda_mu_repulsion * L_repulsion
            + cfg.lambda_div_W * L_div_W
            + cfg.lambda_mask_coverage * L_mask_coverage
            + cfg.lambda_ch_purity * L_ch_purity
        )
        loss_total = cfg.lambda_recon * L_recon + loss_Z

        sel_binary = (sel > 0.5).float()
        n_selected = sel_binary.sum(dim=(1, 2, 3))
        n_selected_per_scale = sel_binary.sum(dim=(0, 1, 2)) if cfg.separated_topk else None
        n_selected_per_channel = sel_binary.sum(dim=(0, 1, 3)) if C_eff > 1 else None

        with torch.no_grad():
            mu_var_batch = mu.var(dim=0, correction=0).mean()
            mu_std_per = mu.var(dim=0, correction=0).sqrt()
            mu_std_median = mu_std_per.median()
            mu_std_pct_low = (mu_std_per < 0.5).float().mean()
            mu_sorted = mu.sort(dim=-1).values
            mu_std_sorted_per = mu_sorted.var(dim=0, correction=0).sqrt()
            mu_std_sorted_median = mu_std_sorted_per.median()
            mu_std_sorted_pct_low = (mu_std_sorted_per < 0.5).float().mean()
            a_abs_mean = a.detach().abs().mean()    # NEW diagnostic: how much skew

        return {
            'H_hat_grid': H_hat_grid,
            'H_hat_grid_aux': H_hat_grid_aux,
            'A0': A0,
            'A_hat': A_hat,
            'mask': mask,
            'g': g,          # asymmetric envelope, so downstream PS/eval reflect skew
            'mu': mu,
            'sigma': sigma,
            'sel': sel,
            'gate': gate,
            'a': a,
            'loss_total': loss_total,
            'loss_Z': loss_Z,
            'L_recon': L_recon,
            'L_auxk': L_auxk,
            'L_repulsion': L_repulsion,
            'L_div_W': L_div_W,
            'L_a0_contrast': L_a0_contrast,
            'L_mask_coverage': L_mask_coverage,
            'L_ch_purity': L_ch_purity,
            'n_dead': n_dead,
            'n_selected': n_selected,
            'n_selected_per_scale': n_selected_per_scale,
            'n_selected_per_channel': n_selected_per_channel,
            'mu_var_batch': mu_var_batch,
            'mu_std_median': mu_std_median,
            'mu_std_pct_low': mu_std_pct_low,
            'mu_std_sorted_median': mu_std_sorted_median,
            'mu_std_sorted_pct_low': mu_std_sorted_pct_low,
            'a_abs_mean': a_abs_mean,
        }
