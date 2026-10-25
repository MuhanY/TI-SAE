"""Standard overcomplete SAE operating on event-level H[B, L, D].

No grid scattering, no Gaussian islands, no variable routing, no channel organization.
A standard sparse autoencoder on per-event backbone embeddings.

Pipeline:
    H[B, L, D] (from backbone, before grid scatter)
    → Encoder: W_enc[K, D]  → z[B, L, K] (sparse, per-event top-k)
    → Decoder: W_dec[K, D]  → H_hat[B, L, D]

Attribution for AOPC: per-event importance scattered back to
(T, V) grid via tau/varis from batch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SimpleSAEConfig:
    D: int
    K: int = 1280
    top_k: int = 32
    sparsity_mode: str = "topk"
    normalize_decoder: bool = True
    lambda_recon: float = 1.0
    lambda_auxk: float = 0.3
    auxk_k: int = 256
    dead_threshold_batches: int = 5


class SimpleSAE(nn.Module):
    """Standard overcomplete SAE on event embeddings H[B, L, D].

    Sparsity: keep top-k activations per event (Gao et al. 2024).
    """

    def __init__(self, config: SimpleSAEConfig):
        super().__init__()
        self.cfg = config
        K, D = config.K, config.D

        self.W_enc = nn.Parameter(torch.randn(K, D) * (1.0 / math.sqrt(D)))
        self.b_enc = nn.Parameter(torch.zeros(K))
        self.W_dec = nn.Parameter(torch.randn(K, D) * (1.0 / math.sqrt(D)))
        self.b_dec = nn.Parameter(torch.zeros(D))

        self.register_buffer('batches_since_fired', torch.zeros(K))

    def encode(self, H: torch.Tensor) -> torch.Tensor:
        """H: [B, L, D] → z_sparse: [B, L, K]"""
        if self.cfg.sparsity_mode != "topk":
            raise ValueError(f"Unknown sparsity_mode: {self.cfg.sparsity_mode}")
        z = F.relu(H @ self.W_enc.T + self.b_enc)  # [B, L, K]
        top_k = min(self.cfg.top_k, self.cfg.K)
        topk_vals, topk_idx = z.topk(top_k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, L, K] → H_hat: [B, L, D]"""
        W_dec = self.W_dec
        if self.cfg.normalize_decoder:
            W_dec = F.normalize(W_dec, p=2, dim=-1)
        return z @ W_dec + self.b_dec

    def forward(self, H: torch.Tensor, obs_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        H: [B, L, D] — per-event hidden states
        obs_mask: [B, L] — 1 for real events, 0 for padding

        Returns dict with reconstructions, losses, diagnostics.
        """
        cfg = self.cfg
        B, L, D = H.shape
        device = H.device

        z = self.encode(H)  # [B, L, K]
        H_hat = self.decode(z)  # [B, L, D]

        # Per-event MSE, masked by obs_mask
        err = (H_hat - H).pow(2).sum(dim=-1)  # [B, L]
        var = (H - H.mean(dim=-1, keepdim=True)).pow(2).sum(dim=-1).clamp_min(1e-8)
        nmse = (err / var).clamp_max(10.0) * obs_mask
        L_recon = nmse.sum() / obs_mask.sum().clamp_min(1)

        # Dead latent tracking + AuxK
        L_auxk = torch.tensor(0.0, device=device)
        n_dead = 0
        if cfg.lambda_auxk > 0 and self.training:
            z_obs = z * obs_mask.unsqueeze(-1)  # [B, L, K]
            fired = (z_obs.abs().sum(dim=(0, 1)) > 1e-6)  # [K]
            with torch.no_grad():
                self.batches_since_fired += 1
                self.batches_since_fired[fired] = 0

            dead_mask = self.batches_since_fired > cfg.dead_threshold_batches  # [K]
            n_dead = dead_mask.sum().item()

            if n_dead > 0:
                residual = (H - H_hat).detach() * obs_mask.unsqueeze(-1)
                z_full = F.relu(H @ self.W_enc.T + self.b_enc)
                z_dead = z_full * dead_mask.unsqueeze(0).unsqueeze(0).float()
                auxk_k = min(cfg.auxk_k, n_dead)
                if auxk_k > 0:
                    dead_scores = z_dead.sum(dim=1).mean(dim=0)  # [K]
                    _, top_dead_idx = dead_scores.topk(auxk_k)
                    aux_mask = torch.zeros(cfg.K, device=device)
                    aux_mask[top_dead_idx] = 1.0
                    z_aux = z_dead * aux_mask.unsqueeze(0).unsqueeze(0)
                    H_aux = self.decode(z_aux)
                    res_var = (residual.pow(2).sum() / (obs_mask.sum() * D).clamp_min(1)).clamp_min(1e-8)
                    L_auxk = ((H_aux - residual).pow(2) * obs_mask.unsqueeze(-1)).sum() / (obs_mask.sum() * D).clamp_min(1) / res_var

        loss_total = cfg.lambda_recon * L_recon
        n_active = (z.abs().sum(dim=(0, 1)) > 1e-6).sum().item()
        n_active_per_event = (z.abs() > 1e-6).float().sum(dim=-1)  # [B, L]
        n_active_per_event = (n_active_per_event * obs_mask).sum() \
            / obs_mask.sum().clamp_min(1)

        return {
            'H_hat': H_hat,
            'z': z,
            'loss_total': loss_total,
            'L_recon': L_recon,
            'L_auxk': L_auxk,
            'n_dead': n_dead,
            'n_active': n_active,
            'n_active_per_event': n_active_per_event,
        }

    def compute_attribution(self, H: torch.Tensor, obs_mask: torch.Tensor,
                            tau: torch.Tensor, varis: torch.Tensor,
                            T: int, V: int) -> torch.Tensor:
        """Compute per-(T,V) cell attribution by scattering event importance.

        Returns: attr [B, T, V] attribution map for AOPC evaluation.
        """
        z = self.encode(H)  # [B, L, K]
        importance = z.abs().sum(dim=-1)  # [B, L]
        importance = importance * obs_mask

        B, L = importance.shape
        attr = torch.zeros(B, T, V, device=H.device)
        for b in range(B):
            valid = obs_mask[b].bool()
            t_idx = tau[b, valid].long().clamp(0, T - 1)
            v_idx = varis[b, valid].long().clamp(0, V - 1)
            imp = importance[b, valid]
            attr[b].index_put_((t_idx, v_idx), imp, accumulate=True)

        return attr
