"""
BackboneAdapter — uniform interface for any sequence-format ICU backbone
that produces per-event hidden states `[B, L, D]` keyed by `(tau_bin, var_id)`.

Why this exists
---------------
This module provides a uniform adapter so that TISAE training, PredSelector,
and attribution methods can consume any backbone through one stable interface.
The adapter scatters per-event hidden states into the canonical
`H_grid[B, T, V, D]` representation TISAE expects.

Identity contract
-----------------
The :class:`TISAEBackboneAdapter` is a no-op wrapper around
:class:`tisae.models.backbone.BinnedTimeTransformer`. All numerical results
remain bit-equivalent to direct backbone usage.

Methods
-------
extract_hidden(batch) → (H_grid[B,T,V,D], obs_grid[B,T,V])
    Frozen forward pass that scatters per-event embeddings into the grid.

forward(batch, *, return_pooled=False, return_attention=False) → dict
    Full forward pass returning logits + optional pooled + attention.

predict_from_grid(H_hat_grid, batch, obs_grid) → logits[B]
    Re-runs the (frozen) pooling + prediction head on a SAE-reconstructed
    grid. This is "live" pooling — the
    backbone parameters are still frozen via `requires_grad=False`, but
    we re-execute the forward operations so gradients flow through the
    grid for prediction-fidelity loss and IG-style attribution.

get_attention_grid(batch, T, V) → [B, T, V]
    Convenience: pull per-event attention weights and scatter into the
    same grid layout used elsewhere.

forward_from_event_embeddings(emb, batch) → logits[B]
    Captum-friendly path: take a fully-formed `[B, L, D]` event-embedding
    tensor (often with `requires_grad=True`) and run only the post-
    embedding part of the backbone. Used by IG / GradientSHAP / DeepLIFT.

get_event_embeddings(batch) → [B, L, D]
    Pre-context (layer-0) event embeddings, used as the input to
    `forward_from_event_embeddings` and as the seed for
    Captum / DynaMask perturbation methods.

Checkpoint format
-----------------
Every backbone checkpoint stores a uniform metadata bundle:

    {
        'backbone_type': 'tisae' | 'strats' | 'hymate',
        'backbone_config': dict,
        'model_state_dict': dict,
        'unified_dataset_metadata': {'V', 'T', 'var_to_ind_hash', ...},
        'val_auroc': float,
        'training_args': dict,
        'is_finetune': bool,
    }

"""

from __future__ import annotations

import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tisae.models.backbone import BackboneConfig, BinnedTimeTransformer
from tisae.utils.grid_ops import build_H_grid, grid_to_H


# =============================================================================
# Abstract base
# =============================================================================


class BackboneAdapter(nn.Module):
    """Abstract base class. Every concrete backbone provides the methods
    documented in this module's header. Subclasses MUST set:

        self.name : str    # 'tisae' | 'strats' | 'hymate'
        self.D : int       # hidden dim
        self.T : int       # number of time bins
        self.V : int       # number of variables
        self.n_value_bins : int  # discrete-bin backbones only; default 0

    """

    name: str = "abstract"

    def __init__(self, T: int, V: int, D: int, n_value_bins: int = 0):
        super().__init__()
        self.T = int(T)
        self.V = int(V)
        self.D = int(D)
        self.n_value_bins = int(n_value_bins)
        self.is_finetune: bool = False
        self.var_to_ind_hash: Optional[str] = None
        self._frozen: bool = False
        # H_grid_mean is registered lazily so that subclasses do not need
        # to know about it at construction time. Use
        # `set_residualization_mean(...)` to install one.
        self.register_buffer("H_grid_mean", None, persistent=False)
        # median_bins[V]: per-variable median value-bin index, used by
        # `get_baseline_event_embeddings(mode='median')` on binned backbones.
        # CVE backbones can ignore this. Use `set_median_bins(...)` to install.
        self.register_buffer("median_bins", None, persistent=False)

    # ------------------------------------------------------------------
    # Frozen-mode contract
    # ------------------------------------------------------------------

    def freeze(self) -> "BackboneAdapter":
        """Set requires_grad=False on every parameter and lock the
        underlying module in eval mode. After freeze(), `self.train()`
        will keep the underlying module in eval mode regardless of the
        argument passed to it (see `train()` override).
        """
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True
        self.eval()
        return self

    def train(self, mode: bool = True) -> "BackboneAdapter":  # type: ignore[override]
        """nn.Module.train() override that respects the frozen contract.
        """
        super().train(mode)
        if self._frozen and hasattr(self, "module"):
            self.module.eval()
        return self

    # ------------------------------------------------------------------
    # Residualization
    # ------------------------------------------------------------------

    def set_residualization_mean(self, mean: Optional[torch.Tensor]) -> None:
        """Install the per-variable `H_grid_mean[V, D]` used by
        `predict_from_grid` to undo the population-mean subtraction
        applied to the SAE input. Pass `None` to disable.
        """
        if mean is None:
            self.H_grid_mean = None  # type: ignore[assignment]
            return
        if mean.dim() != 2:
            raise ValueError(
                f"H_grid_mean must be [V, D]; got shape {tuple(mean.shape)}"
            )
        # Move onto the same device as the rest of the adapter parameters
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = mean.device
        self.H_grid_mean = mean.to(target_device).detach()  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_hidden(
        self,
        batch: Dict[str, torch.Tensor],
        return_event_h: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Return `(H_grid[B,T,V,D], obs_grid[B,T,V])` from a unified-collate
        batch. If `return_event_h=True`, additionally return the per-event
        `H[B,L,D]` for L_recon_H monitoring (3-tuple).
        """
        raise NotImplementedError

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_pooled: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass returning at minimum {'logits': [B]}.
        Optionally returns 'pooled' and 'attention'.

        """
        raise NotImplementedError

    def predict_from_grid(
        self,
        H_hat_grid: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        obs_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Re-run the head on a SAE-reconstructed grid.
        Adds `self.H_grid_mean[varis]` back in event space if a
        residualization mean was installed via
        `set_residualization_mean`.

        Returns [B] logits with gradient flowing through `H_hat_grid`.
        """
        raise NotImplementedError

    def pool_events(
        self,
        H_event: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool event-level `[B, L, D]` embeddings to `(pooled[B, D],
        attention[B, L])`. Used by attribution methods that have an H
        already in hand (e.g. island-prediction-gradient extraction).
        """
        raise NotImplementedError

    def predict_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """Run the prediction head on a `[B, D]` pooled embedding.
        Returns `[B]` logits.
        """
        raise NotImplementedError

    def forward_from_event_embeddings(
        self,
        emb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward from a `[B, L, D]` event-embedding tensor (typically
        `requires_grad=True`) directly to logits, bypassing the embedding
        layer. Used by Captum gradient methods. Returns [B] logits.
        """
        raise NotImplementedError

    def get_event_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return pre-context event embeddings `[B, L, D]`. Used as the
        starting point for Captum / DynaMask perturbation methods.

        """
        raise NotImplementedError

    def get_baseline_event_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "median",
    ) -> torch.Tensor:
        """Return `[B, L, D]` baseline embeddings for captum (IG, DeepLIFT,
        GradientSHAP).

        For binned backbones:
            mode='median' → replaces `value_bins` with the per-variable
                            median bin index (precomputed at adapter
                            construction time, exposed via
                            `self.median_bins`).
            mode='zero'   → replaces `value_bins` with bin index 0.

        For CVE backbones (STraTS/HyMaTE): both modes return the embedding of
        a `values_zscore = 0` baseline (since z-score 0 == population mean).
        The two modes are equivalent for CVE backbones because z-scoring
        already centers the data.

        Subclasses MUST override this method.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Extract layer
    # ------------------------------------------------------------------

    def set_extract_layer(self, layer: int) -> None:
        """Set the transformer layer to extract for the TISAE grid.

        layer=-1 (default): use the last layer (current behavior).
        layer=0:  use the output of the first transformer layer / mamba block.
        layer=-2: second-to-last layer (Python-style negative indexing).

        Negative indices (other than -1) are resolved to positive at set
        time, so the rest of the code only sees -1 or non-negative.
        """
        if layer < -1:
            m = self.module
            n = (m.transformer.N if hasattr(m, "transformer")
                 else m.mamba.blocks if hasattr(m, "mamba")
                 else len(m.layers))
            layer = n + layer
            if layer < 0:
                raise ValueError(f"extract_layer resolves to {layer} (backbone has {n} layers)")
        self._extract_layer = layer

    @property
    def extract_layer(self) -> int:
        return getattr(self, "_extract_layer", -1)

    def set_median_bins(self, median_bins: Optional[torch.Tensor]) -> None:
        """Install the per-variable `[V]` median value-bin tensor used by
        `get_baseline_event_embeddings(mode='median')`. CVE adapters can
        ignore this; binned adapters need it for the median-baseline path.
        Pass `None` to disable.
        """
        if median_bins is None:
            self.median_bins = None  # type: ignore[assignment]
            return
        if median_bins.dim() != 1:
            raise ValueError(
                f"median_bins must be [V]; got shape {tuple(median_bins.shape)}"
            )
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = median_bins.device
        self.median_bins = median_bins.to(target_device).long()  # type: ignore[assignment]

    @torch.no_grad()
    def get_attention_grid(
        self,
        batch: Dict[str, torch.Tensor],
        T: int,
        V: int,
    ) -> torch.Tensor:
        """Return the per-(t,v) cell attention attribution `[B, T, V]`."""
        raise NotImplementedError

    def replace_event_value(
        self,
        batch: Dict[str, torch.Tensor],
        replace_mask: torch.Tensor,
        strategy: str,
        marginal_dist: Optional[torch.Tensor] = None,
        population_values_zscore: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return a new batch dict in which events selected by
        `replace_mask[B, L]` (1 = replace, 0 = keep) have had their values
        modified per `strategy`:

            'removal'      → set obs_mask=0 at those positions
            'marginal'     → sample from per-variable population
            'conditional'  → use the patient's nearest same-variable observed
                             value, falling back to marginal

        Concrete adapters that operate on binned values use `marginal_dist`
        (`[V, n_bins]` per-variable bin probabilities). Adapters that
        operate on continuous z-scored values use
        `population_values_zscore` (per-variable training pool of
        z-scored floats — exposed by the unified dataset).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Migration helpers
    # ------------------------------------------------------------------

    @property
    def has_continuous_values(self) -> bool:
        """True if the backbone consumes continuous `values_zscore` as input
        (CVE backbones: STraTS, HyMaTE). False for binned backbones.

        Used by `attribution_methods.py` to dispatch between embedding-level
        and value-level adaptations of gradient/perturbation methods.
        """
        return False

    def predict_with_obs_mask(
        self,
        batch: Dict[str, torch.Tensor],
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass with a modified obs_mask, returning [B] logits.

        Used by perturbation-based attribution methods that zero out events
        to test prediction sensitivity.

        """
        new_batch = {**batch, "obs_mask": obs_mask}
        return self.forward(new_batch)["logits"]

    def forward_with_value_mask(
        self,
        batch: Dict[str, torch.Tensor],
        soft_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass with `values_zscore * soft_mask`, returning
        [B] logits. CVE adapters (STraTS, HyMaTE) override this; the base
        class default raises NotImplementedError.

        `soft_mask=1` → original value; `soft_mask=0` → population mean
        (z-score 0). Used by value-level dynamask. The intervention is
        "smoothly fade the measured value toward the population mean while
        keeping variable identity, time, and obs_mask unchanged".
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support value-level masking; "
            "this method is only implemented on CVE backbones (STraTS, HyMaTE)."
        )


# =============================================================================
# TISAEBackboneAdapter: no-op wrapper around BinnedTimeTransformer
# =============================================================================


class TISAEBackboneAdapter(BackboneAdapter):
    """Identity-wrapper adapter for the existing binned-time transformer.
    """

    name = "tisae"

    def __init__(
        self,
        backbone: BinnedTimeTransformer,
        T: int,
        V: int,
        var_to_ind_hash: Optional[str] = None,
        is_finetune: bool = False,
    ):
        super().__init__(
            T=T,
            V=V,
            D=backbone.config.hid_dim,
            n_value_bins=int(backbone.config.n_value_bins),
        )
        self.module = backbone
        self.var_to_ind_hash = var_to_ind_hash
        self.is_finetune = is_finetune

    # ------------------------------------------------------------------
    # Forward / extract / predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_hidden(
        self,
        batch: Dict[str, torch.Tensor],
        return_event_h: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        vb = batch["value_bins"]
        tau = batch["tau"]
        varis = batch["varis"]
        obs_mask = batch["obs_mask"]

        emb_out = self.module.get_embeddings(vb, tau, varis, obs_mask)
        if self.extract_layer != -1:
            # extract_layer=0 → after layer 0 → all_layers[1]
            H = emb_out["all_layers"][self.extract_layer + 1]
        else:
            H = emb_out["H"]  # [B, L, D]
        H_grid, obs_grid = build_H_grid(H, tau, varis, obs_mask, self.T, self.V)
        if return_event_h:
            return H_grid, obs_grid, H
        return H_grid, obs_grid

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_pooled: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:

        vb = batch["value_bins"]
        tau = batch["tau"]
        varis = batch["varis"]
        obs_mask = batch["obs_mask"]
        out = self.module(vb, tau, varis, obs_mask)
        result: Dict[str, torch.Tensor] = {
            "logits": out["logits"],
            "H": out["H"],
        }
        if return_pooled:
            result["pooled"] = out["pooled"]
        if return_attention:
            result["attention"] = out["attention"]
        return result

    def predict_from_grid(
        self,
        H_hat_grid: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        obs_grid: torch.Tensor,
    ) -> torch.Tensor:

        tau = batch["tau"]
        varis = batch["varis"]
        obs_mask = batch["obs_mask"]
        H_hat_event = grid_to_H(H_hat_grid, tau, varis)  # [B, L, D]
        if self.H_grid_mean is not None:
            H_hat_event = H_hat_event + self.H_grid_mean[varis]

        if self.extract_layer != -1:
            # Run through remaining layers before pooling
            return self.module.forward_from_layer(
                H_hat_event, obs_mask, start_layer=self.extract_layer + 1
            )

        pooled, _ = self.module.attn_pool(H_hat_event, obs_mask)
        return self.module.pred_head(pooled).squeeze(-1)

    def pool_events(
        self,
        H_event: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the attention pool on a `[B, L, D]` event tensor.
        Returns (pooled[B, D], attention[B, L]).
        """
        return self.module.attn_pool(H_event, obs_mask)

    def predict_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.module.pred_head(pooled).squeeze(-1)

    def forward_from_event_embeddings(
        self,
        emb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Captum-friendly forward from a `[B, L, D]` event-embedding tensor.
        Mirrors `_forward_from_embeddings` in attribution_methods.py.
        """
        obs_mask = batch["obs_mask"]
        x = self.module.dropout(emb)
        for layer in self.module.layers:
            x = layer(x, obs_mask)
        pooled, _ = self.module.attn_pool(x, obs_mask)
        return self.module.pred_head(pooled).squeeze(-1)

    def get_event_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.module.get_event_embeddings(
            batch["value_bins"], batch["tau"], batch["varis"]
        )

    def get_baseline_event_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "median",
    ) -> torch.Tensor:
        """Per-variable median (or zero) baseline embedding for binned
        backbones. mode='median' replaces value_bins with `self.median_bins`
        per variable; mode='zero' replaces value_bins with 0.
        """
        varis = batch["varis"]
        if mode == "median":
            if self.median_bins is None:
                raise ValueError(
                    "median_bins not installed on TISAEBackboneAdapter; call "
                    "`adapter.set_median_bins(median_bins[V])` before requesting "
                    "the median baseline."
                )
            ref_vb = self.median_bins.to(varis.device)[varis]
        elif mode == "zero":
            ref_vb = torch.zeros_like(batch["value_bins"])
        else:
            raise ValueError(f"Unknown baseline mode: {mode!r}; expected 'median' or 'zero'")
        return self.get_event_embeddings({**batch, "value_bins": ref_vb})

    @torch.no_grad()
    def get_attention_grid(
        self,
        batch: Dict[str, torch.Tensor],
        T: int,
        V: int,
    ) -> torch.Tensor:
        out = self.module(
            batch["value_bins"], batch["tau"], batch["varis"], batch["obs_mask"]
        )
        attn = out["attention"]  # [B, L]

        tau = batch["tau"].long()
        varis = batch["varis"].long()
        obs_mask = batch["obs_mask"]

        # Zero attention at padded positions and route padded events to
        # cell (0,0) so the scatter does not depend on garbage tau/varis
        # values left in the padding.
        valid = obs_mask > 0.5
        attn = attn * obs_mask
        safe_tau = torch.where(valid, tau, torch.zeros_like(tau))
        safe_varis = torch.where(valid, varis, torch.zeros_like(varis))
        cell_idx = safe_tau * V + safe_varis  # [B, L]

        B, L = attn.shape
        attr_tv = torch.zeros(B, T * V, device=attn.device, dtype=attn.dtype)
        attr_tv.scatter_add_(1, cell_idx, attn)
        return attr_tv.view(B, T, V)

    # ------------------------------------------------------------------
    # Value replacement
    # ------------------------------------------------------------------

    def _marginal_dist_on(
        self,
        marginal_dist: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:

        if marginal_dist is None:
            return None
        if getattr(self, "_marginal_dist_cache", None) is marginal_dist:
            return self._marginal_dist_cache_dev  # type: ignore[attr-defined]
        cached = marginal_dist.to(device)
        self._marginal_dist_cache = marginal_dist
        self._marginal_dist_cache_dev = cached
        return cached

    def _random_bins(self, n: int, device: torch.device) -> torch.Tensor:

        n_bins = max(self.n_value_bins, 1)
        return torch.randint(0, n_bins, (n,), device=device)

    def _median_bins_for(self, marginal_dist: torch.Tensor) -> torch.Tensor:

        cache_key = (id(marginal_dist), marginal_dist.device)
        cached_key = getattr(self, "_median_bins_cache_key", None)
        if cached_key == cache_key:
            return self._median_bins_cache_value  # type: ignore[attr-defined]
        cumsum = marginal_dist.cumsum(dim=-1)
        medians = (cumsum >= 0.5).float().argmax(dim=-1).long()
        self._median_bins_cache_key = cache_key
        self._median_bins_cache_value = medians
        return medians

    def replace_event_value(
        self,
        batch: Dict[str, torch.Tensor],
        replace_mask: torch.Tensor,
        strategy: str,
        marginal_dist: Optional[torch.Tensor] = None,
        population_values_zscore: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, torch.Tensor]:

        new_batch = {k: v for k, v in batch.items()}
        replace_mask = replace_mask.bool()
        device = batch["value_bins"].device

        if strategy == "removal":
            new_batch["obs_mask"] = batch["obs_mask"] * (~replace_mask).float()
            return new_batch

        vb = batch["value_bins"].clone()
        varis = batch["varis"]
        md_dev = self._marginal_dist_on(marginal_dist, device)

        if strategy == "marginal":
            if replace_mask.any():
                rep_vars = varis[replace_mask]
                if md_dev is not None:
                    sampled = self._median_bins_for(md_dev)[rep_vars]
                else:
                    sampled = self._random_bins(int(replace_mask.sum().item()), device)
                vb[replace_mask] = sampled
            new_batch["value_bins"] = vb
            return new_batch

        if strategy == "conditional":
            tau = batch["tau"]
            obs_mask = batch["obs_mask"]
            keep_mask = (obs_mask > 0.5) & (~replace_mask)  # [B, L]

            same_var = varis.unsqueeze(2) == varis.unsqueeze(1)          # [B, L, L]
            valid = same_var & keep_mask.unsqueeze(1)                     # valid[b,i,j]
            dists = (tau.unsqueeze(2) - tau.unsqueeze(1)).abs().float()   # [B, L, L]
            dists = dists.masked_fill(~valid, float("inf"))

            nearest_j = dists.argmin(dim=2)                               # [B, L]
            nearest_vb = vb.gather(1, nearest_j)                          # [B, L]
            nearest_dist = dists.gather(2, nearest_j.unsqueeze(2)).squeeze(2)
            has_match = nearest_dist < float("inf")                       # [B, L]

            matched_positions = replace_mask & has_match
            vb[matched_positions] = nearest_vb[matched_positions]

            fallback_positions = replace_mask & ~has_match
            if fallback_positions.any():
                fb_vars = varis[fallback_positions]
                if md_dev is not None:
                    vb[fallback_positions] = self._median_bins_for(md_dev)[fb_vars]
                else:
                    vb[fallback_positions] = self._random_bins(
                        int(fallback_positions.sum().item()), device
                    )
            new_batch["value_bins"] = vb
            return new_batch

        raise ValueError(f"Unknown replacement strategy: {strategy!r}")


# =============================================================================
# Factory: load_backbone_adapter
# =============================================================================


def _make_tisae_adapter(
    ckpt: Dict[str, Any],
    device: torch.device,
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> TISAEBackboneAdapter:
    """Build a TISAEBackboneAdapter from a Discrete backbone checkpoint dict.
    """
    bb_cfg_dict = (
        ckpt.get("backbone_config")
        or ckpt.get("config")
        or ckpt.get("model_config")
        or {}
    )
    valid_fields = {f.name for f in fields(BackboneConfig)}
    bb_cfg = BackboneConfig(**{k: v for k, v in bb_cfg_dict.items() if k in valid_fields})

    backbone = BinnedTimeTransformer(bb_cfg)
    state_dict = ckpt["model_state_dict"]
    new_sd = {}
    for k, v in state_dict.items():
        if ".layer." in k and k.startswith("layers."):
            new_sd[k.replace(".layer.", ".")] = v
        else:
            new_sd[k] = v
    missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        msg = (
            f"BinnedTimeTransformer state_dict load mismatch:\n"
            f"  missing keys ({len(missing)}): {missing[:8]}\n"
            f"  unexpected keys ({len(unexpected)}): {unexpected[:8]}\n"
            f"This is a silent partial-load and almost certainly a bug — "
            f"the legacy '.layer.' rename rule may not cover this checkpoint."
        )
        if missing:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
    backbone = backbone.to(device).eval()

    # Resolve T, V, var_to_ind_hash. Priority: explicit metadata bundle > dataset
    # metadata passed by the caller > BackboneConfig fallback.
    md = ckpt.get("unified_dataset_metadata") or {}
    T = md.get("T") or bb_cfg.T
    V = md.get("V") or bb_cfg.V
    h = md.get("var_to_ind_hash")
    if h is None and dataset_meta is not None:
        T = T or dataset_meta.get("T")
        V = V or dataset_meta.get("V")
        h = dataset_meta.get("var_to_ind_hash")

    adapter = TISAEBackboneAdapter(
        backbone,
        T=int(T),
        V=int(V),
        var_to_ind_hash=h,
        is_finetune=bool(ckpt.get("is_finetune", False)),
    )
    return adapter.freeze()


# =============================================================================
# StratsBackboneAdapter — wraps the ported Strats model
# =============================================================================


class StratsBackboneAdapter(BackboneAdapter):
    """BackboneAdapter for the ported STraTS model
    (:class:`tisae.models.external.strats.Strats`).

    Continuous-value backbone: consumes `values_zscore` (NOT `value_bins`) and
    `times_norm` (NOT `tau`) from the unified-collate batch dict. The TISAE
    grid scatter still uses the binned `tau` and `varis` integer indices for
    the cell coordinates.

    """

    name = "strats"

    def __init__(
        self,
        module,  # tisae.models.external.strats.Strats
        T: int,
        V: int,
        var_to_ind_hash: Optional[str] = None,
        is_finetune: bool = False,
    ):
        super().__init__(T=T, V=V, D=module.strats_cfg.hid_dim, n_value_bins=0)
        self.module = module
        self.var_to_ind_hash = var_to_ind_hash
        self.is_finetune = is_finetune


    @torch.no_grad()
    def extract_hidden(
        self,
        batch: Dict[str, torch.Tensor],
        return_event_h: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Run the STraTS transformer on continuous (values_zscore, times_norm)
        and scatter the per-event contextual embedding into a (T, V, D) grid
        keyed by the BINNED `tau` and `varis` indices.

        """
        values = batch["values_zscore"]
        times = batch["times_norm"]
        varis = batch["varis"]
        obs_mask = batch["obs_mask"]
        tau_bins = batch["tau"]  # used only for grid scatter coordinates

        H = self.module.get_contextual_embeddings_at_layer(
            values, times, varis, obs_mask, self.extract_layer
        )
        H_grid, obs_grid = build_H_grid(H, tau_bins, varis, obs_mask, self.T, self.V)
        if return_event_h:
            return H_grid, obs_grid, H
        return H_grid, obs_grid

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_pooled: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:

        values = batch["values_zscore"]
        times = batch["times_norm"]
        varis = batch["varis"]
        obs_mask = batch["obs_mask"]

        contextual_emb = self.module.get_contextual_embeddings(values, times, varis, obs_mask)
        attn = self.module.fusion_att(contextual_emb, obs_mask)  # [B, L]
        ts_emb = (contextual_emb * attn[:, :, None]).sum(dim=1)  # [B, D]

        if self.is_finetune:
            logits = self.module.binary_head(self.module.forecast_head(ts_emb))[:, 0]
        else:
            logits = self.module.binary_head(ts_emb)[:, 0]

        result: Dict[str, torch.Tensor] = {"logits": logits, "H": contextual_emb}
        if return_pooled:
            result["pooled"] = ts_emb
        if return_attention:
            result["attention"] = attn
        return result

    def predict_from_grid(
        self,
        H_hat_grid: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        obs_grid: torch.Tensor,
    ) -> torch.Tensor:

        tau = batch["tau"]
        varis = batch["varis"]
        obs_mask = batch["obs_mask"]
        H_hat_event = grid_to_H(H_hat_grid, tau, varis)
        if self.H_grid_mean is not None:
            H_hat_event = H_hat_event + self.H_grid_mean[varis]

        if self.extract_layer != -1:
            # Run through remaining layers before pooling
            return self.module.forward_from_layer(
                H_hat_event, obs_mask, start_layer=self.extract_layer + 1
            )

        # Default: direct pool + predict (extract_layer == -1)
        attn = self.module.fusion_att(H_hat_event, obs_mask)
        ts_emb = (H_hat_event * attn[:, :, None]).sum(dim=1)

        if self.is_finetune:
            return self.module.binary_head(self.module.forecast_head(ts_emb))[:, 0]
        return self.module.binary_head(ts_emb)[:, 0]

    def pool_events(
        self,
        H_event: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = self.module.fusion_att(H_event, obs_mask)
        pooled = (H_event * attn[:, :, None]).sum(dim=1)
        return pooled, attn

    def predict_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        if self.is_finetune:
            return self.module.binary_head(self.module.forecast_head(pooled))[:, 0]
        return self.module.binary_head(pooled)[:, 0]

    @property
    def has_continuous_values(self) -> bool:
        return True

    def forward_with_value_mask(
        self,
        batch: Dict[str, torch.Tensor],
        soft_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with `values_zscore * soft_mask`, returning [B] logits.

        Used by value-level dynamask. Multiplying z-scored values by a soft
        mask in [0, 1] smoothly fades each measurement toward the population
        mean (z-score 0). Variable identity, time, and obs_mask are
        unchanged.
        """
        new_batch = {**batch, "values_zscore": batch["values_zscore"] * soft_mask}
        return self.forward(new_batch)["logits"]

    def forward_from_event_embeddings(
        self,
        emb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Captum-friendly: take a `[B, L, D]` triplet-embedding tensor (the
        output of `get_event_embeddings`), run it through the FULL transformer
        + fusion attention + head, return logits.

        """
        obs_mask = batch["obs_mask"]
        triplet_emb = F.dropout(emb, self.module.dropout_p, self.module.training)
        contextual_emb = self.module.transformer(triplet_emb, obs_mask)
        attn = self.module.fusion_att(contextual_emb, obs_mask)
        ts_emb = (contextual_emb * attn[:, :, None]).sum(dim=1)
        if self.is_finetune:
            return self.module.binary_head(self.module.forecast_head(ts_emb))[:, 0]
        return self.module.binary_head(ts_emb)[:, 0]

    def get_event_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pre-context triplet embedding `[B, L, hid_dim]`.
        """
        return self.module.get_event_embeddings(
            batch["values_zscore"],
            batch["times_norm"],
            batch["varis"],
        )

    def get_baseline_event_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "median",
    ) -> torch.Tensor:
        """For CVE backbones (STraTS/HyMaTE), the baseline is `values_zscore=0`
        regardless of mode (z-scoring already centers data, so the
        population-mean baseline IS the zero baseline). The `mode` argument
        is accepted for API compatibility with the binned adapter but is
        effectively ignored.
        """
        if mode not in ("median", "zero"):
            raise ValueError(f"Unknown baseline mode: {mode!r}; expected 'median' or 'zero'")
        zero_zscore = torch.zeros_like(batch["values_zscore"])
        return self.get_event_embeddings({**batch, "values_zscore": zero_zscore})

    @torch.no_grad()
    def get_attention_grid(
        self,
        batch: Dict[str, torch.Tensor],
        T: int,
        V: int,
    ) -> torch.Tensor:
        out = self.forward(batch, return_attention=True)
        attn = out["attention"]  # [B, L]

        tau = batch["tau"].long()
        varis = batch["varis"].long()
        obs_mask = batch["obs_mask"]
        valid = obs_mask > 0.5
        attn = attn * obs_mask
        safe_tau = torch.where(valid, tau, torch.zeros_like(tau))
        safe_varis = torch.where(valid, varis, torch.zeros_like(varis))
        cell_idx = safe_tau * V + safe_varis

        B, L = attn.shape
        attr_tv = torch.zeros(B, T * V, device=attn.device, dtype=attn.dtype)
        attr_tv.scatter_add_(1, cell_idx, attn)
        return attr_tv.view(B, T, V)

    # ------------------------------------------------------------------
    # Value replacement — operates on continuous z-scored values
    # ------------------------------------------------------------------

    def _sample_population_zscore(
        self,
        var_ids: torch.Tensor,
        population_values_zscore: Dict[int, np.ndarray],
        device: torch.device,
    ) -> torch.Tensor:
        """Deterministic per-variable median z-score lookup for each element of `var_ids[N]`.
        """
        cache_key = id(population_values_zscore)
        if getattr(self, "_median_zscore_cache_key", None) != cache_key:
            V = max(population_values_zscore.keys()) + 1
            lut = torch.zeros(V, dtype=torch.float32)
            for v, pool in population_values_zscore.items():
                if pool is not None and len(pool) > 0:
                    lut[int(v)] = float(np.median(pool))
            self._median_zscore_cache_key = cache_key
            self._median_zscore_lut = lut

        lut = self._median_zscore_lut.to(device)  # type: ignore[attr-defined]
        return lut[var_ids.long()]

    def replace_event_value(
        self,
        batch: Dict[str, torch.Tensor],
        replace_mask: torch.Tensor,
        strategy: str,
        marginal_dist: Optional[torch.Tensor] = None,
        population_values_zscore: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Edit `values_zscore` per the requested strategy.

        - 'removal'    : zero out the corresponding `obs_mask` positions
        - 'marginal'   : sample replacement z-scored values from
                         `population_values_zscore[var_id]` (training pool)
        - 'conditional': use the patient's nearest same-variable observed
                         z-score; fall back to marginal if no match
        """
        new_batch = {k: v for k, v in batch.items()}
        replace_mask = replace_mask.bool()
        device = batch["values_zscore"].device

        if strategy == "removal":
            new_batch["obs_mask"] = batch["obs_mask"] * (~replace_mask).float()
            return new_batch

        if population_values_zscore is None:
            raise ValueError(
                f"StratsBackboneAdapter.replace_event_value strategy={strategy!r} "
                f"requires `population_values_zscore` from the unified dataset"
            )

        vz = batch["values_zscore"].clone()
        varis = batch["varis"]

        if strategy == "marginal":
            if replace_mask.any():
                rep_vars = varis[replace_mask]
                vz[replace_mask] = self._sample_population_zscore(
                    rep_vars, population_values_zscore, device
                )
            new_batch["values_zscore"] = vz
            return new_batch

        if strategy == "conditional":
            # Vectorized nearest-same-variable match — mirrors the binned adapter's implementation.
            tau = batch["tau"]
            obs_mask = batch["obs_mask"]
            keep_mask = (obs_mask > 0.5) & (~replace_mask)  # [B, L]

            same_var = varis.unsqueeze(2) == varis.unsqueeze(1)           # [B, L, L]
            valid = same_var & keep_mask.unsqueeze(1)
            dists = (tau.unsqueeze(2) - tau.unsqueeze(1)).abs().float()
            dists = dists.masked_fill(~valid, float("inf"))

            nearest_j = dists.argmin(dim=2)                                # [B, L]
            nearest_vz = vz.gather(1, nearest_j)                           # [B, L]
            nearest_dist = dists.gather(2, nearest_j.unsqueeze(2)).squeeze(2)
            has_match = nearest_dist < float("inf")

            matched_positions = replace_mask & has_match
            vz[matched_positions] = nearest_vz[matched_positions]

            fallback_positions = replace_mask & ~has_match
            if fallback_positions.any():
                fb_vars = varis[fallback_positions]
                vz[fallback_positions] = self._sample_population_zscore(
                    fb_vars, population_values_zscore, device
                )
            new_batch["values_zscore"] = vz
            return new_batch

        raise ValueError(f"Unknown replacement strategy: {strategy!r}")


# =============================================================================
# HyMaTEBackboneAdapter — wraps the ported HyMaTE model
# =============================================================================


class HyMaTEBackboneAdapter(StratsBackboneAdapter):
    """BackboneAdapter for the ported HyMaTE model
    (:class:`tisae.models.external.hymate.HyMaTE`).

    HyMaTE has the same per-event interface as STraTS — same
    `get_contextual_embeddings(values, times, varis, obs_mask)` method,
    same `fusion_att`, same `binary_head` / `forecast_head`. The only
    architectural difference (mamba vs transformer) is internal to
    `get_contextual_embeddings`. Therefore this adapter inherits from
    :class:`StratsBackboneAdapter` and only overrides `__init__` and
    `forward_from_event_embeddings` (because the post-embedding pipeline
    differs: HyMaTE runs the triplet through `mamba` instead of
    `transformer`).
    """

    name = "hymate"

    def __init__(
        self,
        module,  # tisae.models.external.hymate.HyMaTE
        T: int,
        V: int,
        var_to_ind_hash: Optional[str] = None,
        is_finetune: bool = False,
    ):
        # Skip StratsBackboneAdapter's super().__init__ because it reads
        # `module.strats_cfg.hid_dim`. Call BackboneAdapter directly.
        BackboneAdapter.__init__(
            self,
            T=T,
            V=V,
            D=module.hymate_cfg.hid_dim,
            n_value_bins=0,
        )
        self.module = module
        self.var_to_ind_hash = var_to_ind_hash
        self.is_finetune = is_finetune

    def forward_from_event_embeddings(
        self,
        emb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Captum-friendly: take a `[B, L, D]` triplet-embedding tensor
        (the output of `get_event_embeddings`), run it through ALL mamba
        blocks + fusion attention + head, return logits.

        Note: this always runs ALL mamba blocks regardless of
        ``extract_layer``. The ``emb`` input is pre-context (no mamba),
        so the full mamba stack must be applied to produce a valid logit.
        """
        obs_mask = batch["obs_mask"]
        triplet_emb = F.dropout(emb, self.module.dropout_p, self.module.training)
        contextual_emb = self.module.mamba(triplet_emb)
        attn = self.module.fusion_att(contextual_emb, obs_mask)
        ts_emb = (contextual_emb * attn[:, :, None]).sum(dim=1)
        if self.is_finetune:
            return self.module.binary_head(self.module.forecast_head(ts_emb))[:, 0]
        return self.module.binary_head(ts_emb)[:, 0]


# =============================================================================
# Factory: dispatchers for each backbone_type
# =============================================================================


def _make_strats_adapter(
    ckpt: Dict[str, Any],
    device: torch.device,
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> StratsBackboneAdapter:
    """Build a StratsBackboneAdapter from a checkpoint produced by
    `train_external_backbone.py --backbone strats`.
    """
    from tisae.models.external.strats import Strats, StratsConfig

    bb_cfg_dict = ckpt.get("backbone_config", {})
    valid_fields = {f.name for f in fields(StratsConfig)}
    cfg = StratsConfig(**{k: v for k, v in bb_cfg_dict.items() if k in valid_fields})

    module = Strats(cfg)
    missing, unexpected = module.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        msg = (
            f"Strats state_dict load mismatch:\n"
            f"  missing keys ({len(missing)}): {missing[:8]}\n"
            f"  unexpected keys ({len(unexpected)}): {unexpected[:8]}"
        )
        if missing:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
    module = module.to(device).eval()

    md = ckpt.get("unified_dataset_metadata") or {}
    T = md.get("T") or (dataset_meta or {}).get("T")
    V = md.get("V") or cfg.V
    h = md.get("var_to_ind_hash") or (dataset_meta or {}).get("var_to_ind_hash")

    adapter = StratsBackboneAdapter(
        module,
        T=int(T),
        V=int(V),
        var_to_ind_hash=h,
        is_finetune=bool(ckpt.get("is_finetune", False)),
    )
    return adapter.freeze()


def _make_hymate_adapter(
    ckpt: Dict[str, Any],
    device: torch.device,
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> "HyMaTEBackboneAdapter":
    """Build a HyMaTEBackboneAdapter from a checkpoint produced by
    `train_external_backbone.py --backbone hymate`.
    """
    from tisae.models.external.hymate import HyMaTE, HyMaTEConfig

    bb_cfg_dict = ckpt.get("backbone_config", {})
    valid_fields = {f.name for f in fields(HyMaTEConfig)}
    cfg = HyMaTEConfig(**{k: v for k, v in bb_cfg_dict.items() if k in valid_fields})

    module = HyMaTE(cfg)
    missing, unexpected = module.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        msg = (
            f"HyMaTE state_dict load mismatch:\n"
            f"  missing keys ({len(missing)}): {missing[:8]}\n"
            f"  unexpected keys ({len(unexpected)}): {unexpected[:8]}"
        )
        if missing:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
    module = module.to(device).eval()

    md = ckpt.get("unified_dataset_metadata") or {}
    T = md.get("T") or (dataset_meta or {}).get("T")
    V = md.get("V") or cfg.V
    h = md.get("var_to_ind_hash") or (dataset_meta or {}).get("var_to_ind_hash")

    adapter = HyMaTEBackboneAdapter(
        module,
        T=int(T),
        V=int(V),
        var_to_ind_hash=h,
        is_finetune=bool(ckpt.get("is_finetune", False)),
    )
    return adapter.freeze()


# Registry of adapter constructors keyed by `backbone_type`.
_ADAPTER_REGISTRY: Dict[str, Any] = {
    "tisae": _make_tisae_adapter,
    "strats": _make_strats_adapter,
    "hymate": _make_hymate_adapter,
}


def register_adapter(name: str, constructor) -> None:
    """Register an adapter constructor for a `backbone_type`.

    `constructor(ckpt_dict, device, dataset_meta=None) -> BackboneAdapter`.
    """
    _ADAPTER_REGISTRY[name] = constructor


def load_backbone_adapter(
    checkpoint_path: str,
    device: str = "cuda",
    dataset_meta: Optional[Dict[str, Any]] = None,
) -> BackboneAdapter:
    """Load a checkpoint and dispatch to the right adapter constructor.

    Args:
        checkpoint_path: Path to a `*.pt` saved by either `train_backbone.py` or `train_external_backbone.py`.
        device:    'cuda' / 'cpu'.
        dataset_meta: Optional dict carrying `{T, V, var_to_ind_hash, ...}`.

    Returns:
        BackboneAdapter ready for evaluation / inference.
    """
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_type = ckpt.get("backbone_type", "tisae")
    if backbone_type not in _ADAPTER_REGISTRY:
        try:
            __import__(f"tisae.models.{backbone_type}_adapter")
        except ImportError:
            pass
    if backbone_type not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"No adapter registered for backbone_type={backbone_type!r}. "
            f"Known types: {sorted(_ADAPTER_REGISTRY)}. "
            f"Did you forget to import the module that registers it?"
        )
    return _ADAPTER_REGISTRY[backbone_type](ckpt, torch.device(device), dataset_meta)
