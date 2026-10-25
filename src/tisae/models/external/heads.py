"""TimeSeriesModelBase — clean prediction-head wrapper for ported backbones.

This module is a stripped-down replacement for the `TimeSeriesModel` class in
the upstream HyMaTE repository. The upstream class entangles the
prediction heads with `argparse.Namespace`, a custom `Logger`, parameter
counting helpers, and the demographics-fusion concat that we explicitly drop
since demographics are excluded from all backbones.

The clean version provides ONLY the four pieces of state that the ported
STraTS / HyMaTE forward passes need:

    self.binary_head      : nn.Linear  -> mortality logit
    self.forecast_head    : nn.Linear  -> 2-h forecast (pretrain / finetune-from-pretrain)
    self.pos_class_weight : torch.Tensor scalar -> BCEWithLogitsLoss pos_weight
    self.pretrain         : bool       -> selects forecast vs binary path
    self.finetune         : bool       -> stack binary_head ∘ forecast_head   

Plus two helper methods:
    binary_cls_final(logits, labels)  -> BCEWithLogitsLoss or sigmoid(logits)
    forecast_final(ts_emb, fv, fm)    -> mean masked-MSE forecast loss

Since demographics are excluded:
    - There is NO `demo_emb` MLP
    - The "ts_demo_emb_size" is just `hid_dim` (no concat with demographics)
    - The forward signature in the ported backbones drops the `demo` argument

The constructor takes a small dict of plain Python values instead of an
`argparse.Namespace` so the ported backbones can be instantiated without any
CLI plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TimeSeriesModelBaseConfig:
    """Minimal config for the prediction-head wrapper.

    Fields:
        V                : number of variables (used to size forecast_head)
        hid_dim          : hidden dimension of the per-event representation
        pretrain         : True for pretraining (forecast_head only, no binary_head)
        finetune         : True for finetune-from-pretrain (binary_head ∘ forecast_head)
                           False for from-scratch supervised (binary_head only).
                           Mutually exclusive with `pretrain`.
        pos_class_weight : positive-class weight for BCEWithLogitsLoss
    """

    V: int
    hid_dim: int
    pretrain: bool = False
    finetune: bool = False
    pos_class_weight: float = 1.0


class TimeSeriesModelBase(nn.Module):
    """Clean replacement for HyMaTE's `TimeSeriesModel`.

    Differences vs the upstream HyMaTE models.py:
      - No `argparse.Namespace` dependency; takes a `TimeSeriesModelBaseConfig`.
      - No `Logger` / `count_parameters` helpers (debug-only).
      - No demographics MLP — `demo_emb` is removed entirely.
      - The "ts_demo_emb" embedding size is just `hid_dim` (was `hid_dim * 2`
        because of the demo concat).
      - `binary_head` input dim is `hid_dim` (was `hid_dim + demo_emb_dim`).
      - `pos_class_weight` is stored as a scalar `nn.Buffer` so it survives
        `.to(device)` and shows up in `state_dict` for reproducibility.

    The three downstream output paths are:

        self.pretrain == True:
            forecast_head(ts_emb) -> [B, V] forecast prediction
            forecast_final(ts_emb, fv, fm) -> mean masked-MSE loss

        self.finetune == True:
            binary_head(forecast_head(ts_emb))[:, 0] -> [B] mortality logit
            binary_cls_final(logits, labels) -> BCEWithLogitsLoss

        else (from-scratch supervised):
            binary_head(ts_emb)[:, 0] -> [B] mortality logit
            binary_cls_final(logits, labels) -> BCEWithLogitsLoss
    """

    def __init__(self, cfg: TimeSeriesModelBaseConfig):
        super().__init__()
        if cfg.pretrain and cfg.finetune:
            raise ValueError(
                "TimeSeriesModelBase: cannot have both pretrain=True and "
                "finetune=True (mutually exclusive head configurations)"
            )
        self.cfg = cfg
        self.pretrain = bool(cfg.pretrain)
        self.finetune = bool(cfg.finetune)

        # ts_emb size is hid_dim per the no-demographics design decision (was hid_dim*2 with demo concat)
        ts_emb_size = cfg.hid_dim

        if self.pretrain:
            # Pretrain path: forecast_head only.
            self.forecast_head = nn.Linear(ts_emb_size, cfg.V)
        elif self.finetune:
            # Finetune-from-pretrain path: binary_head ∘ forecast_head.
            self.forecast_head = nn.Linear(ts_emb_size, cfg.V)
            self.binary_head = nn.Linear(cfg.V, 1)
        else:
            # From-scratch supervised path: binary_head directly.
            self.binary_head = nn.Linear(ts_emb_size, 1)

        # pos_class_weight as a buffer so .to(device) moves it.
        self.register_buffer(
            "pos_class_weight",
            torch.tensor(float(cfg.pos_class_weight)),
        )

    def binary_cls_final(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Either BCEWithLogitsLoss(labels=labels) or sigmoid(logits).

        Matches HyMaTE's signature: when `labels is None` (eval), returns the
        sigmoid probabilities; when `labels` is provided (train), returns the
        BCE loss using `pos_class_weight`.
        """
        if labels is not None:
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_class_weight
            )
        return torch.sigmoid(logits)

    def forecast_final(
        self,
        ts_emb: torch.Tensor,
        forecast_values: torch.Tensor,
        forecast_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean masked MSE between predicted and target 2-h forecast values.

        ts_emb        : [B, hid_dim]   pooled per-sample representation
        forecast_values : [B, V]      target z-scored value at each variable
        forecast_mask : [B, V]        1 = variable observed in [t1, t1+2h]
        """
        pred = self.forecast_head(ts_emb)  # [B, V]
        return (forecast_mask * (pred - forecast_values) ** 2).sum() / forecast_mask.sum().clamp_min(1.0)
