"""Pretraining evaluator — port of HyMaTE's evaluator_pretrain.py.

Computes the mean masked-MSE forecast loss on a held-out split for a
ported STraTS / HyMaTE backbone in pretraining mode (`cfg.pretrain=True`).

"""

from __future__ import annotations

from typing import Dict, List

import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_pretrain_split(
    model,
    dataset,
    split: str,
    eval_batch_size: int = 32,
    device=None,
    n_repeats: int = 3,
    log_fn=print,
) -> Dict[str, float]:
    """Compute the mean per-target masked MSE on a pretraining split.

    The model is expected to be a Strats/HyMaTE backbone whose forward
    signature, in pretraining mode, returns a scalar masked-MSE loss:

        loss = model(values_zscore, times_norm, varis, obs_mask,
                     forecast_values=fv, forecast_mask=fm)

    The total loss is computed as `(loss * num_pred).sum() / sum(num_pred)`
    over all batches and repeats, where `num_pred = forecast_mask.sum()`
    is the number of variables with a forecast target in that sample.
    This matches the upstream HyMaTE evaluator.

    Args:
        model:           ported Strats/HyMaTE module in pretraining mode
        dataset:         a :class:`tisae.data.pretrain_dataset.PretrainDataset`
        split:           "train" / "val"
        eval_batch_size: number of samples per batch
        device:          target device for the batch tensors
        n_repeats:       number of independent `t1` draws per sample (matches
                         upstream's `for i in range(3)` triple sampling)
        log_fn:          logging function

    Returns:
        {"forecast_mse": float, "n_predictions": int}
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    eval_ind = dataset.splits[split]
    num_samples = len(eval_ind)
    if num_samples == 0:
        return {"forecast_mse": float("nan"), "n_predictions": 0}

    log_fn(f"\nPretrain evaluator — split={split}, n={num_samples}, n_repeats={n_repeats}")

    cum_weighted_loss = 0.0
    cum_count = 0.0

    pbar = tqdm(
        range(0, num_samples, eval_batch_size),
        desc=f"pretrain eval {split}",
    )
    n_skipped_nan = 0
    for start in pbar:
        end = min(num_samples, start + eval_batch_size)
        batch_ind = eval_ind[start:end]
        for _ in range(n_repeats):
            batch = dataset.get_batch(batch_ind)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(
                batch["values_zscore"],
                batch["times_norm"],
                batch["varis"],
                batch["obs_mask"],
                forecast_values=batch["forecast_values"],
                forecast_mask=batch["forecast_mask"],
            )
            loss_val = float(loss.item())
            if not (loss_val == loss_val) or loss_val == float("inf") or loss_val == float("-inf"):
                n_skipped_nan += 1
                continue
            num_pred = batch["forecast_mask"].sum()
            cum_weighted_loss += loss_val * float(num_pred.item())
            cum_count += float(num_pred.item())

    if n_skipped_nan > 0:
        log_fn(
            f"  WARN: skipped {n_skipped_nan} val batches with NaN/Inf loss "
            f"(check the dataset for pathological values)"
        )

    if cum_count <= 0:
        return {"forecast_mse": float("nan"), "n_predictions": 0}

    return {
        "forecast_mse": cum_weighted_loss / cum_count,
        "n_predictions": int(cum_count),
        "n_skipped_nan": n_skipped_nan,
    }
