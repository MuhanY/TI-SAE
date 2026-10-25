"""Train an external backbone (STraTS or HyMaTE) on the unified dataset.

Usage
-----

STraTS — pretrain → finetune:

    python -m tisae.training.train_external_backbone \\
        --backbone strats --stage pretrain \\
        --dataset_path data/unified_mimic3_240h.pkl \\
        --cohort mimic_iii \\
        --output_dir experiments/strats/mimic3_mortality/strats_mimic_iii_d32_pretrain \\
        --hid_dim 32 --num_layers 2 --num_heads 4 \\
        --train_batch_size 16 --lr 5e-4 --epochs 50 --patience 10 \\
        --pretrain_max_obs 880

    python -m tisae.training.train_external_backbone \\
        --backbone strats --stage finetune \\
        --dataset_path data/unified_mimic3_48h.pkl \\
        --cohort mimic_iii \\
        --output_dir experiments/strats/mimic3_mortality/strats_mimic_iii_d32_finetune \\
        --load_ckpt_path experiments/strats/mimic3_mortality/strats_mimic_iii_d32_pretrain/best_model.pt \\
        --hid_dim 32 --num_layers 2 --num_heads 4 \\
        --train_batch_size 16 --lr 5e-4 --epochs 50 --patience 10

HyMaTE — pretrain → finetune (adds ``--num_blocks`` for the mamba stack):

    python -m tisae.training.train_external_backbone \\
        --backbone hymate --stage pretrain \\
        --dataset_path data/unified_mimic3_240h.pkl \\
        --cohort mimic_iii \\
        --output_dir experiments/strats/mimic3_mortality/hymate_mimic_iii_d32_pretrain \\
        --hid_dim 32 --num_blocks 3 --num_layers 2 --num_heads 4 \\
        --train_batch_size 16 --lr 5e-4 --epochs 50 --patience 10 \\
        --pretrain_max_obs 880

    python -m tisae.training.train_external_backbone \\
        --backbone hymate --stage finetune \\
        --dataset_path data/unified_mimic3_48h.pkl \\
        --cohort mimic_iii \\
        --output_dir experiments/strats/mimic3_mortality/hymate_mimic_iii_d32_finetune \\
        --load_ckpt_path experiments/strats/mimic3_mortality/hymate_mimic_iii_d32_pretrain/best_model.pt \\
        --hid_dim 32 --num_blocks 3 --num_layers 2 --num_heads 4 \\
        --train_batch_size 16 --lr 5e-4 --epochs 50 --patience 10

Notes:
  * HyMaTE supervised-only: same as the HyMaTE finetune command WITHOUT
    ``--load_ckpt_path``. The model is built from scratch and the
    checkpoint stamps ``is_finetune=False``.
  * MIMIC-IV: swap ``--dataset_path`` to
    ``data/processed_mimiciv/unified_*.pkl`` and ``--cohort mimic_iv``.

The trained checkpoint is saved in the uniform format expected by
:func:`tisae.models.backbone_adapter.load_backbone_adapter`:

    {
        'backbone_type': 'strats' | 'hymate',
        'backbone_config': dict (StratsConfig / HyMaTEConfig fields),
        'model_state_dict': dict,
        'unified_dataset_metadata': {'V', 'T', 'var_to_ind_hash', ...},
        'val_auroc': float,
        'training_args': dict,
        'is_finetune': bool,                # True if --load_ckpt_path was set (finetune-from-pretrain)
    }

Stages
------
- ``--stage finetune`` (this file): supervised mortality training
  with the binary head. ``--load_ckpt_path`` (optional) loads a
  pretraining checkpoint and stacks the binary head on top of the forecast head
  per the finetune dispatch logic.

- ``--stage pretrain``: 2-hour value forecasting via the STraTS-style
  sliding window. Added in :mod:`tisae.training.train_external_backbone` once
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from tisae.utils.seeding import set_seed

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


TASK_BATCH_KEY = {
    'mortality': 'labels',
    'los_gt3d': 'labels_los_gt3d',
    'los_gt7d': 'labels_los_gt7d',
    'readmission': 'labels_readmission',
}

TASK_LABEL_ATTR = {
    'mortality': 'y',
    'los_gt3d': 'y_los_gt3d',
    'los_gt7d': 'y_los_gt7d',
    'readmission': 'y_readmission',
}


# =============================================================================
# Logger (matches train_backbone.py / train_tisae.py)
# =============================================================================


class Logger:
    def __init__(self, log_dir, filename="log.txt"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename

    def write(self, msg: str) -> None:
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")


# =============================================================================
# Dataset loader
# =============================================================================


def _load_unified(path: str):
    """Load a UnifiedTimeSeriesDataset pickle."""
    from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
    return UnifiedTimeSeriesDataset.load(path)


def _compute_pos_class_weight(dataset) -> float:
    """(n_neg / n_pos) on the training split."""
    y_train = np.asarray(dataset.y)[np.asarray(dataset.splits["train"])]
    n_pos = float(y_train.sum())
    if n_pos < 1.0:
        return 1.0
    return float((len(y_train) - n_pos) / n_pos)


# =============================================================================
# Model factory
# =============================================================================


def _build_model(args, dataset):
    """Construct the requested external backbone."""
    if args.backbone == "strats":
        from tisae.models.external.strats import Strats, StratsConfig
        cfg = StratsConfig(
            V=dataset.V,
            hid_dim=args.hid_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            pretrain=(args.stage == "pretrain"),
            finetune=(args.stage == "finetune" and args.load_ckpt_path is not None),
            pos_class_weight=_compute_pos_class_weight(dataset),
            residual=args.residual,
        )
        model = Strats(cfg)
        return model, cfg

    if args.backbone == "hymate":
        # Surfaces a clear error if mamba-ssm is not installed.
        try:
            from tisae.models.external.hymate import HyMaTE, HyMaTEConfig
        except ImportError as e:
            raise NotImplementedError(
                "HyMaTE backbone requires mamba-ssm. "
                "Install with: pip install mamba-ssm causal-conv1d"
            ) from e
        cfg = HyMaTEConfig(
            V=dataset.V,
            hid_dim=args.hid_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            pretrain=(args.stage == "pretrain"),
            finetune=(args.stage == "finetune" and args.load_ckpt_path is not None),
            pos_class_weight=_compute_pos_class_weight(dataset),
        )
        model = HyMaTE(cfg)
        return model, cfg

    raise ValueError(f"Unknown backbone: {args.backbone!r}")


# =============================================================================
# Batch builder for the dict-API
# =============================================================================


def _make_batch(dataset, indices, device) -> Dict[str, torch.Tensor]:
    batch = dataset.get_batch(indices)
    return {k: v.to(device) for k, v in batch.items()}


# =============================================================================
# Finetune training step (supervised mortality)
# =============================================================================


def _finetune_step(model, batch: Dict[str, torch.Tensor], label_key: str = "labels"):
    """Run one supervised forward pass on a STraTS / HyMaTE backbone.

    Returns the BCE loss tensor and the predicted logits (for AUROC tracking).
    """
    values = batch["values_zscore"]
    times = batch["times_norm"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]
    labels = batch[label_key]

    loss, _ = model(values, times, varis, obs_mask, labels=labels)
    return loss


def _pretrain_step(model, batch: Dict[str, torch.Tensor]):
    """Run one 2-h forecast pretraining forward pass on a STraTS / HyMaTE
    backbone in pretrain mode (`cfg.pretrain=True`).

    The forward signature in pretrain mode returns the masked-MSE forecast loss directly:

        loss = model(values_zscore, times_norm, varis, obs_mask,
                     forecast_values=fv, forecast_mask=fm)
    """
    values = batch["values_zscore"]
    times = batch["times_norm"]
    varis = batch["varis"]
    obs_mask = batch["obs_mask"]

    return model(
        values,
        times,
        varis,
        obs_mask,
        forecast_values=batch["forecast_values"],
        forecast_mask=batch["forecast_mask"],
    )


@torch.no_grad()
def _eval_split(model, dataset, split: str, batch_size: int, device: torch.device,
                label_key: str = "labels", task: str = "mortality"):
    """Evaluate the model on a split. Returns dict with loss / auroc / auprc."""
    model.eval()
    indices = np.asarray(dataset.splits[split])
    if task == 'readmission':
        valid = dataset.y_readmission[indices] >= 0
        indices = indices[valid]
    n = len(indices)

    cum_loss = 0.0
    n_seen = 0
    all_logits: List[float] = []
    all_labels: List[float] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        batch = _make_batch(dataset, idx, device)
        loss, _ = model(
            batch["values_zscore"],
            batch["times_norm"],
            batch["varis"],
            batch["obs_mask"],
            labels=batch[label_key],
        )
        cum_loss += loss.item() * (end - start)
        n_seen += end - start

        with torch.no_grad():
            probs = model(
                batch["values_zscore"],
                batch["times_norm"],
                batch["varis"],
                batch["obs_mask"],
                labels=None,
            )
            if isinstance(probs, tuple):
                probs = probs[0]
            all_logits.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(batch[label_key].detach().cpu().numpy().tolist())

    metrics = {"loss": cum_loss / max(n_seen, 1)}
    if HAVE_SKLEARN and len(set(all_labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(all_labels, all_logits))
        metrics["auprc"] = float(average_precision_score(all_labels, all_logits))
    return metrics


# =============================================================================
# Save uniform-format checkpoint
# =============================================================================


def _save_uniform_checkpoint(
    path: Path,
    *,
    backbone_type: str,
    backbone_config,
    model: torch.nn.Module,
    dataset,
    val_metric: float,
    val_metric_name: str,
    training_args: Dict[str, Any],
    is_finetune: bool,
) -> None:
    """Write a checkpoint that `load_backbone_adapter` can read.
    """
    ckpt = {
        "backbone_type": backbone_type,
        "backbone_config": asdict(backbone_config),
        "model_state_dict": model.state_dict(),
        "unified_dataset_metadata": {
            "V": int(dataset.V),
            "T": int(dataset.T),
            "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None),
        },
        "val_metric": float(val_metric),
        "val_metric_name": str(val_metric_name),
        "training_args": training_args,
        "is_finetune": bool(is_finetune),
    }
    if val_metric_name == "val_auroc":
        # Backwards compat: existing scripts that filter by `val_auroc`
        ckpt["val_auroc"] = float(val_metric)
    torch.save(ckpt, path)


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train STraTS / HyMaTE on the unified dataset."
    )

    # backbone selection
    parser.add_argument("--backbone", type=str, required=True, choices=["strats", "hymate"])
    parser.add_argument("--stage", type=str, default="finetune", choices=["finetune", "pretrain"])
    parser.add_argument("--load_ckpt_path", type=str, default=None,
                        help="Pretraining checkpoint to warm-start from. If set "
                             "(and --stage finetune), enables binary_head ∘ "
                             "forecast_head dispatch.")

    parser.add_argument("--task", type=str, default="mortality",
                        choices=["mortality", "los_gt3d", "los_gt7d", "readmission"],
                        help="Prediction task (label to train on)")

    # data
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--cohort", type=str, default="mimic_iii", choices=["mimic_iii", "mimic_iv"],
                        help="Cohort tag (stamped on the checkpoint metadata).")

    # output
    parser.add_argument("--output_dir", type=str, required=True)

    # arch
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=3,
                        help="HyMaTE-specific: mamba block depth")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention_dropout", type=float, default=0.2)
    parser.add_argument("--residual", type=str, default="average",
                        choices=["average", "layernorm"],
                        help="STraTS residual mode: average=(x+h)/2, layernorm=LayerNorm(x+h)")

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)

    # Pretraining-specific (only used when --stage pretrain)
    parser.add_argument("--pretrain_max_obs", type=int, default=1024,
                        help="Max events per pretraining sample (after sliding-window slicing)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir)
    logger.write("=" * 80)
    logger.write(f"train_external_backbone — backbone={args.backbone}, stage={args.stage}")
    logger.write("=" * 80)
    logger.write(f"  cohort={args.cohort}, seed={args.seed}, task={args.task}, dataset={args.dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load unified dataset
    dataset = _load_unified(args.dataset_path)
    logger.write(
        f"  Dataset: N={dataset.N}, V={dataset.V}, T={dataset.T}, "
        f"max_hours={dataset.max_hours}, "
        f"hash={getattr(dataset, 'var_to_ind_hash', 'n/a')}"
    )

    # Wrap as PretrainDataset if --stage pretrain.
    if args.stage == "pretrain":
        from tisae.data.pretrain_dataset import PretrainDataset
        if dataset.max_hours < 60:
            logger.write(
                f"  WARN: dataset.max_hours={dataset.max_hours} is short for "
                f"pretraining; expected ≥ 60h. Did you forget to rebuild the "
                f"unified pickle with max_hours=240?"
            )
        runtime_dataset = PretrainDataset.from_unified(
            dataset, max_obs=args.pretrain_max_obs, seed=args.seed, log_fn=logger.write
        )
    else:
        runtime_dataset = dataset

    # Build model. The model factory reads `args.stage` to set
    # `cfg.pretrain` / `cfg.finetune` correctly.
    model, bb_cfg = _build_model(args, dataset)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write(f"  Model: {n_params:,} params ({n_train:,} trainable)")

    # Load pretraining checkpoint if requested
    is_finetune_from_pretrain = False
    if args.load_ckpt_path is not None:
        logger.write(f"  Loading pretraining ckpt: {args.load_ckpt_path}")
        pt_ckpt = torch.load(args.load_ckpt_path, map_location="cpu", weights_only=False)
        pt_sd = pt_ckpt.get("model_state_dict", pt_ckpt)
        # Only copy keys that exist in the current (finetune) model
        current = model.state_dict()
        copied = 0
        for k, v in pt_sd.items():
            if k in current and current[k].shape == v.shape:
                current[k] = v
                copied += 1
        model.load_state_dict(current)
        logger.write(f"    copied {copied}/{len(pt_sd)} tensors from pretraining")
        is_finetune_from_pretrain = True

    # Save full config
    full_config = {
        "args": vars(args),
        "backbone_config": asdict(bb_cfg),
        "n_params": n_params,
        "n_trainable": n_train,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(full_config, f, default_flow_style=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # Training loop — bypass DataLoader to match HyMaTE's main.py pattern.
    # `runtime_dataset` is either the unified dataset (finetune) or a
    # PretrainDataset wrapper (pretrain). Both expose `splits[...]` and `get_batch(indices)`.
    train_indices = list(np.asarray(runtime_dataset.splits["train"]))
    task_label_key = TASK_BATCH_KEY[args.task]
    if args.task == 'readmission':
        y_readm = runtime_dataset.y_readmission
        train_indices = [i for i in train_indices if y_readm[i] >= 0]
    rng = random.Random(args.seed)

    is_pretrain = (args.stage == "pretrain")
    step_fn = _pretrain_step if is_pretrain else (
        lambda model, batch: _finetune_step(model, batch, label_key=task_label_key)
    )

    best_val_metric = math.inf if is_pretrain else -math.inf
    wait = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_metric": [],  # forecast_mse for pretrain, auroc for finetune
    }

    for epoch in range(1, args.epochs + 1):
        logger.write(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # Reshuffle each epoch
        rng.shuffle(train_indices)
        n = len(train_indices)
        bs = args.train_batch_size

        model.train()
        cum_loss = 0.0
        n_seen = 0
        pbar = tqdm(range(0, n, bs), desc="Training")
        for start in pbar:
            end = min(start + bs, n)
            idx = train_indices[start:end]
            batch = _make_batch(runtime_dataset, idx, device)
            loss = step_fn(model, batch)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            cum_loss += loss.item() * (end - start)
            n_seen += end - start
            pbar.set_postfix({"loss": f"{cum_loss / max(n_seen, 1):.4f}"})

        train_loss = cum_loss / max(n_seen, 1)
        history["train_loss"].append(train_loss)
        logger.write(f"  Train: loss={train_loss:.4f}")

        # Validation
        if is_pretrain:
            from tisae.training.pretrain_eval import evaluate_pretrain_split
            val = evaluate_pretrain_split(
                model, runtime_dataset, "val",
                eval_batch_size=args.eval_batch_size, device=device,
                n_repeats=3, log_fn=logger.write,
            )
            val_loss = val.get("forecast_mse", float("inf"))
            val_metric = val_loss
            history["val_loss"].append(val_loss)
            history["val_metric"].append(val_metric)
            logger.write(
                f"  Val:   forecast_mse={val_loss:.4f}  "
                f"n_predictions={val.get('n_predictions', 0)}"
            )
        else:
            val = _eval_split(model, runtime_dataset, "val", args.eval_batch_size, device,
                             label_key=task_label_key, task=args.task)
            val_loss = val.get("loss", float("inf"))
            val_metric = val.get("auroc", 0.0)
            history["val_loss"].append(val_loss)
            history["val_metric"].append(val_metric)
            logger.write(
                f"  Val:   loss={val_loss:.4f}  "
                f"auroc={val_metric:.4f}  "
                f"auprc={val.get('auprc', 0):.4f}"
            )

        # Improvement check (lower is better for pretrain, higher for finetune)
        improved = (val_metric < best_val_metric) if is_pretrain else (val_metric > best_val_metric)
        if improved:
            best_val_metric = val_metric
            wait = 0
            metric_label = "forecast_mse" if is_pretrain else "val_auroc"
            _save_uniform_checkpoint(
                output_dir / "best_model.pt",
                backbone_type=args.backbone,
                backbone_config=bb_cfg,
                model=model,
                dataset=dataset,  # always stamp the underlying unified dataset metadata
                val_metric=best_val_metric,
                val_metric_name=metric_label,
                training_args=vars(args),
                is_finetune=is_finetune_from_pretrain,
            )
            logger.write(f"  -> Best model saved ({metric_label}: {val_metric:.4f})")
        else:
            wait += 1
            if wait >= args.patience:
                logger.write("\nEarly stopping")
                break

    # Test evaluation
    logger.write("\n" + "=" * 80)
    logger.write("Test Evaluation")
    if is_pretrain:
        test = {"forecast_mse": float(best_val_metric)}
        logger.write(f"Pretrain: best val forecast_mse={best_val_metric:.4f}")
    else:
        test = _eval_split(model, runtime_dataset, "test", args.eval_batch_size, device,
                           label_key=task_label_key, task=args.task)
        logger.write(
            f"Test - loss={test['loss']:.4f}  "
            f"auroc={test.get('auroc', 0):.4f}  "
            f"auprc={test.get('auprc', 0):.4f}"
        )

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "best_val_metric": best_val_metric,
                "test_metrics": test,
                "history": history,
                "stage": args.stage,
            },
            f,
            indent=2,
        )
    logger.write(f"\nComplete! Results in: {output_dir}")


if __name__ == "__main__":
    main()
