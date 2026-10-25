"""Train the BAT (Bi-Axial Transformer) backbone on the unified dataset.

Usage
-----

BAT — supervised only:

    python -m tisae.training.train_bat \\
        --stage finetune \\
        --dataset_path data/unified_mimic3_48h.pkl \\
        --cohort mimic_iii \\
        --output_dir experiments/bat/mimic3_mortality/backbone_mimic_iii \\
        --embed_out 16 --layers 1 --heads 1 \\
        --dropout 0.4 --attn_dropout 0.2 --pooling mean \\
        --train_batch_size 4 --lr 1e-4 --epochs 50 --patience 10

BAT — pretrain -> finetune:

    python -m tisae.training.train_bat \\
        --stage pretrain \\
        --dataset_path data/unified_mimic3_240h.pkl \\
        --cohort mimic_iii \\
        --output_dir experiments/bat/mimic3_mortality/backbone_pretrain \\
        --embed_out 16 --layers 1 --heads 1 \\
        --train_batch_size 16 --lr 5e-4 --epochs 50 --patience 10 \\
        --pretrain_max_obs 1024

    python -m tisae.training.train_bat \\
        --stage finetune \\
        --dataset_path data/unified_mimic3_48h.pkl \\
        --cohort mimic_iii \\
        --output_dir experiments/bat/mimic3_mortality/backbone_finetune \\
        --load_ckpt_path experiments/bat/mimic3_mortality/backbone_pretrain/best_model.pt \\
        --embed_out 16 --layers 1 --heads 1 \\
        --train_batch_size 4 --lr 5e-5 --epochs 50 --patience 10

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
import yaml
from tqdm import tqdm

from tisae.models.external.bat import BAT, BATConfig
from tisae.utils.seeding import set_seed

import tisae.models.bat_adapter  # noqa: F401  (trigger adapter registration)

from tisae.training.train_external_backbone import (
    Logger,
    _load_unified,
    _compute_pos_class_weight,
    _make_batch,
    _save_uniform_checkpoint,
    TASK_BATCH_KEY,
)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


# =============================================================================
# Model factory
# =============================================================================


def _build_bat_model(args, dataset):
    cfg = BATConfig(
        V=dataset.V,
        T=dataset.T,
        embed_out=args.embed_out,
        # sensor_encoding_out auto-computed from V (default -1)
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        use_mask=args.use_mask,
        pooling=args.pooling,
        pretrain=(args.stage == "pretrain"),
        finetune=(args.stage == "finetune" and args.load_ckpt_path is not None),
        pos_class_weight=_compute_pos_class_weight(dataset),
    )
    model = BAT(cfg)
    return model, cfg


# =============================================================================
# Training steps
# =============================================================================


def _finetune_step_bat(model, batch: Dict[str, torch.Tensor], label_key: str = "labels"):
    loss, _ = model(
        batch["values_zscore"],
        batch["times_norm"],
        batch["varis"],
        batch["obs_mask"],
        labels=batch[label_key],
        tau=batch["tau"],
    )
    return loss


def _pretrain_step_bat(model, batch: Dict[str, torch.Tensor]):
    return model(
        batch["values_zscore"],
        batch["times_norm"],
        batch["varis"],
        batch["obs_mask"],
        forecast_values=batch["forecast_values"],
        forecast_mask=batch["forecast_mask"],
        tau=batch["tau"],
    )


# =============================================================================
# Evaluation
# =============================================================================


@torch.no_grad()
def _eval_split_bat(
    model, dataset, split: str, batch_size: int, device: torch.device,
    label_key: str = "labels", task: str = "mortality",
):
    model.eval()
    indices = np.asarray(dataset.splits[split])
    if task == "readmission":
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
            tau=batch["tau"],
        )
        cum_loss += loss.item() * (end - start)
        n_seen += end - start

        probs = model(
            batch["values_zscore"],
            batch["times_norm"],
            batch["varis"],
            batch["obs_mask"],
            labels=None,
            tau=batch["tau"],
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
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BAT backbone.")

    parser.add_argument("--stage", type=str, default="finetune",
                        choices=["finetune", "pretrain"])
    parser.add_argument("--load_ckpt_path", type=str, default=None)
    parser.add_argument("--task", type=str, default="mortality",
                        choices=["mortality", "los_gt3d", "los_gt7d", "readmission"])

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--cohort", type=str, default="mimic_iii",
                        choices=["mimic_iii", "mimic_iv"])
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--embed_out", type=int, default=16,
                        help="Value+mask embedding dim (primary size knob). "
                             "sensor_encoding_out auto-computed from V. "
                             "E = embed_out + sensor_encoding_out, D = 2*E.")
    parser.add_argument("--layers", type=int, default=1,
                        help="Internal depth per attention module")
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--use_mask", action="store_true", default=False,
                        help="Mask empty timepoints in time attention (BAT default: False)")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--pretrain_max_obs", type=int, default=1024)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_dir)
    logger.write("=" * 80)
    logger.write(f"train_bat — stage={args.stage}")
    logger.write("=" * 80)
    logger.write(
        f"  cohort={args.cohort}, seed={args.seed}, task={args.task}, "
        f"dataset={args.dataset_path}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = _load_unified(args.dataset_path)
    logger.write(
        f"  Dataset: N={dataset.N}, V={dataset.V}, T={dataset.T}, "
        f"max_hours={dataset.max_hours}, "
        f"hash={getattr(dataset, 'var_to_ind_hash', 'n/a')}"
    )

    if args.stage == "pretrain":
        from tisae.data.pretrain_dataset import PretrainDataset
        if dataset.max_hours < 60:
            logger.write(
                f"  WARN: dataset.max_hours={dataset.max_hours} too short for "
                f"pretraining; expected >= 60h."
            )
        runtime_dataset = PretrainDataset.from_unified(
            dataset, max_obs=args.pretrain_max_obs, seed=args.seed,
            log_fn=logger.write,
        )
    else:
        runtime_dataset = dataset

    model, bb_cfg = _build_bat_model(args, dataset)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write(
        f"  Model: BAT D={bb_cfg.D}, E={bb_cfg.E}, layers={bb_cfg.layers}, "
        f"heads={bb_cfg.heads} | {n_params:,} params ({n_train:,} trainable)"
    )

    is_finetune_from_pretrain = False
    if args.load_ckpt_path is not None:
        logger.write(f"  Loading pretrain ckpt: {args.load_ckpt_path}")
        pt_ckpt = torch.load(args.load_ckpt_path, map_location="cpu", weights_only=False)
        pt_sd = pt_ckpt.get("model_state_dict", pt_ckpt)
        current = model.state_dict()
        copied = 0
        for k, v in pt_sd.items():
            if k in current and current[k].shape == v.shape:
                current[k] = v
                copied += 1
        model.load_state_dict(current)
        logger.write(f"    copied {copied}/{len(pt_sd)} tensors from pretraining")
        is_finetune_from_pretrain = True

    full_config = {
        "args": vars(args),
        "backbone_config": asdict(bb_cfg),
        "n_params": n_params,
        "n_trainable": n_train,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(full_config, f, default_flow_style=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    task_label_key = TASK_BATCH_KEY[args.task]
    train_indices = list(np.asarray(runtime_dataset.splits["train"]))
    if args.task == "readmission":
        y_readm = runtime_dataset.y_readmission
        train_indices = [i for i in train_indices if y_readm[i] >= 0]
    rng = random.Random(args.seed)

    is_pretrain = (args.stage == "pretrain")
    step_fn = _pretrain_step_bat if is_pretrain else (
        lambda model, batch: _finetune_step_bat(model, batch, label_key=task_label_key)
    )

    best_val_metric = math.inf if is_pretrain else -math.inf
    wait = 0
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_metric": [],
    }

    for epoch in range(1, args.epochs + 1):
        logger.write(f"\n--- Epoch {epoch}/{args.epochs} ---")
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
            val = _eval_split_bat(
                model, runtime_dataset, "val", args.eval_batch_size, device,
                label_key=task_label_key, task=args.task,
            )
            val_loss = val.get("loss", float("inf"))
            val_metric = val.get("auroc", 0.0)
            history["val_loss"].append(val_loss)
            history["val_metric"].append(val_metric)
            logger.write(
                f"  Val:   loss={val_loss:.4f}  "
                f"auroc={val_metric:.4f}  "
                f"auprc={val.get('auprc', 0):.4f}"
            )

        improved = (val_metric < best_val_metric) if is_pretrain else (val_metric > best_val_metric)
        if improved:
            best_val_metric = val_metric
            wait = 0
            metric_label = "forecast_mse" if is_pretrain else "val_auroc"
            _save_uniform_checkpoint(
                output_dir / "best_model.pt",
                backbone_type="bat",
                backbone_config=bb_cfg,
                model=model,
                dataset=dataset,
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

    logger.write("\n" + "=" * 80)
    logger.write("Test Evaluation")
    if is_pretrain:
        test = {"forecast_mse": float(best_val_metric)}
        logger.write(f"Pretrain: best val forecast_mse={best_val_metric:.4f}")
    else:
        test = _eval_split_bat(
            model, runtime_dataset, "test", args.eval_batch_size, device,
            label_key=task_label_key, task=args.task,
        )
        logger.write(
            f"Test - loss={test['loss']:.4f}  "
            f"auroc={test.get('auroc', 0):.4f}  "
            f"auprc={test.get('auprc', 0):.4f}"
        )

    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {"best_val_metric": best_val_metric, "test_metrics": test,
             "history": history, "stage": args.stage},
            f, indent=2,
        )
    logger.write(f"\nComplete! Results in: {output_dir}")


if __name__ == "__main__":
    main()
