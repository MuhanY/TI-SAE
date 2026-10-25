# tisae/training/train_backbone.py
"""
Train Binned-Time Transformer Backbone.

Usage:
    python -m tisae.training.train_backbone \
        --dataset_path data/unified_mimic3_48h.pkl \
        --output_dir experiments/discrete/mimic3_mortality/backbone \
        --task mortality \
        --hid_dim 128 --num_layers 4 --num_heads 8 \
        --dropout 0.3 --weight_decay 5e-3 \
        --batch_size 64 --eval_batch_size 64 \
        --lr 1e-4 --seed 42 --epochs 50 --patience 10 \
        --flash_attention --max_grad_norm 1.0

"""

import os
import argparse
import json
import yaml
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
from tisae.models.backbone import BinnedTimeTransformer, BackboneConfig

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


class Logger:
    def __init__(self, log_dir, filename='log.txt'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename

    def write(self, msg):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')


from tisae.utils.seeding import set_seed, worker_init_fn


TASK_LABEL_ATTR = {
    'mortality': 'y',
    'los_gt3d': 'y_los_gt3d',
    'los_gt7d': 'y_los_gt7d',
    'readmission': 'y_readmission',
}


class BackboneDataset(torch.utils.data.Dataset):
    def __init__(self, binned_dataset, split, task='mortality'):
        self.dataset = binned_dataset
        self._y_attr = TASK_LABEL_ATTR[task]
        indices = binned_dataset.splits[split]
        if task == 'readmission':
            y_arr = getattr(binned_dataset, self._y_attr)
            indices = [i for i in indices if y_arr[i] >= 0]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        y_arr = getattr(self.dataset, self._y_attr)
        return {
            'value_bins': torch.LongTensor(self.dataset.value_bins[real_idx]),
            'tau': torch.LongTensor(self.dataset.tau_bins[real_idx]),
            'tau_real': torch.FloatTensor(self.dataset.tau_real[real_idx]),
            'varis': torch.LongTensor(self.dataset.varis[real_idx]),
            'label': torch.FloatTensor([y_arr[real_idx]])
        }


from tisae.data.unified_dataset import collate_fn


def compute_pos_weight(dataset, task='mortality'):
    y_attr = TASK_LABEL_ATTR[task]
    y_arr = getattr(dataset, y_attr)
    train_inds = dataset.splits['train']
    if task == 'readmission':
        train_inds = [i for i in train_inds if y_arr[i] >= 0]
    y_train = y_arr[train_inds]
    return float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))


# =============================================================================
# Pretrain training / eval (2h value-bin forecast)
# =============================================================================


def pretrain_epoch(model, pretrain_ds, split, batch_size, optimizer, device,
                   max_grad_norm=1.0, training=True):
    """One epoch of 2h value-bin forecast pretraining."""
    import numpy as np
    if training:
        model.train()
    else:
        model.eval()

    indices = np.asarray(pretrain_ds.splits[split])
    if training:
        np.random.shuffle(indices)

    cum_loss = 0.0
    n_batches = 0
    n_total = (len(indices) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(indices), batch_size),
                desc="Pretrain" if training else "Pretrain-val", total=n_total)

    for start in pbar:
        end = min(start + batch_size, len(indices))
        batch = pretrain_ds.get_batch(indices[start:end])
        batch_dev = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            batch_dev["value_bins"], batch_dev["tau"],
            batch_dev["varis"], batch_dev["obs_mask"],
            forecast_value_bins=batch_dev["forecast_value_bins"],
            forecast_mask=batch_dev["forecast_mask"],
        )
        loss = outputs["loss"]

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        cum_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{cum_loss / n_batches:.4f}")

    return {"loss": cum_loss / max(n_batches, 1)}


def train_epoch(model, dataloader, optimizer, pos_weight, device, max_grad_norm=1.0):
    model.train()
    metrics = defaultdict(float)
    n_samples = 0
    all_labels, all_probs = [], []

    for batch in tqdm(dataloader, desc="Training"):
        value_bins = batch['value_bins'].to(device)
        tau = batch['tau'].to(device)
        varis = batch['varis'].to(device)
        obs_mask = batch['obs_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(value_bins, tau, varis, obs_mask)
        loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels, pos_weight=pos_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        B = value_bins.size(0)
        metrics['loss'] += loss.item() * B
        n_samples += B
        all_probs.extend(torch.sigmoid(outputs['logits']).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics['loss'] /= n_samples
    if HAVE_SKLEARN and len(set(all_labels)) > 1:
        metrics['auroc'] = roc_auc_score(all_labels, all_probs)
    return dict(metrics)


@torch.no_grad()
def eval_epoch(model, dataloader, pos_weight, device):
    model.eval()
    metrics = defaultdict(float)
    n_samples = 0
    all_labels, all_probs = [], []

    for batch in tqdm(dataloader, desc="Eval"):
        value_bins = batch['value_bins'].to(device)
        tau = batch['tau'].to(device)
        varis = batch['varis'].to(device)
        obs_mask = batch['obs_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(value_bins, tau, varis, obs_mask)
        loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels, pos_weight=pos_weight)

        B = value_bins.size(0)
        metrics['loss'] += loss.item() * B
        n_samples += B
        all_probs.extend(torch.sigmoid(outputs['logits']).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics['loss'] /= n_samples
    if HAVE_SKLEARN and len(set(all_labels)) > 1:
        metrics['auroc'] = roc_auc_score(all_labels, all_probs)
        metrics['auprc'] = average_precision_score(all_labels, all_probs)
    return dict(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./experiments/backbone')
    parser.add_argument('--dataset_path', type=str, default='data/unified_mimic3_48h.pkl')
    parser.add_argument('--stage', type=str, default='supervised',
                        choices=['supervised', 'pretrain', 'finetune'],
                        help='Training stage: supervised (default), pretrain (2h forecast), '
                             'finetune (supervised from pretrain checkpoint)')
    parser.add_argument('--load_ckpt_path', type=str, default=None,
                        help='Path to pretrain checkpoint for finetune stage')
    parser.add_argument('--task', type=str, default='mortality',
                        choices=['mortality', 'los_gt3d', 'los_gt7d', 'readmission'],
                        help='Prediction task (label to train on)')
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--flash_attention', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain_max_obs', type=int, default=1024,
                        help='Max events per sample for pretrain sliding window')
    args = parser.parse_args()

    device = torch.device('cuda')
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(str(exp_dir))
    logger.write('=' * 80)
    logger.write('Transformer backbone training')
    logger.write('=' * 80)

    set_seed(args.seed)

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)

    is_pretrain = args.stage == 'pretrain'
    is_finetune = args.stage == 'finetune'

    logger.write(f'Stage: {args.stage}')
    logger.write(f'Dataset: N={len(dataset.y)}, V={dataset.V}, T={dataset.T}')

    # Dataset metadata for checkpoint
    ds_meta = {
        'V': dataset.V,
        'T': dataset.T,
        'var_to_ind_hash': getattr(dataset, 'var_to_ind_hash', None),
    }

    # ----------------------------------------------------------------
    # Pretrain stage: wrap in PretrainDataset for sliding-window 2h forecast
    # ----------------------------------------------------------------
    pretrain_ds = None
    if is_pretrain:
        from tisae.data.pretrain_dataset import PretrainDataset
        pretrain_ds = PretrainDataset.from_unified(
            dataset, max_obs=args.pretrain_max_obs, seed=args.seed)
        logger.write(f'PretrainDataset: train={len(pretrain_ds.splits["train"])}, '
                     f'val={len(pretrain_ds.splits["val"])}')

    # ----------------------------------------------------------------
    # Supervised / finetune: standard DataLoader
    # ----------------------------------------------------------------
    train_loader, val_loader, test_loader = None, None, None
    pos_weight = None
    if not is_pretrain:
        train_dataset = BackboneDataset(dataset, 'train', task=args.task)
        val_dataset = BackboneDataset(dataset, 'val', task=args.task)
        test_dataset = BackboneDataset(dataset, 'test', task=args.task)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=args.num_workers,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=args.num_workers,
                                pin_memory=True, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=True, worker_init_fn=worker_init_fn)
        pos_weight = torch.tensor([compute_pos_weight(dataset, task=args.task)], device=device)
        logger.write(f'Task: {args.task}, pos_weight: {pos_weight.item():.2f}')

    # ----------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------
    config = BackboneConfig(
        V=dataset.V, T=dataset.T, n_value_bins=dataset.n_value_bins,
        hid_dim=args.hid_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, dropout=args.dropout,
        use_flash_attention=args.flash_attention,
        pretrain=is_pretrain,
        pos_class_weight=pos_weight.item() if pos_weight is not None else 1.0,
    )
    model = BinnedTimeTransformer(config).to(device)
    logger.write(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Finetune: load pretrain checkpoint (selective key copy)
    if is_finetune and args.load_ckpt_path:
        logger.write(f'Loading pretrain checkpoint: {args.load_ckpt_path}')
        pt_ckpt = torch.load(args.load_ckpt_path, map_location='cpu', weights_only=False)
        pt_sd = pt_ckpt.get('model_state_dict', pt_ckpt)
        current = model.state_dict()
        copied = 0
        for k, v in pt_sd.items():
            if k in current and current[k].shape == v.shape:
                current[k] = v
                copied += 1
        model.load_state_dict(current)
        logger.write(f'  Copied {copied}/{len(pt_sd)} tensors from pretrain '
                     f'(skipped: forecast_head, pred_head if shape mismatch)')

    # Save config
    from dataclasses import asdict
    full_config = {
        'model_config': asdict(config),
        'training_config': {**vars(args), 'pos_weight': pos_weight.item() if pos_weight is not None else 1.0},
    }
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(full_config, f, default_flow_style=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    best_val_metric = float('inf') if is_pretrain else 0.0  # pretrain: lower loss = better
    metric_name = 'forecast_loss' if is_pretrain else 'auroc'
    wait = 0
    history = defaultdict(list)

    def _save_ckpt(epoch, val_metric):
        torch.save({
            'backbone_type': 'tisae',
            'backbone_config': asdict(config),
            'model_state_dict': model.state_dict(),
            'unified_dataset_metadata': ds_meta,
            'val_metric': val_metric,
            'val_metric_name': metric_name,
            'training_args': vars(args),
            'is_finetune': is_finetune,
            'epoch': epoch,
        }, exp_dir / 'best_model.pt')

    for epoch in range(1, args.epochs + 1):
        logger.write(f'\n--- Epoch {epoch}/{args.epochs} ---')

        if is_pretrain:
            train_m = pretrain_epoch(model, pretrain_ds, 'train', args.batch_size,
                                     optimizer, device, max_grad_norm=args.max_grad_norm)
            logger.write(f"Train - forecast_loss: {train_m['loss']:.4f}")
            with torch.no_grad():
                val_m = pretrain_epoch(model, pretrain_ds, 'val', args.eval_batch_size,
                                       None, device, training=False)
            logger.write(f"Val   - forecast_loss: {val_m['loss']:.4f}")
            val_metric = val_m['loss']
            improved = val_metric < best_val_metric
        else:
            train_m = train_epoch(model, train_loader, optimizer, pos_weight, device,
                                  max_grad_norm=args.max_grad_norm)
            logger.write(f"Train - loss: {train_m['loss']:.4f}, auroc: {train_m.get('auroc', 0):.4f}")
            val_m = eval_epoch(model, val_loader, pos_weight, device)
            logger.write(f"Val   - loss: {val_m['loss']:.4f}, auroc: {val_m.get('auroc', 0):.4f}")
            val_metric = float(val_m.get('auroc', 0))
            improved = val_metric > best_val_metric

        for k, v in train_m.items():
            history[f'train_{k}'].append(v)
        for k, v in val_m.items():
            history[f'val_{k}'].append(v)

        scheduler.step()

        if improved:
            best_val_metric = val_metric
            wait = 0
            _save_ckpt(epoch, val_metric)
            logger.write(f"  -> Best ({metric_name}: {val_metric:.4f})")
        else:
            wait += 1
            if wait >= args.patience:
                logger.write("\nEarly stopping")
                break

    # Test evaluation (supervised / finetune only)
    if not is_pretrain:
        logger.write('\n' + '=' * 80)
        logger.write('Test Evaluation')
        ckpt = torch.load(exp_dir / 'best_model.pt', map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        test_metrics = eval_epoch(model, test_loader, pos_weight, device)
        logger.write(f"Test - loss: {test_metrics['loss']:.4f}, auroc: {test_metrics.get('auroc', 0):.4f}, "
                     f"auprc: {test_metrics.get('auprc', 0):.4f}")
    else:
        test_metrics = {}

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump({'best_val_metric': best_val_metric, 'metric_name': metric_name,
                   'test_metrics': test_metrics, 'stage': args.stage}, f, indent=2)
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)
    logger.write(f"\nComplete! Results in: {exp_dir}")


if __name__ == '__main__':
    main()
