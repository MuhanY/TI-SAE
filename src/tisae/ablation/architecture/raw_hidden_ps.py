"""Prediction selector on raw hidden states (dictionary/selector isolation).

Applies the same prediction-aware selector directly to raw hidden states.
Raw H has no concept vocabulary, so the only candidate unit is the observed event (t,v) cell.
This trains the same prediction selector over event candidates,
with each event's feature being a "degenerate island".
Keep/drop is done by masking event embeddings -> pool -> predict (no SAE).

Usage:
    python -m tisae.ablation.architecture.raw_hidden_ps \
        --backbone experiments/discrete/mimic3_mortality/backbone/best_model.pt \
        --dataset_path data/unified_mimic3_48h.pkl \
        --output_dir experiments/ablation/structural/raw_hidden/ps \
        --seed 42
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from tisae.models.pred_selector import (
    PredSelector, PredSelectorConfig, pred_selector_losses,
)
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.training.train_tisae import Logger, TISAEDataset
from tisae.utils.seeding import set_seed, worker_init_fn


@torch.no_grad()
def assemble_cell_features(H, obs_mask, varis, tau, T, V):
    """Per-event 'degenerate island' features -> [B, L, 1, 1, 6+2V].

    Scalars [6]: [t/T, sigma=0, |H_cell| (bn), |H_cell| (bn), gate=1, prior=0].
    Variable profile [V]: one-hot(v).
    Observation profile [V]: one-hot(v).
    """
    B, L, D = H.shape
    valid = obs_mask.float()                                  # [B,L]
    pos = (tau.float() / float(T)).clamp(0, 1)                # [B,L]
    sigma0 = torch.zeros_like(pos)
    mag = H.norm(dim=-1)                                      # [B,L]
    m = (mag * valid).amax(dim=1, keepdim=True).clamp_min(1e-8)
    mag_n = (mag / m) * valid
    gate = valid
    prior = torch.zeros_like(pos)
    scalars = torch.stack([pos, sigma0, mag_n, mag_n, gate, prior], dim=-1)  # [B,L,6]
    v_oh = F.one_hot(varis.long().clamp(0, V - 1), V).float() * valid.unsqueeze(-1)
    feats = torch.cat([scalars, v_oh, v_oh], dim=-1)         # [B,L,6+2V]
    return feats.unsqueeze(2).unsqueeze(2)                    # [B,L,1,1,6+2V]


def ps_step_raw_hidden(adapter, selector, batch, T, V, config, device, soft_tau=1.0):
    batch_dev = {k: v.to(device) for k, v in batch.items()}
    obs_mask = batch_dev['obs_mask']
    varis = batch_dev['varis']
    tau = batch_dev['tau']
    B, L = obs_mask.shape

    with torch.no_grad():
        _, _, H = adapter.extract_hidden(batch_dev, return_event_h=True)  # [B,L,D]
        if adapter.extract_layer != -1:
            out_full = adapter.forward(batch_dev, return_pooled=True)
            h_pool = out_full['pooled']
            z_teacher = out_full['logits']
        else:
            h_pool, _ = adapter.pool_events(H, obs_mask)
            z_teacher = adapter.predict_from_pooled(h_pool)
        z_full = z_teacher  # keeping all events == full backbone
        features = assemble_cell_features(H, obs_mask, varis, tau, T, V)

    s_pred = selector(features, h_pool.detach())             # [B,L,1,1]
    s_flat = s_pred.view(B, -1)                              # [B,L]
    s_flat = s_flat.masked_fill(obs_mask < 0.5, float('-inf'))
    weights = (F.softmax(s_flat / soft_tau, dim=-1) * config.n_global_pred).nan_to_num(0.0)

    H_keep = H * weights.unsqueeze(-1)
    pooled_k, _ = adapter.pool_events(H_keep, obs_mask)
    z_keep = adapter.predict_from_pooled(pooled_k)

    H_drop = H * (1 - weights).clamp_min(0).unsqueeze(-1)
    pooled_d, _ = adapter.pool_events(H_drop, obs_mask)
    z_drop = adapter.predict_from_pooled(pooled_d)

    return pred_selector_losses(z_keep, z_drop, z_full, z_teacher, s_pred, None, config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--d_sel", type=int, default=64)
    ap.add_argument("--mlp_hidden", type=int, default=64)
    ap.add_argument("--n_global_pred", type=int, default=64)
    ap.add_argument("--lambda_logit", type=float, default=1.0)
    ap.add_argument("--lambda_comp", type=float, default=0.1)
    ap.add_argument("--delta_comp", type=float, default=0.5)
    ap.add_argument("--soft_tau", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=52)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    logger = Logger(out)

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V,
          "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    adapter = load_backbone_adapter(args.backbone, device=str(device), dataset_meta=dm)
    adapter.eval()
    T, V, D = dataset.T, dataset.V, adapter.D
    n_feat = 6 + 2 * V

    cfg = PredSelectorConfig(
        d_sel=args.d_sel, mlp_hidden=args.mlp_hidden, n_global_pred=args.n_global_pred,
        lambda_logit=args.lambda_logit, lambda_comp=args.lambda_comp,
        delta_comp=args.delta_comp, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
    )
    selector = PredSelector(cfg, n_feat, D, C_eff=1).to(device)
    logger.write(f"raw-hidden PS: {sum(p.numel() for p in selector.parameters()):,} params, "
                 f"n_feat={n_feat}, D={D}, T={T}, V={V}")

    train_loader = torch.utils.data.DataLoader(
        TISAEDataset(dataset, "train"), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = torch.utils.data.DataLoader(
        TISAEDataset(dataset, "val"), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    opt = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    with open(out / 'config.yaml', 'w') as f:
        yaml.dump({'pred_selector_config': asdict(cfg), 'model_type': 'raw_hidden',
                   'backbone': args.backbone, 'n_island_features': n_feat,
                   'D': D, 'T': T, 'V': V, 'seed': args.seed}, f, default_flow_style=False)

    best, patience = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        selector.train(); tr = defaultdict(float); ntr = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            losses = ps_step_raw_hidden(adapter, selector, batch, T, V, cfg, device, args.soft_tau)
            loss = losses['L_sel']
            if torch.isnan(loss) or torch.isinf(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0); opt.step()
            B = batch['value_bins'].size(0); ntr += B
            for k, v in losses.items():
                if isinstance(v, torch.Tensor): tr[k] += v.item() * B
        selector.eval(); va = defaultdict(float); nva = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                losses = ps_step_raw_hidden(adapter, selector, batch, T, V, cfg, device, args.soft_tau)
                B = batch['value_bins'].size(0); nva += B
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor): va[k] += v.item() * B
        logger.write(f"\n--- Epoch {epoch}/{args.epochs} ---")
        logger.write("Val:   " + " | ".join(f"{k}={v/max(nva,1):.4f}" for k, v in sorted(va.items())))
        vl = va.get('L_sel', float('inf')) / max(nva, 1)
        if vl < best:
            best = vl; patience = 0
            torch.save({'selector_state_dict': selector.state_dict(),
                        'epoch': epoch, 'val_loss': vl}, out / 'best_model.pt')
            logger.write(f"  * New best val L_sel={vl:.6f}")
        else:
            patience += 1
            if patience >= args.patience:
                logger.write(f"Early stopping at epoch {epoch}"); break
    logger.write(f"\nBest val L_sel: {best:.6f}")


if __name__ == "__main__":
    main()
