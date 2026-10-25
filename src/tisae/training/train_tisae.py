# tisae/training/train_tisae.py
"""
TISAE Training

Usage:
    python -m tisae.training.train_tisae \
        --backbone_checkpoint experiments/discrete/mimic3_mortality/backbone/best_model.pt \
        --dataset_path data/unified_mimic3_48h.pkl \
        --output_dir experiments/discrete/mimic3_mortality/tisae \
        --K 1280 --batch_size 32 --lr 1e-4 --epochs 100 --patience 15 \
        --gradient_checkpointing --diag_every 5
"""

import os
import math
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from tisae.models.tisae import TISAE, TISAEConfig
from tisae.models.backbone_adapter import (
    BackboneAdapter,
    load_backbone_adapter,
)
from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
from tisae.utils.seeding import set_seed, worker_init_fn
from tisae.utils.grid_ops import grid_to_H
from tisae.models.ontology import create_ontology_mapping


# =============================================================================
# Logger
# =============================================================================

class Logger:
    def __init__(self, log_dir, filename='log.txt'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename

    def write(self, msg: str):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')


# =============================================================================
# Dataset
# =============================================================================

class TISAEDataset(Dataset):
    def __init__(self, dataset: UnifiedTimeSeriesDataset, split: str):
        self.dataset = dataset
        self.indices = dataset.splits[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.dataset.get_item(real_idx)


from tisae.data.unified_dataset import collate_fn


# =============================================================================
# Gradient Checkpointing
# =============================================================================

class CheckpointedModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if self.training and (
            any(a.requires_grad for a in args if isinstance(a, torch.Tensor))
            or any(p.requires_grad for p in self.module.parameters())
        ):
            return checkpoint(self._forward_fn, *args, use_reentrant=False, **kwargs)
        return self.module(*args, **kwargs)

    def _forward_fn(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def apply_gradient_checkpointing(sae: TISAE):
    sae.encoder = CheckpointedModule(sae.encoder)
    sae.island_params = CheckpointedModule(sae.island_params)
    sae.decoder = CheckpointedModule(sae.decoder)
    return sae


# =============================================================================
# TISAETrainer (Frozen Backbone + Trainable TISAE)
# =============================================================================

class TISAETrainer(nn.Module):
    """
    Training wrapper: frozen backbone (via adapter) + trainable TISAE.
    Subtracts per-variable population mean before encoding.
    """

    def __init__(self, adapter: BackboneAdapter, sae: TISAE,
                 T: int, V: int,
                 extract_layer: int = -1,
                 lambda_pred: float = 0.0,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.adapter = adapter
        self.sae = sae
        self.T = T
        self.V = V
        self.extract_layer = extract_layer
        self.lambda_pred = lambda_pred
        self.gradient_checkpointing = gradient_checkpointing

        # Freeze the underlying backbone via the adapter contract.
        # `freeze()` ensures `train()` keeps the module in eval mode.
        self.adapter.freeze()

    @property
    def backbone(self):
        """Backwards-compat alias: legacy diagnostic code paths reference
        `model.backbone` for things like `.eval()` and `.parameters()`.
        Route those through the adapter so existing call sites still work.
        """
        return self.adapter

    def _pred_from_H_event(self, H_event, obs_mask, p_target):
        """Compute BCE prediction loss from un-residualized event-level H.

        Used by the L_pred path when extract_layer == -1 (last layer) and
        the SAE recon has already been de-residualized in event space.
        """
        pooled, _ = self.adapter.pool_events(H_event, obs_mask)
        logit = self.adapter.predict_from_pooled(pooled)
        return F.binary_cross_entropy_with_logits(logit, p_target)

    def _pred_from_grid(self, H_hat_grid_res, tau, varis, obs_mask, obs_grid, p_target):
        """Compute BCE prediction loss by routing through predict_from_grid.

        Used when extract_layer != -1: the SAE reconstruction is at an
        intermediate layer, so predict_from_grid runs the remaining
        backbone layers before pool + predict.
        """
        batch_mini = {"tau": tau, "varis": varis, "obs_mask": obs_mask}
        logit = self.adapter.predict_from_grid(H_hat_grid_res, batch_mini, obs_grid)
        return F.binary_cross_entropy_with_logits(logit, p_target)

    @staticmethod
    def per_token_centered_nmse(H_hat, H, token_mask=None, eps=1e-8, max_nmse=10.0):
        Hc = H - H.mean(dim=-1, keepdim=True)
        Hhatc = H_hat - H_hat.mean(dim=-1, keepdim=True)
        err = (Hhatc - Hc).pow(2).sum(dim=-1)
        denom = Hc.pow(2).sum(dim=-1).clamp_min(eps)
        nmse = (err / denom).clamp_max(max_nmse)
        if token_mask is not None:
            nmse = nmse * token_mask
            return nmse.sum() / token_mask.sum().clamp_min(1.0)
        return nmse.mean()

    def forward(self, batch_or_vb, tau=None, tau_real=None, varis=None,
                obs_mask=None, compute_fidelity=False):
        """Polymorphic forward.

        Two call shapes:
            forward(batch_dict, compute_fidelity=False)
            forward(value_bins, tau, tau_real, varis, obs_mask, compute_fidelity=False)

        The dict path is the canonical shape (so STraTS/HyMaTE
        adapters can read `values_zscore` / `times_norm` from the batch).
        """
        if isinstance(batch_or_vb, dict):
            batch = batch_or_vb
            tau = batch["tau"]
            varis = batch["varis"]
            obs_mask = batch["obs_mask"]
        else:
            batch = {
                "value_bins": batch_or_vb,
                "tau": tau,
                "tau_real": tau_real,
                "varis": varis,
                "obs_mask": obs_mask,
            }

        # Frozen backbone — extract per-event H AND grid via the adapter
        H_grid, obs_grid, H = self.adapter.extract_hidden(batch, return_event_h=True)

        # Subtract per-variable population mean (H_grid_mean is [V, D]).
        # The adapter also stores H_grid_mean (so predict_from_grid can
        # de-residualize), but for the encoder input we keep the explicit
        # subtract here to preserve bit-equivalence with the legacy code.
        if hasattr(self, 'H_grid_mean'):
            H_grid_input = H_grid - self.H_grid_mean.unsqueeze(0).unsqueeze(1)
            H_grid_input = H_grid_input * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

        # TISAE forward
        sae_out = self.sae(H_grid_input, obs_grid)
        H_hat_grid_res = sae_out['H_hat_grid']
        del H_grid, H_grid_input  # free [B,T,V,D] tensors

        # Convert to event level — add mean at event level
        H_hat_res = grid_to_H(H_hat_grid_res, tau, varis)  # [B, L, D]
        if hasattr(self, 'H_grid_mean'):
            H_hat = H_hat_res + self.H_grid_mean[varis]
        else:
            H_hat = H_hat_res

        # H-level reconstruction (for monitoring)
        L_recon_H = self.per_token_centered_nmse(H_hat, H, obs_mask)

        # Prediction loss
        L_pred = torch.tensor(0.0, device=H.device)
        H_teacher = torch.tensor(0.0, device=H.device)
        p_target = None
        if self.lambda_pred > 0:
            with torch.no_grad():
                if self.extract_layer != -1:
                    # Teacher ALWAYS uses the full backbone (all layers),
                    # even when extract_layer selects an intermediate layer
                    # for the TISAE grid. adapter.forward() is unaffected by
                    # extract_layer — it always runs the full backbone.
                    y_orig = self.adapter.forward(batch)['logits']
                else:
                    pooled_orig, _ = self.adapter.pool_events(H, obs_mask)
                    y_orig = self.adapter.predict_from_pooled(pooled_orig)
                p_target = torch.sigmoid(y_orig)
            if self.extract_layer != -1:
                # SAE reconstruction is at the intermediate layer;
                # route through predict_from_grid to run remaining layers
                if self.gradient_checkpointing and self.training:
                    L_pred = checkpoint(
                        self._pred_from_grid, H_hat_grid_res, tau, varis,
                        obs_mask, obs_grid, p_target, use_reentrant=False)
                else:
                    L_pred = self._pred_from_grid(
                        H_hat_grid_res, tau, varis, obs_mask, obs_grid, p_target)
            elif self.gradient_checkpointing and self.training:
                L_pred = checkpoint(self._pred_from_H_event, H_hat, obs_mask, p_target,
                                    use_reentrant=False)
            else:
                L_pred = self._pred_from_H_event(H_hat, obs_mask, p_target)
            H_teacher = F.binary_cross_entropy(p_target, p_target)

        # L_auxk_pred: dead latents improve prediction
        L_auxk_pred = torch.tensor(0.0, device=H.device)
        H_hat_grid_aux = sae_out.get('H_hat_grid_aux')
        if self.lambda_pred > 0 and H_hat_grid_aux is not None:
            if self.extract_layer != -1:
                # Route through predict_from_grid for remaining layers
                H_aux_grid = H_hat_grid_res.detach() + H_hat_grid_aux
                del H_hat_grid_aux
                if self.gradient_checkpointing and self.training:
                    L_auxk_pred = checkpoint(
                        self._pred_from_grid, H_aux_grid, tau, varis,
                        obs_mask, obs_grid, p_target, use_reentrant=False)
                else:
                    L_auxk_pred = self._pred_from_grid(
                        H_aux_grid, tau, varis, obs_mask, obs_grid, p_target)
            else:
                H_aux = grid_to_H(H_hat_grid_res, tau, varis).detach() \
                      + grid_to_H(H_hat_grid_aux, tau, varis)
                if hasattr(self, 'H_grid_mean'):
                    H_aux = H_aux + self.H_grid_mean[varis]
                del H_hat_grid_aux
                if self.gradient_checkpointing and self.training:
                    L_auxk_pred = checkpoint(self._pred_from_H_event, H_aux, obs_mask, p_target,
                                             use_reentrant=False)
                else:
                    L_auxk_pred = self._pred_from_H_event(H_aux, obs_mask, p_target)

        # Proportional AuxK weighting
        cfg = self.sae.cfg

        L_auxk = sae_out.get('L_auxk', torch.tensor(0.0, device=H.device))
        w_total = cfg.lambda_recon + self.lambda_pred
        if w_total > 0 and self.lambda_pred > 0:
            L_auxk_combined = (cfg.lambda_recon * L_auxk + self.lambda_pred * L_auxk_pred) / w_total
        else:
            L_auxk_combined = L_auxk

        # Total loss
        loss = sae_out['loss_total'] + self.lambda_pred * L_pred + cfg.lambda_auxk * L_auxk_combined

        # cos_H monitoring
        cos_H = torch.tensor(0.0, device=H.device)
        if compute_fidelity:
            with torch.no_grad():
                cos_all = F.cosine_similarity(H_hat, H, dim=-1)
                cos_H = (cos_all * obs_mask).sum() / obs_mask.sum().clamp_min(1.0)

        return {
            'loss': loss,
            'L_recon': sae_out['L_recon'],
            'L_recon_H': L_recon_H,
            'L_pred': L_pred,
            'KL_pred': L_pred - H_teacher,
            'L_auxk_combined': L_auxk_combined,
            'L_auxk_pred': L_auxk_pred,
            'KL_auxk_pred': L_auxk_pred - H_teacher,
            'cos_H': cos_H,
            'H': H,
            'H_hat': H_hat,
            **{k: v for k, v in sae_out.items() if k not in ('H_hat_grid', 'H_hat_grid_aux', 'loss_total')},
        }


# =============================================================================
# H_grid Mean Precomputation
# =============================================================================

@torch.no_grad()
def precompute_H_grid_mean(adapter, dataloader, device, T, V, extract_layer=-1,
                            max_batches=100):
    """Compute per-variable population mean from training data.

    Returns [V, D] mean across all time steps and samples.
    """
    adapter.eval()
    H_sum = None
    obs_count = None
    n = 0

    for batch in dataloader:
        if n >= max_batches:
            break
        # Move ALL keys to device — adapters for CVE backbones (STraTS/HyMaTE)
        # read `values_zscore` and `times_norm` in addition to the binned
        # value_bins/tau pair, and a hand-curated subset would silently miss them.
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        H_grid, obs_grid = adapter.extract_hidden(batch_dev)

        batch_H_sum = (H_grid * obs_grid.unsqueeze(-1)).sum(dim=(0, 1))  # [V, D]
        batch_obs = obs_grid.sum(dim=(0, 1))        # [V]

        if H_sum is None:
            H_sum = batch_H_sum
            obs_count = batch_obs
        else:
            H_sum += batch_H_sum
            obs_count += batch_obs
        n += 1

    return H_sum / obs_count.unsqueeze(-1).clamp_min(1)


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(trainer, optimizer, epoch, metrics, path):
    sae = trainer.sae
    sae_mod = sae.module if hasattr(sae, 'module') else sae
    save_dict = {
        'epoch': epoch,
        'sae_state_dict': sae_mod.state_dict(),
        'metrics': metrics,
    }
    if hasattr(trainer, 'H_grid_mean'):
        save_dict['H_grid_mean'] = trainer.H_grid_mean.cpu()
    torch.save(save_dict, path)


def load_checkpoint(path, trainer):
    ckpt = torch.load(path, map_location='cpu')
    trainer.sae.load_state_dict(ckpt['sae_state_dict'])
    return ckpt


# =============================================================================
# Training & Eval
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, args, compute_grads=False):
    model.train()
    # The adapter's frozen contract keeps `model.adapter.module` in eval
    # mode regardless, but we still call `.eval()` here for clarity and
    # to handle any future adapter that does not honor `_frozen`.
    model.adapter.eval()

    metrics = defaultdict(float)
    n_samples = 0

    grad_norms = {}
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move ALL keys to device — STraTS/HyMaTE adapters need values_zscore
        # and times_norm in addition to the binned (value_bins, tau) pair.
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(batch, compute_fidelity=False)
        loss = out['loss']

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Capture gradient norms on first batch of diag epochs
        if compute_grads and n_samples == 0:
            grad_norms = compute_submodule_grad_norms(model.sae)

        torch.nn.utils.clip_grad_norm_(model.sae.parameters(), max_norm=1.0)
        optimizer.step()

        B = batch['value_bins'].size(0)
        n_samples += B
        metrics['_n_batches'] = metrics.get('_n_batches', 0) + 1

        metrics['loss'] += loss.item() * B
        metrics['L_recon'] += out['L_recon'].item() * B
        metrics['L_recon_H'] += out['L_recon_H'].item() * B

        for key in ['loss_Z', 'L_auxk', 'L_auxk_combined', 'L_pred', 'KL_pred',
                     'L_auxk_pred', 'KL_auxk_pred',
                     'L_repulsion', 'L_div_W',
                     'L_a0_contrast', 'L_mask_coverage',
                     'L_ch_purity',
                     'mu_var_batch', 'mu_std_median', 'mu_std_pct_low',
                     'mu_std_sorted_median', 'mu_std_sorted_pct_low']:
            if key in out and isinstance(out[key], torch.Tensor):
                metrics[key] += out[key].item() * B
        metrics['n_selected'] += out['n_selected'].mean().item() * B
        if 'sigma' in out:
            s = out['sigma']
            metrics['avg_sigma'] = metrics.get('avg_sigma', 0) + s.mean().item() * B
            metrics['std_sigma'] = metrics.get('std_sigma', 0) + s.std().item() * B
        if 'n_dead' in out:
            metrics['n_dead'] += out['n_dead'] * B
        if out.get('n_selected_per_scale') is not None:
            nps = out['n_selected_per_scale'] / B
            if 'n_selected_per_scale' not in metrics:
                metrics['n_selected_per_scale'] = nps.cpu()
            else:
                metrics['n_selected_per_scale'] = metrics['n_selected_per_scale'] + nps.cpu()

        pbar.set_postfix({
            'loss': f"{metrics['loss']/n_samples:.4f}",
            'recon': f"{metrics['L_recon']/n_samples:.6f}",
            'n_sel': f"{metrics['n_selected']/n_samples:.1f}",
        })

    n_sel_per_scale = metrics.pop('n_selected_per_scale', None)
    n_batches = metrics.pop('_n_batches', None)
    for k in metrics:
        metrics[k] /= max(1, n_samples)
    result = dict(metrics)
    if n_sel_per_scale is not None and n_batches is not None:
        result['n_selected_per_scale'] = (n_sel_per_scale / n_batches).tolist()
    if compute_grads:
        result['grad_norms'] = grad_norms
    return result


@torch.no_grad()
def evaluate(model, dataloader, device, compute_fidelity=True):
    model.eval()
    metrics = defaultdict(float)
    n_samples = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, compute_fidelity=compute_fidelity)
        B = batch['value_bins'].size(0)
        n_samples += B

        metrics['loss'] += out['loss'].item() * B
        metrics['L_recon'] += out['L_recon'].item() * B
        metrics['L_recon_H'] += out['L_recon_H'].item() * B
        if compute_fidelity:
            metrics['cos_H'] += out['cos_H'].item() * B

        for key in ['loss_Z', 'L_auxk', 'L_auxk_combined', 'L_pred', 'KL_pred',
                     'L_auxk_pred', 'KL_auxk_pred',
                     'L_repulsion', 'L_div_W',
                     'L_a0_contrast', 'L_mask_coverage',
                     'L_ch_purity',
                     'mu_var_batch', 'mu_std_median', 'mu_std_pct_low',
                     'mu_std_sorted_median', 'mu_std_sorted_pct_low']:
            if key in out and isinstance(out[key], torch.Tensor):
                metrics[key] += out[key].item() * B
        metrics['n_selected'] += out['n_selected'].mean().item() * B
        if 'sigma' in out:
            metrics['avg_sigma'] = metrics.get('avg_sigma', 0) + out['sigma'].mean().item() * B
            metrics['std_sigma'] = metrics.get('std_sigma', 0) + out['sigma'].std().item() * B

    for k in metrics:
        metrics[k] /= max(1, n_samples)
    return dict(metrics)


# =============================================================================
# Diagnostics
# =============================================================================

def _get_sae_mod(model):
    """Unwrap CheckpointedModule wrapper to get raw TISAE."""
    sae = model.sae
    sae_mod = sae.module if hasattr(sae, 'module') else sae
    if hasattr(sae_mod, 'module'):
        sae_mod = sae_mod.module
    return sae_mod


def _run_batches(model, dataloader, device, max_batches=20, compute_fidelity=False):
    """Generator yielding (out, B) for each batch."""
    model.eval()
    for bi, batch in enumerate(dataloader):
        if bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, compute_fidelity=compute_fidelity)
        yield out, batch['value_bins'].size(0)


def compute_submodule_grad_norms(sae):
    """Per-submodule gradient L2 norms."""
    def classify(name):
        if 'island_params' in name:
            for sub in ['conv1', 'conv2', 'fc_sigma', 'fc_score', 'to_center_logits']:
                if sub in name:
                    return f'ip_{sub}'
            return 'island_params'
        if 'encoder' in name:
            return 'encoder'
        if 'decoder' in name:
            return 'decoder'
        return 'other'

    group_names = ['encoder', 'island_params', 'decoder',
                   'ip_conv1', 'ip_conv2', 'ip_fc_sigma', 'ip_fc_score', 'ip_to_center_logits']
    local_norm_sq = defaultdict(float)
    for name, p in sae.named_parameters():
        if p.grad is None:
            continue
        group = classify(name)
        local_norm_sq[group] += p.grad.norm(2).item() ** 2

    stats = {}
    for g in group_names:
        val = local_norm_sq[g] ** 0.5
        if val > 0 or g in ('encoder', 'island_params', 'decoder'):
            stats[g] = val
    ip_total_sq = sum(v**2 for k, v in stats.items() if k.startswith('ip_'))
    stats['island_params'] = ip_total_sq ** 0.5
    return stats


def _get_posthoc_dom_ch(sae_mod):
    """Post-hoc channel assignment for purity mode (C_eff=1).

    Returns dom_ch [K] tensor (argmax channel per latent) or None.
    """
    if not hasattr(sae_mod, 'channel_var_mask_full') or sae_mod.channel_var_mask_full is None:
        return None
    if hasattr(sae_mod.encoder, 'get_W'):
        W = sae_mod.encoder.get_W()
    elif hasattr(sae_mod.decoder, 'get_W'):
        W = sae_mod.decoder.get_W()
    else:
        return None
    ch_mass = W @ sae_mod.channel_var_mask_full.T  # [K, C]
    return ch_mass.argmax(dim=-1)  # [K]


@torch.no_grad()
def compute_state_diagnostics(model, dataloader, device, max_batches=20):
    """Latent usage / collapse diagnostics."""
    model.eval()
    sae_mod = _get_sae_mod(model)
    K_eff, C_eff = sae_mod.K_eff, sae_mod.C_eff
    K = sae_mod.cfg.K

    all_p_bar = []
    all_p_bar_c = []
    all_sigma = []
    all_sigma_std = []
    all_n_sel = []
    n = 0

    for out, B in _run_batches(model, dataloader, device, max_batches):
        A_hat = out['A_hat']  # [B, K, T]

        if C_eff > 1:
            A_hat_4d = A_hat.view(B, C_eff, K_eff, A_hat.size(2)).permute(0, 2, 3, 1)
        else:
            A_hat_4d = A_hat.unsqueeze(-1)

        a_bk = A_hat.abs().sum(dim=2)  # [B, K]
        p = F.softmax(a_bk, dim=1)
        all_p_bar.append(p.mean(dim=0).cpu())

        a_bkc = A_hat_4d.abs().sum(dim=2)  # [B, K_eff, C_eff]
        p_c = F.softmax(a_bkc, dim=1)
        all_p_bar_c.append(p_c.mean(dim=0).cpu())

        all_sigma.append(out['sigma'].mean().cpu())
        all_sigma_std.append(out['sigma'].std().cpu())
        all_n_sel.append(out['n_selected'].mean().cpu())
        n += 1

    if n == 0:
        return None

    p_bar = torch.stack(all_p_bar).mean(dim=0)
    active_K = (p_bar > 0.05).sum().item()
    H_pbar = -(p_bar * (p_bar + 1e-12).log()).sum().item()
    max_H = math.log(K)

    result = {
        'active_K': active_K,
        'H_pbar': H_pbar,
        'H_pbar_normalized': H_pbar / max_H,
        'avg_sigma': torch.stack(all_sigma).mean().item(),
        'std_sigma': torch.stack(all_sigma_std).mean().item(),
        'avg_n_selected': torch.stack(all_n_sel).mean().item(),
    }

    if C_eff > 1:
        p_bar_c = torch.stack(all_p_bar_c).mean(dim=0)
        active_K_per_c = [(p_bar_c[:, c] > 0.05).sum().item() for c in range(C_eff)]
        H_per_c = [-(p_bar_c[:, c] * (p_bar_c[:, c] + 1e-12).log()).sum().item() for c in range(C_eff)]
        max_H_k = math.log(K_eff) if K_eff > 0 else 1.0
        H_norm_per_c = [h / max_H_k for h in H_per_c]
        result['active_K_per_c'] = active_K_per_c
        result['H_pbar_per_c'] = H_per_c
        result['H_pbar_norm_per_c'] = H_norm_per_c

    return result


@torch.no_grad()
def compute_latent_usage_stats(model, dataloader, device, max_batches=20,
                                act_thresh=1e-3, eps=1e-12):
    """Latent usage statistics: K-level and (K_eff, C_eff) pair level."""
    model.eval()
    sae_mod = _get_sae_mod(model)
    K = sae_mod.cfg.K
    K_eff, C_eff = sae_mod.K_eff, sae_mod.C_eff

    usage_count_k = None
    usage_count_kc = None
    total_samples = 0
    effK_list = []
    activeK_list = []
    chan_ent_list = []

    for out, B in _run_batches(model, dataloader, device, max_batches):
        A_hat = out['A_hat']  # [B, K, T]

        m_bk = A_hat.abs().sum(dim=2)  # [B, K]
        active_bk = (m_bk > act_thresh).float()
        if usage_count_k is None:
            usage_count_k = torch.zeros(K, device=A_hat.device)
            usage_count_kc = torch.zeros(K_eff, C_eff, device=A_hat.device)
        usage_count_k += active_bk.sum(dim=0)

        if C_eff > 1:
            A_hat_4d = A_hat.view(B, C_eff, K_eff, A_hat.size(2)).permute(0, 2, 3, 1)
        else:
            A_hat_4d = A_hat.unsqueeze(-1)
        m_bkc = A_hat_4d.abs().sum(dim=2)
        active_bkc = (m_bkc > act_thresh).float()
        usage_count_kc += active_bkc.sum(dim=0)

        total_samples += B
        activeK_list.append(active_bk.sum(dim=1).cpu())

        denom = m_bk.sum(dim=1, keepdim=True).clamp_min(eps)
        p_bk = m_bk / denom
        ent = -(p_bk * (p_bk + eps).log()).sum(dim=1)
        effK_list.append(torch.exp(ent).cpu())

        if C_eff > 1:
            m_bc = m_bkc.sum(dim=1)
            denom_c = m_bc.sum(dim=1, keepdim=True).clamp_min(eps)
            p_bc = m_bc / denom_c
            chan_ent = -(p_bc * (p_bc + eps).log()).sum(dim=1)
            chan_ent_list.append((chan_ent / math.log(C_eff)).cpu())
        elif C_eff == 1:
            dom_ch = _get_posthoc_dom_ch(sae_mod)
            if dom_ch is not None:
                C_full = sae_mod.channel_var_mask_full.size(0)
                m_bc = torch.zeros(B, C_full, device=m_bk.device)
                for c in range(C_full):
                    mask_c = dom_ch == c
                    if mask_c.any():
                        m_bc[:, c] = m_bk[:, mask_c].sum(dim=1)
                denom_c = m_bc.sum(dim=1, keepdim=True).clamp_min(eps)
                p_bc = m_bc / denom_c
                chan_ent = -(p_bc * (p_bc + eps).log()).sum(dim=1)
                chan_ent_list.append((chan_ent / math.log(C_full)).cpu())

    if total_samples == 0:
        return None

    usage_k = usage_count_k / total_samples
    usage_kc = usage_count_kc / total_samples
    usage_kc_flat = usage_kc.view(-1)
    all_effK = torch.cat(effK_list)

    result = {
        'n_used_5pct': (usage_k > 0.05).sum().item(),
        'n_used_1pct': (usage_k > 0.01).sum().item(),
        'K': K,
        'n_kc_used_5pct': (usage_kc_flat > 0.05).sum().item(),
        'n_kc_used_1pct': (usage_kc_flat > 0.01).sum().item(),
        'KC': K_eff * C_eff,
        'active_per_sample_mean': torch.cat(activeK_list).mean().item(),
        'effK_mean': all_effK.mean().item(),
    }
    if chan_ent_list:
        result['channel_entropy_norm'] = torch.cat(chan_ent_list).mean().item()
    return result


@torch.no_grad()
def compute_island_stats(model, dataloader, device, max_batches=20):
    """Island-specific diagnostics: per-latent island count, mu spread, diversity."""
    model.eval()
    sae_mod = _get_sae_mod(model)
    K_eff = sae_mod.K_eff

    all_n_islands = []
    all_sigma = []
    all_sigma_std = []
    all_mu_within = []
    latent_activations = []
    latent_mu_per_k = defaultdict(list)

    for out, B in _run_batches(model, dataloader, device, max_batches):
        sel = out['sel']      # [B, K_eff, C_eff, M]
        sigma = out['sigma']  # [B, K_eff, C_eff, M]
        mu = out['mu']        # [B, K_eff, C_eff, M]
        B_cur, K_e, C_e, M = sel.shape

        n_islands = sel.sum(dim=-1)
        all_n_islands.append(n_islands.mean().cpu())

        sel_mask = sel > 0.5
        if sel_mask.any():
            all_sigma.append(sigma[sel_mask].mean().cpu())
            all_sigma_std.append(sigma[sel_mask].std().cpu())

        for b in range(min(B_cur, 8)):
            for k in range(K_e):
                for c in range(C_e):
                    active_m = sel_mask[b, k, c]
                    if active_m.sum() > 1:
                        mu_active = mu[b, k, c, active_m]
                        all_mu_within.append((mu_active.max() - mu_active.min()).item())

        latent_active = (sel.sum(dim=(2, 3)) > 0).float()
        for b in range(B_cur):
            latent_activations.append(latent_active[b].cpu())

        for b in range(min(B_cur, 8)):
            for k in range(K_e):
                sel_bk = sel_mask[b, k]
                if sel_bk.any():
                    latent_mu_per_k[k].append(mu[b, k][sel_bk].mean().item())

    # Latent diversity (Jaccard)
    if len(latent_activations) > 1:
        act_matrix = torch.stack(latent_activations)
        n_samp = min(100, act_matrix.size(0))
        sims = []
        for i in range(n_samp):
            for j in range(i+1, n_samp):
                inter = (act_matrix[i] * act_matrix[j]).sum()
                union = ((act_matrix[i] + act_matrix[j]) > 0).float().sum()
                if union > 0:
                    sims.append((inter / union).item())
        latent_diversity = 1.0 - (sum(sims) / len(sims)) if sims else 0.0
    else:
        latent_diversity = 0.0

    mu_vars = [torch.tensor(mus).std().item() for mus in latent_mu_per_k.values() if len(mus) > 1]

    return {
        'avg_islands_per_latent': torch.stack(all_n_islands).mean().item() if all_n_islands else 0.0,
        'avg_sigma': torch.stack(all_sigma).mean().item() if all_sigma else 0.0,
        'std_sigma': torch.stack(all_sigma_std).mean().item() if all_sigma_std else 0.0,
        'mu_spread_within': sum(all_mu_within) / len(all_mu_within) if all_mu_within else 0.0,
        'mu_variance_across': sum(mu_vars) / len(mu_vars) if mu_vars else 0.0,
        'latent_diversity': latent_diversity,
    }


@torch.no_grad()
def compute_alive_dead_diagnostics(model, dataloader, device, max_batches=20):
    """Alive vs dead (k_eff, c_eff) pair diagnostics."""
    model.eval()
    sae_mod = _get_sae_mod(model)
    cfg = sae_mod.cfg
    K_eff, C_eff, M = sae_mod.K_eff, sae_mod.C_eff, cfg.M

    bsf = sae_mod.batches_since_fired  # [K_eff, C_eff]
    dead_mask = bsf > cfg.dead_threshold_batches
    n_alive = (~dead_mask).sum().item()
    n_dead = dead_mask.sum().item()

    if n_alive == 0 or n_dead == 0:
        return {'n_alive': n_alive, 'n_dead': n_dead,
                'msg': f"All {'dead' if n_alive == 0 else 'alive'}, skipping"}

    A0_mag_sum = torch.zeros(K_eff, C_eff, device=device)
    A0_tvar_sum = torch.zeros(K_eff, C_eff, device=device)
    sigma_sum = torch.zeros(K_eff, C_eff, device=device)
    mu_std_sum = torch.zeros(K_eff, C_eff, device=device)
    sel_freq_sum = torch.zeros(K_eff, C_eff, device=device)
    mu_per_m_sum = torch.zeros(M, device=device)
    mu_cross_std_per_m_sum = torch.zeros(M, device=device)
    sigma_per_m_sum = torch.zeros(M, device=device)
    n_batches = 0

    for out, B in _run_batches(model, dataloader, device, max_batches):
        A_hat = out['A_hat']
        sigma = out['sigma']
        mu = out['mu']
        sel = out['sel']

        T = A_hat.size(2)
        if C_eff > 1:
            A0_4d = out['A0'].view(B, C_eff, K_eff, T).permute(0, 2, 3, 1)
        else:
            A0_4d = out['A0'].unsqueeze(-1)

        A0_mag_sum += A0_4d.abs().mean(dim=(0, 2))
        A0_tvar_sum += A0_4d.var(dim=2).mean(dim=0)
        sigma_sum += sigma.mean(dim=(0, 3))
        mu_std_sum += mu.std(dim=3).mean(dim=0)
        sel_freq_sum += (sel > 0.5).float().sum(dim=(0, 3)) / (B * M)

        mu_per_m_sum += mu.mean(dim=(0, 1, 2))
        mu_cross_std_per_m_sum += mu.std(dim=0).mean(dim=(0, 1))
        sigma_per_m_sum += sigma.mean(dim=(0, 1, 2))
        n_batches += 1

    if n_batches == 0:
        return {'n_alive': n_alive, 'n_dead': n_dead, 'msg': 'No batches'}

    A0_mag = A0_mag_sum / n_batches
    A0_tvar = A0_tvar_sum / n_batches
    sigma_avg = sigma_sum / n_batches
    mu_std = mu_std_sum / n_batches
    sel_freq = sel_freq_sum / n_batches
    mu_per_m = (mu_per_m_sum / n_batches).cpu().tolist()
    mu_cross_std_per_m = (mu_cross_std_per_m_sum / n_batches).cpu().tolist()
    sigma_per_m = (sigma_per_m_sum / n_batches).cpu().tolist()

    alive_mask = ~dead_mask
    def alive_dead_mean(tensor):
        a = tensor[alive_mask].mean().item() if alive_mask.any() else 0.0
        d = tensor[dead_mask].mean().item() if dead_mask.any() else 0.0
        return a, d

    def safe_ratio(a, d):
        return a / d if abs(d) > 1e-12 else float('inf')

    a0_a, a0_d = alive_dead_mean(A0_mag)
    tv_a, tv_d = alive_dead_mean(A0_tvar)
    si_a, si_d = alive_dead_mean(sigma_avg)
    mu_a, mu_d = alive_dead_mean(mu_std)
    sf_a, sf_d = alive_dead_mean(sel_freq)

    result = {
        'n_alive': n_alive, 'n_dead': n_dead,
        'A0_mag_alive': a0_a, 'A0_mag_dead': a0_d, 'A0_mag_ratio': safe_ratio(a0_a, a0_d),
        'A0_tvar_alive': tv_a, 'A0_tvar_dead': tv_d, 'A0_tvar_ratio': safe_ratio(tv_a, tv_d),
        'sigma_alive': si_a, 'sigma_dead': si_d,
        'mu_std_alive': mu_a, 'mu_std_dead': mu_d,
        'sel_freq_alive': sf_a, 'sel_freq_dead': sf_d,
        'mu_per_m': mu_per_m,
        'mu_cross_std_per_m': mu_cross_std_per_m,
        'sigma_per_m': sigma_per_m,
    }

    if C_eff > 1:
        n_alive_per_c = alive_mask.sum(dim=0).cpu().tolist()
        sel_total_per_c = sel_freq.sum(dim=0)
        sel_total = sel_total_per_c.sum().clamp_min(1e-8)
        sel_share_per_c = (sel_total_per_c / sel_total).cpu().tolist()
        result['n_alive_per_c'] = n_alive_per_c
        result['sel_share_per_c'] = sel_share_per_c
    elif C_eff == 1:
        dom_ch = _get_posthoc_dom_ch(sae_mod)
        if dom_ch is not None:
            C_full = sae_mod.channel_var_mask_full.size(0)
            alive_1d = alive_mask.squeeze(-1)  # [K_eff]
            sel_1d = sel_freq.squeeze(-1)      # [K_eff]
            n_alive_per_c = [int(alive_1d[dom_ch == c].sum().item()) for c in range(C_full)]
            sel_per_c = [sel_1d[dom_ch == c].sum().item() for c in range(C_full)]
            sel_total = sum(sel_per_c) or 1e-8
            sel_share_per_c = [s / sel_total for s in sel_per_c]
            result['n_alive_per_c'] = n_alive_per_c
            result['sel_share_per_c'] = sel_share_per_c

    return result


@torch.no_grad()
def compute_W_stats(model, eps=1e-8):
    """Decoder W matrix diagnostics (sparsity, coverage, diversity)."""
    try:
        sae_mod = _get_sae_mod(model)
        dec = sae_mod.decoder
        dec_mod = dec.module if hasattr(dec, 'module') else dec

        if not hasattr(dec_mod, 'W_raw') or dec_mod.W_raw.dim() < 2:
            # TISAE decoder uses tied W from encoder; check encoder W_raw
            enc = sae_mod.encoder
            enc_mod = enc.module if hasattr(enc, 'module') else enc
            if not hasattr(enc_mod, 'W_raw') or enc_mod.W_raw.dim() < 2:
                return None
            W_raw = enc_mod.W_raw
        else:
            W_raw = dec_mod.W_raw

        W = torch.softplus(W_raw)
        W = W / W.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        K, V = W.shape
        logV = math.log(V)

        nz_frac = (W > eps).float().mean()
        row_ent = -(W * (W + eps).log()).sum(dim=1)
        row_ent_norm = row_ent / logV

        top1_vals, top1_idx = W.max(dim=1)
        unique_top1 = len(set(top1_idx.tolist()))
        unique_top1_frac = unique_top1 / V

        var_mass = W.sum(dim=0)
        var_mass = var_mass / var_mass.sum().clamp_min(eps)
        var_mass_ent = -(var_mass * (var_mass + eps).log()).sum()
        var_mass_ent_norm = var_mass_ent / logV
        n_vars_used = (var_mass > eps).sum().item()

        # Also compute encoder W stats
        enc_stats = None
        enc = sae_mod.encoder
        enc_mod = enc.module if hasattr(enc, 'module') else enc
        if hasattr(enc_mod, 'W_raw') and enc_mod.W_raw.dim() >= 2:
            We = torch.softplus(enc_mod.W_raw)
            if hasattr(enc_mod, 'latent_var_mask') and enc_mod.latent_var_mask is not None:
                We = We * enc_mod.latent_var_mask
            We = We / We.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            enc_nz = (We > eps).float().mean()
            enc_ent = -(We * (We + eps).log()).sum(dim=1)
            enc_stats = {
                'shape': tuple(We.shape),
                'nz_frac': enc_nz.item(),
                'row_entropy_mean': enc_ent.mean().item(),
            }

        return {
            'W_shape': (K, V),
            'nz_frac': nz_frac.item(),
            'row_entropy_mean': row_ent.mean().item(),
            'row_entropy_norm_mean': row_ent_norm.mean().item(),
            'top1_mean': top1_vals.mean().item(),
            'unique_top1_frac': unique_top1_frac,
            'n_vars_used': n_vars_used,
            'var_mass_entropy_norm': var_mass_ent_norm.item(),
            'enc_W': enc_stats,
        }
    except (RuntimeError, AttributeError):
        return None


@torch.no_grad()
def compute_nn_overlap(model, dataloader, device, k=10, n_vec=512, max_batches=30):
    """Compute NN overlap between H and H_hat."""
    model.eval()
    H_list, Hh_list = [], []

    for bi, batch in enumerate(dataloader):
        if bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(batch, compute_fidelity=False)
        H = out['H']
        H_hat = out['H_hat']

        valid = (batch['obs_mask'] > 0.5)
        Hv = H[valid]
        Hhv = H_hat[valid]
        if Hv.numel() == 0:
            continue
        H_list.append(Hv.cpu())
        Hh_list.append(Hhv.cpu())
        total = sum(x.size(0) for x in H_list)
        if total >= n_vec:
            break

    if len(H_list) == 0:
        return {'nn_overlap': 0.0, 'N': 0, 'k': k}

    H_all = torch.cat(H_list, dim=0)[:n_vec]
    Hh_all = torch.cat(Hh_list, dim=0)[:n_vec]
    N = H_all.size(0)
    if N <= k + 1:
        return {'nn_overlap': 0.0, 'N': N, 'k': k}

    H_norm = F.normalize(H_all, p=2, dim=-1)
    Hh_norm = F.normalize(Hh_all, p=2, dim=-1)
    sim_H = H_norm @ H_norm.t()
    sim_Hh = Hh_norm @ Hh_norm.t()
    sim_H.fill_diagonal_(-1e9)
    sim_Hh.fill_diagonal_(-1e9)

    k_actual = min(k, N - 1)
    idx_H = sim_H.topk(k_actual, dim=-1).indices
    idx_Hh = sim_Hh.topk(k_actual, dim=-1).indices

    overlap_scores = []
    for i in range(N):
        set_H = set(idx_H[i].tolist())
        set_Hh = set(idx_Hh[i].tolist())
        overlap_scores.append(len(set_H & set_Hh) / k_actual)

    return {
        'nn_overlap': sum(overlap_scores) / len(overlap_scores),
        'N': N,
        'k': k_actual,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TISAE Training')

    # Paths
    parser.add_argument('--backbone_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./experiments/tisae')
    parser.add_argument('--dataset_path', type=str, default='data/unified_mimic3_48h.pkl')
    parser.add_argument('--warm_start', type=str, default=None,
                        help='Path to experiment dir or checkpoint .pt file to warm-start TISAE weights from')

    # Architecture
    parser.add_argument('--extract_layer', type=int, default=-1)
    parser.add_argument('--K', type=int, default=1280)
    parser.add_argument('--encoder_sparsity_k', type=int, default=5)
    parser.add_argument('--encoder_chunk_size', type=int, default=32)

    # Channel restriction
    parser.add_argument('--restrict_to_channels', action=argparse.BooleanOptionalAction, default=False,
                        help='Restrict decoder W to channel-assigned variables')
    parser.add_argument('--C', type=int, default=10,
                        help='Number of clinical channels')
    parser.add_argument('--lambda_ch_purity', type=float, default=0.1,
                        help='Self-organizing channel purity (0=off)')

    # Islands
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--min_sigma', type=float, default=0.5)
    parser.add_argument('--max_sigma', type=float, default=12.0)
    parser.add_argument('--island_hidden', type=int, default=32)
    parser.add_argument('--island_kernel_size', type=int, default=25)
    parser.add_argument('--n_sigma_iter', type=int, default=3)
    parser.add_argument('--separated_topk', action=argparse.BooleanOptionalAction, default=False)

    # Selector
    parser.add_argument('--n_global', type=int, default=128)
    parser.add_argument('--n_per_channel', type=int, default=None,
                        help='Per-channel selection cap')
    parser.add_argument('--score_mode', type=str, default='energy_gated',
                        choices=['energy', 'energy_gated'])
    parser.add_argument('--gumbel_topk_noise', type=float, default=0.05)
    parser.add_argument('--dead_boost_scale', type=float, default=0.5)
    parser.add_argument('--dead_threshold_batches', type=int, default=5)

    # Decoder
    parser.add_argument('--W_sparsity_k', type=int, default=3)
    parser.add_argument('--normalize_atoms', action=argparse.BooleanOptionalAction, default=True)

    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_pred', type=float, default=0.0)
    parser.add_argument('--lambda_auxk', type=float, default=0.3)
    parser.add_argument('--auxk_k', type=int, default=256)
    parser.add_argument('--lambda_mu_repulsion', type=float, default=0.01)
    parser.add_argument('--lambda_div_W', type=float, default=0.01)
    parser.add_argument('--lambda_mask_coverage', type=float, default=0.005,
                        help='Penalize temporal coverage of selected islands (CRUCIAL for sigma control)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--early_stop_metric', type=str, default='loss',
                        choices=['L_recon', 'loss'])

    # Runtime
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gradient_checkpointing', action='store_true')

    # Logging
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--diag_every', type=int, default=5)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = Path(args.output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(exp_dir)

    logger.write(f'\n{"="*80}')
    logger.write(f'TISAE Training')
    logger.write(f'{"="*80}')

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    logger.write(f'Dataset: N={dataset.N}, V={dataset.V}, T={dataset.T}')
    if hasattr(dataset, 'var_to_ind_hash'):
        logger.write(f'  var_to_ind_hash: {dataset.var_to_ind_hash}')

    # Read prediction task from backbone checkpoint; filter readmission splits
    _bb_ckpt = torch.load(args.backbone_checkpoint, map_location='cpu', weights_only=False)
    _task = _bb_ckpt.get('training_args', {}).get('task', 'mortality')
    del _bb_ckpt
    if _task == 'readmission' and hasattr(dataset, 'y_readmission'):
        for split in dataset.splits:
            inds = np.array(dataset.splits[split])
            valid = dataset.y_readmission[inds] >= 0
            dataset.splits[split] = inds[valid].tolist()
        logger.write(f'Task: readmission — filtered death samples from splits')

    train_ds = TISAEDataset(dataset, 'train')
    val_ds = TISAEDataset(dataset, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    # Backbone — loaded via the BackboneAdapter factory. The
    # factory dispatches on `backbone_type` in the checkpoint metadata
    # and returns a frozen adapter ready for inference.
    logger.write(f'Loading backbone via adapter: {args.backbone_checkpoint}')
    dataset_meta = {
        'T': dataset.T,
        'V': dataset.V,
        'var_to_ind_hash': getattr(dataset, 'var_to_ind_hash', None),
    }
    adapter = load_backbone_adapter(
        args.backbone_checkpoint,
        device=device,
        dataset_meta=dataset_meta,
    )
    logger.write(
        f'  adapter: name={adapter.name}, T={adapter.T}, V={adapter.V}, '
        f'D={adapter.D}, n_value_bins={adapter.n_value_bins}, '
        f'is_finetune={adapter.is_finetune}, '
        f'var_to_ind_hash={adapter.var_to_ind_hash}'
    )

    # hard-fail if the backbone was trained against a different
    # dataset version than the one we're loading now.
    ds_hash = getattr(dataset, 'var_to_ind_hash', None)
    if adapter.var_to_ind_hash is not None and ds_hash is not None:
        if adapter.var_to_ind_hash != ds_hash:
            raise RuntimeError(
                f"var_to_ind_hash mismatch: backbone={adapter.var_to_ind_hash} "
                f"vs dataset={ds_hash}. The backbone was trained against a "
                f"different dataset version (different V, different ordering, "
                f"or different cohort). Refusing to silently corrupt training."
            )

    D = adapter.D
    T = adapter.T
    V = adapter.V
    C = args.C

    # Channel mask (for restrict_to_channels or purity loss)
    channel_var_mask = None
    if args.restrict_to_channels or args.lambda_ch_purity > 0:
        ontology = create_ontology_mapping(dataset.var_names, n_channels=C)
        channel_var_mask = ontology.get_hard_mask()  # [C, V]
        if args.restrict_to_channels:
            K_per_c = args.K // C
            if args.K % C != 0:
                parser.error(f"--K={args.K} must be divisible by C={C} when --restrict_to_channels")
            n_per_c = channel_var_mask.sum(dim=1).long().tolist()
            logger.write(f'  Channel restriction: K_per_c={K_per_c}, vars/ch={n_per_c}')
        if args.lambda_ch_purity > 0 and not args.restrict_to_channels:
            n_per_c = channel_var_mask.sum(dim=1).long().tolist()
            logger.write(f'  Channel purity mode: vars/ch={n_per_c}')

    # TISAEConfig construction
    tisae_config = TISAEConfig(
        D=D, T=T, V=V, K=args.K, C=C,
        restrict_to_channels=args.restrict_to_channels,
        lambda_ch_purity=args.lambda_ch_purity,
        encoder_sparsity_k=args.encoder_sparsity_k,
        encoder_chunk_size=args.encoder_chunk_size,
        M=args.M,
        min_sigma=args.min_sigma, max_sigma=args.max_sigma,
        island_hidden=args.island_hidden, island_kernel_size=args.island_kernel_size,
        n_sigma_iter=args.n_sigma_iter,
        n_global=args.n_global, n_per_channel=args.n_per_channel,
        score_mode=args.score_mode,
        separated_topk=args.separated_topk,
        gumbel_topk_noise=args.gumbel_topk_noise,
        dead_boost_scale=args.dead_boost_scale,
        dead_threshold_batches=args.dead_threshold_batches,
        W_sparsity_k=args.W_sparsity_k,
        normalize_atoms=args.normalize_atoms,
        lambda_recon=args.lambda_recon, lambda_pred=args.lambda_pred,
        lambda_auxk=args.lambda_auxk, auxk_k=args.auxk_k,
        lambda_mu_repulsion=args.lambda_mu_repulsion,
        lambda_div_W=args.lambda_div_W,
        lambda_mask_coverage=args.lambda_mask_coverage,
    )

    logger.write(f'TISAE: K={args.K}, M={args.M}, n_global={args.n_global}, V={V}')
    if args.restrict_to_channels:
        logger.write(f'  C_eff={C}, K_eff={args.K // C}')
        if args.n_per_channel is not None:
            logger.write(f'  n_per_channel={args.n_per_channel}')
    logger.write(f'  score_mode={args.score_mode}')
    logger.write(f'  sigma_mode=em, min_sigma={args.min_sigma}, max_sigma={args.max_sigma}')
    logger.write(f'  n_sigma_iter={args.n_sigma_iter}')
    logger.write(f'  encoder_sparsity_k={args.encoder_sparsity_k}, W_sparsity_k={args.W_sparsity_k}')
    logger.write(f'  lambda_recon={args.lambda_recon}, lambda_auxk={args.lambda_auxk}, '
                 f'lambda_mu_repulsion={args.lambda_mu_repulsion}')
    logger.write(f'  lambda_div_W={args.lambda_div_W}')
    if args.lambda_mask_coverage > 0:
        logger.write(f'  lambda_mask_coverage={args.lambda_mask_coverage}')
    if args.lambda_ch_purity > 0:
        logger.write(f'  lambda_ch_purity={args.lambda_ch_purity} (self-organizing channel purity)')
    if args.lambda_pred > 0:
        logger.write(f'  lambda_pred={args.lambda_pred}')
    logger.write(f'  dead_boost_scale={args.dead_boost_scale}, gumbel_topk_noise={args.gumbel_topk_noise}, '
                 f'dead_threshold_batches={args.dead_threshold_batches}')

    # Create TISAE
    sae = TISAE(tisae_config, channel_var_mask=channel_var_mask)

    # Warm-start
    if args.warm_start:
        ws_path = Path(args.warm_start)
        if ws_path.is_dir():
            ws_path = ws_path / 'final_model.pt'
        logger.write(f'  Warm-start: loading TISAE weights from {ws_path}')
        ws_ckpt = torch.load(ws_path, map_location='cpu')
        ws_sd = ws_ckpt['sae_state_dict']
        ws_sd = {k.replace('.module.', '.'): v for k, v in ws_sd.items()}
        sae.load_state_dict(ws_sd)
        logger.write(f'  Warm-start: loaded successfully')

    if args.gradient_checkpointing:
        sae = apply_gradient_checkpointing(sae)
    sae = sae.to(device)

    # Parameter counting
    enc_p = sum(p.numel() for p in sae.encoder.parameters())
    ip_p = sum(p.numel() for p in sae.island_params.parameters())
    dec_p = sum(p.numel() for p in sae.decoder.parameters())
    total_p = sum(p.numel() for p in sae.parameters())
    total_t = sum(p.numel() for p in sae.parameters() if p.requires_grad)
    logger.write(f'\n--- Parameter Count ---')
    logger.write(f'  Encoder: {enc_p:,} (W_raw[{args.K}, {V}] + d_enc[{args.K}, {D}] + b[{args.K}])')
    logger.write(f'  IslandParamNet: {ip_p:,}')
    logger.write(f'  Decoder: {dec_p:,} (W shared with encoder + P[{args.K}, {D}] + bias[{D}])')
    logger.write(f'  Total: {total_p:,} ({total_t:,} trainable)')

    # Wire extract_layer into the adapter so extract_hidden returns intermediate-layer embeddings.
    adapter.set_extract_layer(args.extract_layer)

    model = TISAETrainer(adapter, sae, T, V,
                          args.extract_layer, lambda_pred=args.lambda_pred,
                          gradient_checkpointing=args.gradient_checkpointing)

    # Subtract per-variable population mean before encoding
    H_grid_mean = precompute_H_grid_mean(adapter, train_loader, device, T, V,
                                          extract_layer=args.extract_layer)
    # NOTE: H_grid_mean is stored on BOTH the trainer (used by the L_recon /
    # L_pred forward path) AND the adapter (used by external callers of
    # adapter.predict_from_grid for eval / IG attribution). They reference
    # the same precomputed tensor and must remain consistent — never reset
    # one without the other.
    model.register_buffer('H_grid_mean', H_grid_mean)
    adapter.set_residualization_mean(H_grid_mean)

    # Save config
    training_config = {
        'lambda_pred': args.lambda_pred,
        'residualize': True,
        'residualize_mode': 'v',
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'early_stop_metric': args.early_stop_metric,
    }
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump({
            'tisae_config': asdict(tisae_config),
            'training': training_config,
            'args': vars(args),
        }, f, default_flow_style=False)

    # Optimizer
    optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    logger.write(f'\n{"="*80}\nTraining\n{"="*80}')

    best_val = float('inf')
    wait = 0
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        logger.write(f'\n--- Epoch {epoch}/{args.epochs} ---')

        compute_grads = (epoch % args.diag_every == 0 or epoch == 1)
        train_m = train_epoch(model, train_loader, optimizer, device, args,
                              compute_grads=compute_grads)
        if epoch % args.log_every == 0:
            nsel_str = f"n_sel={train_m['n_selected']:.1f}"
            if 'n_selected_per_scale' in train_m:
                nps = train_m['n_selected_per_scale']
                nsel_str += f" ([{', '.join(f'{v:.1f}' for v in nps)}])"
            sae_mod = _get_sae_mod(model)
            n_dead_denom = sae_mod.K_eff * sae_mod.C_eff

            logger.write(f"  Train: L_recon={train_m['L_recon']:.6f}  loss_Z={train_m.get('loss_Z', 0):.4f}  "
                         f"L_recon_H={train_m['L_recon_H']:.4f}")
            logger.write(f"         L_repulsion={train_m.get('L_repulsion', 0):.5f}  "
                         f"L_div_W={train_m.get('L_div_W', 0):.5f}")
            logger.write(f"         auxk={train_m.get('L_auxk', 0):.5f}  "
                         f"n_dead={train_m.get('n_dead', 0):.0f}/{n_dead_denom}  {nsel_str}")
            logger.write(f"         a0_contrast={train_m.get('L_a0_contrast', 0):.4f}  "
                         f"mu_std_med={train_m.get('mu_std_median', 0):.2f}h  "
                         f"avg_sigma={train_m.get('avg_sigma', 0):.2f}  "
                         f"std_sigma={train_m.get('std_sigma', 0):.2f}")
            opt_parts = []
            if args.lambda_mask_coverage > 0:
                opt_parts.append(f"mask_cov={train_m.get('L_mask_coverage', 0):.4f}")
            if args.lambda_ch_purity > 0:
                opt_parts.append(f"ch_pur={train_m.get('L_ch_purity', 0):.4f}")
            if opt_parts:
                logger.write(f"         {'  '.join(opt_parts)}")

        # Eval
        if epoch % args.eval_every == 0:
            val_m = evaluate(model, val_loader, device, compute_fidelity=True)
            logger.write(f"  Val:   L_recon={val_m['L_recon']:.6f}  loss_Z={val_m.get('loss_Z', 0):.4f}  "
                         f"L_recon_H={val_m['L_recon_H']:.4f}")
            logger.write(f"         L_repulsion={val_m.get('L_repulsion', 0):.5f}  "
                         f"L_div_W={val_m.get('L_div_W', 0):.5f}  "
                         f"avg_sigma={val_m.get('avg_sigma', 0):.2f}  "
                         f"std_sigma={val_m.get('std_sigma', 0):.2f}")
            logger.write(f"         a0_contrast={val_m.get('L_a0_contrast', 0):.4f}  "
                         f"mu_std_med={val_m.get('mu_std_median', 0):.2f}h")
            logger.write(f"         cos_H={val_m.get('cos_H', 0):.3f}")

            for k, v in train_m.items():
                if isinstance(v, (int, float)):
                    history[f'train_{k}'].append(v)
            for k, v in val_m.items():
                if isinstance(v, (int, float)):
                    history[f'val_{k}'].append(v)

            # Early stopping
            es_metric = val_m[args.early_stop_metric]
            if es_metric < best_val:
                best_val = es_metric
                wait = 0
                save_checkpoint(model, optimizer, epoch, val_m, exp_dir / 'best_model.pt')
                logger.write("  [New best]")
            else:
                wait += 1
                if wait >= args.patience:
                    logger.write(f"\nEarly stopping at epoch {epoch}")
                    break

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_m, exp_dir / f'ckpt_e{epoch}.pt')

        scheduler.step()

    save_checkpoint(model, optimizer, epoch, train_m, exp_dir / 'final_model.pt')
    logger.write(f'\nTraining complete. Best val loss: {best_val:.4f}')


if __name__ == '__main__':
    main()
