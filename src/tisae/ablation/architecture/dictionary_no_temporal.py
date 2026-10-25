"""Dictionary builder for TISAENoTemporal.

Same characterization as canonical (grid encoder/decoder, W_sparsity_k=3),
but loads TISAENoTemporal instead of TISAE. Since there are no Gaussian
islands, temporal positions are derived from peak A0 activation times.

Usage:
    python -m tisae.ablation.dictionary_no_temporal \\
        --experiment_dir experiments_ablation/structural/no_temporal/tisae \\
        --dataset_path data/unified_mimic3_48h.pkl \\
        --output_dir experiments_ablation/dictionaries/no_temporal \\
        --n_samples 500
"""

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import fields
from tqdm import tqdm

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.tisae import TISAEConfig
from tisae.models.ontology import create_ontology_mapping
from tisae.ablation.architecture.tisae_no_temporal import TISAENoTemporal


@torch.no_grad()
def build_dictionary(model, adapter, dataset, test_indices, device,
                     H_grid_mean=None, top_n_episodes=10, min_freq=0.01):
    K = model.cfg.K
    T = dataset.T
    V = dataset.V
    var_names = dataset.var_names

    W_dec = model.get_W_dec().detach().cpu().numpy()  # [K, V]

    # Per-latent accumulators
    latent_stats = defaultdict(lambda: {
        'activation_strengths': [],
        'time_profiles': np.zeros(T),
        'mortality_labels': [],
    })

    n_samples = len(test_indices)
    print(f"Scanning {n_samples} episodes...")

    for i in tqdm(range(n_samples)):
        idx = test_indices[i]
        sample = dataset.get_item(idx)
        batch = collate_fn([sample])
        batch = {k: v.to(device) for k, v in batch.items()}

        H_grid, obs_grid = adapter.extract_hidden(batch)
        if H_grid_mean is not None:
            H_grid = (H_grid - H_grid_mean.unsqueeze(0).unsqueeze(1)) * obs_grid.unsqueeze(-1)

        out = model(H_grid, obs_grid)
        A_hat = out['A_hat'][0].cpu().numpy()  # [K, T]

        label = batch.get("label", batch.get("mortality", None))
        if label is not None:
            label = label[0].item()

        for k in range(K):
            a_k = np.abs(A_hat[k])
            mean_act = a_k.mean()
            if mean_act < 1e-8:
                continue

            stats = latent_stats[k]
            stats['activation_strengths'].append((int(idx), float(mean_act)))
            stats['time_profiles'] += a_k
            if label is not None:
                stats['mortality_labels'].append(label)

    # Baseline mortality
    mort_labels_all = []
    for i in range(min(100, n_samples)):
        sample = dataset.get_item(test_indices[i])
        batch = collate_fn([sample])
        label = batch.get("label", batch.get("mortality", None))
        if label is not None:
            mort_labels_all.append(label[0].item())
    baseline_mort = np.mean(mort_labels_all) if mort_labels_all else 0.14

    # Build dictionary
    dictionary = []
    n_alive = 0
    for k in range(K):
        stats = latent_stats[k]
        n_episodes = len(stats['activation_strengths'])
        freq = n_episodes / n_samples
        if freq < min_freq:
            continue
        n_alive += 1

        # Variable composition from decoder W
        w_row = np.abs(W_dec[k])
        top_var_idx = w_row.argsort()[-3:][::-1]
        w_sum = w_row.sum() if w_row.sum() > 0 else 1
        top_vars = [
            {"var_name": var_names[v], "weight": float(w_row[v]), "var_idx": int(v)}
            for v in top_var_idx if w_row[v] > 1e-6
        ]

        # Temporal profile (from aggregated A_hat, no islands)
        tp = stats['time_profiles']
        tp_sum = tp.sum() if tp.sum() > 0 else 1
        peak_time = float(tp.argmax())
        temporal_entropy = float(
            -np.sum((tp / tp_sum) * np.log(np.clip(tp / tp_sum, 1e-12, None))) / np.log(T))

        # Mortality
        mort_labels = stats['mortality_labels']
        mort_rate = float(np.mean(mort_labels)) if mort_labels else 0.0
        mort_or = (mort_rate / (1 - mort_rate + 1e-8)) / (
            baseline_mort / (1 - baseline_mort + 1e-8)) if baseline_mort > 0 else 1.0

        # Top episodes
        sorted_eps = sorted(stats['activation_strengths'], key=lambda x: x[1], reverse=True)

        # Variable entropy
        w_norm = w_row / w_sum
        var_entropy = float(
            -np.sum(w_norm * np.log(np.clip(w_norm, 1e-12, None))) / np.log(V))

        dictionary.append({
            "latent_k": int(k),
            "activation_frequency": float(freq),
            "n_episodes_active": n_episodes,
            "variable_composition": top_vars,
            "variable_entropy": var_entropy,
            "peak_time_bin": peak_time,
            "temporal_entropy": temporal_entropy,
            "temporal_note": "no_temporal: flat activation, no Gaussian islands",
            "mortality_rate_when_active": mort_rate,
            "mortality_rate_baseline": float(baseline_mort),
            "mortality_odds_ratio": float(mort_or),
            "top_episodes": [{"episode_idx": ep[0], "activation": ep[1]}
                             for ep in sorted_eps[:top_n_episodes]],
        })

    dictionary.sort(key=lambda x: x["activation_frequency"], reverse=True)
    return dictionary, n_alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    exp_dir = Path(args.experiment_dir)
    with open(exp_dir / 'config.yaml') as f:
        config = yaml.safe_load(f)
    cfg_dict = config.get('tisae_config', {})
    valid = {f.name for f in fields(TISAEConfig)}
    cfg = TISAEConfig(**{k: v for k, v in cfg_dict.items() if k in valid})

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V,
          "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}

    channel_var_mask = None
    if cfg.lambda_ch_purity > 0:
        ontology = create_ontology_mapping(dataset.var_names, n_channels=cfg.C)
        channel_var_mask = ontology.get_hard_mask()

    model = TISAENoTemporal(cfg, channel_var_mask=channel_var_mask)
    ckpt = torch.load(exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['sae_state_dict'])
    model = model.to(device).eval()

    bb_path = config.get('args', {}).get('backbone_checkpoint')
    adapter = load_backbone_adapter(bb_path, device=str(device), dataset_meta=dm)
    extract_layer = config.get('args', {}).get('extract_layer', -1)
    adapter.set_extract_layer(extract_layer)

    H_grid_mean = ckpt.get('H_grid_mean')
    if H_grid_mean is not None:
        H_grid_mean = H_grid_mean.to(device)
        adapter.set_residualization_mean(H_grid_mean)

    test_indices = dataset.splits["test"][:args.n_samples]
    dictionary, n_alive = build_dictionary(
        model, adapter, dataset, test_indices, device, H_grid_mean)

    with open(output_dir / 'latent_dictionary.json', 'w') as f:
        json.dump(dictionary, f, indent=2)

    var_ents = [d['variable_entropy'] for d in dictionary]
    temp_ents = [d['temporal_entropy'] for d in dictionary]

    print(f"\n{'='*60}")
    print(f"No-Temporal Dictionary Summary")
    print(f"{'='*60}")
    print(f"Alive latents (freq > 1%): {n_alive}/{cfg.K}")
    print(f"Variable entropy:  mean={np.mean(var_ents):.3f}  (canonical ≈ 0)")
    print(f"Temporal entropy:  mean={np.mean(temp_ents):.3f}  (1.0 = uniform across time)")

    print(f"\nTop 5 latents:")
    for d in dictionary[:5]:
        vars_str = ", ".join([f"{v['var_name']}({v['weight']:.2f})" for v in d['variable_composition'][:3]])
        print(f"  k={d['latent_k']:>4}  freq={d['activation_frequency']:.3f}  "
              f"OR={d['mortality_odds_ratio']:.2f}  var_ent={d['variable_entropy']:.3f}  "
              f"temp_ent={d['temporal_entropy']:.3f}")
        print(f"         vars: {vars_str}")

    print(f"\nSaved to {output_dir / 'latent_dictionary.json'}")


if __name__ == "__main__":
    main()
