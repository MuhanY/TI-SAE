"""Dictionary builder for SimpleSAE — characterize latents by top-activating episodes.

Since SimpleSAE has no decoder W[K,V] or Gaussian islands,
characterize each latent by its activation patterns across episodes:
  - Which events (variables, time bins) trigger it most
  - Variable profile: distribution of activating variables
  - Temporal profile: when activating events occur
  - Mortality association

Usage:
    python -m tisae.ablation.dictionary_simple_sae \\
        --experiment_dir experiments_ablation/structural/simple_sae/model \\
        --dataset_path data/unified_mimic3_48h.pkl \\
        --output_dir experiments_ablation/dictionaries/simple_sae \\
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
from tisae.ablation.architecture.simple_sae import SimpleSAE, SimpleSAEConfig


@torch.no_grad()
def build_dictionary(model, adapter, dataset, test_indices, device,
                     top_n_episodes=10, min_freq=0.01):
    """Build per-latent dictionary from SimpleSAE activations."""
    K = model.cfg.K
    T = dataset.T
    V = dataset.V
    var_names = dataset.var_names

    exp_dir = Path(args.experiment_dir)

    # Per-latent accumulators
    latent_stats = defaultdict(lambda: {
        'activation_strengths': [],  # (episode_idx, mean_activation)
        'var_counts': np.zeros(V),   # how many events of each variable activate this latent
        'time_counts': np.zeros(T),  # temporal distribution of activating events
        'mortality_labels': [],
        'total_events': 0,
    })

    n_samples = len(test_indices)
    print(f"Scanning {n_samples} episodes...")

    for i in tqdm(range(n_samples)):
        idx = test_indices[i]
        sample = dataset.get_item(idx)
        batch = collate_fn([sample])
        batch = {k: v.to(device) for k, v in batch.items()}

        _, _, H = adapter.extract_hidden(batch, return_event_h=True)
        obs_mask = batch["obs_mask"][0]  # [L]
        tau = batch["tau"][0]            # [L]
        varis = batch["varis"][0]        # [L]
        label = batch.get("labels", batch.get("label", batch.get("mortality", None)))
        if label is not None:
            label = label[0].item()

        # No residualization for simple_sae
        H_input = H[0] * obs_mask.unsqueeze(-1)

        z = model.encode(H_input.unsqueeze(0))[0]  # [L, K]

        valid = obs_mask.bool()
        z_valid = z[valid]       # [n_valid, K]
        tau_valid = tau[valid].long().clamp(0, T - 1)
        var_valid = varis[valid].long().clamp(0, V - 1)

        # Per-latent: which events activate it
        for k in range(K):
            z_k = z_valid[:, k]  # [n_valid]
            active = z_k > 0
            if not active.any():
                continue

            mean_act = z_k[active].mean().item()
            stats = latent_stats[k]
            stats['activation_strengths'].append((int(idx), mean_act))

            # Variable distribution
            active_vars = var_valid[active].cpu().numpy()
            for v in active_vars:
                stats['var_counts'][v] += 1

            # Temporal distribution
            active_times = tau_valid[active].cpu().numpy()
            for t in active_times:
                stats['time_counts'][t] += 1

            stats['total_events'] += active.sum().item()

            if label is not None:
                stats['mortality_labels'].append(label)

    # Build dictionary entries
    baseline_mort = np.mean([
        collate_fn([dataset.get_item(test_indices[i])]).get(
            "labels", collate_fn([dataset.get_item(test_indices[i])]).get(
                "label", torch.tensor([0])))[0].item()
        for i in range(min(100, n_samples))
    ])

    dictionary = []
    n_alive = 0
    for k in range(K):
        stats = latent_stats[k]
        n_episodes = len(stats['activation_strengths'])
        freq = n_episodes / n_samples

        if freq < min_freq:
            continue
        n_alive += 1

        # Top variables
        vc = stats['var_counts']
        top_var_idx = vc.argsort()[-5:][::-1]
        vc_sum = vc.sum() if vc.sum() > 0 else 1
        top_vars = [
            {"var_name": var_names[v], "weight": float(vc[v] / vc_sum), "var_idx": int(v)}
            for v in top_var_idx if vc[v] > 0
        ]

        # Temporal profile
        tc = stats['time_counts']
        tc_sum = tc.sum() if tc.sum() > 0 else 1
        peak_time = float(tc.argmax())
        temporal_entropy = float(-np.sum((tc / tc_sum) * np.log(np.clip(tc / tc_sum, 1e-12, None))) / np.log(T))

        # Mortality
        mort_labels = stats['mortality_labels']
        mort_rate = float(np.mean(mort_labels)) if mort_labels else 0.0
        mort_or = (mort_rate / (1 - mort_rate + 1e-8)) / (baseline_mort / (1 - baseline_mort + 1e-8)) if baseline_mort > 0 else 1.0

        # Top episodes
        sorted_eps = sorted(stats['activation_strengths'], key=lambda x: x[1], reverse=True)
        top_episodes = [{"episode_idx": ep[0], "activation": ep[1]} for ep in sorted_eps[:top_n_episodes]]

        # Variable concentration (effective number of variables)
        vc_norm = vc / vc_sum
        var_entropy = float(-np.sum(vc_norm * np.log(np.clip(vc_norm, 1e-12, None))) / np.log(V))

        dictionary.append({
            "latent_k": int(k),
            "activation_frequency": float(freq),
            "n_episodes_active": n_episodes,
            "total_activating_events": stats['total_events'],
            "variable_composition": top_vars,
            "variable_entropy": var_entropy,  # 0=specialized, 1=uniform
            "peak_time_bin": peak_time,
            "temporal_entropy": temporal_entropy,  # 0=localized, 1=uniform
            "mortality_rate_when_active": mort_rate,
            "mortality_rate_baseline": float(baseline_mort),
            "mortality_odds_ratio": float(mort_or),
            "top_episodes": top_episodes,
        })

    dictionary.sort(key=lambda x: x["activation_frequency"], reverse=True)
    return dictionary, n_alive


def main():
    global args
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
    cfg = SimpleSAEConfig(**config.get('simple_sae_config', {}))
    model = SimpleSAE(cfg)
    ckpt = torch.load(exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['sae_state_dict'])
    model = model.to(device).eval()

    # Load backbone
    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V,
          "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    bb_path = config.get('args', {}).get('backbone_checkpoint')
    adapter = load_backbone_adapter(bb_path, device=str(device), dataset_meta=dm)
    extract_layer = config.get('args', {}).get('extract_layer', -1)
    adapter.set_extract_layer(extract_layer)

    test_indices = dataset.splits["test"][:args.n_samples]

    dictionary, n_alive = build_dictionary(model, adapter, dataset, test_indices, device)

    # Save
    with open(output_dir / 'latent_dictionary.json', 'w') as f:
        json.dump(dictionary, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"SimpleSAE Dictionary Summary")
    print(f"{'='*60}")
    print(f"Alive latents (freq > 1%): {n_alive}/{cfg.K}")
    print(f"Total entries: {len(dictionary)}")

    # Variable entropy stats
    var_ents = [d['variable_entropy'] for d in dictionary]
    temp_ents = [d['temporal_entropy'] for d in dictionary]
    print(f"\nVariable entropy:  mean={np.mean(var_ents):.3f}  (0=specialized, 1=uniform)")
    print(f"Temporal entropy:  mean={np.mean(temp_ents):.3f}  (0=localized, 1=uniform)")

    # Show top 5 latents
    print(f"\nTop 5 latents by frequency:")
    for d in dictionary[:5]:
        vars_str = ", ".join([f"{v['var_name']}({v['weight']:.2f})" for v in d['variable_composition'][:3]])
        print(f"  k={d['latent_k']:>4}  freq={d['activation_frequency']:.3f}  "
              f"OR={d['mortality_odds_ratio']:.2f}  var_ent={d['variable_entropy']:.3f}  "
              f"temp_ent={d['temporal_entropy']:.3f}")
        print(f"         vars: {vars_str}")

    # Compare to canonical
    print(f"\nFor comparison, canonical TI-SAE has:")
    print(f"  Variable entropy: ~0 (exactly 3 vars per latent, W_sparsity_k=3)")
    print(f"  Temporal entropy: varies (Gaussian islands localize in time)")

    print(f"\nSaved to {output_dir / 'latent_dictionary.json'}")


if __name__ == "__main__":
    main()
