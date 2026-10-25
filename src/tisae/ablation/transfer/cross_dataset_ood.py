"""Cross-dataset OOD generalization: MIMIC-III trained model on MIMIC-IV data.

Tests whether backbone predictions and SAE dictionary concepts transfer
across datasets.

Usage:
    python -m tisae.ablation.cross_dataset_ood \
        --experiment_dir experiments/discrete/mimic3_mortality/tisae \
        --pred_selector_dir experiments/discrete/mimic3_mortality/ps \
        --source_dataset data/unified_48h.pkl \
        --target_dataset data/unified_48h.pkl \
        --n_samples 500 --seed 42 \
        --output_dir experiments_ablation/cross_dataset_ood
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
from tisae.evaluation.utils import load_evaluation_models
from tisae.evaluation.faithfulness.run_evaluation import load_pred_selector
from tisae.models.pred_selector import assemble_island_features
from tisae.ablation.knockout.concept_knockout import concept_knockout


def build_var_remap(source_var_names, target_var_names):
    """Build index remap from target variable space to source variable space.

    Returns: tensor[target_V] where remap[i] = source index for target var i.
    Vars in target but not in source get -1 (should not happen if target ⊂ source).
    """
    src_to_idx = {name: i for i, name in enumerate(source_var_names)}
    remap = []
    missing = []
    for i, name in enumerate(target_var_names):
        if name in src_to_idx:
            remap.append(src_to_idx[name])
        else:
            remap.append(0)  # placeholder, should not happen
            missing.append(name)
    if missing:
        print(f"  WARNING: {len(missing)} target vars not in source: {missing}")
    return torch.tensor(remap, dtype=torch.long)


def remap_batch(batch, var_remap, device):
    """Remap batch varis indices from target space to source space."""
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["varis"] = var_remap[batch["varis"].cpu()].to(device)
    return batch


@torch.no_grad()
def evaluate_prediction(backbone, dataset, var_remap, n_samples, batch_size,
                        device, task="mortality", seed=42):
    """Evaluate backbone prediction on target dataset."""
    test_indices = np.array(dataset.splits["test"])
    rng = np.random.RandomState(seed)
    if len(test_indices) > n_samples:
        test_indices = rng.choice(test_indices, n_samples, replace=False)

    all_logits = []
    all_labels = []

    for start in tqdm(range(0, len(test_indices), batch_size), desc="Prediction"):
        end = min(start + batch_size, len(test_indices))
        idx = test_indices[start:end]
        batch = dataset.get_batch(idx)
        batch = remap_batch(batch, var_remap, device)

        logits = backbone.forward(batch)["logits"]
        all_logits.append(logits.cpu())

        if task == "mortality":
            labels = torch.tensor([dataset.y[i] for i in idx])
        elif task == "readmission":
            labels = torch.tensor([dataset.y_readmission[i] for i in idx])
        else:
            labels = torch.tensor([dataset.y_los_gt7d[i] for i in idx])
        all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits).numpy()
    labels = labels.numpy()

    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    prevalence = labels.mean()

    return {
        "auroc": auroc, "auprc": auprc,
        "prevalence": float(prevalence),
        "n_samples": len(labels),
    }


@torch.no_grad()
def evaluate_dictionary_transfer(backbone, sae, dataset, var_remap,
                                 n_samples, batch_size, device,
                                 H_grid_mean=None, seed=42):
    """Check which latents activate on target data and compare frequencies."""
    cfg = sae.cfg
    K, M = cfg.K, cfg.M
    K_eff, C_eff = sae.K_eff, sae.C_eff

    test_indices = np.array(dataset.splits["test"])
    rng = np.random.RandomState(seed)
    if len(test_indices) > n_samples:
        test_indices = rng.choice(test_indices, n_samples, replace=False)

    # Count latent selection frequency (island selector, not raw encoder)
    latent_sel_counts = torch.zeros(K_eff, device=device)
    n_total = 0

    for start in tqdm(range(0, len(test_indices), batch_size), desc="Dictionary"):
        end = min(start + batch_size, len(test_indices))
        idx = test_indices[start:end]
        batch = dataset.get_batch(idx)
        batch = remap_batch(batch, var_remap, device)
        B = batch["obs_mask"].shape[0]

        H_grid, obs_grid = backbone.extract_hidden(batch)

        if H_grid_mean is not None:
            if H_grid_mean.dim() == 2:
                H_grid_input = H_grid - H_grid_mean.unsqueeze(0).unsqueeze(1)
            else:
                H_grid_input = H_grid - H_grid_mean.unsqueeze(0)
            H_grid_input = H_grid_input * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

        A0 = sae.encoder(H_grid_input, obs_grid)  # [B, K, T]
        T_len = A0.size(2)

        if C_eff > 1:
            A0_4d = A0.view(B, C_eff, K_eff, T_len).permute(0, 2, 3, 1)
        else:
            A0_4d = A0.unsqueeze(-1)

        # Run island parameterization + selection (same as training)
        mu, sigma, gate = sae.island_params(A0_4d)
        g = sae.gaussian_mask.build_gaussians(mu, sigma.detach())
        eff_len = g.sum(dim=-1).clamp_min(1e-6)
        A0_perm = A0_4d.permute(0, 1, 3, 2)
        energy = (A0_perm.unsqueeze(3) * g).sum(dim=-1)
        scores = energy / eff_len.sqrt() * gate
        sel = sae.selector(scores)  # [B, K_eff, C_eff, M]

        # A latent is "selected" if any of its islands is selected
        selected = sel.any(dim=-1).any(dim=-1).float()  # [B, K_eff]
        latent_sel_counts += selected.sum(dim=0)
        n_total += B

    freq = (latent_sel_counts / n_total).cpu().numpy()

    return {
        "n_active_1pct": int((freq >= 0.01).sum()),
        "n_active_5pct": int((freq >= 0.05).sum()),
        "n_dead": int((freq < 0.001).sum()),
        "mean_freq": float(freq.mean()),
        "median_freq": float(np.median(freq)),
        "n_samples": n_total,
        "freq": freq.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset OOD generalization")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--pred_selector_dir", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True,
                        help="Dataset the model was trained on (for var_names)")
    parser.add_argument("--target_dataset", type=str, required=True,
                        help="OOD dataset to evaluate on")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default="mortality")
    args = parser.parse_args()

    device = args.device

    # Load M3-trained models
    models = load_evaluation_models(
        args.experiment_dir, args.source_dataset, device=device)
    backbone = models["backbone"]
    sae = models["sae"]
    H_grid_mean = models.get("H_grid_mean")
    source_var_names = models["var_names"]

    ps_data = load_pred_selector(args.pred_selector_dir, device=device)
    selector = ps_data["selector"]

    # Load M4 dataset
    target_dataset = UnifiedTimeSeriesDataset.load(args.target_dataset)
    target_var_names = target_dataset.var_names

    # Build var remap
    var_remap = build_var_remap(source_var_names, target_var_names)
    n_shared = len(set(source_var_names) & set(target_var_names))
    print(f"\nVariable mapping: {len(target_var_names)} target vars → "
          f"{len(source_var_names)} source vars ({n_shared} shared)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prediction performance
    print("\n=== Prediction Performance (OOD) ===")
    pred_results = evaluate_prediction(
        backbone, target_dataset, var_remap,
        args.n_samples, args.batch_size, device,
        task=args.task, seed=args.seed)
    print(f"AUROC: {pred_results['auroc']:.4f}")
    print(f"AUPRC: {pred_results['auprc']:.4f}")
    print(f"Prevalence: {pred_results['prevalence']:.4f}")

    # 2. Dictionary transfer
    print("\n=== Dictionary Transfer ===")
    dict_results = evaluate_dictionary_transfer(
        backbone, sae, target_dataset, var_remap,
        args.n_samples, args.batch_size, device,
        H_grid_mean=H_grid_mean, seed=args.seed)
    print(f"Active latents (≥1%): {dict_results['n_active_1pct']}")
    print(f"Active latents (≥5%): {dict_results['n_active_5pct']}")
    print(f"Dead latents (<0.1%): {dict_results['n_dead']}")
    print(f"Mean freq: {dict_results['mean_freq']:.4f}")

    # 3. Concept knockout on OOD data
    print("\n=== Concept Knockout (OOD) ===")
    # Wrap target dataset to remap on get_batch
    class RemappedDataset:
        def __init__(self, dataset, var_remap, device):
            self.dataset = dataset
            self.var_remap = var_remap
            self.device = device
            self.splits = dataset.splits
            self.var_names = dataset.var_names
            self.T = dataset.T
            self.V = dataset.V
        def get_batch(self, indices):
            batch = self.dataset.get_batch(indices)
            batch["varis"] = self.var_remap[batch["varis"]]
            return batch

    remapped = RemappedDataset(target_dataset, var_remap, device)
    ko_results = concept_knockout(
        backbone, sae, selector, remapped,
        n_samples=args.n_samples, batch_size=args.batch_size,
        device=device, H_grid_mean=H_grid_mean, seed=args.seed)

    # Print knockout summary
    ps = ko_results["ps_ranked"]
    rd = ko_results["random"]
    print(f"{'k':>6s}  {'PS Δprob':>10s}  {'Rand Δprob':>10s}")
    print("-" * 30)
    for k in [1, 5, 10, 20, 50, 64]:
        sk = str(k)
        if sk in ps and sk in rd:
            print(f"{k:>6d}  {ps[sk]['mean_delta_prob']:10.4f}  "
                  f"{rd[sk]['mean_delta_prob']:10.4f}")

    # Save all results
    output = {
        "prediction": pred_results,
        "dictionary_transfer": {k: v for k, v in dict_results.items() if k != "freq"},
        "dictionary_freq": dict_results["freq"],
        "concept_knockout": ko_results,
        "config": {
            "experiment_dir": args.experiment_dir,
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
            "task": args.task,
            "n_samples": args.n_samples,
            "n_source_vars": len(source_var_names),
            "n_target_vars": len(target_var_names),
            "n_shared_vars": n_shared,
        },
    }
    with open(out_dir / "ood_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_dir / 'ood_results.json'}")


if __name__ == "__main__":
    main()
