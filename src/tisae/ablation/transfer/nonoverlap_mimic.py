"""Disjoint-cohort cross-dataset transfer (MIMIC-III -> MIMIC-IV).

MIMIC-III (2001-2012) and MIMIC-IV (2008-2019) share the 2008-2012 MetaVision era.
To remove this, the frozen MIMIC-III model is evaluated on disjoint MIMIC-IV,
split by `anchor_year_group`:

  - DISJOINT (>=2014): strictly after MIMIC-III's 2012 cutoff.
  - OVERLAP  (<=2013): 2008-2013 -> potential patient overlap.

Note: both datasets derive from the same hospital system, so this removes
patient and temporal overlap but not institution-level shift.

Stages:
    # 1) CPU: reconstruct sample->stay_id, join anchor_year_group, save index sets
    #    (reads raw icustays/patients CSVs; set MIMIC4_DIR or pass --icustays/--patients)
    python -m tisae.ablation.transfer.nonoverlap_mimic --stage reconstruct \
        --raw data/raw_mimic4/mimic_iv_48h.pkl \
        --output_dir experiments/ablation/disjoint_transfer

    # 2) frozen MIMIC-III model prediction on full/disjoint/overlap (full sets)
    python -m tisae.ablation.transfer.nonoverlap_mimic --stage predict \
        --experiment_dir experiments/discrete/mimic3_mortality/tisae \
        --source_dataset data/unified_mimic3_48h.pkl \
        --target_dataset data/unified_mimic4_48h.pkl \
        --output_dir experiments/ablation/disjoint_transfer

    # 3) derive m3-space pickles with splits['test'] replaced by the disjoint /
    #    overlap subsets, so the stock run_evaluation / run_plausibility and
    #    concept_knockout can be run unchanged
    python -m tisae.ablation.transfer.nonoverlap_mimic --stage derive_pickles \
        --m3space data/unified_mimic4_48h_m3space.pkl \
        --output_dir experiments/ablation/disjoint_transfer
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Deidentified real-era bins that are strictly after MIMIC-III's 2012 cutoff.
DISJOINT_YEAR_MIN = 2014

_MIMIC4_DIR = os.environ.get("MIMIC4_DIR", "")
DEFAULT_ICUSTAYS = os.path.join(_MIMIC4_DIR, "icu", "icustays.csv.gz")
DEFAULT_PATIENTS = os.path.join(_MIMIC4_DIR, "hosp", "patients.csv.gz")

# Expected pickle split sizes for the canonical 48h MIMIC-IV cohort
EXPECTED_SIZES = {"train": 29606, "val": 7223, "test": 9311, "N": 46140}


def reconstruct_sup_ts_ids(raw_path, max_hours=48.0):
    """Replicate UnifiedTimeSeriesDataset.from_raw sample ordering.

    Returns (sup_ts_ids, n_train, n_val, n_test). sample index i -> sup_ts_ids[i].
    Defaults train_frac=1.0, run='1o1' => no sub-selection (as used to build the canonical pickle).
    """
    with open(raw_path, "rb") as f:
        data, oc, train_ids, val_ids, test_ids = pickle.load(f)

    max_minutes = max_hours * 60
    d = data.loc[(data.minute >= 0) & (data.minute <= max_minutes)].copy()
    # Age outlier fix then static-var drop (mirrors from_raw).
    d.loc[(d.variable == "Age") & (d.value > 200), "value"] = 91.4
    d = d.loc[~d.variable.isin(["Age", "Gender"])]
    # Keep only variables present in the training set (mirrors from_raw).
    train_variables = d.loc[d.ts_id.isin(train_ids)].variable.unique()
    d = d.loc[d.variable.isin(train_variables)]

    curr_ids = d.ts_id.unique()
    train_i = np.intersect1d(train_ids, curr_ids)  # np.intersect1d => sorted
    val_i = np.intersect1d(val_ids, curr_ids)
    test_i = np.intersect1d(test_ids, curr_ids)
    sup = np.concatenate((train_i, val_i, test_i))
    return sup, len(train_i), len(val_i), len(test_i)


def reconstruct_and_join(raw_path, icustays_path, patients_path, out_dir, seed=1234):
    """CPU: map MIMIC-IV test stays -> anchor_year_group, emit disjoint/overlap indices."""
    sup, n_tr, n_va, n_te = reconstruct_sup_ts_ids(raw_path)
    N = len(sup)
    print(f"reconstructed sizes: train/val/test = {n_tr}/{n_va}/{n_te}  N={N}")
    print(f"expected  (pickle):  train/val/test = "
          f"{EXPECTED_SIZES['train']}/{EXPECTED_SIZES['val']}/{EXPECTED_SIZES['test']}  "
          f"N={EXPECTED_SIZES['N']}")
    if (n_tr, n_va, n_te, N) != (
        EXPECTED_SIZES["train"], EXPECTED_SIZES["val"],
        EXPECTED_SIZES["test"], EXPECTED_SIZES["N"],
    ):
        raise RuntimeError(
            "Reconstructed split sizes do not match the pickle. The sample->stay_id "
            "mapping is unsafe; aborting. (Check train_frac/run/max_hours or that "
            "this is the raw pickle the unified build consumed.)")
    print("VALIDATION MATCH: True")

    test_start = n_tr + n_va          # first test sample index
    test_stay_ids = sup[test_start:N].astype(np.int64)  # == sorted(test n events)

    icu = pd.read_csv(icustays_path, usecols=["subject_id", "stay_id"])
    pat = pd.read_csv(patients_path, usecols=["subject_id", "anchor_year_group"])
    stay2subj = dict(zip(icu.stay_id.astype(np.int64), icu.subject_id.astype(np.int64)))
    subj2grp = dict(zip(pat.subject_id.astype(np.int64), pat.anchor_year_group))

    groups, missing = [], 0
    for sid in test_stay_ids:
        su = stay2subj.get(int(sid))
        g = subj2grp.get(su) if su is not None else None
        if g is None:
            missing += 1
        groups.append(g if g is not None else "UNKNOWN")
    groups = np.array(groups)

    def year_start(g):
        try:
            return int(str(g).split(" - ")[0])
        except Exception:
            return -1
    ys = np.array([year_start(g) for g in groups])

    disjoint_mask = ys >= DISJOINT_YEAR_MIN
    overlap_mask = (ys > 0) & (ys < DISJOINT_YEAR_MIN)

    print(f"\nmissing joins: {missing}")
    print(f"anchor_year_group of the {len(test_stay_ids)} test stays:")
    for v, c in zip(*np.unique(groups, return_counts=True)):
        print(f"  {v}: {c} ({100*c/len(test_stay_ids):.1f}%)")
    print(f"DISJOINT (>={DISJOINT_YEAR_MIN}): {disjoint_mask.sum()} "
          f"({100*disjoint_mask.sum()/len(test_stay_ids):.1f}%)")
    print(f"OVERLAP  (<={DISJOINT_YEAR_MIN-1}): {overlap_mask.sum()} "
          f"({100*overlap_mask.sum()/len(test_stay_ids):.1f}%)")

    # Absolute sample indices into the data arrays; shuffle (seeded) so that a
    # downstream test[:N] draw (run_evaluation) is a representative subsample
    # rather than the smallest stay_ids.
    rng = np.random.RandomState(seed)
    dj = test_start + np.where(disjoint_mask)[0]
    ov = test_start + np.where(overlap_mask)[0]
    rng.shuffle(dj)
    rng.shuffle(ov)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "indices.npz",
        disjoint=dj, overlap=ov, test_start=test_start,
        test_stay_ids=test_stay_ids, year_start=ys, year_group=groups,
    )
    summary = {
        "n_test": int(len(test_stay_ids)),
        "n_disjoint": int(disjoint_mask.sum()),
        "n_overlap": int(overlap_mask.sum()),
        "missing_joins": int(missing),
        "disjoint_year_min": DISJOINT_YEAR_MIN,
        "year_group_counts": {
            str(v): int(c) for v, c in zip(*np.unique(groups, return_counts=True))
        },
        "shuffle_seed": seed,
    }
    with open(out_dir / "reconstruction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved indices + summary -> {out_dir}")
    return out_dir / "indices.npz"


def _evaluate_prediction_on_indices(backbone, dataset, var_remap, indices,
                                    device, batch_size, task="mortality"):
    """Full-set prediction AUROC/AUPRC on an explicit index list (no sampling)."""
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    from tqdm import tqdm
    from tisae.ablation.transfer.cross_dataset_ood import remap_batch

    indices = np.asarray(indices)
    logits_all, labels_all = [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc="pred"):
            idx = indices[start:start + batch_size]
            batch = dataset.get_batch(idx)
            batch = remap_batch(batch, var_remap, device)
            logits_all.append(backbone.forward(batch)["logits"].cpu())
            if task == "mortality":
                labels_all.append(torch.tensor([dataset.y[i] for i in idx]))
            elif task == "readmission":
                labels_all.append(torch.tensor([dataset.y_readmission[i] for i in idx]))
            else:
                labels_all.append(torch.tensor([dataset.y_los_gt7d[i] for i in idx]))
    probs = torch.sigmoid(torch.cat(logits_all)).numpy()
    labels = torch.cat(labels_all).numpy()
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "prevalence": float(labels.mean()),
        "n_samples": int(len(labels)),
    }


def predict_subsets(args):
    import torch
    from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
    from tisae.evaluation.utils import load_evaluation_models
    from tisae.ablation.transfer.cross_dataset_ood import build_var_remap

    npz = np.load(Path(args.output_dir) / "indices.npz", allow_pickle=True)
    disjoint, overlap = npz["disjoint"], npz["overlap"]
    test_start = int(npz["test_start"])

    models = load_evaluation_models(args.experiment_dir, args.source_dataset,
                                    device=args.device)
    backbone = models["backbone"]
    source_var_names = models["var_names"]

    target = UnifiedTimeSeriesDataset.load(args.target_dataset)
    var_remap = build_var_remap(source_var_names, target.var_names)
    full_test = np.asarray(target.splits["test"])
    assert full_test.min() == test_start, (
        f"test_start mismatch: pickle={full_test.min()} recon={test_start} "
        "(target_dataset ordering differs from the raw reconstruction)")

    results = {}
    for name, idx in [("full", full_test), ("disjoint", disjoint), ("overlap", overlap)]:
        print(f"\n=== prediction: {name} (n={len(idx)}) ===")
        results[name] = _evaluate_prediction_on_indices(
            backbone, target, var_remap, idx, args.device, args.batch_size,
            task=args.task)
        print(f"  AUROC {results[name]['auroc']:.4f}  "
              f"AUPRC {results[name]['auprc']:.4f}  "
              f"prev {results[name]['prevalence']:.4f}")

    out = Path(args.output_dir) / "prediction_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")
    print(f"\nHEADLINE: full {results['full']['auroc']:.3f} | "
          f"disjoint {results['disjoint']['auroc']:.3f} | "
          f"overlap {results['overlap']['auroc']:.3f}")
    return results


def derive_pickles(args):
    """Write m3-space copies with splits['test'] replaced by disjoint / overlap,
    so the stock run_evaluation / run_plausibility can be run unchanged."""
    npz = np.load(Path(args.output_dir) / "indices.npz", allow_pickle=True)
    subsets = {"disjoint": npz["disjoint"], "overlap": npz["overlap"]}
    with open(args.m3space, "rb") as f:
        base = pickle.load(f)
    out_dir = Path(args.m3space).parent
    stem = Path(args.m3space).stem
    for name, idx in subsets.items():
        d = dict(base)
        d["splits"] = dict(base["splits"])
        d["splits"]["test"] = np.asarray(idx).tolist()
        p = out_dir / f"{stem}_{name}.pkl"
        with open(p, "wb") as f:
            pickle.dump(d, f)
        print(f"wrote {p}  (test={len(idx)})")


def main():
    ap = argparse.ArgumentParser(
        description="Disjoint-cohort MIMIC-III -> MIMIC-IV transfer")
    ap.add_argument("--stage", choices=["reconstruct", "predict", "derive_pickles", "all"],
                    default="all")
    ap.add_argument("--raw", default="data/raw_mimic4/mimic_iv_48h.pkl")
    ap.add_argument("--icustays", default=DEFAULT_ICUSTAYS)
    ap.add_argument("--patients", default=DEFAULT_PATIENTS)
    ap.add_argument("--experiment_dir")
    ap.add_argument("--source_dataset", default="data/unified_mimic3_48h.pkl")
    ap.add_argument("--target_dataset", default="data/unified_mimic4_48h.pkl")
    ap.add_argument("--m3space", default="data/unified_mimic4_48h_m3space.pkl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--task", default="mortality")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.stage in ("reconstruct", "all"):
        reconstruct_and_join(args.raw, args.icustays, args.patients,
                             args.output_dir, seed=args.seed)
    if args.stage in ("predict", "all"):
        if not args.experiment_dir:
            raise SystemExit("--experiment_dir required for predict stage")
        predict_subsets(args)
    if args.stage == "derive_pickles":
        derive_pickles(args)


if __name__ == "__main__":
    main()
