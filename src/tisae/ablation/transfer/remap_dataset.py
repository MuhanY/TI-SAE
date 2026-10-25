"""Remap a target dataset's variable indices to match a source dataset.

Creates a new pickle where varis indices are in source variable space,
V is set to source V, and missing vars simply have no observations.
All existing eval code works unchanged on the output.

Usage:
    python -m tisae.ablation.transfer.remap_dataset \
        --source data/unified_mimic3_48h.pkl \
        --target data/unified_mimic4_48h.pkl \
        --output data/unified_mimic4_48h_m3space.pkl
"""

import argparse
import pickle
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                        help="Source dataset (defines variable space)")
    parser.add_argument("--target", type=str, required=True,
                        help="Target dataset to remap")
    parser.add_argument("--output", type=str, required=True,
                        help="Output remapped pickle")
    args = parser.parse_args()

    print(f"Loading source: {args.source}")
    with open(args.source, "rb") as f:
        src = pickle.load(f)
    print(f"Loading target: {args.target}")
    with open(args.target, "rb") as f:
        tgt = pickle.load(f)

    src_names = src["var_names"]
    tgt_names = tgt["var_names"]
    src_to_idx = {name: i for i, name in enumerate(src_names)}

    shared = [n for n in tgt_names if n in src_to_idx]
    missing = [n for n in tgt_names if n not in src_to_idx]
    print(f"Source V={len(src_names)}, Target V={len(tgt_names)}, "
          f"Shared={len(shared)}, Missing={len(missing)}")
    if missing:
        print(f"  WARNING: target vars not in source (dropped): {missing}")

    # Build remap: tgt_idx -> src_idx
    remap = np.array([src_to_idx[n] for n in tgt_names if n in src_to_idx])
    # For vars not in source, we'd need to drop events
    # but since all M4 vars are in M3, this shouldn't happen
    assert len(missing) == 0, f"Cannot remap vars not in source: {missing}"

    remap_full = np.array([src_to_idx[n] for n in tgt_names])

    # Build source quantile boundaries as array indexed by source var idx
    n_bins = src["n_value_bins"]
    src_qb = src["quantile_boundaries"]

    # Re-quantize: for each event, look up its raw value, find the correct
    # bin using the SOURCE variable's quantile boundaries.
    print("Remapping varis + re-quantizing value_bins...")
    new_varis = []
    new_value_bins = []
    n_requantized = 0
    n_events_total = 0

    for i in range(tgt["N"]):
        v_old = tgt["varis"][i]           # [L] target var indices
        vals_raw = tgt["values_raw"][i]   # [L] raw float values
        v_new = remap_full[v_old]         # [L] source var indices

        # Re-quantize using source boundaries
        bins_new = np.zeros_like(v_old)
        for j in range(len(v_old)):
            src_var_name = src_names[v_new[j]]
            boundaries = src_qb.get(src_var_name, None)
            if boundaries is not None and len(boundaries) > 0:
                b = np.searchsorted(boundaries, vals_raw[j], side="right") - 1
                bins_new[j] = np.clip(b, 0, n_bins - 1)
            else:
                bins_new[j] = 0
        n_events_total += len(v_old)

        new_varis.append(v_new)
        new_value_bins.append(bins_new)

    print(f"  Re-quantized {n_events_total} events using source boundaries")

    # Also recompute values_zscore using source means/stds
    src_means_stds = src.get("means_stds", {})
    new_values_zscore = []
    if "values_zscore" in tgt:
        print("Re-normalizing values_zscore with source stats...")
        for i in range(tgt["N"]):
            v_new = new_varis[i]
            vals_raw = tgt["values_raw"][i]
            zs = np.zeros_like(vals_raw, dtype=np.float32)
            for j in range(len(v_new)):
                src_var_name = src_names[v_new[j]]
                ms = src_means_stds.get(src_var_name, None)
                if ms is not None:
                    mean = ms["mean"] if isinstance(ms, dict) else ms[0]
                    std = ms["std"] if isinstance(ms, dict) else ms[1]
                    zs[j] = (vals_raw[j] - mean) / max(std, 1e-8)
                else:
                    zs[j] = vals_raw[j]
            new_values_zscore.append(zs)

    # Build output — copy everything, override var-related fields
    out = dict(tgt)
    out["varis"] = new_varis
    out["value_bins"] = new_value_bins
    out["var_names"] = list(src_names)  # use source var list (V=135)
    out["V"] = len(src_names)
    out["var_to_ind"] = src["var_to_ind"]
    out["var_to_ind_hash"] = src["var_to_ind_hash"]
    out["quantile_boundaries"] = src["quantile_boundaries"]
    out["means_stds"] = src_means_stds
    out["n_value_bins"] = n_bins
    if new_values_zscore:
        out["values_zscore"] = new_values_zscore

    print(f"Saving to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    print(f"Done. V={out['V']}, hash={out['var_to_ind_hash']}, "
          f"N={out['N']}")


if __name__ == "__main__":
    main()
