"""Dictionary-only seed reproducibility (Stage A).

Computes dictionary-reproducibility metrics (decoder cosine, variable overlap,
latent energy correlation, temporal firing cosine) on dictionaries retrained across seeds.
Two-stage matching (decoder + temporal) as in seed_reproducibility.

Usage:
    python -m tisae.ablation.robustness.dict_activation_metrics \
        --backbone experiments/discrete/mimic3_mortality/backbone/best_model.pt \
        --dataset_path data/unified_mimic3_48h.pkl \
        --ref 42=experiments/discrete/mimic3_mortality/tisae \
        --seeds 43=experiments/ablation/seed_stability/seed_43/tisae \
                44=experiments/ablation/seed_stability/seed_44/tisae \
                45=experiments/ablation/seed_stability/seed_45/tisae \
                46=experiments/ablation/seed_stability/seed_46/tisae \
        --n_samples 500 \
        --output experiments/ablation/seed_stability/dict_full_metrics.json
"""
import argparse, itertools, json
import numpy as np
import torch
from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.ablation.robustness.seed_reproducibility import (
    load_sae, two_stage_match, random_two_stage_match,
)


@torch.no_grad()
def collect_sae(backbone, sae, hgm, dataset, test_idx, device, bs=8):
    traces, energy = [], []
    for st in range(0, len(test_idx), bs):
        samples = [dataset.get_item(int(i)) for i in test_idx[st:st + bs]]
        batch = collate_fn(samples)
        batch = {k: v.to(device) for k, v in batch.items()}
        H_grid, obs_grid = backbone.extract_hidden(batch)
        if hgm is not None:
            H_grid = (H_grid - hgm.unsqueeze(0)) * obs_grid.unsqueeze(-1)
        out = sae(H_grid, obs_grid)
        traces.append(out["A_hat"].cpu())
        A0 = sae.encoder(H_grid, obs_grid)
        A0_4d = A0.unsqueeze(-1)
        mu, sigma, gate = sae.island_params(A0_4d)
        g = sae.gaussian_mask.build_gaussians(mu, sigma.detach())
        A0_perm = A0_4d.permute(0, 1, 3, 2)
        e = (A0_perm.unsqueeze(3) * g).sum(-1).sum(dim=(2, 3))
        energy.append(e.cpu())
    W = sae.get_W_dec().detach().cpu().numpy()
    return {"traces": torch.cat(traces).numpy(), "energy": torch.cat(energy).numpy(),
            "W_raw": W, "W_normed": W / np.linalg.norm(W, axis=1, keepdims=True).clip(1e-8)}


def metrics(a, b, midx, n_samples, sub=5):
    n = len(midx)
    dcos = float(np.mean([float(np.dot(a["W_normed"][i], b["W_normed"][midx[i]])) for i in range(n)]))
    ov = float(np.mean([1 if len(set(np.abs(a["W_raw"][i]).argsort()[-3:]) &
                                   set(np.abs(b["W_raw"][midx[i]]).argsort()[-3:])) >= 2 else 0
                        for i in range(n)]))
    ec = []
    for i in range(n):
        ea, eb = a["energy"][:, i], b["energy"][:, midx[i]]
        if ea.std() > 1e-8 and eb.std() > 1e-8:
            ec.append(float(np.corrcoef(ea, eb)[0, 1]))
    tc = []
    for i in range(n):
        cs = []
        for ns in range(0, n_samples, sub):
            ta, tb = a["traces"][ns, i, :], b["traces"][ns, midx[i], :]
            na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
            if na > 1e-8 and nb > 1e-8:
                cs.append(float(np.dot(ta, tb) / (na * nb)))
        if cs:
            tc.append(np.mean(cs))
    return {"decoder_cos": dcos, "var_overlap": ov,
            "energy_corr": float(np.mean(ec)) if ec else 0.0,
            "temporal_cos": float(np.mean(tc)) if tc else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--ref", required=True, help="label=dir for reference seed")
    ap.add_argument("--seeds", nargs="+", required=True, help="label=dir entries")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--cos_threshold", type=float, default=0.7)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    dirs = {}
    for e in [args.ref] + args.seeds:
        lab, d = e.split("=", 1)
        dirs[lab] = d

    dataset = UnifiedTimeSeriesDataset.load(args.dataset_path)
    dm = {"T": dataset.T, "V": dataset.V, "var_to_ind_hash": getattr(dataset, "var_to_ind_hash", None)}
    backbone = load_backbone_adapter(args.backbone, device=str(device), dataset_meta=dm)
    backbone.eval()
    test_idx = np.array(dataset.splits["test"][:args.n_samples])

    data = {}
    for lab, d in dirs.items():
        print(f"[collect] {lab} <- {d}", flush=True)
        sae, hgm = load_sae(d)
        sae = sae.to(device).eval()
        if hgm is not None:
            hgm = hgm.to(device)
        data[lab] = collect_sae(backbone, sae, hgm, dataset, test_idx, device)
        del sae
        torch.cuda.empty_cache()

    labs = list(dirs.keys())
    pairs = list(itertools.combinations(labs, 2))
    keys = ["decoder_cos", "var_overlap", "energy_corr", "temporal_cos"]
    acc = {k: [] for k in keys}
    first_ncands = None
    for a, b in pairs:
        midx, ncands = two_stage_match(data[a], data[b], cos_threshold=args.cos_threshold)
        if first_ncands is None:
            first_ncands = (a, b, ncands)
        m = metrics(data[a], data[b], midx, args.n_samples)
        for k in keys:
            acc[k].append(m[k])
        print(f"  {a}-{b}: " + " ".join(f"{k}={m[k]:.3f}" for k in keys))

    a, b, ncands = first_ncands
    rnd = {k: [] for k in keys}
    for _ in range(10):
        rmatch = random_two_stage_match(data[a], data[b], ncands)
        m = metrics(data[a], data[b], rmatch, args.n_samples)
        for k in keys:
            rnd[k].append(m[k])

    result = {"seeds": labs, "n_pairs": len(pairs), "n_samples": int(len(test_idx)),
              "actual": {k: {"mean": float(np.mean(acc[k])), "std": float(np.std(acc[k]))} for k in keys},
              "random_null": {k: float(np.mean(rnd[k])) for k in keys}}
    json.dump(result, open(args.output, "w"), indent=2)
    print("\n=== 1a full dictionary reproducibility ===")
    for k in keys:
        print(f"  {k}: {result['actual'][k]['mean']:.3f} ± {result['actual'][k]['std']:.3f}  (null {result['random_null'][k]:.3f})")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
