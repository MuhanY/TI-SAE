"""Dictionary structure analysis: separability, groupings, outcome significance, label diversity.

Key design choices:
  - Separability is measured by activation correlation across patients,
    because sparse decoder atoms are near-orthogonal by construction.
  - Statistical significance = active-vs-inactive odds ratio + 95% CI.

Usage:
  python -m tisae.ablation.dict_structure_analysis \
      --experiment_dir experiments/discrete/mimic3_mortality/tisae \
      --dataset data/unified_mimic3_48h.pkl \
      --dictionary experiments/discrete/mimic3_mortality/dictionary/latent_dictionary.json
"""
import argparse, json
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", default="experiments/discrete/mimic3_mortality/tisae")
    ap.add_argument("--dataset", default="data/unified_mimic3_48h.pkl")
    ap.add_argument("--dictionary", default="experiments/discrete/mimic3_mortality/dictionary/latent_dictionary.json",
                    help="latent_dictionary.json (list of entries) or a keyed characterization json")
    ap.add_argument("--n_funcsep", type=int, default=3000, help="patients for activation-correlation separability")
    ap.add_argument("--n_orci", type=int, default=12000, help="patients for OR+CI")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from tisae.evaluation.utils import load_evaluation_models, run_tisae_forward
    from tisae.data.unified_dataset import UnifiedTimeSeriesDataset
    from tisae.models.ontology import VARIABLE_CHANNEL_MAP, CHANNEL_DEFINITIONS
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    m = load_evaluation_models(args.experiment_dir, args.dataset, device=args.device)
    adapter, sae, Hmean = m["backbone"], m["sae"], m.get("H_grid_mean")
    ds = UnifiedTimeSeriesDataset.load(args.dataset)
    vn = list(ds.var_names); T, V = ds.T, ds.V
    W = sae.decoder.get_W().detach().cpu().numpy()          # [K, V]
    dc = json.load(open(args.dictionary))
    if isinstance(dc, list):
        dc = {f"{e['latent_k']}_{e['channel_c']}": e for e in dc}
    keys = list(dc.keys())
    idxs = np.array([int(k.split("_")[0]) for k in keys])   # flat decoder index
    CN = [c.short_name for c in CHANNEL_DEFINITIONS]

    def organ(row):
        ch = VARIABLE_CHANNEL_MAP.get(vn[int(np.argmax(np.abs(row)))], [9])
        return ch[0] if isinstance(ch, list) else ch
    org = np.array([organ(W[i]) for i in idxs])

    # ---- 1) SEPARABILITY: geometric ----
    A = W[idxs]; An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    iu = np.triu_indices(len(idxs), 1); gcos = (An @ An.T)[iu]
    rng = np.random.default_rng(0); Rn = An.copy()
    for i in range(len(Rn)): rng.shuffle(Rn[i])
    gnull = (Rn @ Rn.T)[iu]
    print("=== 1a) GEOMETRIC separability (decoder cosine; CONFOUNDED by sparsity) ===")
    print(f"  active median={np.median(gcos):.3f} mean={gcos.mean():.3f} %>0.5={100*(gcos>0.5).mean():.1f}%  "
          f"| shuffled-null mean={gnull.mean():.3f}")

    # ---- 1b) SEPARABILITY: functional (activation correlation) ----
    tr = np.asarray(ds.splits["train"]); sel = rng.choice(tr, min(args.n_funcsep, len(tr)), replace=False)
    acts = []
    for s in range(0, len(sel), 64):
        b = ds.get_batch(sel[s:s+64]); b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        Hg, obs = adapter.extract_hidden(b)
        Hin = (Hg - Hmean.unsqueeze(0)) * obs.unsqueeze(-1) if Hmean is not None else Hg
        with torch.no_grad(): A0 = sae.encoder(Hin, obs)
        acts.append(A0.amax(dim=-1).cpu().numpy()[:, idxs])
    Am = np.concatenate(acts, 0); Az = (Am - Am.mean(0)) / (Am.std(0) + 1e-9)
    Cc = (Az.T @ Az) / len(Az); cv = np.abs(Cc[iu])
    same = org[iu[0]] == org[iu[1]]
    print("=== 1b) FUNCTIONAL separability (activation corr; CONFOUND-FREE) ===")
    print(f"  overall mean|corr|={cv.mean():.3f}  %>0.5={100*(cv>0.5).mean():.1f}%  within-ch={cv[same].mean():.3f} cross-ch={cv[~same].mean():.3f}")
    print("  in-channel |act-corr| per organ:", {CN[c]: round(float(np.abs(Cc[np.ix_(np.where(org==c)[0], np.where(org==c)[0])][np.triu_indices((org==c).sum(),1)]).mean()), 3)
                                                  for c in range(10) if (org==c).sum() > 1})

    # ---- 2) GROUPINGS ----
    km = KMeans(10, n_init=10, random_state=0).fit(An)
    print("=== 2) GROUPINGS ===")
    print(f"  KMeans(10) silhouette={silhouette_score(An, km.labels_):.3f}  ARI(vs organ)={adjusted_rand_score(org, km.labels_):.3f}")
    gc = (An @ An.T)[iu]
    print(f"  within-organ cos={gc[same].mean():.3f} cross-organ={gc[~same].mean():.3f}")

    # ---- 3) OUTCOME SIGNIFICANCE (OR + 95% CI, active-vs-inactive) ----
    samp = rng.choice(ds.N, min(args.n_orci, ds.N), replace=False)
    act = []; mort = []
    for s in range(0, len(samp), 64):
        b = ds.get_batch(samp[s:s+64]); b = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = run_tisae_forward(adapter, sae, b, Hmean, T, V)
        act.append((out["sel"].amax(-1).amax(-1) >= 0.5).cpu().numpy())
        lab = b["labels"]; mort.append((lab.cpu().numpy() if torch.is_tensor(lab) else np.asarray(lab)).astype(int))
    act = np.concatenate(act); mort = np.concatenate(mort).ravel().astype(int); AC = act[:, idxs]

    def orci(col, y):
        a = ((col == 1) & (y == 1)).sum(); b = ((col == 1) & (y == 0)).sum()
        c = ((col == 0) & (y == 1)).sum(); d = ((col == 0) & (y == 0)).sum()
        a2, b2, c2, d2 = a+.5, b+.5, c+.5, d+.5
        orr = (a2*d2)/(b2*c2); se = np.sqrt(1/a2+1/b2+1/c2+1/d2)
        return orr, np.exp(np.log(orr)-1.96*se), np.exp(np.log(orr)+1.96*se)
    R = [orci(AC[:, j], mort) for j in range(len(idxs))]
    LO = np.array([r[1] for r in R]); HI = np.array([r[2] for r in R])
    sig = (LO > 1) | (HI < 1)
    print("=== 3) OUTCOME SIGNIFICANCE (OR + 95% CI) ===")
    print(f"  mort base={mort.mean():.3f}  significant (CI excl 1): {sig.sum()}/{len(idxs)} ({100*sig.mean():.0f}%)  "
          f"risk={int((LO>1).sum())} protective={int((HI<1).sum())}")

    # ---- 4) LABEL DIVERSITY ----
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer
    labels = [(e.get("clinical_label") or "").strip() for e in dc.values()
              if (e.get("clinical_label") or "").strip()]
    if not labels:
        print("=== 4) LABEL DIVERSITY: skipped (dictionary has no clinical_label fields) ===")
        return
    uniq = len(set(l.lower() for l in labels))
    Vz = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(labels).toarray()
    Vz = Vz / (np.linalg.norm(Vz, axis=1, keepdims=True) + 1e-9)
    lu = np.triu_indices(len(labels), 1); lsim = (Vz @ Vz.T)[lu]
    print("=== 4) LABEL DIVERSITY (surface; NOT a substitute for clinician panel) ===")
    print(f"  unique={uniq}/{len(labels)} ({100*uniq/len(labels):.1f}%)  mean lexical sim={lsim.mean():.3f}  near-dup(>0.8)={int((lsim>0.8).sum())}")
    print(f"  top head-words: {Counter(l.lower().split()[0] for l in labels if l.split()).most_common(5)}")


if __name__ == "__main__":
    main()
