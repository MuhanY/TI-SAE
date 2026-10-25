"""Realized composed-mask shapes of PS-selected concepts.

Quantifies on the trained model with no retraining:
(1) how many islands each selected concept composes (windows-per-concept distribution)
(2) the goodness-of-fit of the best single symmetric Gaussian to each realized (clamped-sum) mask.
"""
import argparse, json
from dataclasses import fields
from pathlib import Path
import numpy as np, torch, yaml

ap=argparse.ArgumentParser()
ap.add_argument("--experiment_dir", default="experiments/discrete/mimic3_mortality/tisae")
ap.add_argument("--ps_dir", default="experiments/discrete/mimic3_mortality/ps")
ap.add_argument("--dataset_path", default="data/unified_mimic3_48h.pkl")
ap.add_argument("--n_episodes", type=int, default=300)
ap.add_argument("--out", default="experiments/ablation/mask_shapes.json")
args=ap.parse_args()

from tisae.data.unified_dataset import UnifiedTimeSeriesDataset, collate_fn
from tisae.models.backbone_adapter import load_backbone_adapter
from tisae.models.tisae import TISAE, TISAEConfig
from tisae.evaluation.faithfulness.attribution_methods import extract_pred_selector_softs
from tisae.evaluation.faithfulness.run_evaluation import load_pred_selector

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp=Path(args.experiment_dir)
ds=UnifiedTimeSeriesDataset.load(args.dataset_path); T,V=ds.T,ds.V
dm={"T":T,"V":V,"var_to_ind_hash":getattr(ds,"var_to_ind_hash",None)}
cfg=yaml.safe_load(open(exp/"config.yaml"))
cd=cfg.get("tisae_config",cfg.get("sae_config",{}))
vf={f.name for f in fields(TISAEConfig)}
tcfg=TISAEConfig(**{k:v for k,v in cd.items() if k in vf})
bb=cfg.get("args",{}).get("backbone_checkpoint") or cfg.get("training",{}).get("backbone_checkpoint")
adapter=load_backbone_adapter(bb,device=str(device),dataset_meta=dm).eval()
sae=TISAE(tcfg); ck=torch.load(exp/"best_model.pt",map_location="cpu",weights_only=False)
sae.load_state_dict(ck["sae_state_dict"],strict=False); sae=sae.to(device).eval()
Hgm=ck.get("H_grid_mean"); Hgm=Hgm.to(device) if Hgm is not None else None
psr=load_pred_selector(args.ps_dir,device=str(device)); selector=psr["selector"]; pc=psr.get("prior_cache")
NG=getattr(psr.get("config"),"n_global_pred",64) or 64
print(f"PS selection budget n_global_pred={NG} (truncating to top-{NG} islands per episode)")

test=ds.splits["test"][:args.n_episodes]
tt=np.arange(T)

# best single symmetric Gaussian fit: vectorized basis bank over a mu/sigma grid
# (mu allowed outside the window so edge/monotonic shapes can fit; free amplitude >= 0).
# R^2 = 1 - SSR/SST.
_MUS=np.arange(-10, T+10, 0.5); _SIGS=np.arange(0.5, 20.01, 0.5)
_B=np.array([np.exp(-0.5*((tt-mu)/s)**2) for mu in _MUS for s in _SIGS], dtype=float)  # [n_basis,T]
_BN=(_B*_B).sum(1)
def gauss_r2(g):
    g=g.astype(float); sst=((g-g.mean())**2).sum()
    if sst<1e-9: return 1.0
    A=np.clip((_B@g)/_BN, 0, None)
    ssr=((A[:,None]*_B - g[None,:])**2).sum(1)
    return float(1-ssr.min()/sst)

from collections import Counter
nisl=Counter(); nconcepts=0; r2s=[]; r2_by_isl={1:[],2:[],3:[],4:[]}
for i,si in enumerate(test):
    batch=collate_fn([ds.get_item(int(si))]); batch={k:v.to(device) for k,v in batch.items()}
    _,isl=extract_pred_selector_softs(adapter,sae,selector,batch,T,V,H_grid_mean=Hgm,prior_cache=pc)
    isl=isl[0]
    isl=sorted(isl,key=lambda d:-float(d.get("importance",0)))[:NG]   # PS-SELECTED islands only
    byk={}
    for d in isl:
        byk.setdefault(int(d["k"]),[]).append((float(d["mu"]),float(d["sigma"])))
    for k,islands in byk.items():
        g_raw=np.zeros(T)
        for mu,sg in islands:
            sg=max(sg,0.3); g_raw+=np.exp(-0.5*((tt-mu)/sg)**2)
        g=np.clip(g_raw,0,1)
        if g.max()<0.2: continue
        r2=gauss_r2(g)
        nconcepts+=1; nisl[len(islands)]+=1; r2s.append(r2)
        r2_by_isl.setdefault(len(islands),[]).append(r2)

r2s=np.array(r2s)
print(f"\n=== single-symmetric-Gaussian goodness-of-fit of composed masks ===")
print(f"concepts: {nconcepts} over {len(test)} episodes | islands/concept: {dict(sorted(nisl.items()))}")
print(f"\nR^2 of BEST single symmetric Gaussian fit to each composed mask:")
print(f"  mean={r2s.mean():.3f}  median={np.median(r2s):.3f}  p25={np.percentile(r2s,25):.3f}")
print(f"\nfraction NOT representable by a single symmetric Gaussian:")
for tau in (0.99,0.95,0.90,0.80):
    print(f"  R^2 < {tau:.2f} : {(r2s<tau).mean()*100:5.1f}%")
print(f"\nmean R^2 by #islands used: " + " ".join(f"{k}:{np.mean(v):.3f}(n={len(v)})" for k,v in sorted(r2_by_isl.items()) if v))
json.dump({"n_concepts":nconcepts,"islands_per_concept":dict(nisl),
           "r2_mean":float(r2s.mean()),"r2_median":float(np.median(r2s)),
           "frac_below":{str(t):float((r2s<t).mean()) for t in (0.99,0.95,0.90,0.80)}},
          open(args.out,"w"),indent=2)
print("saved",args.out)
