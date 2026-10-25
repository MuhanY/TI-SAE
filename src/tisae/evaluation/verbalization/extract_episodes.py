"""
Extract episode data for LLM-as-a-Judge evaluation.

Loads pre-computed attribution maps, extracts TI-SAE concepts on the fly
(requires PredSelector + SAE forward), and builds structured episode dicts.

Usage:
    python -m tisae.evaluation.verbalization.extract_episodes \
        --experiment_dir experiments/<backbone>/<condition>/tisae \
        --ps_dir experiments/<backbone>/<condition>/ps \
        --dataset_path data/unified_48h.pkl \
        --attribution_dir experiments/<backbone>/<condition>/eval/attributions \
        --output ./outputs/episodes_data.json
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tisae.evaluation.utils import load_evaluation_models
from tisae.models.ontology import CHANNEL_DEFINITIONS, create_ontology_mapping
from tisae.models.pred_selector import assemble_island_features
from tisae.dictionary.dictionary_builder import load_pred_selector


class EpisodeExtractor:
    """Extract structured episode data for LLM-as-a-Judge pipeline."""

    def __init__(
        self,
        experiment_dir: str,
        ps_dir: str,
        dataset_path: str,
        device: str = "cuda",
        attribution_dir: Optional[str] = None,
        dictionary_path: Optional[str] = None,
        top_n_concepts: int = 5,
        labeled_only: bool = False,
    ):
        self.experiment_dir = experiment_dir
        self.ps_dir = ps_dir
        self.dataset_path = dataset_path
        self.device = device
        self.attribution_dir = attribution_dir
        self.dictionary_path = dictionary_path
        self.top_n_concepts = top_n_concepts
        self.labeled_only = labeled_only

        self.adapter = None
        self.sae = None
        self.selector = None
        self.dataset = None
        self.var_names = None
        self.H_grid_mean = None
        self.dictionary = None
        self.K_eff = self.C_eff = self.K = self.M = self.T = self.V = None

    def load_models(self):
        """Load adapter, SAE, PredSelector, dataset, and optional dictionary."""
        print("Loading models...")
        models = load_evaluation_models(
            self.experiment_dir, self.dataset_path, device=self.device
        )
        self.adapter = models["adapter"]
        self.sae = models["sae"]
        self.dataset = models["dataset"]
        self.H_grid_mean = models["H_grid_mean"]
        self.var_names = models["var_names"]
        self.T = models["T"]
        self.V = models["V"]
        self.K_eff = models["K_eff"]
        self.C_eff = models["C_eff"]
        self.M = models["M"]
        self.K = self.sae.cfg.K

        self.selector = load_pred_selector(self.ps_dir, device=self.device)
        self.n_global_pred = self.selector.config.n_global_pred

        # Inferred channel assignment
        self.ontology = create_ontology_mapping(self.var_names)
        ch_mask = self.ontology.get_hard_mask()
        W = self.sae.decoder.get_W().detach().cpu().abs()
        self.latent_ch_map = (W @ ch_mask.float().T).argmax(dim=-1).numpy()

        if self.dictionary_path and Path(self.dictionary_path).exists():
            with open(self.dictionary_path) as f:
                raw = json.load(f)
            self.dictionary = {
                f"{e['latent_k']}_{e['channel_c']}": e for e in raw
            }
            print(f"Loaded dictionary with {len(self.dictionary)} entries")

        print(f"Models loaded: K_eff={self.K_eff}, C_eff={self.C_eff}, "
              f"M={self.M}, T={self.T}, V={self.V}, n_global_pred={self.n_global_pred}")

    def load_attributions(self, methods: List[str]) -> Dict[str, Dict[int, np.ndarray]]:
        """Load pre-computed attribution maps from .pt files."""
        if self.attribution_dir is None:
            return {}

        attr_dir = Path(self.attribution_dir)
        result = {}
        for method in methods:
            path = attr_dir / f"{method}.pt"
            if not path.exists():
                print(f"  WARNING: {path} not found, skipping {method}")
                continue
            data = torch.load(path, map_location="cpu", weights_only=False)
            attr_tensor = data["attr"]
            indices = data["indices"]
            obs_grid = data.get("obs_grid")
            lookup = {}
            for i, idx in enumerate(indices):
                attr_tv = attr_tensor[i].numpy()
                if obs_grid is not None:
                    attr_tv = attr_tv * obs_grid[i].numpy()
                lookup[int(idx)] = attr_tv
            result[method] = lookup
            print(f"  Loaded {method}: {len(lookup)} episodes")
        return result

    def select_episodes(
        self, n_episodes: int = 50, seed: int = 42,
        restrict_to: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Stratified selection: n/2 HIGH + n/2 LOW risk from test split.

        Args:
            restrict_to: if provided, only select from these indices
                (e.g., episodes that have saved attributions).
        """
        rng = np.random.RandomState(seed)
        if restrict_to is not None:
            test_indices = np.array(restrict_to)
        else:
            test_indices = np.array(self.dataset.splits["test"])

        print(f"Computing predictions for {len(test_indices)} episodes...")
        preds = []
        batch_size = 64
        for start in range(0, len(test_indices), batch_size):
            end = min(start + batch_size, len(test_indices))
            idx = test_indices[start:end]
            batch = self.dataset.get_batch(idx)
            batch_dev = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                out = self.adapter.forward(batch_dev)
                prob = torch.sigmoid(out["logits"]).cpu().numpy()
            preds.append(prob)

        preds = np.concatenate(preds)
        high_mask = preds > 0.5
        high_idx = test_indices[high_mask]
        low_idx = test_indices[~high_mask]

        n_half = n_episodes // 2
        rng.shuffle(high_idx)
        rng.shuffle(low_idx)
        selected = np.concatenate([high_idx[:n_half], low_idx[:n_half]])

        print(f"Selected {len(selected)}: {min(n_half, len(high_idx))} HIGH + "
              f"{min(n_half, len(low_idx))} LOW")
        return selected

    @torch.no_grad()
    def extract_tisae_concepts(self, ep_idx: int) -> Tuple[List[dict], int]:
        """Extract top-N TI-SAE concepts for one episode."""
        batch = self.dataset.get_batch(np.array([ep_idx]))
        batch_dev = {k: v.to(self.device) for k, v in batch.items()}
        obs_mask = batch_dev["obs_mask"]
        tau_np = batch["tau"][0].numpy()
        varis_np = batch["varis"][0].numpy()
        om_np = batch["obs_mask"][0].numpy()
        vr_np = batch["values_raw"][0].numpy()

        K_eff, C_eff, M, T, V = self.K_eff, self.C_eff, self.M, self.T, self.V

        # Adapter → H_grid, obs_grid + h_pool
        H_grid, obs_grid, H = self.adapter.extract_hidden(batch_dev, return_event_h=True)
        if self.adapter.extract_layer != -1:
            h_pool = self.adapter.forward(batch_dev, return_pooled=True)["pooled"]
        else:
            h_pool, _ = self.adapter.pool_events(H, obs_mask)

        # Residualize
        if self.H_grid_mean is not None:
            H_grid_input = (H_grid - self.H_grid_mean.unsqueeze(0)) * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

        # SAE encoder
        A0 = self.sae.encoder(H_grid_input, obs_grid)
        A0_4d = (A0.view(1, C_eff, K_eff, T).permute(0, 2, 3, 1)
                 if C_eff > 1 else A0.unsqueeze(-1))

        mu, sigma, gate = self.sae.island_params(A0_4d)
        g = self.sae.gaussian_mask.build_gaussians(mu, sigma.detach())
        W_dec = self.sae.decoder.get_W().detach()

        # PredSelector scoring
        include_obs = (self.selector.n_island_features != 6 + V)
        features = assemble_island_features(
            A0_4d, g, mu, sigma, gate, obs_grid, W_dec,
            None, K_eff, C_eff, include_obs_profile=include_obs,
        )
        s_pred = self.selector(features, h_pool)
        weights = F.softmax(s_pred.view(1, -1), dim=-1) * self.n_global_pred
        weights_4d = weights.view(1, K_eff, C_eff, M)

        # Aggregate per latent
        latent_weights = {}
        latent_islands = defaultdict(list)
        for k in range(K_eff):
            for c in range(C_eff):
                total_w = 0.0
                for m in range(M):
                    w = weights_4d[0, k, c, m].item()
                    total_w += w
                    mu_val = mu[0, k, c, m].item()
                    sig_val = sigma[0, k, c, m].item()
                    latent_islands[(k, c)].append({
                        "m": m, "weight": round(w, 4),
                        "mu": round(mu_val, 1), "sigma": round(sig_val, 1),
                        "t_start": max(0, int(mu_val - 2 * sig_val)),
                        "t_end": min(T - 1, int(mu_val + 2 * sig_val)),
                    })
                latent_weights[(k, c)] = total_w

        if self.labeled_only and self.dictionary:
            candidates = {
                kc: w for kc, w in latent_weights.items()
                if (entry := self.dictionary.get(f"{kc[0]}_{kc[1]}"))
                and entry.get("clinical_label")
            }
        else:
            candidates = latent_weights
        top_n = sorted(candidates.items(), key=lambda x: -x[1])[:self.top_n_concepts]
        covered_cells = set()
        W_np = W_dec.cpu().numpy()

        concepts = []
        for rank, ((k_idx, c_idx), total_w) in enumerate(top_n):
            k_flat = c_idx * K_eff + k_idx if C_eff > 1 else k_idx
            w_k = W_np[k_flat]
            top_vi = np.argsort(np.abs(w_k))[-5:][::-1]
            top_vars = [(self.var_names[vi], round(float(w_k[vi]), 3))
                        for vi in top_vi if abs(w_k[vi]) > 0.01]

            ch_idx = int(self.latent_ch_map[k_flat])
            ch_name = CHANNEL_DEFINITIONS[ch_idx].name if ch_idx < len(CHANNEL_DEFINITIONS) else "Other"

            islands = sorted(latent_islands[(k_idx, c_idx)], key=lambda x: -x["weight"])
            for isl in islands:
                isl_vals = {}
                for vn, vw in top_vars[:3]:
                    vi_idx = self.var_names.index(vn) if vn in self.var_names else -1
                    if vi_idx >= 0:
                        obs_in_win = []
                        for ev in range(len(varis_np)):
                            if (om_np[ev] > 0 and varis_np[ev] == vi_idx
                                    and isl["t_start"] <= tau_np[ev] <= isl["t_end"]):
                                obs_in_win.append({
                                    "t": int(tau_np[ev]),
                                    "value_raw": round(float(vr_np[ev]), 2),
                                })
                                covered_cells.add((int(tau_np[ev]), vi_idx))
                        isl_vals[vn] = obs_in_win
                isl["values"] = isl_vals

            clinical_label = clinical_description = None
            if self.dictionary:
                entry = self.dictionary.get(f"{k_idx}_{c_idx}")
                if entry:
                    clinical_label = entry.get("clinical_label")
                    clinical_description = entry.get("clinical_description")

            concepts.append({
                "rank": rank, "total_weight": round(total_w, 3),
                "k_eff": int(k_idx), "c_eff": int(c_idx),
                "channel": ch_name,
                "clinical_label": clinical_label,
                "clinical_description": clinical_description,
                "top_variables": top_vars, "islands": islands,
            })

        return concepts, len(covered_cells)

    def extract_top_k_cells(self, attr_tv: np.ndarray, ep_idx: int, top_k: int) -> List[dict]:
        """Extract top-K attributed cells from a [T, V] attribution map."""
        batch = self.dataset.get_batch(np.array([ep_idx]))
        tau_np = batch["tau"][0].numpy()
        varis_np = batch["varis"][0].numpy()
        om_np = batch["obs_mask"][0].numpy()
        vr_np = batch["values_raw"][0].numpy()

        flat = attr_tv.flatten()
        top_indices = np.argsort(flat)[-top_k:][::-1]

        cells = []
        for fi in top_indices:
            ti, vi = divmod(fi, self.V)
            raw_val = None
            for ev in range(len(varis_np)):
                if om_np[ev] > 0 and varis_np[ev] == vi and tau_np[ev] == ti:
                    raw_val = round(float(vr_np[ev]), 2)
                    break
            cells.append({
                "t": int(ti), "variable": self.var_names[vi],
                "importance": round(float(flat[fi]), 4), "value_raw": raw_val,
            })
        return cells

    def compute_matched_k(self, tisae_n_cells_list: List[int]) -> int:
        """Compute K for attribution methods based on TI-SAE coverage."""
        avg_cells = np.mean(tisae_n_cells_list)
        k = max(5, int(round(avg_cells)))
        print(f"Budget matching: TI-SAE covers {avg_cells:.1f} obs cells -> K={k}")
        return k

    def _extract_raw_values(self, ep_idx: int) -> List[dict]:
        batch = self.dataset.get_batch(np.array([ep_idx]))
        tau_np = batch["tau"][0].numpy()
        varis_np = batch["varis"][0].numpy()
        vb_np = batch["value_bins"][0].numpy()
        om_np = batch["obs_mask"][0].numpy()
        vr_np = batch["values_raw"][0].numpy()

        raw_all = []
        for ev in range(len(varis_np)):
            if om_np[ev] > 0:
                raw_all.append({
                    "t": int(tau_np[ev]),
                    "variable": self.var_names[int(varis_np[ev])],
                    "value_raw": round(float(vr_np[ev]), 2),
                    "bin": int(vb_np[ev]),
                })
        raw_all.sort(key=lambda x: (x["t"], x["variable"]))
        return raw_all

    def _get_pred_prob(self, ep_idx: int) -> float:
        batch = self.dataset.get_batch(np.array([ep_idx]))
        batch_dev = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.adapter.forward(batch_dev)
            return torch.sigmoid(out["logits"]).item()

    def extract_all(
        self, indices: np.ndarray, methods: List[str],
        attributions: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Build all episode data dicts."""
        if attributions is None:
            attributions = {}

        print(f"\nExtracting TI-SAE concepts for {len(indices)} episodes...")
        tisae_data = {}
        tisae_n_cells = []
        for i, ep_idx in enumerate(indices):
            concepts, n_obs = self.extract_tisae_concepts(int(ep_idx))
            tisae_data[int(ep_idx)] = concepts
            tisae_n_cells.append(n_obs)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(indices)} done")

        if top_k is None:
            top_k = self.compute_matched_k(tisae_n_cells)

        print(f"\nBuilding episode dicts (top_k={top_k})...")
        episodes = []
        for i, ep_idx in enumerate(indices):
            ep_idx_int = int(ep_idx)
            pred_prob = self._get_pred_prob(ep_idx_int)
            raw_all = self._extract_raw_values(ep_idx_int)

            attr_cells = {}
            for method in methods:
                attr_tv = attributions.get(method, {}).get(ep_idx_int)
                if attr_tv is not None:
                    attr_cells[method] = self.extract_top_k_cells(attr_tv, ep_idx_int, top_k)

            episodes.append({
                "ep_idx": ep_idx_int,
                "pred_prob": round(pred_prob, 4),
                "risk_level": "HIGH" if pred_prob > 0.5 else "LOW",
                "n_observations": len(raw_all),
                "raw_values_all": raw_all,
                "attribution_cells": attr_cells,
                "tisae_top5_latents": tisae_data[ep_idx_int],
                "tisae_n_obs_cells": tisae_n_cells[i],
            })
        return episodes

    def save(self, episodes: List[dict], output_path: str):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(episodes, f, indent=2)
        print(f"Saved {len(episodes)} episodes to {out}")

        metadata = [{
            "ep_idx": ep["ep_idx"],
            "actual_label": int(self.dataset.y[ep["ep_idx"]]),
            "pred_prob": ep["pred_prob"],
            "risk_level": ep["risk_level"],
        } for ep in episodes]
        meta_path = out.parent / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {meta_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract episode data for LLM-as-a-Judge evaluation.")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--ps_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--attribution_dir", default=None)
    parser.add_argument("--dictionary_path", default=None)
    parser.add_argument("--methods", nargs="+",
                        default=["attention", "ig", "tsr"])
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_n_concepts", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./outputs/episodes_data.json")
    return parser.parse_args()


if __name__ == "__main__":
    from tisae.utils.seeding import set_seed
    args = parse_args()
    set_seed(args.seed)

    extractor = EpisodeExtractor(
        experiment_dir=args.experiment_dir, ps_dir=args.ps_dir,
        dataset_path=args.dataset_path, device=args.device,
        attribution_dir=args.attribution_dir,
        dictionary_path=args.dictionary_path,
        top_n_concepts=args.top_n_concepts,
    )
    extractor.load_models()
    attributions = extractor.load_attributions(args.methods)
    indices = extractor.select_episodes(args.n_episodes, args.seed)
    episodes = extractor.extract_all(indices, args.methods, attributions, args.top_k)
    extractor.save(episodes, args.output)
