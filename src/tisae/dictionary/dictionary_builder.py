"""
Latent Dictionary Builder — characterizes TISAE latents for LLM labeling.

Characterizes each (k, c) latent-channel pair across the dataset:
  Phase 1: Variable composition from decoder W
  Phase 2: Dataset-wide scan for activation statistics + PredSelector weights
  Phase 3: Exemplar analysis with dual ranking (SAE activation + PredSelector)
  Phase 4: LLM-ready prompt formatting (requests clinical_label + clinical_description)

Usage:
    python -m tisae.dictionary.dictionary_builder \
        --experiment_dir experiments/<backbone>/<condition>/tisae \
        --dataset_path data/unified_48h.pkl
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from tisae.models.tisae import TISAE
from tisae.models.backbone_adapter import BackboneAdapter


class DictionaryBuilder:
    """Build a latent dictionary from a trained TISAE."""

    def __init__(
        self,
        sae: TISAE,
        adapter: BackboneAdapter,
        dataset,
        var_names: List[str],
        channel_names: List[str],
        device: str = "cuda",
        H_grid_mean: Optional[torch.Tensor] = None,
        selector=None,
    ):
        self.sae = sae
        self.adapter = adapter
        self.dataset = dataset
        self.var_names = var_names
        self.channel_names = channel_names
        self.device = device
        self.H_grid_mean = H_grid_mean
        self.selector = selector

        self.K_eff = sae.K_eff
        self.C_eff = sae.C_eff
        self.K = sae.cfg.K
        self.C = self.C_eff
        self.M = sae.cfg.M
        self.T = sae.cfg.T
        self.V = sae.cfg.V

    def _compute_inferred_channels(self) -> Dict[int, Tuple[int, str]]:
        """Assign each latent to a clinical channel via decoder W affinity.

        Used when C_eff=1 (purity mode). Returns {k: (ch_idx, ch_name)}.
        """
        from tisae.models.ontology import CHANNEL_DEFINITIONS, create_ontology_mapping
        ontology = create_ontology_mapping(self.var_names)
        ch_mask = ontology.get_hard_mask()  # [C, V]
        W = self.sae.decoder.get_W().detach().cpu().abs()
        affinity = W @ ch_mask.float().T  # [K, C]
        assignments = affinity.argmax(dim=-1).numpy()  # [K]
        result = {}
        for k in range(self.K):
            ch_idx = int(assignments[k])
            ch_name = CHANNEL_DEFINITIONS[ch_idx].name if ch_idx < len(CHANNEL_DEFINITIONS) else f"Ch{ch_idx}"
            result[k] = (ch_idx, ch_name)
        return result

    def extract_variable_composition(self, top_n: int = 3) -> Dict[Tuple[int, int], list]:
        """Extract top variables per (k, c) from decoder W."""
        W = self.sae.decoder.get_W().detach().cpu()
        result = {}

        for c in range(self.C):
            for k in range(self.K_eff):
                k_flat = c * self.K_eff + k if self.C_eff > 1 else k
                row = W[k_flat]

                n = min(top_n, row.size(0))
                vals, idx = row.topk(n)
                vars_list = []
                for v_idx, w in zip(idx.tolist(), vals.tolist()):
                    if w < 1e-6:
                        break
                    vars_list.append({
                        "var_name": self.var_names[v_idx] if v_idx < len(self.var_names) else f"Var_{v_idx}",
                        "weight": round(float(w), 4),
                        "var_idx": v_idx,
                    })
                result[(k, c)] = vars_list

        return result

    @torch.no_grad()
    def _run_inference_one(self, ep_idx: int) -> dict:
        """Run TISAE forward pass for a single episode via BackboneAdapter."""
        batch = self.dataset.get_batch(np.array([ep_idx]))
        batch_dev = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        label = float(batch_dev["labels"][0].item())

        H_grid, obs_grid = self.adapter.extract_hidden(batch_dev)

        if self.H_grid_mean is not None:
            H_grid_input = (H_grid - self.H_grid_mean.unsqueeze(0)) * obs_grid.unsqueeze(-1)
        else:
            H_grid_input = H_grid

        out = self.sae(H_grid_input, obs_grid)

        ep_result = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in out.items()
        }
        ep_result["label"] = label
        ep_result["episode_idx"] = ep_idx

        if self.selector is not None:
            A0 = out["A0"]
            B_dim, K_total, T_len = A0.shape

            if self.C_eff > 1:
                A0_4d = A0.view(B_dim, self.C_eff, self.K_eff, T_len).permute(0, 2, 3, 1)
            else:
                A0_4d = A0.unsqueeze(-1)

            mu = out["mu"]
            sigma_det = out["sigma"].detach()
            g = self.sae.gaussian_mask.build_gaussians(mu, sigma_det)
            gate = out.get("gate")

            # h_pool for PredSelector
            if self.adapter.extract_layer != -1:
                out_full = self.adapter.forward(batch_dev, return_pooled=True)
                h_pool = out_full["pooled"]
            else:
                _, obs_grid_ev = self.adapter.extract_hidden(batch_dev)
                H_ev_result = self.adapter.extract_hidden(batch_dev, return_event_h=True)
                H_event = H_ev_result[2]
                obs_mask = batch_dev["obs_mask"]
                h_pool, _ = self.adapter.pool_events(H_event, obs_mask)

            ep_result["_ps_intermediates"] = {
                "A0_4d": A0_4d,
                "g": g,
                "mu": mu,
                "sigma": sigma_det,
                "gate": gate,
                "obs_grid": obs_grid,
                "h_pool": h_pool,
            }

        return ep_result

    @torch.no_grad()
    def _compute_ps_weights(self, intermediates: dict) -> torch.Tensor:
        """Compute PredSelector per-island weights. Returns [K_eff, C_eff, M]."""
        from tisae.models.pred_selector import assemble_island_features

        W_dec = self.sae.decoder.get_W().detach()

        features = assemble_island_features(
            A0_4d=intermediates["A0_4d"],
            g=intermediates["g"],
            mu=intermediates["mu"],
            sigma=intermediates["sigma"],
            gate=intermediates["gate"],
            obs_grid=intermediates["obs_grid"],
            W_dec=W_dec,
            prior_scores=None,
            K_eff=self.K_eff,
            C_eff=self.C_eff,
            include_obs_profile=True,
        )

        s_pred = self.selector(features, intermediates["h_pool"])
        return s_pred[0].cpu()

    def scan_dataset(self, episode_indices: List[int]) -> dict:
        """Run all episodes through TISAE, collect per-(k,c) statistics."""
        per_kc = defaultdict(lambda: {
            "activation_strengths": [],
            "mortality_labels": [],
            "island_params": [],
            "ps_weights": [],
        })
        total_episodes = len(episode_indices)
        total_mortality = 0
        has_ps = self.selector is not None

        print(f"Scanning {total_episodes} episodes "
              f"(K_eff={self.K_eff}, C_eff={self.C_eff}, "
              f"PredSelector={'yes' if has_ps else 'no'})...")

        for ep_idx in tqdm(episode_indices, desc="Scanning"):
            ep = self._run_inference_one(ep_idx)
            label = ep["label"]
            if label > 0.5:
                total_mortality += 1

            sel = ep["sel"][0].numpy() if isinstance(ep["sel"], torch.Tensor) else ep["sel"][0]
            mu = ep["mu"][0].numpy() if isinstance(ep["mu"], torch.Tensor) else ep["mu"][0]
            sigma = ep["sigma"][0].numpy() if isinstance(ep["sigma"], torch.Tensor) else ep["sigma"][0]

            A_hat_raw = ep["A_hat"][0]
            if isinstance(A_hat_raw, torch.Tensor):
                A_hat_raw = A_hat_raw.numpy()
            if self.C_eff > 1:
                A_hat = A_hat_raw.reshape(self.C_eff, self.K_eff, -1)
                A_hat = np.transpose(A_hat, (1, 2, 0))
            else:
                A_hat = A_hat_raw[:, :, np.newaxis]

            ps_weights = None
            if has_ps and "_ps_intermediates" in ep:
                ps_weights = self._compute_ps_weights(ep["_ps_intermediates"])
                ps_weights = ps_weights.numpy()
                del ep["_ps_intermediates"]

            for k in range(self.K_eff):
                for c in range(self.C_eff):
                    sel_kc = sel[k, c]
                    if sel_kc.max() < 0.5:
                        continue
                    strength = float(np.abs(A_hat[k, :, c]).mean())
                    if strength < 1e-6:
                        continue

                    entry = per_kc[(k, c)]
                    entry["activation_strengths"].append((ep_idx, strength))
                    entry["mortality_labels"].append(int(label > 0.5))

                    islands = []
                    for m in range(self.M):
                        if sel_kc[m] > 0.5:
                            islands.append((float(mu[k, c, m]), float(sigma[k, c, m]), m))
                    entry["island_params"].append((ep_idx, islands))

                    if ps_weights is not None:
                        ps_weight = float(sum(
                            ps_weights[k, c, m] for m in range(self.M) if sel_kc[m] > 0.5
                        ))
                        entry["ps_weights"].append((ep_idx, ps_weight))

        return {
            "per_kc": dict(per_kc),
            "total_episodes": total_episodes,
            "total_mortality": total_mortality,
        }

    def extract_exemplar_patterns(
        self,
        scan_results: dict,
        var_composition: dict,
        top_n_episodes: int = 10,
        top_n_vars: int = 3,
    ) -> dict:
        """Extract exemplar episodes with dual ranking (SAE + PredSelector)."""
        per_kc = scan_results["per_kc"]
        result = {}
        has_ps = any(bool(data.get("ps_weights")) for data in per_kc.values())

        print(f"Extracting exemplar patterns for {len(per_kc)} active (k,c) pairs"
              f" (dual ranking: {'yes' if has_ps else 'no'})...")

        for (k, c), data in tqdm(per_kc.items(), desc="Exemplars"):
            if not data["activation_strengths"]:
                continue

            var_comp = var_composition.get((k, c), [])
            track_var_indices = [v["var_idx"] for v in var_comp[:top_n_vars]]
            track_var_names = [v["var_name"] for v in var_comp[:top_n_vars]]
            if not track_var_indices:
                continue

            island_lookup = {ep_idx: islands for ep_idx, islands in data["island_params"]}
            ps_lookup = {ep_idx: w for ep_idx, w in data.get("ps_weights", [])}

            sae_ranked = sorted(data["activation_strengths"], key=lambda x: x[1], reverse=True)
            ps_ranked = sorted(data.get("ps_weights", []), key=lambda x: x[1], reverse=True) if has_ps else []

            exemplar_sae = self._extract_traces(
                sae_ranked[:top_n_episodes], island_lookup, ps_lookup,
                track_var_indices, track_var_names, rank_field="activation_strength",
            )

            exemplar_ps = []
            if ps_ranked:
                strength_lookup = {ep_idx: s for ep_idx, s in data["activation_strengths"]}
                exemplar_ps = self._extract_traces(
                    ps_ranked[:top_n_episodes], island_lookup, ps_lookup,
                    track_var_indices, track_var_names, rank_field="ps_weight",
                    strength_lookup=strength_lookup,
                )

            result[(k, c)] = {
                "exemplar_episodes_sae": exemplar_sae,
                "exemplar_episodes_ps": exemplar_ps,
            }

        return result

    def _extract_traces(
        self,
        ranked_episodes: List[Tuple],
        island_lookup: dict,
        ps_lookup: dict,
        track_var_indices: List[int],
        track_var_names: List[str],
        rank_field: str = "activation_strength",
        strength_lookup: Optional[dict] = None,
    ) -> List[dict]:
        """Extract variable traces from ranked episodes."""
        exemplars = []

        for ep_idx, rank_value in ranked_episodes:
            islands = island_lookup.get(ep_idx, [])
            if not islands:
                continue

            t_min = min(max(0, mu - 2 * sig) for mu, sig, _ in islands)
            t_max = max(min(self.T - 1, mu + 2 * sig) for mu, sig, _ in islands)

            batch = self.dataset.get_batch(np.array([ep_idx]))
            values_raw = batch["values_raw"][0].numpy()
            varis = batch["varis"][0].numpy()
            tau = batch["tau"][0].numpy()
            obs_mask = batch["obs_mask"][0].numpy() > 0.5
            label = float(batch["labels"][0].numpy())

            ep_traces = {}
            for v_idx, v_name in zip(track_var_indices, track_var_names):
                times_v = []
                vals_v = []
                for e in range(len(varis)):
                    if not obs_mask[e]:
                        continue
                    if int(varis[e]) != v_idx:
                        continue
                    t = int(tau[e])
                    if t_min <= t <= t_max:
                        times_v.append(t)
                        vals_v.append(float(values_raw[e]))

                if times_v:
                    sorted_pairs = sorted(zip(times_v, vals_v))
                    ep_traces[v_name] = [
                        {"t": int(t), "val": round(v, 4)} for t, v in sorted_pairs
                    ]

            if rank_field == "activation_strength":
                act_strength = rank_value
                ps_weight = ps_lookup.get(ep_idx, 0.0)
            else:
                ps_weight = rank_value
                act_strength = strength_lookup.get(ep_idx, 0.0) if strength_lookup else 0.0

            exemplars.append({
                "episode_idx": ep_idx,
                "mortality": int(label > 0.5),
                "activation_strength": round(act_strength, 4),
                "ps_weight": round(ps_weight, 4),
                "island_windows": [
                    {"mu": round(mu, 2), "sigma": round(sig, 2), "scale": s}
                    for mu, sig, s in islands
                ],
                "variable_traces": ep_traces,
            })

        return exemplars

    def format_for_llm(self, entry: dict) -> str:
        """Format one (k,c) dictionary entry as LLM prompt."""
        lines = []
        has_channels = self.C > 1

        lines.append("# Context")
        if has_channels:
            ch_list = ", ".join(self.channel_names[:self.C])
            lines.append(
                "We trained a Temporal Island Sparse Autoencoder (TI-SAE) on ICU patient "
                "data from MIMIC-III. The input is 48 hours of irregularly-sampled clinical "
                "time-series per patient (labs, vitals, medications, procedures). The "
                "downstream task is in-hospital mortality prediction. The TI-SAE decomposes "
                f"each patient's clinical trajectory into {self.K_eff} sparse latent "
                f"concepts per channel across {self.C} clinical channels ({ch_list}). "
                "Each latent is restricted to one channel and activates via Gaussian "
                "'islands' at specific time windows."
            )
        else:
            lines.append(
                "We trained a Temporal Island Sparse Autoencoder (TI-SAE) on ICU patient "
                "data from MIMIC-III. The input is 48 hours of irregularly-sampled clinical "
                "time-series per patient (labs, vitals, medications, procedures). The "
                "downstream task is in-hospital mortality prediction. The TI-SAE decomposes "
                f"each patient's clinical trajectory into {self.K} sparse latent concepts "
                f"that self-organize to focus on clinical variable groups. Each latent "
                f"selects a sparse set of variables from {self.V} total clinical variables "
                "and activates via Gaussian 'islands' at specific time windows."
            )
        lines.append("")
        lines.append(
            "ICU data is irregularly sampled: lab tests and vitals are ordered based on "
            "clinical need, NOT on a fixed schedule. A variable with zero observations in "
            "a window means the clinician did not order that test -- this is itself "
            "informative (e.g. stable patients get fewer labs). Sparse traces (2-3 data "
            "points over 48 hours) are normal for many lab values. Vitals (heart rate, "
            "blood pressure) are measured more frequently than labs (creatinine, bilirubin)."
        )
        lines.append("")
        lines.append(
            "Each latent activates via Gaussian \"islands\" with center mu and width sigma. "
            "The temporal footprint of one island is approximately mu +/- 2*sigma (hours). "
            "Islands at different time positions across exemplar episodes reflect "
            "patient-specific timing of the same clinical pattern. Interpret each "
            "island's value patterns within its temporal window."
        )
        lines.append("")

        if has_channels:
            lines.append(f"# Latent k={entry['latent_k']}, Channel: {entry['channel_name']}")
        else:
            lines.append(f"# Latent k={entry['latent_k']}")
        lines.append("")

        lines.append("## Variable Composition (from decoder weights)")
        for i, v in enumerate(entry.get("variable_composition", []), 1):
            lines.append(f"{i}. {v['var_name']} (weight: {v['weight']:.3f})")
        lines.append("")

        lines.append("## Statistics")
        lines.append(f"- Active in {entry['activation_frequency'] * 100:.1f}% of episodes")
        mort_active = entry.get("mortality_rate_when_active", 0)
        mort_base = entry.get("mortality_rate_baseline", 0)
        odds = entry.get("mortality_odds_ratio", 1.0)
        lines.append(f"- Mortality rate when active: {mort_active * 100:.1f}% "
                      f"(baseline: {mort_base * 100:.1f}%, OR: {odds:.2f})")
        lines.append("")

        exemplars_sae = entry.get("exemplar_episodes_sae", [])
        tracked_vars = [v["var_name"] for v in entry.get("variable_composition", [])]
        if exemplars_sae:
            lines.append(f"## Exemplar Episodes (top {len(exemplars_sae)} by SAE activation)")
            lines.append("")
            self._format_exemplars(lines, exemplars_sae, tracked_vars, show_field="activation_strength")

        exemplars_ps = entry.get("exemplar_episodes_ps", [])
        if exemplars_ps:
            lines.append(f"## Exemplar Episodes (top {len(exemplars_ps)} by prediction selector)")
            lines.append("")
            self._format_exemplars(lines, exemplars_ps, tracked_vars, show_field="ps_weight")

        lines.append("## Task")
        lines.append("Provide a clinical interpretation of what this latent captures.")
        lines.append("")
        lines.append("Guidelines:")
        lines.append(
            "- Use standard clinical terminology (e.g., \"Septic Shock\", \"Acute Kidney "
            "Injury\", \"Metabolic Acidosis\") when the variable pattern matches a known "
            "condition or physiological state."
        )
        lines.append(
            "- The label should name a clinical STATE or CONDITION, not describe the "
            "variables or measurement process. If values are within normal ranges, "
            "name the physiological state (e.g., \"Hemodynamic Stability\" not "
            "\"Stable BP Monitoring\")."
        )
        lines.append(
            "- Island timing varies across episodes. Focus on the clinical pattern within "
            "each island window, not its absolute time position."
        )
        lines.append("")
        lines.append("Respond with ONLY a JSON object (no markdown fences, no extra text):")
        lines.append("""{
  "clinical_label": "<2-6 word clinical diagnosis or state>",
  "clinical_description": "<1-3 sentence interpretation of the clinical condition>"
}""")

        return "\n".join(lines)

    def _format_exemplars(
        self,
        lines: List[str],
        exemplars: List[dict],
        tracked_vars: List[str],
        show_field: str = "activation_strength",
    ):
        for ep in exemplars:
            mort_str = "YES" if ep["mortality"] else "NO"
            if show_field == "ps_weight":
                rank_str = f"ps_weight: {ep['ps_weight']:.3f}"
            else:
                rank_str = f"strength: {ep['activation_strength']:.3f}"
            lines.append(f"### Episode {ep['episode_idx']} "
                          f"(mortality: {mort_str}, {rank_str})")

            for isl in ep.get("island_windows", []):
                t_start = max(0, isl["mu"] - 2 * isl["sigma"])
                t_end = min(48, isl["mu"] + 2 * isl["sigma"])
                lines.append(f"Island: hours {t_start:.0f}-{t_end:.0f} "
                              f"(mu={isl['mu']:.1f}, sigma={isl['sigma']:.1f})")

            observed_vars = set()
            for v_name, trace in ep.get("variable_traces", {}).items():
                observed_vars.add(v_name)
                if trace:
                    trace_str = ", ".join(
                        f"t{p['t']}h={p['val']:.2f}" for p in trace[:20]
                    )
                    if len(trace) > 20:
                        trace_str += f" ... ({len(trace)} total)"
                    lines.append(f"- {v_name}: {trace_str}")

            for v_name in tracked_vars:
                if v_name not in observed_vars:
                    lines.append(f"- {v_name}: NOT MEASURED in this window")
            lines.append("")

    def build_dictionary(
        self,
        episode_indices: List[int],
        top_n_episodes: int = 10,
        top_n_vars: int = 3,
        min_activation_freq: float = 0.01,
    ) -> List[dict]:
        """Full pipeline: build the latent dictionary."""
        print("Phase 1: Extracting variable composition from decoder W...")
        var_comp = self.extract_variable_composition(top_n=top_n_vars)

        print("Phase 2: Scanning dataset...")
        scan = self.scan_dataset(episode_indices)
        total_eps = scan["total_episodes"]
        total_mort = scan["total_mortality"]
        baseline_mortality = total_mort / max(total_eps, 1)

        print("Phase 3: Extracting exemplar patterns...")
        exemplar_data = self.extract_exemplar_patterns(
            scan, var_comp,
            top_n_episodes=top_n_episodes,
            top_n_vars=top_n_vars,
        )

        print("Phase 4: Assembling dictionary...")
        inferred_ch = self._compute_inferred_channels() if self.C_eff == 1 else {}
        dictionary = []
        per_kc = scan["per_kc"]

        for (k, c), data in sorted(per_kc.items()):
            n_active = len(data["activation_strengths"])
            freq = n_active / max(total_eps, 1)
            if freq < min_activation_freq:
                continue

            strengths = [s for _, s in data["activation_strengths"]]
            mean_strength = float(np.mean(strengths))

            mort_labels = data["mortality_labels"]
            mort_when_active = sum(mort_labels) / max(len(mort_labels), 1)

            if 0 < baseline_mortality < 1:
                odds_active = mort_when_active / (1 - mort_when_active + 1e-8)
                odds_base = baseline_mortality / (1 - baseline_mortality + 1e-8)
                odds_ratio = odds_active / (odds_base + 1e-8)
            else:
                odds_ratio = 1.0

            all_positions = defaultdict(list)
            for _, islands in data["island_params"]:
                for mu_val, sigma_val, scale in islands:
                    all_positions[scale].append(mu_val)
            mean_positions = []
            for s in sorted(all_positions.keys()):
                mean_positions.append(round(float(np.mean(all_positions[s])), 2))

            ex = exemplar_data.get((k, c), {})

            entry = {
                "latent_k": k,
                "channel_c": c,
                "channel_name": self.channel_names[c] if c < len(self.channel_names) else f"Ch{c}",
                "variable_composition": var_comp.get((k, c), []),
                "activation_frequency": round(freq, 4),
                "mean_activation_strength": round(mean_strength, 4),
                "mortality_rate_when_active": round(float(mort_when_active), 4),
                "mortality_rate_baseline": round(float(baseline_mortality), 4),
                "mortality_odds_ratio": round(float(odds_ratio), 3),
                "mean_island_positions": mean_positions,
                "exemplar_episodes_sae": ex.get("exemplar_episodes_sae", []),
                "exemplar_episodes_ps": ex.get("exemplar_episodes_ps", []),
            }

            if k in inferred_ch:
                entry["inferred_channel_idx"] = inferred_ch[k][0]
                entry["inferred_channel_name"] = inferred_ch[k][1]

            entry["llm_prompt"] = self.format_for_llm(entry)
            dictionary.append(entry)

        print(f"Dictionary built: {len(dictionary)} active (k,c) entries "
              f"out of {self.K_eff * self.C_eff} total pairs")

        return dictionary


def save_dictionary(
    dictionary: List[dict],
    output_dir: str,
    save_llm_prompts: bool = True,
):
    """Save dictionary as JSON + summary markdown + individual LLM prompts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_entries = []
    for entry in dictionary:
        e = {k: v for k, v in entry.items() if k != "llm_prompt"}
        json_entries.append(e)

    with open(out / "latent_dictionary.json", "w") as f:
        json.dump(json_entries, f, indent=2)
    print(f"Saved: {out / 'latent_dictionary.json'}")

    lines = ["# Latent Dictionary Summary", ""]
    lines.append(f"Total active (k,c) pairs: {len(dictionary)}")
    lines.append("")
    lines.append("| k | c | Channel | Top Variables | Freq | Mort OR |")
    lines.append("|---|---|---------|---------------|------|---------|")

    for entry in sorted(dictionary, key=lambda e: e["mortality_odds_ratio"], reverse=True):
        k = entry["latent_k"]
        c = entry["channel_c"]
        ch = entry["channel_name"]
        vars_str = ", ".join(v["var_name"] for v in entry["variable_composition"][:3])
        freq = f"{entry['activation_frequency'] * 100:.1f}%"
        odds = f"{entry['mortality_odds_ratio']:.2f}"
        lines.append(f"| {k} | {c} | {ch} | {vars_str} | {freq} | {odds} |")

    with open(out / "latent_dictionary_summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out / 'latent_dictionary_summary.md'}")

    if save_llm_prompts:
        prompt_dir = out / "llm_prompts"
        prompt_dir.mkdir(exist_ok=True)
        for entry in dictionary:
            k, c = entry["latent_k"], entry["channel_c"]
            prompt = entry.get("llm_prompt", "")
            if prompt:
                fname = prompt_dir / f"latent_k{k}_c{c}.txt"
                with open(fname, "w") as f:
                    f.write(prompt)
        print(f"Saved {len(dictionary)} LLM prompts to: {prompt_dir}")


def load_pred_selector(pred_selector_dir: str, device: str = "cuda"):
    """Load a trained PredSelector from checkpoint."""
    import yaml
    from tisae.models.pred_selector import PredSelector, PredSelectorConfig

    ps_dir = Path(pred_selector_dir)
    with open(ps_dir / "config.yaml") as f:
        ps_config = yaml.safe_load(f)

    ps_cfg_dict = ps_config.get("pred_selector_config", ps_config.get("config", {}))
    ps_cfg = PredSelectorConfig(**{
        k: v for k, v in ps_cfg_dict.items()
        if k in PredSelectorConfig.__dataclass_fields__
    })

    n_island_features = ps_config.get("n_island_features", ps_cfg_dict.get("n_island_features"))
    D = ps_config.get("D", ps_cfg_dict.get("D"))
    C_eff = ps_config.get("C_eff", ps_cfg_dict.get("C_eff", 1))

    selector = PredSelector(ps_cfg, n_island_features, D, C_eff).to(device)
    ckpt = torch.load(ps_dir / "best_model.pt", map_location=device, weights_only=False)
    selector.load_state_dict(ckpt["selector_state_dict"])
    selector.eval()

    return selector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TISAE latent dictionary for LLM labeling")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="TISAE experiment directory")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to UnifiedTimeSeriesDataset pickle")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: experiment_dir/latent_dictionary)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of episodes to scan (default: all test split)")
    parser.add_argument("--top_n_episodes", type=int, default=10)
    parser.add_argument("--top_n_vars", type=int, default=3)
    parser.add_argument("--min_activation_freq", type=float, default=0.01)
    parser.add_argument("--pred_selector_dir", type=str, default=None,
                        help="Optional PredSelector experiment directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    from tisae.utils.seeding import set_seed
    from tisae.evaluation.utils import load_evaluation_models

    args = parse_args()
    set_seed(args.seed)

    models = load_evaluation_models(args.experiment_dir, args.dataset_path, device=args.device)
    adapter = models["adapter"]
    sae = models["sae"]
    dataset = models["dataset"]
    H_grid_mean = models["H_grid_mean"]
    var_names = models["var_names"]
    channel_names = models["channel_names"]

    selector = None
    if args.pred_selector_dir:
        selector = load_pred_selector(args.pred_selector_dir, device=args.device)
        print(f"Loaded PredSelector from {args.pred_selector_dir}")

    builder = DictionaryBuilder(
        sae=sae, adapter=adapter, dataset=dataset,
        var_names=var_names, channel_names=channel_names,
        device=args.device, H_grid_mean=H_grid_mean, selector=selector,
    )

    test_indices = list(dataset.splits["test"])
    if args.n_samples is not None:
        test_indices = test_indices[:args.n_samples]

    dictionary = builder.build_dictionary(
        episode_indices=test_indices,
        top_n_episodes=args.top_n_episodes,
        top_n_vars=args.top_n_vars,
        min_activation_freq=args.min_activation_freq,
    )

    output_dir = args.output_dir or str(Path(args.experiment_dir) / "latent_dictionary")
    save_dictionary(dictionary, output_dir)
