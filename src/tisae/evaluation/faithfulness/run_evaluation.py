"""
Main evaluation script: TISAE vs baselines.

Usage
    python -m tisae.evaluation.faithfulness.run_evaluation \
        --experiment_dir experiments/discrete/mimic3_mortality/tisae \
        --pred_selector_dir experiments/discrete/mimic3_mortality/ps \
        --dataset_path data/unified_mimic3_48h.pkl \
        --output_dir experiments/discrete/mimic3_mortality/eval \
        --n_samples 500 --save_attributions \
        --captum_baseline median \
        --methods attention ig sae pred_selector pred_selector_soft


Computes attribution maps for each method, evaluates faithfulness,
and outputs comparison tables.

Masking strategies for comp/suff:
  - removal: zero out events (obs_mask=0)
  - marginal: replace with population-sampled values
  - conditional: replace with patient's nearest same-variable value

For perturbation-based methods (dynamask, timeshap), the internal perturbation is
recomputed per masking strategy (attribution depends on perturbation).
For non-perturbation methods, attribution is computed once and evaluated with all strategies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from functools import partial
from tqdm import tqdm


from tisae.utils.seeding import set_seed
from tisae.evaluation.utils import load_evaluation_models
from tisae.evaluation.faithfulness.attribution_methods import (
    compute_random_attribution,
    compute_uniform_attribution,
    compute_attention_attribution,
    compute_captum_attribution,
    compute_sae_attribution,
    compute_grad_act_attribution,
    compute_pred_selector_attribution,
    extract_pred_selector_softs,
    compute_dynamask_attribution,
    compute_timeshap_attribution,
    train_fit_generator,
    compute_fit_attribution,
    compute_fit_removal_attribution,
    compute_winit_attribution,
    compute_winit_removal_attribution,
    compute_tsr_attribution,
    compute_concept_attribution,
    build_concept_vectors,
    _build_obs_grid,
    masked_predict,
    marginal_masked_predict,
    conditional_masked_predict,
    compute_median_bins,
)
from tisae.evaluation.faithfulness.baseline_timex import (
    HAS_TXAI, HAS_TXAI_PP, HAS_EXTREMAL,
    train_timex_explainer,
    compute_timex_attribution,
    train_timexpp_explainer,
    compute_timexpp_attribution,
    compute_extremal_mask_attribution,
)
from tisae.evaluation.faithfulness.faithfulness import (
    evaluate_faithfulness,
    evaluate_faithfulness_island,
)


def load_pred_selector(experiment_dir, device='cuda'):
    """Load a trained PredSelector from experiment directory."""
    import yaml
    from dataclasses import fields as dc_fields
    from tisae.models.pred_selector import PredSelector, PredSelectorConfig

    exp_dir = Path(experiment_dir)
    with open(exp_dir / "config.yaml") as f:
        full_config = yaml.safe_load(f)

    ps_cfg_dict = full_config["pred_selector_config"]
    valid_fields = {f.name for f in dc_fields(PredSelectorConfig)}
    config = PredSelectorConfig(**{k: v for k, v in ps_cfg_dict.items() if k in valid_fields})

    n_island_features = full_config["n_island_features"]
    D = full_config["D"]
    C_eff = full_config["C_eff"]

    ckpt = torch.load(exp_dir / "best_model.pt", map_location="cpu")
    selector = PredSelector(config, n_island_features, D, C_eff)
    selector.load_state_dict(ckpt["selector_state_dict"])
    selector = selector.to(device).eval()

    prior_cache = None
    sae_exp_dir = full_config.get("sae_experiment_dir")
    if sae_exp_dir:
        prior_path = Path(sae_exp_dir) / "grad_act_prior.pt"
        if prior_path.exists():
            prior_data = torch.load(prior_path, map_location="cpu")
            prior_cache = prior_data["prior_cache"].float()

    return dict(selector=selector, config=config, prior_cache=prior_cache)


# =============================================================================
# Method choices
# =============================================================================

ALL_METHODS = [
    "random", "uniform", "attention",
    # Captum gradient methods — both embedding-level (all backbones) and
    # value-level (CVE-only). The "best of all viable adaptations" protocol
    # runs both on CVE backbones; the aggregation step picks the highest-AOPC
    # variant per (family, backbone) cell. The legacy bare names ("ig",
    # "gradient_shap", etc.) are kept as backwards-compat aliases for the
    # `_emb` variant.
    "ig", "ig_emb", "ig_value",
    "gradient_shap", "gradient_shap_emb", "gradient_shap_value",
    "grad_x_input", "grad_x_input_emb", "grad_x_input_value",
    "deeplift", "deeplift_emb", "deeplift_value",
    "tsr", "tsr_emb", "tsr_value",
    "dynamask", "dynamask_emb", "dynamask_value",
    "sae", "grad_act",
    "pred_selector", "pred_selector_soft",
    "timeshap",
    "fit_removal", "winit_removal",
    "concept",
    "timex", "timexpp", "extremal_mask",
]

# Method-name → (family, variant) mapping for the aggregation step
# adds `aggregate_best_variant` which uses this to pick the highest-AOPC
# variant per family per backbone for the main paper table).
METHOD_FAMILIES = {
    "ig":                ("ig", "emb"),
    "ig_emb":            ("ig", "emb"),
    "ig_value":          ("ig", "value"),
    "gradient_shap":         ("gradient_shap", "emb"),
    "gradient_shap_emb":     ("gradient_shap", "emb"),
    "gradient_shap_value":   ("gradient_shap", "value"),
    "grad_x_input":         ("grad_x_input", "emb"),
    "grad_x_input_emb":     ("grad_x_input", "emb"),
    "grad_x_input_value":   ("grad_x_input", "value"),
    "deeplift":         ("deeplift", "emb"),
    "deeplift_emb":     ("deeplift", "emb"),
    "deeplift_value":   ("deeplift", "value"),
    "tsr":              ("tsr", "emb"),
    "tsr_emb":          ("tsr", "emb"),
    "tsr_value":        ("tsr", "value"),
    "dynamask":         ("dynamask", "emb"),
    "dynamask_emb":     ("dynamask", "emb"),
    "dynamask_value":   ("dynamask", "value"),
}


def _split_captum_method_variant(method: str):
    """Split a method name like 'ig_value' or 'ig' into (base, variant).
    Returns ('ig', 'value') / ('ig', 'emb') / etc.
    """
    if method.endswith("_value"):
        return method[:-len("_value")], "value"
    if method.endswith("_emb"):
        return method[:-len("_emb")], "emb"
    return method, "emb"  # bare name → embedding variant

SAE_METHODS = {"sae", "grad_act", "pred_selector"}

# Perturbation-based methods: attribution IS computed via perturbation,
# so the internal perturbation must match the evaluation masking strategy.
# These methods recompute attribution per masking strategy.
PERTURBATION_METHODS = {
    "dynamask", "dynamask_emb", "dynamask_value",
    "timeshap",
    "extremal_mask",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate attribution methods")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to Stage 2 experiment directory")
    parser.add_argument("--dataset_path", type=str,
                        default="data/unified_mimic3_48h.pkl")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of test samples to evaluate")
    parser.add_argument("--methods", nargs="+",
                        default=ALL_METHODS,
                        help="Methods to evaluate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: experiments/eval_results/{experiment_name})")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_offset", type=int, default=0,
                        help="Skip first N test samples (for parallel sample splitting)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

    parser.add_argument("--ig_steps", type=int, default=50,
                        help="Number of IG interpolation steps")
    parser.add_argument("--n_concepts", type=int, default=50,
                        help="Number of concepts for ACE")
    parser.add_argument("--g_threshold", type=float, default=0.607,
                        help="Gaussian cutoff for island boundary (0.607 ~ 1 sigma, total width 2sigma)")
    parser.add_argument("--timeshap_nsamples", type=int, default=2000,
                        help="TimeSHAP coalition samples per level")
    parser.add_argument("--dynamask_epochs", type=int, default=100,
                        help="DynaMask optimization epochs per sample")
    parser.add_argument("--fit_epochs", type=int, default=300,
                        help="FIT generator training epochs")
    parser.add_argument("--fit_n_samples", type=int, default=10,
                        help="FIT Monte Carlo samples per feature")
    parser.add_argument("--timex_epochs", type=int, default=50,
                        help="TimeX training epochs")
    parser.add_argument("--timex_batch_size", type=int, default=128,
                        help="TimeX/TimeX++ training batch size")
    parser.add_argument("--timexpp_epochs", type=int, default=50,
                        help="TimeX++ training epochs")
    parser.add_argument("--extremal_mask_epochs", type=int, default=500,
                        help="ExtremalMask per-sample optimization epochs")
    parser.add_argument("--tsr_method", type=str, default="IG",
                        choices=["IG", "GRAD", "GS"],
                        help="Base saliency method for TSR")
    parser.add_argument("--captum_baseline", type=str, default="zero",
                        choices=["zero", "median"],
                        help="Baseline for gradient-based methods (IG, GradSHAP, DeepLIFT, TSR). "
                             "'zero': zero embedding. 'median': per-variable median value bin "
                             "(preserves time+variable embeddings).")

    # Masking strategies for comp/suff
    parser.add_argument("--masking_strategies", nargs="+",
                        default=["removal", "marginal", "conditional"],
                        choices=["removal", "marginal", "conditional", "all"],
                        help="Masking strategies for comp/suff. 'all' = all three.")

    # PredSelector
    parser.add_argument("--pred_selector_dir", type=str, default=None,
                        help="PredSelector experiment dir (required if method=pred_selector)")

    # Attribution saving for downstream plausibility + LLM-as-a-Judge pipelines.
    # Default ON because tisae.evaluation.plausibility.run_plausibility hard-requires
    # these .pt files; also useful for re-scoring attributions without recomputing.
    parser.add_argument("--save_attributions", action="store_true", default=True,
                        help="Save raw [N,T,V] attribution tensors as .pt files "
                             "for downstream plausibility + LLM-as-a-Judge evaluation. "
                             "Default: True. Use --no_save_attributions to disable.")
    parser.add_argument("--no_save_attributions", dest="save_attributions",
                        action="store_false",
                        help="Skip saving attribution .pt files (disables downstream "
                             "plausibility analysis).")
    return parser.parse_args()


def get_attribution_fn(method, backbone, sae, T, V, D, args, dataset,
                       fit_data=None,
                       concept_data=None, H_grid_mean=None,
                       pred_selector_data=None,
                       masking_strategy=None, marginal_dist=None, predict_fn=None,
                       median_bins=None,
                       timex_data=None, timexpp_data=None):
    """Return a callable(batch) -> attr[B, T, V] for each method.

    For perturbation methods (dynamask, timeshap), masking_strategy
    and marginal_dist/predict_fn control the internal perturbation semantics.
    """
    baseline_mode = getattr(args, "captum_baseline", "zero")

    if method == "random":
        return lambda batch: compute_random_attribution(batch, T, V)

    elif method == "uniform":
        return lambda batch: compute_uniform_attribution(batch, T, V)

    elif method == "attention":
        return lambda batch: compute_attention_attribution(backbone, batch, T, V)

    elif method in (
        "ig", "ig_emb", "ig_value",
        "gradient_shap", "gradient_shap_emb", "gradient_shap_value",
        "grad_x_input", "grad_x_input_emb", "grad_x_input_value",
        "deeplift", "deeplift_emb", "deeplift_value",
    ):
        base_method, variant = _split_captum_method_variant(method)
        # Skip _value variants on binned backbones (binned input is not
        # differentiable). Returning None would crash; instead we let the
        # caller see a clear ValueError at attribution time.
        # aggregation step will skip cells with no result.
        n_steps = getattr(args, "ig_steps", 50)
        _bm = baseline_mode
        _mb = median_bins
        return lambda batch, m=base_method, bm=_bm, mb=_mb, var=variant: compute_captum_attribution(
            backbone, batch, T, V, method=m, n_steps=n_steps,
            baseline_mode=bm, median_bins=mb, variant=var,
        )

    elif method == "sae":
        return lambda batch: compute_sae_attribution(
            backbone, sae, batch, T, V, H_grid_mean=H_grid_mean,
        )

    elif method == "grad_act":
        return lambda batch: compute_grad_act_attribution(
            backbone, sae, batch, T, V, H_grid_mean=H_grid_mean,
        )

    elif method == "pred_selector":
        if pred_selector_data is None:
            raise ValueError("pred_selector requires pred_selector_data=(selector, prior_cache)")
        selector, prior_cache = pred_selector_data
        return lambda batch: compute_pred_selector_attribution(
            backbone, sae, selector, batch, T, V,
            H_grid_mean=H_grid_mean, prior_cache=prior_cache,
        )

    elif method in ("dynamask", "dynamask_emb", "dynamask_value"):
        _, variant = _split_captum_method_variant(method)
        # Value-level dynamask only supports removal-style perturbation
        # (smooth fade of values_zscore toward population mean). For
        # marginal/conditional on CVE, fall back to embedding-level.
        if variant == "value" and (masking_strategy or "removal") != "removal":
            variant = "emb"
        strat = masking_strategy or "removal"
        md = marginal_dist
        return lambda batch, _s=strat, _md=md, var=variant: compute_dynamask_attribution(
            backbone, batch, T, V, n_epoch=args.dynamask_epochs,
            masking_strategy=_s, marginal_dist=_md, variant=var,
        )

    elif method == "timeshap":
        pf = predict_fn
        return lambda batch, _pf=pf: compute_timeshap_attribution(
            backbone, batch, T, V, nsamples=args.timeshap_nsamples,
            predict_fn=_pf,
        )

    elif method == "fit":
        if fit_data is None:
            raise ValueError("FIT requires pre-trained generator (train_fit_generator)")
        fit_gen, fit_n_bins = fit_data
        return lambda batch: compute_fit_attribution(
            backbone, batch, T, V, fit_gen,
            n_bins=fit_n_bins,
            n_samples=args.fit_n_samples,
        )

    elif method == "fit_removal":
        return lambda batch: compute_fit_removal_attribution(
            backbone, batch, T, V,
        )

    elif method == "winit_removal":
        return lambda batch: compute_winit_removal_attribution(
            backbone, batch, T, V, window_size=10,
        )

    elif method == "winit":
        if fit_data is None:
            raise ValueError("WinIT requires pre-trained generator (train_fit_generator)")
        fit_gen, fit_n_bins = fit_data
        return lambda batch: compute_winit_attribution(
            backbone, batch, T, V, fit_gen,
            n_bins=fit_n_bins,
            n_samples=args.fit_n_samples,
            window_size=10,
        )

    elif method in ("tsr", "tsr_emb", "tsr_value"):
        _, variant = _split_captum_method_variant(method)
        _bm = baseline_mode
        _mb = median_bins
        return lambda batch, bm=_bm, mb=_mb, var=variant: compute_tsr_attribution(
            backbone, batch, T, V, method=args.tsr_method,
            baseline_mode=bm, median_bins=mb, variant=var,
        )

    elif method == "concept":
        if concept_data is None:
            raise ValueError("Concept method requires precomputed concept_data")
        vectors, importances = concept_data
        return lambda batch: compute_concept_attribution(
            backbone, batch, vectors, importances, T, V
        )

    elif method == "timex":
        if timex_data is None:
            raise ValueError("TimeX requires pre-trained explainer (train_timex_explainer)")
        timex_model, tmean, tstd = timex_data
        return lambda batch: compute_timex_attribution(
            backbone, timex_model, batch, T, V, tmean, tstd)

    elif method == "timexpp":
        if timexpp_data is None:
            raise ValueError("TimeX++ requires pre-trained explainer (train_timexpp_explainer)")
        timexpp_model, tmean, tstd = timexpp_data
        return lambda batch: compute_timexpp_attribution(
            backbone, timexpp_model, batch, T, V, tmean, tstd)

    elif method == "extremal_mask":
        strat = masking_strategy or "removal"
        md = marginal_dist
        return lambda batch, _s=strat, _md=md: compute_extremal_mask_attribution(
            backbone, batch, T, V,
            n_epoch=getattr(args, 'extremal_mask_epochs', 500),
            masking_strategy=_s, marginal_dist=_md)

    else:
        raise ValueError(f"Unknown method: {method}")


def _isnan(v):
    import math as _m
    try:
        return _m.isnan(v)
    except (TypeError, ValueError):
        return True


def format_results_table(all_results: Dict[str, Dict[str, float]]) -> str:
    """Format results as a markdown table (generic, metrics as columns)."""
    if not all_results:
        return "No results to display."

    methods = list(all_results.keys())
    all_keys = set()
    for r in all_results.values():
        all_keys.update(r.keys())
    metrics = sorted(all_keys)

    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|---" + "|---" * len(metrics) + "|"

    rows = [header, separator]
    for method in methods:
        vals = all_results[method]
        row = f"| {method} | " + " | ".join(
            f"{vals.get(m, float('nan')):.4f}" for m in metrics
        ) + " |"
        rows.append(row)

    return "\n".join(rows)


def format_faithfulness_table(all_results: Dict[str, Dict[str, float]]) -> str:
    """Format faithfulness in paper style: one table per strategy,
    methods as rows, Comp/Suff columns grouped by fraction.
    AOPC/NAOPC appended as extra columns if present.
    """
    if not all_results:
        return "No results to display."

    methods = list(all_results.keys())

    # Discover strategies and fractions from keys like "conditional/comp@0.1"
    strategies = set()
    fractions = set()
    for vals in all_results.values():
        for k in vals.keys():
            if "/" not in k:
                continue
            strat, metric = k.split("/", 1)
            strategies.add(strat)
            if "@" in metric and not metric.startswith("i"):
                fractions.add(float(metric.split("@")[1]))
    strategies = sorted(strategies)
    fractions = sorted(fractions)

    if not strategies or not fractions:
        return format_results_table(all_results)

    sections = []
    for strat in strategies:
        lines = [f"### {strat}\n"]

        has_aopc = any(f"{strat}/aopc" in all_results[m] for m in methods)
        has_naopc = any(f"{strat}/naopc" in all_results[m] for m in methods)

        # Cell-level comp/suff
        frac_labels = [f"@{f:.2f}" for f in fractions]
        header = "| Method | " + " | ".join(f"C{fl}" for fl in frac_labels)
        header += " | " + " | ".join(f"S{fl}" for fl in frac_labels)
        if has_aopc:
            header += " | AOPC"
        if has_naopc:
            header += " | NAOPC"
        header += " |"
        n_cols = len(fractions) * 2 + int(has_aopc) + int(has_naopc)
        sep = "|---" + "|---" * n_cols + "|"
        lines.extend([header, sep])

        for method in methods:
            vals = all_results[method]
            comp_vals = [vals.get(f"{strat}/comp@{f}", float("nan")) for f in fractions]
            suff_vals = [vals.get(f"{strat}/suff@{f}", float("nan")) for f in fractions]
            row = f"| {method} | "
            row += " | ".join(f"{v:.3f}" for v in comp_vals)
            row += " | " + " | ".join(f"{v:.3f}" for v in suff_vals)
            if has_aopc:
                row += f" | {vals.get(f'{strat}/aopc', float('nan')):.3f}"
            if has_naopc:
                row += f" | {vals.get(f'{strat}/naopc', float('nan')):.3f}"
            row += " |"
            lines.append(row)

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_name = Path(args.experiment_dir).name
    output_dir = Path(args.output_dir or f"experiments/eval/{exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading models from {args.experiment_dir}...")
    models = load_evaluation_models(args.experiment_dir, args.dataset_path, device=args.device)

    backbone = models["backbone"]
    sae = models["sae"]
    dataset = models["dataset"]
    var_names = models["var_names"]
    H_grid_mean = models.get("H_grid_mean")

    T = models["T"]
    V = models["V"]
    D = models["tisae_config"].D
    C = models["C_eff"]

    print(f"T={T}, V={V}, D={D}, C={C}")

    needs_sae = bool(SAE_METHODS & set(args.methods))
    if needs_sae and H_grid_mean is not None:
        print(f"H_grid_mean loaded: shape={H_grid_mean.shape}, norm={H_grid_mean.norm():.2f}")

    # Test sample indices
    test_indices = np.array(dataset.splits["test"])
    n_eval = min(args.n_samples, len(test_indices))
    if args.sample_offset > 0:
        test_indices = test_indices[args.sample_offset:]
        n_eval = min(args.n_samples, len(test_indices))
    sample_indices = test_indices[:n_eval]
    print(f"Evaluating on {n_eval} test samples (offset={args.sample_offset})")

    # Resolve masking strategies
    masking_strategies = args.masking_strategies
    if "all" in masking_strategies:
        masking_strategies = ["removal", "marginal", "conditional"]
    print(f"Masking strategies: {masking_strategies}")

    # Precompute marginal distributions if needed
    marginal_dist = None
    if "marginal" in masking_strategies or "conditional" in masking_strategies:
        print("Computing per-variable marginal distributions from training data...")
        n_bins = 10
        train_indices_marg = np.array(dataset.splits["train"])
        counts = torch.zeros(V, n_bins)
        n_marg = min(5000, len(train_indices_marg))
        for i in range(0, n_marg, 100):
            end_i = min(i + 100, n_marg)
            batch_m = dataset.get_batch(train_indices_marg[i:end_i])
            for b in range(batch_m['obs_mask'].shape[0]):
                om = batch_m['obs_mask'][b]
                vb_b = batch_m['value_bins'][b][om > 0.5]
                va_b = batch_m['varis'][b][om > 0.5]
                for v_idx in range(V):
                    mask_v = va_b == v_idx
                    if mask_v.any():
                        bc = torch.bincount(vb_b[mask_v].long(), minlength=n_bins)
                        counts[v_idx] += bc[:n_bins].float()
        counts_sum = counts.sum(dim=1, keepdim=True).clamp_min(1)
        marginal_dist = (counts / counts_sum).clamp_min(1e-6).to(args.device)
        print(f"  Marginal dist computed from {n_marg} training samples")

    # Precompute median bins if needed for gradient-based baselines.
    # Include both bare names (legacy) and the new `_emb` / `_value` variants
    # (median bins are only used by `_emb`-variant captum, but the check is
    # cheap and median_bins is just precomputed unused for `_value`).
    median_bins = None
    gradient_methods = {
        "ig", "ig_emb", "ig_value",
        "gradient_shap", "gradient_shap_emb", "gradient_shap_value",
        "grad_x_input", "grad_x_input_emb", "grad_x_input_value",
        "deeplift", "deeplift_emb", "deeplift_value",
        "tsr", "tsr_emb", "tsr_value",
    }
    if args.captum_baseline == "median" and (gradient_methods & set(args.methods)):
        print("Computing per-variable median value bins from training data...")
        median_bins = compute_median_bins(dataset, V).to(args.device)
        print(f"  Median bins computed (range: [{median_bins.min()}, {median_bins.max()}])")

    # Build predict functions for each masking strategy.
    # Bind BOTH `marginal_dist` (binned backbones) and `population_values_zscore`
    # (CVE backbones) to each predict_fn. The adapter's `replace_event_value`
    # picks the right one based on the backbone type and ignores the other.
    population_values_zscore = getattr(dataset, "population_values_zscore", None)
    if population_values_zscore is None and (
        "marginal" in masking_strategies or "conditional" in masking_strategies
    ):
        print("WARNING: dataset has no population_values_zscore — CVE backbones "
              "will fall back to standard-normal sampling for marginal/conditional.")

    predict_fns = {}
    for strat in masking_strategies:
        if strat == "removal":
            predict_fns[strat] = masked_predict
        elif strat == "marginal":
            predict_fns[strat] = partial(
                marginal_masked_predict,
                marginal_dist=marginal_dist,
                population_values_zscore=population_values_zscore,
            )
        elif strat == "conditional":
            predict_fns[strat] = partial(
                conditional_masked_predict,
                marginal_dist=marginal_dist,
                population_values_zscore=population_values_zscore,
            )

    # Pre-train FIT generator if needed (shared by FIT and WinIT)
    fit_data = None
    if "fit" in args.methods or "winit" in args.methods:
        print("Training FIT conditional generator...")
        train_indices = np.array(dataset.splits["train"])
        n_fit_train = min(len(train_indices), 50 * args.batch_size)
        fit_train_idx = train_indices[:n_fit_train]

        fit_data = train_fit_generator(
            dataset, fit_train_idx, T, V,
            device=args.device,
            batch_size=args.batch_size,
            max_epochs=args.fit_epochs,
        )
        print("FIT generator training complete.")

    # Pre-train TimeX explainer if needed
    timex_cache = os.path.join(args.output_dir, "timex_precompute.pt")
    timex_data = None
    if "timex" in args.methods:
        print("Training TimeX explainer...")
        train_indices = np.array(dataset.splits["train"])
        timex_data = train_timex_explainer(
            backbone, dataset, train_indices, T, V,
            device=args.device,
            n_epochs=args.timex_epochs,
            batch_size=args.timex_batch_size,
            cache_path=timex_cache,
        )
        print("TimeX training complete.")

    # Pre-train TimeX++ explainer if needed
    timexpp_data = None
    if "timexpp" in args.methods:
        print("Training TimeX++ explainer...")
        train_indices = np.array(dataset.splits["train"])
        timexpp_data = train_timexpp_explainer(
            backbone, dataset, train_indices, T, V,
            device=args.device,
            n_epochs=args.timexpp_epochs,
            batch_size=args.timex_batch_size,
            cache_path=timex_cache,
        )
        print("TimeX++ training complete.")

    # Build concepts if needed
    concept_data = None
    if "concept" in args.methods:
        print("Building concept vectors...")
        train_indices = np.array(dataset.splits["train"])
        n_concept_batches = min(30, len(train_indices) // args.batch_size)
        concept_batches = []
        for i in range(n_concept_batches):
            idx = train_indices[i * args.batch_size:(i + 1) * args.batch_size]
            batch = dataset.get_batch(idx)
            concept_batches.append(batch)
        concept_data = build_concept_vectors(
            backbone, concept_batches,
            n_concepts=args.n_concepts, device=args.device,
        )
        print("Concept vectors built.")

    # Load PredSelector if needed
    pred_selector_data = None
    if "pred_selector" in args.methods or "pred_selector_soft" in args.methods:
        ps_dir = getattr(args, 'pred_selector_dir', None)
        if ps_dir is None:
            raise ValueError("pred_selector requires --pred_selector_dir")
        print(f"Loading PredSelector from {ps_dir}...")
        ps_models = load_pred_selector(ps_dir, args.device)
        pred_selector_data = (ps_models['selector'], ps_models.get('prior_cache'))
        print(f"PredSelector loaded (n_global_pred={ps_models['config'].n_global_pred})")

    # Build attribution functions
    # Separate island methods from regular — they need special handling
    island_methods = {"pred_selector_soft"}
    regular_methods = [m for m in args.methods if m not in island_methods]
    run_ps_soft = "pred_selector_soft" in args.methods

    # Print estimated time warnings for slow methods
    if "timeshap" in args.methods:
        est_passes = n_eval * 3 * args.timeshap_nsamples
        n_strats = len([s for s in masking_strategies]) if "timeshap" in PERTURBATION_METHODS else 1
        print(f"WARNING: TimeSHAP is slow (~{est_passes} forward passes × {n_strats} strategies for {n_eval} samples)")
    if any(m in args.methods for m in ("dynamask", "dynamask_emb", "dynamask_value")):
        est_passes = n_eval * args.dynamask_epochs * len([0.1, 0.2, 0.3, 0.5])
        n_strats = len(masking_strategies) if "dynamask" in PERTURBATION_METHODS else 1
        print(f"WARNING: DynaMask optimization (~{est_passes} forward passes × {n_strats} strategies for {n_eval} samples)")
    if any(m in args.methods for m in ("tsr", "tsr_emb", "tsr_value")):
        print(f"WARNING: TSR is per-sample with gradient computation (T+V ablations per sample)")
    # Build attribution functions for non-perturbation methods
    # Perturbation methods build per-strategy fns in the eval loop
    attr_fns = {}
    for method in regular_methods:
        if method not in PERTURBATION_METHODS:
            attr_fns[method] = get_attribution_fn(
                method, backbone, sae, T, V, D, args, dataset,
                fit_data=fit_data, concept_data=concept_data,
                H_grid_mean=H_grid_mean,
                pred_selector_data=pred_selector_data,
                median_bins=median_bins,
                timex_data=timex_data, timexpp_data=timexpp_data,
            )

    # Attribution saving setup
    save_attributions = getattr(args, "save_attributions", False)
    if save_attributions:
        attr_save_dir = output_dir / "attributions"
        attr_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Will save attributions to {attr_save_dir}/")

    # Evaluate each method with all masking strategies
    all_faithfulness = {}
    per_sample_faithfulness = {}

    for method in regular_methods:
        is_perturbation = method in PERTURBATION_METHODS

        print(f"\n{'='*60}")
        if is_perturbation:
            print(f"Evaluating: {method} (perturbation — recomputes per strategy)")
        else:
            print(f"Evaluating: {method}")
        print(f"{'='*60}")

        # For perturbation methods, build per-strategy attribution functions
        if is_perturbation:
            strat_attr_fns = {}
            for strat_name in masking_strategies:
                strat_attr_fns[strat_name] = get_attribution_fn(
                    method, backbone, sae, T, V, D, args, dataset,
                    masking_strategy=strat_name,
                    marginal_dist=marginal_dist,
                    predict_fn=predict_fns.get(strat_name),
                    timex_data=timex_data, timexpp_data=timexpp_data,
                )

        # Attribution saving accumulators
        if save_attributions:
            saved_attrs = []
            saved_indices = []
            saved_obs_grids = []
            # Per-strategy accumulators for perturbation methods
            if is_perturbation:
                saved_attrs_per_strat = {s: [] for s in masking_strategies}

        # Accumulators: keyed by "{strategy}/{metric}" for faithfulness
        faith_accum = {}
        n_batches = 0
        t_start = time.time()

        for start in tqdm(range(0, n_eval, args.batch_size), desc=f"{method}"):
            end = min(start + args.batch_size, n_eval)
            idx = sample_indices[start:end]
            batch = dataset.get_batch(idx)
            batch = {k: v.to(args.device) for k, v in batch.items()}

            if is_perturbation:
                # Perturbation method: recompute attribution per strategy,
                # evaluate faithfulness with matched strategy only
                struct_attr = None
                for strat_name in masking_strategies:
                    attr = strat_attr_fns[strat_name](batch)

                    # Sanity check
                    if attr.shape != (len(idx), T, V):
                        print(f"  WARNING: {method}/{strat_name} attr shape {attr.shape} != expected")
                        continue

                    # Use first strategy's attribution for structural metrics
                    if struct_attr is None:
                        struct_attr = attr

                    # Save per-strategy attribution
                    if save_attributions:
                        saved_attrs_per_strat[strat_name].append(attr.cpu())

                    # Faithfulness with matched strategy only
                    faith = evaluate_faithfulness(
                        backbone, batch, attr, args.fractions,
                        predict_fn=predict_fns[strat_name])
                    for k, v in faith.items():
                        faith_accum.setdefault(f"{strat_name}/{k}", []).append(v)

                attr = struct_attr
                if attr is None:
                    continue
            else:
                # Non-perturbation method: compute attribution once,
                # evaluate faithfulness with all strategies
                attr = attr_fns[method](batch)

                # Sanity check
                if attr.shape != (len(idx), T, V):
                    print(f"  WARNING: attr shape {attr.shape} != expected ({len(idx)}, {T}, {V})")
                    continue
                if attr.abs().sum() < 1e-12:
                    print(f"  WARNING: all-zero attribution for batch starting at {start}")

                # Faithfulness for each masking strategy
                for strat_name, pf in predict_fns.items():
                    faith = evaluate_faithfulness(backbone, batch, attr, args.fractions, predict_fn=pf)
                    for k, v in faith.items():
                        faith_accum.setdefault(f"{strat_name}/{k}", []).append(v)

            # Accumulate attributions for saving
            if save_attributions and attr is not None:
                obs_grid = _build_obs_grid(batch, T, V, device=args.device)
                saved_attrs.append(attr.cpu())
                saved_indices.append(idx)
                saved_obs_grids.append(obs_grid.cpu())

            n_batches += 1

        elapsed = time.time() - t_start

        # Save attributions for this method
        if save_attributions and saved_attrs:
            attr_tensor = torch.cat(saved_attrs, dim=0)
            indices_arr = np.concatenate(saved_indices)
            obs_tensor = torch.cat(saved_obs_grids, dim=0)
            save_path = attr_save_dir / f"{method}.pt"
            torch.save({
                "attr": attr_tensor,
                "indices": indices_arr,
                "obs_grid": obs_tensor,
            }, save_path)
            print(f"  Saved attributions: {save_path} ({attr_tensor.shape})")

            # For perturbation methods, also save per-strategy attributions
            if is_perturbation:
                for strat_name, strat_attrs in saved_attrs_per_strat.items():
                    if strat_attrs:
                        strat_tensor = torch.cat(strat_attrs, dim=0)
                        strat_path = attr_save_dir / f"{method}_{strat_name}.pt"
                        torch.save({
                            "attr": strat_tensor,
                            "indices": indices_arr,
                            "obs_grid": obs_tensor,
                        }, strat_path)
                        print(f"  Saved attributions: {strat_path} ({strat_tensor.shape})")

        # Average results.
        # For metrics with a `<metric>_samples` list (comp@f, suff@f), aggregate
        # by flattening all per-sample values and taking the true mean. The
        # prior code averaged per-batch means, which biases results when
        # `n_samples % batch_size != 0` (unequal last batch weighted higher).
        per_sample_flat: Dict[str, list] = {}
        for k, v in faith_accum.items():
            if k.endswith("_samples"):
                per_sample_flat[k[:-len("_samples")]] = [x for sub in v for x in sub]
        averaged: Dict[str, float] = {}
        for k, v in faith_accum.items():
            if k.endswith("_samples"):
                continue
            if k in per_sample_flat and len(per_sample_flat[k]) > 0:
                averaged[k] = float(np.mean(per_sample_flat[k]))
            else:
                averaged[k] = float(np.mean(v))
        all_faithfulness[method] = averaged
        per_sample_faithfulness[method] = {k: [x for sub in v for x in sub] for k, v in faith_accum.items() if k.endswith("_samples")}

        # Derive AOPC and NAOPC per strategy from averaged comp@f values
        faith = all_faithfulness[method]
        for strat in masking_strategies:
            comp_vals = [faith.get(f"{strat}/comp@{f}") for f in args.fractions]
            comp_vals = [v for v in comp_vals if v is not None]
            if comp_vals:
                aopc = float(np.mean(comp_vals))
                faith[f"{strat}/aopc"] = aopc
                p_orig_mean = faith.get(f"{strat}/p_orig_mean", 0)
                faith[f"{strat}/naopc"] = aopc / max(p_orig_mean, 1e-6)
            # Remove p_orig_mean from output (internal only)
            faith.pop(f"{strat}/p_orig_mean", None)

        print(f"  Time: {elapsed:.1f}s ({n_batches} batches)")

    # PredSelector soft-selection evaluation
    if run_ps_soft and pred_selector_data is not None:
        print(f"\n{'='*60}")
        print(f"Evaluating: pred_selector_soft (g_threshold={args.g_threshold})")
        print(f"{'='*60}")

        ps_selector, ps_prior_cache = pred_selector_data
        faith_cell_accum = {}
        faith_island_accum = {}
        ps_soft_attrs = []
        ps_soft_indices = []
        ps_soft_obs = []
        n_batches = 0
        t_start = time.time()

        for start in tqdm(range(0, n_eval, args.batch_size), desc="ps_soft"):
            end = min(start + args.batch_size, n_eval)
            idx = sample_indices[start:end]
            batch = dataset.get_batch(idx)
            batch = {k: v.to(args.device) for k, v in batch.items()}

            attr, islands = extract_pred_selector_softs(
                backbone, sae, ps_selector, batch, T, V,
                H_grid_mean=H_grid_mean, prior_cache=ps_prior_cache,
                g_threshold=args.g_threshold,
            )

            for strat_name, pf in predict_fns.items():
                faith_cell = evaluate_faithfulness(backbone, batch, attr, args.fractions, predict_fn=pf)
                for k, v in faith_cell.items():
                    faith_cell_accum.setdefault(f"{strat_name}/{k}", []).append(v)

                faith_island = evaluate_faithfulness_island(
                    backbone, batch, islands, T, V, args.fractions, predict_fn=pf,
                )
                for k, v in faith_island.items():
                    faith_island_accum.setdefault(f"{strat_name}/{k}", []).append(v)

            if save_attributions:
                obs_grid = _build_obs_grid(batch, T, V, device=args.device)
                ps_soft_attrs.append(attr.cpu())
                ps_soft_indices.append(idx)
                ps_soft_obs.append(obs_grid.cpu())

            n_batches += 1

        if save_attributions and ps_soft_attrs:
            attr_tensor = torch.cat(ps_soft_attrs, dim=0)
            indices_arr = np.concatenate(ps_soft_indices)
            obs_tensor = torch.cat(ps_soft_obs, dim=0)
            save_path = attr_save_dir / "pred_selector_soft.pt"
            torch.save({
                "attr": attr_tensor,
                "indices": indices_arr,
                "obs_grid": obs_tensor,
            }, save_path)
            print(f"  Saved attributions: {save_path} ({attr_tensor.shape})")

        elapsed = time.time() - t_start
        # See main loop: prefer flattened `*_samples` lists for aggregation
        # to avoid unequal-last-batch bias.
        def _avg_from_accum(accum: Dict[str, list]) -> Dict[str, float]:
            per_sample = {
                k[:-len("_samples")]: [x for sub in v for x in sub]
                for k, v in accum.items()
                if k.endswith("_samples")
            }
            out: Dict[str, float] = {}
            for k, v in accum.items():
                if k.endswith("_samples"):
                    continue
                if k in per_sample and len(per_sample[k]) > 0:
                    out[k] = float(np.mean(per_sample[k]))
                else:
                    out[k] = float(np.mean(v))
            return out

        cell_avg = _avg_from_accum(faith_cell_accum)
        island_avg = _avg_from_accum(faith_island_accum)
        per_sample_faithfulness["pred_selector_soft"] = {
            k: [x for sub in v for x in sub]
            for k, v in {**faith_cell_accum, **faith_island_accum}.items()
            if k.endswith("_samples")
        }
        ps_faith = {**cell_avg, **island_avg}
        for strat in masking_strategies:
            comp_vals = [ps_faith.get(f"{strat}/comp@{f}") for f in args.fractions]
            comp_vals = [v for v in comp_vals if v is not None]
            if comp_vals:
                aopc = float(np.mean(comp_vals))
                ps_faith[f"{strat}/aopc"] = aopc
                p_orig_mean = ps_faith.get(f"{strat}/p_orig_mean", 0)
                ps_faith[f"{strat}/naopc"] = aopc / max(p_orig_mean, 1e-6)
            ps_faith.pop(f"{strat}/p_orig_mean", None)

        all_faithfulness["pred_selector_soft"] = ps_faith

        print(f"  Time: {elapsed:.1f}s ({n_batches} batches)")

    # Combine all results
    all_combined = {}
    for method in args.methods:
        all_combined[method] = {}
        all_combined[method].update(all_faithfulness.get(method, {}))

    # Output tables
    print("\n" + "=" * 80)
    print("FAITHFULNESS RESULTS")
    print("=" * 80)
    print(format_faithfulness_table(all_faithfulness))

    # Save results
    results_path = output_dir / "evaluation_results.json"
    save_data = {
        "faithfulness": all_faithfulness,
        "combined": all_combined,
        "config": {
            "experiment_dir": args.experiment_dir,
            "n_samples": n_eval,
            "methods": args.methods,
            "fractions": args.fractions,
            "masking_strategies": masking_strategies,
            "ig_steps": args.ig_steps,
            "T": T, "V": V, "D": D, "C": C,
        },
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save per-sample scores for bootstrap CI computation
    if per_sample_faithfulness:
        samples_path = output_dir / "per_sample_faithfulness.json"
        with open(samples_path, "w") as f:
            json.dump(per_sample_faithfulness, f)
        print(f"Per-sample scores saved to {samples_path}")

    # Save markdown table
    table_path = output_dir / "evaluation_table.md"
    with open(table_path, "w") as f:
        f.write("# Attribution Method Evaluation\n\n")
        f.write(f"T={T}, V={V}, D={D}, C={C}\n\n")
        f.write("## Faithfulness\n\n")
        f.write(format_faithfulness_table(all_faithfulness))
    print(f"Table saved to {table_path}")


if __name__ == "__main__":
    main()
