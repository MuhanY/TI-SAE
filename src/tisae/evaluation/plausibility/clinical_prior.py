"""
Clinical prior for plausibility evaluation of attribution methods.

Builds a [B, T, V] "clinical severity" map using SOFA scoring
(Vincent et al., 1996). Values_raw is in original units (not normalized).

6 SOFA organ systems with combined scoring for multi-variable organs
(PF ratio, GCS total, MAP + vasopressors). Score = SOFA sub-score / 4.

Usage:
    from tisae.evaluation.plausibility.clinical_prior import (
        compute_clinical_prior, compute_plausibility_metrics,
    )

    prior = compute_clinical_prior(val_grid, obs_grid, var_names, mode="sofa")
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from tisae.models.ontology import VARIABLE_CHANNEL_MAP


# =============================================================================
# SOFA single-variable thresholds (score = n_crossed / 4)
# =============================================================================

# (thresholds, direction, sofa_offset)
# sofa_offset: SOFA sub-score for first threshold crossing.
#   e.g. Urine <500 = SOFA 3 → offset=3, so score = (3+0)/4, (3+1)/4 = 0.75, 1.0
SOFA_SINGLE_VAR: Dict[str, Tuple[List[float], str, int]] = {
    # Coagulation: Platelets → SOFA 1,2,3,4
    "Platelet Count":    ([150, 100, 50, 20],       "below", 1),
    # Liver: Bilirubin → SOFA 1,2,3,4
    "Bilirubin (Total)": ([1.2, 2.0, 6.0, 12.0],   "above", 1),
    # Renal: Creatinine → SOFA 1,2,3,4
    "Creatinine Blood":  ([1.2, 2.0, 3.5, 5.0],    "above", 1),
    # Renal: Urine → SOFA 3,4 only
    "Urine":             ([500, 200],                "below", 3),
}


# =============================================================================
# SOFA multi-variable organ definitions
# =============================================================================

# PF ratio SOFA thresholds: < each → SOFA 1,2,3,4
PF_THRESHOLDS = [400, 300, 200, 100]

# GCS total → SOFA CNS score
# (max_total, sofa_score): if total <= max_total → sofa_score
GCS_SOFA = [(5, 4), (9, 3), (12, 2), (14, 1)]

# CV vasopressor SOFA scores: drug → [(dose_threshold, sofa_score), ...]
# Only drugs in the original SOFA definition (Vincent et al., 1996)
CV_VASO_SCORES = {
    "Dopamine":       [(0, 2), (5, 3), (15, 4)],
    "Norepinephrine": [(0, 3), (0.1, 4)],
    "Epinephrine":    [(0, 3), (0.1, 4)],
}

CV_VASOPRESSOR_VARS = tuple(CV_VASO_SCORES.keys())

MAX_GAP = 6  # max time bins for nearest co-observation

# =============================================================================
# Helper: nearest co-observation
# =============================================================================

def _find_nearest(obs_times: np.ndarray, target_t: int, max_gap: int) -> Optional[int]:
    """Find nearest observed time to target_t within max_gap."""
    if len(obs_times) == 0:
        return None
    gaps = np.abs(obs_times - target_t)
    best = gaps.argmin()
    if gaps[best] <= max_gap:
        return int(obs_times[best])
    return None


# =============================================================================
# Multi-variable organ scoring
# =============================================================================

def _score_respiratory(
    val_np: np.ndarray, obs_np: np.ndarray,
    po2_idx: int, fio2_idx: int,
    B: int, T: int,
) -> Dict[int, np.ndarray]:
    """Score respiratory via PF ratio → SOFA 0-4 / 4."""
    scores = {}
    for idx in (po2_idx, fio2_idx):
        if idx >= 0:
            scores[idx] = np.zeros((B, T))

    if po2_idx < 0 or fio2_idx < 0:
        return scores

    for b in range(B):
        po2_times = np.where(obs_np[b, :, po2_idx] > 0.5)[0]
        fio2_times = np.where(obs_np[b, :, fio2_idx] > 0.5)[0]

        # All time bins where either PO2 or FiO2 is observed
        all_times = set(po2_times.tolist()) | set(fio2_times.tolist())

        for t in all_times:
            # Get PO2 value
            po2_val = None
            if obs_np[b, t, po2_idx] > 0.5:
                po2_val = val_np[b, t, po2_idx]
                po2_t = t
            else:
                po2_t = _find_nearest(po2_times, t, MAX_GAP)
                if po2_t is not None:
                    po2_val = val_np[b, po2_t, po2_idx]

            # Get FiO2 value
            fio2_val = None
            if obs_np[b, t, fio2_idx] > 0.5:
                fio2_val = val_np[b, t, fio2_idx]
            else:
                fio2_t = _find_nearest(fio2_times, t, MAX_GAP)
                if fio2_t is not None:
                    fio2_val = val_np[b, fio2_t, fio2_idx]

            if po2_val is None or fio2_val is None or fio2_val < 0.01:
                continue

            pf = po2_val / fio2_val
            sofa = 0
            for thresh in PF_THRESHOLDS:
                if pf < thresh:
                    sofa += 1
            score = sofa / 4.0

            # Assign to observed cells at this time bin
            if obs_np[b, t, po2_idx] > 0.5:
                scores[po2_idx][b, t] = score
            if obs_np[b, t, fio2_idx] > 0.5:
                scores[fio2_idx][b, t] = score

    return scores


def _score_cns(
    val_np: np.ndarray, obs_np: np.ndarray,
    eye_idx: int, verb_idx: int, mot_idx: int,
    B: int, T: int,
) -> Dict[int, np.ndarray]:
    """Score CNS via GCS total → SOFA CNS 0-4 / 4."""
    idxs = [eye_idx, verb_idx, mot_idx]
    scores = {idx: np.zeros((B, T)) for idx in idxs if idx >= 0}
    if any(i < 0 for i in idxs):
        return scores

    for b in range(B):
        obs_per = [np.where(obs_np[b, :, idx] > 0.5)[0] for idx in idxs]

        all_times = set()
        for ot in obs_per:
            all_times.update(ot.tolist())

        for t in all_times:
            vals = []
            for comp_i, idx in enumerate(idxs):
                if obs_np[b, t, idx] > 0.5:
                    vals.append(val_np[b, t, idx])
                else:
                    nearest = _find_nearest(obs_per[comp_i], t, MAX_GAP)
                    if nearest is not None:
                        vals.append(val_np[b, nearest, idx])

            if len(vals) != 3:
                continue

            total = sum(vals)
            sofa = 0
            for max_total, sofa_score in GCS_SOFA:
                if total <= max_total:
                    sofa = sofa_score
                    break
            score = sofa / 4.0

            for idx in idxs:
                if obs_np[b, t, idx] > 0.5:
                    scores[idx][b, t] = score

    return scores


def _score_cardiovascular(
    val_np: np.ndarray, obs_np: np.ndarray,
    var_names: List[str],
    mbp_idx: int, sbp_idx: int, dbp_idx: int,
    vaso_idxs: Dict[str, int],
    B: int, T: int,
) -> Dict[int, np.ndarray]:
    """Score cardiovascular via MAP + vasopressors → SOFA CV 0-4 / 4."""
    all_cv_idxs = [i for i in [mbp_idx, sbp_idx, dbp_idx] + list(vaso_idxs.values()) if i >= 0]
    scores = {idx: np.zeros((B, T)) for idx in all_cv_idxs}

    for b in range(B):
        mbp_times = np.where(obs_np[b, :, mbp_idx] > 0.5)[0] if mbp_idx >= 0 else np.array([])
        sbp_times = np.where(obs_np[b, :, sbp_idx] > 0.5)[0] if sbp_idx >= 0 else np.array([])
        dbp_times = np.where(obs_np[b, :, dbp_idx] > 0.5)[0] if dbp_idx >= 0 else np.array([])

        all_times = set()
        for idx in all_cv_idxs:
            all_times.update(np.where(obs_np[b, :, idx] > 0.5)[0].tolist())

        for t in all_times:
            # Step 1: Get MAP
            map_val = None
            if mbp_idx >= 0 and obs_np[b, t, mbp_idx] > 0.5:
                map_val = val_np[b, t, mbp_idx]
            elif (sbp_idx >= 0 and dbp_idx >= 0 and
                  obs_np[b, t, sbp_idx] > 0.5 and obs_np[b, t, dbp_idx] > 0.5):
                map_val = (val_np[b, t, sbp_idx] + 2 * val_np[b, t, dbp_idx]) / 3.0
            else:
                if mbp_idx >= 0:
                    nearest = _find_nearest(mbp_times, t, MAX_GAP)
                    if nearest is not None:
                        map_val = val_np[b, nearest, mbp_idx]
                if map_val is None and sbp_idx >= 0 and dbp_idx >= 0:
                    sbp_t = _find_nearest(sbp_times, t, MAX_GAP)
                    dbp_t = _find_nearest(dbp_times, t, MAX_GAP)
                    if sbp_t is not None and dbp_t is not None:
                        map_val = (val_np[b, sbp_t, sbp_idx] + 2 * val_np[b, dbp_t, dbp_idx]) / 3.0

            # Step 2: Vasopressor scores
            max_vaso_sofa = 0
            for drug_name, drug_idx in vaso_idxs.items():
                if drug_idx < 0:
                    continue
                drug_val = None
                if obs_np[b, t, drug_idx] > 0.5:
                    drug_val = val_np[b, t, drug_idx]
                else:
                    drug_times = np.where(obs_np[b, :, drug_idx] > 0.5)[0]
                    nearest = _find_nearest(drug_times, t, MAX_GAP)
                    if nearest is not None:
                        drug_val = val_np[b, nearest, drug_idx]

                if drug_val is not None and drug_val > 0:
                    for dose_thresh, sofa_score in CV_VASO_SCORES[drug_name]:
                        if drug_val > dose_thresh:
                            max_vaso_sofa = max(max_vaso_sofa, sofa_score)

            # Step 3: Combined SOFA CV
            sofa = max_vaso_sofa
            if sofa == 0 and map_val is not None and map_val < 70:
                sofa = 1
            score = sofa / 4.0

            for idx in all_cv_idxs:
                if obs_np[b, t, idx] > 0.5:
                    scores[idx][b, t] = score

    return scores


# =============================================================================
# Single-variable SOFA scoring
# =============================================================================

def _score_sofa_single(
    val_np: np.ndarray, obs_np: np.ndarray,
    var_idx: int, thresholds: List[float], direction: str, sofa_offset: int,
    B: int, T: int,
) -> np.ndarray:
    """Score single SOFA variable. Returns [B, T].

    score = (sofa_offset + n_additional_crossings - 1) / 4  if any crossed.
    """
    scores = np.zeros((B, T))
    obs_mask = obs_np[:, :, var_idx] > 0.5
    vals = val_np[:, :, var_idx]

    n_crossed = np.zeros((B, T))
    for thresh in thresholds:
        if direction == "above":
            n_crossed += (vals > thresh) & obs_mask
        else:
            n_crossed += (vals < thresh) & obs_mask

    any_crossed = n_crossed > 0
    scores[any_crossed] = (sofa_offset + n_crossed[any_crossed] - 1) / 4.0
    return scores




# =============================================================================
# Deterioration
# =============================================================================

def _compute_deterioration(
    severity: np.ndarray, obs_np: np.ndarray,
    window: int, B: int, T: int, V: int,
) -> np.ndarray:
    """Deterioration: increase in severity between consecutive observations.

    NOTE: this component is experimental. It has NO effect on any reported result.
    It adds mass only at cells whose static severity is already > 0,
    so the binary AUROC/AUPRC is bit-identical with or without it.
    It only shifts the unreported supplementary metrics.
    """
    det = np.zeros((B, T, V))

    for v in range(V):
        for b in range(B):
            obs_times = np.where(obs_np[b, :, v] > 0.5)[0]
            if len(obs_times) < 2:
                continue
            for i in range(1, len(obs_times)):
                t_cur = obs_times[i]
                t_prev = obs_times[i - 1]
                if t_cur - t_prev > window:
                    continue
                delta = severity[b, t_cur, v] - severity[b, t_prev, v]
                if delta > 0:
                    det[b, t_cur, v] = delta

    # Normalize per variable to [0, 1]
    for v in range(V):
        d_max = det[:, :, v].max()
        if d_max > 0:
            det[:, :, v] /= d_max

    return det


# =============================================================================
# Main clinical prior
# =============================================================================

def compute_clinical_prior(
    val_grid: torch.Tensor,
    obs_grid: torch.Tensor,
    var_names: List[str],
    mode: str = "sofa",
    temporal_weight: bool = False,
    include_deterioration: bool = True,
    deterioration_window: int = 6,
    deterioration_weight: float = 0.5,
) -> torch.Tensor:
    """Compute clinical severity prior.

    Args:
        val_grid: [B, T, V] raw clinical values (original units)
        obs_grid: [B, T, V] binary observation mask
        var_names: list of V variable names
        mode: "sofa"
        temporal_weight: apply exponential recency weighting
        include_deterioration: include deterioration component
        deterioration_window: max bins for deterioration lookback
        deterioration_weight: weight for deterioration

    Returns:
        prior: [B, T, V] in [0, 1]. Zero at unobserved cells.
    """
    if mode != "sofa":
        raise ValueError(f"Unknown mode: {mode!r}. Use 'sofa'.")

    B, T, V = val_grid.shape
    device = val_grid.device
    val_np = val_grid.cpu().numpy()
    obs_np = obs_grid.cpu().numpy()
    severity = np.zeros((B, T, V))

    def _idx(name):
        try:
            return var_names.index(name)
        except ValueError:
            return -1

    # --- Multi-variable SOFA organs ---

    # Respiratory: PF ratio
    po2_idx, fio2_idx = _idx("PO2"), _idx("FiO2")
    resp_scores = _score_respiratory(val_np, obs_np, po2_idx, fio2_idx, B, T)
    for idx, s in resp_scores.items():
        severity[:, :, idx] = s

    # CNS: GCS total
    eye_idx, verb_idx, mot_idx = _idx("GCS_eye"), _idx("GCS_verbal"), _idx("GCS_motor")
    cns_scores = _score_cns(val_np, obs_np, eye_idx, verb_idx, mot_idx, B, T)
    for idx, s in cns_scores.items():
        severity[:, :, idx] = s

    # Cardiovascular: MAP + vasopressors
    mbp_idx, sbp_idx, dbp_idx = _idx("MBP"), _idx("SBP"), _idx("DBP")
    vaso_idxs = {name: _idx(name) for name in CV_VASOPRESSOR_VARS}
    cv_scores = _score_cardiovascular(
        val_np, obs_np, var_names,
        mbp_idx, sbp_idx, dbp_idx, vaso_idxs, B, T,
    )
    for idx, s in cv_scores.items():
        severity[:, :, idx] = s

    # Track which indices are already scored by multi-var organs
    scored_idxs = set()
    scored_idxs.update(resp_scores.keys())
    scored_idxs.update(cns_scores.keys())
    scored_idxs.update(cv_scores.keys())

    # --- Single-variable SOFA ---
    for var_name, (thresholds, direction, offset) in SOFA_SINGLE_VAR.items():
        idx = _idx(var_name)
        if idx < 0 or idx in scored_idxs:
            continue
        severity[:, :, idx] = _score_sofa_single(
            val_np, obs_np, idx, thresholds, direction, offset, B, T,
        )
        scored_idxs.add(idx)


    # --- Deterioration ---
    if include_deterioration:
        det = _compute_deterioration(severity, obs_np, deterioration_window, B, T, V)
        severity = severity + deterioration_weight * det

    # --- Temporal recency ---
    if temporal_weight:
        t_weight = np.exp(np.linspace(-1.0, 0.0, T))
        severity = severity * t_weight[np.newaxis, :, np.newaxis]

    # Mask and normalize per sample
    severity = severity * obs_np
    s_max = severity.reshape(B, -1).max(axis=1).clip(min=1e-8)
    severity = severity / s_max[:, np.newaxis, np.newaxis]

    return torch.from_numpy(severity).float().to(device)


# =============================================================================
# Plausibility metrics
# =============================================================================

def compute_plausibility_metrics(
    attr: torch.Tensor,
    prior: torch.Tensor,
    obs_grid: torch.Tensor,
    topk_fractions: Tuple[float, ...] = (0.05, 0.1, 0.2),
) -> Dict[str, float]:
    """Compute plausibility agreement between attribution and clinical prior.

    Returns dict with metrics averaged over samples:
      - auroc_binary, auprc_binary: GT = prior > 0 vs == 0
      - spearman_rho: rank correlation at observed cells
      - topk_jaccard@f, mass_accuracy@f: top-k overlap
      - ndcg: normalized discounted cumulative gain
    """
    B = attr.shape[0]
    metrics_accum: Dict[str, list] = {}

    for b in range(B):
        mask = obs_grid[b].bool()
        n_obs = mask.sum().item()
        if n_obs < 10:
            continue

        a = attr[b][mask].cpu().numpy().astype(np.float64)
        p = prior[b][mask].cpu().numpy().astype(np.float64)

        if a.std() < 1e-10 or p.std() < 1e-10:
            continue

        # --- AUC metrics ---
        gt_binary = (p > 0).astype(np.int32)
        if 0 < gt_binary.sum() < len(gt_binary):
            metrics_accum.setdefault("auroc_binary", []).append(
                roc_auc_score(gt_binary, a))
            metrics_accum.setdefault("auprc_binary", []).append(
                average_precision_score(gt_binary, a))

        # --- Rank correlation ---
        rho, _ = spearmanr(a, p)
        if not np.isnan(rho):
            metrics_accum.setdefault("spearman_rho", []).append(rho)

        # --- Top-k overlap ---
        for frac in topk_fractions:
            k = max(1, int(frac * n_obs))
            topk_attr = set(np.argsort(a)[-k:])
            topk_prior = set(np.argsort(p)[-k:])
            intersection = len(topk_attr & topk_prior)
            union = len(topk_attr | topk_prior)
            metrics_accum.setdefault(f"topk_jaccard@{frac}", []).append(
                intersection / union if union > 0 else 0.0)
            metrics_accum.setdefault(f"mass_accuracy@{frac}", []).append(
                intersection / k if k > 0 else 0.0)

        # --- NDCG ---
        ranked = np.argsort(a)[::-1]
        p_ranked = p[ranked]
        discount = np.log2(np.arange(2, len(p_ranked) + 2))
        dcg = np.sum(p_ranked / discount)
        idcg = np.sum(np.sort(p)[::-1] / discount)
        metrics_accum.setdefault("ndcg", []).append(dcg / idcg if idcg > 0 else 0.0)

    return {k: float(np.mean(v)) for k, v in metrics_accum.items()}
