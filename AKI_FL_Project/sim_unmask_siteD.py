"""
mimic_ftl_simulation_phase3_aligned.py
=========================================
Phase III data simulation — APPROACH 2 (anchor-based, one row per patient),
GPC-ALIGNED FEATURE SET. Aligned with Liu et al. 2018 (AMIA, PMC5977670)
and with GPC RF feature importance lists (KUMC, MCW, UIOWA, UPITT, UTSW,
UofU; Zijian, KU shared drive).

PHASE III CHANGES vs phase2 script
-----------------------------------
1. Reads the GPC-aligned 24h-lookback cohort
   (aki_anchor_based_24h_lookback_aligned_features.csv, 163,038 patients,
   159 features, AKI prevalence 12.5%) — produced by
   AKI_Anchor_Based_Approach2_aligned_features.ipynb.
2. FEATURE_GROUPS split into 8 GPC-aligned groups (was 5): renal,
   metabolic_panel, hepatic, hematology, inflammatory, vitals_gpc,
   vitals_mimic, clinical. vitals_gpc (SBP/DBP/BMI) maps directly to
   GPC's VITAL_TIME class (SYSTOLIC/DIASTOLIC/ORIGINAL_BMI); vitals_mimic
   (heart_rate/resp_rate/spo2/temperature/gcs_total) has no GPC RF-list
   equivalent but is retained as MIMIC-IV-only enrichment.
3. ANCHOR_PREVALENCE updated to 0.125 (GPC-aligned cohort AKI rate,
   up from 0.09 — SCr>1.3 admission exclusion removed, 24h lookback).
4. FIXED_PREVALENCES re-scaled around the new 12.5% global rate.
5. SITE_CONFIGS groups updated to draw from the 8 new feature groups so
   site-level feature masking remains clinically coherent (e.g. site_E
   keeps renal+inflammatory+clinical, no hepatic/hematology panel).

INHERITED FROM PHASE II / APPROACH 2
-------------------------------------
1. Feature groups use prefix matching (e.g. "creatinine" matches
   creatinine_most_recent, creatinine_min, creatinine_max etc.)
2. sample_with_prevalence respects train/test split — only samples
   from train split to prevent test leakage into federated site CSVs
3. TARGET_N_PER_SITE auto-capped at available training patients
4. baseline_method excluded from feature groups (string column)

WHAT IS NEW vs Phase I (mimic_ftl_simulation3.py)
--------------------------------------------------
1. site_C local dominance
   - site_C is capped at ~33k samples (same as all other sites) so the
     comparison is fair in terms of data volume.
   - site_C retains its full 159-feature set (8 feature groups) and true
     GPC-aligned MIMIC-IV prevalence (12.5%). With richer features than
     any other site, a locally trained model at site_C is expected to
     outperform the federated global model for site_C's own patients —
     WITHOUT artificially reducing its N.
   - This motivates personalisation: even a strong contributor should keep
     a local head rather than fully merging into the global model.

2. FL-gain index
   - A per-site composite score computed BEFORE any model training.
   - Combines positive case scarcity, class imbalance (minority class
     fraction), and feature sparsity (fraction of full feature set
     available).
   - Exported as fl_gain_index.csv alongside the site CSVs.
   - Hypothesis: FL-gain index correlates with observed AUROC improvement
     from federation vs local-only training — validated in Phase III.

3. Confirmed fixed prevalences (GPC-aligned cohort, 12.5% global rate)
   site_A  43%   ICU / Tertiary
   site_B  17%   General Ward
   site_C  12.5% Academic anchor (GPC-aligned cohort rate)
   site_D  10%   Community clinic
   site_E   6%   Resource-limited / Rural  ← primary benefitter
   (site_F removed — semi-supervised site dropped for Approach 2)

4. SiteInputAdapter specification
   - Each site exports a metadata JSON recording its local feature list
     and embedding_dim target. This is consumed by fedadapt_model.py and
     ALL baseline FL models (FedAvg, FedProx, SCAFFOLD) so that input
     heterogeneity is handled uniformly and comparisons are fair.
   - The adapter is infrastructure, not a novelty; FedAdapt's novelty is
     the GRL group alignment and the personal head.

PHASE I / PHASE II COMPATIBILITY
----------------------------------
This script reads from the GPC-aligned anchor-based CSV produced by
AKI_Anchor_Based_Approach2_aligned_features.ipynb. It is independent of
Phase I and Phase II outputs — run all three side by side if needed;
they share only the upstream MIMIC-IV data source.

USAGE
-----
python mimic_ftl_simulation_phase3_aligned.py \\
    --input  /path/to/aki_anchor_based_24h_lookback_aligned_features.csv \\
    --label  AKI_label \\
    --alpha  0.5 \\
    --gamma  0.75 \\
    --output ./phase3_sites_aligned/

Optional flags:
    --alpha   float   Dirichlet concentration (default 0.5)
    --gamma   float   Covariate shift severity (default 0.75)
    --seed    int     Random seed (default 42)
    --sweep           Run alpha in [0.1, 0.3, 0.5, 1.0, 10.0] and
                      gamma in [0.0, 0.5, 0.75, 1.0]
    --embedding_dim   int  Shared embedding dim for SiteInputAdapter (default 64)
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

TARGET_N_PER_SITE = 33_000      # target — auto-capped at available train patients
ANCHOR_PREVALENCE = 0.125       # GPC-aligned cohort AKI rate (24h lookback,
                                 # SCr>1.3 admission exclusion removed)

# Fixed AKI prevalences — clinically grounded, re-scaled around the new
# 12.5% global rate (was 9.0% before GPC alignment). Relative ordering
# and spread preserved; site_C fixed to actual cohort rate.
FIXED_PREVALENCES: Dict[str, Optional[float]] = {
    "site_A": 0.43,    # ICU — literature range 20-50%, using upper-mid
    "site_B": 0.17,    # General ward — literature range 12-17%, upper end
    "site_C": 0.125,   # Academic anchor — actual GPC-aligned cohort rate
    "site_D": 0.10,    # Community clinic — literature range 5-10%, upper end
    "site_E": 0.06,    # Rural/resource-limited — literature range 3-7%, mid
}

# ─── FEATURE GROUPS ───────────────────────────────────────────────────────────

# GPC-ALIGNED: feature names are prefixes matching all stat-suffix variants
# e.g. "creatinine" matches creatinine_most_recent, creatinine_min,
#       creatinine_max, creatinine_mean, creatinine_hours_since
# Exact names (baseline_scr, age_at_admission etc.) matched as-is.
# baseline_method excluded — string column, not numeric feature.
#
# Groups mirror GPC RF feature list `feature_class` / `clinical_meaning`
# categories (see mimic_gpc_feature_alignment.csv) so that site-level
# feature masking produces feature-sparsity conditions directly
# comparable to GPC's 6-site feature coverage.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "renal": [
        "baseline_scr", "hours_to_anchor",
        "creatinine", "bun",
    ],
    "metabolic_panel": [
        # GPC-aligned: Sodium, Potassium, Chloride, Bicarbonate, Calcium,
        # Phosphate, Magnesium, Glucose — all in GPC shared98 (mean rank
        # 4.7 - 78.5 across 6 GPC sites)
        "sodium", "potassium", "chloride", "bicarbonate",
        "calcium", "phosphate", "magnesium", "glucose",
    ],
    "hepatic": [
        # GPC-aligned: Albumin, Protein Mass, Bilirubin total/direct
        "albumin", "total_protein", "bilirubin", "bilirubin_dir",
    ],
    "hematology": [
        # wbc, platelets, hemoglobin retained (no direct GPC match found
        # for wbc/platelets, but standard CBC labs — kept as MIMIC-IV
        # clinical signal). rdw/basophils_pct/lymphocyte_pct are
        # GPC-aligned (mean rank 5.2 - 70.5).
        "wbc", "hemoglobin", "platelets",
        "rdw", "basophils_pct", "lymphocyte_pct",
    ],
    "inflammatory": [
        "lactate",
    ],
    "vitals_gpc": [
        # GPC-aligned: maps directly to GPC VITAL_TIME class
        # (SYSTOLIC, DIASTOLIC, ORIGINAL_BMI)
        "sbp", "dbp", "bmi",
    ],
    "vitals_mimic": [
        # MIMIC-IV only — no GPC RF-list equivalent found, retained as
        # local enrichment (does not claim GPC comparability)
        "heart_rate", "resp_rate", "spo2", "temperature", "gcs_total",
    ],
    "clinical": [
        "admission_type", "gender", "age_at_admission",
        "has_diabetes", "has_hypertension", "has_chf",
        "has_sepsis", "has_liver_disease", "has_cancer",
        "nephrotoxic_flag", "nephrotoxic_count", "n_distinct_meds",
    ],
}

# ─── SITE CONFIGURATIONS ──────────────────────────────────────────────────────

SITE_CONFIGS: Dict[str, dict] = {
    "site_A": {
        "description": "ICU / Tertiary hospital",
        # ICU: full renal + inflammatory workup, vitals (both GPC + MIMIC-only
        # monitoring), but no routine metabolic/hepatic panel beyond renal
        "groups":      ["renal", "inflammatory", "vitals_gpc", "vitals_mimic", "clinical"],
        "acuity_bias": +2.0,    # draws sicker patients
        "spread_scale": 0.70,   # focused / narrower feature distributions
        "unlabeled":   False,
    },
    "site_B": {
        "description": "General ward / Secondary hospital",
        # General ward: renal + metabolic panel + GPC-aligned vitals
        "groups":      ["renal", "metabolic_panel", "vitals_gpc", "clinical"],
        "acuity_bias": +0.5,
        "spread_scale": 1.10,
        "unlabeled":   False,
    },
    "site_C": {
        "description": "Academic Medical Centre — MIMIC-IV anchor, full feature set",
        # Anchor: all 8 feature groups, richest feature coverage
        "groups":      ["renal", "metabolic_panel", "hepatic", "hematology",
                         "inflammatory", "vitals_gpc", "vitals_mimic", "clinical"],
        "acuity_bias": 0.0,     # anchor: unbiased sampling
        "spread_scale": 1.00,
        "unlabeled":   False,
        "is_anchor":   True,
        # NOTE: site_C local dominance
        # site_C has the richest feature set vs all other sites.
        # With equal N (~33k) its locally trained model is expected to
        # outperform the global model for site_C patients — motivating
        # a personal head even for a strong contributor.
    },
    "site_D": {
        "description": "Community / Primary care clinic",
        # Community: metabolic panel + GPC-aligned vitals only — no renal
        # labs drawn routinely, no specialised hepatic/hematology workup
        "groups":      ["metabolic_panel", "vitals_gpc", "clinical"],
        "acuity_bias": -0.5,
        "spread_scale": 1.20,
        "unlabeled":   False,
    },
    "site_E": {
        "description": "Resource-limited / Rural hospital — PRIMARY FL BENEFITTER",
        # Rural: renal + inflammatory only — sparsest feature coverage
        "groups":      ["renal", "inflammatory", "clinical"],
        "acuity_bias": -0.8,
        "spread_scale": 1.40,   # wide / heterogeneous referral patterns
        "unlabeled":   False,
    },
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def resolve_feature_columns(df: pd.DataFrame, groups: List[str]) -> List[str]:
    """
    Return columns present in df that belong to the requested groups.

    APPROACH 2: feature names are prefixes. A column matches if:
      - it equals the prefix exactly (e.g. baseline_scr, age_at_admission), OR
      - it starts with prefix + "_" (e.g. creatinine_most_recent, bun_min)
    String columns (baseline_method) are automatically excluded.
    """
    wanted_prefixes = []
    for g in groups:
        wanted_prefixes.extend(FEATURE_GROUPS.get(g, []))

    matched = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue   # skip string/object columns
        for prefix in wanted_prefixes:
            if col == prefix or col.startswith(prefix + "_"):
                if col not in matched:
                    matched.append(col)
                break
    return matched


def sample_with_prevalence(
    df: pd.DataFrame,
    label_col: str,
    target_prev: float,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw n rows from df such that AKI prevalence ≈ target_prev.
    Samples AKI and non-AKI strata separately then concatenates.

    APPROACH 2: if a "split" column is present, sample only from the
    train split — test patients must not appear in any site CSV.
    One row per patient (no multi-row expansion), so replace=False
    unless the site target N exceeds available training patients.
    """
    # Restrict to train split if present
    pool = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()

    aki     = pool[pool[label_col] == 1]
    non_aki = pool[pool[label_col] == 0]

    # Cap n to available training patients
    max_available = len(aki) + len(non_aki)
    n = min(n, max_available)

    n_aki     = min(int(round(n * target_prev)), len(aki))
    n_non_aki = min(n - n_aki, len(non_aki))

    sampled_aki     = aki.sample(n=n_aki,     replace=len(aki)     < n_aki,     random_state=int(rng.integers(1e6)))
    sampled_non_aki = non_aki.sample(n=n_non_aki, replace=len(non_aki) < n_non_aki, random_state=int(rng.integers(1e6)))

    return pd.concat([sampled_aki, sampled_non_aki]).sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)


def apply_covariate_shift(
    df: pd.DataFrame,
    feature_cols: List[str],
    global_stats: pd.DataFrame,
    acuity_bias: float,
    spread_scale: float,
    gamma: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Apply two-mechanism covariate shift scaled by gamma.

    Mechanism A — acuity-driven mean shift (z-score units):
        delta_z ~ N(acuity_bias * gamma, 0.1)

    Mechanism B — spread perturbation:
        effective_scale = 1.0 + gamma * (spread_scale - 1.0)
    """
    df = df.copy()
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols or gamma == 0.0:
        return df

    # Draw per-site delta_z once (coherent shift across all features)
    delta_z        = rng.normal(acuity_bias * gamma, 0.1)
    eff_scale      = 1.0 + gamma * (spread_scale - 1.0)

    for col in numeric_cols:
        if col not in global_stats.index:
            continue
        mu  = global_stats.loc[col, "mean"]
        sig = global_stats.loc[col, "std"]
        if sig == 0 or np.isnan(sig):
            continue
        # Shift in raw units: delta_z standard deviations
        df[col] = df[col] + delta_z * sig
        # Scale spread around new mean
        new_mean   = mu + delta_z * sig
        df[col]    = new_mean + eff_scale * (df[col] - new_mean)

    return df


# ─── FL-GAIN INDEX ────────────────────────────────────────────────────────────

def compute_fl_gain_index(
    site_dfs: Dict[str, pd.DataFrame],
    label_col: str,
    total_features: int,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute a per-site FL-gain index BEFORE any model training.

    COMPONENTS (each in [0, 1])
    ───────────────────────────
    positive_case_scarcity = 1 - (site_n_positive / total_positive_all_sites)
                       Measures AKI case contribution scarcity.
                       site_A (ICU, many positives) scores low → contributor.
                       site_E (rural, few positives) scores high → benefitter.
                       Non-zero even when all sites have equal sample count N.

    class_imbalance  = 1 - minority_class_fraction
                       minority = min(prevalence, 1 - prevalence)
                       site_E: prev=0.05 → minority=0.05 → imbalance=0.95
                       site_A: prev=0.43 → minority=0.43 → imbalance=0.57

    feature_sparsity = 1 - (site_features / total_features)
                       site_E (18 feat) scores high; site_C (39 feat) scores 0.

    (label_efficiency removed — redundant with positive_case_scarcity
     when all sites have equal N=33k)

    FL-GAIN INDEX
    ─────────────
    Weighted mean of the four components. Default weights reflect the
    clinical insight that label scarcity and feature poverty drive FL
    benefit more than raw data volume when N is equalised:

        default weights = {
            data_rarity:      0.10,   # near-zero signal at equal N
            class_imbalance:  0.30,   # symmetric imbalance penalty
            feature_sparsity: 0.30,   # feature poverty
            label_efficiency: 0.30,   # positive label scarcity (AKI)
        }

    Override via the `weights` argument, e.g.:
        weights = {"data_rarity": 0.0, "class_imbalance": 0.25,
                   "feature_sparsity": 0.35, "label_efficiency": 0.40}

    INTERPRETATION
    ──────────────
        0.0 → rich data, balanced/high AKI rate, full features → low FL benefit
        1.0 → scarce data, rare AKI, few features              → high FL benefit

    HYPOTHESIS (validated in Phase III)
    ────────────────────────────────────
    FL-gain index correlates positively with observed AUROC improvement
    (federated global model vs local-only model) per site.
    """
    # Default weights — data_rarity down-weighted since N is equalised
    # 3-component index — data_rarity and label_efficiency replaced by
    # positive_case_scarcity which is non-zero even when all sites have equal N.
    DEFAULT_WEIGHTS = {
        "positive_case_scarcity": 0.30,  # AKI case contribution scarcity
        "class_imbalance":        0.30,  # symmetric imbalance penalty
        "feature_sparsity":       0.40,  # feature poverty (strongest driver)
    }
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    w_total = sum(w.values())
    w = {k: v / w_total for k, v in w.items()}

    # Total positive cases across all sites
    total_positive = sum(int(df[label_col].sum()) for df in site_dfs.values())

    records = []

    for site_id, df in site_dfs.items():
        n          = len(df)
        prev       = float(df[label_col].mean())
        n_positive = int(df[label_col].sum())

        # positive_case_scarcity: low AKI-case share → high scarcity → benefitter
        contributor_score      = n_positive / max(total_positive, 1)
        positive_case_scarcity = float(np.clip(1.0 - contributor_score, 0, 1))

        # class_imbalance: symmetric penalty for skewed prevalence
        minority        = min(prev, 1.0 - prev)
        class_imbalance = float(np.clip(1.0 - minority, 0, 1))

        # feature_sparsity: fraction of full feature set missing at this site
        n_features       = sum(
            1 for c in df.columns
            if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        )
        feature_sparsity = float(np.clip(1.0 - (n_features / total_features), 0, 1))

        components = {
            "positive_case_scarcity": positive_case_scarcity,
            "class_imbalance":        class_imbalance,
            "feature_sparsity":       feature_sparsity,
        }

        fl_gain = float(sum(w[k] * v for k, v in components.items()))

        records.append({
            "site_id":               site_id,
            "n_samples":             n,
            "aki_prevalence":        round(prev, 4),
            "n_positive":            n_positive,
            "n_features":            n_features,
            "contributor_score":     round(contributor_score, 4),
            **{k: round(v, 4) for k, v in components.items()},
            "fl_gain_index":         round(fl_gain, 4),
            "role":                  _infer_role(site_id, fl_gain),
            "weight_positive_case_scarcity": round(w["positive_case_scarcity"], 3),
            "weight_class_imbalance":        round(w["class_imbalance"],        3),
            "weight_feature_sparsity":       round(w["feature_sparsity"],       3),
        })

    return pd.DataFrame(records).sort_values("fl_gain_index", ascending=False).reset_index(drop=True)


def _infer_role(site_id: str, fl_gain: float) -> str:
    """
    Role thresholds calibrated for 3-component index with positive_case_scarcity.
    site_A (ICU, many AKI cases, rich features) expected ~0.45 → contributor
    site_C (anchor, full features)               expected ~0.55 → conditional
    site_E (rural, few cases, sparse)            expected ~0.75 → primary_benefitter
    """
    if fl_gain >= 0.65:
        return "primary_benefitter"
    if fl_gain >= 0.52:
        return "conditional_benefitter"
    return "contributor"


# ─── SITE INPUT ADAPTER METADATA ─────────────────────────────────────────────

def export_adapter_metadata(
    site_dfs: Dict[str, pd.DataFrame],
    label_col: str,
    embedding_dim: int,
    output_dir: Path,
) -> None:
    """
    Export a JSON metadata file per site consumed by fedadapt_model.py
    and ALL baseline FL models (FedAvg, FedProx, SCAFFOLD).

    The SiteInputAdapter is SHARED INFRASTRUCTURE across all methods —
    not a FedAdapt novelty. It exists because feature heterogeneity
    (18–39 features) means every FL algorithm needs a site-specific
    projection to a common embedding dimension before federation.

    Schema:
    {
        "site_id":       "site_E",
        "input_dim":     18,
        "embedding_dim": 64,
        "feature_names": ["baseline_scr", ..., "has_cancer"],
        "aki_prevalence": 0.05,
        "unlabeled":     false
    }
    """
    for site_id, df in site_dfs.items():
        feature_cols = [
            c for c in df.columns
            if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        prev = float(df[label_col].mean())

        meta = {
            "site_id":        str(site_id),
            "input_dim":      int(len(feature_cols)),
            "embedding_dim":  int(embedding_dim),
            "feature_names":  [str(c) for c in feature_cols],
            "aki_prevalence": float(prev),
            "unlabeled":      False,
            "description":    str(SITE_CONFIGS[site_id]["description"]),
            "feature_groups": [str(g) for g in SITE_CONFIGS[site_id]["groups"]],
        }

        out_path = output_dir / f"{site_id}_adapter_meta.json"
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [adapter] {out_path.name}  input_dim={len(feature_cols)}  → embedding_dim={embedding_dim}")


# ─── SIMULATION PLOTS ─────────────────────────────────────────────────────────

def plot_simulation_summary(
    site_dfs: Dict[str, pd.DataFrame],
    fl_gain_df: pd.DataFrame,
    label_col: str,
    alpha: float,
    gamma: float,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [plot] matplotlib not available — skipping plots")
        return

    sites   = list(site_dfs.keys())
    colors  = {
        "contributor":           "#4C72B0",
        "conditional_benefitter":"#DD8452",
        "primary_benefitter":    "#C44E52",
        }
    role_map = dict(zip(fl_gain_df["site_id"], fl_gain_df["role"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Phase III simulation summary (GPC-aligned)  |  α={alpha}  γ={gamma}\n"
        f"site_C local dominance: equal N (~33k), richest features (159) → "
        f"local model expected to outperform global",
        fontsize=11,
    )

    # Panel 1 — AKI prevalence
    ax = axes[0]
    prevs = []
    bar_colors = []
    for s in sites:
        df = site_dfs[s]
        prevs.append(float(df[label_col].mean()))
        bar_colors.append(colors.get(role_map.get(s, "contributor"), "#4C72B0"))
    bars = ax.bar(sites, prevs, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(ANCHOR_PREVALENCE, color="navy", linestyle="--", linewidth=1, label=f"Anchor ({ANCHOR_PREVALENCE:.1%})")
    for bar, p in zip(bars, prevs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{p:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("AKI prevalence")
    ax.set_title("AKI prevalence per site")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)

    # Panel 2 — FL-gain index
    ax = axes[1]
    gain_dict = dict(zip(fl_gain_df["site_id"], fl_gain_df["fl_gain_index"]))
    gains = [gain_dict.get(s, 0) for s in sites]
    gc    = [colors.get(role_map.get(s, "contributor"), "#4C72B0") for s in sites]
    bars2 = ax.bar(sites, gains, color=gc, edgecolor="white", linewidth=0.5)
    for bar, g in zip(bars2, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{g:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Benefitter threshold (0.5)")
    ax.set_ylabel("FL-gain index")
    ax.set_title("FL-gain index per site\n(higher = benefits more from federation)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)

    # Panel 3 — Feature count
    ax = axes[2]
    fcounts = []
    for s in sites:
        df = site_dfs[s]
        fcounts.append(sum(
            1 for c in df.columns
            if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        ))
    fc_colors = [colors.get(role_map.get(s, "contributor"), "#4C72B0") for s in sites]
    bars3 = ax.bar(sites, fcounts, color=fc_colors, edgecolor="white", linewidth=0.5)
    for bar, f in zip(bars3, fcounts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(f), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of features")
    ax.set_title("Feature count per site\n(site_C has most features → local dominance)")

    # Legend for roles
    patches = [mpatches.Patch(color=c, label=r.replace("_", " ").title())
               for r, c in colors.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fname = output_dir / f"phase3_summary_alpha{alpha}_gamma{gamma}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {fname.name}")


def plot_fl_gain_decomposition(
    fl_gain_df: pd.DataFrame,
    alpha: float,
    gamma: float,
    output_dir: Path,
) -> None:
    """
    Stacked bar showing all four FL-gain components per site.
    Bars are weighted contributions (weight × raw component value)
    so the total bar height equals the FL-gain index directly.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df    = fl_gain_df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"FL-gain index decomposition  |  α={alpha}  γ={gamma}\n"
        f"Left: weighted contributions (bar height = FL-gain index)  |  "
        f"Right: raw component scores",
        fontsize=11,
    )

    sites = df["site_id"].tolist()
    x     = np.arange(len(sites))
    bw    = 0.55

    COMPONENT_COLORS = {
        "positive_case_scarcity": "#4C72B0",
        "class_imbalance":        "#DD8452",
        "feature_sparsity":       "#C44E52",
    }
    COMPONENT_LABELS = {
        "positive_case_scarcity": "Positive-case scarcity (w=0.30)",
        "class_imbalance":        "Class imbalance (w=0.30)",
        "feature_sparsity":       "Feature sparsity (w=0.40)",
    }
    W = {"positive_case_scarcity": 0.30, "class_imbalance": 0.30,
         "feature_sparsity": 0.40}

    # ── Panel 1: weighted contributions ──────────────────────────────────────
    ax   = axes[0]
    bottom = np.zeros(len(df))
    for comp, color in COMPONENT_COLORS.items():
        weighted = df[comp].values * W[comp]
        ax.bar(x, weighted, bw, bottom=bottom,
               label=COMPONENT_LABELS[comp], color=color, edgecolor="white", linewidth=0.4)
        bottom += weighted

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["fl_gain_index"] + 0.008,
                f'{row["fl_gain_index"]:.3f}', ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("Weighted FL-gain contribution")
    ax.set_title("Weighted contributions\n(bar height = FL-gain index)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.text(len(sites) - 0.45, 0.51, "benefitter threshold", fontsize=7, color="gray")
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel 2: raw component scores (unweighted) ────────────────────────────
    ax2    = axes[1]
    bottom2 = np.zeros(len(df))
    for comp, color in COMPONENT_COLORS.items():
        raw = df[comp].values
        ax2.bar(x, raw, bw, bottom=bottom2,
                label=comp.replace("_", " ").title(), color=color,
                edgecolor="white", linewidth=0.4, alpha=0.85)
        bottom2 += raw

    ax2.set_xticks(x)
    ax2.set_xticklabels(sites)
    ax2.set_ylabel("Raw component score (unweighted)")
    ax2.set_title("Raw component scores\n(for comparison — not the actual index)")
    ax2.set_ylim(0, 4.2)
    ax2.axhline(2.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fname = output_dir / f"fl_gain_decomposition_alpha{alpha}_gamma{gamma}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {fname.name}")


# ─── MAIN SIMULATION ──────────────────────────────────────────────────────────

def run_simulation(
    df_source: pd.DataFrame,
    label_col: str,
    alpha: float,
    gamma: float,
    embedding_dim: int,
    output_dir: Path,
    rng: np.random.Generator,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Phase III simulation (GPC-aligned)  |  α={alpha}  γ={gamma}")
    print(f"{'='*60}")

    # Global stats for covariate shift (computed once from full source)
    all_numeric = df_source.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in all_numeric:
        all_numeric.remove(label_col)
    global_stats = df_source[all_numeric].agg(["mean", "std"]).T

    total_features = len(all_numeric)

    site_dfs: Dict[str, pd.DataFrame] = {}

    for site_id, cfg in SITE_CONFIGS.items():
        print(f"\n  → {site_id}: {cfg['description']}")

        # 1. Resolve feature columns
        feat_cols = resolve_feature_columns(df_source, cfg["groups"])
        if not feat_cols:
            print(f"     WARNING: no matching feature columns found — skipping")
            continue

        # 2. Sample with alpha-interpolated prevalence (Option A).
        #
        #   blend = alpha / (alpha + 1)
        #   target_prev = (1 - blend) x archetype + blend x global_rate
        #
        #   alpha -> 0 : target_prev -> archetype  (clinical archetype)
        #   alpha -> inf : target_prev -> global_rate (~12.6%, IID baseline)
        #   midpoint at alpha=1.0 (50/50 blend)
        #
        #   site_C (anchor) always uses its fixed rate.
        archetype_prev = FIXED_PREVALENCES[site_id]
        global_prev = float(df_source[label_col].mean())

        if site_id == "site_C":
            target_prev = archetype_prev
            blend = 0.0
        else:
            blend       = float(alpha) / (float(alpha) + 1.0)
            target_prev = (1.0 - blend) * archetype_prev + blend * global_prev

        print(f"     target_prev={target_prev:.3f}  "
              f"(archetype={archetype_prev:.3f}, global={global_prev:.3f}, blend={blend:.2f})")

        site_df = sample_with_prevalence(
            df_source, label_col, target_prev,
            n=TARGET_N_PER_SITE, rng=rng,
        )

        # 3. Mask to site feature set + label
        keep_cols = feat_cols + [label_col]
        keep_cols = [c for c in keep_cols if c in site_df.columns]
        site_df   = site_df[keep_cols].copy()

        # 4. Apply covariate shift
        site_df = apply_covariate_shift(
            site_df, feat_cols, global_stats,
            acuity_bias=cfg["acuity_bias"],
            spread_scale=cfg["spread_scale"],
            gamma=gamma, rng=rng,
        )

        obs_prev = float(site_df[label_col].mean())
        print(f"     N={len(site_df):,}  features={len(feat_cols)}  AKI_prev={obs_prev:.3f}")

        site_dfs[site_id] = site_df

        # 6. Save CSV
        csv_path = output_dir / f"{site_id}_alpha{alpha}_gamma{gamma}.csv"
        site_df.to_csv(csv_path, index=False)

    # 7. FL-gain index
    print("\n  Computing FL-gain index...")
    fl_gain_df = compute_fl_gain_index(site_dfs, label_col, total_features)
    gain_path  = output_dir / f"fl_gain_index_alpha{alpha}_gamma{gamma}.csv"
    fl_gain_df.to_csv(gain_path, index=False)
    print(fl_gain_df[["site_id", "n_samples", "aki_prevalence", "n_positive",
                       "n_features", "positive_case_scarcity", "class_imbalance",
                       "feature_sparsity", "fl_gain_index", "role"]].to_string(index=False))

    # 8. Adapter metadata (shared by all FL methods)
    print("\n  Exporting SiteInputAdapter metadata...")
    export_adapter_metadata(site_dfs, label_col, embedding_dim, output_dir)

    # 9. Plots
    print("\n  Generating plots...")
    plot_simulation_summary(site_dfs, fl_gain_df, label_col, alpha, gamma, output_dir)
    plot_fl_gain_decomposition(fl_gain_df, alpha, gamma, output_dir)

    print(f"\n  ✅ Done — outputs in {output_dir}")

    # Return per-site stats so the sweep loop can build the cross-alpha chart
    site_stats = {}
    for sid, sdf in site_dfs.items():
        n_aki = int(sdf[label_col].sum())
        n_no  = int((sdf[label_col] == 0).sum())
        site_stats[sid] = {
            "n_aki":       n_aki,
            "n_no_aki":    n_no,
            "description": SITE_CONFIGS[sid]["description"],
        }
    return site_stats


def plot_class_distribution_by_alpha(
    sweep: dict,          # {alpha: {site_id: {"n_aki": int, "n_no_aki": int, "description": str}}}
    output_dir: Path,
    global_aki_rate: float = 0.125,
) -> None:
    """
    Horizontal stacked-bar chart: one panel per alpha, one bar per site.

    HIGH alpha → all sites converge toward global_aki_rate (~12.6 %)
    LOW  alpha → sites diverge toward clinical archetypes
                 (ICU ~45-50 %, rural ~5-9 %, etc.)

    A dashed red reference line marks global_aki_rate on every panel.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    CLASS_COLORS = {"No AKI": "#D1D5DB", "AKI": "#1D4ED8"}

    def _regime(a):
        if a <= 0.1:  return "extreme non-IID"
        if a <= 0.3:  return "non-IID"
        if a <= 1.0:  return "moderate non-IID"
        if a <= 3.0:  return "mild non-IID"
        return f"near-IID  (→ {global_aki_rate*100:.1f}% global)"

    alphas   = sorted(sweep.keys())
    n_panels = len(alphas)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(3.4 * n_panels, 5.8),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        site_data = sweep[alpha]
        site_ids  = sorted(site_data.keys())

        for i, sid in enumerate(site_ids):
            info     = site_data[sid]
            n_aki    = info.get("n_aki", 0)
            n_no_aki = info.get("n_no_aki", 0)
            total    = n_aki + n_no_aki
            if total == 0:
                continue
            frac_no = n_no_aki / total
            frac_ak = n_aki    / total

            ax.barh(i, frac_no, color=CLASS_COLORS["No AKI"],
                    edgecolor="white", linewidth=0.4)
            ax.barh(i, frac_ak, left=frac_no,
                    color=CLASS_COLORS["AKI"],
                    edgecolor="white", linewidth=0.4)

            if frac_ak > 0.06:
                ax.text(frac_no + frac_ak / 2, i,
                        f"{frac_ak*100:.0f}%",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")

        # reference line at global AKI rate
        ax.axvline(x=global_aki_rate, color="#EF4444",
                   linestyle="--", linewidth=1.3, alpha=0.9)

        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.set_xlabel("Proportion of patients", fontsize=9)
        ax.set_title(f"α = {alpha}\n{_regime(alpha)}",
                     fontsize=9, fontweight="bold", pad=6)
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        short = [s.replace("site_", "") for s in site_ids]
        descs = [site_data[s].get("description", "").split("/")[0].strip()[:12]
                 for s in site_ids]
        ylabels = [f"{sh}\n({d})" for sh, d in zip(short, descs)]
        ax.set_yticks(range(len(site_ids)))
        if ax is axes[0]:
            ax.set_yticklabels(ylabels, fontsize=8)
            ax.set_ylabel("Site", fontsize=10)
        else:
            ax.set_yticklabels([])

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS["No AKI"], label="No AKI"),
        mpatches.Patch(color=CLASS_COLORS["AKI"],    label="AKI"),
        plt.Line2D([0], [0], color="#EF4444", linestyle="--", linewidth=1.3,
                   label=f"Global AKI rate ({global_aki_rate*100:.1f}% — Approach 2)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "AKI Label Distribution per Site Across α Values\n"
        "High α → converge to global rate  |  Low α → diverge to clinical archetypes",
        fontsize=11, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    out = output_dir / "class_distribution_by_alpha.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [plot] class_distribution_by_alpha.png  →  {out}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",         required=True,  help="Path to MIMIC-IV AKI CSV")
    p.add_argument("--label",         default="AKI_label", help="Label column name")
    p.add_argument("--alpha",         type=float, default=0.5, help="Dirichlet α")
    p.add_argument("--gamma",         type=float, default=0.75, help="Covariate shift γ")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--embedding_dim", type=int,   default=64, help="Shared embedding dim for SiteInputAdapter")
    p.add_argument("--output",        default="./phase3_sites_aligned/", help="Output directory")
    p.add_argument("--sweep",         action="store_true", help="Sweep α and γ values")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.output)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  Source: {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Overall AKI rate: {df[args.label].mean():.3f}")

    if args.sweep:
        alphas = [0.1, 0.3, 0.5, 1.0, 10.0]
        gammas = [0.0, 0.5, 0.75, 1.0]
        # Collect per-alpha stats (using gamma=0.75 as the representative slice)
        # so we can draw the cross-alpha label-distribution chart at the end.
        sweep_stats: dict = {}
        for a in alphas:
            for g in gammas:
                sub_dir   = out_dir / f"alpha{a}_gamma{g}"
                site_stats = run_simulation(df, args.label, a, g, args.embedding_dim, sub_dir, rng)
                if g == 0.75:          # representative gamma for the chart
                    sweep_stats[a] = site_stats
        # One chart across all alpha values → saved at the top-level output dir
        out_dir.mkdir(parents=True, exist_ok=True)
        global_rate = float(df[args.label].mean())
        plot_class_distribution_by_alpha(sweep_stats, out_dir, global_aki_rate=global_rate)
    else:
        run_simulation(df, args.label, args.alpha, args.gamma,
                       args.embedding_dim, out_dir, rng)


if __name__ == "__main__":
    main()