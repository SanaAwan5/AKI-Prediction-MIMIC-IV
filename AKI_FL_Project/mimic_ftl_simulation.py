"""
mimic_ftl_simulation3.py
========================
Simulate three orthogonal axes of non-IID heterogeneity across 6 federated
sites from a MIMIC-IV AKI CSV.

  1. Label shift        — Dirichlet alpha controls AKI prevalence divergence
                          across sites. Lower alpha = more extreme imbalance.

  2. Feature masking    — Clinical archetypes determine which feature groups
                          each site observes (e.g. ICU has renal+inflammatory,
                          community has only metabolic+clinical).

  3. Covariate shift    — gamma controls feature distribution shift via two
                          CLEANLY SEPARATED mechanisms:

        a) Subpopulation sampling (drives MEAN shift using real data)
           Sites preferentially draw patients matching their clinical archetype
           via a softmax over a composite acuity score. ICU draws sicker
           patients (higher creatinine, lactate etc.) naturally. The mean
           shift comes from real patient heterogeneity, not synthetic numbers.

        b) Gaussian SPREAD perturbation only (no mean shift)
           After sampling, the spread of each numeric feature is scaled by a
           site-specific factor controlled by gamma and spread_scale.
           This simulates measurement/calibration differences between sites
           (e.g. less precise analysers at rural hospitals = wider spread).
           The mean is NEVER moved synthetically -- only the variance.


Columns expected in CSV
-----------------------
Renal:        baseline_scr, scr_first_24h_*, bun_first_24h_*
Inflammatory: lactate_first_24h_*, wbc_first_24h_*, platelets_first_24h_*
Metabolic:    sodium_first_24h_*, potassium_first_24h_*,
              bicarbonate_first_24h_*, hemoglobin_first_24h_*
Clinical:     admission_type, insurance, gender, anchor_age,
              age_at_admission, has_diabetes, has_hypertension,
              has_chf, has_sepsis, has_liver_disease, has_cancer,
              baseline_method
Hemodynamic:  sbp, dbp, map, heart_rate, spo2 (reserved; add if available)
Label:        AKI_label

Usage
-----
    # Single run, no covariate shift (IID baseline)
    python mimic_ftl_simulation.py --input aki_features_iid.csv --alpha 0.3 --gamma 0.0

    # Single run with covariate shift
    python mimic_ftl_simulation.py --input aki_features_iid.csv --alpha 0.3 --gamma 0.75

    # Sweep alpha values at fixed gamma
    python mimic_ftl_simulation.py --input aki_features_iid.csv --sweep --gamma 0.5

    # Sweep both alpha and gamma (ablation grid)
    python mimic_ftl_simulation.py --input aki_features_iid.csv --sweep --gamma_sweep

    # Control sample sizes
    python mimic_ftl_simulation.py --input aki_features_iid.csv --alpha 0.3 --gamma 0.5 --n_site 2000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "renal": [
        "baseline_scr", "scr_first_24h", "bun_first_24h",
    ],
    "hemodynamic": [
        "sbp", "dbp", "map", "heart_rate", "pulse", "spo2",
    ],
    "inflammatory": [
        "lactate_first_24h", "wbc_first_24h", "platelets_first_24h",
        "crp", "procalcitonin",
    ],
    "metabolic": [
        "sodium_first_24h", "potassium_first_24h", "bicarbonate_first_24h",
        "hemoglobin_first_24h", "glucose", "albumin", "calcium",
        "phosphate", "magnesium",
    ],
    "clinical": [
        "admission_type", "insurance", "gender", "anchor_age",
        "age_at_admission", "has_diabetes", "has_hypertension",
        "has_chf", "has_sepsis", "has_liver_disease", "has_cancer",
        "baseline_method",
    ],
}

SITE_CONFIGS = {
    "site_C": {
        "description":    "Academic Medical Centre (MIMIC-IV anchor, full feature set)",
        "groups":         ["renal", "hemodynamic", "inflammatory", "metabolic", "clinical"],
        "label_bias":     None,
        "is_anchor":      True,
        "acuity_bias_max": 0.0,
        "spread_scale":    1.0,
    },
    "site_A": {
        "description":    "ICU / Tertiary Hospital",
        "groups":         ["renal", "inflammatory", "metabolic", "clinical"],
        "label_bias":     "high_aki",
        "acuity_bias_max": 8.0,
        "spread_scale":    0.5,
    },
    "site_B": {
        "description":    "General Ward / Secondary Hospital",
        "groups":         ["renal", "inflammatory", "metabolic", "clinical"],
        "label_bias":     "medium_aki",
        "acuity_bias_max": 0.8,
        "spread_scale":    1.0,
    },
    "site_D": {
        "description":    "Community / Primary Care Clinic",
        "groups":         ["metabolic", "clinical"],
        "label_bias":     "low_aki",
        "acuity_bias_max": -3.0,
        "spread_scale":    1.5,
    },
    "site_E": {
        "description":    "Resource-limited / Rural Hospital",
        "groups":         ["renal", "clinical"],
        "label_bias":     "very_low_aki",
        "acuity_bias_max": -5.0,
        "spread_scale":    3.0,
    },
    "site_F": {
        "description":    "Semi-supervised Site (AKI labels withheld)",
        "groups":         ["renal", "inflammatory", "metabolic", "clinical"],
        "label_bias":     "medium_aki",
        "unlabeled":      True,
        "acuity_bias_max": 1.5,
        "spread_scale":    1.1,
    },
}

DIRICHLET_BIAS = {
    "high_aki":     [0.60, 0.40],
    "medium_aki":   [0.825, 0.175],
    "low_aki":      [0.90, 0.10],
    "very_low_aki": [0.95, 0.05],
}

GROUP_COLORS = {
    "renal":        "#1D4ED8",
    "hemodynamic":  "#0891B2",
    "inflammatory": "#B45309",
    "metabolic":    "#065F46",
    "clinical":     "#6B21A8",
    "other":        "#6B7280",
}

SITE_PALETTE = plt.colormaps.get_cmap("tab10")


# ─── FEATURE UTILITIES ───────────────────────────────────────────────────────

def assign_columns_to_groups(columns, feature_groups):
    assigned = {g: [] for g in feature_groups}
    assigned["other"] = []
    claimed = set()
    for group, patterns in feature_groups.items():
        for col in columns:
            if col in claimed:
                continue
            if any(pat.lower() in col.lower() for pat in patterns):
                assigned[group].append(col)
                claimed.add(col)
    for col in columns:
        if col not in claimed:
            assigned["other"].append(col)
    return assigned


def compute_global_stats(df, label_col, group_map):
    clinical = set(group_map.get("clinical", []))
    other    = set(group_map.get("other", []))
    skip     = clinical | other | {label_col}
    stats = {}
    for col in df.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        stats[col] = {
            "mean": float(df[col].mean()),
            "std":  float(df[col].std()),
        }
    return stats


def build_acuity_score(df, label_col, group_map, anchor_mu=None, anchor_sigma=None):
    ACUITY_POSITIVE_KEYS = {
        "creatinine", "bun", "scr", "baseline_scr",
        "lactate", "wbc", "crp", "procalcitonin",
    }
    all_candidates = (
        group_map.get("renal", []) +
        group_map.get("inflammatory", [])
    )
    priority = [c for c in all_candidates
                if any(key in c.lower() for key in ACUITY_POSITIVE_KEYS)]
    numeric = [c for c in priority
               if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if not numeric:
        return df[label_col].fillna(0).astype(float)

    X     = df[numeric].fillna(df[numeric].median())
    score = X.mean(axis=1)

    if anchor_mu is not None and anchor_sigma is not None:
        mu, sigma = anchor_mu, anchor_sigma
    else:
        mu, sigma = score.mean(), score.std()

    if sigma < 1e-9:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return (score - mu) / sigma


def apply_feature_mask(df, keep_groups, group_map, label_col):
    keep = {label_col}
    for g in keep_groups:
        keep.update(group_map.get(g, []))
    keep.update(group_map.get("other", []))
    return df[[c for c in df.columns if c in keep]].copy()


# ─── SAMPLING ────────────────────────────────────────────────────────────────

def subpopulation_sample(df, n_samples, acuity_scores, acuity_bias_max, gamma, rng):
    effective_bias = gamma * acuity_bias_max
    if abs(effective_bias) < 1e-6:
        replace = n_samples > len(df)
        return df.sample(n=n_samples, replace=replace,
                         random_state=int(rng.integers(1_000_000))).reset_index(drop=True)

    logits = effective_bias * acuity_scores.values
    logits = logits - logits.max()
    probs  = np.exp(logits)
    probs  = probs / probs.sum()

    replace = n_samples > len(df)
    idx = rng.choice(len(df), size=n_samples, replace=replace, p=probs)
    return df.iloc[idx].reset_index(drop=True)


def within_class_acuity_sample(df, label_col, n_samples, class_props,
                               acuity_scores, acuity_bias_max, gamma, rng):
    frames = []
    for cls, n_needed in class_props.items():
        cls_df     = df[df[label_col] == cls].copy()
        cls_acuity = acuity_scores.reindex(cls_df.index)

        if n_needed <= 0:
            continue

        effective_bias = gamma * acuity_bias_max
        if abs(effective_bias) < 1e-6 or cls_acuity.isna().all():
            replace = n_needed > len(cls_df)
            frames.append(cls_df.sample(n=n_needed, replace=replace,
                                        random_state=int(rng.integers(1_000_000))))
        else:
            logits = effective_bias * cls_acuity.fillna(0).values
            logits = logits - logits.max()
            probs  = np.exp(logits)
            probs  = probs / probs.sum()
            replace = n_needed > len(cls_df)
            idx = rng.choice(len(cls_df), size=n_needed, replace=replace, p=probs)
            frames.append(cls_df.iloc[idx])

    if not frames:
        return df.sample(n=n_samples, replace=True,
                         random_state=int(rng.integers(1_000_000))).reset_index(drop=True)

    return pd.concat(frames).sample(
        frac=1, random_state=int(rng.integers(1_000_000))
    ).reset_index(drop=True)


def dirichlet_label_sample(pool, label_col, alpha, n_samples, bias_vector, rng):
    if bias_vector is None:
        replace = n_samples > len(pool)
        return pool.sample(n=n_samples, replace=replace,
                           random_state=int(rng.integers(1_000_000))).reset_index(drop=True)

    classes   = sorted(pool[label_col].dropna().unique())
    n_cls     = len(classes)
    bias      = np.array(bias_vector[:n_cls], dtype=float)
    bias_norm = bias / bias.sum()
    # Concentration: conc = alpha * N_SCALE * bias_norm
    # N_SCALE=100 ensures min(conc) >= 1.5 even at alpha=0.3, site_E (bias=0.05)
    # This prevents degenerate near-binary Dirichlet draws.
    # E[proportion] = bias_norm for all alpha (mean always at clinical target)
    # alpha controls variance around that target:
    #   alpha=0.3  -> std ~3-9%  (high variance, extreme non-IID)
    #   alpha=0.5  -> std ~3-7%  (moderate variance, standard non-IID)
    #   alpha=1.0  -> std ~2-5%  (mild variance)
    #   alpha=10.0 -> std ~1-2%  (near-deterministic, tight to targets)
    # IID is achieved via no_label_shift=True (bias_vector=None), not high alpha
    N_SCALE = 100
    conc  = np.clip(alpha * N_SCALE * bias_norm, 1e-3, None)
    props = rng.dirichlet(conc)

    frames, remaining = [], n_samples
    for i, cls in enumerate(classes):
        cls_df = pool[pool[label_col] == cls]
        n = int(round(props[i] * n_samples)) if i < n_cls - 1 else remaining
        n = max(1, min(n, remaining - (n_cls - i - 1)))
        frames.append(cls_df.sample(n=n, replace=n > len(cls_df),
                                    random_state=int(rng.integers(1_000_000))))
        remaining -= n
        if remaining <= 0:
            break

    return pd.concat(frames).sample(frac=1,
                                    random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


# ─── SPREAD PERTURBATION ─────────────────────────────────────────────────────

def apply_spread_perturbation(df, label_col, group_map, spread_scale, gamma, rng):
    if gamma < 1e-6:
        return df.copy(), 1.0

    clinical = set(group_map.get("clinical", []))
    other    = set(group_map.get("other", []))
    skip     = clinical | other | {label_col}

    target_scale = 1.0 + gamma * (spread_scale - 1.0)
    noise        = rng.normal(0.0, gamma * 0.03)
    eff_scale    = max(target_scale + noise, 0.1)

    df_out = df.copy()
    for col in df.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals      = df[col].values.astype(float)
        nan_mask  = np.isnan(vals)
        site_mean = np.nanmean(vals)
        deviation = vals - site_mean
        x_out     = site_mean + eff_scale * deviation
        x_out[nan_mask] = np.nan
        df_out[col]     = x_out

    return df_out, eff_scale


def site_stats(df, label_col, site_id, cfg, gamma, eff_scale):
    labeled   = df[label_col].dropna()
    n_aki     = int((labeled == 1).sum())
    n_labeled = len(labeled)
    return {
        "site":           site_id,
        "description":    cfg["description"],
        "n_total":        len(df),
        "n_labeled":      n_labeled,
        "n_aki":          n_aki,
        "n_no_aki":       n_labeled - n_aki,
        "aki_pct":        round(100 * n_aki / n_labeled, 1) if n_labeled else None,
        "n_features":     len([c for c in df.columns if c != label_col]),
        "feature_cols":   [c for c in df.columns if c != label_col],
        "feature_groups": cfg["groups"],
        "unlabeled":      cfg.get("unlabeled", False),
        "is_anchor":      cfg.get("is_anchor", False),
        "acuity_bias_max": cfg.get("acuity_bias_max", 0.0),
        "spread_scale":   cfg.get("spread_scale", 1.0),
        "eff_scale":      eff_scale,
        "gamma":          gamma,
    }


# ─── PLOTS ───────────────────────────────────────────────────────────────────

def _site_color(i):
    return SITE_PALETTE(i)


def plot_summary(stats_list, group_map, alpha, gamma, output_dir):
    fig = plt.figure(figsize=(17, 10))
    fig.suptitle(
        f"FTL Simulation Summary  —  alpha={alpha},  gamma={gamma}",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs    = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
    sites = [s["site"] for s in stats_list]
    acolors = ["#1D4ED8" if s["is_anchor"] else "#6B7280" for s in stats_list]

    ax1 = fig.add_subplot(gs[0, 0])
    aki_pct = [s["aki_pct"] or 0 for s in stats_list]
    bars = ax1.bar(sites, aki_pct, color=acolors, edgecolor="white")
    ax1.set_title("AKI Prevalence (%)", fontweight="bold")
    ax1.set_ylim(0, 115); ax1.tick_params(axis="x", rotation=35)
    for bar, v in zip(bars, aki_pct):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    n_tots = [s["n_total"] for s in stats_list]
    ax2.bar(sites, n_tots, color="#374151", edgecolor="white")
    ax2.set_title("Sample Size per Site", fontweight="bold")
    ax2.tick_params(axis="x", rotation=35)
    for i, v in enumerate(n_tots):
        ax2.text(i, v + max(n_tots)*0.01, f"{v:,}", ha="center", va="bottom", fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    n_feats = [s["n_features"] for s in stats_list]
    ax3.bar(sites, n_feats, color="#4B5563", edgecolor="white")
    ax3.set_title("Feature Count per Site", fontweight="bold")
    ax3.tick_params(axis="x", rotation=35)
    for i, v in enumerate(n_feats):
        ax3.text(i, v + 0.15, str(v), ha="center", va="bottom", fontsize=9)

    ax4 = fig.add_subplot(gs[1, :2])
    active = [g for g in FEATURE_GROUPS if group_map.get(g)]
    hm = np.array([[1 if g in s["feature_groups"] else 0
                    for g in active] for s in stats_list], dtype=float)
    ax4.imshow(hm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax4.set_xticks(range(len(active)))
    ax4.set_xticklabels([g.capitalize() for g in active], rotation=30, ha="right")
    ax4.set_yticks(range(len(stats_list))); ax4.set_yticklabels(sites)
    ax4.set_title("Feature Groups Available per Site", fontweight="bold")
    for i in range(len(stats_list)):
        for j in range(len(active)):
            ax4.text(j, i, "Y" if hm[i, j] else "–", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if hm[i, j] else "#BBBBBB")

    ax5 = fig.add_subplot(gs[1, 2])
    aki    = [s["n_aki"]    for s in stats_list]
    no_aki = [s["n_no_aki"] for s in stats_list]
    x = np.arange(len(sites))
    ax5.bar(x, no_aki, label="No AKI", color="#D1D5DB", edgecolor="white")
    ax5.bar(x, aki, bottom=no_aki, label="AKI", color="#1D4ED8", edgecolor="white")
    ax5.set_xticks(x); ax5.set_xticklabels(sites, rotation=35, fontsize=9)
    ax5.set_title("Class Distribution per Site", fontweight="bold")
    ax5.set_ylabel("N patients"); ax5.legend(fontsize=9)

    plt.savefig(output_dir / f"summary_alpha{alpha}_gamma{gamma}.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [plot] summary_alpha{alpha}_gamma{gamma}.png")


def plot_feature_coverage(stats_list, group_map, output_dir):
    all_cols, seen = [], set()
    for s in stats_list:
        for c in s["feature_cols"]:
            if c not in seen:
                all_cols.append(c); seen.add(c)

    hm = np.array([[1 if c in set(s["feature_cols"]) else 0
                    for c in all_cols] for s in stats_list], dtype=float)
    col_to_grp = {}
    for col in all_cols:
        for grp, cols in group_map.items():
            if col in cols:
                col_to_grp[col] = grp; break
        else:
            col_to_grp[col] = "other"

    fig, ax = plt.subplots(figsize=(max(14, len(all_cols) * 0.4), 4.5))
    ax.imshow(hm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(stats_list)))
    ax.set_yticklabels([s["site"] for s in stats_list], fontsize=10)
    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=75, ha="right", fontsize=8)
    for tick, col in zip(ax.get_xticklabels(), all_cols):
        tick.set_color(GROUP_COLORS.get(col_to_grp.get(col, "other"), "#000"))
    ax.set_title("Feature Coverage per Site  (colour = clinical group)",
                 fontsize=11, fontweight="bold", pad=10)
    patches = [plt.matplotlib.patches.Patch(color=v, label=k.capitalize())
               for k, v in GROUP_COLORS.items() if k != "other"]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_coverage.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [plot] feature_coverage.png")


def plot_class_distribution_by_alpha(sweep, output_dir):
    alphas   = sorted(sweep.keys())
    site_ids = list(sweep[alphas[0]].keys())
    n_alpha  = len(alphas)
    CC = {"AKI": "#1D4ED8", "No AKI": "#D1D5DB"}

    fig, axes = plt.subplots(
        1, n_alpha,
        figsize=(3.5 * n_alpha, max(4, len(site_ids) * 0.6)),
        sharey=True,
    )
    if n_alpha == 1:
        axes = [axes]

    fig.suptitle(
        "Class Distribution per Site across Dirichlet Alpha Values\n"
        "(each bar = one site; segments = AKI vs no-AKI sample counts)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    y_pos = np.arange(len(site_ids))

    for ax, alpha in zip(axes, alphas):
        aki_c, naki_c, tot_c = [], [], []
        for sid in site_ids:
            s = sweep[alpha][sid]
            aki_c.append(s["n_aki"]); naki_c.append(s["n_no_aki"])
            tot_c.append(s["n_total"])
        max_tot = max(tot_c) if tot_c else 1

        ax.barh(y_pos, naki_c, color=CC["No AKI"], edgecolor="white",
                linewidth=0.5, height=0.75)
        ax.barh(y_pos, aki_c, left=naki_c, color=CC["AKI"],
                edgecolor="white", linewidth=0.5, height=0.75)

        for i, (sid, a, n, tot) in enumerate(
                zip(site_ids, aki_c, naki_c, tot_c)):
            if sweep[alpha][sid].get("unlabeled"):
                ax.text(tot + max_tot * 0.01, i, "labels withheld",
                        ha="left", va="center", fontsize=7,
                        color="#6B7280", fontstyle="italic")
            elif tot > 0:
                pct = 100 * a / tot
                mid = n + a / 2
                if a / max_tot > 0.05:
                    ax.text(mid, i, f"{pct:.0f}%", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
                else:
                    ax.text(tot + max_tot * 0.01, i, f"{pct:.0f}%",
                            ha="left", va="center", fontsize=7,
                            color="#1D4ED8", fontweight="bold")

        ax.set_title(f"alpha = {alpha}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max_tot * 1.3)
        ax.set_xlabel("Sample count", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks(y_pos); ax.set_yticklabels(site_ids, fontsize=8)
        ax.set_ylabel("Site", fontsize=10)
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.legend(
        handles=[plt.matplotlib.patches.Patch(color=CC["No AKI"], label="No AKI"),
                 plt.matplotlib.patches.Patch(color=CC["AKI"],    label="AKI")],
        loc="lower center", ncol=2, fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, -0.06),
    )
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_by_alpha.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [plot] class_distribution_by_alpha.png")


def plot_alpha_sweep(sweep, output_dir, gamma):
    alphas   = sorted(sweep.keys())
    site_ids = list(sweep[alphas[0]].keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Effect of Dirichlet alpha on Site Distributions  (gamma={gamma})",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    for i, sid in enumerate(site_ids):
        vals = [sweep[a][sid]["aki_pct"] or 0 for a in alphas]
        ax.plot(alphas, vals, marker="o", label=sid,
                color=_site_color(i), linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet alpha (log scale)")
    ax.set_ylabel("AKI Prevalence (%)")
    ax.set_title("AKI Prevalence vs alpha\n(lower alpha = more extreme class imbalance)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    feat_counts = [sweep[alphas[0]][sid]["n_features"] for sid in site_ids]
    descs = [sweep[alphas[0]][sid]["description"].split("/")[0].strip()
             for sid in site_ids]
    bars = ax2.barh(site_ids, feat_counts,
                    color=[_site_color(i) for i in range(len(site_ids))],
                    edgecolor="white")
    ax2.set_xlabel("Number of features")
    ax2.set_title("Feature Count per Site\n(clinical masking, alpha-independent)")
    for bar, v, desc in zip(bars, feat_counts, descs):
        ax2.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{v}  ({desc})", va="center", fontsize=8)
    ax2.set_xlim(0, max(feat_counts) * 1.9)
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_sweep.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [plot] alpha_sweep.png")


def _shared_numeric_cols(site_dfs, anchor_df, label_col, group_map):
    clinical = set(group_map.get("clinical", []))
    other    = set(group_map.get("other", []))
    skip     = clinical | other | {label_col}
    candidates = []
    for col in anchor_df.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(anchor_df[col]):
            continue
        n = sum(1 for df in site_dfs.values() if col in df.columns)
        if n >= 2:
            candidates.append(col)
    return candidates


def plot_covariate_shift_violins(site_dfs, label_col, group_map,
                                 alpha, gamma, output_dir):
    anchor    = next(sid for sid, cfg in SITE_CONFIGS.items() if cfg.get("is_anchor"))
    plot_cols = _shared_numeric_cols(site_dfs, site_dfs[anchor],
                                     label_col, group_map)[:8]
    if not plot_cols:
        return

    site_ids    = list(site_dfs.keys())
    colors      = {sid: _site_color(i) for i, sid in enumerate(site_ids)}
    n_cols_plot = min(4, len(plot_cols))
    n_rows_plot = (len(plot_cols) + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(5 * n_cols_plot, 4 * n_rows_plot),
                             squeeze=False)
    fig.suptitle(
        f"Feature Distributions across Sites  (alpha={alpha},  gamma={gamma})\n"
        "Wider violin = more patients at that value. "
        "Offset between sites = covariate shift.",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for idx, col in enumerate(plot_cols):
        row, col_i = divmod(idx, n_cols_plot)
        ax = axes[row][col_i]
        data, labels, fcolors = [], [], []
        for sid in site_ids:
            df = site_dfs[sid]
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) < 5:
                continue
            data.append(vals); labels.append(sid); fcolors.append(colors[sid])

        if len(data) < 2:
            ax.set_visible(False); continue

        parts = ax.violinplot(data, positions=range(len(data)),
                              showmedians=True, showextrema=True, widths=0.7)
        for pc, fc in zip(parts["bodies"], fcolors):
            pc.set_facecolor(fc); pc.set_alpha(0.75)
        parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
        for key in ["cmins", "cmaxes", "cbars"]:
            parts[key].set_color("#374151")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylabel("Value", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(len(plot_cols), n_rows_plot * n_cols_plot):
        row, col_i = divmod(idx, n_cols_plot)
        axes[row][col_i].set_visible(False)

    fig.legend(
        handles=[plt.matplotlib.patches.Patch(color=colors[sid], label=sid)
                 for sid in site_ids],
        loc="lower center", ncol=len(site_ids), fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"covariate_shift_violins_alpha{alpha}_gamma{gamma}.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [plot] covariate_shift_violins_alpha{alpha}_gamma{gamma}.png")


def plot_covariate_shift_kde(site_dfs, label_col, group_map,
                             alpha, gamma, output_dir):
    anchor    = next(sid for sid, cfg in SITE_CONFIGS.items() if cfg.get("is_anchor"))
    plot_cols = _shared_numeric_cols(site_dfs, site_dfs[anchor],
                                     label_col, group_map)[:6]
    if not plot_cols:
        return

    site_ids    = list(site_dfs.keys())
    colors      = {sid: _site_color(i) for i, sid in enumerate(site_ids)}
    n_cols_plot = min(3, len(plot_cols))
    n_rows_plot = (len(plot_cols) + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                             figsize=(5 * n_cols_plot, 3.5 * n_rows_plot),
                             squeeze=False)
    fig.suptitle(
        f"KDE Feature Distributions  (alpha={alpha},  gamma={gamma})\n"
        "Peaks = where patients cluster. "
        "Horizontal offset = mean shift. Width difference = spread shift.",
        fontsize=11, fontweight="bold", y=1.02,
    )

    for idx, col in enumerate(plot_cols):
        row, col_i = divmod(idx, n_cols_plot)
        ax = axes[row][col_i]
        for sid in site_ids:
            df = site_dfs[sid]
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) < 10:
                continue
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            v = vals[(vals >= lo) & (vals <= hi)]
            if len(v) < 5:
                continue
            x_grid = np.linspace(lo, hi, 200)
            try:
                kde = gaussian_kde(v, bw_method="scott")
                ax.plot(x_grid, kde(x_grid), label=sid,
                        color=colors[sid], linewidth=2, alpha=0.85)
                ax.fill_between(x_grid, kde(x_grid), alpha=0.08, color=colors[sid])
            except Exception:
                continue

        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8); ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(len(plot_cols), n_rows_plot * n_cols_plot):
        row, col_i = divmod(idx, n_cols_plot)
        axes[row][col_i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"covariate_shift_kde_alpha{alpha}_gamma{gamma}.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [plot] covariate_shift_kde_alpha{alpha}_gamma{gamma}.png")


def plot_shift_verification(site_dfs, acuity_map, shift_meta,
                            global_stats, label_col, group_map,
                            alpha, gamma, output_dir):
    site_ids = list(site_dfs.keys())
    anchor   = next(sid for sid in site_ids if shift_meta[sid]["is_anchor"])
    anchor_df = site_dfs[anchor]
    colors    = {sid: _site_color(i) for i, sid in enumerate(site_ids)}

    def shared_cols(sid):
        clinical = set(group_map.get("clinical", []))
        other    = set(group_map.get("other", []))
        skip     = clinical | other | {label_col}
        return [c for c in site_dfs[sid].columns
                if c in anchor_df.columns and c not in skip
                and pd.api.types.is_numeric_dtype(site_dfs[sid][c])
                and pd.api.types.is_numeric_dtype(anchor_df[c])]

    def signed_mean_shift(sid):
        DIRECTION = {
            "creatinine": +1, "bun": +1, "scr": +1, "baseline_scr": +1,
            "lactate": +1, "wbc": +1, "crp": +1, "procalcitonin": +1,
            "bicarbonate": -1, "hemoglobin": -1, "platelet": -1,
            "sodium": 0, "potassium": 0, "glucose": 0,
            "albumin": 0, "calcium": 0, "phosphate": 0, "magnesium": 0,
        }
        def get_direction(col):
            col_l = col.lower()
            for key, d in DIRECTION.items():
                if key in col_l:
                    return d
            return None

        clinical = set(group_map.get("clinical", []))
        other    = set(group_map.get("other", []))
        skip     = clinical | other | {label_col}
        diffs = []
        for col in site_dfs[sid].columns:
            if col not in anchor_df.columns or col in skip:
                continue
            if not pd.api.types.is_numeric_dtype(site_dfs[sid][col]):
                continue
            direction = get_direction(col)
            if direction is None or direction == 0:
                continue
            sig = global_stats.get(col, {}).get("std", 1.0)
            if sig < 1e-9:
                continue
            raw_diff = (site_dfs[sid][col].mean() - anchor_df[col].mean()) / sig
            diffs.append(direction * raw_diff)
        return float(np.mean(diffs)) if diffs else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Covariate Shift Verification  (alpha={alpha},  gamma={gamma})\n"
        "Confirms mechanism A (subpopulation sampling drives mean shift) "
        "and mechanism B (spread perturbation only).",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(site_ids))

    # Panel 1 — Class-stratified acuity
    ax = axes[0, 0]
    aki_acuity_means, noaki_acuity_means = [], []
    for sid in site_ids:
        df_site = site_dfs[sid].copy()
        scores  = acuity_map.get(sid, pd.Series(dtype=float, index=df_site.index))
        scores  = scores.reindex(df_site.index)
        labeled = df_site[label_col].dropna()
        aki_idx   = labeled[labeled == 1].index
        noaki_idx = labeled[labeled == 0].index
        aki_sc    = scores.loc[aki_idx].dropna()
        noaki_sc  = scores.loc[noaki_idx].dropna()
        aki_acuity_means.append(float(aki_sc.mean())   if len(aki_sc)   > 0 else np.nan)
        noaki_acuity_means.append(float(noaki_sc.mean()) if len(noaki_sc) > 0 else np.nan)

    anchor_aki_mean   = aki_acuity_means[site_ids.index(anchor)]
    anchor_noaki_mean = noaki_acuity_means[site_ids.index(anchor)]
    ax.plot(x, aki_acuity_means,   "o-", color="#1D4ED8", linewidth=2,
            label="Mean acuity — AKI patients", markersize=7)
    ax.plot(x, noaki_acuity_means, "s--", color="#9CA3AF", linewidth=2,
            label="Mean acuity — no-AKI patients", markersize=7)
    ax.axhline(anchor_aki_mean,   color="#1D4ED8", linestyle=":", linewidth=1,
               alpha=0.5, label=f"Anchor AKI mean ({anchor_aki_mean:.2f})")
    ax.axhline(anchor_noaki_mean, color="#9CA3AF", linestyle=":", linewidth=1,
               alpha=0.5, label=f"Anchor no-AKI mean ({anchor_noaki_mean:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(site_ids, rotation=30, fontsize=9)
    ax.set_ylabel("Mean acuity score (z-score units, anchor-referenced)")
    ax.set_title("Panel 1 — Mechanism A: Class-Stratified Acuity Check\n"
                 "Acuity z-scored relative to anchor (0 = anchor mean). "
                 "ICU AKI above 0; rural AKI below 0.",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2 — Spread
    ax2 = axes[0, 1]
    configured_eff_scale = [shift_meta[sid]["eff_scale"] for sid in site_ids]
    observed_std_ratio   = []
    for sid in site_ids:
        cols = shared_cols(sid)
        if not cols:
            observed_std_ratio.append(1.0); continue
        ratios = [site_dfs[sid][c].std() / anchor_df[c].std()
                  for c in cols if anchor_df[c].std() > 1e-9]
        observed_std_ratio.append(float(np.median(ratios)) if ratios else 1.0)

    ax2.bar(x - 0.2, configured_eff_scale, width=0.35, color="#6B7280",
            label="Configured effective scale", edgecolor="white")
    ax2.bar(x + 0.2, observed_std_ratio, width=0.35,
            color=[colors[s] for s in site_ids],
            label="Observed std ratio vs anchor", edgecolor="white")
    ax2.axhline(1.0, color="#9CA3AF", linestyle="--", linewidth=1,
                label="No spread change")
    ax2.set_xticks(x); ax2.set_xticklabels(site_ids, rotation=30, fontsize=9)
    ax2.set_ylabel("Spread ratio (site / anchor)")
    ax2.set_title("Panel 2 — Mechanism B: Spread Perturbation\n"
                  "<1 = narrower than anchor,  >1 = wider than anchor",
                  fontweight="bold", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)

    # Panel 3 — Sign-adjusted mean shift
    ax3 = axes[1, 0]
    mean_shifts = [signed_mean_shift(sid) for sid in site_ids]
    bars3 = ax3.bar(x, mean_shifts, color=[colors[s] for s in site_ids],
                    edgecolor="white")
    ax3.axhline(0, color="#9CA3AF", linestyle="--", linewidth=1)
    for bar, v in zip(bars3, mean_shifts):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 v + (0.01 if v >= 0 else -0.04),
                 f"{v:+.3f}", ha="center", fontsize=8, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(site_ids, rotation=30, fontsize=9)
    ax3.set_ylabel("Sign-adjusted mean shift vs anchor (all labs)")
    ax3.set_title("Panel 3 — Sign-Adjusted Mean Shift from Anchor\n"
                  "Positive = sicker than anchor (all lab groups, sign-corrected)",
                  fontweight="bold", fontsize=10)
    ax3.grid(axis="y", alpha=0.25); ax3.spines[["top", "right"]].set_visible(False)

    # Panel 4 — Configured vs observed scatter
    ax4 = axes[1, 1]
    for i, sid in enumerate(site_ids):
        ax4.scatter(configured_eff_scale[i], observed_std_ratio[i],
                    color=colors[sid], s=120, zorder=3, label=sid)
        ax4.annotate(sid, (configured_eff_scale[i], observed_std_ratio[i]),
                     textcoords="offset points", xytext=(6, 3), fontsize=8)
    lims = [min(configured_eff_scale + observed_std_ratio) - 0.05,
            max(configured_eff_scale + observed_std_ratio) + 0.05]
    ax4.plot(lims, lims, "--", color="#9CA3AF", linewidth=1.5,
             label="y = x (perfect agreement)")
    ax4.set_xlim(lims); ax4.set_ylim(lims)
    ax4.set_xlabel("Configured effective scale")
    ax4.set_ylabel("Observed std ratio vs anchor")
    ax4.set_title("Panel 4 — Spread: Configured vs Observed\n"
                  "Points near y=x line = perturbation working correctly",
                  fontweight="bold", fontsize=10)
    ax4.legend(fontsize=7, frameon=False)
    ax4.grid(alpha=0.25); ax4.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"shift_verification_alpha{alpha}_gamma{gamma}.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [plot] shift_verification_alpha{alpha}_gamma{gamma}.png")


def plot_gamma_sweep(gamma_sweep_results, global_stats, label_col,
                     group_map, output_dir, alpha):
    gammas   = sorted(gamma_sweep_results.keys())
    site_ids = [sid for sid in SITE_CONFIGS if not SITE_CONFIGS[sid].get("is_anchor")]
    anchor   = next(sid for sid in SITE_CONFIGS if SITE_CONFIGS[sid].get("is_anchor"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Gamma Sensitivity  (alpha={alpha})\n"
        "Left: sign-adjusted mean shift grows with gamma. "
        "Right: spread ratio departs from 1 as gamma increases.",
        fontsize=11, fontweight="bold",
    )

    DIRECTION = {
        "creatinine": +1, "bun": +1, "scr": +1, "baseline_scr": +1,
        "lactate": +1, "wbc": +1,
        "bicarbonate": -1, "hemoglobin": -1, "platelet": -1,
        "sodium": 0, "potassium": 0, "glucose": 0,
        "albumin": 0, "calcium": 0, "phosphate": 0, "magnesium": 0,
    }
    def get_dir(col):
        for k, d in DIRECTION.items():
            if k in col.lower():
                return d
        return None

    for i, sid in enumerate(site_ids):
        mean_shifts, std_ratios = [], []
        for g in gammas:
            entry     = gamma_sweep_results[g]
            site_df   = entry.get(sid + "_df")
            anchor_df = entry.get(anchor + "_df")
            if site_df is None or anchor_df is None:
                mean_shifts.append(np.nan); std_ratios.append(np.nan); continue

            clinical = set(group_map.get("clinical", []))
            other    = set(group_map.get("other", []))
            skip     = clinical | other | {label_col}
            all_cols = [c for c in site_df.columns
                        if c in anchor_df.columns and c not in skip
                        and pd.api.types.is_numeric_dtype(site_df[c])]
            diffs = []
            for c in all_cols:
                d = get_dir(c)
                if d is None or d == 0:
                    continue
                sig = global_stats.get(c, {}).get("std", 1) + 1e-9
                diffs.append(d * (site_df[c].mean() - anchor_df[c].mean()) / sig)
            ratios = [site_df[c].std() / anchor_df[c].std()
                      for c in all_cols if anchor_df[c].std() > 1e-9]
            mean_shifts.append(float(np.mean(diffs))  if diffs  else 0.0)
            std_ratios.append(float(np.median(ratios)) if ratios else 1.0)

        axes[0].plot(gammas, mean_shifts, marker="o", label=sid,
                     color=_site_color(i), linewidth=2)
        axes[1].plot(gammas, std_ratios, marker="o", label=sid,
                     color=_site_color(i), linewidth=2)

    axes[0].axhline(0,   color="#9CA3AF", linestyle="--", linewidth=1)
    axes[1].axhline(1.0, color="#9CA3AF", linestyle="--", linewidth=1)
    axes[0].set_xlabel("gamma"); axes[0].set_ylabel("Sign-adjusted mean shift vs anchor")
    axes[1].set_xlabel("gamma"); axes[1].set_ylabel("Median std ratio vs anchor")
    axes[0].set_title("Mean Shift vs gamma", fontweight="bold")
    axes[1].set_title("Spread Ratio vs gamma", fontweight="bold")
    for ax in axes:
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, alpha=0.3); ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"gamma_sweep_alpha{alpha}.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [plot] gamma_sweep_alpha{alpha}.png")


# ─── CORE ────────────────────────────────────────────────────────────────────

def run_simulation(df, label_col, alpha, gamma, seed, output_dir,
                   site_sizes=None, no_label_shift=False):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c != label_col]
    group_map    = assign_columns_to_groups(feature_cols, FEATURE_GROUPS)

    print("\n── Feature group assignment ──────────────────────────────────")
    for grp, cols in group_map.items():
        if cols:
            print(f"  {grp:15s} ({len(cols):2d}): {', '.join(cols[:6])}"
                  + (" ..." if len(cols) > 6 else ""))

    with open(output_dir / "group_map.json", "w") as f:
        json.dump(group_map, f, indent=2)

    print("  Computing acuity scores ...")
    acuity_global = build_acuity_score(df, label_col, group_map)

    n_total = len(df)
    if site_sizes is None:
        n_non    = sum(1 for c in SITE_CONFIGS.values() if not c.get("is_anchor"))
        n_anchor = int(n_total * 0.30)
        n_each   = max(1, (n_total - n_anchor) // n_non)
        site_sizes = {sid: (n_anchor if cfg.get("is_anchor") else n_each)
                      for sid, cfg in SITE_CONFIGS.items()}

    anchor_id  = next(sid for sid, cfg in SITE_CONFIGS.items() if cfg.get("is_anchor"))
    anchor_pre = dirichlet_label_sample(
        df, label_col, alpha, site_sizes[anchor_id], None,
        np.random.default_rng(seed),
    )
    global_stats = compute_global_stats(anchor_pre, label_col, group_map)

    priority_cols = (
        group_map.get("renal", []) +
        group_map.get("inflammatory", []) +
        group_map.get("hemodynamic", [])
    )
    numeric_cols = [c for c in priority_cols
                    if c in anchor_pre.columns
                    and pd.api.types.is_numeric_dtype(anchor_pre[c])]
    if numeric_cols:
        _raw = anchor_pre[numeric_cols].fillna(anchor_pre[numeric_cols].median())
        _raw_score = _raw.mean(axis=1)
        anchor_acuity_mu    = float(_raw_score.mean())
        anchor_acuity_sigma = float(_raw_score.std())
        if anchor_acuity_sigma < 1e-9:
            anchor_acuity_sigma = 1.0
    else:
        anchor_acuity_mu, anchor_acuity_sigma = 0.0, 1.0

    label_mode = "NO LABEL SHIFT" if no_label_shift else f"alpha={alpha}"
    print(f"\n── Simulation  {label_mode}  gamma={gamma}  seed={seed}  N={n_total:,} ──")

    anchor_presampled = {anchor_id: anchor_pre}
    stats_all, stats_list, site_dfs, acuity_map, shift_meta = {}, [], {}, {}, {}

    for site_id, cfg in SITE_CONFIGS.items():
        bias_vec = None if no_label_shift else (
            DIRICHLET_BIAS.get(cfg["label_bias"]) if cfg.get("label_bias") else None
        )

        if cfg.get("is_anchor"):
            sampled = anchor_presampled[site_id]
        else:
            n_site = site_sizes[site_id]
            dirichlet_sample = dirichlet_label_sample(
                df, label_col, alpha, n_site, bias_vec, rng,
            )
            if abs(gamma * cfg.get("acuity_bias_max", 0.0)) < 1e-6:
                sampled = dirichlet_sample
            else:
                classes     = sorted(df[label_col].dropna().unique())
                class_props = {cls: int((dirichlet_sample[label_col] == cls).sum())
                               for cls in classes}
                sampled = within_class_acuity_sample(
                    df, label_col, n_site, class_props,
                    acuity_global, cfg.get("acuity_bias_max", 0.0), gamma, rng,
                )

        acuity_map[site_id] = build_acuity_score(
            sampled, label_col, group_map,
            anchor_mu=anchor_acuity_mu, anchor_sigma=anchor_acuity_sigma,
        )

        masked = apply_feature_mask(sampled, cfg["groups"], group_map, label_col)

        eff_scale = 1.0
        if not cfg.get("is_anchor"):
            masked, eff_scale = apply_spread_perturbation(
                masked, label_col, group_map,
                cfg.get("spread_scale", 1.0), gamma, rng,
            )

        shift_meta[site_id] = {
            "effective_bias": gamma * cfg.get("acuity_bias_max", 0.0),
            "spread_scale":   cfg.get("spread_scale", 1.0),
            "eff_scale":      eff_scale,
            "is_anchor":      cfg.get("is_anchor", False),
        }

        if cfg.get("unlabeled"):
            masked = masked.copy()
            masked[label_col] = np.nan

        # Include no_label_shift tag in filename so training script can
        # distinguish IID (no label shift) from label-shifted conditions
        ls_suffix = "_no_label_shift" if no_label_shift else ""
        masked.to_csv(
            output_dir / f"{site_id}_alpha{alpha}_gamma{gamma}{ls_suffix}.csv",
            index=False,
        )
        site_dfs[site_id] = masked

        s = site_stats(masked, label_col, site_id, cfg, gamma, eff_scale)
        stats_all[site_id] = s
        stats_list.append(s)

        aki_str = (f"{s['n_aki']:,} AKI / {s['n_no_aki']:,} no-AKI  ({s['aki_pct']}%)"
                   if not cfg.get("unlabeled") else "labels withheld")
        print(f"  {site_id:8s} | {s['n_total']:6,} rows | {s['n_features']:2d} feat "
              f"| eff_bias={gamma * cfg.get('acuity_bias_max',0):+.2f} "
              f"| eff_scale={eff_scale:.3f} | {aki_str}")

    ls_suffix = "_no_label_shift" if no_label_shift else ""
    with open(output_dir / f"stats_alpha{alpha}_gamma{gamma}{ls_suffix}.json", "w") as f:
        json.dump(stats_all, f, indent=2)

    global_stats_path = output_dir / "global_stats.json"
    if not global_stats_path.exists():
        with open(global_stats_path, "w") as f:
            json.dump(global_stats, f, indent=2)
        print(f"  [saved] global_stats.json")

    group_map_path = output_dir / "group_map.json"
    if not group_map_path.exists():
        with open(group_map_path, "w") as f:
            json.dump(group_map, f, indent=2)
        print(f"  [saved] group_map.json")

    plot_summary(stats_list, group_map, alpha, gamma, output_dir)
    plot_feature_coverage(stats_list, group_map, output_dir)
    plot_covariate_shift_violins(site_dfs, label_col, group_map, alpha, gamma, output_dir)
    plot_covariate_shift_kde(site_dfs, label_col, group_map, alpha, gamma, output_dir)
    plot_shift_verification(site_dfs, acuity_map, shift_meta, global_stats,
                            label_col, group_map, alpha, gamma, output_dir)

    return stats_all, site_dfs, group_map, global_stats


# ─── CLI ─────────────────────────────────────────────────────────────────────

def load_and_validate(input_path, label_col):
    print(f"\nLoading {input_path} ...")
    df = pd.read_csv(input_path)
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} columns")

    if label_col not in df.columns:
        candidates = [c for c in df.columns
                      if "aki" in c.lower() or "label" in c.lower()]
        print(f"\n[ERROR] Label column '{label_col}' not found.")
        if candidates:
            print(f"  Try: --label {candidates[0]}")
        sys.exit(1)

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    n_null = df[label_col].isna().sum()
    if n_null:
        print(f"  [!] Dropping {n_null} rows with null label")
        df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)

    counts = df[label_col].value_counts().sort_index().to_dict()
    print(f"  Label distribution: {counts}  "
          f"(AKI rate: {round(100*counts.get(1,0)/len(df),1)}%)")
    return df


def main():
    p = argparse.ArgumentParser(
        description="Simulate non-IID federated sites from aki_features_iid.csv"
    )
    p.add_argument("--input",          default="aki_features_iid.csv",
                   help="Input CSV file (default: aki_features_iid.csv)")
    p.add_argument("--label",          default="AKI_label")
    p.add_argument("--alpha",          type=float, default=0.3)
    p.add_argument("--gamma",          type=float, default=0.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output",         default="./simulated_sites")
    p.add_argument("--sweep",          action="store_true")
    p.add_argument("--gamma_sweep",    action="store_true")
    p.add_argument("--n_site",         type=int, default=None)
    p.add_argument("--no_label_shift", action="store_true")
    args = p.parse_args()

    df         = load_and_validate(args.input, args.label)
    output_dir = Path(args.output)

    site_sizes = None
    if args.n_site:
        site_sizes = {sid: args.n_site for sid in SITE_CONFIGS}
        anchor = next(sid for sid, c in SITE_CONFIGS.items() if c.get("is_anchor"))
        site_sizes[anchor] = int(args.n_site * 2.143)

    alphas = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0] if args.sweep else [args.alpha]
    gammas = ([0.0, 0.25, 0.5, 0.75, 1.0]
              if (args.sweep and args.gamma_sweep) else [args.gamma])

    sweep_results     = {}
    gamma_sweep_store = {}
    group_map_ref     = None
    global_stats_ref  = None

    for gamma in gammas:
        sweep_results = {}
        for alpha in alphas:
            print(f"\n{'='*60}\n  alpha={alpha}   gamma={gamma}\n{'='*60}")
            stats, site_dfs, group_map_ref, global_stats_ref = run_simulation(
                df, args.label, alpha, gamma, args.seed, output_dir, site_sizes,
                no_label_shift=args.no_label_shift,
            )
            sweep_results[alpha] = stats
            if args.gamma_sweep:
                if gamma not in gamma_sweep_store:
                    gamma_sweep_store[gamma] = {}
                for sid, sdf in site_dfs.items():
                    gamma_sweep_store[gamma][sid + "_df"] = sdf

        if args.sweep:
            plot_alpha_sweep(sweep_results, output_dir, gamma)
            plot_class_distribution_by_alpha(sweep_results, output_dir)

    if args.gamma_sweep and len(gammas) > 1 and group_map_ref and global_stats_ref:
        plot_gamma_sweep(
            gamma_sweep_store, global_stats_ref, args.label,
            group_map_ref, output_dir, alpha=alphas[-1],
        )

    print(f"\nDone -> {output_dir}/")


if __name__ == "__main__":
    main()
