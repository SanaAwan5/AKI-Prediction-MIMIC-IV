"""
Pre-training per-site diagnostic metrics for the FTL/AKI site_D investigation.

Computes three families of metrics, none requiring federated training:

1. Prevalence distance from the federated mean (pure label statistics).
2. Local decision-boundary distance + signal-to-noise ratio, using the
   features common to ALL 5 sites (demographics/comorbidities only — no
   labs, since lab availability is exactly what differs by site).
3. The richer site_D-vs-site_B PAIRWISE comparison, using their much
   larger shared lab-feature set (Jaccard=0.848) — this is the one most
   likely to actually carry clinical signal, since it isn't bottlenecked
   by the lowest-common-denominator across all 5 sites.

Usage:
    python3 compute_pretraining_metrics.py --sites_dir phase3_sites_aligned \
        --conditions alpha0.3_gamma0.75 alpha0.1_gamma1.0 \
        --out_dir pretraining_metrics_out

Run with --conditions all to sweep the full 5x4 grid (20 conditions).
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SITES = ["A", "B", "C", "D", "E"]
COMMON_5SITE_FEATURES = [
    "admission_type", "age_at_admission", "gender", "has_cancer", "has_chf",
    "has_diabetes", "has_hypertension", "has_liver_disease", "has_sepsis",
    "n_distinct_meds", "nephrotoxic_count", "nephrotoxic_flag",
]
N_BOOTSTRAP = 8
BOOTSTRAP_SEED = 42


def load_site(sites_dir: str, site: str, condition: str) -> pd.DataFrame:
    path = os.path.join(sites_dir, condition, f"site_{site}_{condition}.csv")
    return pd.read_csv(path)


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    return X


def fit_and_bootstrap(X_scaled: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP):
    """Fit once on the full data, plus n_boot bootstrap refits for SNR."""
    clf = LogisticRegression(max_iter=1000).fit(X_scaled, y)
    coef = clf.coef_.flatten()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_coefs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        try:
            boot_clf = LogisticRegression(max_iter=500).fit(X_scaled[idx], y[idx])
            boot_coefs.append(boot_clf.coef_.flatten())
        except Exception:
            continue
    boot_coefs = np.array(boot_coefs)
    coef_std = boot_coefs.std(axis=0).mean() if len(boot_coefs) else np.nan
    coef_mean_mag = np.abs(boot_coefs.mean(axis=0)).mean() if len(boot_coefs) else np.nan
    snr = coef_mean_mag / (coef_std + 1e-8) if len(boot_coefs) else np.nan
    return coef, snr, coef_std


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compute_prevalence_metrics(sites_dir: str, condition: str) -> pd.DataFrame:
    rows = []
    prevalences = {}
    for s in SITES:
        df = load_site(sites_dir, s, condition)
        prevalences[s] = df["AKI_label"].mean()
    fed_mean = np.mean(list(prevalences.values()))
    for s in SITES:
        rows.append({
            "condition": condition, "site": s,
            "prevalence": prevalences[s],
            "prevalence_distance": abs(prevalences[s] - fed_mean),
        })
    return pd.DataFrame(rows)


def compute_5site_boundary_metrics(sites_dir: str, condition: str) -> pd.DataFrame:
    site_data = {}
    for s in SITES:
        df = load_site(sites_dir, s, condition)
        X = encode_categoricals(df[COMMON_5SITE_FEATURES])
        y = df["AKI_label"].values
        site_data[s] = (X.values, y)

    X_pooled = np.vstack([site_data[s][0] for s in SITES])
    y_pooled = np.hstack([site_data[s][1] for s in SITES])
    imputer = SimpleImputer(strategy="median").fit(X_pooled)
    X_pooled = imputer.transform(X_pooled)
    scaler = StandardScaler().fit(X_pooled)
    X_pooled_scaled = scaler.transform(X_pooled)
    pooled_coef, _, _ = fit_and_bootstrap(X_pooled_scaled, y_pooled, n_boot=0)

    rows = []
    for s in SITES:
        X_s, y_s = site_data[s]
        X_s = imputer.transform(X_s)
        X_s_scaled = scaler.transform(X_s)
        local_coef, snr, coef_std = fit_and_bootstrap(X_s_scaled, y_s)
        rows.append({
            "condition": condition, "site": s,
            "boundary_cosine_sim_5site": cosine_sim(local_coef, pooled_coef),
            "signal_to_noise_5site": snr,
            "coef_bootstrap_std_5site": coef_std,
        })
    return pd.DataFrame(rows)


def compute_bd_pairwise_metrics(sites_dir: str, condition: str) -> pd.DataFrame:
    dfB = load_site(sites_dir, "B", condition)
    dfD = load_site(sites_dir, "D", condition)
    shared = sorted((set(dfB.columns) & set(dfD.columns)) - {"AKI_label"})

    XB = encode_categoricals(dfB[shared]).values
    yB = dfB["AKI_label"].values
    XD = encode_categoricals(dfD[shared]).values
    yD = dfD["AKI_label"].values

    X_pooled = np.vstack([XB, XD])
    imputer = SimpleImputer(strategy="median").fit(X_pooled)
    XB = imputer.transform(XB)
    XD = imputer.transform(XD)
    X_pooled = imputer.transform(X_pooled)
    scaler = StandardScaler().fit(X_pooled)
    XB_scaled = scaler.transform(XB)
    XD_scaled = scaler.transform(XD)

    coefB, snrB, stdB = fit_and_bootstrap(XB_scaled, yB)
    coefD, snrD, stdD = fit_and_bootstrap(XD_scaled, yD)
    bd_cosine = cosine_sim(coefB, coefD)

    return pd.DataFrame([
        {"condition": condition, "site": "B", "n_shared_features": len(shared),
         "signal_to_noise_BD": snrB, "coef_bootstrap_std_BD": stdB,
         "boundary_cosine_sim_B_vs_D": bd_cosine},
        {"condition": condition, "site": "D", "n_shared_features": len(shared),
         "signal_to_noise_BD": snrD, "coef_bootstrap_std_BD": stdD,
         "boundary_cosine_sim_B_vs_D": bd_cosine},
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sites_dir", required=True)
    p.add_argument("--conditions", nargs="+", required=True,
                    help="Condition folder names, e.g. alpha0.3_gamma0.75, or 'all' to sweep every folder found")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.conditions == ["all"]:
        conditions = sorted(
            os.path.basename(d) for d in glob.glob(os.path.join(args.sites_dir, "alpha*_gamma*"))
            if os.path.isdir(d)
        )
    else:
        conditions = args.conditions

    prev_rows, b5_rows, bd_rows = [], [], []
    for cond in conditions:
        print(f"[metrics] {cond} ...")
        prev_rows.append(compute_prevalence_metrics(args.sites_dir, cond))
        b5_rows.append(compute_5site_boundary_metrics(args.sites_dir, cond))
        bd_rows.append(compute_bd_pairwise_metrics(args.sites_dir, cond))

    prev_df = pd.concat(prev_rows, ignore_index=True)
    b5_df = pd.concat(b5_rows, ignore_index=True)
    bd_df = pd.concat(bd_rows, ignore_index=True)

    merged = prev_df.merge(b5_df, on=["condition", "site"], how="outer")
    merged = merged.merge(bd_df, on=["condition", "site"], how="left")

    merged.to_csv(os.path.join(args.out_dir, "pretraining_metrics_all_sites.csv"), index=False)
    bd_df.to_csv(os.path.join(args.out_dir, "boundary_distance_B_vs_D.csv"), index=False)
    print(f"\nSaved: {args.out_dir}/pretraining_metrics_all_sites.csv")
    print(f"Saved: {args.out_dir}/boundary_distance_B_vs_D.csv")


if __name__ == "__main__":
    main()
