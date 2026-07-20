"""
Progressive feature-enrichment experiment (Dr. Li's proposal) — Tier 1, cheap version.

Builds a synthetic low-performer site ("site_F") by stratified 10%-sampling
site_C (the one site with all 159 aligned features), held out from a
"reference" model built on the other 90% of site_C. Starting from the
12-feature sparse baseline (shared across all 5 real sites), features are
added in RF-importance-ranked order, and at each step we measure:

  - local_auroc: site_F's own cross-validated AUROC at this feature subset
                 (feature CONTRIBUTION — does adding this feature help
                 site_F's own local model?)
  - boundary_cosine_sim: cosine similarity between site_F's local decision
                 boundary and the reference model's boundary, same subset
                 (feature SENSITIVITY / convergence — does site_F's boundary
                 become more similar to a well-resourced site's as features
                 are added?)

Caveat this run does NOT cover: validating on the REAL site_D (which needs
the master unmasked dataset, not available in this environment) — this is
the synthetic-testbed half of the design only.

Usage:
    python3 progressive_feature_enrichment.py \
        --sites_dir phase3_sites_aligned --condition alpha0.3_gamma0.75 \
        --out_dir enrichment_out --step_size 10
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

BASELINE_FEATURES = [
    "admission_type", "age_at_admission", "gender", "has_cancer", "has_chf",
    "has_diabetes", "has_hypertension", "has_liver_disease", "has_sepsis",
    "n_distinct_meds", "nephrotoxic_count", "nephrotoxic_flag",
]
RANDOM_STATE = 42


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    return X


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sites_dir", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--step_size", type=int, default=10,
                    help="How many features to add per enrichment step")
    p.add_argument("--C", type=float, default=1.0,
                    help="Inverse L2 regularization strength (sklearn LogisticRegression default C=1.0). "
                         "Lower = stronger regularization — worth testing smaller values given the "
                         "small site_F sample size (3,300 rows) vs. growing feature dimensionality.")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load site_C (full 159-feature site) and split ---
    path = os.path.join(args.sites_dir, args.condition, f"site_C_{args.condition}.csv")
    df = pd.read_csv(path)
    all_features = [c for c in df.columns if c != "AKI_label"]
    assert set(BASELINE_FEATURES).issubset(set(all_features)), "Baseline features missing from site_C"
    candidate_features = [f for f in all_features if f not in BASELINE_FEATURES]

    site_f_df, reference_df = train_test_split(
        df, test_size=0.90, stratify=df["AKI_label"], random_state=RANDOM_STATE
    )
    print(f"site_F (synthetic low-performer): {len(site_f_df)} rows "
          f"(AKI rate {site_f_df['AKI_label'].mean():.3f})")
    print(f"reference (well-resourced, held out from site_F): {len(reference_df)} rows "
          f"(AKI rate {reference_df['AKI_label'].mean():.3f})")

    # --- RF feature importance, computed from the REFERENCE set only (no leakage) ---
    X_ref_full = encode_categoricals(reference_df[all_features])
    imputer_full = SimpleImputer(strategy="median").fit(X_ref_full)
    X_ref_full_imp = imputer_full.transform(X_ref_full)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_ref_full_imp, reference_df["AKI_label"])
    importances = pd.Series(rf.feature_importances_, index=all_features)
    ranked_candidates = importances.loc[candidate_features].sort_values(ascending=False)
    ranked_candidates.to_csv(os.path.join(args.out_dir, "feature_importance_ranking.csv"))
    print(f"\nTop 10 candidate features by importance (from reference set):")
    print(ranked_candidates.head(10))

    # --- Progressive enrichment loop ---
    results = []
    feature_order = list(ranked_candidates.index)
    n_steps = list(range(0, len(feature_order) + 1, args.step_size))
    if n_steps[-1] != len(feature_order):
        n_steps.append(len(feature_order))

    for n_added in n_steps:
        current_features = BASELINE_FEATURES + feature_order[:n_added]

        X_f = encode_categoricals(site_f_df[current_features])
        y_f = site_f_df["AKI_label"].values
        X_r = encode_categoricals(reference_df[current_features])
        y_r = reference_df["AKI_label"].values

        X_pooled = pd.concat([X_f, X_r], ignore_index=True)
        imputer = SimpleImputer(strategy="median").fit(X_pooled)
        X_f_imp = imputer.transform(X_f)
        X_r_imp = imputer.transform(X_r)
        scaler = StandardScaler().fit(np.vstack([X_f_imp, X_r_imp]))
        X_f_scaled = scaler.transform(X_f_imp)
        X_r_scaled = scaler.transform(X_r_imp)

        # local_auroc: 5-fold CV AUROC of site_F's OWN model on its OWN data
        clf_f = LogisticRegression(max_iter=1000, C=args.C)
        try:
            cv_scores = cross_val_score(clf_f, X_f_scaled, y_f, cv=5, scoring="roc_auc")
            local_auroc = cv_scores.mean()
        except Exception:
            local_auroc = np.nan

        # boundary_cosine_sim: fit both on full data, compare coefficients
        clf_f_full = LogisticRegression(max_iter=1000, C=args.C).fit(X_f_scaled, y_f)
        clf_r_full = LogisticRegression(max_iter=1000, C=args.C).fit(X_r_scaled, y_r)
        boundary_sim = cosine_sim(clf_f_full.coef_.flatten(), clf_r_full.coef_.flatten())

        results.append({
            "n_features_added": n_added,
            "n_total_features": len(current_features),
            "local_auroc": round(local_auroc, 4),
            "boundary_cosine_sim_vs_reference": round(boundary_sim, 4),
        })
        print(f"  +{n_added:3d} features ({len(current_features):3d} total): "
              f"local_auroc={local_auroc:.4f}  boundary_sim={boundary_sim:.4f}")

    res_df = pd.DataFrame(results)
    res_df["marginal_auroc_gain"] = res_df["local_auroc"].diff()
    res_df.to_csv(os.path.join(args.out_dir, "progressive_enrichment_results.csv"), index=False)
    print(f"\nSaved: {args.out_dir}/progressive_enrichment_results.csv")
    print(f"Saved: {args.out_dir}/feature_importance_ranking.csv")


if __name__ == "__main__":
    main()
