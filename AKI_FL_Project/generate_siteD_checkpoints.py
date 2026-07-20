"""
Generate intermediate feature-count checkpoints for the site_D unmasking
experiment — run this locally in your AKI_FL_Project folder. Uses only
files you already have: site_C (full 159-feature reference), the original
site_D (67-feature baseline), and the already-unmasked site_D (159-feature).

Produces, for each checkpoint in CHECKPOINTS below:
    site_D_alpha0.3_gamma0.75_{N}feat.csv
    site_D_adapter_meta_{N}feat.json

Usage:
    python3 generate_siteD_checkpoints.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
CONDITION = "alpha0.3_gamma0.75"
CHECKPOINTS = [15, 45]  # features to add beyond the 67-feature baseline -> 82, 112

# Output goes into its own condition-scoped subfolder, not the project root —
# avoids any ambiguity between separate runs (e.g. re-running with different
# CHECKPOINTS values at the same condition, or running at a different
# condition later) about which generated files are current.
OUTPUT_DIR = f"siteD_checkpoints_{CONDITION}"

ORIG_SITE_D_PATH = f"phase3_sites_aligned/{CONDITION}/site_D_{CONDITION}.csv"
UNMASKED_SITE_D_PATH = f"phase3_sites_UNMASKED_D/{CONDITION}/site_D_{CONDITION}.csv"
SITE_C_PATH = f"phase3_sites_aligned/{CONDITION}/site_C_{CONDITION}.csv"

LABEL_COL = "AKI_label"


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    return X


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}/")

    print("Loading local files...")
    orig_d = pd.read_csv(ORIG_SITE_D_PATH)
    unmasked_d = pd.read_csv(UNMASKED_SITE_D_PATH)
    site_c = pd.read_csv(SITE_C_PATH)

    orig_features = [c for c in orig_d.columns if c != LABEL_COL]
    all_features = [c for c in unmasked_d.columns if c != LABEL_COL]
    newly_available = [f for f in all_features if f not in orig_features]
    print(f"Baseline: {len(orig_features)} features. "
          f"Newly available (previously masked): {len(newly_available)} features.")

    # RF feature importance, computed from a 90% holdout of site_C
    # (the one site with all 159 features) — same methodology used
    # for the original Tier-1 enrichment test.
    print("Computing RF feature importance from site_C holdout...")
    _, reference_df = train_test_split(
        site_c, test_size=0.90, stratify=site_c[LABEL_COL], random_state=RANDOM_STATE
    )
    X_ref = encode_categoricals(reference_df[all_features])
    imputer = SimpleImputer(strategy="median").fit(X_ref)
    X_ref_imp = imputer.transform(X_ref)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_ref_imp, reference_df[LABEL_COL])
    importances = pd.Series(rf.feature_importances_, index=all_features)
    ranked_new = importances.loc[importances.index.intersection(newly_available)].sort_values(ascending=False)
    feature_order = list(ranked_new.index)
    print(f"Ranked {len(feature_order)} newly-available features by importance.")

    for n_added in CHECKPOINTS:
        current_features = orig_features + feature_order[:n_added]
        n_total = len(current_features)
        subset = unmasked_d[current_features + [LABEL_COL]]

        csv_path = f"{OUTPUT_DIR}/site_D_{CONDITION}_{n_total:03d}feat.csv"
        if os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} already exists — overwriting. "
                  f"If this wasn't intentional, check CHECKPOINTS for a repeat value.")
        subset.to_csv(csv_path, index=False)

        meta = {
            "site_id": "site_D",
            "input_dim": n_total,
            "embedding_dim": 64,
            "feature_names": current_features,
            "aki_prevalence": float(subset[LABEL_COL].mean()),
            "unlabeled": False,
        }
        meta_path = f"{OUTPUT_DIR}/site_D_adapter_meta_{n_total:03d}feat.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {n_total} features -> {csv_path}, {meta_path} "
              f"(prevalence={meta['aki_prevalence']:.4f})")

    print(f"\nDone. Files are in {OUTPUT_DIR}/. Copy each into its "
          f"phase3_sites_UNMASKED_D_XXX/{CONDITION}/ folder, "
          f"renaming to site_D_{CONDITION}.csv / site_D_adapter_meta.json.")


if __name__ == "__main__":
    main()
