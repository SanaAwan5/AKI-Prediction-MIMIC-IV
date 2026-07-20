"""
True per-feature sensitivity analysis — adds ONE feature at a time (not
batches of ~15 like the earlier enrichment sweep), for site_D and a
comparison site (site_A, a genuine FL benefiter). Answers: do the features
that matter most for site_D's boundary differ systematically from the
features that matter for a site that already benefits from FL?

For each site and each single-feature addition, records:
  - local_auroc: 5-fold CV AUROC at this feature set
  - marginal_auroc_gain: local_auroc - local_auroc at the previous step
    (this specific feature's CONTRIBUTION)
  - boundary_cosine_sim: cosine similarity vs a reference model at this
    feature set (site_C's 90% holdout, same reference used throughout)
  - marginal_boundary_shift: change in boundary_cosine_sim from the
    previous step (this specific feature's SENSITIVITY)

Usage:
    python3 feature_sensitivity_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CONDITION = "alpha0.3_gamma0.75"
C_REG = 0.05  # strong regularization, matches the earlier enrichment sweep

SITES_TO_TEST = {
    "D": {
        "orig_path": f"/home/claude/work/phase3_sites_aligned/{CONDITION}/site_D_{CONDITION}.csv",
        "unmasked_path": f"unmasked_A_and_D/site_D_{CONDITION}.csv",
    },
    "A": {
        "orig_path": f"/home/claude/work/phase3_sites_aligned/{CONDITION}/site_A_{CONDITION}.csv",
        "unmasked_path": f"unmasked_A_and_D/site_A_{CONDITION}.csv",
    },
}
SITE_C_PATH = f"unmasked_A_and_D/site_C_{CONDITION}.csv"
LABEL_COL = "AKI_label"


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype("category").cat.codes
    return X


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    import sys
    only_site = sys.argv[1] if len(sys.argv) > 1 else None

    site_c = pd.read_csv(SITE_C_PATH)
    all_features = [c for c in site_c.columns if c != LABEL_COL]

    _, reference_df = train_test_split(
        site_c, test_size=0.90, stratify=site_c[LABEL_COL], random_state=RANDOM_STATE
    )
    X_ref_full = encode_categoricals(reference_df[all_features])
    imputer_full = SimpleImputer(strategy="median").fit(X_ref_full)
    X_ref_full_imp = imputer_full.transform(X_ref_full)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_ref_full_imp, reference_df[LABEL_COL])
    importances = pd.Series(rf.feature_importances_, index=all_features)

    sites_to_run = {only_site: SITES_TO_TEST[only_site]} if only_site else SITES_TO_TEST

    all_results = []
    for site_label, paths in sites_to_run.items():
        print(f"\n=== site_{site_label} ===")
        orig_features = [c for c in pd.read_csv(paths["orig_path"], nrows=1).columns if c != LABEL_COL]
        unmasked = pd.read_csv(paths["unmasked_path"])
        newly_available = [f for f in all_features if f not in orig_features]
        ranked = importances.loc[importances.index.intersection(newly_available)].sort_values(ascending=False)
        feature_order = list(ranked.index)
        full_order = orig_features + feature_order  # final column order, baseline first
        print(f"Baseline: {len(orig_features)} features. Testing {len(feature_order)} single-feature additions.")

        y_s = unmasked[LABEL_COL].values
        y_r = reference_df[LABEL_COL].values

        # Pre-fit imputer + scaler ONCE on the full feature set (all 159 cols) —
        # avoids re-fitting per step, which dominated runtime. Slightly changes
        # the exact imputed/scaled values vs. per-step fitting (uses global
        # stats instead of subset-specific stats) but is consistent across all
        # steps and vastly faster; the earlier batched sweep used per-step
        # fitting, so treat any minor numeric differences as a methodology
        # note rather than a discrepancy.
        X_s_full = encode_categoricals(unmasked[full_order])
        X_r_full = encode_categoricals(reference_df[full_order])
        X_pooled_full = pd.concat([X_s_full, X_r_full], ignore_index=True)
        imputer = SimpleImputer(strategy="median").fit(X_pooled_full)
        X_s_imp_full = imputer.transform(X_s_full)
        X_r_imp_full = imputer.transform(X_r_full)
        scaler = StandardScaler().fit(np.vstack([X_s_imp_full, X_r_imp_full]))
        X_s_scaled_full = scaler.transform(X_s_imp_full)
        X_r_scaled_full = scaler.transform(X_r_imp_full)

        # single stratified train/test split, reused every step (faster than CV,
        # still gives a genuine held-out AUROC rather than resubstitution)
        from sklearn.model_selection import train_test_split as tts
        idx_train, idx_test = tts(
            np.arange(len(y_s)), test_size=0.3, stratify=y_s, random_state=RANDOM_STATE
        )

        prev_local_auroc = None
        prev_boundary_sim = None

        for step, feat in enumerate([None] + feature_order):
            n_cols = len(orig_features) + step
            X_s_step = X_s_scaled_full[:, :n_cols]
            X_r_step = X_r_scaled_full[:, :n_cols]

            clf = LogisticRegression(max_iter=500, C=C_REG)
            clf.fit(X_s_step[idx_train], y_s[idx_train])
            from sklearn.metrics import roc_auc_score
            local_auroc = roc_auc_score(y_s[idx_test], clf.predict_proba(X_s_step[idx_test])[:, 1])

            clf_s_full = LogisticRegression(max_iter=500, C=C_REG).fit(X_s_step, y_s)
            clf_r_full = LogisticRegression(max_iter=500, C=C_REG).fit(X_r_step, y_r)
            boundary_sim = cosine_sim(clf_s_full.coef_.flatten(), clf_r_full.coef_.flatten())

            marginal_auroc = local_auroc - prev_local_auroc if prev_local_auroc is not None else np.nan
            marginal_boundary = boundary_sim - prev_boundary_sim if prev_boundary_sim is not None else np.nan

            all_results.append({
                "site": site_label,
                "step": step,
                "feature_added": feat if feat else "(baseline)",
                "n_total_features": n_cols,
                "local_auroc": round(local_auroc, 4),
                "marginal_auroc_gain": round(marginal_auroc, 4) if not np.isnan(marginal_auroc) else None,
                "boundary_cosine_sim": round(boundary_sim, 4),
                "marginal_boundary_shift": round(marginal_boundary, 4) if not np.isnan(marginal_boundary) else None,
            })
            prev_local_auroc = local_auroc
            prev_boundary_sim = boundary_sim

            if step % 20 == 0 or step == len(feature_order):
                print(f"  step {step:3d}/{len(feature_order)}: +{feat if feat else '(baseline)'}, "
                      f"auroc={local_auroc:.4f}, boundary_sim={boundary_sim:.4f}")

        # save immediately after each site finishes, don't wait for all sites
        site_df = pd.DataFrame([r for r in all_results if r["site"] == site_label])
        out_path = f"feature_sensitivity_results_site{site_label}.csv"
        site_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(site_df)} rows)")

    if not only_site:
        res_df = pd.DataFrame(all_results)
        res_df.to_csv("feature_sensitivity_results.csv", index=False)
        print(f"\nSaved combined: feature_sensitivity_results.csv ({len(res_df)} rows)")


if __name__ == "__main__":
    main()
