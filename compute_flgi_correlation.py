#!/usr/bin/env python3
"""
compute_flgi_correlation.py

Standalone, reusable computation of the FL Gain Index (FLGI) and its
correlation with observed federated gain, for any set of sites.

This reproduces the formula embedded in the v2.5 training scripts
(compute_fl_gain() / compute_fl_gain_revised()) as a standalone tool, so
the manuscript's FL Gain Index tables/correlations can be regenerated
directly from site characteristics without needing to re-run training.

FORMULA (Eq. 1, pre-training FL Gain Index):
    FLGI_i = 0.30*S_i + 0.30*I_i + 0.40*F_i
where, across the sites being compared:
    S_i = positive-case scarcity = (max(n_pos) - n_pos_i) / (max(n_pos) - min(n_pos))
    I_i = class imbalance        = 1 - prevalence_i   (since AKI is always the minority class)
    F_i = feature sparsity       = (max(n_feat) - n_feat_i) / (max(n_feat) - min(n_feat))
Role thresholds: FLGI >= 0.65 -> primary benefitter
                 FLGI >= 0.52 -> conditional benefitter
                 else          -> contributor

Usage:
    python3 compute_flgi_correlation.py sites.csv

Input CSV must have columns: site_id, n, n_features, prevalence, observed_delta_auroc
    site_id                - name of the site
    n                       - number of patients at the site
    n_features              - number of features available at the site
    prevalence              - AKI prevalence (fraction, e.g. 0.144 for 14.4%)
    observed_delta_auroc    - confirmed observed ΔAUROC from actual federated training

Output: per-site FLGI decomposition + role, and Pearson/Spearman correlation
against observed_delta_auroc, printed to stdout. Pass --csv-out <path> to
additionally write the per-site decomposition table to a CSV file.
"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_THRESHOLD = 0.65
CONDITIONAL_THRESHOLD = 0.52


def classify_role(flgi):
    if flgi >= PRIMARY_THRESHOLD:
        return "primary"
    elif flgi >= CONDITIONAL_THRESHOLD:
        return "conditional"
    else:
        return "contributor"


def compute_flgi(df):
    """
    df must have columns: site_id, n, n_features, prevalence, observed_delta_auroc
    Returns a new DataFrame with S, I, F, FLGI, role columns added, sorted by
    FLGI descending.
    """
    required = {"site_id", "n", "n_features", "prevalence", "observed_delta_auroc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {sorted(missing)}")

    n_pos = df["n"].values * df["prevalence"].values
    n_feat = df["n_features"].values
    prev = df["prevalence"].values

    if n_pos.max() == n_pos.min():
        print("  [warning] all sites have identical positive-case counts; "
              "S_i is undefined (0/0) and set to 0.5 for all sites.", file=sys.stderr)
        S = np.full_like(n_pos, 0.5, dtype=float)
    else:
        S = (n_pos.max() - n_pos) / (n_pos.max() - n_pos.min())

    I = 1 - prev

    if n_feat.max() == n_feat.min():
        print("  [warning] all sites have identical feature counts; "
              "F_i is undefined (0/0) and set to 0.5 for all sites.", file=sys.stderr)
        F = np.full_like(n_feat, 0.5, dtype=float)
    else:
        F = (n_feat.max() - n_feat) / (n_feat.max() - n_feat.min())

    FLGI = 0.30 * S + 0.30 * I + 0.40 * F

    out = df.copy()
    out["S_i"] = S
    out["I_i"] = I
    out["F_i"] = F
    out["FLGI"] = FLGI
    out["role"] = [classify_role(v) for v in FLGI]
    out = out.sort_values("FLGI", ascending=False).reset_index(drop=True)
    return out


def compute_correlation(df):
    """Returns (pearson_r, pearson_p, spearman_rho, spearman_p)."""
    n = len(df)
    if n < 3:
        print(f"  [warning] only {n} sites -- correlation is not meaningful "
              f"below n=3; scipy will likely return NaN.", file=sys.stderr)
    r, p_r = stats.pearsonr(df["FLGI"], df["observed_delta_auroc"])
    rho, p_rho = stats.spearmanr(df["FLGI"], df["observed_delta_auroc"])
    return r, p_r, rho, p_rho


def significance_threshold(n, alpha=0.05):
    """The |r| needed to reach the given two-tailed significance level at
    this sample size -- included so small-n results can be read in context
    rather than compared against significance in the abstract."""
    if n < 3:
        return float("nan")
    df = n - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return t_crit / np.sqrt(df + t_crit**2)


def main():
    parser = argparse.ArgumentParser(
        description="Compute the FL Gain Index and its correlation with observed federated gain."
    )
    parser.add_argument("input_csv", help="CSV with columns: site_id, n, n_features, prevalence, observed_delta_auroc")
    parser.add_argument("--csv-out", default=None, help="Optional path to write the per-site decomposition table to")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    result = compute_flgi(df)

    print("Per-site FL Gain Index decomposition:")
    print(result[["site_id", "S_i", "I_i", "F_i", "FLGI", "role", "observed_delta_auroc"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    r, p_r, rho, p_rho = compute_correlation(result)
    n = len(result)
    r_needed = significance_threshold(n)

    print()
    print(f"n = {n} sites")
    print(f"Pearson  r   = {r:.3f}  (p = {p_r:.3f})")
    print(f"Spearman rho = {rho:.3f}  (p = {p_rho:.3f})")
    if not np.isnan(r_needed):
        print(f"(|r| >= {r_needed:.3f} would be needed to reach p<0.05 at n={n})")

    if args.csv_out:
        result.to_csv(args.csv_out, index=False)
        print(f"\nWrote per-site decomposition to {args.csv_out}")


if __name__ == "__main__":
    main()
