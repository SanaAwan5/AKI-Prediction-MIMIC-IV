#!/usr/bin/env python3
"""
score_fixed_coefficients.py -- score the already-published, archetype-cohort
prevalence coefficients against NEW data WITHOUT refitting.

WHY THIS IS A SEPARATE SCRIPT FROM flgain_sign_rank.py
--------------------------------------------------------
flgain_sign_rank.py always refits leave-one-site-out on whatever --data you
give it. That is the right test when you want to know whether the published
*relationship* (gain depends on prevalence) re-derives from a new cohort on
its own. It is the wrong test for the question Phase 3 actually asks: does
the *already-fitted* line -- the one reported in the manuscript, estimated
once on the 25-site expanded archetype cohort -- correctly score sites it
was never fitted on, real or simulated. Refitting on 6 real sites (Phase 3)
has no statistical power on its own (see the manuscript's Section 5.3 /
README's "Controlled negative" discussion); scoring the fixed coefficients
does not have that problem, because nothing is estimated from the 6 sites.

This mirrors exactly what the manuscript did for the simulated GPC-aligned
cohort ("Scoring the fitted equation on these six sites without refitting"),
just pointed at real Phase 3 data instead.

USAGE
    python3 score_fixed_coefficients.py --data gains_phase3.csv \
        --target delta_auroc --a -0.0154 --b 0.0460

    # FedAdaptProto, dAUROC (the manuscript's primary arm):
    python3 score_fixed_coefficients.py --data gains_phase3.csv \
        --target delta_auroc --a -0.0154 --b 0.0460

    # FedAdaptProto, dAUPRC:
    python3 score_fixed_coefficients.py --data gains_phase3.csv \
        --target delta_auprc --a -0.0440 --b 0.1330

Published coefficients (25-site expanded archetype cohort, alpha=0.1,
gamma=0.0 -- see Table in sections/joining_site.tex / README "The
replacement"):

    Method          Metric      a         b
    FedAvg          dAUROC   -0.0144   +0.0432
    FedAvg          dAUPRC   -0.0580   +0.1719
    FedAdaptProto   dAUROC   -0.0154   +0.0460
    FedAdaptProto   dAUPRC   -0.0440   +0.1330
    Pooled          dAUROC   -0.0149   +0.0446
    Pooled          dAUPRC   -0.0510   +0.1524

INPUT
-----
A tidy CSV with one row per site (or per site-condition, pre-pooled with
pool_site_conditions.py): columns "site", "prevalence", and the target
column named by --target (e.g. "delta_auroc"). This is the same schema
flgain_sign_rank.py and joining_site_report.py expect -- gains_phase3.csv
from recompute_gains.py, merged with each real site's own prevalence.
"""
import argparse

import numpy as np
import pandas as pd


def balanced_accuracy(truth, pred):
    """Mean of per-class recall -- a constant predictor scores 0.5."""
    recalls = []
    for c in (True, False):
        m = truth == c
        if m.sum():
            recalls.append((pred[m] == c).mean())
    return float(np.mean(recalls)) if recalls else np.nan


def kendall_tau(obs, pred):
    """Concordant-minus-discordant pairs, manual (matches
    flgain_sign_rank.py's rank_report, no scipy dependency)."""
    n = len(obs)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            so = np.sign(obs[i] - obs[j])
            sp = np.sign(pred[i] - pred[j])
            if so == 0 or sp == 0:
                continue
            if so == sp:
                conc += 1
            else:
                disc += 1
    n_pairs = conc + disc
    return (conc - disc) / n_pairs if n_pairs else np.nan, n_pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="tidy CSV: site, prevalence, target")
    ap.add_argument("--target", default="delta_auroc")
    ap.add_argument("--a", type=float, required=True, help="published intercept")
    ap.add_argument("--b", type=float, required=True, help="published slope")
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    if "site" in d.columns and d["site"].duplicated().any():
        # one row per site: average any repeat (condition/seed) rows first
        d = d.groupby("site", as_index=False).agg(
            prevalence=("prevalence", "mean"), **{args.target: (args.target, "mean")}
        )
    d = d.dropna(subset=["prevalence", args.target]).reset_index(drop=True)

    pred = args.a + args.b * d["prevalence"].values
    obs = d[args.target].values

    truth_pos = obs > 0
    pred_pos = pred > 0
    b_acc = balanced_accuracy(truth_pos, pred_pos)
    tau, n_pairs = kendall_tau(obs, pred)
    r = np.corrcoef(d["prevalence"].values, obs)[0, 1] if len(d) > 1 else np.nan

    print("=" * 74)
    print(f"SCORING FIXED COEFFICIENTS (no refit)   target {args.target}")
    print(f"  delta = {args.a:+.4f} {args.b:+.4f} * prevalence")
    print("=" * 74)
    print(f"  {len(d)} sites, {truth_pos.sum()}/{len(d)} actually positive")
    print(f"  sign, balanced accuracy : {b_acc:.3f}   (chance 0.500; NOT estimable")
    print(f"                             reliably below ~2 sites per class)")
    print(f"  ranking, Kendall tau    : {tau:+.3f}   ({n_pairs} pairs)")
    print(f"  r(prevalence, {args.target}) : {r:+.3f}")
    print()
    print(f"  {'site':<15}{'prevalence':>12}{'predicted':>12}{'observed':>12}{'sign ok':>10}")
    for s, p_, pr, ob in zip(d["site"], d["prevalence"], pred, obs):
        ok = "yes" if (pr > 0) == (ob > 0) else "NO"
        print(f"  {s:<15}{p_:>12.4f}{pr:>+12.4f}{ob:>+12.4f}{ok:>10}")
    print()
    print("  Caution: with 6 sites this is a directional check, not a powered")
    print("  validation -- report alongside this caveat, not as a pass/fail.")


if __name__ == "__main__":
    main()
