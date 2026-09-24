#!/usr/bin/env python3
"""
matched_method_pvalues.py -- paired significance tests between FL strategies,
computed against the ONE shared matched-effort local baseline.

WHY NOT compute_method_pvalues.py
---------------------------------
That script reads each run's own fl_gain_correlation.csv, i.e. each method's
gain against whatever baseline its own job happened to use. Under the matched
protocol every method must be scored against the same counterfactual, so the
input here is the output of recompute_gains.py instead.

The difference is not cosmetic: on the GPC cohort the matched baseline moves
every method's mean gain by about +0.07 AUROC and flips its sign. A p-value
computed from the old source is testing a different quantity.

WHAT IS PAIRED WITH WHAT
------------------------
Two granularities are reported, because they are different tests and can
disagree:

  job-level   n = conditions x seeds (9 for Phase 2). One mean per job,
              pooled across that job's sites, then paired across methods.
              Sites within a job share the same federated model, so they are
              not independent; pooling first is the defensible choice and is
              what the manuscript reports.

  site-level  n = conditions x seeds x sites (54 for Phase 2). More apparent
              power, weaker independence assumption. Reported for
              transparency, not as the headline.

Wilcoxon signed-rank is reported beside the paired t-test because at n=9 the
normality assumption is not checkable.

USAGE
    python3 matched_method_pvalues.py gains_matched_p2.csv
    python3 matched_method_pvalues.py gains_matched_p2.csv --metric delta_auprc
    python3 matched_method_pvalues.py gains_matched_p2.csv --reference fedadaptproto \\
        --csv-out pvalues_matched_p2.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:
    sys.exit("scipy is required:  pip install scipy --break-system-packages")


def paired(d, ref, metric, level):
    """Return one row per non-reference method."""
    keys = ["cond", "seed"] + (["site"] if level == "site" else [])
    g = d.groupby(["method"] + keys)[metric].mean().reset_index()
    wide = g.pivot_table(index=keys, columns="method", values=metric)
    if ref not in wide.columns:
        sys.exit(f"reference method {ref!r} not in {list(wide.columns)}")
    rows = []
    for m in sorted(c for c in wide.columns if c != ref):
        pair = wide[[ref, m]].dropna()
        if len(pair) < 3:
            rows.append(dict(method=m, n=len(pair), diff=np.nan,
                             t_p=np.nan, w_p=np.nan, note="too few pairs"))
            continue
        a, b = pair[ref].values, pair[m].values
        t_p = ttest_rel(a, b).pvalue
        try:
            w_p = wilcoxon(a, b).pvalue
        except ValueError:
            w_p = np.nan                     # all differences zero
        rows.append(dict(method=m, n=len(pair), ref_mean=a.mean(),
                         method_mean=b.mean(), diff=(a - b).mean(),
                         t_p=t_p, w_p=w_p, note=""))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gains", help="gains_matched_*.csv from recompute_gains.py")
    ap.add_argument("--reference", default="fedadaptproto")
    ap.add_argument("--config", default=None,
                    help="comma-separated configuration name(s) to keep (the "
                         "'config' column from recompute_gains.py). Several may "
                         "be pooled ONLY if no method appears in more than one "
                         "of them -- that is the condition under which pooling "
                         "compares methods rather than versions of one method.")
    ap.add_argument("--metric", default="delta_auroc",
                    choices=["delta_auroc", "delta_auprc"])
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    d = pd.read_csv(args.gains)
    need = {"method", "cond", "seed", "site", args.metric}
    missing = need - set(d.columns)
    if missing:
        sys.exit(f"{args.gains} is missing {sorted(missing)}.\n"
                 f"Expected the output of recompute_gains.py.")
    d = d.dropna(subset=[args.metric])

    # [CONFIG GUARD] recompute_gains.py reports configurations separately and
    # refuses to average them; this script must not undo that. A method present
    # in two configurations (e.g. fedadaptproto under both v2.3_K2 and
    # v2.5_bestckpt_fix) would otherwise be pooled into a single mean that
    # belongs to neither, and compared against single-configuration rivals.
    if "config" in d.columns:
        cfgs = sorted(d["config"].astype(str).unique())
        if args.config:
            want = [c.strip() for c in args.config.split(",") if c.strip()]
            bad = [c for c in want if c not in cfgs]
            if bad:
                sys.exit(f"--config value(s) not found: {bad}\nAvailable: {cfgs}")
            d = d[d["config"].astype(str).isin(want)]
            # A method present in two of the selected configurations would be
            # averaged across them, producing a mean belonging to neither. That
            # is the failure this guard exists to prevent; selecting several
            # configurations is safe only when each method sits in exactly one.
            dup = (d.groupby("method")["config"].nunique()
                     .loc[lambda x: x > 1])
            if len(dup):
                detail = (d[d["method"].isin(dup.index)]
                          .groupby(["method", "config"])[args.metric]
                          .agg(["size", "mean"]))
                sys.exit("these method(s) appear in more than one selected "
                         f"configuration:\n\n{detail}\n\nPooling them would "
                         "average a method across configurations. Narrow "
                         "--config\nso that each method comes from exactly one.")
            if len(want) > 1:
                src = d.groupby("method")["config"].first()
                print("Pooling configurations -- each method comes from exactly "
                      "one:\n")
                for m, c in src.items():
                    print(f"  {m:<18} <- {c}")
                print()
        elif len(cfgs) > 1:
            per = (d.groupby(["config", "method"])[args.metric]
                     .agg(["size", "mean"]).reset_index())
            print("This file holds several configurations:\n")
            for c in cfgs:
                sub = per[per["config"] == c]
                print(f"  {c}")
                for _, r in sub.iterrows():
                    print(f"      {r['method']:<16}{int(r['size']):>6}"
                          f"{r['mean']:>+10.4f}")
            sys.exit("\nPass --config <name> to select one. Averaging them would "
                     "pool a method's\ndistinct configurations into a mean that "
                     "belongs to neither.")

    # ------------------------------------------------------------- inventory
    print("=" * 78)
    print("INVENTORY  -- read this before the p-values")
    print("=" * 78)
    print(f"  file      {args.gains}")
    print(f"  metric    {args.metric}")
    print(f"  {len(d)} observations, {d['method'].nunique()} methods, "
          f"{d['cond'].nunique()} conditions, {d['seed'].nunique()} seeds, "
          f"{d['site'].nunique()} sites")
    cnt = d.groupby("method").agg(n=(args.metric, "size"),
                                  jobs=("seed", lambda s: len(set(zip(
                                      d.loc[s.index, "cond"], s)))),
                                  mean=(args.metric, "mean"))
    print(f"\n  {'method':<18}{'n':>6}{'jobs':>7}{'mean':>10}")
    for m, r in cnt.iterrows():
        print(f"  {m:<18}{int(r['n']):>6}{int(r['jobs']):>7}{r['mean']:>+10.4f}")
    if cnt["n"].nunique() > 1:
        print("\n  WARNING: methods have unequal observation counts. Pairing "
              "drops unmatched\n  cells silently -- check the n column in each "
              "test below.")

    # ---------------------------------------------------------------- tests
    csv_rows = []
    for level, label in (("job", "JOB-LEVEL  (one mean per condition-seed, "
                                 "pooled across sites)"),
                         ("site", "SITE-LEVEL  (each site-condition-seed paired "
                                  "separately)")):
        r = paired(d, args.reference, args.metric, level)
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)
        print(f"  reference: {args.reference}")
        print(f"  {'vs method':<18}{'n':>5}{'ref':>10}{'other':>10}"
              f"{'diff':>10}{'t-test p':>11}{'wilcoxon p':>12}")
        for _, x in r.iterrows():
            if x["note"]:
                print(f"  {x['method']:<18}{x['n']:>5}  {x['note']}")
                continue
            print(f"  {x['method']:<18}{int(x['n']):>5}{x['ref_mean']:>+10.4f}"
                  f"{x['method_mean']:>+10.4f}{x['diff']:>+10.4f}"
                  f"{x['t_p']:>11.4f}{x['w_p']:>12.4f}")
            csv_rows.append(dict(level=level, metric=args.metric,
                                 reference=args.reference, **x))

    print("\n" + "=" * 78)
    print("READING THESE")
    print("=" * 78)
    print("  A significant difference between two methods whose means are both")
    print("  negative says one loses less than the other -- not that either")
    print("  improves on local-only training. Report the means beside every")
    print("  p-value so that distinction survives into the text.")
    print("  At n=9 the t-test and Wilcoxon can disagree; where they do, prefer")
    print("  Wilcoxon and say so.")

    if args.csv_out and csv_rows:
        pd.DataFrame(csv_rows).to_csv(args.csv_out, index=False)
        print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
