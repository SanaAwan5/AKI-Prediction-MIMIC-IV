#!/usr/bin/env python3
"""
recompute_gains.py -- recompute every method's FL gain against the ONE shared
matched-effort local baseline, instead of whatever baseline its own run used.

WHY
---
A method's reported gain is  fed_metric - local_baseline.  If two methods are
scored against different baselines their gains are not comparable, and in this
project the baselines differed twice over (generic MLP vs real FedAdaptClient;
cached 20-epoch vs per-run matched-effort). This reads the federated metric
each run already stored and subtracts the shared baseline, so every method is
measured against the same counterfactual. No retraining -- pure arithmetic.

Nothing is dropped quietly:
  * configurations under one root (v2.3_K2, v2.5_bestckpt, the baseline dir)
    are reported as SEPARATE rows, never averaged into one number
  * results files that cannot be used are listed with the reason
  * (condition, site) pairs with no matched baseline are named before dropping
  * a baseline without a val_source column, or with val_source == "internal",
    is called out: the federated side selects on the training script's own
    validation split, so the baseline must use that same split

USAGE
-----
    python3 recompute_gains.py ./results_phase2_auprc_valsel \\
            --data-dir ./phase2_data_disjoint \\
            --csv-out gains_matched_p2_valsel.csv
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


def cond_of(path):
    """(alpha, gamma) as floats-as-strings, from a path or filename."""
    m = re.search(r"(?:alpha|a)(\d+(?:\.\d+)?)[_/]*(?:gamma|g)(\d+(?:\.\d+)?)",
                  path.lower())
    if not m:
        return None
    return f"{float(m.group(1))}_{float(m.group(2))}"


def base_site(sid):
    return re.sub(r"_alpha.*$", "", str(sid))


def load_baseline(data_dir):
    rows, warn = [], []
    # [SUFFIX GUARD] The glob must not pick up variant baselines. A
    # directory accumulates local_baseline_matched_*_tf0.25.csv (train-fraction
    # sweep), *_ve3.csv (coarsened validation), *_mlp.csv (architecture arm)
    # and friends. They all match alpha*_gamma*.csv and all parse to the SAME
    # condition, so each result row then merges against several baseline rows
    # and the output is silently multiplied: 324 observations became 1944 with
    # six variants present. Keep only the canonical unsuffixed file unless the
    # caller asks for a specific variant.
    files = sorted(
        f for f in glob.glob(os.path.join(
            data_dir, "local_baseline_matched_alpha*_gamma*.csv"))
        if re.fullmatch(r"local_baseline_matched_alpha[\d.]+_gamma[\d.]+\.csv",
                        os.path.basename(f)))
    skipped_variants = sorted(
        os.path.basename(f) for f in glob.glob(os.path.join(
            data_dir, "local_baseline_matched_alpha*_gamma*.csv"))
        if not re.fullmatch(r"local_baseline_matched_alpha[\d.]+_gamma[\d.]+\.csv",
                            os.path.basename(f)))
    if not files:
        sys.exit(f"no local_baseline_matched_*.csv in {data_dir}\n"
                 f"Run:  bash run_valselect_rerun.sh baseline")
    if skipped_variants:
        warn.append(f"ignored {len(skipped_variants)} variant baseline file(s) "
                    f"(suffixed: {', '.join(sorted({re.sub(r'.*_gamma[0-9.]+', '', v) for v in skipped_variants}))}) "
                    f"-- only the unsuffixed baseline is used")
    for fp in files:
        c = cond_of(os.path.basename(fp))
        if c is None:
            continue
        df = pd.read_csv(fp)
        if "val_source" not in df.columns:
            warn.append(f"{os.path.basename(fp)}: no val_source column -- this "
                        f"baseline predates the val-selection patch")
        else:
            bad = sorted(set(df["val_source"].astype(str)) - {"script"})
            if bad:
                warn.append(f"{os.path.basename(fp)}: val_source={bad} -- the "
                            f"baseline chose its epoch on a DIFFERENT validation "
                            f"split than the federated runs used")
        df["cond"] = c
        df["site"] = df["site_id"].map(base_site)
        rows.append(df[["site", "cond", "local_auroc", "local_auprc",
                        "prevalence", "best_epoch_mean"]])
    return pd.concat(rows, ignore_index=True), warn


def shape_of(rel):
    """Directory shape with alpha/gamma/seed normalised away."""
    d = os.path.dirname(rel) or "."
    d = re.sub(r"(alpha|a)\d+\.?\d*", "A", d)
    d = re.sub(r"(gamma|g)\d+\.?\d*", "G", d)
    d = re.sub(r"seed\d+", "S", d)
    return d


def load_results(results_dir):
    """
    Federated metrics come from final_metrics.csv, NOT fl_gain_correlation.csv.

    The two script families disagree about the latter's schema: v2.3 writes
    local_auroc / fed_auroc / improvement (+ the AUPRC trio), while v2.5 writes
    local_auroc / delta_auroc / selected_k and no AUPRC at all. Keying on
    fed_auroc silently drops every v2.5 job -- which is exactly the failure
    that once removed v2.5 from a p-value table without a word. final_metrics.csv
    carries site_id / auroc / auprc in both families, so it is the one uniform
    source. fl_gain_correlation.csv is read only for the run's OWN gain, as a
    reference column, and its absence costs nothing.
    """
    rows, shapes, skipped = [], set(), []
    # [ROOT COND] The condition may live in the ROOT directory's name rather
    # than in the path below it -- e.g. results_x25_a10.0_g0.0_fedavg/seed44/
    # has no alpha/gamma once made relative. Fall back to the root so a run
    # organised by seed instead of by condition is not silently discarded.
    root_cond = cond_of(os.path.abspath(results_dir))
    for dirpath, _, files in os.walk(results_dir):
        if "final_metrics.csv" not in files:
            continue
        fp = os.path.join(dirpath, "final_metrics.csv")
        rel = os.path.relpath(fp, results_dir)
        c = cond_of(rel) or root_cond
        if c is None:
            skipped.append((rel, "no alpha/gamma in path or in the root dir name"))
            continue
        seed = re.search(r"seed(\d+)", rel)
        method = os.path.basename(dirpath)
        # The JOB directory (parent of the method dir) with alpha/gamma/seed
        # normalised away identifies the configuration: v2.3_K2_lepoch3,
        # v2.5_bestckpt_fix and the plain baseline dir are three deliberate
        # configurations of one grid, not three run generations. They are
        # reported separately rather than averaged together.
        config = (shape_of(os.path.dirname(rel)) or ".").split(os.sep)[-1]
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            skipped.append((rel, f"unreadable ({e.__class__.__name__})"))
            continue
        if not {"site_id", "auroc"} <= set(df.columns):
            skipped.append((rel, f"columns {list(df.columns)[:5]}"))
            continue
        shapes.add(config)

        # the run's own reported gain, purely for the reference column
        own = {}
        corr = os.path.join(dirpath, "fl_gain_correlation.csv")
        if os.path.exists(corr):
            try:
                cdf = pd.read_csv(corr)
                gcol = next((g for g in ("improvement", "delta_auroc",
                                         "improvement_auroc") if g in cdf.columns), None)
                if gcol and "site_id" in cdf.columns:
                    own = {base_site(s): float(v)
                           for s, v in zip(cdf["site_id"], cdf[gcol])}
            except Exception:
                pass

        for _, r in df.iterrows():
            site = base_site(r["site_id"])
            rows.append({
                "cond": c,
                "seed": seed.group(1) if seed else "",
                "config": config,
                "method": method,
                "job": os.path.dirname(rel),
                "site": site,
                "fed_auroc": float(r["auroc"]),
                "fed_auprc": (float(r["auprc"]) if "auprc" in df.columns
                              and pd.notna(r.get("auprc")) else np.nan),
                "own_gain": own.get(site, np.nan),
            })
    return pd.DataFrame(rows), sorted(shapes), skipped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--data-dir", required=True,
                    help="directory holding local_baseline_matched_*.csv")
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    base, warn = load_baseline(args.data_dir)
    res, shapes, skipped = load_results(args.results_dir)
    if res.empty:
        msg = [f"no usable final_metrics.csv under {args.results_dir}"]
        if skipped:
            msg.append(f"{len(skipped)} file(s) were found but skipped:")
            for rel, why in skipped[:10]:
                msg.append(f"   {rel}  ({why})")
            if len(skipped) > 10:
                msg.append(f"   ... and {len(skipped)-10} more")
        else:
            msg.append("No final_metrics.csv was found at all -- check the path.")
        sys.exit("\n".join(msg))

    print(f"results  : {len(res)} site-observations, {res['job'].nunique()} jobs, "
          f"{res['method'].nunique()} methods, {res['cond'].nunique()} conditions")
    print(f"baseline : {len(base)} site-conditions from {args.data_dir}")
    if skipped:
        print(f"\n  {len(skipped)} results file(s) SKIPPED -- listed rather than "
              f"dropped quietly:")
        for rel, why in skipped[:12]:
            print(f"     {rel}  ({why})")
        if len(skipped) > 12:
            print(f"     ... and {len(skipped) - 12} more")
    if warn:
        print("\n  BASELINE WARNINGS")
        for w in warn:
            print("   ! " + w)

    if len(shapes) > 1:
        print(f"\n  {len(shapes)} configurations under this root -- reported "
              f"SEPARATELY below, never averaged together:")
        for s in shapes:
            print(f"     {s}")

    m = res.merge(base, on=["site", "cond"], how="left")
    missing = m[m["local_auroc"].isna()]
    if len(missing):
        pairs = sorted(set(zip(missing["cond"], missing["site"])))
        print(f"\n  {len(pairs)} (condition, site) pair(s) have no matched "
              f"baseline and are dropped:")
        for c, s in pairs[:10]:
            print(f"     alpha_gamma={c}  site={s}")
        if len(pairs) > 10:
            print(f"     ... and {len(pairs) - 10} more")
        m = m[m["local_auroc"].notna()].copy()

    m["delta_auroc"] = m["fed_auroc"] - m["local_auroc"]
    m["delta_auprc"] = m["fed_auprc"] - m["local_auprc"]

    print("\n" + "=" * 84)
    print("GAIN vs SHARED MATCHED BASELINE")
    print("=" * 84)
    print(f"{'configuration / method':<44}{'n':>5}{'dAUROC':>10}{'dAUPRC':>10}"
          f"{'own d':>9}{'% >0':>7}")
    print("-" * 84)
    for (cfg, meth), s in m.groupby(["config", "method"]):
        own = s["own_gain"].mean()
        new = s["delta_auroc"].mean()
        label = f"{cfg} / {meth}" if cfg not in (".", "") else meth
        print(f"{label[:43]:<44}{len(s):>5}{new:>+10.4f}"
              f"{s['delta_auprc'].mean():>+10.4f}"
              f"{(f'{own:+.4f}' if pd.notna(own) else 'n/a'):>9}"
              f"{(s['delta_auroc'] > 0).mean()*100:>6.0f}%")
    print("-" * 84)
    print(f"{'ALL':<22}{len(m):>5}{m['delta_auroc'].mean():>+10.4f}"
          f"{m['delta_auprc'].mean():>+10.4f}")

    print("\nPer condition (AUROC):")
    for c, s in m.groupby("cond"):
        print(f"  alpha_gamma={c:<10}  n={len(s):<4}  "
              f"mean dAUROC={s['delta_auroc'].mean():+.4f}  "
              f"sites>0={100*(s['delta_auroc']>0).mean():.0f}%")

    print(f"\nLocal baseline peaked at epoch "
          f"{base['best_epoch_mean'].mean():.1f} on average -- if this is tiny "
          f"next to\nthe matched budget, the local ceiling is data-limited "
          f"rather than effort-limited.")

    if args.csv_out:
        m.to_csv(args.csv_out, index=False)
        print(f"\nwrote {args.csv_out}  ({len(m)} rows)")


if __name__ == "__main__":
    main()
