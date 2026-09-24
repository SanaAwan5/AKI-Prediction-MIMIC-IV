#!/usr/bin/env python3
"""
fix_results_naming.py -- make every results directory self-identifying, and
rename only the ones whose names actively mislead.

THE PROBLEM
-----------
Directory names in this project do not indicate their contents:

  results_phase2/           holds the PHASE 1 archetype grid (site_A..site_F)
  results_phase2_approach2/ holds a DIFFERENT archetype cohort (site_A..E)
  results_phase2_auprc/     holds the actual GPC-aligned (Phase 2) runs
  results_phase4_*/         also hold GPC-aligned runs

So `results_phase2` and `results_phase2_auprc` differ by a suffix and hold
different cohorts, whose site sets, prevalences and feature counts differ.
A table built from the wrong one looks entirely normal.

WHAT THIS DOES
--------------
Two independent actions, both off by default:

  --manifest  Write COHORT.txt inside every results directory, recording the
              cohort, sites, methods, conditions and seeds found there. Zero
              risk: it only adds a file. Do this first.

  --rename    Rename ONLY the directories whose name names the wrong cohort.
              Refuses to move a directory whose old name is referenced by any
              .py/.sh/.md file in the tree, and prints those references
              instead. Writes RENAME_LOG.txt with the exact inverse commands.

Default (no flags) is a dry run: it reports what each action would do.

USAGE
    python3 fix_results_naming.py                # dry run, report only
    python3 fix_results_naming.py --manifest     # write COHORT.txt files
    python3 fix_results_naming.py --rename       # perform the renames
"""

import argparse
import glob
import os
import re
import shutil
import sys
from datetime import datetime

import pandas as pd

# Directories whose NAME states the wrong cohort. Only these are candidates;
# a merely uninformative name (results_v2_2_k3_warmup) is left alone, because
# renaming everything would break more than it fixes.
RENAMES = {
    "results_phase2": "results_phase1_grid_early",
    "results_phase2_approach2": "results_phase1_grid_approach2",
}


def survey(root):
    sites, methods, conds, seeds, n = set(), set(), set(), set(), 0
    for f in glob.glob(os.path.join(root, "**", "final_metrics.csv"),
                       recursive=True):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        n += 1
        if "site_id" in d.columns:
            sites |= {re.sub(r"_alpha.*$", "", str(s)) for s in d["site_id"]}
        methods.add(os.path.basename(os.path.dirname(f)))
        m = re.search(r"alpha(\d+(?:\.\d+)?)_gamma(\d+(?:\.\d+)?)", f)
        if m:
            conds.add(f"{m.group(1)}_{m.group(2)}")
        m = re.search(r"[/_]a(\d+(?:\.\d+)?)_g(\d+(?:\.\d+)?)", f)
        if m:
            conds.add(f"{m.group(1)}_{m.group(2)}")
        s = re.search(r"seed(\d+)", f)
        if s:
            seeds.add(s.group(1))
    cohort = ("GPC-aligned (manuscript Phase 2)"
              if any(x.startswith("sim_") for x in sites)
              else "clinical-archetype (manuscript Phase 1)"
              if sites else "unknown (no site_id found)")
    return dict(cohort=cohort, sites=sorted(sites), methods=sorted(methods),
                conds=sorted(conds), seeds=sorted(seeds), files=n)


def references(name, tree="."):
    """Files that mention this directory name, excluding the directory itself."""
    hits = []
    for pat in ("**/*.py", "**/*.sh", "**/*.md", "**/*.tex"):
        for f in glob.glob(os.path.join(tree, pat), recursive=True):
            if os.path.abspath(f).startswith(os.path.abspath(name) + os.sep):
                continue
            try:
                txt = open(f, errors="ignore").read()
            except Exception:
                continue
            for i, line in enumerate(txt.splitlines(), 1):
                if name in line:
                    hits.append((f, i, line.strip()[:100]))
    return hits


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="store_true",
                    help="write COHORT.txt in every results directory")
    ap.add_argument("--rename", action="store_true",
                    help="perform the renames (refuses if names are referenced)")
    args = ap.parse_args()
    dry = not (args.manifest or args.rename)

    roots = sorted(d for d in glob.glob("./results*") if os.path.isdir(d))
    if not roots:
        sys.exit("no ./results* directories here. Run from the project root.")
    print(f"{len(roots)} results directories\n")

    # ------------------------------------------------------------- manifests
    wrote = 0
    for r in roots:
        info = survey(r)
        if info["files"] == 0:
            continue
        text = (f"COHORT: {info['cohort']}\n"
                f"written by fix_results_naming.py on "
                f"{datetime.now():%Y-%m-%d %H:%M}\n\n"
                f"files   : {info['files']} final_metrics.csv\n"
                f"sites   : {', '.join(info['sites'])}\n"
                f"methods : {', '.join(info['methods'])}\n"
                f"conds   : {', '.join(info['conds']) or '(none in path)'}\n"
                f"seeds   : {', '.join(info['seeds']) or '(none in path)'}\n")
        if args.manifest:
            with open(os.path.join(r, "COHORT.txt"), "w") as f:
                f.write(text)
            wrote += 1
        elif dry:
            short = info["cohort"].split(" (")[0]
            print(f"  would write {r}/COHORT.txt  [{short}, "
                  f"{len(info['sites'])} sites, {info['files']} files]")
    if args.manifest:
        print(f"wrote {wrote} COHORT.txt file(s)")

    # --------------------------------------------------------------- renames
    print("\n" + "=" * 74)
    print("RENAMES -- only directories whose name states the wrong cohort")
    print("=" * 74)
    log = []
    for old, new in RENAMES.items():
        if not os.path.isdir(old):
            print(f"  {old}: not present, skipping")
            continue
        info = survey(old)
        print(f"\n  {old}  ->  {new}")
        print(f"     contains: {info['cohort']}")
        print(f"     sites   : {', '.join(info['sites'])}")
        if os.path.exists(new):
            print(f"     REFUSED: {new} already exists")
            continue
        refs = references(old)
        if refs:
            print(f"     REFUSED: {len(refs)} reference(s) to this name -- "
                  f"update them first:")
            for f, i, line in refs[:10]:
                print(f"       {f}:{i}  {line}")
            if len(refs) > 10:
                print(f"       ... and {len(refs)-10} more")
            continue
        if args.rename:
            shutil.move(old, new)
            log.append(f"mv {new} {old}")
            print("     RENAMED")
        else:
            print("     would rename (no references found)")

    if args.rename and log:
        with open("RENAME_LOG.txt", "w") as f:
            f.write(f"# fix_results_naming.py, {datetime.now():%Y-%m-%d %H:%M}\n")
            f.write("# to undo, run these lines:\n")
            f.write("\n".join(log) + "\n")
        print(f"\nwrote RENAME_LOG.txt ({len(log)} inverse command(s))")

    if dry:
        print("\nDry run. Re-run with --manifest to write the COHORT.txt files,")
        print("then --rename to perform the renames.")


if __name__ == "__main__":
    main()
