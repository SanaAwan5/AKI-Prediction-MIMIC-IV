# Bug-fix log — AKI FTL pipeline

Moved out of `run_complete.md` so that file stays a clean run/reproduce
log rather than a bug-hunting narrative. This document is the historical
record of real code defects found and fixed during development — distinct
from the matched-baseline **protocol** (a deliberate methodology choice,
documented in `run_complete.md` Section 14.8 and `README.md`, not a bug).

Each entry: what broke, what it affected, how it was caught, and the fix.

---

## 1. KDIGO baseline-SCr / CKD-exclusion fix (both cohorts)

Both cohort-generation notebooks previously applied MDRD-estimated
baseline SCr to all patients regardless of CKD history. Fixed to the
standard 3-tier KDIGO hierarchy: 7-day-prior most recent → 7–365-day-prior
mean → CKD history with no SCr in the past year drops the encounter
(rather than MDRD-estimating it), non-CKD gets MDRD-estimated. This took
the cohort from 163,038 to 114,720 patients. Confirmed both notebooks use
identical exclusion criteria and produce the identical underlying patient
population and train/test split (`record_train_test_numbers.py`).

## 2. Cross-site patient overlap fix (both cohorts)

Sites previously could and did share ~35% of their patients with each
other. Fixed via disjoint-sites simulation scripts
(`phase1_archetype_simulation.py`, `phase2_gpc_aligned_simulation.py`).
Confirmed zero overlap across all site pairs, all conditions, both
cohorts, via `check_overlap.py`.

## 3. v2.5 root-cause diagnosis and fix (both cohorts)

The original v2.5 scripts (`fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py`
for GPC-aligned, and an archetype counterpart not kept in this repository)
reported strongly negative results on both cohorts (GPC-aligned: −0.0267;
archetype: −0.0945). Both were traced to the same two causes:

1. **Missing best-checkpoint restoration in Phase 1 (warmup).** v2.5 runs
   a two-phase procedure: Phase 1 trains at a uniform placeholder
   `K=k_min` to generate embeddings for silhouette-based K selection,
   then Phase 2 resets the head and retrains at the selected per-site K.
   Phase 2 already restores each site to its own best-checkpoint round
   (same convention as v2.3); Phase 1 did not, so the state handed to
   Phase 2 was Phase 1's *final* round — already well past its own peak
   and into an overfit decline, not Phase 1's best round.
2. **An uncontrolled `local_epochs` mismatch.** v2.5's script default is
   `5`; v2.3 uses `1`. Comparisons that did not explicitly pass
   `--local_epochs 1` were not comparing like with like.

Fixed scripts: `phase2_gpc_aligned_train_v25.py` (GPC-aligned) and
`phase1_archetype_train_v25.py` (archetype). Both add Phase 1
best-checkpoint tracking, identical in structure to Phase 2's own;
behavior is unchanged from the original script when `--local_epochs` is
left unset, so `--local_epochs 1` must still be passed explicitly for a
fair v2.3 comparison.

**Confirmed result after the fix, both cohorts:**
- GPC-aligned: −0.0267 → **+0.0052 ± 0.0085** (n=54, full 9-run grid)
- Archetype, primary condition: −0.0945 → **−0.0044 ± 0.0104** (n=15)
- Archetype, full 20-condition grid: **−0.0023 ± 0.0096** (n=300)

All three are now statistically comparable to v2.3 and the simple
baselines on their respective cohorts.

An intermediate diagnostic script (adding a `--head_fix_rounds` flag to
freeze the body during an initial head-only warmup at the start of
Phase 2) tested a *different* hypothesis ("gradient shock" from the
reset) and found no improvement. Negative result; the script is not part
of the confirmed pipeline and is not kept in this repository — use the
`_bestckpt_fix` scripts, not `_head_fix`.

## 4. Phase 1 v2.5 site-discovery and baseline-cache bugs

Found during the Phase 1 v2.5 re-run, distinct from bug 3 above (bug 3 is
the checkpoint/local_epochs issue; this is two separate defects found
later, in the same script family):

1. **No alpha/gamma filtering in site discovery.** Every job silently
   trained a ~100-site mega-federation (5 real sites × 20 conditions)
   instead of the intended 5 sites for its specific condition.
2. **Condition-independent baseline cache path.** The local-baseline
   cache lived at a single path regardless of condition, despite the
   docstring claiming otherwise. Even after fix (1), every job after the
   first loaded a stale, cross-condition-polluted cached baseline.

Both fixed in the delivered `_bestckpt_fix` scripts (both Phase 1 and
Phase 2 versions) and confirmed via direct inspection of real output on
both cohorts.

**Verification coverage — confirmed vs. not yet confirmed.** The
underlying text only documents this pair of bug classes as explicitly
checked in one other script: the Phase 2 v2.3 script (`grep`/direct
inspection), confirmed **absent** there — no change needed. **Phase 1
v2.3 was not documented as separately checked for this pair of bugs.**
Given v2.3's architecture doesn't share v2.5's two-phase warmup-then-K-select
structure (so bug (1)'s class doesn't obviously apply) and v2.3 has no
baseline-cache mechanism of any kind per Section 0's prerequisite notes
(so bug (2)'s class doesn't apply either), there's a structural reason to
expect it's clean — but that's an inference, not a direct check. If this
matters for a claim in the manuscript, run the equivalent inspection
(`grep` for the site-discovery filter logic and for any cache path
construction) against `phase1_archetype_train_v23.py` before citing it as
confirmed clean.

## 5. Interrupted Phase 1 v2.3+baselines grid — stale mixed-provenance data

Surfaced during Phase 1's v2.3+baselines grid, unrelated to bug 4. An
interruption from an external cause (no traceback — likely system sleep
or an OOM kill) stopped the grid mid-run, and restarting it with a
non-resumable script version caused 3 of 5 methods
(scaffold/fedadapt/fedprox) to retain stale, mixed-provenance data from
an earlier, separate attempt rather than being freshly recomputed. Caught
by checking per-method timestamps (not just job counts) after the
"completed" grid still looked suspicious. Fixed by adding a resume-skip
check to all three grid shell scripts (skips a job if its output file
already exists, so an interrupted run can resume safely instead of
restarting from the top), then re-running the full 300-job grid once,
cleanly, in a single pass. Confirmed the staleness was substantively
real, not just cosmetic — the discarded numbers for scaffold/fedadapt
were both meaningfully lower than the clean re-run's.

## 6. AUPRC + v2.5 local-baseline architecture confound (pending, not yet re-run)

Found while adding an AUPRC-based FL-Gain experiment (see `run_complete.md`
Section 13): the pooled n=390 fit behind the withdrawn FL Gain Index's
post-local-baseline equation mixed two different definitions of the
local-ceiling term. v2.3's local-only baseline (`run_local_only()`)
trains the real `FedAdaptClient` locally with no federation rounds — a
matched counterfactual. v2.5's local-only baseline
(`compute_or_load_shared_local_baseline()`) instead trained a *separate,
generic 3-layer MLP*, on both cohorts. Since the n=390 fit pools v2.3
(n=336) and v2.5 (n=54) rows together and treats `C_i^raw` as one
consistent quantity, this was a real architecture confound sitting inside
the reported regression, not just a modeling nuance.

**Status: fixed in code, not yet re-run.** v2.5's local-only baseline now
trains the real `FedAdaptClient` as well, and the fix additionally
computes local AUPRC (previously discarded by v2.3, never computed by
v2.5's MLP baseline). This item is moot for the current manuscript
headline (the FL Gain Index it fed into is withdrawn — see
`run_complete.md` Section 12), kept here only in case that equation is
ever revived.

---

## Bugs vs. protocol — how to tell them apart

A **bug** here means: the code did not do what its own docstring or
design intent claimed, and produced numbers state as reproducible until
someone happened to notice. All six items above are that.

The **matched-baseline protocol** (`run_complete.md` §14.8, `README.md`
§Results) is not that: `recompute_gains.py` and the conventional-protocol
tables both do exactly what their own logic says they do. They differ
because they answer different questions — one baseline per method's own
incidentally-selected checkpoint vs. one shared baseline given the same
checkpoint-selection privilege as the federated run. Neither is broken;
they're two measurements, and the paper's methodology commits to the
second one. Don't file that distinction here.
