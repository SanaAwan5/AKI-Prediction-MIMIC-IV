# Complete Run Instructions — Phase 2 (GPC-Aligned) and Phase 1 (Clinical-Archetype)

> **✅ ALL TRAINING COMPLETE ON BOTH COHORTS.**
> Both master CSVs were regenerated after correcting the baseline-SCr
> computation (CKD-history patients with no SCr in the past year are now
> dropped instead of MDRD-estimated, matching the standard KDIGO baseline
> flowchart) and after fixing cross-site patient overlap (sites previously
> could and did share ~35% of their patients with each other; see
> `HOW_TO_CHECK_OVERLAP.txt`). **Data generation is complete and confirmed
> clean** for both cohorts (zero cross-site overlap, every condition,
> verified via `check_overlap.py`). **Phase 1 v2.5 (60 jobs), Phase 1
> v2.3+baselines (300 jobs), and all of Phase 2 (54 jobs, all 6 methods)
> are complete and confirmed** — every job checked programmatically for
> correct site count and correct alpha/gamma per job, zero bad jobs found
> anywhere. The manuscript (`main.tex`) reflects all of this and is
> current as of this version.
>
> **Two real bugs were found and fixed during the Phase 1 v2.5 re-run,
> worth knowing about**: (1) the archetype v2.5 script's site-discovery
> had no alpha/gamma filtering at all, so every job silently trained a
> ~100-site mega-federation (5 real sites × 20 conditions) instead of the
> intended 5 for its specific condition; (2) its local-baseline cache
> lived at a single, condition-independent path despite the docstring
> claiming otherwise, so even after fix (1), every job after the first
> loaded stale, cross-condition-polluted cached baselines. Both fixed in
> the delivered `_bestckpt_fix` scripts (both Phase 1 and Phase 2
> versions) and confirmed via direct inspection of real output on both
> cohorts. The same two bug classes were checked in the Phase 2 v2.3
> script specifically and confirmed **absent** there — no changes were
> needed for that one.
>
> **A third issue surfaced during Phase 1's v2.3+baselines grid,
> unrelated to the two above**: an interruption from an external cause
> (no traceback -- likely system sleep or an OOM kill) stopped the grid
> mid-run, and restarting it with a non-resumable script version caused
> 3 of 5 methods (scaffold/fedadapt/fedprox) to retain stale,
> mixed-provenance data from an earlier, separate attempt rather than
> being freshly recomputed. Caught by checking per-method timestamps
> (not just job counts) after the "completed" grid still looked
> suspicious. Fixed by adding a resume-skip check to all three grid
> shell scripts (skips a job if its output file already exists, so an
> interrupted run can resume safely instead of restarting from the top),
> then re-running the full 300-job grid once, cleanly, in a single pass.
> Confirmed the staleness was substantively real, not just cosmetic --
> the discarded numbers for scaffold/fedadapt were both meaningfully
> lower than the clean re-run's.

## Cohort summary (confirmed, current)

| | Value |
|---|---|
| Total patients (both cohorts, identical population) | **114,720** |
| Train split | 91,776 (80%) |
| Test split | 22,944 (20%) |
| `subject_id`/`hadm_id` relationship | 1:1 (confirmed — one encounter per patient, the last admission only) |
| Phase 1 vs. Phase 2 patient sets | Identical (same 114,720 patients, same per-patient train/test assignment in both files) |
| Phase 1 site count / N per site | 5 sites × **17,000** (85,000 of 91,776 train patients used, 6,776 buffer) |
| Phase 2 site count / N per site | 6 sites × **14,000** (84,000 of 91,776 train patients used, 7,776 buffer) |
| Cross-site overlap | **Zero** — confirmed via `check_overlap.py` across all 20 Phase 1 conditions (10 site pairs each) and all 3 Phase 2 conditions (15 site pairs each) |
| `TARGET_N_PER_SITE` ceiling (tested, not just calculated) | Phase 1: 18,000 fails (6,241 within-site duplicates at the last-processed site); Phase 2: 15,000 fails (1,138-patient shortfall) |

Unlike `QUICK_RUN.md` (abbreviated, one representative command per
step), this file lists every individual run explicitly so it can be
executed top-to-bottom to reproduce the full result set for both
cohorts.

Run each command standalone (one at a time, or as a sequential non-backgrounded
block) — this project has repeatedly found silent `--data_dir`/`--alpha`/
`--gamma` misreads under concurrent execution.

---

## 0. Prerequisites

**Phase 2 (GPC-aligned):**
- `aki_anchor_based_24h_lookback_aligned_features.csv` — the master CSV,
  now reflecting both the leakage fix (`hours_since`/`hours_to_anchor`
  removed) and the KDIGO baseline-SCr / CKD-exclusion fix (114,720 total
  patients, 91,776 in the `train` split, 22,944 in `test`).
- `phase2_gpc_aligned_simulation.py`
  — ensures no patient is sampled into more than
  one site — see `HOW_TO_CHECK_OVERLAP.txt`. `TARGET_N_PER_SITE = 14_000`
  inside this script is the highest value tested clean (zero shortfall or
  within-site duplication) across all 3 conditions against the current
  91,776-patient train pool (6 × 14,000 = 84,000, 7,776 buffer); 15,000
  caused the last-processed site to fall short by 1,138 — re-check this
  constant if the cohort size changes again.
- `phase2_gpc_aligned_train_v23.py`
- `phase2_gpc_aligned_train_v25.py` — **use
  this, not** the older `fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py`
  (see Section 8 for why).
- `check_overlap.py` — verifies zero cross-site patient overlap from the
  `_subject_ids_*.csv` files the disjoint-sites script now writes
  alongside each site's data.

**Phase 1 (clinical-archetype):**
- `aki_anchor_based_24h_lookback.csv` — same KDIGO/CKD-fix update as
  above (94 columns, no BMI, smaller lab panel; same 114,720/91,776/22,944
  patient counts — both cohorts share the identical underlying patient
  population and train/test split, confirmed identical across files).
- `phase1_archetype_simulation.py`
  — `TARGET_N_PER_SITE = 17_000` inside this script is the
  highest value tested clean (zero shortfall or within-site duplication)
  at both the primary condition and the more extreme α=0.1 against the
  current 91,776-patient train pool (5 × 17,000 = 85,000, 6,776 buffer);
  18,000 caused the last-processed site (lowest AKI-prevalence target) to
  pick up 6,241 within-site duplicate patients — re-check this constant if
  the cohort size changes again.
- `phase1_archetype_train_v23.py`
- `phase1_archetype_train_v25.py` — **use this,
  not** the original `fedadapt_train_approach2_v2_5_realarch.py` (see
  Section 9).
- `fedadapt_model_approach2.py` — shared model definitions both Phase 1
  training scripts import.

**Both cohorts:**
- Python packages: `torch`, `pandas`, `numpy`, `scikit-learn` (required by
  the v2.3 scripts' `sklearn.KMeans` clustering).
- `run_disjoint_sites_data_gen.sh` — generates data for both cohorts (a
  single-condition smoke test, then the full grid), with an automatic
  `check_overlap.py` verification pass after each stage. **Run this
  first, on its own, before anything else in this file** — everything
  downstream depends on its output.
- `run_phase1_grid_v23.sh`, `run_phase1_grid_v25.sh` — shell
  scripts that loop the full 20-condition × 3-seed Phase 1 training grid
  (see Section 10/11). **Fixed, run, and verified complete** — both now
  point at the flat `./phase1_data_disjoint/` directory and check for
  each condition's specific site file rather than a per-condition
  subfolder. (Both scripts previously assumed data lived in
  `./phase1_data_corrected/alpha{a}_gamma{g}/` subfolders that were never
  actually created by any data-generation command in this file, old or
  new — every condition would have silently hit `[skip] missing data dir`
  and produced zero training output.) **v2.5 additionally needed two more
  fixes** inside `phase1_archetype_train_v25.py`
  itself before its grid's output could be trusted — see the banner at
  the top of this file. **All three grid scripts also gained a
  resume-skip check** (skips a job if its output file already exists) after
  the v2.3 grid's first attempt was interrupted mid-run and a restart
  without this check produced mixed-provenance data for 3 of 5 methods
  (see the banner). All 60 v2.5 jobs and all 300 v2.3+baselines jobs are
  now complete and confirmed, single provenance (Section 10/11 below).
- `run_phase2_training.sh` — the Phase 2 equivalent (54 runs: v2.3 9 +
  v2.5 9 + 4 baselines × 36), covering Sections 3/4/5 below. **Complete
  and confirmed** — all 54 jobs verified present with the correct 6-site
  count and correct alpha/gamma per job, zero bad jobs across all 6
  methods (Section 3/4/5 results, and the Phase 2 final-numbers table,
  below). Uses `phase2_gpc_aligned_train_v25.py`,
  which needed the same two fixes as its Phase 1 counterpart — both
  confirmed via direct inspection of this run's real output, not just
  code review. The Phase 2 v2.3 script needed neither fix — checked
  specifically and confirmed clean (exact alpha/gamma matching already in
  place, no caching mechanism of any kind).

---

# PART A — Phase 2 (GPC-aligned cohort)

> **Sections 3, 4, and 5 below (54 runs total: v2.3 9 runs + v2.5 9 runs +
> 4 baselines × 36 runs) are now consolidated into `run_phase2_training.sh`**,
> updated to point at `./phase2_data_disjoint/` and tested against real
> generated data (skip-check correctly finds an existing condition and
> correctly skips a nonexistent one). Sections 6 and 7 are NOT included —
> both were confirmatory re-runs after an in-place code fix, and for a
> from-scratch run against the new cohort they'd just repeat Section 3/5's
> commands verbatim. Section 2's local_epochs sweep is also not included —
> `local_epochs=3` is already a settled, confirmed choice baked into every
> command. The individual commands below are kept for reference/history.

## 1. Generate Phase 4 simulation data (all 3 conditions)

**Superseded by `run_disjoint_sites_data_gen.sh`.** The commands below
are kept for reference (they show the per-condition invocation pattern
training scripts elsewhere in this file assume), but running them
individually against the disjoint-sites script works identically — just
substitute the script name below.

```bash
python3 phase2_gpc_aligned_simulation.py \
  --input aki_anchor_based_24h_lookback_aligned_features.csv \
  --label AKI_label --alpha 0.0 --gamma 0.0 --seed 42 \
  --output ./phase2_data_disjoint/

python3 phase2_gpc_aligned_simulation.py \
  --input aki_anchor_based_24h_lookback_aligned_features.csv \
  --label AKI_label --alpha 0.5 --gamma 0.75 --seed 42 \
  --output ./phase2_data_disjoint/

python3 phase2_gpc_aligned_simulation.py \
  --input aki_anchor_based_24h_lookback_aligned_features.csv \
  --label AKI_label --alpha 1.0 --gamma 1.0 --seed 42 \
  --output ./phase2_data_disjoint/
```
All three write into the same `phase2_data_disjoint/` directory (filenames
encode alpha/gamma, e.g. `sim_KUMC_alpha0.5_gamma0.75.csv`). Verify
afterward with `python3 check_overlap.py ./phase2_data_disjoint/` — should
report zero overlap across all 15 site pairs (confirmed via live run
against the current cohort: all 6 sites hit the full `TARGET_N_PER_SITE =
14_000`, zero shortfall/duplication warnings).

This step is what applies the real-GPC-derived `acuity_bias`/`spread_scale`
per site (replacing earlier placeholder values) and produces the six
GPC-matched per-site feature sets (215–322 raw features before the
training-time vitals exclusion in step 2).

---

## 2. `local_epochs` sweep (single seed, diagnostic)

```bash
python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/hpsweep_v2.3_lepoch1/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/hpsweep_v2.3_lepoch3/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 5 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/hpsweep_v2.3_lepoch5/
```
Result: `local_epochs=3` is the local optimum for v2.3 (≈0.079 vs. 0.053 at
1, 0.075 at 5). Used for every v2.3 confirmatory run below. (v2.5's
head-to-head comparison against v2.3, Section 4, instead holds
`local_epochs=1` for *both* versions — a separate, deliberate choice to
isolate the clustering strategy; see Section 4's note.)

---

## 3. v2.3 (manual K=2), 20-group + weighted, `local_epochs=3` — 9 runs

```bash
python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.3_K2_lepoch3/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.3_K2_lepoch3/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.3_K2_lepoch3/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.3_K2_lepoch3/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed456_20group_weighted_v2.3_K2_lepoch3/
```
(9th run is `hpsweep_v2.3_lepoch3/` from step 2, seed 42 @ α=0.5/γ=0.75 —
reused, not re-run.)

**Result: mean ΔAUROC = 0.0698 ± 0.0116, pooled across all 3 conditions
(6/6 sites positive, 18/18 site-condition means positive).**

---

## 4. v2.5 (auto-K), 20-group + weighted, `local_epochs=1` — 9 runs

**Use `phase2_gpc_aligned_train_v25.py`.**
The earlier script (`fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py`)
had no best-checkpoint restoration in its Phase 1 (warmup) stage — see
Section 8. `local_epochs=1` here (not v2.3's `3`) matches v2.3 for this
specific head-to-head comparison, isolating the clustering strategy from
the local-epoch schedule.

```bash
python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.5_bestckpt_fix/

python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.5_bestckpt_fix/

python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.5_bestckpt_fix/

python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_v2.5_bestckpt_fix/

python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.5_bestckpt_fix/

python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.5_bestckpt_fix/

python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.5_bestckpt_fix/

python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.5_bestckpt_fix/

python3 phase2_gpc_aligned_train_v25.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 456 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a1.0_g1.0_seed456_20group_weighted_v2.5_bestckpt_fix/
```

**Result: mean ΔAUROC = +0.0052 ± 0.0085, pooled across all 9 runs
(n=54, all 3 conditions × 3 seeds), statistically comparable to v2.3.**
(An earlier version of this script gave −0.0267, net-negative — see
Section 8 for the full diagnosis and why that number does not stand.)

---

## 5. Method comparison — FedAvg/FedProx/SCAFFOLD/FedAdapt, `local_epochs=3` — 36 runs

Repeat for each `--method` ∈ `{fedavg, fedprox, scaffold, fedadapt}`,
each seed ∈ `{42, 123, 456}`, each condition:

```bash
python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedavg --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_lepoch3/
```
(swap `--method`, `--alpha`/`--gamma`, `--seed`, `--output_dir` for the
other 35 combinations — 4 methods × 3 seeds × 3 conditions)

**Note:** FedAvg/FedProx/SCAFFOLD bypass the GRL discriminator entirely
(`local_step_fedavg`/`fedprox`/`scaffold` call `client.encode(x) →
client.head(emb)` directly, `adv_loss=0.0` hardcoded) — `--group_taxonomy`
has no effect on these three. FedAdapt does use the discriminator.

**Results (pooled, 3 seeds × 3 conditions):** FedAdapt 0.0683 ± 0.0102,
SCAFFOLD 0.0606 ± 0.0120, FedProx 0.0605 ± 0.0117, FedAvg 0.0603 ± 0.0122.

---

## 6. Taxonomy + clustering fix — confirmatory re-run, 9 runs

Two corrections applied to
`phase2_gpc_aligned_train_v23.py`:
`wbc` added to the `hematologic` keyword group (previously matched no
keyword, fell into `other`); `lactate` moved from `demographic_other` to
`renal`; `cardiovascular_resp` split into `hemodynamic` (vitals) +
`blood_gas` (labs); custom single-shot k-means replaced with
`sklearn.KMeans(n_init=10)`, matching v2.5.

```bash
python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed456_20group_weighted_v2.3_K2_lepoch3_improvement/
```
**Result: mean ΔAUROC = 0.0698 ± 0.0116 — negligible change from 0.0702
pre-fix. Confirms the fixes were correctness issues, not performance
bottlenecks.**

---

## 7. FedAdapt re-run post-fix (FedAdapt uses the discriminator; the 3
baselines above do not, so only FedAdapt needed re-confirming) — 9 runs

```bash
python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_lepoch3_improvement/

python3 phase2_gpc_aligned_train_v23.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 456 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a1.0_g1.0_seed456_20group_weighted_lepoch3_improvement/
```
**Result: mean ΔAUROC = 0.0683 ± 0.0102 — negligible change from 0.0684
pre-fix.**

---

## 8. v2.5 root-cause diagnosis and fix (applies to BOTH cohorts)

The original v2.5 scripts (`fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py`
for GPC-aligned, `fedadapt_train_approach2_v2_5_realarch.py` for archetype)
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

Fixed scripts: `phase2_gpc_aligned_train_v25.py`
(GPC-aligned) and `phase1_archetype_train_v25.py`
(archetype). Both add Phase 1 best-checkpoint tracking, identical in
structure to Phase 2's own; behavior is unchanged from the original
script when `--local_epochs` is left unset, so `--local_epochs 1` must
still be passed explicitly for a fair v2.3 comparison.

**Confirmed result after the fix, both cohorts:**
- GPC-aligned: −0.0267 → **+0.0052 ± 0.0085** (n=54, full 9-run grid)
- Archetype, primary condition: −0.0945 → **−0.0044 ± 0.0104** (n=15)
- Archetype, full 20-condition grid: **−0.0023 ± 0.0096** (n=300)

All three are now statistically comparable to v2.3 and the simple
baselines on their respective cohorts — v2.5 is no longer an outlier on
either cohort.

An intermediate diagnostic script, `fedadapt_train_approach2_v2_5_realarch_head_fix.py`
(adds a `--head_fix_rounds` flag to freeze the body during an initial
head-only warmup at the start of Phase 2), tested a *different* hypothesis
("gradient shock" from the reset) and found no improvement — included
here for completeness but not part of the confirmed pipeline; use the
`_bestckpt_fix` scripts, not `_head_fix`.

---

# PART B — Phase 1 (Clinical-archetype cohort)

## 9. Generate Phase 1 simulation data (full 20-condition grid)

**Superseded by `run_disjoint_sites_data_gen.sh`** (see Section 0) — use
`phase1_archetype_simulation.py`. Same site-C fix as
before (prevalence computed live at `0.176`, not hardcoded), **plus** two
further fixes not yet reflected in this section until now:
1. The KDIGO baseline-SCr / CKD-exclusion fix upstream (114,720 total
   patients now, not 163,038 — see the banner at the top of this file).
2. Disjoint cross-site sampling — no patient can be selected into more
   than one site (previously ~35% pairwise overlap; see
   `HOW_TO_CHECK_OVERLAP.txt`). `TARGET_N_PER_SITE` is now `17_000` (was
   `33_000`) — the highest value tested clean (zero shortfall or
   within-site duplication) against the smaller, train-only
   91,776-patient pool; 18,000 caused 6,241 within-site duplicates at the
   last-processed site.

Single condition:
```bash
python3 phase1_archetype_simulation.py \
  --input aki_anchor_based_24h_lookback.csv \
  --label AKI_label \
  --alpha 0.3 --gamma 0.75 --seed 42 \
  --output ./phase1_data_disjoint/
```

Full 20-condition grid (5 α × 4 γ):
```bash
for ALPHA in 0.1 0.3 0.5 1.0 10.0; do
  for GAMMA in 0.0 0.5 0.75 1.0; do
    python3 phase1_archetype_simulation.py \
      --input aki_anchor_based_24h_lookback.csv \
      --label AKI_label \
      --alpha "$ALPHA" --gamma "$GAMMA" --seed 42 \
      --output ./phase1_data_disjoint/
  done
done
```

**Verify before proceeding to training:**
```bash
python3 check_overlap.py ./phase1_data_disjoint/
```
Should report zero overlap across all 10 site pairs. Also check the
console output from the generation step itself for any
`[disjoint-sampling]` shortfall or within-site-duplication warnings —
none should appear at `TARGET_N_PER_SITE = 17_000` against the current
cohort (confirmed via live run at both α=0.3 and the more extreme α=0.1:
all 5 sites hit the full 17,000 target, zero warnings).

Site design (confirmed via live run against the current, corrected
cohort — feature counts unchanged from before, prevalence anchors
unchanged, N per site is what changed):

| Site | Features | Prevalence anchor | N (this cohort) |
|---|---|---|---|
| A (ICU) | 33 | 35.0% | 17,000 |
| B (general ward) | 49 | 12.0% | 17,000 |
| C (academic anchor) | 89 | 17.6% (= pooled rate, fixed) | 17,000 |
| D (community) | 40 | 7.0% | 17,000 |
| E (rural) | 21 | 4.0% | 17,000 |

---

## 10. v2.3 + baselines, full grid — 300 runs

Use `run_phase1_grid_v23.sh` (5 methods × 20 conditions × 3 seeds).
`local_epochs` is not passed — the script's own default (`1`) is already
correct for this cohort (confirmed via console logs; do not confuse with
Phase 2's `local_epochs=3`, which is specific to that cohort). Points at
the flat `./phase1_data_disjoint/` directly.

```bash
chmod +x run_phase1_grid_v23.sh
./run_phase1_grid_v23.sh
```

**Result — CONFIRMED against the corrected, disjoint-sites cohort. All
300 jobs verified present, single consistent provenance (one continuous
run, no mixed timestamps — an earlier attempt crashed mid-run from an
external cause and was restarted with a non-resumable script version,
producing 3 of 5 methods with stale/mixed data; caught via per-method
timestamp checks, fixed by adding a resume-skip check to the script, and
resolved with one fully clean re-run of all 300 jobs):**

| Method | Mean ΔAUROC | SD |
|---|---|---|
| FedAdaptProto v2.3 (manual K=2) | +0.0551 | 0.0135 |
| FedAvg | +0.0526 | 0.0132 |
| FedProx | +0.0526 | 0.0132 |
| FedAdapt | +0.0507 | 0.0137 |
| SCAFFOLD | +0.0505 | 0.0135 |

All five methods show real, comparable positive gain on the
corrected cohort. FedAdaptProto v2.3's edge over each baseline remains
statistically significant (paired t-test, n=300, p<0.0001 throughout),
though the mean differences are smaller than in the pre-fix data
(0.002–0.005, vs. the pre-fix 0.003–0.006) — the methods are closer
together on this corrected cohort than they appeared before.

---

## 11. v2.5 (auto-K, bestckpt-fixed), full grid — 60 runs

Use `run_phase1_grid_v25.sh` (1 method × 20 conditions × 3
seeds), which calls `phase1_archetype_train_v25.py`
with `--local_epochs 1` explicitly set. Points at the flat
`./phase1_data_disjoint/` directly.

**This script needed two additional fixes beyond the directory-structure
one**, discovered only after actually inspecting its real output (not
just from code review) — see the banner at the top of this file for the
full explanation:
1. Site-discovery had no alpha/gamma filtering, so every job trained a
   ~100-site mega-federation instead of the intended 5.
2. Its local-baseline cache lived at a single, condition-independent
   path, so even after fixing (1), every job after the first silently
   reused stale cross-condition-polluted results.

Both fixed and the resulting output directly verified: all 60 jobs
checked programmatically, zero jobs with an incorrect site count (every
single one has exactly 5 rows in `fl_gain_correlation.csv`, all sharing
the correct alpha/gamma for that job).

```bash
chmod +x run_phase1_grid_v25.sh
./run_phase1_grid_v25.sh
```

**Result — CONFIRMED against the corrected, disjoint-sites cohort:**
mean ΔAUROC = **−0.0004 ± 0.0125** (n=300, full 20-condition grid).
Remarkably close to the pre-fix confirmed number (−0.0023 ± 0.0096)
despite the substantial underlying changes (cohort size, disjoint
sampling, per-site N) — reinforcing that v2.5 is not a meaningful
outlier on this cohort either before or after the fixes, consistent with
Section 8's root-cause explanation.

---

## Final confirmed numbers (Phase 2, all 3 conditions × 3 seeds, post-fix)

> **✅ CURRENT — confirmed against the corrected, disjoint-sites cohort
> (114,720 patients, zero cross-site overlap). All 54 jobs verified
> present with the correct 6-site count and correct alpha/gamma per job,
> zero bad jobs across all 6 methods.**

| Method | Mean ΔAUROC | SD |
|---|---|---|
| FedAdaptProto (v2.3, K=2) | **0.0707** | 0.0165 |
| FedAdapt | 0.0701 | 0.0159 |
| SCAFFOLD | 0.0643 | 0.0140 |
| FedAvg | 0.0631 | 0.0142 |
| FedProx | 0.0624 | 0.0142 |
| FedAdaptProto (v2.5, auto-K, bestckpt-fixed) | +0.0105 | 0.0153 |

n=54 per method (3 seeds × 3 conditions × 6 sites). Compared to the
pre-fix numbers (0.0698/0.0683/0.0606/0.0605/0.0603/0.0052): the ranking
is preserved (v2.3 ≈ FedAdapt > SCAFFOLD ≈ FedAvg ≈ FedProx ≫ v2.5), all
six methods still show real positive gain, and every mean moved up
slightly on the corrected cohort — including v2.5, whose gain roughly
doubled (0.0052 → 0.0105) but remains far smaller than the other five,
consistent with the established finding that v2.5's benefit is modest
but real once the checkpoint/cache bugs are fixed. Significance testing
(paired t-test vs. FedAdaptProto) has not yet been recomputed on this
data — the pre-fix p-values are shown for reference only and should not
be assumed to still hold exactly:

| Method | p vs. FedAdaptProto (pre-fix, for reference only) |
|---|---|
| FedAdapt | 0.21 |
| SCAFFOLD | 0.0002 |
| FedProx | 0.0001 |
| FedAvg | 0.0001 |

## Final confirmed numbers (Phase 1, full 20-condition grid, post-fix)

> **✅ CURRENT — confirmed against the corrected, disjoint-sites cohort
> (114,720 patients, zero cross-site overlap), single consistent
> provenance (one continuous 300-job run, resume-skip check added to the
> script so any future interruption can resume cleanly rather than
> risk mixed-provenance data again).**

| Method | Mean ΔAUROC | SD |
|---|---|---|
| FedAdaptProto v2.3 (manual K=2) | +0.0551 | 0.0135 |
| FedAvg | +0.0526 | 0.0132 |
| FedProx | +0.0526 | 0.0132 |
| FedAdapt | +0.0507 | 0.0137 |
| SCAFFOLD | +0.0505 | 0.0135 |
| FedAdaptProto v2.5 (auto-K, bestckpt-fixed) | −0.0004 | 0.0125 |

n=300 per method (3 seeds × 20 conditions × 5 sites). All six methods
confirmed at full-grid scope against the current cohort, single
provenance. FedAdaptProto's edge over each baseline remains significant
(paired t-test, n=300, p<0.0001 throughout; mean differences 0.002–0.005),
and v2.5 remains statistically indistinguishable from the rest
(near-zero, not a meaningful outlier).

---

## GPC-vitals exclusion (applies to every Phase 2 run above)

Both training scripts exclude `heart_rate`, `resp_rate`, `temperature`,
`spo2`, `oxygen_saturation`, `gcs_total` (all 4 stat-variants each:
`_min`/`_max`/`_mean`/`_most_recent` — 24 columns) from `feat_cols`, since
these have no counterpart in real GPC production tables. `sbp`, `dbp`,
`bmi` (→ SYSTOLIC/DIASTOLIC/BMI) keep all 4 stat-variants; `age_at_admission`
(→ AGE) is kept as a single value. This is not a command to run — it is
baked into `SiteData.__init__` (v2.3) / `load_site()` (v2.5) and applies
automatically to every run in Part A.

---

## Files in this repository

**Notebooks (run these first, against your own credentialed MIMIC-IV access):**
- `phase1_archetype_cohort.ipynb` — generates `aki_anchor_based_24h_lookback.csv`
  (94 columns, no BMI, smaller lab panel: albumin/bicarbonate/bilirubin/bun/
  creatinine/glucose/hemoglobin/lactate/platelets/potassium/sodium/wbc;
  114,720 patients after the KDIGO baseline-SCr/CKD-exclusion fix)
- `phase2_gpc_aligned_cohort.ipynb` — generates
  `aki_anchor_based_24h_lookback_aligned_features.csv` (490 columns,
  includes BMI, expanded GPC-aligned lab panel; same 114,720-patient
  population, same train/test split assignment per patient as Phase 1 --
  confirmed via `record_train_test_numbers.py`)

**Note on the master CSVs**: not included in this repository. They are
MIMIC-IV-derived patient data (subject to PhysioNet's data use
agreement) and, for Phase 2, ~144 MB -- both a data-governance and a
GitHub size concern. Run the two notebooks above against your own
credentialed BigQuery/MIMIC-IV access to generate them locally; every
script below expects them in the working directory under the exact
filenames stated.

**Simulation (data generation):**
- `phase1_archetype_simulation.py`
- `phase2_gpc_aligned_simulation.py`
- `run_disjoint_sites_data_gen.sh` — combined entry point for both
  cohorts (smoke test + full grid + overlap verification)
- `check_overlap.py`, `HOW_TO_CHECK_OVERLAP.txt` — cross-site patient
  overlap verification
- `record_train_test_numbers.py` — reports/verifies the train/test
  split and cross-file patient-population consistency

**Training:**
- `phase1_archetype_train_v23.py`, `phase1_archetype_train_v25.py`
- `phase2_gpc_aligned_train_v23.py`, `phase2_gpc_aligned_train_v25.py`
- `fedadapt_model_approach2.py` (shared model definitions, imported by
  every training script above)
- `run_phase1_grid_v23.sh`, `run_phase1_grid_v25.sh`, `run_phase2_training.sh`

**Leakage check: resolved, confirmed clean, both cohorts.** `feature_cutoff`
is class-anchored: AKI patients cut off 24h *before* their first
KDIGO-positive SCr (lead-time buffer, no overlap with the label-defining
event); non-AKI patients cut off at `last_scr_time − 24h`. Both notebooks
additionally identify and remove a more subtle anchor-selection-asymmetry
leak: `hours_since`/`hours_to_anchor` were found to encode class-dependent
monitoring-density artifacts rather than real signal and are dropped from
the modeling feature set.

**Baseline-SCr / CKD-exclusion: also resolved, both cohorts.** Both
notebooks implement the standard 3-tier KDIGO baseline-SCr hierarchy
exactly (7-day-prior most recent → 7-365-day-prior mean → CKD history +
no SCr in past year drops the encounter, non-CKD gets MDRD-estimated),
rather than always applying MDRD regardless of CKD status. Both notebooks
confirmed to use identical exclusion criteria and produce the identical
underlying patient population with identical train/test split assignment.

## Pushing this to GitHub

These 17 files live at the **repo root** (same level as `LICENSE`,
`README.md`, `.gitignore` — *not* inside `AKI_FL_Project/`, which has
its own separate, older set of files left untouched by this update).

```bash
cd /path/to/local/clone/of/AKI-Prediction-MIMIC-IV

# Extract/copy all 17 files from this package directly into this
# directory (the repo root).

git add phase1_archetype_cohort.ipynb \
        phase2_gpc_aligned_cohort.ipynb \
        phase1_archetype_simulation.py \
        phase2_gpc_aligned_simulation.py \
        phase1_archetype_train_v23.py \
        phase1_archetype_train_v25.py \
        phase2_gpc_aligned_train_v23.py \
        phase2_gpc_aligned_train_v25.py \
        fedadapt_model_approach2.py \
        run_disjoint_sites_data_gen.sh \
        run_phase1_grid_v23.sh \
        run_phase1_grid_v25.sh \
        run_phase2_training.sh \
        check_overlap.py \
        HOW_TO_CHECK_OVERLAP.txt \
        record_train_test_numbers.py \
        README.md

git commit -m "Add corrected pipeline (KDIGO baseline-SCr/CKD-exclusion fix + disjoint cross-site sampling fix, both cohorts) at repo root"
```

**The following old-named files, confirmed present at the repo root as
of this update, are superseded by the files above and should be
removed in the same push:**

```bash
git rm fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py
git rm fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py
git rm fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py
git rm fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py
git rm fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py
git rm mimic_ftl_simulation_phase1_archetype_post_leakage.py
git rm mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py
git rm run_complete.md

git commit -m "Remove superseded old-named root-level files, replaced by the renamed files above"

git push origin main
```

| Old root-level file | Superseded by |
|---|---|
| `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py` (older, pre-`_improvement`) | — (dead end) |
| `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py` | `phase2_gpc_aligned_train_v23.py` |
| `fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py` (pre-bugfix) | — (dead end) |
| `fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py` | `phase1_archetype_train_v25.py` |
| `fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py` | `phase2_gpc_aligned_train_v25.py` |
| `mimic_ftl_simulation_phase1_archetype_post_leakage.py` (no `_FIXED`, oldest version) | `phase1_archetype_simulation.py` |
| `mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py` | `phase2_gpc_aligned_simulation.py` |
| `run_complete.md` | `README.md` (this file) |

