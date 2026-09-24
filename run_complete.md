# Complete Run Instructions — Phase 2 (GPC-Aligned) and Phase 1 (Clinical-Archetype)

> **✅ ALL TRAINING COMPLETE ON BOTH COHORTS.**
> Both master CSVs reflect the corrected baseline-SCr computation (KDIGO
> 3-tier hierarchy) and disjoint cross-site sampling (zero overlap,
> verified via `check_overlap.py`). **Phase 1 v2.5 (60 jobs), Phase 1
> v2.3+baselines (300 jobs), and all of Phase 2 (54 jobs, all 6 methods)
> are complete and confirmed** — every job checked programmatically for
> correct site count and correct alpha/gamma per job, zero bad jobs found
> anywhere. The manuscript (`main.tex`) reflects all of this and is
> current as of this version.
>
> **Six real code defects were found and fixed along the way — see
> `BUGFIX_LOG.md`** for the full history (KDIGO/CKD-exclusion fix,
> disjoint-sampling fix, the v2.5 checkpoint/local_epochs bug, the v2.5
> site-discovery/baseline-cache bug, an interrupted-grid stale-data
> incident, and a pending AUPRC/local-ceiling architecture confound tied
> to the withdrawn FL Gain Index). Section 8 below keeps a short summary
> of the one that affects reading the v2.5 numbers in this file; the rest
> live only in the log, since none of them bear on how to run or
> reproduce the current pipeline.

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
  — **use this, not** the pre-disjoint-sites version, which is also still
  in the repo under its original name
  `mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py`. Ensures no
  patient is sampled into more than one site — see `HOW_TO_CHECK_OVERLAP.txt`. `TARGET_N_PER_SITE = 14_000`
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
  — **use this, not** the pre-disjoint-sites version, which is also still
  in the repo under its original name
  `mimic_ftl_simulation_phase1_archetype_post_leakage.py`, for the same
  reason as Phase 2 above. `TARGET_N_PER_SITE = 17_000` inside this script is the
  highest value tested clean (zero shortfall or within-site duplication)
  at both the primary condition and the more extreme α=0.1 against the
  current 91,776-patient train pool (5 × 17,000 = 85,000, 6,776 buffer);
  18,000 caused the last-processed site (lowest AKI-prevalence target) to
  pick up 6,241 within-site duplicate patients — re-check this constant if
  the cohort size changes again.
- `phase1_archetype_train_v23.py`
- `phase1_archetype_train_v25.py` — incorporates the Phase 1
  best-checkpoint fix described in Section 8.
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

**Result (historical, pre-disjoint-sites data — see "Final confirmed
numbers (Phase 2)" below for the current, correct figure of 0.0707):
mean ΔAUROC = 0.0698 ± 0.0116, pooled across all 3 conditions
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

**Result (historical, pre-disjoint-sites data — see "Final confirmed
numbers (Phase 2)" below for the current, correct figure of +0.0105):
mean ΔAUROC = +0.0052 ± 0.0085, pooled across all 9 runs
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
**Result (historical, pre-disjoint-sites data — see "Final confirmed
numbers (Phase 2)" below for the current, correct figure of 0.0707):
mean ΔAUROC = 0.0698 ± 0.0116 — negligible change from 0.0702
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
**Result (historical, pre-disjoint-sites data — see "Final confirmed
numbers (Phase 2)" below for the current, correct figure of +0.0701):
mean ΔAUROC = 0.0683 ± 0.0102 — negligible change from 0.0684
pre-fix.**

---

## 8. v2.5 fix — what changed and the confirmed result (full diagnosis: `BUGFIX_LOG.md` §3)

v2.5's original scripts reported strongly negative results on both
cohorts (GPC-aligned: −0.0267; archetype: −0.0945) from two causes: a
missing best-checkpoint restoration after v2.5's Phase-1 warmup stage,
and an uncontrolled `local_epochs` mismatch (v2.5 defaults to `5`, v2.3
to `1`). Fixed in `phase2_gpc_aligned_train_v25.py` and
`phase1_archetype_train_v25.py` — pass `--local_epochs 1` explicitly for
a fair v2.3 comparison; behavior is unchanged from the original script
when it's left unset.

**Confirmed result after the fix, both cohorts:**
- GPC-aligned: −0.0267 → **+0.0052 ± 0.0085** (n=54, full 9-run grid)
- Archetype, primary condition: −0.0945 → **−0.0044 ± 0.0104** (n=15)
- Archetype, full 20-condition grid: **−0.0023 ± 0.0096** (n=300)

All three are now statistically comparable to v2.3 and the simple
baselines on their respective cohorts — v2.5 is no longer an outlier on
either cohort under this fix. (A separate, later-discovered pair of bugs
in the same script family — site-discovery filtering and the
baseline-cache path — is documented in `BUGFIX_LOG.md` §4, not here.)

---

# PART B — Phase 1 (Clinical-archetype cohort)

## 9. Generate Phase 1 simulation data (full 20-condition grid)

**Superseded by `run_disjoint_sites_data_gen.sh`** (see Section 0) — use
`phase1_archetype_simulation.py`,
not the version without the `_disjoint_sites` suffix. Same site-C fix as
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

> **Protocol note — see Section 14.8 for the matched-baseline evaluation.**
> This table's baseline is each method's own incidentally-selected
> checkpoint (the conventional protocol). Section 14.8 reports the same
> runs scored against a baseline given the same checkpoint-selection
> privilege (the matched-baseline protocol) — a second, independent
> measurement, not a correction of this one. The relative ranking agrees
> between the two; the absolute gains do not.

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

> **Protocol note — see Section 14.8 for the matched-baseline evaluation.**
> Same as the Phase 2 table above: relative ranking agrees between the
> conventional and matched-baseline protocols; the absolute gains do not.

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

## 12. FL Gain Index computation (post-training analysis) — **WITHDRAWN**

> **⚠️ This section documents a superseded result. Do not report its
> numbers.** The three-term FL Gain Index was withdrawn on 2026-09-21.
> Two of its three terms are outcome prevalence entered twice under
> different names: `class_imbalance` is `1 - prevalence` by definition,
> and `positive_case_scarcity` derives from `n_positive`, which equals
> `33,000 x prevalence` because `n_samples` is constant across sites.
> Both correlate **-1.000** with prevalence on the 25-site cohort. Two
> collinear terms fitted with independent weights can trade off freely
> without changing the prediction, which is why the class-imbalance
> coefficient swung from `-1.508` to `+0.063` when the local baseline was
> corrected. The third term, `feature_sparsity`, is genuinely independent
> and does not predict (LOO balanced accuracy `0.500` on both metrics and
> both methods).
>
> The correlations tabulated below (r = 0.457, n=6; r = 0.637, n=5) were
> additionally computed against the pre-correction local baseline and at
> sample sizes where neither reaches significance, as the section itself
> notes.
>
> **Replacement: Section 14.** The section is kept unedited below for
> traceability, and `compute_flgi_correlation.py` still reproduces its
> numbers as a regression test against the old formula — that is now its
> only purpose.

Once both cohorts' training is confirmed (Sections 3–5, 10–11, and the
"Final confirmed numbers" tables above), the FL Gain Index and its
correlation with observed federated gain can be computed directly with
`compute_flgi_correlation.py` — a standalone script that reproduces the
formula embedded in the v2.5 training scripts
(`compute_fl_gain()`/`compute_fl_gain_revised()`), without needing to
re-run any training.

**Input**: a CSV with columns `site_id, n, n_features, prevalence,
observed_delta_auroc` — one row per site, using the confirmed N, feature
count, AKI prevalence, and observed ΔAUROC from the tables above (Table 2
+ Table 8 for GPC-aligned; Table 1 + Table 6 for archetype).

```bash
python3 compute_flgi_correlation.py example_gpc_aligned_sites.csv
python3 compute_flgi_correlation.py example_archetype_sites.csv
```

**Confirmed output** (matches the manuscript's Table 10 and the
archetype-cohort correlation exactly):

| Cohort | Pearson r | Spearman ρ | n |
|---|---|---|---|
| GPC-aligned | 0.457 (p=0.363) | 0.486 (p=0.329) | 6 |
| Archetype (primary condition) | 0.637 (p=0.247) | 0.700 (p=0.188) | 5 |

Neither reaches statistical significance at these sample sizes (n=6
would need \|r\| ≥ 0.811, n=5 would need \|r\| ≥ 0.878 to reach p<0.05 —
printed automatically by the script for whatever n is passed in). Both
`example_gpc_aligned_sites.csv` and `example_archetype_sites.csv` are
included as regression tests: re-running the script against them should
always reproduce the numbers above exactly. Pass `--csv-out <path>` to
additionally write the full per-site decomposition (S_i, I_i, F_i, FLGI,
role) to a CSV file.

---

## 13. AUPRC experiment + v2.5 local-baseline architecture fix (addendum)

> **⏳ PENDING — code fixed and compiled, not yet run.** This section
> documents two changes bundled into one addendum, both discovered while
> setting up an AUPRC-based FL-Gain experiment on top of the existing
> AUROC one:
>
> 1. **AUPRC was never carried through to `fl_gain_correlation.csv` /
>    `local_baseline_fl_gain_revised*.csv`.** v2.3's `evaluate_site()`
>    already computed it (`{"auroc", "f1", "auprc"}`) but
>    `run_local_only()` kept only `auroc`. v2.5's local-baseline loop
>    computed AUROC only and never called `average_precision_score` at
>    all.
> 2. **v2.5's local-only baseline used the wrong architecture** (see the
>    banner at the top of this file) — a separate generic 3-layer MLP
>    instead of the real `FedAdaptClient`, unlike v2.3's matched
>    counterfactual. This directly affects `local_ceiling` / `C_i^raw`,
>    the dominant term in Eq. flgi-post, for every v2.5 row in the pooled
>    n=390 fit.

### Files (already using clean repo-style names — see the mapping table
below for how they replace the existing repo files)

| File | What changed |
|---|---|
| `phase1_archetype_train_v23.py` | AUPRC plumbed through `run_local_only()` → `plot_fl_gain_correlation()` → `fl_gain_correlation.csv` (new `local_auprc`, `fed_auprc`, `improvement_auprc` columns). No architecture change — v2.3 was already correct. |
| `phase1_archetype_train_v25.py` | Same AUPRC plumbing inside `compute_or_load_shared_local_baseline()`, **plus** the local-only baseline now trains the real `FedAdaptClient` (no federation rounds, no adversarial branch) instead of the generic MLP. Cache filename now carries a `_realmodel` suffix (`local_baseline_fl_gain_revised{_alphaX_gammaY}_realmodel.csv`) so it can never silently reuse a stale, MLP-based cache. |
| `phase2_gpc_aligned_train_v25.py` | Identical fix to the Phase 1 v2.5 script (same two bugs, same fix), applied to the GPC-aligned script. |
| `run_phase1_grid_v25.sh` | Updated grid runner. **`--force_baseline` removed** — no longer needed since the `_realmodel` cache-filename suffix already guarantees a fresh, correctly-architected baseline on first use; the baseline now trains once per (alpha, gamma) condition and is reused across its 3 seeds, as originally intended, instead of retraining 3x per condition. Added a third version-safeguard grep for the architecture-fix signature (`ARCH FIX`/`_realmodel`). |
| `run_phase1_grid_v23.sh` | **Unchanged** — v2.3 never had the MLP-baseline issue, and the AUPRC patch is a pure addition (no new CLI flags), so the existing runner works against the updated `phase1_archetype_train_v23.py` as-is. |
| `run_phase2_training.sh` | **Unchanged** — same reasoning; the existing runner's calls to `phase2_gpc_aligned_train_v25.py`/`_train_v23.py` pick up both fixes automatically once those files are replaced (see the mapping table below). |

### Run

```bash
chmod +x run_phase1_grid_v25.sh run_phase1_grid_v23.sh

# v2.3 + baselines, all 5 methods, full 20-condition grid, now with AUPRC
# (300 jobs — same grid as Section 10, re-run to pick up the AUPRC columns)
./run_phase1_grid_v23.sh auprc_rerun

# v2.5, auto-K, full 20-condition grid, now with AUPRC + the real-model
# local baseline (60 jobs — same grid as Section 11)
./run_phase1_grid_v25.sh auprc_rerun

# Phase 2: re-run via the existing run_phase2_training.sh once
# phase2_gpc_aligned_train_v25.py / _train_v23.py are replaced (see mapping
# table below) -- no changes needed to run_phase2_training.sh itself.
./run_phase2_training.sh
```

The `auprc_rerun` tag keeps this output in `./results_phase1_grid_v23_auprc_rerun/`
and `./results_phase1_grid_v25_auprc_rerun/`, separate from the existing
Section 10/11 output, so the old AUROC-only results aren't overwritten
until the new run is confirmed clean.

### Expected outputs

- `local_baseline_fl_gain_revised{_alphaX_gammaY}_realmodel.csv` (v2.5,
  both cohorts) and the existing (unchanged-name) v2.3 baseline cache —
  now with `local_auprc`/`local_auprc_std` columns alongside
  `local_auroc`/`local_auroc_std`.
- `fl_gain_correlation.csv` (all methods, both versions) — now with
  `local_auprc`, `fed_auprc`, `improvement_auprc` columns alongside the
  existing AUROC columns.

### Still to do once the re-run is complete

1. Pool the new v2.3 + v2.5 `fl_gain_correlation.csv` outputs (all
   methods, both cohorts) the same way the existing n=390 AUROC pool was
   built (Section 12 / methods.tex).
2. Confirm `C_i^raw` (local ceiling) no longer differs systematically
   between v2.3 and v2.5 rows now that both use the same architecture —
   this was the actual point of the fix, so it's worth checking directly
   rather than assuming.
3. Fit the AUPRC analog of Eq. flgi-post (OLS: `improvement_auprc ~
   local_auprc_ceiling + I_i`) and report its $R^2$/Pearson/Spearman
   alongside the AUROC version.
4. Re-fit Eq. flgi-post itself on the corrected, consistent-basis AUROC
   data and compare the new coefficients/$R^2$ to the current
   $1.854 - 1.508\,C_i^{\text{raw}} - 0.844\,I_i$ ($R^2=0.7044$) — expect
   this to change, possibly materially, since roughly 14% of the pooled
   rows (54/390) had their `C_i^raw` recomputed under a different model.
5. Update methods.tex (Eq. flgi-post + its surrounding discussion),
   results.tex (the n=390 pooled-fit paragraph, Table/Figure
   `fig:flgi-post-scatter`, and the "v2.5 occupies the lower end of the
   predicted range" sentence, which was written to explain away a
   difference that may partly be this confound), and discussion.tex
   (the $R^2=0.6015$/$R^2=0.7044$ figures cited there) once the refit is
   in hand.

---

## 14. Joining-site prediction (replaces Section 12)

> **Use the `_auprc` training scripts.** `phase1_archetype_train_v25.py`
> and `phase2_gpc_aligned_train_v25.py` do **not** carry the val-selection
> patch (`grep -c VALSELECT` returns 0 on both; the `_auprc` variants return
> 11). Run a baseline against an unpatched script and it silently carves its
> own validation split, so the local baseline and the federated run select
> checkpoints on different rows — the mismatch this whole section exists to
> avoid. Verify before any baseline run:
> ```bash
> grep -c VALSELECT phase1_archetype_train_v25_auprc.py >                   phase2_gpc_aligned_train_v25_auprc.py
> ```
> Every baseline run should print `val split from script`. If it prints
> `internal`, stop.

Predicts whether a site not yet in the federation will benefit, and by
roughly how much. Replaces the withdrawn FL Gain Index. Every number is
leave-one-**site**-out: a joining site is by definition one the model was
not fitted on, so holding out a row (which leaves the same site in the
training set under other conditions) would be meaningless.

### 14.1 Why the cohort was expanded to 25 sites

At 5 sites the site-level claims cannot be validated: 10 ranking pairs,
Kendall tau moving in steps of 0.20, and a permutation null reaching
+0.40. The archetype cohort is simulated, so more sites is not a
compromise — it is the same generator sampled more densely. The five
published archetypes are preserved verbatim as anchors.

```bash
# Design: Latin hypercube over prevalence, acuity_bias, spread_scale,
# panel breadth. Rejects designs whose parameters arrive correlated
# (non-separable coefficients) -- re-run with a different --seed if so.
python3 design_expanded_cohort.py --n 25 --out cohort25

# Teach the simulator to read a cohort file (idempotent, one-time)
python3 apply_site_config_patch.py

# Generate. alpha=0.1 keeps the designed prevalence spread; see 14.5.
python3 phase1_archetype_simulation.py \
    --input ./aki_anchor_based_24h_lookback.csv \
    --site_config cohort25.json \
    --alpha 0.1 --gamma 0.0 --output ./phase1_data_x25
```

**Check the generate step before spending compute.** Sites with under
~200 positives train to near-chance and add noise rather than
information; the script reports per-site rows, positives and prevalence
for exactly this reason.

### 14.2 Baselines and federated runs

```bash
# Selection must never read the test split -- this is an AST check, not
# a grep, and it is a precondition for every baseline below.
python3 verify_no_test_selection.py

python3 compute_matched_baseline.py \
    --script phase1_archetype_train_v25_auprc.py \
    --data_dir ./phase1_data_x25 \
    --alpha 0.1 --gamma 0.0 --epochs 50 --seeds 3

for S in 42 43 44; do
  python3 phase1_archetype_train_v23.py \
      --data_dir ./phase1_data_x25 --alpha 0.1 --gamma 0.0 --seed $S \
      --rounds 50 --local_epochs 1 --method fedavg \
      --output_dir ./results_x25_a0.1_g0.0_fedavg/seed${S}/
done
```

Repeat with `--method fedadaptproto` into its own `--output_dir`. Both
methods are needed: agreement between them is what distinguishes a
result from a single algorithm's quirk.

### 14.3 Assembly and the headline fit

```bash
python3 recompute_gains.py ./results_x25_a0.1_g0.0_fedavg \
    --data-dir ./phase1_data_x25 --csv-out gains_x25_a0.1_fedavg.csv

python3 pool_site_conditions.py gains_x25_a0.1_fedavg.csv \
    --method fedavg --out pooled_x25_a0.1_fedavg.csv

python3 flgain_sign_rank.py --data pooled_x25_a0.1_fedavg.csv \
    --predictors prevalence --perms 2000

python3 joining_site_report.py --data pooled_x25_a0.1_fedavg.csv \
    --predictors prevalence --new-site 0.22
python3 joining_site_report.py --data pooled_x25_a0.1_fedavg.csv \
    --predictors prevalence --target delta_auprc --new-site 0.22
```

`--predictors` is required. Without it `flgain_sign_rank.py` falls back
to its original two terms (`local_auroc`, `I`), both of which are now
excluded — see 14.4.

**Confirmed output** (25 sites, alpha=0.1, gamma=0.0, 3 seeds):

| Method | Metric | Fit | Break-even | R2 |
|---|---|---|---|---|
| FedAvg | dAUROC | `-0.0144 + 0.0432*prev` | 0.333 | 0.370 |
| FedAvg | dAUPRC | `-0.0580 + 0.1719*prev` | 0.338 | 0.707 |
| FedAdaptProto | dAUROC | `-0.0154 + 0.0460*prev` | 0.335 | 0.392 |
| FedAdaptProto | dAUPRC | `-0.0440 + 0.1330*prev` | 0.331 | 0.606 |

| Test | Observed | Null (95th) | |
|---|---|---|---|
| Sign, dAUROC | 0.691 | 0.500 | clears |
| Sign, dAUPRC | 0.807 | 0.500 | clears |
| Ranking, Kendall tau | +0.353 | +0.167 | clears |

Output is an interval, not a point — the LOO residual SD (0.0071) is the
same order as the between-site spread (0.0082):

```
A site with prevalence 0.22:
  expected gain -0.005, 90% interval [-0.018, +0.005]
```

Achieved coverage (84% against a nominal 90%) is printed beside the
interval rather than assumed. For mid-range sites the interval spans
zero and the correct report is that the direction is unresolved.

### 14.4 Rejected predictors (optional, for the paper's defence)

```bash
python3 site_predictors.py ./phase1_data_x25 alpha0.1_gamma0.0 \
    site_predictors.csv

# Gradient alignment needs --track_gradient_conflict, which is NOT set
# by the runs in 14.2. Ten rounds is enough.
for S in 42 43 44; do
  python3 phase1_archetype_train_v23.py \
      --data_dir ./phase1_data_x25 --alpha 0.1 --gamma 0.0 --seed $S \
      --rounds 10 --local_epochs 1 --method fedavg \
      --track_gradient_conflict \
      --output_dir ./results_gc_x25_a0.1/seed${S}/
done
python3 gradient_predictor.py --gc-dir ./results_gc_x25_a0.1 \
    --gains pooled_x25_a0.1_fedavg.csv

# MLP ceiling. The _mlp suffix keeps it from overwriting the real
# baseline, but it DOES match recompute_gains.py's glob and parses to
# the same condition -- leave it in the data dir and every site gets two
# baseline rows. Move it out before running anything else.
python3 compute_matched_baseline.py \
    --script phase1_archetype_train_v25_auprc.py \
    --data_dir ./phase1_data_x25 \
    --alpha 0.1 --gamma 0.0 --epochs 50 --seeds 3 --arch mlp
mkdir -p ./mlp_baseline
mv ./phase1_data_x25/local_baseline_matched_alpha0.1_gamma0.0_mlp.csv \
   ./mlp_baseline/
ls ./phase1_data_x25/local_baseline_matched_*   # must show no _mlp file

python3 mlp_ceiling_test.py \
    --mlp ./mlp_baseline/local_baseline_matched_alpha0.1_gamma0.0_mlp.csv \
    --gains pooled_x25_a0.1_fedavg.csv --draws 2000
```

Nothing displaced prevalence; every added term lowered balanced accuracy
while raising its own null. Three failure modes, kept distinct because
they imply different things for other cohorts:

| Predictor | Bal. acc. | Null | Outcome |
|---|---|---|---|
| Prevalence | 0.691 | 0.500 | **retained** |
| Positive count | — | — | r = 1.000 with prevalence |
| Feature overlap | — | — | no variance (1.000 at all 25) |
| Gradient alignment | 0.504 | 0.500 | r = -0.897 with prevalence |
| Local ceiling C | 0.382 | 0.533 | inside the target |
| MLP ceiling | 0.596 | 0.566 | rho = 0.923 with C |
| Federation coverage | 0.500 | 0.500 | no signal |
| Distributional distance | 0.500 | 0.500 | no signal |

Two cautions that cost real time here:

1. **The MLP ceiling needs a calibrated null.** At `rho = 0.923` with
   `C_fed`, a synthetic ceiling carrying *zero* information still shows
   `r ~ +0.54` with the gain, because `delta = F - C_fed` contains
   `C_fed`. `mlp_ceiling_test.py` builds its null at the observed rho;
   only the excess over that null is evidence. Read raw correlations
   here and you will report an effect that is entirely the estimator.
2. **Distributional distance must use shared features only.** Computed
   over the union of all features it gives 0.784-1.000, which is
   median-imputation showing through in columns a site does not measure,
   not case-mix difference. Restricted to shared features: 0.499-0.662.

### 14.5 Controlled negative (run this)

The relationship requires prevalence heterogeneity to exist. Verify it
rather than assert it — regenerate the identical sites with the blending
parameter compressing the prevalence range, and confirm the result
disappears.

```bash
python3 phase1_archetype_simulation.py \
    --input ./aki_anchor_based_24h_lookback.csv \
    --site_config cohort25.json \
    --alpha 10.0 --gamma 0.0 --output ./phase1_data_x25
# then 14.2 and 14.3 with --alpha 10.0
```

| | alpha=0.1 | alpha=10.0 |
|---|---|---|
| prevalence spread | 0.1160 | 0.0116 |
| sign, balanced accuracy | 0.691 | 0.500 |
| ranking, Kendall tau | +0.353 | +0.113 |
| null 95th (tau) | +0.167 | +0.180 |
| verdict | clears both | clears neither |

Same generator, sites, pipeline and predictor; both tests collapse to
chance. A pipeline that manufactures apparent skill would have produced
a comparable fit in both conditions.

### 14.6 Data-scarcity sweep (the `--train_frac` learning curve)

Answers whether a site's local ceiling is set by how much data it has or
by how the model is trained. Validation and test stay fixed; only the
training split shrinks.

```bash
for TF in 0.02 0.05 0.1 0.25; do
  python3 compute_matched_baseline.py \
      --script phase1_archetype_train_v25_auprc.py \
      --data_dir ./phase1_data_x25 \
      --alpha 0.1 --gamma 0.0 --epochs 50 --seeds 3 --train_frac $TF
done
# the 100% point is the ordinary unsuffixed baseline, already present
ls ./phase1_data_x25/local_baseline_matched_alpha0.1_gamma0.0*
```

Each writes a distinct `_tf` suffix; none collides with the unsuffixed
baseline the gains table depends on.

**Confirmed output.** Seven condition-cohort combinations, five points
each, all matched protocol:

| Cohort | Condition | Sites | Budget | AUROC/doubling | AUPRC/doubling | R2 |
|---|---|---|---|---|---|---|
| Archetype | a=0.1 g=0 | 5 | 50 | +0.00959 | +0.01937 | 0.994 |
| Archetype | a=0.5 g=1 | 5 | 50 | +0.01141 | +0.02479 | 0.999 |
| Archetype | a=10 g=0 | 5 | 50 | +0.00985 | +0.02369 | 0.992 |
| Archetype (x25) | a=0.1 g=0 | 25 | 50 | +0.00912 | +0.02001 | 0.997 |
| GPC-aligned | a=0 g=0 | 6 | 150 | +0.01199 | +0.01726 | 0.986 |
| GPC-aligned | a=0.5 g=0.75 | 6 | 150 | +0.01143 | +0.01790 | 0.944 |
| GPC-aligned | a=1 g=1 | 6 | 150 | +0.01093 | +0.01677 | 0.992 |

The 25-site curve in full:

| train_frac | n_train | AUROC | AUPRC |
|---|---|---|---|
| 2% | 462 | 0.7474 | 0.4396 |
| 5% | 1,155 | 0.7603 | 0.4644 |
| 10% | 2,310 | 0.7667 | 0.4788 |
| 25% | 5,775 | 0.7801 | 0.5103 |
| 100% | 23,100 | 0.7992 | 0.5520 |

`AUROC = 0.6665 + 0.00912*log2(n_train)`, R2 = 0.997. **No saturation in
any condition** — in five of seven the last increment is the largest. All
slopes fall in +0.0091 to +0.0120, a 1.3x spread, while alpha varies 100x,
site count 5x and the epoch budget 3x. The data-quantity effect is
therefore separable from the prevalence-heterogeneity effect of Sec. 14.5,
which alpha abolishes entirely.

Do not report `corr(alpha, slope)` from three points per cohort — it
reaches -0.999 on Phase 2 across a 1.10x range of slopes. The spread is the
meaningful statistic.

**Interpretation.** Local data is the binding constraint. One doubling is
worth +0.0091 AUROC; federation across 25 sites delivers -0.0049 — a 0.0141
shortfall against what a single doubling would have supplied. Report that
conservative comparison rather than extrapolating the fit to 25x pooling
(+0.0424), which lies 4.6 doublings beyond the fitted range and assumes
pooled heterogeneous data behaves like more of the same.

**Two protocol checks this sweep requires.** Both are enforced by
`fit_scarcity.py`, which refuses to fit a condition whose points disagree:

```bash
python3 fit_scarcity.py ./scarcity_cohort
python3 fit_scarcity.py ./phase2_data_disjoint
```

1. **Every point must cover every site and use one protocol.** The script
   prints the point count and the shared (sites, seeds, budget, val_source);
   a condition that reports `(4 points)` is missing a file, and one that
   reports a `val_source` disagreement has an unpatched script in the mix.
2. **Phase 2's original baselines predate the `n_train` column.** The 100%
   point therefore has no x-value; `fit_scarcity.py` derives it as
   `n_train(tf)/tf` from the largest subsampled point (9,800 rows for the
   GPC cohort) and says so. Note the 2% GPC run used 256 rows where 2% of
   9,800 is 196 — a floor, and the recorded value is the one the fit needs.

### 14.7 Cohort identification before reusing any data

Three cohorts in this project share filenames and differ in content. A
table mixing them would look entirely normal. Before running a baseline
against site CSVs, fingerprint them — per-site prevalence and column
count identify a cohort uniquely:

```bash
python3 - <<'PY'
import glob, os, pandas as pd
D = './phase1_data_x25'          # directory under test
for f in sorted(glob.glob(os.path.join(D, 'site_*_alpha*_gamma*.csv'))):
    s = os.path.basename(f).split('_alpha')[0]
    df = pd.read_csv(f)
    lab = next((c for c in df.columns if c.lower() == 'aki_label'), None)
    print(f'{s:<10} rows {len(df):>7,}  cols {df.shape[1]:>4}  '
          f'prev {df[lab].mean():.4f}')
PY
```

Compare against the `prevalence` and feature counts recorded in the
matching `local_baseline_matched_*.csv`. If they disagree, the baselines
and the site data are from different cohorts and any gain computed
across them is meaningless.

### 14.8 Matched-baseline protocol — final numbers (both cohorts)

> **✅ CURRENT — a second, independent evaluation under a symmetric
> protocol, alongside the "Final confirmed numbers (Phase 1 / Phase 2)"
> tables above.** Those tables score each federated method against
> whatever local baseline its own job happened to produce — a different
> counterfactual method-to-method, since each method's own training run
> has its own incidentally-selected checkpoint (the conventional
> protocol). This section scores every method against the **one shared,
> matched-effort local baseline** per (site, condition), given the same
> checkpoint-selection privilege (`val_source=script`) the federated run
> gets (the matched-baseline protocol). The two are different
> measurements of the same runs, not a fix of one by the other — but they
> disagree materially, and only this section's numbers should be quoted
> as the paper's headline gains, since symmetric checkpoint selection is
> the protocol the paper's methodology commits to.

**Recompute (no retraining — both cohorts' federated runs and matched
baselines already existed):**

```bash
# Phase 1: matched baselines already existed for the full 20-condition
# grid (val_source=script confirmed), so this is the only step needed.
python3 recompute_gains.py ./results_phase1_grid_v23_auprc_valsel \
    --data-dir ./phase1_data_disjoint --csv-out gains_matched_p1.csv

# Phase 2
python3 recompute_gains.py ./results_phase2_auprc_valsel \
    --data-dir ./phase2_data_disjoint --csv-out gains_matched_p2.csv

# Significance, both metrics, both cohorts
python3 matched_method_pvalues.py gains_matched_p1.csv \
    --metric delta_auroc --csv-out pvalues_matched_p1_auroc.csv
python3 matched_method_pvalues.py gains_matched_p1.csv \
    --metric delta_auprc --csv-out pvalues_matched_p1_auprc.csv
python3 matched_method_pvalues.py gains_matched_p2.csv \
    --metric delta_auroc --csv-out pvalues_matched_p2_auroc.csv
python3 matched_method_pvalues.py gains_matched_p2.csv \
    --metric delta_auprc --csv-out pvalues_matched_p2_auprc.csv
```

`recompute_gains.py` guards against the variant baseline files
(`_tf*`/`_ve3`/`_mlp` suffixes, produced by Sections 14.4/14.6) leaking
into the match — it keeps only the canonical
`local_baseline_matched_alpha{a}_gamma{g}.csv` file per condition and
prints `ignored N variant baseline file(s)`. If that line does not
appear, or the observation count printed by the INVENTORY block is not
1,500 for Phase 1 / 324 for Phase 2, stop and check the data directory
for stray variant files before trusting the output.

**Phase 1 (n=1,500 = 5 methods × 20 conditions × 3 seeds × 5 sites,
local baseline peaked at epoch 7.7 of 50):**

| Method | Mean ΔAUROC | Mean ΔAUPRC | Sites > 0 |
|---|---|---|---|
| FedAdaptProto v2.3 | −0.0029 | −0.0099 | 43% |
| FedAdapt | −0.0048 | −0.0163 | 39% |
| FedAvg | −0.0060 | −0.0204 | 33% |
| FedProx | −0.0060 | −0.0204 | 33% |
| SCAFFOLD | −0.0087 | −0.0290 | 30% |
| **ALL (pooled)** | **−0.0060** | **−0.0204** | — |

**Phase 2 (n=324 = 6 methods × 3 conditions × 3 seeds × 6 sites, local
baseline peaked at epoch 1.5 of 150):**

| Method | Mean ΔAUROC | Mean ΔAUPRC | Sites > 0 |
|---|---|---|---|
| FedAdaptProto v2.3 (K=2) | −0.0069 | −0.0139 | 35% |
| FedAdapt | −0.0091 | −0.0178 | 28% |
| SCAFFOLD | −0.0129 | −0.0233 | 22% |
| FedAvg | −0.0138 | −0.0248 | 19% |
| FedProx | −0.0141 | −0.0251 | 17% |
| FedAdaptProto v2.5 (auto-K, bestckpt) | −0.0366 | −0.0485 | 0% |

**Two protocols, two findings.**
- **Absolute gain, matched-baseline protocol**: every method's mean gain
  is negative under a baseline that gets the same checkpoint-selection
  privilege as the federated run — the opposite sign from the
  conventional-protocol tables above. No method shows a positive
  matched-baseline gain on either cohort.
- **Relative ranking, both protocols**: unchanged between the two —
  including the FedAvg/FedProx tie in Phase 1 — FedAdaptProto v2.3 leads
  under both, SCAFFOLD (Phase 1) / FedAdaptProto v2.5 (Phase 2) trails
  under both.
- **Not marginal**: FedAdaptProto v2.5 is unambiguously the weakest
  configuration on Phase 2 (0% of sites positive, −0.0366 mean AUROC),
  well outside the spread of the other five methods.

**Significance (Wilcoxon signed-rank, job-level pairing — one mean per
condition-seed pooled across sites, vs. FedAdaptProto v2.3):**

| Phase 1 (n=60 jobs) | AUROC p | AUPRC p |
|---|---|---|
| vs. FedAdapt | <1e-6 | <1e-6 |
| vs. FedAvg | <1e-6 | <1e-6 |
| vs. FedProx | <1e-6 | <1e-6 |
| vs. SCAFFOLD | <1e-6 | <1e-6 |

| Phase 2 (n=9 jobs) | AUROC p | AUPRC p |
|---|---|---|
| vs. FedAdapt | not significant | not significant |
| vs. SCAFFOLD/FedAvg/FedProx | significant | mixed / not significant |
| vs. FedAdaptProto v2.5 | significant | significant |

Phase 2's AUPRC non-significance is underpowered (n=9), not
contradictory: Phase 1 (n=60) shows the same comparisons significant at
p<1e-6 on AUPRC with 2–3x larger effect sizes than AUROC, and its
significance fully replicates Phase 2's AUROC-level ranking. **Do not
re-run extra Phase 2 seeds to chase AUPRC significance there** — Phase 1
already demonstrates the pattern is real at adequate power; report the
Phase 1 result as the powered test and the Phase 2 result as consistent
but underpowered.

One caveat on FedAdaptProto: its Phase 2 v2.3 (K=2, −0.0069/−0.0139) and
v2.5 (auto-K, bestckpt, −0.0366/−0.0485) rows come from two distinct
algorithm configurations. `matched_method_pvalues.py`'s `--config` guard
refuses to pool them into one mean (this refusal caught a real bug — an
earlier naive pooling reversed the sign of the finding) — always pass
`--config` to select the configuration(s) intended, and check the
printed "Pooling configurations" list names each method exactly once.

**Report the means beside every p-value.** A significant difference
between two methods whose means are both negative says one loses less
than the other, not that either improves on local-only training.

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

## Files needed for GitHub

**Repo filenames are simplified from the local working names used
throughout this document** — the local names carry internal tracking
detail (which bug-fix stage, which script version) that matters while
actively developing, but is noise once something is finalized and
pushed. The table below is the authoritative local ↔ repo mapping.

**As of this version, every filename used elsewhere in this file is the
REPO name**, not the local working name — this file is now meant to be
executed directly against a fresh clone of the repo, with no mental
translation step. The table is retained so the older local names remain
traceable, and the `cp` staging block below is the one place both names
still appear side by side (it is what produces the repo names in the
first place). The only two exceptions, called out where they appear, are
the pre-disjoint-sites simulation scripts
(`mimic_ftl_simulation_phase1_archetype_post_leakage.py`,
`mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py`), which are in
the repo under those names and are referenced only to say *not* to use
them.

| Local working filename | Repo filename |
|---|---|
| `AKI_Anchor_Based_Approach2_phase1_post_leakage_updated_exclusion_criteria.ipynb` | `phase1_archetype_cohort.ipynb` |
| `AKI_Anchor_Based_Approach2_aligned_features_PHASE4_leakage_fixed_updated_exclusion_criteria.ipynb` | `phase2_gpc_aligned_cohort.ipynb` |
| `mimic_ftl_simulation_phase1_archetype_post_leakage_FIXED_disjoint_sites.py` | `phase1_archetype_simulation.py` |
| `mimic_ftl_simulation_phase4_gpc_aligned_post_leakage_disjoint_sites.py` | `phase2_gpc_aligned_simulation.py` |
| `fedadapt_train_approach2_v2_3_phase1_archetype_post_leakage.py` | `phase1_archetype_train_v23.py` |
| `fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py` | `phase1_archetype_train_v25.py` |
| `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py` | `phase2_gpc_aligned_train_v23.py` |
| `fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py` | `phase2_gpc_aligned_train_v25.py` |
| `run_phase1_grid_v25_bestckpt_fix.sh` | `run_phase1_grid_v25.sh` |
| `phase1_archetype_train_v23_auprc.py` (Section 13, replaces the row above) | `phase1_archetype_train_v23.py` |
| `phase1_archetype_train_v25_auprc.py` (Section 13, replaces the row above) | `phase1_archetype_train_v25.py` |
| `phase2_gpc_aligned_train_v25_auprc.py` (Section 13, replaces the row above) | `phase2_gpc_aligned_train_v25.py` |
| `run_phase1_grid_v25_auprc.sh` (Section 13, replaces the row above) | `run_phase1_grid_v25.sh` |
| `aki_anchor_based_24h_lookback.csv` | *(unchanged)* |
| `aki_anchor_based_24h_lookback_aligned_features.csv` | *(unchanged)* |
| `fedadapt_model_approach2.py` | *(unchanged)* |
| `run_phase1_grid_v23.sh` | *(unchanged)* |
| `run_phase2_training.sh` | *(unchanged)* |
| `run_disjoint_sites_data_gen.sh` | *(unchanged)* |
| `check_overlap.py` | *(unchanged)* |
| `HOW_TO_CHECK_OVERLAP.txt` | *(unchanged)* |
| `record_train_test_numbers.py` | *(unchanged)* |
| `compute_flgi_correlation.py` | *(unchanged)* |
| `example_gpc_aligned_sites.csv` | *(unchanged)* |
| `example_archetype_sites.csv` | *(unchanged)* |
| `run_complete.md` | *(unchanged)* |

**Phase 2/4 (GPC-aligned) pipeline:**
- `phase2_gpc_aligned_cohort.ipynb`
  — generates `aki_anchor_based_24h_lookback_aligned_features.csv`
  (490 columns, includes BMI, expanded GPC-aligned lab panel; now
  114,720 patients post KDIGO-baseline/CKD-exclusion fix)
- `aki_anchor_based_24h_lookback_aligned_features.csv` — the master input
  to step 1 above
- `phase2_gpc_aligned_simulation.py`
- `phase2_gpc_aligned_train_v23.py`
- `phase2_gpc_aligned_train_v25.py`
- `run_phase2_training.sh` — complete and confirmed (all 54 jobs, see
  the Phase 2 final-numbers table above)

**Phase 1 (archetype cohort) pipeline:**
- `phase1_archetype_cohort.ipynb`
  — generates `aki_anchor_based_24h_lookback.csv` (94 columns, no BMI,
  smaller lab panel: albumin/bicarbonate/bilirubin/bun/creatinine/
  glucose/hemoglobin/lactate/platelets/potassium/sodium/wbc; now 114,720
  patients post KDIGO-baseline/CKD-exclusion fix)
- `aki_anchor_based_24h_lookback.csv` — the master input (~40 MB, under
  GitHub's 50 MB warning threshold, no LFS required for this one
  specifically)
- `phase1_archetype_simulation.py`
- `phase1_archetype_train_v23.py`
- `phase1_archetype_train_v25.py`
- `run_phase1_grid_v23.sh`
- `run_phase1_grid_v25.sh`

**Both:**
- `fedadapt_model_approach2.py` (shared model definitions, imported by
  every training script in both parts)
- `run_disjoint_sites_data_gen.sh` — combined data-generation entry point
  for both cohorts (smoke test + full grid + overlap verification)
- `check_overlap.py`, `HOW_TO_CHECK_OVERLAP.txt` — cross-site patient
  overlap verification
- `record_train_test_numbers.py` — reports/verifies the train/test split
  and cross-file patient-population consistency from the two master CSVs
- `compute_flgi_correlation.py`, `example_gpc_aligned_sites.csv`,
  `example_archetype_sites.csv` — standalone FL Gain Index computation
  and correlation (Section 12, **WITHDRAWN**); the two example CSVs
  double as regression tests, confirmed to reproduce r=0.457/r=0.637
  exactly. Retained solely so the superseded formula stays reproducible
- Joining-site prediction (Section 14), which replaces the above:
  - `design_expanded_cohort.py` — Latin-hypercube cohort design
  - `apply_site_config_patch.py` — adds `--site_config` to the simulator
  - `apply_valselect_patch.py` — moves checkpoint selection off the test
    split (produces the `_auprc`/`val_source=script` scripts §14 requires)
  - `apply_epochckpt_patch.py` — adds `--ckpt_every_epoch`/`--eval_every`
  - `compute_matched_baseline.py` — matched-effort local baselines
  - `recompute_gains.py`, `pool_site_conditions.py` — gain assembly
  - `flgain_sign_rank.py` — sign + ranking, leave-one-site-out
  - `joining_site_report.py` — prediction intervals with coverage check
    (**not** `joining_site_recommendation.py` — that's an earlier tool
    built for the withdrawn FL-Gain-Index approach; nothing in this
    document calls it, don't push it as part of this pipeline)
  - `site_predictors.py` — n_positive, feature overlap/coverage, distance
  - `gradient_predictor.py` — gradient-alignment test
  - `mlp_ceiling_test.py` — MLP ceiling against a leakage-calibrated null
  - `fit_scarcity.py` — fits the `--train_frac` learning-curve sweep
    (Section 14.6)
  - `verify_no_test_selection.py` — AST check that selection never reads
    the test split (run before every baseline)
  - `matched_method_pvalues.py` — paired significance tests (job-level +
    site-level, t-test + Wilcoxon) against the shared matched baseline;
    `--config` guard refuses to pool a method across distinct algorithm
    configurations
  - `gains_matched_p1.csv`, `gains_matched_p2.csv` — recompute_gains.py
    output, matched-baseline gains for every method/condition/seed/site,
    both cohorts (Section 14.8)
  - `pvalues_matched_p1_auroc.csv`, `pvalues_matched_p1_auprc.csv`,
    `pvalues_matched_p2_auroc.csv`, `pvalues_matched_p2_auprc.csv` —
    matched_method_pvalues.py output, both metrics, both cohorts
  - `fix_results_naming.py` — writes `COHORT.txt` into every results
    directory (self-identifying contents); only renames the 2 directories
    whose name states the wrong cohort, and only if nothing still
    references the old name
- This file (`run_complete.md`)

**Leakage check: resolved, confirmed clean, both cohorts.** `feature_cutoff`
is class-anchored (Cell 30, STEP 8): AKI patients cut off 24h *before*
their first KDIGO-positive SCr (lead-time buffer, no overlap with the
label-defining event); non-AKI patients cut off at `last_scr_time − 24h`.
Both notebooks additionally identify and remove a more subtle
anchor-selection-asymmetry leak (Cell 38): `hours_since`/`hours_to_anchor`
were found to encode class-dependent monitoring-density artifacts rather
than real signal and are dropped from the modeling feature set.

**Baseline-SCr / CKD-exclusion: also resolved, both cohorts** (see the
banner at the top of this file) — both notebooks now implement the
standard 3-tier KDIGO baseline-SCr hierarchy exactly (7-day-prior most
recent → 7-365-day-prior mean → CKD history + no SCr in past year drops
the encounter, non-CKD gets MDRD-estimated), rather than always applying
MDRD regardless of CKD status. This is what took the cohort from 163,038
to 114,720 patients. Both notebooks confirmed to use identical exclusion
criteria and produce the identical underlying patient population with
identical train/test split assignment (verified via
`record_train_test_numbers.py`).

```bash
cd /path/to/AKI-Prediction-MIMIC-IV/AKI_FL_Project

# Stage copies under the clean repo names (see mapping table above) --
# this leaves your local working files, with their traceable names,
# completely untouched. Files not in the mapping table keep their name
# (cp X X is a harmless no-op).
#
# NOTE: the four training/grid scripts are NOT staged from their old
# local working names any more -- see the warning below this block.
cp AKI_Anchor_Based_Approach2_phase1_post_leakage_updated_exclusion_criteria.ipynb        phase1_archetype_cohort.ipynb
cp AKI_Anchor_Based_Approach2_aligned_features_PHASE4_leakage_fixed_updated_exclusion_criteria.ipynb  phase2_gpc_aligned_cohort.ipynb
cp mimic_ftl_simulation_phase1_archetype_post_leakage_FIXED_disjoint_sites.py             phase1_archetype_simulation.py
cp mimic_ftl_simulation_phase4_gpc_aligned_post_leakage_disjoint_sites.py                 phase2_gpc_aligned_simulation.py

# Section 13 (AUPRC + v2.5 real-model local baseline) -- these four are
# now the ONLY source for their repo names. Each already uses a clean
# repo-style name, so this is a straight overwrite, not a rename.
cp phase1_archetype_train_v23_auprc.py       phase1_archetype_train_v23.py
cp phase1_archetype_train_v25_auprc.py       phase1_archetype_train_v25.py
cp phase2_gpc_aligned_train_v25_auprc.py     phase2_gpc_aligned_train_v25.py
cp run_phase1_grid_v25_auprc.sh              run_phase1_grid_v25.sh

git add aki_anchor_based_24h_lookback.csv \
        aki_anchor_based_24h_lookback_aligned_features.csv \
        phase1_archetype_cohort.ipynb \
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
        compute_flgi_correlation.py \
        example_gpc_aligned_sites.csv \
        example_archetype_sites.csv \
        run_complete.md

git commit -m "KDIGO baseline-SCr/CKD-exclusion fix + disjoint cross-site sampling fix (both cohorts); Phase 1 training re-confirmed on corrected data; add standalone FL Gain Index computation"
git push origin main
```

> **⚠️ Why four `cp` lines were removed from the staging block above.**
> It previously also staged the training/grid scripts from their old
> local working names:
>
> ```
> cp fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py   phase1_archetype_train_v25.py
> cp fedadapt_train_approach2_v2_3_phase1_archetype_post_leakage.py   phase1_archetype_train_v23.py
> cp fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py  phase2_gpc_aligned_train_v23.py
> cp fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py phase2_gpc_aligned_train_v25.py
> ```
>
> The first of those was actively destructive and was verified as such on
> the real working directory:
> `fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py` is
> **not** a copy of `phase1_archetype_train_v25.py` — it has the
> site-discovery filter fix but is missing the condition-specific
> baseline-cache fix (`grep -c "_cond_suffix"` returns **0** on it,
> **3** on the clean-named file). Running that `cp` therefore silently
> replaced the good, fully-fixed Phase 1 v2.5 script with the half-fixed
> one, reintroducing the stale-cross-condition-cache bug described in the
> banner at the top of this file. The other three were byte-identical
> duplicates, harmless in themselves but pointless once the repo keeps
> only the clean names.
>
> **The four repo-name training/grid scripts are now sourced only from
> the Section 13 `*_auprc` files**, which carry every fix. The old local
> names should also be untracked so the duplicates stop reappearing —
> this removes them from GitHub while leaving every file on disk intact:
>
> ```bash
> git rm --cached fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py \
>                 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
>                 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
>                 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py
>
> printf '%s\n' \
>   'fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py' \
>   'fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py' \
>   'fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py' \
>   'fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py' >> .gitignore
>
> git add .gitignore
> git commit -m "Untrack local working-name duplicates; repo keeps only the clean names"
> ```

**Size note:** both master CSVs are large (the Phase 2/4 one is ~144 MB,
the Phase 1 one ~40 MB). GitHub warns above 50 MB and blocks plain pushes
above 100 MB without Git LFS. Either enable LFS for `*.csv` before
committing, or commit only the generating notebooks (which reproduce the
CSVs) and `.gitignore` the data itself — also worth a data-governance
check given this is MIMIC-IV-derived patient data, separate from the
size issue.

```bash
# if using LFS:
git lfs install
git lfs track "*.csv"
git add .gitattributes
```

**If replacing existing files already in the repo**: the `git add`
command above works identically whether the clean repo-name file is new
or already tracked — git detects the difference and stages a
modification instead of an addition automatically.

**If the repo currently has files under the *old* long local-style
names** (from before this mapping table existed) rather than the clean
names above, those old-named files are now orphaned — the commands above
never reference them, so they'll just sit in the repo unchanged and
increasingly out of date. Remove them explicitly in the same commit:
```bash
git rm AKI_Anchor_Based_Approach2_phase1_post_leakage.ipynb  # or whatever the old repo filenames actually are
```
(substitute whatever old names are actually present in the repo — check
with `git ls-files` if unsure).

---

## Pushing the matched-baseline correction (Section 14.8) — final fl-gain files

A focused commit for just what changed in this round: the matched-baseline
scripts, their CSV output, and the two docs (this file and `README.md`).
Nothing here retrains anything — it's the recompute/significance/naming
layer added after the checkpoint-selection-asymmetry finding.

```bash
cd /path/to/AKI-Prediction-MIMIC-IV/AKI_FL_Project

git add recompute_gains.py \
        matched_method_pvalues.py \
        fix_results_naming.py \
        gains_matched_p1.csv \
        gains_matched_p2.csv \
        pvalues_matched_p1_auroc.csv \
        pvalues_matched_p1_auprc.csv \
        pvalues_matched_p2_auroc.csv \
        pvalues_matched_p2_auprc.csv \
        README.md \
        run_complete.md

git commit -m "Matched-baseline correction: shared checkpoint-selection privilege for local vs. federated comparison, both cohorts

Every method's absolute gain flips negative against the matched
baseline; relative ranking is preserved exactly. Phase 1 (n=60 jobs)
significant at p<1e-6 on both AUROC and AUPRC for all four comparisons;
Phase 2 (n=9 jobs) AUPRC nulls explained as underpowered, not
contradictory."

git push origin main
```

**Not included, per standing instruction (\"no need to push texes\")**:
the manuscript `.tex` sections (`joining_site.tex`, `expanded_cohort.tex`,
etc.) and `RESULTS_SUMMARY.md`. Add them to the `git add` line above if
that's changed. The `COHORT.txt` manifest files (131 of them, written by
`fix_results_naming.py --manifest`) are also left out here — they live
inside the `results*/` directories, which are not part of the file lists
above and are assumed untracked/gitignored; if `results*/` is tracked in
this repo, decide separately whether 131 small marker files are worth a
commit.

### Section 14 pipeline scripts (second commit — these were still missing)

The commit above only covered the recompute/significance layer. The
scripts Section 14's own commands call by name — the matched-baseline
computation, cohort design, and predictor-rejection tests — were never
pushed at all. This is the complete, live set; nothing withdrawn or
superseded is in it (see the "Analysis — superseded" note in README.md's
Repository contents: `joining_site_recommendation.py`, an earlier tool
for the withdrawn FL-Gain-Index approach, is deliberately excluded, as is
any ad hoc diagnostic script used only during development and never
called by a documented command here).

```bash
cd "/Users/awans/Documents/AKI Prediction on MIMIC-IV/AKI_FL_Project"

git add design_expanded_cohort.py \
        apply_site_config_patch.py \
        verify_no_test_selection.py \
        compute_matched_baseline.py \
        pool_site_conditions.py \
        flgain_sign_rank.py \
        joining_site_report.py \
        site_predictors.py \
        gradient_predictor.py \
        mlp_ceiling_test.py \
        fit_scarcity.py

git commit -m "Add joining-site prediction pipeline scripts (Section 14)

Matched-effort baseline computation, 25-site Latin-hypercube cohort
design, predictor-rejection tests (gradient alignment, MLP ceiling),
and the data-scarcity sweep. These are called by name in run_complete.md
Section 14; without them the documented commands are not reproducible
from a fresh clone."

git push origin main
```

**`apply_valselect_patch.py` and `apply_epochckpt_patch.py` are deliberately
excluded from this push.** They are one-time patch-appliers: their job was
to inject the VALSELECT / checkpoint-epoch logic into a training script.
That already happened — the `_auprc` training scripts Section 14 calls
(`phase1_archetype_train_v23_auprc.py`, `phase1_archetype_train_v25_auprc.py`,
`phase2_gpc_aligned_train_v23_auprc.py`, `phase2_gpc_aligned_train_v25_auprc.py`)
already carry the patch (`grep -c VALSELECT` returns 11 on them). No command
in this document re-runs the patcher, so these two scripts are provenance
(how the patched files came to exist), not a dependency of reproducing any
result here. Push them only if you want the patch process itself
documented/reproducible in the repo, not because anything requires it.

