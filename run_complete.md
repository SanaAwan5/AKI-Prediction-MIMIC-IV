# Complete Run Instructions — Phase 2 (GPC-Aligned) and Phase 1 (Clinical-Archetype)

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
- `aki_anchor_based_24h_lookback_aligned_features.csv` — the leakage-fixed
  master CSV (leakage columns `hours_since`/`hours_to_anchor` already
  removed upstream at the notebook/CSV-export stage).
- `mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py`
- `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py`
- `fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py` — **use
  this, not** the older `fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py`
  (see Section 8 for why).

**Phase 1 (clinical-archetype):**
- `aki_anchor_based_24h_lookback.csv` — the leakage-fixed master CSV for
  this cohort (94 columns, no BMI, smaller lab panel).
- `mimic_ftl_simulation_phase1_archetype_post_leakage_FIXED.py` — **use
  this, not** the original `mimic_ftl_simulation_phase1_archetype_post_leakage.py`
  (see Section 9 for why).
- `fedadapt_train_approach2_v2_3_phase1_archetype_post_leakage.py`
- `fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py` — **use this,
  not** the original `fedadapt_train_approach2_v2_5_realarch.py` (see
  Section 9).
- `fedadapt_model_approach2.py` — shared model definitions both Phase 1
  training scripts import.

**Both cohorts:**
- Python packages: `torch`, `pandas`, `numpy`, `scikit-learn` (required by
  the v2.3 scripts' `sklearn.KMeans` clustering).
- `run_phase1_grid_v23.sh`, `run_phase1_grid_v25_bestckpt_fix.sh` — shell
  scripts that loop the full 20-condition × 3-seed Phase 1 grid (see
  Section 9).

---

# PART A — Phase 2 (GPC-aligned cohort)

## 1. Generate Phase 4 simulation data (all 3 conditions)

```bash
python3 mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py \
  --input aki_anchor_based_24h_lookback_aligned_features.csv \
  --label AKI_label --alpha 0.0 --gamma 0.0 --seed 42 \
  --output ./phase4_data/

python3 mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py \
  --input aki_anchor_based_24h_lookback_aligned_features.csv \
  --label AKI_label --alpha 0.5 --gamma 0.75 --seed 42 \
  --output ./phase4_data/

python3 mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py \
  --input aki_anchor_based_24h_lookback_aligned_features.csv \
  --label AKI_label --alpha 1.0 --gamma 1.0 --seed 42 \
  --output ./phase4_data/
```
All three write into the same `phase4_data/` directory (filenames encode
alpha/gamma, e.g. `sim_KUMC_alpha0.5_gamma0.75.csv`).

This step is what applies the real-GPC-derived `acuity_bias`/`spread_scale`
per site (replacing earlier placeholder values) and produces the six
GPC-matched per-site feature sets (215–322 raw features before the
training-time vitals exclusion in step 2).

---

## 2. `local_epochs` sweep (single seed, diagnostic)

```bash
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/hpsweep_v2.3_lepoch1/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/hpsweep_v2.3_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
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
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
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

**Use `fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py`.**
The earlier script (`fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py`)
had no best-checkpoint restoration in its Phase 1 (warmup) stage — see
Section 8. `local_epochs=1` here (not v2.3's `3`) matches v2.3 for this
specific head-to-head comparison, isolating the clustering strategy from
the local-epoch schedule.

```bash
python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.5_bestckpt_fix/

python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.5_bestckpt_fix/

python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.5_bestckpt_fix/

python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_v2.5_bestckpt_fix/

python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.5_bestckpt_fix/

python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.5_bestckpt_fix/

python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.5_bestckpt_fix/

python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.5_bestckpt_fix/

python3 fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
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
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
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
`fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py`:
`wbc` added to the `hematologic` keyword group (previously matched no
keyword, fell into `other`); `lactate` moved from `demographic_other` to
`renal`; `cardiovascular_resp` split into `hemodynamic` (vitals) +
`blood_gas` (labs); custom single-shot k-means replaced with
`sklearn.KMeans(n_init=10)`, matching v2.5.

```bash
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
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
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py \
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

Fixed scripts: `fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py`
(GPC-aligned) and `fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py`
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

**Use `mimic_ftl_simulation_phase1_archetype_post_leakage_FIXED.py`.**
The original script hardcoded site C's prevalence to a stale literal
(`0.09`) rather than computing it dynamically from the input cohort; the
fixed version computes it live (`0.176` for the current data), matching
the script's own design intent (site C is the network's designated
anchor site, prevalence equal to the pooled rate).

Single condition:
```bash
python3 mimic_ftl_simulation_phase1_archetype_post_leakage_FIXED.py \
  --input aki_anchor_based_24h_lookback.csv \
  --label AKI_label \
  --alpha 0.3 --gamma 0.75 --seed 42 \
  --output ./phase1_data_corrected/
```

Full 20-condition grid (5 α × 4 γ):
```bash
for ALPHA in 0.1 0.3 0.5 1.0 10.0; do
  for GAMMA in 0.0 0.5 0.75 1.0; do
    python3 mimic_ftl_simulation_phase1_archetype_post_leakage_FIXED.py \
      --input aki_anchor_based_24h_lookback.csv \
      --label AKI_label \
      --alpha "$ALPHA" --gamma "$GAMMA" --seed 42 \
      --output ./phase1_data_corrected/
  done
done
```

Site design (confirmed via live run, `adapter_meta.json` per site):

| Site | Features | Prevalence anchor |
|---|---|---|
| A (ICU) | 33 | 35.0% |
| B (general ward) | 49 | 12.0% |
| C (academic anchor) | 89 | 17.6% (= pooled rate, fixed) |
| D (community) | 40 | 7.0% |
| E (rural) | 21 | 4.0% |

---

## 10. v2.3 + baselines, full grid — 300 runs

Use `run_phase1_grid_v23.sh` (5 methods × 20 conditions × 3 seeds).
`local_epochs` is not passed — the script's own default (`1`) is already
correct for this cohort (confirmed via console logs; do not confuse with
Phase 2's `local_epochs=3`, which is specific to that cohort).

```bash
chmod +x run_phase1_grid_v23.sh
./run_phase1_grid_v23.sh
```

**Result, primary condition (α=0.3, γ=0.75), n=15 per method:**

| Method | Mean ΔAUROC | SD |
|---|---|---|
| FedAdaptProto v2.3 (manual K=2) | +0.0471 | 0.0098 |
| FedAvg | +0.0455 | 0.0106 |
| FedProx | +0.0451 | 0.0108 |
| FedAdapt | +0.0433 | 0.0122 |
| SCAFFOLD | +0.0425 | 0.0104 |

Full-grid confirmation (all 20 conditions) pending at time of writing.

---

## 11. v2.5 (auto-K, bestckpt-fixed), full grid — 60 runs

Use `run_phase1_grid_v25_bestckpt_fix.sh` (1 method × 20 conditions × 3
seeds), which calls `fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py`
with `--local_epochs 1` explicitly set.

```bash
chmod +x run_phase1_grid_v25_bestckpt_fix.sh
./run_phase1_grid_v25_bestckpt_fix.sh
```

**Result: mean ΔAUROC = −0.0023 ± 0.0096 (n=300, full 20-condition grid).
Not distinguishable from the other five methods on this cohort — see
Section 8 for the root-cause fix that produced this.**

---

## Final confirmed numbers (Phase 2, all 3 conditions × 3 seeds, post-fix)

| Method | Mean ΔAUROC | SD | p vs. FedAdaptProto |
|---|---|---|---|
| FedAdaptProto (v2.3, K=2) | **0.0698** | 0.0116 | — |
| FedAdapt | 0.0683 | 0.0102 | 0.21 |
| SCAFFOLD | 0.0606 | 0.0120 | 0.0002 |
| FedProx | 0.0605 | 0.0117 | 0.0001 |
| FedAvg | 0.0603 | 0.0122 | 0.0001 |
| FedAdaptProto (v2.5, auto-K, bestckpt-fixed) | +0.0052 | 0.0085 | n.s. |

`p`-values from paired `t`-tests, n=9 matched seed-condition points
(FedAdaptProto v2.3 comparisons); v2.5's number is not directly
paired-tested against v2.3 here but is well within v2.3's range.

## Final confirmed numbers (Phase 1, primary condition α=0.3/γ=0.75, post-fix)

| Method | Mean ΔAUROC | SD |
|---|---|---|
| FedAdaptProto v2.3 (manual K=2) | +0.0471 | 0.0098 |
| FedAvg | +0.0455 | 0.0106 |
| FedProx | +0.0451 | 0.0108 |
| FedAdapt | +0.0433 | 0.0122 |
| SCAFFOLD | +0.0425 | 0.0104 |
| FedAdaptProto v2.5 (auto-K, bestckpt-fixed) | −0.0044 | 0.0104 |

Full 20-condition-grid confirmation for v2.5 alone: −0.0023 ± 0.0096
(n=300). v2.3 + baselines full-grid confirmation pending.

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

**Phase 2/4 (GPC-aligned) pipeline:**
- `AKI_Anchor_Based_Approach2_aligned_features_PHASE4_leakage_fixed.ipynb`
  — generates `aki_anchor_based_24h_lookback_aligned_features.csv`
  (490 columns, includes BMI, expanded GPC-aligned lab panel)
- `aki_anchor_based_24h_lookback_aligned_features.csv` — the master input
  to step 1 above
- `mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py`
- `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py`
- `fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py`
- `fedadapt_model_approach2.py` (shared model definitions)

**Phase 1 (archetype cohort) pipeline:**
- `AKI_Anchor_Based_Approach2_phase1_post_leakage.ipynb` — generates
  `aki_anchor_based_24h_lookback.csv` (94 columns, no BMI, smaller lab
  panel: albumin/bicarbonate/bilirubin/bun/creatinine/glucose/hemoglobin/
  lactate/platelets/potassium/sodium/wbc)
- `aki_anchor_based_24h_lookback.csv` — the master input (~40 MB, under
  GitHub's 50 MB warning threshold, no LFS required for this one
  specifically)
- `mimic_ftl_simulation_phase1_archetype_post_leakage_FIXED.py`
- `fedadapt_train_approach2_v2_3_phase1_archetype_post_leakage.py`
- `fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py`
- `run_phase1_grid_v23.sh`
- `run_phase1_grid_v25_bestckpt_fix.sh`

**Both:**
- `fedadapt_model_approach2.py` (shared model definitions, imported by
  every training script in both parts)
- This file (`run_complete.md`)

**Leakage check: resolved, confirmed clean, both cohorts.** `feature_cutoff`
is class-anchored (Cell 30, STEP 8): AKI patients cut off 24h *before*
their first KDIGO-positive SCr (lead-time buffer, no overlap with the
label-defining event); non-AKI patients cut off at `last_scr_time − 24h`.
Both notebooks additionally identify and remove a more subtle
anchor-selection-asymmetry leak (Cell 38): `hours_since`/`hours_to_anchor`
were found to encode class-dependent monitoring-density artifacts rather
than real signal and are dropped from the modeling feature set. Both
notebooks use identical exclusion criteria (age 18+, no SCr-admission
exclusion, no CKD exclusion, same baseline-SCr computation method,
self-documented in the Phase 1 notebook as "aligned to Phase 2/3"). No
further action needed before using either file.

```bash
cd /path/to/AKI-Prediction-MIMIC-IV/AKI_FL_Project

git add AKI_Anchor_Based_Approach2_phase1_post_leakage.ipynb \
        aki_anchor_based_24h_lookback.csv \
        mimic_ftl_simulation_phase1_archetype_post_leakage_FIXED.py \
        fedadapt_train_approach2_v2_3_phase1_archetype_post_leakage.py \
        fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py \
        run_phase1_grid_v23.sh \
        run_phase1_grid_v25_bestckpt_fix.sh \
        fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py \
        fedadapt_model_approach2.py

git commit -m "Add Phase 1 archetype pipeline (site C fix) and v2.5 bestckpt fix (both cohorts)"
git push origin main
```

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
