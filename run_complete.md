# Phase 2 (GPC-Aligned) — Complete Run Instructions

Unlike `QUICK_RUN.md` (abbreviated, one representative command per
step), this file lists every individual run explicitly so it can be
executed top-to-bottom to reproduce the full result set.

Run each command standalone (one at a time, or as a sequential non-backgrounded
block) — this project has repeatedly found silent `--data_dir`/`--alpha`/
`--gamma` misreads under concurrent execution.

---

## 0. Prerequisites

- `aki_anchor_based_24h_lookback_aligned_features.csv` — the leakage-fixed
  master CSV (leakage columns `hours_since`/`hours_to_anchor` already
  removed upstream at the notebook/CSV-export stage).
- `mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py`
- `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py`
- `fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py`
- Python packages: `torch`, `pandas`, `numpy`, `scikit-learn` (required by
  the v2.3 script's `sklearn.KMeans` clustering).

---

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
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/hpsweep_v2.3_lepoch1/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/hpsweep_v2.3_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 5 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/hpsweep_v2.3_lepoch5/
```
Result: `local_epochs=3` is the local optimum (≈0.079 vs. 0.053 at 1, 0.075
at 5). Used for every confirmatory run below.

---

## 3. v2.3 (manual K=2), 20-group + weighted, `local_epochs=3` — 9 runs

```bash
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.3_K2_lepoch3/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed456_20group_weighted_v2.3_K2_lepoch3/
```
(9th run is `hpsweep_v2.3_lepoch3/` from step 2, seed 42 @ α=0.5/γ=0.75 —
reused, not re-run.)

**Result: mean ΔAUROC = 0.0702 ± 0.0108, pooled across all 3 conditions
(6/6 sites positive, 18/18 site-condition means positive).**

---

## 4. v2.5 (auto-K), 20-group + weighted, `local_epochs=1` — 9 runs

```bash
python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.5_autoK/

python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.5_autoK/

python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.5_autoK/

python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_v2.5_autoK/

python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.5_autoK/

python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.5_autoK/

python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.5_autoK/

python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.5_autoK/

python3 fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 456 --local_epochs 1 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --auto_k --k_min 2 --k_max 5 \
  --output_dir ./results/a1.0_g1.0_seed456_20group_weighted_v2.5_autoK/
```
**Requires two script fixes, both already in `..._leakage_fixed.py`:**
site-loading now filters by the exact requested `--alpha`/`--gamma`
(previously pooled all 3 conditions' data into one federation on every
run), and `--local_epochs` must be passed explicitly (script default is 5).

**Result: mean ΔAUROC = −0.0267 ± 0.0102, pooled — net-negative, all 54
site-condition-seed observations negative.**

---

## 5. Method comparison — FedAvg/FedProx/SCAFFOLD/FedAdapt, `local_epochs=3` — 36 runs

Repeat for each `--method` ∈ `{fedavg, fedprox, scaffold, fedadapt}`,
each seed ∈ `{42, 123, 456}`, each condition:

```bash
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
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

**Results (pooled, 3 seeds × 3 conditions):** FedAdapt 0.0684 ± 0.0102,
SCAFFOLD 0.0606 ± 0.0120, FedProx 0.0605 ± 0.0117, FedAvg 0.0603 ± 0.0122.

---

## 6. Taxonomy + clustering fix — confirmatory re-run, 9 runs

Two corrections applied to
`fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py`:
`wbc` added to the `hematologic` keyword group (previously matched no
keyword, fell into `other`); `lactate` moved from `demographic_other` to
`renal`; `cardiovascular_resp` split into `hemodynamic` (vitals) +
`blood_gas` (labs); custom single-shot k-means replaced with
`sklearn.KMeans(n_init=10)`, matching v2.5.

```bash
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
  --discriminator_target group --n_clusters 2 \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_v2.3_K2_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
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
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed42_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed123_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.0 --gamma 0.0 --seed 456 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.0_g0.0_seed456_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed42_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed123_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 0.5 --gamma 0.75 --seed 456 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a0.5_g0.75_seed456_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 42 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a1.0_g1.0_seed42_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 123 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a1.0_g1.0_seed123_20group_weighted_lepoch3_improvement/

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py \
  --data_dir ./phase4_data/ --alpha 1.0 --gamma 1.0 --seed 456 --local_epochs 3 \
  --method fedadapt --group_taxonomy phase4_20group --group_class_weighting \
  --output_dir ./results/a1.0_g1.0_seed456_20group_weighted_lepoch3_improvement/
```
**Result: mean ΔAUROC = 0.0683 ± 0.0102 — negligible change from 0.0684
pre-fix.**

---

## Final confirmed numbers (all 3 conditions × 3 seeds, post-fix)

| Method | Mean ΔAUROC | SD | p vs. FedAdaptProto |
|---|---|---|---|
| FedAdaptProto (v2.3, K=2) | **0.0698** | 0.0116 | — |
| FedAdapt | 0.0683 | 0.0102 | 0.21 |
| SCAFFOLD | 0.0606 | 0.0120 | 0.0002 |
| FedProx | 0.0605 | 0.0117 | 0.0001 |
| FedAvg | 0.0603 | 0.0122 | 0.0001 |
| FedAdaptProto (v2.5, auto-K) | −0.0267 | 0.0102 | (net-negative) |

`p`-values from paired `t`-tests, n=9 matched seed-condition points.

---

## GPC-vitals exclusion (applies to every run above)

Both training scripts exclude `heart_rate`, `resp_rate`, `temperature`,
`spo2`, `oxygen_saturation`, `gcs_total` (all 4 stat-variants each:
`_min`/`_max`/`_mean`/`_most_recent` — 24 columns) from `feat_cols`, since
these have no counterpart in real GPC production tables. `sbp`, `dbp`,
`bmi` (→ SYSTOLIC/DIASTOLIC/BMI) keep all 4 stat-variants; `age_at_admission`
(→ AGE) is kept as a single value. This is not a command to run — it is
baked into `SiteData.__init__` (v2.3) / `load_site()` (v2.5) and applies
automatically to every run in this document.

---

## Files needed for GitHub

**Phase 2/4 (GPC-aligned) pipeline — everything this document's commands
actually use:**
- `AKI_Anchor_Based_Approach2_aligned_features_PHASE4_leakage_fixed.ipynb`
  — generates `aki_anchor_based_24h_lookback_aligned_features.csv`
  (490 columns, includes BMI, expanded GPC-aligned lab panel)
- `aki_anchor_based_24h_lookback_aligned_features.csv` — the master input
  to step 1 above
- `mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py`
- `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py`
- `fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py`
- This file (`run_complete.md`)

**Phase 1 (archetype cohort) pipeline — separate, not used by any command
above:**
- `AKI_Anchor_Based_Approach2_phase1_post_leakage.ipynb` — generates
  `aki_anchor_based_24h_lookback.csv` (94 columns, no BMI, smaller lab
  panel: albumin/bicarbonate/bilirubin/bun/creatinine/glucose/hemoglobin/
  lactate/platelets/potassium/sodium/wbc)
- Its output CSV (locally downloaded as
  `1788107013092_aki_anchor_based_24h_lookback.csv`, ~40 MB — under
  GitHub's 50 MB warning threshold, no LFS required for this one
  specifically; the numeric prefix is a Colab download timestamp, safe
  to rename before committing)

**Leakage check: resolved, confirmed clean.** `feature_cutoff` is
class-anchored (Cell 30, STEP 8): AKI patients cut off 24h *before* their
first KDIGO-positive SCr (lead-time buffer, no overlap with the
label-defining event); non-AKI patients cut off at `last_scr_time − 24h`.
The notebook additionally identifies and removes a more subtle
anchor-selection-asymmetry leak (Cell 38): `hours_since`/`hours_to_anchor`
were found to encode class-dependent monitoring-density artifacts rather
than real signal (recency-only features reached 0.62 AUROC on the
Phase 2/3 GPC-aligned cohort under the same construction) and are dropped
from the modeling feature set. No further action needed before using
this file.

Both files are already present in the local project folder — no copying
needed, just add them directly:

```bash
cd /path/to/AKI-Prediction-MIMIC-IV/AKI_FL_Project

git add AKI_Anchor_Based_Approach2_phase1_post_leakage.ipynb \
        1788107013092_aki_anchor_based_24h_lookback.csv

git commit -m "Add Phase 1 (archetype cohort) data pipeline: notebook + generated CSV"
git push origin main
```

**Size note:** both CSVs are large (the Phase 2/4 one is ~144 MB).
GitHub warns above 50 MB and blocks plain pushes above 100 MB without
Git LFS. Either enable LFS for `*.csv` before committing, or commit only
the generating notebooks (which reproduce the CSVs) and `.gitignore` the
data itself — also worth a data-governance check given this is
MIMIC-IV-derived patient data, separate from the size issue.

```bash
# if using LFS:
git lfs install
git lfs track "*.csv"
git add .gitattributes
```
