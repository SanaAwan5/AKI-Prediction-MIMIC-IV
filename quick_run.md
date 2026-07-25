# FTL/AKI — Quickstart for Real GPC Deployment

This folder's scripts were developed and validated on **MIMIC-IV, simulating
6 real GPC sites' feature patterns** (Phase 4) — not on real multi-institution GPC
data. 

---


## File Guide 

| File | Status | What it is |
|---|---|---|
| `fedadapt_train_approach2_v2_3_ftablation_taxtest.py` | **CANONICAL (v2.3)** | All 5 methods. Defaults = runD (`local_epochs=1`, `finetune_epochs=30`, `ft_lr_mult=0.1`, `ft_pos_weight=off`, `ft_grad_clip=5.0`). Best config found: `--group_taxonomy phase4_20group --group_class_weighting` (+0.0664, confirmed) — beats both native and `--discriminator_target site` (+0.0589). |
| `fedadapt_train_approach2_v2_5_grouptest.py` | **CANONICAL (v2.5)** | fedadaptproto only. Defaults to `local_epochs=5` **deliberately** — always pass `--local_epochs 1 --n_rounds 50` explicitly. Best config: `--discriminator_target group --group_taxonomy phase4_20group`. |
| `fedadapt_model_approach2.py` | **CANONICAL (shared)** | Model classes + loss functions. Required in the same directory as either training script above. |
| `mimic_ftl_simulation_phase4_gpc_aligned.py` | Simulation only | Builds the 6 simulated sites from a single MIMIC-IV cohort CSV. Not needed for real GPC data. |
| `AKI_Anchor_Based_Approach2_aligned_features_PHASE4.ipynb` | Simulation only | Extracts the MIMIC-IV cohort CSV feeding the simulation script. Not needed for real GPC data. |
| `PHASE4_runbook_configs.md` | Reference | Full parameter tables |


---

## Step-Wise Pipeline, With Example Commands

### Step 1 — Feature extraction at each site
Each real site independently produces a CSV: one row per admission, the shared
GPC common-data-model features + that site's own additional local features +
an AKI outcome label defined identically across sites. 

### Step 2 — Adapt the data-loading layer
`load_all_sites()` (v2.3) / `load_site()` (v2.5) currently expect files named
`<site_id>_alpha<A>_gamma<G>.csv` (the alpha/gamma suffix is a simulation
artifact — drop it for real data) plus a matching `<site_id>_adapter_meta.json`
with `input_dim`, `embedding_dim`, `feature_names`, `aki_prevalence`. This
layer needs rewriting to point at each site's real extracted CSV.

### Step 3 — Confirm the feature-group taxonomy transfers
`phase4_20group` uses standard ICD-9 concepts + lab-panel groupings — should
transfer directly since it's based on public coding standards, not anything
MIMIC-IV-specific. Confirm real GPC sites use ICD-9 (not ICD-10 or a mix)
before assuming `icd9_chapter()` applies as-is.

### Step 4 — Single-condition sanity run (simulation, to confirm the pipeline works)
```bash
cd "/Users/awans/Documents/AKI Prediction on MIMIC-IV/AKI_FL_Project"
source flamby_env/bin/activate

python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
    --data_dir ./phase4_gpc_aligned_sites_expanded/ \
    --output_dir ./results_sanity_check/ \
    --method fedadaptproto --n_clusters 3 \
    --group_taxonomy phase4_20group --group_class_weighting \
    --alpha 0.5 --gamma 0.75 --seed 42
```

**[Optional] Same sanity check with v2.5** — v2.3 and v2.5 both do best on
`group`-target (an earlier version of this doc stated they preferred opposite
targets; that was traced to a fine-tune-recipe bug and has been corrected).
Note `--local_epochs 1 --n_rounds 50` must be passed explicitly; v2.5's
script deliberately defaults to `local_epochs=5` to force this decision
every time rather than risk a silent mismatch:
```bash
python3 fedadapt_train_approach2_v2_5_grouptest.py \
    --data_dir ./phase4_gpc_aligned_sites_expanded/ \
    --output_dir ./results_sanity_check_v25/ \
    --method fedadaptproto --n_clusters 3 --alpha_proto 0.5 \
    --local_epochs 1 --n_rounds 50 \
    --ft_lr_mult 1.0 --ft_epochs 10 --ft_pos_weight --ft_grad_clip 0.0 \
    --discriminator_target group --group_taxonomy phase4_20group \
    --alpha 0.5 --gamma 0.75 --seed 42
```

### Step 5 — Multi-seed (confirm results aren't a lucky single draw)
This investigation found single-seed comparisons misleading three separate
times (see runbook) — don't skip this step, on simulated or real data.
```bash
cat > run_multiseed.sh << 'SCRIPT_EOF'
#!/bin/bash
for seed in 1 2 3 4 5; do
    for m in fedadapt fedadaptproto fedavg fedprox scaffold; do
        echo "=== seed$seed / $m ==="
        python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
            --data_dir ./phase4_gpc_aligned_sites_expanded/ \
            --output_dir ./results_multiseed/seed${seed}/ \
            --method $m --n_clusters 3 --group_taxonomy phase4_20group --group_class_weighting \
            --alpha 0.5 --gamma 0.75 --seed $seed
    done
done
SCRIPT_EOF
bash run_multiseed.sh
```

### Step 6 — Multi-condition (confirm results hold across heterogeneity levels)
Requires the simulation re-run with `--sweep` first (simulation-only — for
real GPC data, "conditions" would instead mean testing across whatever
natural site-mix/time-period splits are meaningful in the real network,
since there's no `alpha`/`gamma` to sweep on real data at all):
```bash
python3 mimic_ftl_simulation_phase4_gpc_aligned.py \
    --input ./aki_anchor_based_24h_lookback_aligned_features.csv \
    --sweep \
    --output ./phase4_gpc_aligned_sites_expanded_sweep/

cat > run_multicondition.sh << 'SCRIPT_EOF'
#!/bin/bash
CONDITIONS=("alpha0.1_gamma0.0" "alpha0.1_gamma1.0" "alpha10.0_gamma0.0" "alpha10.0_gamma1.0" "alpha0.5_gamma0.75")
METHODS=("fedadapt" "fedadaptproto" "fedavg" "fedprox" "scaffold")
for combo in "${CONDITIONS[@]}"; do
    alpha_val=$(echo "$combo" | sed -E 's/alpha([0-9.]+)_gamma.*/\1/')
    gamma_val=$(echo "$combo" | sed -E 's/.*_gamma([0-9.]+)/\1/')
    for m in "${METHODS[@]}"; do
        echo "=== $combo / $m ==="
        python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
            --data_dir "./phase4_gpc_aligned_sites_expanded_sweep/${combo}/" \
            --output_dir "./results_multicondition/${combo}/" \
            --method "$m" --n_clusters 3 --group_taxonomy phase4_20group --group_class_weighting --seed 42 \
            --alpha "$alpha_val" --gamma "$gamma_val"
    done
done
SCRIPT_EOF
bash run_multicondition.sh
```

### Step 7 — Real GPC data run (once Steps 1–2 adaptation is done)
Same command shape as Step 4, just pointed at real data — **treat the specific
config choices below as starting points, not guaranteed-best**, per Step 8:
```bash
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
    --data_dir ./real_gpc_sites/ \
    --output_dir ./results_real_gpc/ \
    --method fedadaptproto --n_clusters 3 \
    --group_taxonomy phase4_20group --group_class_weighting \
    --seed 42
# no --alpha/--gamma needed for real data -- those flags are logging-only,
# a simulation artifact with no real-data equivalent
```

**[Optional] Same run with v2.5:**
```bash
python3 fedadapt_train_approach2_v2_5_grouptest.py \
    --data_dir ./real_gpc_sites/ \
    --output_dir ./results_real_gpc_v25/ \
    --method fedadaptproto --n_clusters 3 --alpha_proto 0.5 \
    --local_epochs 1 --n_rounds 50 \
    --ft_lr_mult 1.0 --ft_epochs 10 --ft_pos_weight --ft_grad_clip 0.0 \
    --discriminator_target group --group_taxonomy phase4_20group \
    --seed 42
```
Worth actually running both on real data, not just v2.3 — there is a ~0.047 mean
Δauroc gap between v2.3's and v2.5's best simulation configs.

Then repeat Steps 5–6's multi-seed/multi-condition pattern on real data before
trusting any single result — this matters at least as much on real data as it
did in simulation, likely more given real institutional variation is less
controlled than the simulation's.

### Step 8 — Re-validate configuration choices, don't assume they transfer
Start from the best-found simulation configs (see File Guide) but **treat the
specific numbers as MIMIC-IV-simulation results, not predictions for real
data.** Real GPC data will have genuinely different site heterogeneity,
feature distributions, and sample sizes. Re-validate `--discriminator_target`
specifically — it was the single most consequential, least intuitive finding
in this whole investigation ("group-target consistently beats site-target for both v2.3 and v2.5"), so `group`-target is
the safer starting point on real data too; though still worth re-confirming
rather than assuming.

---

## Known Open Question

Even with architecture, prototype aggregation, discriminator taxonomy +
weighting, `local_epochs`, and discriminator target all matched between v2.3
and v2.5, a real gap in mean Δauroc remains.
