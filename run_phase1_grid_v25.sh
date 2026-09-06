#!/bin/bash
# Full 20-condition x 3-seed grid, v2.5 (auto-K) FedAdaptProto only,
# on the disjoint-sites-corrected phase1_data_disjoint/ (patients disjoint
# across sites; cohort updated after the KDIGO baseline-SCr/CKD-exclusion
# fix -- see run_complete.md), using the bestckpt_fix script (Phase 1
# best-checkpoint restoration) and local_epochs=1 (matching v2.3's actual
# archetype-cohort value, confirmed via console logs and Table 3 in the
# manuscript -- NOT local_epochs=3, which is specific to the
# GPC-aligned/Phase 2 cohort only).
# 1 method x 20 conditions x 3 seeds = 60 jobs.
#
# IMPORTANT: mimic_ftl_simulation_*.py writes ALL conditions into the SAME
# flat directory (one CSV per site per condition, e.g.
# site_A_alpha0.3_gamma0.75.csv) -- there are no per-condition subfolders.
# The training script's load_site() already handles condition selection by
# matching the alpha/gamma-suffixed filename within that flat directory, so
# DATA_DIR below is the same flat root for every iteration, not a
# per-condition path.

set -e

ALPHAS=(0.1 0.3 0.5 1.0 10.0)
GAMMAS=(0.0 0.5 0.75 1.0)
SEEDS=(42 123 456)

DATA_DIR="./phase1_data_disjoint"
OUT_ROOT="./results_phase1_grid_v25_bestckpt_fix"

for ALPHA in "${ALPHAS[@]}"; do
  for GAMMA in "${GAMMAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      OUT_DIR="${OUT_ROOT}/alpha${ALPHA}_gamma${GAMMA}_seed${SEED}/"

      # [RESUME] Skip any job whose output already exists from an
      # earlier, interrupted run of this same script.
      if [ -f "${OUT_DIR}fedadaptproto/fl_gain_correlation.csv" ]; then
        echo "[resume-skip] already completed: alpha=$ALPHA gamma=$GAMMA seed=$SEED"
        continue
      fi

      # Check the specific condition's site files exist in the flat
      # data directory (rather than checking for a subfolder, which
      # doesn't exist -- see note above).
      if ! ls "${DATA_DIR}"/site_A_alpha${ALPHA}_gamma${GAMMA}.csv >/dev/null 2>&1; then
        echo "[skip] missing site files for alpha=$ALPHA gamma=$GAMMA in $DATA_DIR"
        continue
      fi

      echo "=== v2.5 [bestckpt_fix] alpha=$ALPHA gamma=$GAMMA seed=$SEED ==="
      python3 phase1_archetype_train_v25.py \
        --data_dir "$DATA_DIR" \
        --method fedadaptproto --alpha "$ALPHA" --gamma "$GAMMA" --seed "$SEED" \
        --local_epochs 1 \
        --warmup_rounds 10 --early_stop_patience 0 \
        --auto_k --k_min 2 --k_max 5 --k_warmup_epochs 5 \
        --output_dir "$OUT_DIR"
    done
  done
done

echo "Done: v2.5 auto-K grid, bestckpt_fix + local_epochs=1 (20 conditions x 3 seeds = 60 jobs)."
