#!/bin/bash
# Full 20-condition x 3-seed grid, v2.3-based methods (fedadaptproto, fedavg,
# fedprox, scaffold, fedadapt), on the disjoint-sites-corrected
# phase1_data_disjoint/ (patients disjoint across sites; cohort updated
# after the KDIGO baseline-SCr/CKD-exclusion fix -- see run_complete.md).
# 5 methods x 20 conditions x 3 seeds = 300 jobs.
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
METHODS=(fedadaptproto fedavg fedprox scaffold fedadapt)

DATA_DIR="./phase1_data_disjoint"
OUT_ROOT="./results_phase1_grid_v23"

for METHOD in "${METHODS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for GAMMA in "${GAMMAS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${OUT_ROOT}/alpha${ALPHA}_gamma${GAMMA}_seed${SEED}/"

        # [RESUME] Skip any job whose output already exists from an
        # earlier, interrupted run of this same script -- makes the whole
        # grid safely restartable after a crash/interruption without
        # redoing already-completed jobs. Checks for the final CSV each
        # job produces, one specific to this exact method/condition/seed.
        if [ -f "${OUT_DIR}${METHOD}/fl_gain_correlation.csv" ]; then
          echo "[resume-skip] already completed: method=$METHOD alpha=$ALPHA gamma=$GAMMA seed=$SEED"
          continue
        fi

        # Check the specific condition's site files exist in the flat
        # data directory (rather than checking for a subfolder, which
        # doesn't exist -- see note above).
        if ! ls "${DATA_DIR}"/site_A_alpha${ALPHA}_gamma${GAMMA}.csv >/dev/null 2>&1; then
          echo "[skip] missing site files for alpha=$ALPHA gamma=$GAMMA in $DATA_DIR"
          continue
        fi

        EXTRA_ARGS=""
        if [ "$METHOD" == "fedadaptproto" ]; then
          EXTRA_ARGS="--n_clusters 2"
        fi

        echo "=== method=$METHOD alpha=$ALPHA gamma=$GAMMA seed=$SEED ==="
        python3 phase1_archetype_train_v23.py \
          --data_dir "$DATA_DIR" \
          --alpha "$ALPHA" --gamma "$GAMMA" --seed "$SEED" \
          --method "$METHOD" $EXTRA_ARGS \
          --embedding_dim 64 --hidden_dim 128 \
          --output_dir "$OUT_DIR"
      done
    done
  done
done

echo "Done: v2.3-based grid (5 methods x 20 conditions x 3 seeds = 300 jobs)."
