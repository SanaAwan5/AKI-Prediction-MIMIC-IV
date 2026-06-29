#!/bin/bash
# =============================================================================
# run_phase2_approach2.sh
# FedAdapt Phase II — Approach 2 (anchor-based, one row per patient)
#
# WHAT THIS SCRIPT DOES
# ─────────────────────
# Step 1: Run Phase II simulation (all alpha/gamma combinations via --sweep)
#         Produces site CSVs + adapter metadata JSONs in ./phase2_sites_approach2/
#         Also produces class_distribution_by_alpha.png at the end of the sweep.
#
# Step 2: Run federated training for all 4 methods across all alpha/gamma
#         combinations. Results saved to ./results_phase2_approach2/
#
# APPROACH 2 CHANGES vs original run_phase2.sh
# ─────────────────────────────────────────────
# 1. Uses mimic_ftl_simulation_phase2_approach2.py (prefix-based features,
#    train-split-aware sampling, 5 sites — site_F removed)
# 2. Uses fedadapt_train_approach2.py (prefix feature matching, no unlabeled)
# 3. MIMIC_CSV points to anchor-based preprocessing output
#    (aki_anchor_based_48h_lookback.csv — 48h lead time, primary experiment)
# 4. python3 used explicitly throughout
# 5. Separate output directories to avoid overwriting Phase II v1 results
#
# BEFORE RUNNING
# ──────────────
# 1. Install dependencies:
#       pip install torch scikit-learn scipy matplotlib pandas numpy
#
# 2. Set your CSV path below (MIMIC_CSV variable).
#    Default: aki_anchor_based_48h_lookback.csv (48h lead time)
#    For 24h experiment: aki_anchor_based_24h_lookback.csv
#
# 3. Make sure these files are in the same folder as this script:
#       mimic_ftl_simulation_phase2_approach2.py
#       fedadapt_model_approach2.py
#       fedadapt_train_approach2.py
#
# 4. Make this script executable:
#       chmod +x run_phase2_approach2.sh
#
# USAGE
# ─────
#   ./run_phase2_approach2.sh                    # full pipeline (sim + training, all 4 methods)
#   ./run_phase2_approach2.sh --sim-only          # simulation only
#   ./run_phase2_approach2.sh --train-only        # training only (sim already done)
#   ./run_phase2_approach2.sh --test              # smoke-test: 1 combo, 5 rounds
#   ./run_phase2_approach2.sh --fedadapt-only          # only run FedAdapt (GRL)
#   ./run_phase2_approach2.sh --fedadaptproto-only     # only run FedAdapt-Proto (GRL + prototype alignment)
#   ./run_phase2_approach2.sh --fedadaptproto-only --alpha_proto=1.0   # custom prototype loss weight
#   ./run_phase2_approach2.sh --fedadaptproto-only --embedding_dim=128 # larger embedding dim
#   ./run_phase2_approach2.sh --fedadaptproto-only --alpha_proto=1.0 --embedding_dim=128  # combine both
# =============================================================================

set -euo pipefail

# ── USER CONFIG — edit this ───────────────────────────────────────────────────
MIMIC_CSV="./aki_anchor_based_48h_lookback.csv"   # Approach 2 anchor-based CSV
LABEL_COL="AKI_label"
SIM_DIR="./phase2_sites_approach2"                     # simulation outputs
RESULTS_DIR="./results_phase2_approach2"               # training results
ROUNDS=50
LOCAL_EPOCHS=5
EMBEDDING_DIM=64
# ─────────────────────────────────────────────────────────────────────────────

ALPHAS="0.1 0.3 0.5 1.0 10.0"
GAMMAS="0.0 0.5 0.75 1.0"
METHODS="fedavg fedprox scaffold fedadapt fedadaptproto"

# ── Parse flags ───────────────────────────────────────────────────────────────
RUN_SIM=true
RUN_TRAIN=true
TEST_MODE=false

FORCE=false
ALPHA_PROTO=0.5
for arg in "$@"; do
    case $arg in
        --sim-only)   RUN_TRAIN=false ;;
        --train-only) RUN_SIM=false   ;;
        --test)       TEST_MODE=true  ;;
        --force)      FORCE=true      ;;
        --fedadapt-only)      METHODS="fedadapt" ;;
        --fedadaptproto-only) METHODS="fedadaptproto" ;;
        --alpha_proto=*)     ALPHA_PROTO="${arg#*=}" ;;
        --embedding_dim=*)   EMBEDDING_DIM="${arg#*=}" ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log()       { echo "[$(date '+%H:%M:%S')] $*"; }
separator() { echo "============================================================"; }

separator
log "FedAdapt Phase II — Approach 2 Pipeline"
log "CSV:     $MIMIC_CSV"
log "Sim dir: $SIM_DIR"
log "Results: $RESULTS_DIR"
log "Sites:   A B C D E  (site_F removed)"
log "Force:   $FORCE  (--force to re-run completed jobs)"
separator

# ── Step 1: Simulation ────────────────────────────────────────────────────────
if $RUN_SIM; then
    if [ ! -f "$MIMIC_CSV" ]; then
        echo "[ERROR] CSV not found: $MIMIC_CSV"
        echo "        Edit MIMIC_CSV at the top of this script."
        exit 1
    fi

    log "Starting Phase II Approach 2 simulation (--sweep) ..."
    log "Generates all alpha/gamma site partitions in $SIM_DIR/"

    if $TEST_MODE; then
        python3 mimic_ftl_simulation_phase2_approach2.py \
            --input         "$MIMIC_CSV" \
            --label         "$LABEL_COL" \
            --alpha         0.5 \
            --gamma         0.75 \
            --embedding_dim $EMBEDDING_DIM \
            --output        "$SIM_DIR/alpha0.5_gamma0.75"
    else
        python3 mimic_ftl_simulation_phase2_approach2.py \
            --input         "$MIMIC_CSV" \
            --label         "$LABEL_COL" \
            --embedding_dim $EMBEDDING_DIM \
            --output        "$SIM_DIR" \
            --sweep
    fi

    log "Simulation complete. Outputs in $SIM_DIR/"
    separator
fi

# ── Step 2: Training ──────────────────────────────────────────────────────────
if $RUN_TRAIN; then
    if $TEST_MODE; then
        ALPHAS="0.5"; GAMMAS="0.75"
        ROUNDS=5; LOCAL_EPOCHS=2
        log "TEST MODE: 1 combo × method=$METHODS × $ROUNDS rounds"
    fi

    N_ALPHAS=$(echo $ALPHAS | wc -w)
    N_GAMMAS=$(echo $GAMMAS | wc -w)
    N_METHODS=$(echo $METHODS | wc -w)
    TOTAL=$((N_ALPHAS * N_GAMMAS * N_METHODS))

    log "Starting training: $N_ALPHAS alphas × $N_GAMMAS gammas × $N_METHODS methods = $TOTAL runs"
    separator

    DONE=0
    FAILED=0

    for ALPHA in $ALPHAS; do
        for GAMMA in $GAMMAS; do
            DATA_DIR="$SIM_DIR/alpha${ALPHA}_gamma${GAMMA}"
            if [ ! -d "$DATA_DIR" ]; then
                log "[SKIP] $DATA_DIR not found — run simulation first"
                continue
            fi

            for METHOD in $METHODS; do
                OUT_DIR="$RESULTS_DIR/alpha${ALPHA}_gamma${GAMMA}/${METHOD}"
                DONE_FLAG="${OUT_DIR}/.done"

                if [ -f "$DONE_FLAG" ] && [ "$FORCE" = "false" ]; then
                    log "[skip] α=$ALPHA γ=$GAMMA $METHOD — already done"
                    DONE=$((DONE + 1))
                    continue
                fi

                log "Running α=$ALPHA  γ=$GAMMA  method=$METHOD ..."
                mkdir -p "$OUT_DIR"

                if python3 fedadapt_train_approach2.py \
                        --data_dir     "$DATA_DIR" \
                        --method       "$METHOD" \
                        --alpha        "$ALPHA" \
                        --gamma        "$GAMMA" \
                        --rounds       "$ROUNDS" \
                        --local_epochs "$LOCAL_EPOCHS" \
                        --embedding_dim "$EMBEDDING_DIM" \
                        --alpha_proto  "$ALPHA_PROTO" \
                        --output_dir   "$OUT_DIR" \
                        2>&1 | tee "${OUT_DIR}/train.log"; then
                    touch "$DONE_FLAG"
                    DONE=$((DONE + 1))
                    log "  ✅ Done ($DONE/$TOTAL)"
                else
                    FAILED=$((FAILED + 1))
                    log "  ❌ FAILED — see ${OUT_DIR}/train.log"
                fi
            done
        done
    done

    separator
    log "Training complete.  ✅ $DONE succeeded   ❌ $FAILED failed"
    log "Results in: $RESULTS_DIR/"
    separator
fi