#!/bin/bash
# =============================================================================
# run_phase2.sh
# FedAdapt Phase II — simulation + training pipeline
#
# WHAT THIS SCRIPT DOES
# ─────────────────────
# Step 1: Run Phase II simulation (all alpha/gamma combinations via --sweep)
#         Produces site CSVs + adapter metadata JSONs in ./phase2_sites/
#         Also produces class_distribution_by_alpha.png at the end of the sweep.
#
# Step 2: Run federated training for all 4 methods across all alpha/gamma
#         combinations. Results saved to ./results_phase2/
#
# BEFORE RUNNING
# ──────────────
# 1. Install dependencies:
#       pip install torch scikit-learn scipy matplotlib pandas numpy
#
# 2. Set your MIMIC-IV CSV path below (MIMIC_CSV variable).
#
# 3. Make sure these files are in the same folder as this script:
#       mimic_ftl_simulation_phase2.py
#       fedadapt_model.py
#       fedadapt_train.py
#
# 4. Make this script executable:
#       chmod +x run_phase2.sh
#
# USAGE
# ─────
#   ./run_phase2.sh               # full pipeline (sim + all training)
#   ./run_phase2.sh --sim-only    # simulation only (no training)
#   ./run_phase2.sh --train-only  # training only (simulation already done)
#   ./run_phase2.sh --test        # quick smoke-test: 1 combo, 1 method, 5 rounds
# =============================================================================

set -euo pipefail

# ── USER CONFIG — edit this ───────────────────────────────────────────────────
MIMIC_CSV="./aki_features_iid.csv"     # path to your MIMIC-IV AKI CSV
LABEL_COL="AKI_label"                  # label column in the CSV
SIM_DIR="./phase2_sites"               # where simulation outputs go
RESULTS_DIR="./results_phase2"         # where training results go
PYTHON="${PYTHON:-python3}"            # override with: PYTHON=python ./run_phase2.sh
ROUNDS=50
LOCAL_EPOCHS=5
EMBEDDING_DIM=64
# ─────────────────────────────────────────────────────────────────────────────

ALPHAS="0.1 0.3 0.5 1.0 10.0"
GAMMAS="0.0 0.5 0.75 1.0"
METHODS="fedavg fedprox scaffold fedadapt"

# ── Parse flags ───────────────────────────────────────────────────────────────
RUN_SIM=true
RUN_TRAIN=true
TEST_MODE=false

for arg in "$@"; do
    case $arg in
        --sim-only)   RUN_TRAIN=false ;;
        --train-only) RUN_SIM=false   ;;
        --test)       TEST_MODE=true  ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log()       { echo "[$(date '+%H:%M:%S')] $*"; }
separator() { echo "============================================================"; }

separator
log "FedAdapt Phase II Pipeline"
log "CSV:     $MIMIC_CSV"
log "Sim dir: $SIM_DIR"
log "Results: $RESULTS_DIR"
separator

# ── Step 1: Simulation ────────────────────────────────────────────────────────
if $RUN_SIM; then
    if [ ! -f "$MIMIC_CSV" ]; then
        echo "[ERROR] CSV not found: $MIMIC_CSV"
        echo "        Edit MIMIC_CSV at the top of this script."
        exit 1
    fi

    log "Starting Phase II simulation (--sweep) ..."
    log "This generates all alpha/gamma site partitions AND the"
    log "class_distribution_by_alpha.png chart in $SIM_DIR/"

    if $TEST_MODE; then
        $PYTHON mimic_ftl_simulation_phase2.py \
            --input         "$MIMIC_CSV" \
            --label         "$LABEL_COL" \
            --alpha         0.5 \
            --gamma         0.75 \
            --embedding_dim $EMBEDDING_DIM \
            --output        "$SIM_DIR"
    else
        $PYTHON mimic_ftl_simulation_phase2.py \
            --input         "$MIMIC_CSV" \
            --label         "$LABEL_COL" \
            --embedding_dim $EMBEDDING_DIM \
            --output        "$SIM_DIR" \
            --sweep
    fi

    log "Simulation complete. Chart saved to $SIM_DIR/class_distribution_by_alpha.png"
    separator
fi

# ── Step 2: Training ──────────────────────────────────────────────────────────
if $RUN_TRAIN; then
    if $TEST_MODE; then
        ALPHAS="0.5"; GAMMAS="0.75"; METHODS="fedadapt"
        ROUNDS=5; LOCAL_EPOCHS=2
        log "TEST MODE: 1 combo × 1 method × $ROUNDS rounds"
    fi

    TOTAL=$(echo $ALPHAS | wc -w)
    TOTAL=$((TOTAL * $(echo $GAMMAS | wc -w) * $(echo $METHODS | wc -w)))
    log "Starting training: $(echo $ALPHAS | wc -w) alphas × $(echo $GAMMAS | wc -w) gammas × $(echo $METHODS | wc -w) methods = $TOTAL runs"
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

                if [ -f "$DONE_FLAG" ]; then
                    log "[skip] α=$ALPHA γ=$GAMMA $METHOD — already done"
                    DONE=$((DONE + 1))
                    continue
                fi

                log "Running α=$ALPHA  γ=$GAMMA  method=$METHOD ..."
                mkdir -p "$OUT_DIR"

                if $PYTHON fedadapt_train.py \
                        --data_dir   "$DATA_DIR" \
                        --method     "$METHOD" \
                        --alpha      "$ALPHA" \
                        --gamma      "$GAMMA" \
                        --rounds     "$ROUNDS" \
                        --local_epochs "$LOCAL_EPOCHS" \
                        --output_dir "$OUT_DIR" 2>&1 | tee "${OUT_DIR}/train.log"; then
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
