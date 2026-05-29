#!/bin/bash
# ============================================================================
# run_simulation.sh — Full FL pipeline: simulate then train
# Run from the folder containing all scripts and aki_features_iid.csv
#
# Condition → simulation parameters:
#   IID:            no_label_shift, gamma=0.0  → all sites ~12.6% AKI
#   Covariate only: no_label_shift, gamma=0.75 → features shifted, labels ~12.6%
#   Label only:     alpha=0.5, gamma=0.0       → A~43%, B~17%, D~13%, E~5%
#   Both:           alpha=0.5, gamma=0.75      → label + covariate shift
#   Label extreme:  alpha=0.3, gamma=0.0       → higher label variance
#   Both extreme:   alpha=0.3, gamma=0.75
#   Label max:      alpha=0.1, gamma=0.0       → maximum label variance
#   Both max:       alpha=0.1, gamma=1.0       → maximum heterogeneity
# ============================================================================

DATA_DIR="${1:-./simulated_sites}"
OUTPUT_DIR="${2:-./fl_results_v2}"
INPUT_CSV="aki_features_iid.csv"
LABEL="AKI_label"
N_SEEDS=20
ROUNDS=100
LOCAL_EPOCHS=10
LR=0.01
MU=0.1
CV_CLIP=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_SCRIPT="$SCRIPT_DIR/mimic_ftl_simulation.py"
TRAIN_SCRIPT="$SCRIPT_DIR/mimic_ftl_training.py"

echo "============================================"
echo " FL Pipeline"
echo "============================================"
echo " Input CSV:    $INPUT_CSV"
echo " Data dir:     $DATA_DIR"
echo " Output dir:   $OUTPUT_DIR"
echo " Seeds:        $N_SEEDS"
echo " Rounds:       $ROUNDS"
echo " Local epochs: $LOCAL_EPOCHS"
echo " LR:           $LR  mu=$MU  cv_clip=$CV_CLIP"
echo "============================================"

# ── Check scripts exist ───────────────────────────────────────────────────
for f in "$SIM_SCRIPT" "$TRAIN_SCRIPT"; do
    [ ! -f "$f" ] && echo "ERROR: Missing: $f" && exit 1
done
[ ! -f "$INPUT_CSV" ] && echo "ERROR: '$INPUT_CSV' not found in $(pwd)" && exit 1

# ── Step 1: Simulate all conditions ──────────────────────────────────────
echo ""
echo "STEP 1: Simulating federated sites..."
echo ""

mkdir -p "$DATA_DIR"

sim() {
    local alpha=$1; local gamma=$2; local extra=$3; local label=$4
    echo "  [$label]  alpha=$alpha  gamma=$gamma  $extra"
    python3 "$SIM_SCRIPT" \
        --input "$INPUT_CSV" --label "$LABEL" \
        --alpha $alpha --gamma $gamma \
        --n_site 31041 --output "$DATA_DIR" $extra \
        || { echo "ERROR: Simulation failed ($label)"; exit 1; }
}

sim 0.5 0.0  "--no_label_shift"  "IID"
sim 0.5 0.75 "--no_label_shift"  "Covariate only"
sim 0.5 0.0  ""                  "Label only"
sim 0.5 0.75 ""                  "Both"
sim 0.3 0.0  ""                  "Label extreme"
sim 0.3 0.75 ""                  "Both extreme"
sim 0.1 0.0  ""                  "Label max"
sim 0.1 1.0  ""                  "Both max"

echo ""
echo "Simulation complete: $(ls "$DATA_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ') CSVs in $DATA_DIR"

# ── Step 2: FL training ───────────────────────────────────────────────────
echo ""
echo "STEP 2: Running FL training ($N_SEEDS seeds, $ROUNDS rounds, $LOCAL_EPOCHS local epochs)..."
echo "Estimated time: 1-2 hours on CPU"
echo ""

mkdir -p "$OUTPUT_DIR"

# Remove stale checkpoint if it exists so we get a clean run
if [ -f "$OUTPUT_DIR/fl_results_v2.json" ]; then
    echo "  [info] Found existing checkpoint — resuming from it."
    echo "  [info] To force a full re-run: rm $OUTPUT_DIR/fl_results_v2.json"
    echo ""
fi

python3 "$TRAIN_SCRIPT" \
    --data_dir     "$DATA_DIR" \
    --label        "$LABEL" \
    --seeds        $N_SEEDS \
    --rounds       $ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --lr           $LR \
    --mu           $MU \
    --cv_clip      $CV_CLIP \
    --h1 64 --h2 32 \
    --output       "$OUTPUT_DIR" \
    || { echo "ERROR: Training failed"; exit 1; }

echo ""
echo "============================================"
echo " DONE — outputs in: $OUTPUT_DIR"
echo "============================================"
echo "  fl_results_v2.json"
echo "  results_all_conditions_ci.png"
echo "  fairness_per_site.png"
echo "  scaffold_stability_comparison.png"
echo "  alpha_comparison.png"
echo ""