#!/bin/bash
# run_v24_5seeds.sh
# Runs fedadapt_train_approach2_v2_4.py 5 times with seeds 42,43,44,45,46
# then aggregates mean +/- std across sites.
#
# Usage:
#   bash run_v24_5seeds.sh
#
# Output:
#   results_v24_seed*/  — per-seed result dirs
#   results_v24_5seeds_summary.csv  — mean +/- std table

SCRIPT="fedadapt_train_approach2_v2_4.py"
DATA_DIR="./phase2_sites_approach2/alpha0.3_gamma0.75"
SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
    echo "============================================"
    echo "  Running seed $SEED"
    echo "============================================"
    python3 $SCRIPT \
        --data_dir $DATA_DIR \
        --method fedadaptproto \
        --alpha 0.3 --gamma 0.75 \
        --warmup_rounds 10 --early_stop_patience 0 \
        --auto_k --k_min 2 --k_max 5 \
        --output_dir ./results_v24_seed${SEED}/ \
        --seed $SEED
    echo "Seed $SEED done."
done

echo ""
echo "All seeds complete. Aggregating results..."

python3 - << 'PYEOF'
import os, csv, numpy as np

seeds = [42, 43, 44, 45, 46]
all_results = {}   # {site_id: {metric: [values]}}

for seed in seeds:
    fpath = f"./results_v24_seed{seed}/fedadaptproto/final_metrics.csv"
    if not os.path.exists(fpath):
        print(f"WARNING: {fpath} not found — skipping seed {seed}")
        continue
    with open(fpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["site_id"].split("_alpha")[0]  # strip alpha suffix
            if sid not in all_results:
                all_results[sid] = {
                    "auroc": [], "f1": [], "auprc": [],
                    "delta_auroc": [], "selected_k": []
                }
            all_results[sid]["auroc"].append(float(row["auroc"]))
            all_results[sid]["f1"].append(float(row["f1"]))
            all_results[sid]["auprc"].append(float(row["auprc"]))
            all_results[sid]["delta_auroc"].append(float(row["delta_auroc"]))
            all_results[sid]["selected_k"].append(int(row["selected_k"]))

print("\n" + "="*75)
print(f"  FedAdaptProto v2.4 — 5-seed summary (seeds {seeds})")
print("="*75)
print(f"{'Site':<12} {'AUROC':>18} {'F1':>14} {'AUPRC':>14} {'Delta':>14} {'K'}")
print("-"*75)

summary_rows = []
for sid in sorted(all_results.keys()):
    r = all_results[sid]
    n = len(r["auroc"])
    if n == 0:
        continue
    auroc_m, auroc_s = np.mean(r["auroc"]), np.std(r["auroc"])
    f1_m,    f1_s    = np.mean(r["f1"]),    np.std(r["f1"])
    auprc_m, auprc_s = np.mean(r["auprc"]), np.std(r["auprc"])
    delta_m, delta_s = np.mean(r["delta_auroc"]), np.std(r["delta_auroc"])
    k_vals = r["selected_k"]
    k_str = str(k_vals[0]) if len(set(k_vals)) == 1 else f"{min(k_vals)}-{max(k_vals)}"

    sign = "+" if delta_m >= 0 else ""
    print(f"{sid:<12} {auroc_m:.4f}±{auroc_s:.4f}  "
          f"{f1_m:.4f}±{f1_s:.4f}  "
          f"{auprc_m:.4f}±{auprc_s:.4f}  "
          f"{sign}{delta_m:.4f}±{delta_s:.4f}  K={k_str}")

    summary_rows.append({
        "site_id": sid, "n_seeds": n,
        "auroc_mean": round(auroc_m, 6), "auroc_std": round(auroc_s, 6),
        "f1_mean":    round(f1_m, 6),    "f1_std":    round(f1_s, 6),
        "auprc_mean": round(auprc_m, 6), "auprc_std": round(auprc_s, 6),
        "delta_mean": round(delta_m, 6), "delta_std": round(delta_s, 6),
        "selected_k": k_str,
    })

print("="*75)

# Save summary CSV
out = "results_v24_5seeds_summary.csv"
if summary_rows:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary saved → {out}")
PYEOF