"""
IID Distribution Script for MIMIC-IV AKI Dataset

Assigns patients to hypothetical centers using IID (Independent and Identically Distributed) approach.

Input:  aki_features_patient_level.csv (all center_id = 0)
Output: aki_features_iid.csv (center_id = 0-9, uniformly distributed)
"""

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_CSV = '/Users/awans/Documents/GitHub/AKI_FL_Project/aki_features_patient_level.csv'
OUTPUT_CSV = '/Users/awans/Documents/GitHub/AKI_FL_Project/aki_features_iid.csv'
NUM_CENTERS = 10
RANDOM_SEED = 42

print("="*80)
print("IID DISTRIBUTION - ASSIGN CENTER IDs")
print("="*80)

# Load
print(f"\n[1/4] Loading {INPUT_CSV}...")
if not os.path.exists(INPUT_CSV):
    print(f"❌ ERROR: File not found: {INPUT_CSV}")
    exit(1)

df = pd.read_csv(INPUT_CSV)
print(f"  ✓ Loaded {len(df):,} patients")

# Assign
print(f"\n[2/4] Assigning to {NUM_CENTERS} centers...")
np.random.seed(RANDOM_SEED)
df['center_id'] = np.random.randint(0, NUM_CENTERS, size=len(df))
#df = df.drop(columns=['aki_criterion', 'max_scr_during_admission'])
leakage_cols = ['aki_criterion', 'max_scr_during_admission', 'aki_detected']
existing_leakage = [c for c in leakage_cols if c in df.columns]
if existing_leakage:
    df = df.drop(columns=existing_leakage)
    print(f"  ✓ Removed leakage columns: {existing_leakage}")
else:
    print(f"  ✓ No leakage columns found (already clean)")

# Verify
print(f"\n[3/4] Distribution:")
print(f"{'Center':<10} {'Patients':<12} {'%':<10} {'AKI %':<10}")
print("-" * 45)

for center_id in range(NUM_CENTERS):
    center_data = df[df['center_id'] == center_id]
    count = len(center_data)
    pct = count / len(df) * 100
    aki_pct = center_data['AKI_label'].mean() * 100
    print(f"{center_id:<10} {count:<12,} {pct:>6.2f}%    {aki_pct:>6.2f}%")

# Save
print(f"\n[4/4] Saving {OUTPUT_CSV}...")
df.to_csv(OUTPUT_CSV, index=False)
print(f"  ✓ Saved ({os.path.getsize(OUTPUT_CSV)/1024/1024:.2f} MB)")

print("\n" + "="*80)
print("✓ COMPLETE! Next: Update DATA_PATH in dataset_2.py")
print("="*80)