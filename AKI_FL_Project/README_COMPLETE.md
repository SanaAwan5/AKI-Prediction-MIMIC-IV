# MIMIC-IV AKI → FLamby FL Pipeline

Complete pipeline to extract features from MIMIC-IV, create CSV with IID distribution across 10 centers, and run federated learning experiments.

---

## 📋 Overview

**3 Python scripts** → **1 CSV file** → **FLamby FL experiments**

1. **`extract_aki_features.py`** - Extract features from MIMIC-IV last encounters
2. **`dataset_2_simple.py`** - FLamby dataset adapter
3. **`run_flamby_experiments.py`** - Run FL experiments

---

## 🚀 Quick Start (3 Steps)

### STEP 1: Extract Features

```bash
# Edit extract_aki_features.py
# Line 20: MIMIC_DATA_PATH = '/path/to/mimic-iv/'

# Run extraction
python extract_aki_features.py
```

**Output:** `aki_features_iid.csv`
- Demographics: age, gender, admission type, insurance (8 features)
- Lab values: creatinine, BUN, electrolytes, etc. (~60 features)
- **AKI_label**: Binary (0 or 1) based on KDIGO criteria
- **center_id**: IID distribution across 10 centers (0-9)

**Runtime:** 10-30 minutes

---

### STEP 2: Setup FLamby Dataset

```bash
# Copy dataset adapter
cp dataset_2_simple.py FLamby/flamby/datasets/fed_heart_disease/dataset_2.py

# Edit dataset_2.py
# Line 15: DATA_PATH = "/path/to/aki_features_iid.csv"

# Verify it works
cd FLamby
python -m flamby.datasets.fed_heart_disease.dataset_2
```

**Expected output:**
```
✓ Dataset found: 10 centers, 68 features
[train] center_0  :   XXX samples,  68 features, AKI=XX.XX%
[test ] center_0  :    XX samples,  68 features, AKI=XX.XX%
...
✓ VERIFICATION PASSED
```

---

### STEP 3: Run FL Experiments

```bash
# Edit run_flamby_experiments.py
# Lines 15-16:
#   CSV_FILE = '/path/to/aki_features_iid.csv'
#   FLAMBY_ROOT = '/path/to/FLamby'

# Run experiments
python run_flamby_experiments.py
```

**Menu options:**
1. Quick test (~5-10 min)
2. Full benchmark (~30-60 min)
3. Run single strategy
4. Exit

---

## 📊 Detailed Instructions

### Part 1: Feature Extraction

#### What Gets Extracted

**From your Colab notebook pipeline:**
- **Last encounter per patient** (one admission per patient)
- **First 48 hours** of lab measurements
- **KDIGO AKI criteria** applied

**Demographics (8 features):**
```python
- age
- gender_male (1=M, 0=F)
- admission_emergency (1/0)
- admission_urgent (1/0)
- admission_elective (1/0)
- insurance_medicare (1/0)
- insurance_medicaid (1/0)
- insurance_other (1/0)
```

**Lab Features (~60 features):**
For each lab type (creatinine, BUN, potassium, sodium, etc.):
```python
- {lab}_first   # First measurement
- {lab}_last    # Last measurement
- {lab}_min     # Minimum value
- {lab}_max     # Maximum value
- {lab}_mean    # Mean value
```

**Lab types extracted:**
- Creatinine (for AKI definition)
- BUN (Blood Urea Nitrogen)
- Potassium, Sodium, Chloride, Bicarbonate
- Platelet, WBC, Hemoglobin, Hematocrit
- Anion gap, Glucose

**AKI Definition (KDIGO):**
- sCr increase ≥0.3 mg/dL within 48 hours, OR
- sCr increase ≥1.5× baseline within 7 days

**IID Distribution:**
- Patients randomly assigned to 10 centers
- Each center gets ~10% of patients
- AKI prevalence balanced across centers
- Coefficient of variation < 0.05 (highly uniform)

#### CSV Structure

```csv
age,gender_male,admission_emergency,...,creatinine_first,creatinine_last,...,AKI_label,center_id
65.3,1,1,...,1.2,1.5,...,0,3
72.1,0,0,...,2.1,3.2,...,1,7
...
```

**Columns:**
- Features: ~68 columns
- AKI_label: 1 column (binary: 0 or 1)
- center_id: 1 column (0-9)

---

### Part 2: Dataset Integration

#### dataset_2.py Features

**Auto-detection:**
```python
# Automatically detects from CSV:
NUM_CLIENTS = 10  # from max(center_id) + 1
num_features = 68  # from column count
```

**Data preprocessing:**
- Missing values → filled with median
- Z-score normalization
- 80/20 train-test split per center
- Reproducible splits (fixed seed)

**Neural Network:**
```
Input (68 features)
  ↓
Layer 1: 128 neurons + BatchNorm + ReLU + Dropout(0.3)
  ↓
Layer 2: 64 neurons + BatchNorm + ReLU + Dropout(0.3)
  ↓
Layer 3: 32 neurons + ReLU
  ↓
Output: 1 neuron + Sigmoid
```

**Evaluation:** AUC-ROC score

#### Verification

```bash
cd FLamby
python -m flamby.datasets.fed_heart_disease.dataset_2
```

This will:
1. Load data for all 10 centers
2. Test model creation
3. Run forward pass
4. Compute loss and AUC

---

### Part 3: Running Experiments

#### Available FL Strategies

1. **FedAvg** - Standard federated averaging
2. **FedProx** - With proximal term (mu=0.01)
3. **Scaffold** - With control variates

#### Quick Test (Recommended First)

```bash
python run_flamby_experiments.py
# Choose option 1: Quick test
```

This runs FedAvg for 50 local updates to verify everything works.

**Expected output:**
```
Test       Metric
Test0      0.7234
Test1      0.7156
...
Pooled     0.7189

Mean AUC: 0.7189
```

#### Full Benchmark

```bash
python run_flamby_experiments.py
# Choose option 2: Full benchmark
```

This runs all strategies (FedAvg, FedProx, Scaffold) with full training.

**Results saved to:** `FLamby/aki_results/full_benchmark.csv`

#### Single Strategy

```bash
python run_flamby_experiments.py
# Choose option 3: Run single strategy
# Enter: FedAvg (or FedProx, Scaffold)
```

---

## 📈 Understanding Results

### Result Files

All results saved to: `FLamby/aki_results/`

**Files created:**
- `quick_test.csv` - Quick test results
- `full_benchmark.csv` - All strategies
- `fedavg.csv`, `fedprox.csv`, `scaffold.csv` - Individual strategies

### CSV Format

```csv
Test,Strategy,Metric,Seed
Test0,FedAvg,0.7234,42
Test1,FedAvg,0.7156,42
...
Pooled,FedAvg,0.7189,42
```

**Columns:**
- **Test**: Which center's test set (Test0-Test9, Pooled)
- **Strategy**: FL algorithm (FedAvg, FedProx, Scaffold)
- **Metric**: AUC-ROC score
- **Seed**: Random seed used

### Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_csv('FLamby/aki_results/full_benchmark.csv')

# Summary by strategy
summary = df.groupby('Strategy')['Metric'].agg(['mean', 'std'])
print(summary)

# Best strategy
best = summary['mean'].idxmax()
print(f"Best: {best}")
```

### Expected Performance

**Typical AUC-ROC scores:**
- Centralized (pooled): 0.75-0.85
- FedAvg: 0.72-0.80
- FedProx: 0.73-0.82
- Scaffold: 0.74-0.83

**Factors affecting performance:**
- Dataset quality
- AKI prevalence (~10-20% typical)
- Sample size per center
- Feature engineering

---

## 🔧 Customization

### Change Number of Centers

**In `extract_aki_features.py`:**
```python
class Config:
    NUM_CENTERS = 5  # Change from 10 to 5
```

**In `run_flamby_experiments.py`:**
```python
NUM_CENTERS = 5  # Change from 10 to 5
```

Re-run both scripts.

### Adjust Hyperparameters

**In `run_flamby_experiments.py`:**
```python
BATCH_SIZE = 64  # Increase
LEARNING_RATE = 0.01  # Increase
NUM_EPOCHS = 200  # More epochs
```

**Or modify config directly:**
```python
config = {
    "strategies": {
        "FedAvg": {
            "learning_rate": 0.01,
            "num_updates": 200,
            "nrounds": 500
        }
    }
}
```

### Modify Neural Network

**In `dataset_2_simple.py`, edit `Baseline` class:**
```python
# Larger model
self.fc1 = torch.nn.Linear(input_dim, 256)
self.fc2 = torch.nn.Linear(256, 128)
self.fc3 = torch.nn.Linear(128, 64)
self.fc4 = torch.nn.Linear(64, 32)
self.fc5 = torch.nn.Linear(32, 1)
```

---

## 🐛 Troubleshooting

### Issue: "CSV file not found"

**Solution:**
```bash
# Check path
ls /path/to/aki_features_iid.csv

# Use absolute path in dataset_2.py
DATA_PATH = "/absolute/path/to/aki_features_iid.csv"
```

### Issue: "MIMIC-IV files not found"

**Solution:**
```python
# In extract_aki_features.py
MIMIC_DATA_PATH = '/absolute/path/to/mimic-iv/'

# Verify structure:
# mimic-iv/
#   └── core/
#       ├── admissions.csv.gz
#       └── patients.csv.gz
#   └── hosp/
#       └── labevents.csv.gz
```

### Issue: Low AUC (<0.6)

**Solutions:**
1. Check data balance:
   ```python
   df = pd.read_csv('aki_features_iid.csv')
   print(df['AKI_label'].value_counts())
   ```

2. Increase model size (see Customization)

3. Adjust learning rate:
   ```python
   LEARNING_RATE = 0.01  # Try higher
   ```

4. More training:
   ```python
   'nrounds': 500  # Instead of 300
   ```

### Issue: "CUDA out of memory"

**Solution:**
```python
# In run_flamby_experiments.py
BATCH_SIZE = 16  # Reduce from 32
```

### Issue: Slow training

**Solutions:**
1. Reduce rounds: `'nrounds': 150`
2. Reduce updates: `'num_updates': 50`
3. GPU will be used automatically if available

---

## 📝 Complete Example Workflow

```bash
# ============================================
# STEP 1: Extract Features
# ============================================

# Edit extract_aki_features.py (line 20)
nano extract_aki_features.py
# Set: MIMIC_DATA_PATH = '/data/mimic-iv/'

# Run extraction
python extract_aki_features.py
# Output: aki_features_iid.csv (10-30 min)

# Verify CSV
head aki_features_iid.csv
wc -l aki_features_iid.csv

# ============================================
# STEP 2: Setup FLamby
# ============================================

# Copy dataset file
cp dataset_2_simple.py FLamby/flamby/datasets/fed_heart_disease/dataset_2.py

# Edit DATA_PATH (line 15)
nano FLamby/flamby/datasets/fed_heart_disease/dataset_2.py
# Set: DATA_PATH = "/data/aki_features_iid.csv"

# Verify
cd FLamby
python -m flamby.datasets.fed_heart_disease.dataset_2
# Should see: ✓ VERIFICATION PASSED

# ============================================
# STEP 3: Run Experiments
# ============================================

cd ..

# Edit paths (lines 15-16)
nano run_flamby_experiments.py
# Set paths to CSV and FLamby

# Run experiments
python run_flamby_experiments.py
# Choose: 1 (Quick test)
# Wait 5-10 minutes

# If successful, run full benchmark
python run_flamby_experiments.py
# Choose: 2 (Full benchmark)
# Wait 30-60 minutes

# ============================================
# STEP 4: Check Results
# ============================================

# View results
cat FLamby/aki_results/full_benchmark.csv

# Or analyze in Python
python
>>> import pandas as pd
>>> df = pd.read_csv('FLamby/aki_results/full_benchmark.csv')
>>> print(df.groupby('Strategy')['Metric'].describe())
```

---

## 📚 File Descriptions

### extract_aki_features.py
- **Purpose:** Extract features from MIMIC-IV
- **Input:** MIMIC-IV CSV files (admissions, patients, labevents)
- **Output:** `aki_features_iid.csv`
- **Key function:** `main()` - runs full pipeline
- **Runtime:** 10-30 minutes

### dataset_2_simple.py
- **Purpose:** FLamby dataset adapter
- **Input:** `aki_features_iid.csv`
- **Output:** PyTorch datasets for FL
- **Key classes:**
  - `FedHeartDisease` - Dataset class
  - `Baseline` - Neural network model
  - `BaselineLoss` - BCE loss
- **Key function:** `metric()` - AUC-ROC calculation

### run_flamby_experiments.py
- **Purpose:** Run FL experiments
- **Input:** CSV file + FLamby installation
- **Output:** Result CSV files in `aki_results/`
- **Key functions:**
  - `verify_setup()` - Check prerequisites
  - `run_quick_test()` - Fast verification
  - `run_full_benchmark()` - All strategies

---

## 🎯 Expected Timeline

| Task | Time | Output |
|------|------|--------|
| Feature extraction | 10-30 min | CSV file (~68 features) |
| Dataset verification | 1-2 min | Verification passed |
| Quick test | 5-10 min | Initial AUC scores |
| Full benchmark | 30-60 min | All strategies tested |

---

## ✅ Checklist

- [ ] Updated `MIMIC_DATA_PATH` in extract_aki_features.py
- [ ] Ran `python extract_aki_features.py`
- [ ] Got `aki_features_iid.csv` file
- [ ] Copied `dataset_2_simple.py` to FLamby
- [ ] Updated `DATA_PATH` in dataset_2.py
- [ ] Verified with `python -m flamby.datasets.fed_heart_disease.dataset_2`
- [ ] Updated paths in `run_flamby_experiments.py`
- [ ] Ran quick test successfully
- [ ] Ran full benchmark
- [ ] Analyzed results

---

## 🆘 Getting Help

**Debug checklist:**
1. All paths are absolute (not relative)
2. MIMIC-IV files are readable
3. CSV has AKI_label and center_id columns
4. Python packages installed (pandas, numpy, torch, sklearn)

**View logs:**
```bash
python extract_aki_features.py 2>&1 | tee extract.log
python -m flamby.datasets.fed_heart_disease.dataset_2 2>&1 | tee verify.log
```

---

**Ready? Start here:**

```bash
# Quick verification
python --version  # Should be 3.7+
pip list | grep -E "torch|pandas|numpy|sklearn"

# Begin extraction
python extract_aki_features.py
```

Good luck! 🚀
