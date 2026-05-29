"""


Usage
─────
# Full experiment (5 seeds, all conditions, all baselines):
python mimic_ftl_training_v2.py \
    --data_dir ./simulated_sites \
    --label AKI_label \
    --seeds 5 \
    --rounds 100 \
    --local_epochs 5 \
    --cv_clip 1.0 \
    --output ./fl_results_v2

# Quick test (1 seed):
python mimic_ftl_training_v2.py \
    --data_dir ./simulated_sites \
    --label AKI_label \
    --seeds 1 \
    --rounds 100 \
    --output ./fl_results_v2

References
──────────
McMahan et al. (2017) FedAvg. AISTATS.
Li et al. (2020) FedProx. MLSys.
Karimireddy et al. (2020) SCAFFOLD. ICML.
Zhang et al. (2025) TACO. arXiv:2504.17528.
"""

from __future__ import annotations

import argparse
import json
import warnings
from scipy import stats as scipy_stats
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────

TEST_FRAC    = 0.20
L2_REG       = 1e-4
GRAD_CLIP    = 5.0    # max gradient norm — prevents exploding gradients on large sites

ALGO_COLORS = {
    "fedavg":          "#2563EB",
    "fedprox":         "#D97706",
    "scaffold":        "#16A34A",
    "scaffold_clip":   "#059669",
    "local":           "#9333EA",
    "centralised":     "#DC2626",
}
ALGO_LABELS = {
    "fedavg":          "FedAvg",
    "fedprox":         "FedProx",
    "scaffold":        "SCAFFOLD",
    "scaffold_clip":   "SCAFFOLD+clip",
    "local":           "Local only",
    "centralised":     "Centralised",
}

FL_ALGOS = ["fedavg", "fedprox", "scaffold", "scaffold_clip"]

# 2×2 + extended conditions
# Condition definitions — alpha, gamma, label_shift
#
# KEY INSIGHT: In the Dirichlet formula used here,
#   conc = max(alpha * 20 * bias_norm, 5 * bias_norm)
#   E[proportion] = bias_norm  ALWAYS, regardless of alpha
#   Alpha only controls VARIANCE around that mean (higher alpha = tighter)
#
# Therefore:
#   - IID requires no_label_shift=True + gamma=0.0, NOT a high alpha value
#     (no_label_shift disables bias vectors entirely -> all sites ~12.6% AKI)
#   - Alpha controls how tightly sites cluster around their TARGET AKI rate:
#       alpha=0.3: high variance -> sites can deviate far from target (extreme)
#       alpha=0.5: moderate variance -> sites reliably near target (standard)
#   - The target rates are set by DIRICHLET_BIAS in the simulation script:
#       site_A ~40% AKI, site_B ~17%, site_D ~10%, site_E ~5%
#
# Condition → expected AKI rates per site:
#   IID:            all sites ~12.6%  (anchor rate, uniform sampling)
#   Covariate only: all sites ~12.6%  (no label bias, but features shifted)
#   Label only:     A=40%, B=17%, C=12.6%, D=10%, E=5%  (alpha=0.5)
#   Both:           A=40%, B=17%, C=12.6%, D=10%, E=5%  + feature shift
#   Label extreme:  A=40%, B=17%, C=12.6%, D=10%, E=5%  + higher variance (alpha=0.3)
#   Both extreme:   same + feature shift
# Condition design — two axes of heterogeneity, independently controlled:
#
#   Label shift axis  — controlled by alpha + label_shift flag:
#     no_label_shift=True          : uniform sampling, all sites ~12.6% AKI (true IID)
#     label_shift=True, alpha=0.5  : sites near clinical targets, moderate variance
#     label_shift=True, alpha=0.3  : same targets, higher variance (more extreme)
#
#   Covariate shift axis — controlled by gamma:
#     gamma=0.0  : no acuity skew, no spread perturbation
#     gamma=0.75 : subpopulation sampling + spread perturbation active
#
# Expected AKI rates per condition:
#   IID            : all sites ~12.6%  (uniform sampling, no label bias)
#   Covariate only : all sites ~12.6%  (no label bias, features shifted)
#   Label only     : A~40%, B~17%, C~12.6%, D~10%, E~5%  (alpha=0.5)
#   Both           : A~40%, B~17%, C~12.6%, D~10%, E~5%  + feature shift
#   Label extreme  : same targets as Label only, higher cross-run variance (alpha=0.3)
#   Both extreme   : same targets + feature shift, higher variance
CONDITIONS = {
    # Standard 2x2 conditions
    "IID":              {"label_shift": False, "gamma": 0.0,  "alpha": 0.5},
    "Covariate only":   {"label_shift": False, "gamma": 0.75, "alpha": 0.5},
    "Label only":       {"label_shift": True,  "gamma": 0.0,  "alpha": 0.5},
    "Both":             {"label_shift": True,  "gamma": 0.75, "alpha": 0.5},
    # Extreme conditions — alpha=0.3 (higher label variance)
    "Label extreme":    {"label_shift": True,  "gamma": 0.0,  "alpha": 0.3},
    "Both extreme":     {"label_shift": True,  "gamma": 0.75, "alpha": 0.3},
    # Maximum heterogeneity — alpha=0.1, gamma=1.0 (stress test)
    "Label max":        {"label_shift": True,  "gamma": 0.0,  "alpha": 0.1},
    "Both max":         {"label_shift": True,  "gamma": 1.0,  "alpha": 0.1},
}

# ─── Data structures ──────────────────────────────────────────────────────────

class SiteData:
    def __init__(self, site_id, features, X_train, X_test, y_train, y_test):
        self.site_id  = site_id
        self.features = features
        self.X_train  = X_train
        self.X_test   = X_test
        self.y_train  = y_train
        self.y_test   = y_test
        self.n_train  = len(y_train)
        self.n_test   = len(y_test)

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_condition(data_dir: Path, label_col: str, gamma: float,
                   label_shift: bool, alpha: float,
                   all_features: List[str],
                   rng: np.random.Generator) -> Optional[Dict[str, SiteData]]:
    """
    Load site CSVs for a given (gamma, label_shift, alpha) condition.

    Imputation is strictly LOCAL — only each site's own observed data is used:
      - Within-column NaNs: filled with that site's own training-set column median
      - Entirely missing columns (feature masking): filled with MASK_VALUE (-3.0)
        in post-standardisation z-score space, distinguishable from real data
      - No anchor/global statistics are accessed or shared

    MASK_VALUE = -3.0: three z-score units below the site mean. This is
    well outside the normal measurement range for any lab value, so the
    model learns to treat it as "not measured here" rather than "measured
    and average". Better than 0.0 (= site mean) which is ambiguous.

    Returns None if no matching CSV files found.
    """
    import re as _re

    # ── Find matching CSV files ───────────────────────────────────────────────
    # Simulation produces filenames like:
    #   site_A_alpha0.5_gamma0.0_no_label_shift.csv  (no label shift)
    #   site_A_alpha0.5_gamma0.0.csv                 (label shift, no suffix)
    # We must match BOTH alpha AND gamma AND label_shift tag precisely.

    def _matches(fname: str) -> bool:
        """Return True if filename matches this condition exactly."""
        # Must contain alpha value
        if f"alpha{alpha}" not in fname:
            return False
        # Must contain gamma value (try both 1 and 2 decimal representations)
        gamma_strs = [f"gamma{gamma}", f"gamma{gamma:.1f}", f"gamma{gamma:.2f}"]
        if not any(g in fname for g in gamma_strs):
            return False
        # Must match label_shift status
        has_no_label = "no_label" in fname
        if label_shift and has_no_label:
            return False       # want label shift but file has no_label tag
        if not label_shift and not has_no_label:
            return False       # want no label shift but file has no tag
        return True

    candidates = [
        f for f in data_dir.glob("*.csv")
        if _matches(f.name)
    ]

    if not candidates:
        return None

    MASK_VALUE = -3.0   # z-score sentinel for "feature not measured at this site"

    site_data = {}
    for csv_path in candidates:
        stem = csv_path.stem
        m = _re.search(r'(site_[A-Za-z0-9]+)', stem)
        site_id = m.group(1) if m else stem.split("_")[0]

        df = pd.read_csv(csv_path)
        if label_col not in df.columns:
            continue

        y         = df[label_col].values.astype(float)
        feat_cols = [c for c in df.columns if c != label_col]
        feat_set  = set(feat_cols)

        # ── Stratified train/test split ───────────────────────────────────────
        # Stratify by label so AKI rate is preserved exactly in train and test.
        # Removes label-proportion variance as a source of seed-to-seed noise.
        n     = len(df)
        y_raw = df[label_col].fillna(-1).values.astype(int)

        aki_idx   = np.where(y_raw == 1)[0]
        noaki_idx = np.where(y_raw == 0)[0]
        nan_idx   = np.where(y_raw == -1)[0]   # unlabeled (site_F)

        rng.shuffle(aki_idx)
        rng.shuffle(noaki_idx)

        n_test_aki   = max(1, int(len(aki_idx)   * TEST_FRAC))
        n_test_noaki = max(1, int(len(noaki_idx) * TEST_FRAC))

        test_idx  = np.concatenate([aki_idx[:n_test_aki],
                                    noaki_idx[:n_test_noaki],
                                    nan_idx])
        train_idx = np.concatenate([aki_idx[n_test_aki:],
                                    noaki_idx[n_test_noaki:]])

        # ── Build feature matrix using only locally observed columns ──────────
        # Columns the site actually has: standardise using site's own train stats
        # Columns the site does NOT have: fill with MASK_VALUE after standardisation
        X_obs = np.full((n, len(all_features)), np.nan, dtype=np.float64)

        for j, feat in enumerate(all_features):
            if feat in feat_set:
                raw = df[feat]
                # Categorical columns: encode locally using integer codes.
                # Each site encodes independently from its own observed categories —
                # no shared vocabulary, no cross-site information.
                # Binary/ordinal categoricals (gender, admission_type, etc.) become
                # integer codes 0,1,2,... which are then z-scored like any numeric.
                if not pd.api.types.is_numeric_dtype(raw):
                    raw = raw.astype("category").cat.codes.astype(float)
                    raw = raw.where(raw >= 0, other=np.nan)  # -1 codes → NaN
                col = raw.values.astype(float)
                X_obs[:, j] = col   # NaNs within observed cols handled below

        # Standardise observed columns using site's OWN training-set mean/std
        # (computed only from rows in train_idx — no data leakage to test set)
        X_tr_raw = X_obs[train_idx].copy()
        X_te_raw = X_obs[test_idx].copy()

        for j, feat in enumerate(all_features):
            if feat not in feat_set:
                # Feature entirely absent at this site — use mask sentinel
                X_tr_raw[:, j] = MASK_VALUE
                X_te_raw[:, j] = MASK_VALUE
                continue

            col_tr = X_tr_raw[:, j]
            # Within-column NaN: fill with site's own training median (local only)
            finite_mask = np.isfinite(col_tr)
            if finite_mask.any():
                local_median = float(np.median(col_tr[finite_mask]))
                local_mean   = float(np.mean(col_tr[finite_mask]))
                local_std    = float(np.std(col_tr[finite_mask]))
            else:
                # All values NaN for this feature — treat as missing
                X_tr_raw[:, j] = MASK_VALUE
                X_te_raw[:, j] = MASK_VALUE
                continue

            # Fill within-column NaNs with local training median
            col_tr = np.where(np.isfinite(col_tr), col_tr, local_median)
            col_te = X_te_raw[:, j].copy()
            col_te = np.where(np.isfinite(col_te), col_te, local_median)

            # Z-score using local training mean/std
            if local_std > 1e-8:
                col_tr = (col_tr - local_mean) / local_std
                col_te = (col_te - local_mean) / local_std
            else:
                col_tr = np.zeros_like(col_tr)
                col_te = np.zeros_like(col_te)

            X_tr_raw[:, j] = col_tr
            X_te_raw[:, j] = col_te

        X_tr = X_tr_raw.astype(np.float32)
        X_te = X_te_raw.astype(np.float32)

        # Final safety: clip extreme values (±10 SD) to prevent overflow
        X_tr = np.clip(X_tr, -10.0, 10.0)
        X_te = np.clip(X_te, -10.0, 10.0)

        # ── Labels ────────────────────────────────────────────────────────────
        y_tr = y[train_idx].astype(float)
        y_te = y[test_idx].astype(float)
        # Drop NaN-labelled rows from training (site_F is unlabeled)
        train_mask = ~np.isnan(y_tr)

        # Skip fully-unlabeled sites (e.g. site_F) — 0 labelled training rows
        if train_mask.sum() == 0:
            print(f"    [skip] {site_id}: no labelled training rows (unlabeled site)")
            continue

        site_data[site_id] = SiteData(
            site_id=site_id,
            features=feat_cols,
            X_train=X_tr[train_mask], X_test=X_te,
            y_train=y_tr[train_mask], y_test=y_te,
        )

    return site_data if site_data else None


# ─── MLP (numpy, no torch dependency) ────────────────────────────────────────

def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def relu(x):
    return np.maximum(0.0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

class MLP:
    """
    2-hidden-layer MLP: input → H1 (ReLU) → H2 (ReLU) → 1 (sigmoid).
    Weights stored as flat numpy array for easy aggregation.
    """
    def __init__(self, n_in: int, h1: int = 64, h2: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / n_in)
        scale2 = np.sqrt(2.0 / h1)
        scale3 = np.sqrt(2.0 / h2)
        self.shapes = [
            (n_in, h1), (h1,),
            (h1,  h2), (h2,),
            (h2,  1),  (1,),
        ]
        self.W1 = rng.normal(0, scale1, (n_in, h1)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (h1,  h2)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = rng.normal(0, scale3, (h2,  1)).astype(np.float32)
        self.b3 = np.zeros(1,  dtype=np.float32)

    def forward(self, X):
        self._z1 = X @ self.W1 + self.b1
        self._a1 = relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = relu(self._z2)
        self._z3 = self._a2 @ self.W3 + self.b3
        self._out = sigmoid(self._z3).ravel()
        return self._out

    def predict(self, X):
        return self.forward(X)

    def loss_and_grad(self, X, y, mu=0.0, w_global=None, l2=L2_REG):
        """BCE + optional FedProx proximal + L2."""
        p   = self.forward(X)
        eps = 1e-7
        bce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        # Backprop
        n   = len(y)
        d3  = (p - y).reshape(-1, 1) / n          # (n,1)
        dW3 = self._a2.T @ d3
        db3 = d3.sum(axis=0)

        d2  = (d3 @ self.W3.T) * relu_grad(self._z2)
        dW2 = self._a1.T @ d2
        db2 = d2.sum(axis=0)

        d1  = (d2 @ self.W2.T) * relu_grad(self._z1)
        dW1 = X.T @ d1
        db1 = d1.sum(axis=0)

        # L2
        dW1 += l2 * self.W1;  dW2 += l2 * self.W2;  dW3 += l2 * self.W3

        # FedProx proximal term
        if mu > 0.0 and w_global is not None:
            w_local = self.to_flat()
            diff    = w_local - w_global
            bce    += (mu / 2.0) * np.dot(diff, diff)
            # Add proximal gradient to each param
            offset = 0
            for arr, grad in [
                (self.W1, dW1), (self.b1, db1),
                (self.W2, dW2), (self.b2, db2),
                (self.W3, dW3), (self.b3, db3),
            ]:
                sz = arr.size
                grad += mu * diff[offset:offset+sz].reshape(arr.shape)
                offset += sz

        return bce, [dW1, db1, dW2, db2, dW3, db3]

    def to_flat(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel(),
            self.W3.ravel(), self.b3.ravel(),
        ])

    def from_flat(self, w: np.ndarray):
        idx = 0
        for arr in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]:
            sz = arr.size
            arr[:] = w[idx:idx+sz].reshape(arr.shape)
            idx += sz

    def param_list(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def flat_size(self) -> int:
        return sum(a.size for a in self.param_list())


# ─── SGD local training step ─────────────────────────────────────────────────

def sgd_epoch(model: MLP, X: np.ndarray, y: np.ndarray,
              lr: float, mu: float = 0.0,
              w_global: Optional[np.ndarray] = None,
              batch_size: int = 256) -> float:
    """One pass through local data with mini-batch SGD + gradient clipping."""
    n   = len(y)
    idx = np.random.permutation(n)
    total_loss = 0.0
    n_batches  = 0
    params = model.param_list()

    for start in range(0, n, batch_size):
        b_idx = idx[start:start + batch_size]
        Xb, yb = X[b_idx], y[b_idx]

        # Skip batch if inputs contain NaN (unimputed missing features)
        if not np.isfinite(Xb).all():
            Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)

        loss, grads = model.loss_and_grad(Xb, yb, mu=mu, w_global=w_global)

        # Skip batch if loss exploded
        if not np.isfinite(loss):
            continue

        # Global gradient clipping — prevents exploding gradients on large sites
        global_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if global_norm > GRAD_CLIP:
            clip_coef = GRAD_CLIP / (global_norm + 1e-6)
            grads = [g * clip_coef for g in grads]

        for p, g in zip(params, grads):
            p -= lr * g

        total_loss += loss
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─── FL algorithms ────────────────────────────────────────────────────────────

def fed_aggregate(updates: Dict[str, np.ndarray],
                  weights: Dict[str, float]) -> np.ndarray:
    """Weighted average of flat parameter vectors."""
    total = sum(weights.values())
    agg   = np.zeros_like(next(iter(updates.values())), dtype=np.float64)
    for sid, w in updates.items():
        agg += (weights[sid] / total) * w.astype(np.float64)
    return agg.astype(np.float32)


def run_fl(
    algo: str,
    site_data: Dict[str, SiteData],
    n_in: int,
    rounds: int,
    local_epochs: int,
    lr: float,
    mu: float = 0.1,
    h1: int = 64, h2: int = 32,
    cv_clip: Optional[float] = None,
    seed: int = 42,
) -> Tuple[Dict[str, float], List[float]]:
    """
    Unified runner for FedAvg / FedProx / SCAFFOLD / SCAFFOLD+clip.

    algo options: "fedavg", "fedprox", "scaffold", "scaffold_clip"
    cv_clip: max norm of per-site CV update (SCAFFOLD stability fix)

    Returns:
        per_site_auroc: {site_id: auroc}
        macro_curve:    list of macro AUROC per round
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    use_scaffold = algo in ("scaffold", "scaffold_clip")
    use_clip     = (algo == "scaffold_clip") and (cv_clip is not None)
    use_prox     = (algo == "fedprox")

    # Initialise global model
    global_model = MLP(n_in, h1=h1, h2=h2, seed=seed)
    global_w     = global_model.to_flat()
    d            = len(global_w)

    # SCAFFOLD: server and per-site control variates
    server_cv = np.zeros(d, dtype=np.float32)
    site_cvs  = {sid: np.zeros(d, dtype=np.float32) for sid in site_data}

    # Training weights proportional to training set size
    train_weights = {sid: sd.n_train for sid, sd in site_data.items()}

    macro_curve = []

    # Learning rate schedule: linear warmup then cosine decay
    # Warmup prevents large early updates from poor initialisation.
    # Cosine decay ensures smooth convergence in later rounds.
    WARMUP_ROUNDS = min(10, rounds // 10)

    def get_lr(r):
        if r < WARMUP_ROUNDS:
            return lr * (r + 1) / WARMUP_ROUNDS          # linear warmup
        progress = (r - WARMUP_ROUNDS) / max(1, rounds - WARMUP_ROUNDS)
        return lr * 0.5 * (1 + np.cos(np.pi * progress)) # cosine decay

    for r in range(rounds):
        round_lr       = get_lr(r)
        local_updates  = {}
        delta_cvs      = {}

        for sid, sd in site_data.items():
            # Copy global weights to local model
            local_model = MLP(n_in, h1=h1, h2=h2, seed=seed)
            local_model.from_flat(global_w.copy())
            w_before = local_model.to_flat().copy()

            w_global_for_prox = global_w.copy() if use_prox else None

            for _ in range(local_epochs):
                if use_scaffold:
                    # SCAFFOLD: modify gradient by (site_cv - server_cv)
                    cv_correction = site_cvs[sid] - server_cv
                    # One SGD epoch; then apply CV correction post-update
                    sgd_epoch(local_model, sd.X_train, sd.y_train, lr=round_lr)
                    # Apply control variate correction
                    w_new = local_model.to_flat() - round_lr * cv_correction
                    local_model.from_flat(w_new)
                else:
                    sgd_epoch(local_model, sd.X_train, sd.y_train,
                              lr=round_lr, mu=mu if use_prox else 0.0,
                              w_global=w_global_for_prox)

            w_after = local_model.to_flat()
            local_updates[sid] = w_after

            # SCAFFOLD: compute delta_cv (Option II update)
            if use_scaffold:
                K = local_epochs
                delta_cv = (
                    site_cvs[sid]
                    - server_cv
                    - (w_after - w_before) / (K * round_lr)
                )
                # CV clipping for stability (scaffold_clip variant)
                if use_clip:
                    norm = np.linalg.norm(delta_cv)
                    if norm > cv_clip:
                        delta_cv = delta_cv * (cv_clip / norm)

                delta_cvs[sid] = delta_cv

        # Aggregate
        global_w = fed_aggregate(local_updates, train_weights)
        global_model.from_flat(global_w)

        # SCAFFOLD: update server and site CVs
        if use_scaffold:
            n_sites = len(site_data)
            for sid in site_data:
                site_cvs[sid] = site_cvs[sid] - delta_cvs[sid]
            agg_delta = np.mean(list(delta_cvs.values()), axis=0)
            server_cv = server_cv - agg_delta

        # Evaluate global AUROC every 5 rounds — pool all sites' test data
        if (r + 1) % 5 == 0 or r == rounds - 1:
            y_pool, p_pool = [], []
            for sid, sd in site_data.items():
                mask = ~np.isnan(sd.y_test)
                if mask.sum() == 0:
                    continue
                p = global_model.predict(sd.X_test)
                if not np.isfinite(p).all():
                    continue
                y_pool.append(sd.y_test[mask])
                p_pool.append(p[mask])
            if y_pool:
                y_all = np.concatenate(y_pool)
                p_all = np.concatenate(p_pool)
                if len(np.unique(y_all)) >= 2:
                    macro_curve.append(float(roc_auc_score(y_all, p_all)))
                else:
                    macro_curve.append(0.5)
            else:
                macro_curve.append(0.5)

    # Final evaluation — per-site AUROC + pooled global AUROC
    per_site = {}
    y_pool, p_pool = [], []

    for sid, sd in site_data.items():
        mask = ~np.isnan(sd.y_test)
        y_valid = sd.y_test[mask]
        if len(np.unique(y_valid)) < 2:
            per_site[sid] = float("nan")
            continue
        p = global_model.predict(sd.X_test)
        if not np.isfinite(p).all():
            per_site[sid] = float("nan")
            continue
        per_site[sid] = float(roc_auc_score(y_valid, p[mask]))
        y_pool.append(y_valid)
        p_pool.append(p[mask])

    # Pooled global AUROC — global model on aggregated test data
    if y_pool:
        y_all = np.concatenate(y_pool)
        p_all = np.concatenate(p_pool)
        per_site["_global"] = (
            float(roc_auc_score(y_all, p_all))
            if len(np.unique(y_all)) >= 2 else float("nan")
        )
    else:
        per_site["_global"] = float("nan")

    return per_site, macro_curve


def run_local(site_data: Dict[str, SiteData],
              n_in: int,
              local_epochs: int,
              lr: float,
              rounds: int,
              h1: int = 64, h2: int = 32,
              seed: int = 42) -> Dict[str, float]:
    """
    Each site trains entirely on its own data — no federation.

    Two design choices matching FL literature (Li et al. 2020 FedProx):
      1. Training budget = rounds * local_epochs (same as FL total local steps)
      2. Local features only — site trains and evaluates using only its own
         observed feature columns, not the full padded feature space.
         Missing features are not visible to the local model at all.
         This is the fair comparison: local model has less data breadth,
         FL global model benefits from all sites' feature distributions.
    """
    np.random.seed(seed)
    per_site = {}
    total_epochs = rounds * local_epochs   # match FL total local training steps

    for sid, sd in site_data.items():
        # Identify which columns are real observations vs mask sentinel (-3.0)
        # A column is "real" if its training values are not all exactly MASK_VALUE
        MASK_VALUE = -3.0
        real_cols = []
        for j in range(sd.X_train.shape[1]):
            col = sd.X_train[:, j]
            if not np.all(col == MASK_VALUE):
                real_cols.append(j)

        if not real_cols:
            per_site[sid] = float("nan")
            continue

        # Train and evaluate on local features only
        real_cols = np.array(real_cols)
        X_tr_local = sd.X_train[:, real_cols]
        X_te_local = sd.X_test[:,  real_cols]
        n_in_local = len(real_cols)

        local_model = MLP(n_in_local, h1=min(h1, n_in_local * 2),
                          h2=min(h2, n_in_local), seed=seed)
        for _ in range(total_epochs):
            sgd_epoch(local_model, X_tr_local, sd.y_train, lr=lr)

        y_valid = sd.y_test[~np.isnan(sd.y_test)]
        if len(np.unique(y_valid)) < 2:
            per_site[sid] = float("nan")
            continue

        p = local_model.predict(X_te_local)
        if not np.isfinite(p).all():
            per_site[sid] = float("nan")
            continue

        per_site[sid] = float(roc_auc_score(
            y_valid, p[~np.isnan(sd.y_test)]
        ))

    return per_site


def run_centralised(site_data: Dict[str, SiteData],
                    n_in: int,
                    local_epochs: int,
                    lr: float,
                    h1: int = 64, h2: int = 32,
                    seed: int = 42) -> Dict[str, float]:
    """Pool all training data, evaluate per site — upper bound."""
    np.random.seed(seed)
    X_all = np.concatenate([sd.X_train for sd in site_data.values()])
    y_all = np.concatenate([sd.y_train for sd in site_data.values()])

    model = MLP(n_in, h1=h1, h2=h2, seed=seed)
    for _ in range(local_epochs * 20):
        sgd_epoch(model, X_all, y_all, lr=lr)

    per_site = {}
    for sid, sd in site_data.items():
        if len(np.unique(sd.y_test)) < 2:
            per_site[sid] = float("nan")
        else:
            p = model.predict(sd.X_test)
            per_site[sid] = float(roc_auc_score(sd.y_test, p))
    return per_site


# ─── Multi-seed runner ────────────────────────────────────────────────────────

def run_multi_seed(
    algo: str,
    site_data_fn,          # callable(seed) -> Dict[str,SiteData]
    n_in: int,
    seeds: List[int],
    rounds: int,
    local_epochs: int,
    lr: float,
    mu: float = 0.1,
    h1: int = 64, h2: int = 32,
    cv_clip: Optional[float] = None,
) -> Dict:
    """
    Run an algorithm across multiple seeds, return mean ± std per site and macro.
    """
    all_per_site = []
    all_macro_curve = []

    for seed in seeds:
        site_data = site_data_fn(seed)
        if site_data is None:
            print(f"    [skip] no data for seed {seed}")
            continue

        if algo in ("fedavg", "fedprox", "scaffold", "scaffold_clip"):
            per_site, macro_curve = run_fl(
                algo, site_data, n_in, rounds, local_epochs,
                lr, mu=mu, h1=h1, h2=h2, cv_clip=cv_clip, seed=seed,
            )
        elif algo == "local":
            per_site = run_local(site_data, n_in, local_epochs, lr,
                                   rounds=rounds, h1=h1, h2=h2, seed=seed)
            macro_curve = []
        elif algo == "centralised":
            per_site = run_centralised(site_data, n_in, local_epochs, lr, h1, h2, seed)
            macro_curve = []
        else:
            raise ValueError(f"Unknown algo: {algo}")

        all_per_site.append(per_site)
        all_macro_curve.append(macro_curve)

    if not all_per_site:
        return {}

    # Aggregate across seeds
    all_sites = [k for k in all_per_site[0].keys() if not k.startswith("_")]
    result = {}

    for sid in all_sites:
        vals = [ps[sid] for ps in all_per_site if not np.isnan(ps.get(sid, np.nan))]
        result[sid] = {
            "mean": float(np.mean(vals)) if vals else np.nan,
            "std":  float(np.std(vals))  if len(vals) > 1 else 0.0,
            "runs": len(vals),
        }

    # Use pooled global AUROC (_global key) as the primary macro metric
    # Falls back to mean of per-site AUROCs if _global not available
    seed_macros = []
    for ps in all_per_site:
        if "_global" in ps and not np.isnan(ps["_global"]):
            seed_macros.append(float(ps["_global"]))
        else:
            vals = [ps[sid] for sid in all_sites
                    if sid in ps and not np.isnan(ps.get(sid, np.nan))]
            if vals:
                seed_macros.append(float(np.nanmean(vals)))

    result["_macro"] = {
        "mean":   float(np.nanmean(seed_macros)) if seed_macros else float("nan"),
        "std":    float(np.std(seed_macros)) if len(seed_macros) > 1 else 0.0,
        "runs":   len(all_per_site),
        "values": seed_macros,   # per-seed values for paired t-tests
    }

    # Store mean macro curve for FL algos
    if all_macro_curve and all_macro_curve[0]:
        min_len = min(len(c) for c in all_macro_curve)
        curves  = np.array([c[:min_len] for c in all_macro_curve])
        result["_curve_mean"] = curves.mean(axis=0).tolist()
        result["_curve_std"]  = curves.std(axis=0).tolist()

    return result


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results_with_ci(all_results: Dict, output_dir: Path):
    """
    Main results figure: macro AUROC with ±1 std error bars across all
    conditions and algorithms, including baselines.
    """
    conditions = list(CONDITIONS.keys())
    algos_to_plot = FL_ALGOS + ["local", "centralised"]

    fig, ax = plt.subplots(figsize=(14, 6))

    n_cond  = len(conditions)
    n_algo  = len(algos_to_plot)
    width   = 0.12
    x       = np.arange(n_cond)

    for i, algo in enumerate(algos_to_plot):
        means, stds = [], []
        for cond in conditions:
            res = all_results.get(cond, {}).get(algo, {})
            macro = res.get("_macro", {})
            means.append(macro.get("mean", np.nan))
            stds.append(macro.get("std",  0.0))

        offset = (i - n_algo / 2 + 0.5) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=ALGO_COLORS[algo], label=ALGO_LABELS[algo],
               alpha=0.85, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", color="black", capsize=3,
                    linewidth=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in conditions], fontsize=10)
    ax.set_ylabel("Macro AUROC", fontsize=12)
    ax.set_title("FL Algorithm Comparison — All Conditions (mean ± std, multiple seeds)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.70, 0.95)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    plt.tight_layout()
    path = output_dir / "results_all_conditions_ci.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_fairness_panel(all_results: Dict, output_dir: Path):
    """
    Fairness panel: per-site AUROC across conditions for each FL algorithm.
    Highlights site_E degradation.
    """
    conditions = ["IID", "Covariate only", "Label only", "Both",
                  "Both extreme"]
    algos = FL_ALGOS

    fig, axes = plt.subplots(1, len(algos), figsize=(5 * len(algos), 5),
                             sharey=True)

    site_colors = {
        "site_A": "#EF4444",
        "site_B": "#3B82F6",
        "site_C": "#10B981",
        "site_D": "#F59E0B",
        "site_E": "#8B5CF6",
    }

    for ax, algo in zip(axes, algos):
        for sid, color in site_colors.items():
            means, stds = [], []
            for cond in conditions:
                res  = all_results.get(cond, {}).get(algo, {})
                site = res.get(sid, {})
                means.append(site.get("mean", np.nan))
                stds.append(site.get("std", 0.0))

            x = np.arange(len(conditions))
            lw = 2.5 if sid == "site_E" else 1.2
            ls = "--" if sid == "site_E" else "-"
            ax.plot(x, means, color=color, linewidth=lw, linestyle=ls,
                    marker="o", markersize=5, label=sid)
            if any(s > 0 for s in stds):
                ax.fill_between(x,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color, alpha=0.15)

        ax.set_title(ALGO_LABELS[algo], fontweight="bold", fontsize=11)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c.replace(" ", "\n") for c in conditions],
                           fontsize=8)
        ax.set_ylim(0.70, 0.95)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("AUROC", fontsize=12)
    axes[-1].legend(fontsize=9, title="Site", bbox_to_anchor=(1.02, 0.5),
                    loc="center left")
    fig.suptitle(
        "Per-Site AUROC Across Conditions — site_E (dashed) is fairness watchpoint",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = output_dir / "fairness_per_site.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_scaffold_comparison(all_results: Dict, output_dir: Path):
    """
    SCAFFOLD vs SCAFFOLD+clip curves in the Both condition.
    Shows the stability fix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cond in zip(axes, ["Both", "Both extreme"]):
        for algo in ["scaffold", "scaffold_clip", "fedavg", "fedprox"]:
            res = all_results.get(cond, {}).get(algo, {})
            curve_mean = res.get("_curve_mean", [])
            curve_std  = res.get("_curve_std",  [])
            if not curve_mean:
                continue
            x = np.arange(len(curve_mean))
            ax.plot(x, curve_mean, color=ALGO_COLORS[algo],
                    label=ALGO_LABELS[algo], linewidth=2)
            if curve_std and any(s > 0 for s in curve_std):
                ax.fill_between(x,
                    [m - s for m, s in zip(curve_mean, curve_std)],
                    [m + s for m, s in zip(curve_mean, curve_std)],
                    color=ALGO_COLORS[algo], alpha=0.15)

        ax.set_title(f"Training Curves — {cond}", fontweight="bold")
        ax.set_xlabel("Evaluation checkpoint (every 5 rounds)")
        ax.set_ylabel("Macro AUROC")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.75, 0.92)

    plt.tight_layout()
    path = output_dir / "scaffold_stability_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_alpha_comparison(all_results: Dict, output_dir: Path):
    """
    Compare alpha=0.5 vs alpha=0.3 — does sharper label imbalance
    widen algorithm gaps?
    """
    compare_pairs = [
        ("Label only", "Label extreme", "Label shift only"),
        ("Both",       "Both extreme",  "Label + Covariate shift"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (cond_05, cond_03, title) in zip(axes, compare_pairs):
        algos = FL_ALGOS
        x     = np.arange(len(algos))
        width = 0.35

        for offset, (cond, label_sfx, hatch) in enumerate([
            (cond_05, " (α=0.5)", ""),
            (cond_03, " (α=0.3)", "//"),
        ]):
            means = []
            stds  = []
            for algo in algos:
                res   = all_results.get(cond, {}).get(algo, {})
                macro = res.get("_macro", {})
                means.append(macro.get("mean", np.nan))
                stds.append(macro.get("std",  0.0))

            bars = ax.bar(x + (offset - 0.5) * width, means, width,
                          color=[ALGO_COLORS[a] for a in algos],
                          alpha=0.7 + 0.3 * (1 - offset),
                          hatch=hatch, label=f"α=0.{'5' if offset==0 else '3'}")
            ax.errorbar(x + (offset - 0.5) * width, means, yerr=stds,
                        fmt="none", color="black", capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in algos])
        ax.set_ylabel("Macro AUROC")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.80, 0.92)

    plt.suptitle("Effect of Dirichlet α on Algorithm Gaps — α=0.3 increases heterogeneity",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = output_dir / "alpha_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def print_results_table(all_results: Dict):
    """Print a publication-ready results table to stdout."""
    conditions = list(CONDITIONS.keys())
    algos = FL_ALGOS + ["local", "centralised"]

    print("\n" + "=" * 90)
    print("RESULTS TABLE — Global AUROC (pooled test set) mean ± std (N seeds per cell)")
    print("=" * 90)
    header = f"{'Condition':<20}" + "".join(f"  {ALGO_LABELS[a]:<16}" for a in algos)
    print(header)
    print("-" * 90)

    for cond in conditions:
        row = f"{cond:<20}"
        # Collect FedAvg seed values for paired t-test reference
        fa_vals = all_results.get(cond, {}).get("fedavg", {}).get("_macro", {}).get("values", [])
        for algo in algos:
            res   = all_results.get(cond, {}).get(algo, {})
            macro = res.get("_macro", {})
            m     = macro.get("mean", float("nan"))
            s     = macro.get("std",  0.0)
            n     = macro.get("runs", 0)
            vals  = macro.get("values", [])
            if np.isnan(m):
                row += f"  {'N/A':<18}"
            elif algo == "fedavg" or not fa_vals or not vals or len(fa_vals) != len(vals):
                row += f"  {m:.3f}±{s:.3f}(n={n})   "
            else:
                # Paired t-test vs FedAvg
                t, p = scipy_stats.ttest_rel(vals, fa_vals)
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
                row += f"  {m:.3f}±{s:.3f}(n={n}){sig} "
        print(row)
    print()
    print("  * p<0.05 vs FedAvg (paired t-test)  ** p<0.01")

    print("=" * 90)

    # Degradation from IID
    print("\nDEGRADATION FROM IID (FL algos only):")
    iid_macros = {}
    for algo in FL_ALGOS:
        res = all_results.get("IID", {}).get(algo, {})
        iid_macros[algo] = res.get("_macro", {}).get("mean", np.nan)

    for cond in conditions:
        if cond == "IID":
            continue
        row = f"  {cond:<20}"
        for algo in FL_ALGOS:
            res   = all_results.get(cond, {}).get(algo, {})
            macro = res.get("_macro", {}).get("mean", np.nan)
            drop  = iid_macros[algo] - macro if not np.isnan(macro) else np.nan
            row  += f"  {ALGO_LABELS[algo]}: {drop:+.3f}   "
        print(row)

    # site_E fairness
    print("\nSITE_E AUROC (fairness watchpoint):")
    for cond in conditions:
        row = f"  {cond:<20}"
        for algo in FL_ALGOS:
            res  = all_results.get(cond, {}).get(algo, {})
            site = res.get("site_E", {})
            m    = site.get("mean", np.nan)
            row += f"  {ALGO_LABELS[algo]}={m:.3f}  "
        print(row)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upgraded FL training: multi-seed, baselines, alpha sweep"
    )
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--label",        default="AKI_label")
    parser.add_argument("--seeds",        type=int, default=5,
                        help="Number of random seeds per condition")
    parser.add_argument("--rounds",       type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--mu",           type=float, default=0.1,
                        help="FedProx proximal coefficient")
    parser.add_argument("--cv_clip",      type=float, default=1.0,
                        help="CV clip norm for scaffold_clip (0=disable)")
    parser.add_argument("--h1",           type=int, default=64)
    parser.add_argument("--h2",           type=int, default=32)
    parser.add_argument("--output",       default="./fl_results_v2")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    seeds    = list(range(args.seeds))
    cv_clip  = args.cv_clip if args.cv_clip > 0 else None

    print(f"Data dir:  {data_dir}")
    print(f"Seeds:     {seeds}")
    print(f"Rounds:    {args.rounds}")
    print(f"CV clip:   {cv_clip}")

    # Build the union feature list from all CSVs in data_dir.
    # No global statistics are loaded — imputation uses only local site data.
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")

    # Union of all feature columns across all site CSVs (excluding label)
    # This defines the shared input dimension for the MLP.
    feat_union: dict = {}   # ordered, preserving first-seen order
    for csv_path in csvs:
        df0 = pd.read_csv(csv_path, nrows=0)   # header only — no data loaded
        for c in df0.columns:
            if c != args.label and c not in feat_union:
                feat_union[c] = True
    all_features = list(feat_union.keys())
    n_in         = len(all_features)
    print(f"Features:  {n_in}  (union across all site CSVs, no global stats loaded)")

    # All algorithms to run
    all_algos = FL_ALGOS + ["local", "centralised"]

    # ── Checkpoint: load any previously completed results ─────────────────────
    results_path = output_dir / "fl_results_v2.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results: Dict = json.load(f)
        completed = {
            cond: set(algos.keys())
            for cond, algos in all_results.items()
        }
        print(f"\n[checkpoint] Loaded {results_path}")
        for cond, algos in completed.items():
            print(f"  Already done: {cond} — {sorted(algos)}")
    else:
        all_results: Dict = {}
        completed: Dict = {}

    def _save_checkpoint():
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [checkpoint saved] {results_path}")

    for cond_name, cond_cfg in CONDITIONS.items():
        print(f"\n{'='*60}")
        label_shift_str = (
            "no_label_shift — all sites uniform ~12.6% AKI"
            if not cond_cfg["label_shift"]
            else f"label_shift=True, alpha={cond_cfg['alpha']}"
        )
        print(f"Condition: {cond_name}  "
              f"({label_shift_str}, gamma={cond_cfg['gamma']})")
        print(f"{'='*60}")
        # Use more rounds for harder conditions to ensure convergence
        cond_rounds = args.rounds * 2 if "max" in cond_name else args.rounds

        # Factory: returns site_data for a given seed
        DATA_SEED = 0   # Fixed across all algo seeds — isolates algorithm variance
        def make_site_data(seed, cfg=cond_cfg):
            # Data split uses a fixed seed so all algo seeds see the same
            # train/test partition. Variance across seeds then reflects only
            # weight initialisation and SGD stochasticity, not data sampling.
            rng = np.random.default_rng(DATA_SEED)
            return load_condition(
                data_dir, args.label,
                gamma=cfg["gamma"],
                label_shift=cfg["label_shift"],
                alpha=cfg["alpha"],
                all_features=all_features,
                rng=rng,
            )

        # Check at least one seed works
        test_data = make_site_data(0)
        if test_data is None:
            print(f"  [skip] No data files found for this condition")
            continue

        print(f"  Sites found: {sorted(test_data.keys())}")
        for sid, sd in sorted(test_data.items()):
            aki_rate = sd.y_train.mean() * 100
            print(f"    {sid}: train={sd.n_train}  test={sd.n_test}  "
                  f"AKI={aki_rate:.1f}%")

        if cond_name not in all_results:
            all_results[cond_name] = {}

        for algo in all_algos:
            # Skip if already completed in a previous run
            if algo in completed.get(cond_name, set()):
                print(f"\n  ── {ALGO_LABELS[algo]} — [already done, skipping] ──")
                macro = all_results[cond_name][algo].get("_macro", {})
                m = macro.get("mean", float("nan"))
                s_val = macro.get("std", 0.0)
                print(f"    Global AUROC (pooled): {m:.4f} ± {s_val:.4f}")
                continue

            print(f"\n  ── {ALGO_LABELS[algo]} ──")
            result = run_multi_seed(
                algo=algo,
                site_data_fn=make_site_data,
                n_in=n_in,
                seeds=seeds,
                rounds=cond_rounds,
                local_epochs=args.local_epochs,
                lr=args.lr,
                mu=args.mu,
                h1=args.h1, h2=args.h2,
                cv_clip=cv_clip,
            )
            all_results[cond_name][algo] = result

            macro = result.get("_macro", {})
            m = macro.get("mean", np.nan)
            s_val = macro.get("std",  0.0)
            print(f"    Global AUROC (pooled): {m:.4f} ± {s_val:.4f}")
            for sid in sorted(k for k in result if not k.startswith("_")):
                site = result[sid]
                print(f"    {sid}: {site['mean']:.4f} ± {site['std']:.4f}")

            # Save after every completed algorithm
            _save_checkpoint()

    # Final save
    _save_checkpoint()
    print(f"\n[saved] {results_path}")

    # Print results table
    print_results_table(all_results)

    # Plots
    print("\nGenerating plots...")
    plot_results_with_ci(all_results, output_dir)
    plot_fairness_panel(all_results, output_dir)
    plot_scaffold_comparison(all_results, output_dir)
    plot_alpha_comparison(all_results, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
