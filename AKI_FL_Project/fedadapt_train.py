"""
fedadapt_train.py
=================
Unified training loop for federated AKI prediction experiments.

Supports four methods via --method flag:
    fedadapt  — proposed: input adapter + shared body + GRL + personal head
    fedavg    — FedAvg baseline: adapter + shared body + shared head (no GRL)
    fedprox   — FedProx baseline: fedavg + proximal penalty on body weights
    scaffold  — SCAFFOLD baseline: fedavg + control variates for client drift

All methods share:
    - SiteInputAdapter (local, never federated) — handles 18–39 feature heterogeneity
    - SharedBody (federated via weighted FedAvg)
    - Same data loading, train/test split, evaluation metrics

FedAdapt additionally uses:
    - GRLGroupDiscriminator (local adversarial head, group-invariant embedding)
    - PersonalHead (local classifier, fine-tuned after federation)

OUTPUTS
-------
results/<method>/
    round_auroc.csv          — per-site AUROC at each federation round
    final_metrics.csv        — AUROC, F1, AUPRC per site after head fine-tuning
    fl_gain_correlation.png  — FL-gain index vs AUROC improvement (Phase III plot)
    training_curves.png      — per-site AUROC across rounds
    loss_curves.png          — per-site task + adv loss across rounds
    site_<X>_model.pt        — saved client model per site

USAGE
-----
# FedAdapt (proposed)
python fedadapt_train.py --data_dir ./phase2_sites/ --method fedadapt

# All baselines in one go (run sequentially)
for METHOD in fedavg fedprox scaffold fedadapt; do
    python fedadapt_train.py --data_dir ./phase2_sites/ --method $METHOD
done

KEY HYPERPARAMETERS
-------------------
--rounds        T   Federation rounds            (default 50)
--local_epochs  K   Local epochs per round       (default 5)
--finetune_epochs E Head fine-tune epochs        (default 10, FedAdapt only)
--lr            float Learning rate              (default 1e-3)
--batch_size    int  Mini-batch size             (default 256, CPU-safe)
--embedding_dim int  Shared embedding dim        (default 64, must match sim)
--hidden_dim    int  MLP hidden dim              (default 128)
--lambda_adv    float GRL adversarial weight     (default 0.5, FedAdapt only)
--mu            float FedProx proximal weight    (default 0.01)
--test_frac     float Local held-out test split  (default 0.20)
--seed          int  Random seed                 (default 42)
--fl_gain_csv   path FL-gain index CSV from sim  (auto-detected if present)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ── import model components ───────────────────────────────────────────────────
# Assumes fedadapt_model.py is in the same directory or on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
from fedadapt_model import (
    FedAdaptClient,
    FedAdaptServer,
    SharedBody,
    fedadapt_loss,
    fedprox_penalty,
)

# ─── FEATURE GROUP DEFINITIONS (must match simulation) ────────────────────────

FEATURE_GROUPS = {
    "renal":        ["baseline_scr", "scr_first_24h_min", "scr_first_24h_max",
                     "bun_first_24h_min", "bun_first_24h_max"],
    "inflammatory": ["lactate_first_24h_min", "lactate_first_24h_max",
                     "wbc_first_24h_min", "wbc_first_24h_max",
                     "platelets_first_24h_min", "platelets_first_24h_max"],
    "metabolic":    ["sodium_first_24h_min", "sodium_first_24h_max",
                     "potassium_first_24h_min", "potassium_first_24h_max",
                     "bicarbonate_first_24h_min", "bicarbonate_first_24h_max",
                     "hemoglobin_first_24h_min", "hemoglobin_first_24h_max",
                     "glucose_first_24h_min", "glucose_first_24h_max"],
    "hemodynamic":  ["sbp_first_24h_min", "sbp_first_24h_max",
                     "dbp_first_24h_min", "dbp_first_24h_max",
                     "heart_rate_first_24h_min", "heart_rate_first_24h_max",
                     "spo2_first_24h_min"],
    "clinical":     ["admission_type", "gender", "age_at_admission",
                     "has_diabetes", "has_hypertension", "has_chf",
                     "has_sepsis", "has_liver_disease", "has_cancer",
                     "baseline_method"],
}
GROUP_INDEX = {g: i for i, g in enumerate(FEATURE_GROUPS)}   # name → int label
N_GROUPS    = len(FEATURE_GROUPS)   # 5


# ─── SEEDS ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

class SiteData:
    """
    Holds train/test tensors, DataLoaders, and metadata for one site.

    Group label assignment:
        Each sample is assigned the index of the dominant feature group —
        the group whose columns have the fewest missing values in that row.
        This gives the GRL discriminator a meaningful group signal even
        when sites have overlapping feature sets.
    """

    def __init__(
        self,
        site_id:    str,
        meta:       dict,
        df:         pd.DataFrame,
        label_col:  str,
        test_frac:  float,
        batch_size: int,
        seed:       int,
    ):
        self.site_id        = site_id
        self.meta           = meta
        self.feature_names  = meta["feature_names"]
        self.aki_prevalence = meta.get("aki_prevalence") or df[label_col].mean()
        self.unlabeled      = meta.get("unlabeled", False)
        self.n_samples      = len(df)

        # ── feature matrix ──────────────────────────────────────────────────
        feat_cols = [c for c in self.feature_names if c in df.columns]
        X = df[feat_cols].fillna(0.0).values.astype(np.float32)

        # ── standardise per site (z-score) ──────────────────────────────────
        mu  = X.mean(axis=0, keepdims=True)
        sig = X.std(axis=0, keepdims=True) + 1e-8
        X   = (X - mu) / sig

        # ── labels ──────────────────────────────────────────────────────────
        if self.unlabeled or label_col not in df.columns:
            y = np.zeros(len(df), dtype=np.float32)   # placeholder
        else:
            y = df[label_col].fillna(0).values.astype(np.float32)

        # ── group labels ─────────────────────────────────────────────────────
        g = self._assign_group_labels(df, feat_cols)

        # ── train / test split ──────────────────────────────────────────────
        rng   = np.random.default_rng(seed)
        idx   = rng.permutation(len(df))
        n_test = max(1, int(len(df) * test_frac))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        self.X_train = torch.tensor(X[train_idx])
        self.y_train = torch.tensor(y[train_idx])
        self.g_train = torch.tensor(g[train_idx], dtype=torch.long)
        self.X_test  = torch.tensor(X[test_idx])
        self.y_test  = torch.tensor(y[test_idx])

        # ── class-weighted sampler for imbalanced sites ──────────────────────
        if not self.unlabeled and self.y_train.sum() > 0:
            n_pos  = self.y_train.sum().item()
            n_neg  = len(self.y_train) - n_pos
            w_pos  = len(self.y_train) / (2 * n_pos)
            w_neg  = len(self.y_train) / (2 * n_neg)
            sample_weights = torch.where(self.y_train == 1,
                                         torch.tensor(w_pos),
                                         torch.tensor(w_neg))
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                            replacement=True)
            self.train_loader = DataLoader(
                TensorDataset(self.X_train, self.y_train, self.g_train),
                batch_size=batch_size, sampler=sampler,
            )
        else:
            self.train_loader = DataLoader(
                TensorDataset(self.X_train, self.y_train, self.g_train),
                batch_size=batch_size, shuffle=True,
            )

        # ── pos_weight for BCE loss (handles class imbalance) ────────────────
        if not self.unlabeled and self.y_train.sum() > 0:
            n_pos = self.y_train.sum().item()
            n_neg = len(self.y_train) - n_pos
            self.pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
        else:
            self.pos_weight = None

    def _assign_group_labels(self, df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
        """Assign dominant feature group index to each row."""
        group_labels = np.zeros(len(df), dtype=np.int64)
        group_scores = np.zeros((len(df), N_GROUPS))

        for g_name, g_idx in GROUP_INDEX.items():
            cols_present = [c for c in FEATURE_GROUPS[g_name] if c in feat_cols]
            if not cols_present:
                continue
            # Score = fraction of group columns that are non-null / non-zero
            sub = df[cols_present].fillna(0)
            group_scores[:, g_idx] = (sub != 0).mean(axis=1).values

        group_labels = group_scores.argmax(axis=1)
        return group_labels


def load_all_sites(
    data_dir:   Path,
    label_col:  str,
    alpha:      float,
    gamma:      float,
    test_frac:  float,
    batch_size: int,
    seed:       int,
) -> Dict[str, SiteData]:
    """
    Load all site CSVs and their adapter metadata JSONs from data_dir.
    Matches files by alpha/gamma suffix produced by mimic_ftl_simulation_phase2.py.
    """
    sites: Dict[str, SiteData] = {}

    for meta_path in sorted(data_dir.glob("*_adapter_meta.json")):
        site_id = meta_path.stem.replace("_adapter_meta", "")

        with open(meta_path) as f:
            meta = json.load(f)

        csv_path = data_dir / f"{site_id}_alpha{alpha}_gamma{gamma}.csv"
        if not csv_path.exists():
            # Try without alpha/gamma suffix (fallback)
            candidates = list(data_dir.glob(f"{site_id}*.csv"))
            if not candidates:
                print(f"  [skip] {site_id}: no CSV found")
                continue
            csv_path = candidates[0]

        df = pd.read_csv(csv_path)
        print(f"  [load] {site_id}: {len(df):,} rows × {df.shape[1]} cols  "
              f"(AKI={df[label_col].mean():.3f})" if label_col in df.columns and not df[label_col].isna().all()
              else f"  [load] {site_id}: {len(df):,} rows (unlabeled)")

        sites[site_id] = SiteData(
            site_id=site_id, meta=meta, df=df,
            label_col=label_col, test_frac=test_frac,
            batch_size=batch_size, seed=seed,
        )

    if not sites:
        raise FileNotFoundError(
            f"No site files found in {data_dir}. "
            f"Run mimic_ftl_simulation_phase2.py first."
        )

    return sites


# ─── CLIENT FACTORY ───────────────────────────────────────────────────────────

def build_clients(
    sites:         Dict[str, SiteData],
    embedding_dim: int,
    hidden_dim:    int,
    grl_lambda:    float,
) -> Dict[str, FedAdaptClient]:
    """Build one FedAdaptClient per site using adapter metadata."""
    clients = {}
    for site_id, sd in sites.items():
        clients[site_id] = FedAdaptClient.from_metadata(
            sd.meta,
            hidden_dim=hidden_dim,
            n_groups=N_GROUPS,
            grl_lambda_max=grl_lambda,
        )
        print(f"  [model] {site_id}: input_dim={sd.meta['input_dim']}  "
              f"→ embedding_dim={embedding_dim}  hidden={hidden_dim}")
    return clients


# ─── SCAFFOLD CONTROL VARIATES ────────────────────────────────────────────────

def init_scaffold_variates(
    clients: Dict[str, FedAdaptClient],
) -> Tuple[Dict[str, dict], dict]:
    """
    Initialise per-client (c_i) and server (c) control variates for SCAFFOLD.
    Both are zero-initialised state dicts matching SharedBody parameters.
    """
    zero_state = {
        k: torch.zeros_like(v)
        for k, v in next(iter(clients.values())).body.state_dict().items()
    }
    client_variates = {sid: deepcopy(zero_state) for sid in clients}
    server_variate  = deepcopy(zero_state)
    return client_variates, server_variate


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate_site(
    client:   FedAdaptClient,
    sd:       SiteData,
) -> Dict[str, float]:
    """
    Compute AUROC, F1, and AUPRC on the site's held-out test set.
    Returns zeros for unlabeled sites.
    """
    if sd.unlabeled or sd.y_test.sum() == 0:
        return {"auroc": 0.0, "f1": 0.0, "auprc": 0.0}

    client.eval()
    with torch.no_grad():
        logits, _ = client(sd.X_test)
        probs     = torch.sigmoid(logits).numpy()
        labels    = sd.y_test.numpy()

    try:
        from sklearn.metrics import (
            roc_auc_score, f1_score,
            average_precision_score, precision_recall_curve,
        )
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))
        preds = (probs >= 0.5).astype(int)
        f1    = float(f1_score(labels, preds, zero_division=0))
    except Exception:
        auroc, auprc, f1 = 0.0, 0.0, 0.0

    return {"auroc": auroc, "f1": f1, "auprc": auprc}


# ─── LOCAL TRAINING STEPS ─────────────────────────────────────────────────────

def local_step_fedadapt(
    client:     FedAdaptClient,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    lambda_adv: float,
    pos_weight: Optional[Tensor],
) -> Dict[str, float]:
    """One epoch of FedAdapt local training (task + GRL adversarial)."""
    client.train()
    totals = {"total": 0.0, "task": 0.0, "adv": 0.0}
    n = 0

    for x, y, g in loader:
        optimizer.zero_grad()
        task_logit, group_logits = client(x)
        total, task, adv = fedadapt_loss(
            task_logit, y, group_logits, g,
            lambda_adv=lambda_adv, pos_weight=pos_weight,
        )
        total.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()

        b = len(y)
        totals["total"] += total.item() * b
        totals["task"]  += task.item()  * b
        totals["adv"]   += adv.item()   * b
        n += b

    return {k: v / max(n, 1) for k, v in totals.items()}


def local_step_fedavg(
    client:     FedAdaptClient,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    pos_weight: Optional[Tensor],
) -> Dict[str, float]:
    """One epoch of FedAvg local training (task loss only, no GRL)."""
    client.train()
    total_loss, n = 0.0, 0

    for x, y, _ in loader:
        optimizer.zero_grad()
        # For baselines: use encode() + head directly (bypasses GRL)
        emb    = client.encode(x)
        logit  = client.head(emb)
        loss   = F.binary_cross_entropy_with_logits(
            logit, y.float(), pos_weight=pos_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)

    return {"total": total_loss / max(n, 1), "task": total_loss / max(n, 1), "adv": 0.0}


def local_step_fedprox(
    client:            FedAdaptClient,
    loader:            DataLoader,
    optimizer:         torch.optim.Optimizer,
    pos_weight:        Optional[Tensor],
    global_body_state: dict,
    mu:                float,
) -> Dict[str, float]:
    """One epoch of FedProx (FedAvg + proximal penalty on SharedBody)."""
    client.train()
    total_loss, n = 0.0, 0

    for x, y, _ in loader:
        optimizer.zero_grad()
        emb   = client.encode(x)
        logit = client.head(emb)
        task  = F.binary_cross_entropy_with_logits(
            logit, y.float(), pos_weight=pos_weight
        )
        prox  = fedprox_penalty(client, global_body_state, mu=mu)
        loss  = task + prox
        loss.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)

    return {"total": total_loss / max(n, 1), "task": total_loss / max(n, 1), "adv": 0.0}


def local_step_scaffold(
    client:          FedAdaptClient,
    loader:          DataLoader,
    optimizer:       torch.optim.Optimizer,
    pos_weight:      Optional[Tensor],
    c_i:             dict,
    c_server:        dict,
) -> Dict[str, float]:
    """
    One epoch of SCAFFOLD local training.
    After the standard gradient step, applies the control variate correction:
        w ← w - lr * (c_i - c_server)
    This corrects the client drift caused by heterogeneous local data.
    Note: sign is c_i - c_server (local minus global), not the reverse.
    """
    client.train()
    total_loss, n = 0.0, 0
    lr = optimizer.param_groups[0]["lr"]

    for x, y, _ in loader:
        optimizer.zero_grad()
        emb   = client.encode(x)
        logit = client.head(emb)
        loss  = F.binary_cross_entropy_with_logits(
            logit, y.float(), pos_weight=pos_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()

        # Apply control variate correction to SharedBody params only
        # SCAFFOLD correction: subtract (c_i - c_server) from gradient
        # i.e. w ← w - lr*(grad + c_i - c_server)
        # Equivalent to: param -= lr * (c_i - c_server) AFTER the grad step
        with torch.no_grad():
            for name, param in client.body.named_parameters():
                if name in c_server and name in c_i:
                    correction = c_i[name] - c_server[name]   # FIX: was c_server - c_i (sign flipped)
                    correction = correction.clamp(-1.0, 1.0)  # FIX: clip to prevent accumulation
                    param.data -= lr * correction

        total_loss += loss.item() * len(y)
        n += len(y)

    return {"total": total_loss / max(n, 1), "task": total_loss / max(n, 1), "adv": 0.0}


def update_scaffold_variates(
    client:         FedAdaptClient,
    c_i:            dict,
    c_server:       dict,
    global_body:    dict,
    local_body_pre: dict,
    local_epochs:   int,
    lr:             float,
) -> dict:
    """
    Compute updated client control variate c_i_new after local training.
    SCAFFOLD Option II:
        c_i_new = c_i - c_server + (w_global - w_local) / (K * lr)
    Returns delta = c_i_new - c_i for server aggregation.
    """
    c_i_new  = {}
    delta_c  = {}
    for key in c_i:
        w_g = global_body[key].float()
        w_l = client.body.state_dict()[key].float()
        c_i_new[key] = (c_i[key].float()
                        - c_server[key].float()
                        + (w_g - w_l) / (local_epochs * lr + 1e-8))
        delta_c[key] = c_i_new[key] - c_i[key]
    return c_i_new, delta_c


# ─── HEAD FINE-TUNING (FedAdapt only) ────────────────────────────────────────

def finetune_heads(
    clients:         Dict[str, FedAdaptClient],
    sites:           Dict[str, SiteData],
    finetune_epochs: int,
    lr:              float,
) -> None:
    """
    Freeze SharedBody. Fine-tune PersonalHead locally per site.
    FedBABU-style: body is fixed, only head weights update.
    """
    print("\n  ── Head fine-tuning (FedAdapt) ──")
    for site_id, client in clients.items():
        sd = sites[site_id]
        if sd.unlabeled:
            continue

        client.freeze_body()
        opt = torch.optim.Adam(client.head_parameters(), lr=lr)

        for epoch in range(finetune_epochs):
            client.train()
            for x, y, _ in sd.train_loader:
                opt.zero_grad()
                emb   = client.encode(x)
                logit = client.head(emb)
                loss  = F.binary_cross_entropy_with_logits(
                    logit, y.float(), pos_weight=sd.pos_weight
                )
                loss.backward()
                opt.step()

        client.unfreeze_body()
        metrics = evaluate_site(client, sd)
        print(f"    {site_id}: AUROC={metrics['auroc']:.4f}  "
              f"F1={metrics['f1']:.4f}  AUPRC={metrics['auprc']:.4f}")


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def plot_training_curves(
    round_records: List[dict],
    method:        str,
    output_dir:    Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df     = pd.DataFrame(round_records)
    sites  = [c for c in df.columns if c.startswith("auroc_")]
    labels = [c.replace("auroc_", "") for c in sites]

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Training curves — {method.upper()}", fontsize=12)

    ax = axes[0]
    for i, (col, lbl) in enumerate(zip(sites, labels)):
        ax.plot(df["round"], df[col], label=lbl, color=COLORS[i % len(COLORS)], linewidth=1.8)
    ax.set_xlabel("Federation round")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-site AUROC across rounds")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    loss_cols = [c for c in df.columns if c.startswith("task_loss_")]
    for i, col in enumerate(loss_cols):
        lbl = col.replace("task_loss_", "")
        ax2.plot(df["round"], df[col], label=lbl, color=COLORS[i % len(COLORS)],
                 linewidth=1.8)
    ax2.set_xlabel("Federation round")
    ax2.set_ylabel("Task loss")
    ax2.set_title("Per-site task loss across rounds")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = output_dir / "training_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {p.name}")


def plot_fl_gain_correlation(
    final_metrics:   pd.DataFrame,
    local_metrics:   Dict[str, float],
    fl_gain_csv:     Optional[Path],
    method:          str,
    output_dir:      Path,
) -> None:
    """
    Phase III validation figure:
    FL-gain index (x) vs AUROC improvement over local-only (y).

    Each point is one site. If the FL-gain index is a good predictor,
    we expect a positive correlation.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("  [skip] fl_gain_correlation: matplotlib/scipy not available")
        return

    if fl_gain_csv is None or not fl_gain_csv.exists():
        print("  [skip] fl_gain_correlation: fl_gain_csv not found")
        return

    fl_df = pd.read_csv(fl_gain_csv)

    rows = []
    for _, row in fl_df.iterrows():
        sid = row["site_id"]
        if sid not in final_metrics["site_id"].values:
            continue
        fed_auroc   = float(final_metrics.loc[final_metrics["site_id"] == sid, "auroc"].iloc[0])
        local_auroc = local_metrics.get(sid, 0.0)
        improvement = fed_auroc - local_auroc
        rows.append({
            "site_id":      sid,
            "fl_gain":      float(row["fl_gain_index"]),
            "local_auroc":  local_auroc,
            "fed_auroc":    fed_auroc,
            "improvement":  improvement,
            "role":         row.get("role", "unknown"),
        })

    if len(rows) < 3:
        print("  [skip] fl_gain_correlation: not enough labelled sites")
        return

    corr_df = pd.DataFrame(rows)

    ROLE_COLORS = {
        "contributor":            "#4C72B0",
        "conditional_benefitter": "#DD8452",
        "primary_benefitter":     "#C44E52",
        "transfer_benefitter":    "#8172B2",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"FL-gain index vs AUROC improvement — {method.upper()}\n"
        f"Phase III validation: does FL-gain index predict benefit from federation?",
        fontsize=11,
    )

    # ── Panel 1: FL-gain vs AUROC improvement ────────────────────────────────
    ax = axes[0]
    for _, row in corr_df.iterrows():
        color = ROLE_COLORS.get(row["role"], "#333333")
        ax.scatter(row["fl_gain"], row["improvement"],
                   color=color, s=120, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(row["site_id"], (row["fl_gain"], row["improvement"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Regression line
    x_vals = corr_df["fl_gain"].values
    y_vals = corr_df["improvement"].values
    if len(x_vals) >= 3:
        m, b   = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, m * x_line + b, "k--", linewidth=1.2, alpha=0.6)
        try:
            r, p_val = pearsonr(x_vals, y_vals)
            rho, _   = spearmanr(x_vals, y_vals)
            ax.set_title(
                f"Pearson r={r:.3f}  Spearman ρ={rho:.3f}  (p={p_val:.3f})"
            )
        except Exception:
            ax.set_title("FL-gain vs AUROC improvement")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("FL-gain index")
    ax.set_ylabel("AUROC improvement (federated − local-only)")
    ax.grid(alpha=0.3)

    # ── Panel 2: side-by-side AUROC bar chart ────────────────────────────────
    ax2    = axes[1]
    n      = len(corr_df)
    x_pos  = np.arange(n)
    bw     = 0.35
    colors = [ROLE_COLORS.get(r, "#333") for r in corr_df["role"]]

    ax2.bar(x_pos - bw / 2, corr_df["local_auroc"], bw,
            label="Local-only", color="#AAAAAA", edgecolor="white")
    ax2.bar(x_pos + bw / 2, corr_df["fed_auroc"], bw,
            label=f"{method.upper()} (federated)", color=colors, edgecolor="white")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(corr_df["site_id"], rotation=15, ha="right")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Local-only vs federated AUROC per site")
    ax2.set_ylim(0.4, 1.0)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Save correlation table
    corr_df.to_csv(output_dir / "fl_gain_correlation.csv", index=False)

    plt.tight_layout()
    p = output_dir / "fl_gain_correlation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {p.name}")


def plot_final_comparison(
    final_metrics: pd.DataFrame,
    method:        str,
    output_dir:    Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df     = final_metrics.copy()
    sites  = df["site_id"].tolist()
    x      = np.arange(len(sites))
    bw     = 0.25

    COLORS = {"auroc": "#4C72B0", "f1": "#DD8452", "auprc": "#55A868"}
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (metric, color) in enumerate(COLORS.items()):
        vals = df[metric].tolist()
        bars = ax.bar(x + (i - 1) * bw, vals, bw, label=metric.upper(),
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Final evaluation metrics — {method.upper()}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = output_dir / "final_metrics.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {p.name}")


# ─── LOCAL-ONLY BASELINE ─────────────────────────────────────────────────────

def run_local_only(
    sites:         Dict[str, SiteData],
    embedding_dim: int,
    hidden_dim:    int,
    local_epochs:  int,
    lr:            float,
) -> Dict[str, float]:
    """
    Train each site independently (no federation) for the same number of
    epochs as one federation round × T rounds. Used as the lower bound
    for the FL-gain correlation plot.
    """
    print("\n  ── Local-only baseline ──")
    local_aurocs: Dict[str, float] = {}

    for site_id, sd in sites.items():
        if sd.unlabeled:
            local_aurocs[site_id] = 0.0
            continue

        client = FedAdaptClient.from_metadata(
            sd.meta, hidden_dim=hidden_dim, n_groups=N_GROUPS,
        )
        opt = torch.optim.Adam(client.parameters(), lr=lr)

        for _ in range(local_epochs):
            for x, y, g in sd.train_loader:
                opt.zero_grad()
                emb   = client.encode(x)
                logit = client.head(emb)
                loss  = F.binary_cross_entropy_with_logits(
                    logit, y.float(), pos_weight=sd.pos_weight
                )
                loss.backward()
                opt.step()

        m = evaluate_site(client, sd)
        local_aurocs[site_id] = m["auroc"]
        print(f"    {site_id}: local AUROC={m['auroc']:.4f}")

    return local_aurocs


# ─── MAIN FEDERATION LOOP ─────────────────────────────────────────────────────

def run_federation(
    method:          str,
    sites:           Dict[str, SiteData],
    clients:         Dict[str, FedAdaptClient],
    args:            argparse.Namespace,
    output_dir:      Path,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Run T federation rounds for the chosen method.
    Returns (final_metrics_df, round_records).
    """
    server = FedAdaptServer(
        body=next(iter(clients.values())).body
    )
    server.broadcast(list(clients.values()))

    # Per-client optimisers — all params for fedadapt, body+head for baselines
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    for sid, client in clients.items():
        optimizers[sid] = torch.optim.Adam(client.parameters(), lr=args.lr)

    # SCAFFOLD variates
    if method == "scaffold":
        client_variates, server_variate = init_scaffold_variates(clients)

    # Global body snapshot for FedProx
    global_body_snapshot: dict = {}

    round_records: List[dict] = []

    print(f"\n  ── Federation [{method.upper()}]  {args.rounds} rounds × {args.local_epochs} local epochs ──")

    for rnd in range(1, args.rounds + 1):
        progress = rnd / args.rounds

        # ── broadcast ──────────────────────────────────────────────────────
        server.broadcast(list(clients.values()))
        if method == "fedprox":
            global_body_snapshot = deepcopy(
                next(iter(clients.values())).body.state_dict()
            )

        body_states:   List[dict]  = []
        sample_counts: List[float] = []
        delta_variates: Dict[str, dict] = {}

        # ── local training ─────────────────────────────────────────────────
        round_losses: Dict[str, dict] = {}
        for sid, client in clients.items():
            sd = sites[sid]
            if sd.unlabeled:
                continue

            client.set_training_progress(progress)
            opt = optimizers[sid]

            for _ in range(args.local_epochs):
                if method == "fedadapt":
                    losses = local_step_fedadapt(
                        client, sd.train_loader, opt,
                        lambda_adv=args.lambda_adv,
                        pos_weight=sd.pos_weight,
                    )
                elif method == "fedprox":
                    losses = local_step_fedprox(
                        client, sd.train_loader, opt,
                        pos_weight=sd.pos_weight,
                        global_body_state=global_body_snapshot,
                        mu=args.mu,
                    )
                elif method == "scaffold":
                    losses = local_step_scaffold(
                        client, sd.train_loader, opt,
                        pos_weight=sd.pos_weight,
                        c_i=client_variates[sid],
                        c_server=server_variate,
                    )
                else:  # fedavg
                    losses = local_step_fedavg(
                        client, sd.train_loader, opt,
                        pos_weight=sd.pos_weight,
                    )

            round_losses[sid] = losses

            # SCAFFOLD: update client variate after local steps
            if method == "scaffold":
                c_i_new, delta_c = update_scaffold_variates(
                    client,
                    c_i=client_variates[sid],
                    c_server=server_variate,
                    global_body=server.global_body.state_dict(),
                    local_body_pre=global_body_snapshot,
                    local_epochs=args.local_epochs,
                    lr=args.lr,
                )
                client_variates[sid] = c_i_new
                delta_variates[sid]  = delta_c

            body_states.append(client.get_body_state())
            sample_counts.append(float(sd.n_samples))

        # ── aggregate ──────────────────────────────────────────────────────
        server.aggregate(body_states, sample_counts)

        # SCAFFOLD: update server variate
        if method == "scaffold" and delta_variates:
            n_participants = len(delta_variates)
            for key in server_variate:
                server_variate[key] = server_variate[key] + sum(
                    dv[key] for dv in delta_variates.values()
                ) / n_participants

        # ── evaluate ───────────────────────────────────────────────────────
        server.broadcast(list(clients.values()))
        record: dict = {"round": rnd}
        for sid, client in clients.items():
            sd = sites[sid]
            m  = evaluate_site(client, sd)
            record[f"auroc_{sid}"]     = round(m["auroc"], 4)
            record[f"task_loss_{sid}"] = round(round_losses.get(sid, {}).get("task", 0.0), 4)
            record[f"adv_loss_{sid}"]  = round(round_losses.get(sid, {}).get("adv", 0.0), 4)

        round_records.append(record)

        if rnd % max(1, args.rounds // 10) == 0 or rnd == args.rounds:
            aurocs = "  ".join(
                f"{sid}={record[f'auroc_{sid}']:.3f}"
                for sid in clients
            )
            print(f"    Round {rnd:3d}/{args.rounds}  |  {aurocs}")

    # ── head fine-tuning (FedAdapt only) ───────────────────────────────────
    if method == "fedadapt":
        finetune_heads(clients, sites, args.finetune_epochs, args.lr)

    # ── final evaluation ───────────────────────────────────────────────────
    final_rows = []
    for sid, client in clients.items():
        m = evaluate_site(client, sites[sid])
        final_rows.append({"site_id": sid, **m})
    final_df = pd.DataFrame(final_rows)

    return final_df, round_records


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir",       required=True,  help="Phase II simulation output dir")
    p.add_argument("--label",          default="AKI_label")
    p.add_argument("--alpha",          type=float, default=0.5,  help="Dirichlet α used in sim")
    p.add_argument("--gamma",          type=float, default=0.75, help="Gamma used in sim")
    p.add_argument("--fl_gain_csv",    default=None,   help="Path to fl_gain_index_*.csv")

    # Method
    p.add_argument("--method",         default="fedadapt",
                   choices=["fedadapt", "fedavg", "fedprox", "scaffold"])

    # Federation
    p.add_argument("--rounds",         type=int,   default=50)
    p.add_argument("--local_epochs",   type=int,   default=5,    help="K local epochs per round")
    p.add_argument("--finetune_epochs",type=int,   default=10,   help="Head fine-tune epochs (FedAdapt)")

    # Model
    p.add_argument("--embedding_dim",  type=int,   default=64)
    p.add_argument("--hidden_dim",     type=int,   default=128)

    # Optimisation
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--batch_size",     type=int,   default=256,  help="CPU-safe batch size")
    p.add_argument("--lambda_adv",     type=float, default=0.1,  help="GRL max weight (FedAdapt); was 0.5 which caused early collapse")
    p.add_argument("--mu",             type=float, default=0.01, help="Proximal weight (FedProx)")

    # Misc
    p.add_argument("--test_frac",      type=float, default=0.20)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output_dir",     default="./results/",
                   help="Root output dir; method subdir created automatically")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  FedAdapt training  |  method={args.method.upper()}")
    print(f"  rounds={args.rounds}  local_epochs={args.local_epochs}  "
          f"lr={args.lr}  batch={args.batch_size}")
    print(f"  output → {output_dir}")
    print(f"{'='*65}\n")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    print("Loading site data...")
    sites = load_all_sites(
        data_dir, args.label, args.alpha, args.gamma,
        args.test_frac, args.batch_size, args.seed,
    )

    # ── 2. Auto-detect FL-gain CSV ───────────────────────────────────────────
    fl_gain_csv = None
    if args.fl_gain_csv:
        fl_gain_csv = Path(args.fl_gain_csv)
    else:
        candidates = list(data_dir.glob("fl_gain_index_*.csv"))
        if candidates:
            fl_gain_csv = candidates[0]
            print(f"  [fl_gain] auto-detected: {fl_gain_csv.name}")

    # ── 3. Local-only baseline (for correlation plot) ─────────────────────────
    print("\nRunning local-only baseline...")
    local_aurocs = run_local_only(
        sites, args.embedding_dim, args.hidden_dim,
        args.local_epochs * args.rounds, args.lr,
    )

    # ── 4. Build clients ─────────────────────────────────────────────────────
    print("\nBuilding FedAdaptClient per site...")
    clients = build_clients(sites, args.embedding_dim, args.hidden_dim, args.lambda_adv)

    # ── 5. Federation ────────────────────────────────────────────────────────
    final_df, round_records = run_federation(
        method=args.method,
        sites=sites,
        clients=clients,
        args=args,
        output_dir=output_dir,
    )

    # ── 6. Save results ───────────────────────────────────────────────────────
    final_df.to_csv(output_dir / "final_metrics.csv", index=False)
    pd.DataFrame(round_records).to_csv(output_dir / "round_auroc.csv", index=False)
    print(f"\n  Final metrics:\n{final_df.to_string(index=False)}")

    # ── 7. Save models ────────────────────────────────────────────────────────
    for sid, client in clients.items():
        torch.save(client.state_dict(), output_dir / f"{sid}_model.pt")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_training_curves(round_records, args.method, output_dir)
    plot_final_comparison(final_df, args.method, output_dir)
    plot_fl_gain_correlation(
        final_df, local_aurocs, fl_gain_csv, args.method, output_dir
    )

    print(f"\n✅  Done — all outputs in {output_dir}/")


if __name__ == "__main__":
    main()
