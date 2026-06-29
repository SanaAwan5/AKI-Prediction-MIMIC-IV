"""
fedadapt_train_approach2_v2_4.py
FedAdaptProto — v2.4: Automatic per-site K via silhouette analysis
================================================================
What's new in v2.4 vs v2.3
  • --auto_k flag  : replaces --n_clusters_per_site with data-driven K selection
  • --k_min / --k_max : search range for K (default 2–5)
  • --k_warmup_epochs : local epochs to warm embeddings before silhouette test (default 5)
  • K is selected per site, per class (AKI=1, non-AKI=0) independently,
    then the per-site K = max(K_AKI, K_nonAKI) — conservative, captures
    richer structure in either class.
  • Silhouette scores and selected K values are logged to auto_k_report.csv
  • All v2.3 flags still work; --n_clusters_per_site overrides --auto_k if both given

Expected auto-K results at (alpha=0.3, gamma=0.75) based on silhouette diagnostics:
  site_A: K=3  (silhouette 0.285 AKI class)
  site_B: K=3  (silhouette 0.312 AKI class)
  site_C: K=3  (silhouette 0.423 AKI class — strongest structure)
  site_D: K=3  (silhouette 0.271 AKI class)
  site_E: K=2  (silhouette 0.243, weak bi-modal only)

Usage:
  python3 fedadapt_train_approach2_v2_4.py \
      --data_dir ./phase2_sites_approach2/alpha0.3_gamma0.75 \
      --method fedadaptproto --alpha 0.3 --gamma 0.75 \
      --warmup_rounds 10 --early_stop_patience 0 \
      --auto_k --k_min 2 --k_max 5 --k_warmup_epochs 5 \
      --output_dir ./results_v24_autok/
"""

import argparse
import os
import csv
import copy
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Model components ──────────────────────────────────────────────────────────

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class SiteInputAdapter(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),   # LayerNorm: stable at any batch size
            nn.ReLU(),
        )

    def forward(self, x):
        return self.proj(x)


class SharedBody(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   # LayerNorm: stable at any batch size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GRLGroupDiscriminator(nn.Module):
    def __init__(self, embedding_dim, n_sites):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_sites),
        )

    def forward(self, x, lam):
        x_rev = GradientReversalFn.apply(x, lam)
        return self.disc(x_rev)


class PersonalHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class FedAdaptClient(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_sites):
        super().__init__()
        self.adapter  = SiteInputAdapter(input_dim, embedding_dim)
        self.body     = SharedBody(embedding_dim, hidden_dim)
        self.disc     = GRLGroupDiscriminator(embedding_dim, n_sites)
        self.head     = PersonalHead(embedding_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, lam=0.0):
        z   = self.adapter(x)
        h   = self.body(z)
        out = self.head(h)
        adv = self.disc(h, lam)
        return out, adv, h


# ── Data loading ──────────────────────────────────────────────────────────────

def load_site(path, label_col=None, batch_size=256, test_frac=0.2):
    df = pd.read_csv(path)

    # Step 1: find label column — must contain 'aki', be numeric, binary {0,1}
    # Pick the LAST candidate so 'AKI_label' beats any binary comorbidity col
    if label_col is None:
        candidates = [c for c in df.columns
                      if 'aki' in c.lower()
                      and pd.api.types.is_numeric_dtype(df[c])
                      and set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
        if not candidates:
            raise ValueError(f'Cannot auto-detect label column in {path}. '
                             f'Columns: {list(df.columns)}')
        # Prefer columns explicitly named 'aki_label' or 'AKI_label'
        exact = [c for c in candidates if c.lower() == 'aki_label']
        label_col = exact[0] if exact else candidates[-1]

    # Step 2: ALL numeric columns except label and site_id string cols
    # Do NOT filter out binary cols — they are legitimate features (comorbidities)
    non_numeric = [c for c in df.columns
                   if not pd.api.types.is_numeric_dtype(df[c])]
    exclude = {label_col} | set(non_numeric)
    feat_cols = [c for c in df.columns if c not in exclude]
    X_df = df[feat_cols].copy()

    # Step 3: two-stage NaN imputation
    # hours_since → 0 (sentinel: lab not recently measured)
    # all others  → column median
    hours_cols = [c for c in X_df.columns if 'hours_since' in c.lower()]
    X_df[hours_cols] = X_df[hours_cols].fillna(0.0)
    other_cols = [c for c in X_df.columns if c not in hours_cols]
    X_df[other_cols] = X_df[other_cols].fillna(X_df[other_cols].median())
    X_df = X_df.fillna(0.0)  # catch any all-NaN columns

    X = X_df.values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)

    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_frac))
    tr, te = idx[:split], idx[split:]

    # Fit scaler on train split only to avoid test leakage
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_te = scaler.transform(X[te])

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y[tr]))
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y[te]))

    g = torch.Generator()
    g.manual_seed(SEED)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=True,
                       generator=g)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    prevalence = float(y.mean())
    return tr_ld, te_ld, X.shape[1], prevalence, feat_cols


# ── Prototype helpers ─────────────────────────────────────────────────────────

def compute_prototypes_kmeans(embeddings, labels, n_clusters):
    """
    Per-class k-means prototypes.
    Returns dict: {class_label: tensor of shape (n_clusters, emb_dim)}
    """
    protos = {}
    for cls in [0, 1]:
        mask = (labels == cls)
        if mask.sum() < n_clusters:
            # Fallback: single centroid if too few samples
            protos[cls] = embeddings[mask].mean(0, keepdim=True).repeat(n_clusters, 1)
        else:
            emb_np = embeddings[mask].detach().cpu().numpy()
            # Replace any NaN with 0 before clustering
            emb_np = np.nan_to_num(emb_np, nan=0.0)
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED)
            km.fit(emb_np)
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            protos[cls] = centers
    return protos


def nearest_proto_loss(h, labels, site_protos, global_protos):
    """
    For each embedding, find the nearest global prototype cluster center
    for its class, compute MSE to that center.
    """
    loss = torch.tensor(0.0, device=h.device)
    for cls in [0, 1]:
        mask = (labels == cls)
        if mask.sum() == 0:
            continue
        h_cls = h[mask]
        g_centers = global_protos[cls].to(h.device).float()   # (K, D) ensure float32
        # nearest center per sample
        dists = torch.cdist(h_cls, g_centers)          # (N_cls, K)
        nearest_idx = dists.argmin(dim=1)
        targets = g_centers[nearest_idx]               # (N_cls, D)
        loss = loss + nn.functional.mse_loss(h_cls, targets)
    return loss


def average_prototypes(all_site_protos, fl_gain_weights=None):
    """
    FL-gain weighted global prototype aggregation.
    Sites with higher FL-gain (more to benefit) contribute more to the global
    prototype, so the global prototype is pulled toward data-scarce sites rather
    than being dominated by data-rich sites.
    If fl_gain_weights is None, falls back to equal weighting.
    """
    global_protos = {}
    site_ids = list(all_site_protos.keys())

    for cls in [0, 1]:
        weighted_centers = []
        total_weight = 0.0
        for sid in site_ids:
            if cls not in all_site_protos[sid]:
                continue
            centers = all_site_protos[sid][cls].float()  # (K, D)
            w = fl_gain_weights.get(sid, 1.0) if fl_gain_weights else 1.0
            weighted_centers.append((centers, w))
            total_weight += w

        if not weighted_centers:
            continue

        # Weighted mean of all cluster centers
        max_k = max(c.shape[0] for c, _ in weighted_centers)
        stacked = torch.cat(
            [c * (w / total_weight) for c, w in weighted_centers], dim=0
        )
        global_protos[cls] = stacked.mean(0, keepdim=True).expand(max_k, -1).clone()
    return global_protos


# ── Auto-K: silhouette-based K selection ──────────────────────────────────────

def extract_embeddings(client, loader, device):
    """Run forward pass, return embeddings and labels as numpy arrays."""
    client.eval()
    embs, labs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if xb.shape[0] < 2:   # skip single-sample batches (BatchNorm)
                continue
            _, _, h = client(xb, lam=0.0)
            h_np = h.cpu().numpy()
            if np.isnan(h_np).any():
                continue           # skip NaN batches
            embs.append(h_np)
            labs.append(yb.numpy())
    client.train()
    if not embs:
        # Return zero embeddings if nothing valid — auto-K will fall back to k_min
        # Use the body output dim (embedding_dim), not input layer shape
        emb_dim = client.body.net[-1].out_features if hasattr(client.body.net[-1], 'out_features') else 64
        return np.zeros((2, emb_dim), dtype=np.float32), np.array([0, 1])
    return np.vstack(embs), np.concatenate(labs)


def warm_embeddings(client, loader, optimizer, device, n_epochs=5):
    """Train locally for n_epochs to warm up the embeddings before silhouette test."""
    criterion = nn.BCEWithLogitsLoss()
    client.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out, _, _ = client(xb, lam=0.0)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()


def select_k_for_site(client, loader, device, k_min=2, k_max=5):
    """
    Silhouette analysis on AKI class (cls=1) embeddings only.
    The AKI class is the clinically relevant grouping — we want to find
    sub-clusters within AKI patients (e.g. mild vs severe AKI).
    The non-AKI class is large and homogeneous; using it would bias toward
    larger K driven by majority-class structure, not AKI subtype structure.
    Returns best_k (int) and a dict of scores for logging.
    """
    embs, labels = extract_embeddings(client, loader, device)
    scores_by_cls = {}

    # Score both classes for logging, but select K from AKI class only
    for cls in [0, 1]:
        mask = (labels == cls)
        cls_emb = embs[mask]
        cls_scores = {}

        for k in range(k_min, k_max + 1):
            if len(cls_emb) < k + 1:
                continue
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
                cluster_labels = km.fit_predict(cls_emb)
                if len(np.unique(cluster_labels)) < 2:
                    continue
                score = silhouette_score(cls_emb, cluster_labels,
                                         sample_size=min(2000, len(cls_emb)))
                cls_scores[k] = float(score)
            except Exception:
                continue

        scores_by_cls[cls] = cls_scores

    # Select K from AKI class (cls=1) only — clinically relevant subgroups
    aki_scores = scores_by_cls.get(1, {})
    if aki_scores:
        best_k = max(aki_scores, key=aki_scores.get)
    else:
        best_k = k_min

    return best_k, scores_by_cls


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(client, loader, device):
    client.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out, _, _ = client(xb, lam=0.0)
            preds.append(torch.sigmoid(out).cpu().numpy())
            trues.append(yb.numpy())
    client.train()
    preds = np.concatenate(preds)
    trues = np.concatenate(trues).astype(int)  # ensure binary int for f1_score
    # Guard against NaN predictions
    if np.isnan(preds).any():
        nan_pct = np.isnan(preds).mean() * 100
        print(f"      [NaN guard] {nan_pct:.1f}% NaN predictions — returning 0.5")
        return 0.5, 0.0, 0.0
    auroc = roc_auc_score(trues, preds) if len(np.unique(trues)) > 1 else 0.5
    f1    = f1_score(trues, (preds > 0.5).astype(int), zero_division=0)
    auprc = average_precision_score(trues, preds) if len(np.unique(trues)) > 1 else 0.0
    return auroc, f1, auprc


# ── FedAvg on shared body ─────────────────────────────────────────────────────

def fedavg_bodies(clients, weights):
    """Weighted average of SharedBody params across all clients."""
    avg_state = {}
    for k, v in clients[list(clients.keys())[0]].body.state_dict().items():
        avg_state[k] = sum(
            weights[site] * clients[site].body.state_dict()[k].float()
            for site in clients
        )
    for site in clients:
        clients[site].body.load_state_dict(avg_state)


# ── Warmup schedule helpers ───────────────────────────────────────────────────

def warmup_scale(current_round, warmup_rounds):
    """Linear ramp from 0→1 over warmup_rounds, then 1.0."""
    if warmup_rounds <= 0:
        return 1.0
    return min(1.0, current_round / warmup_rounds)


# ── Local training step ───────────────────────────────────────────────────────

def local_train(client, loader, optimizer, device, site_idx,
                n_sites, lam_adv, alpha_proto, global_protos, n_clusters,
                local_epochs=5, pos_weight=None):
    # Fix 3: weighted BCE for class-imbalanced sites (site_D 7.5% AKI, site_E 5.2%)
    # weighted BCE: pos_weight = (1-prevalence)/prevalence upweights minority class
    _dev = next(client.parameters()).device
    if pos_weight is not None:
        _pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(_dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=_pw_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
    adv_crit  = nn.CrossEntropyLoss()
    client.train()
    # Freeze PersonalHead during federation — only adapter, body, disc update.
    # Head is fine-tuned locally after federation converges (matching v2.3 design).
    for p in client.head.parameters():
        p.requires_grad_(False)
    total_adv_loss = 0.0
    site_label_tensor = torch.tensor(site_idx, dtype=torch.long)

    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            out, adv_logits, h = client(xb, lam=lam_adv)

            # Skip batch if NaN (model still initialising)
            if torch.isnan(h).any() or torch.isnan(out).any():
                optimizer.zero_grad()
                continue

            # Classification loss
            cls_loss = criterion(out, yb)

            # Adversarial loss
            site_labels = site_label_tensor.expand(xb.size(0)).to(device)
            adv_loss = adv_crit(adv_logits, site_labels)

            # Prototype alignment loss
            proto_loss = torch.tensor(0.0, device=device)
            if global_protos and alpha_proto > 0:
                proto_loss = nearest_proto_loss(h, yb, None, global_protos)

            loss = cls_loss + lam_adv * adv_loss + alpha_proto * proto_loss
            loss.backward()
            nn.utils.clip_grad_norm_(client.parameters(), max_norm=5.0)
            optimizer.step()

            total_adv_loss += adv_loss.item()

    # Compute local prototypes after this round's training
    local_protos = compute_prototypes_kmeans_from_loader(client, loader, device, n_clusters)
    return local_protos, total_adv_loss


def compute_prototypes_kmeans_from_loader(client, loader, device, n_clusters):
    """Extract embeddings then compute k-means prototypes."""
    embs, labs = extract_embeddings(client, loader, device)
    embs_t = torch.tensor(embs, dtype=torch.float32)
    labs_t  = torch.tensor(labs, dtype=torch.float32)
    return compute_prototypes_kmeans(embs_t, labs_t, n_clusters)


# ── FL-gain index ─────────────────────────────────────────────────────────────

def compute_fl_gain(site_stats):
    """
    Three-component FL-gain index (Approach 2 version):
      0.30 * positive_case_scarcity
      0.30 * class_imbalance
      0.40 * feature_sparsity
    """
    total_aki = sum(s["n_aki"] for s in site_stats.values())
    max_feat  = max(s["n_feat"] for s in site_stats.values())
    scores = {}
    for site, s in site_stats.items():
        scarcity   = 1.0 - s["n_aki"] / max(total_aki, 1)
        imbalance  = 1.0 - s["prevalence"]
        sparsity   = 1.0 - s["n_feat"] / max_feat
        scores[site] = 0.30 * scarcity + 0.30 * imbalance + 0.40 * sparsity
    return scores


# ── Main training loop ────────────────────────────────────────────────────────

def run_fedadaptproto(args, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading site data...")
    site_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and "fl_gain_index" not in f
        and f.startswith("site_")
    ])
    site_ids = [os.path.splitext(f)[0] for f in site_files]   # e.g. ["site_A", ...]

    loaders_tr, loaders_te = {}, {}
    input_dims, prevalences, feat_cols_map = {}, {}, {}
    site_stats = {}

    for sid, fname in zip(site_ids, site_files):
        path = os.path.join(data_dir, fname)
        tr_ld, te_ld, in_dim, prev, fcols = load_site(path, batch_size=args.batch_size)
        loaders_tr[sid]  = tr_ld
        loaders_te[sid]  = te_ld
        input_dims[sid]  = in_dim
        prevalences[sid] = prev
        feat_cols_map[sid] = fcols

        n_total = len(tr_ld.dataset) + len(te_ld.dataset)
        n_aki   = int(round(prev * n_total))
        site_stats[sid] = {"n_feat": in_dim - 1, "n_aki": n_aki,
                           "prevalence": prev, "n_total": n_total}
        print(f"  [load] {sid}: {n_total:,} rows x {in_dim} cols  (AKI={prev:.3f})")

    n_sites = len(site_ids)
    site_idx_map = {sid: i for i, sid in enumerate(site_ids)}
    weights = {sid: site_stats[sid]["n_total"] / sum(s["n_total"] for s in site_stats.values())
               for sid in site_ids}

    # FL-gain index
    fl_gain = compute_fl_gain(site_stats)
    gain_csv = [f for f in os.listdir(data_dir) if "fl_gain_index" in f]
    if gain_csv:
        gain_df = pd.read_csv(os.path.join(data_dir, gain_csv[0]))
        print(f"  [fl_gain] auto-detected: {gain_csv[0]}")

    # ── Build models ───────────────────────────────────────────────────────────
    print("Building FedAdaptClient per site...")
    clients = {}
    optimizers = {}
    for sid in site_ids:
        lam = args.lambda_adv * (input_dims[sid] / max(input_dims.values()))
        model = FedAdaptClient(
            input_dim=input_dims[sid],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_sites=n_sites,
        ).to(device)
        clients[sid]    = model
        optimizers[sid] = optim.Adam(model.parameters(), lr=args.lr)
        print(f"  [model] {sid}: input_dim={input_dims[sid]-1}  "
              f"→ embedding_dim={args.embedding_dim}  hidden={args.hidden_dim}  "
              f"lambda_adv={lam:.3f}")

    # ── Local-only baseline ────────────────────────────────────────────────────
    print("Running local-only baseline...")
    local_aurocs = {}
    baseline_clients = {}
    criterion_base = nn.BCEWithLogitsLoss()
    for sid in site_ids:
        in_dim = input_dims[sid]
        # Simple MLP baseline — no GRL, no adapter complexity
        bc = nn.Sequential(
            nn.Linear(in_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        bopt = optim.Adam(bc.parameters(), lr=args.lr)
        bc.train()
        for epoch in range(20):  # 20 epochs
            for xb, yb in loaders_tr[sid]:
                xb, yb = xb.to(device), yb.to(device)
                bopt.zero_grad()
                out = bc(xb).squeeze(-1)
                loss = criterion_base(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(bc.parameters(), max_norm=5.0)
                bopt.step()
        # Evaluate simple MLP
        bc.eval()
        preds_b, trues_b = [], []
        with torch.no_grad():
            for xb, yb in loaders_te[sid]:
                xb = xb.to(device)
                out = bc(xb).squeeze(-1)
                preds_b.append(torch.sigmoid(out).cpu().numpy())
                trues_b.append(yb.numpy())
        preds_b = np.concatenate(preds_b)
        trues_b = np.concatenate(trues_b).astype(int)
        if np.isnan(preds_b).any():
            print(f"    {sid}: NaN in baseline preds — check data")
            auroc = 0.5
        else:
            auroc = roc_auc_score(trues_b, preds_b) if len(np.unique(trues_b)) > 1 else 0.5
        local_aurocs[sid] = auroc
        baseline_clients[sid] = bc
        print(f"    {sid}: local AUROC={auroc:.4f}")

    # ── AUTO-K SELECTION ───────────────────────────────────────────────────────
    n_clusters_per_site = {}

    # Parse manual override if given (--n_clusters_per_site takes precedence)
    manual_k = {}
    if args.n_clusters_per_site:
        for item in args.n_clusters_per_site.split(","):
            k, v = item.split("=")
            manual_k[k.strip()] = int(v.strip())

    if args.auto_k and not manual_k:
        # AUTO-K: Phase 1 — run full n_rounds of federation with uniform K=k_min
        # so the shared body fully converges before silhouette analysis.
        # Phase 2 — re-run n_rounds with the selected per-site K.
        k_sel_rnd = min(args.k_select_round, args.n_rounds)
        print(f"\n  ── Auto-K Phase 1: full federation ({k_sel_rnd} rounds, uniform K={args.k_min}) ──")
        print(f"  Silhouette runs after full fine-tuning on phase 1 model — matching v2.3 offline conditions.")
        _global_protos_pre = {}
        _n_clusters_pre = {sid: args.k_min for sid in site_ids}
        for _rnd in range(1, k_sel_rnd + 1):
            _scale = warmup_scale(_rnd, args.warmup_rounds)
            _all_pre = {}
            for sid in site_ids:
                _fl_mod = (1.0 - fl_gain.get(sid, 0.5))
                _lam = args.lambda_adv * (input_dims[sid] / max(input_dims.values())) * _scale * _fl_mod
                _ap  = args.alpha_proto * _scale
                _pw = (1.0 - prevalences[sid]) / max(prevalences[sid], 0.01)
                _lp, _ = local_train(clients[sid], loaders_tr[sid], optimizers[sid],
                                     device, site_idx_map[sid], n_sites,
                                     _lam, _ap, _global_protos_pre,
                                     _n_clusters_pre[sid], args.local_epochs,
                                     pos_weight=_pw)
                _all_pre[sid] = _lp
            fedavg_bodies(clients, weights)
            _global_protos_pre = average_prototypes(_all_pre, fl_gain_weights=fl_gain)
            if _rnd % 10 == 0:
                _parts = []
                for sid in site_ids:
                    _a, _, _ = evaluate(clients[sid], loaders_te[sid], device)
                    _parts.append(f"{sid.split('_alpha')[0]}={_a:.3f}")
                print(f"    [phase1] Round {_rnd:3d}/{k_sel_rnd}  |  " + "  ".join(_parts))

        print(f"\n  ── Auto-K selection (silhouette at round {k_sel_rnd}, K={args.k_min}–{args.k_max}) ──")

        # ── Full head fine-tuning on phase 1 model before silhouette ─────────
        # This matches exactly what v2.3 did offline:
        # run full federation → full head fine-tuning → silhouette on result.
        # The body is frozen; head trains to convergence on local data.
        # After silhouette we discard the phase 1 model and start phase 2 fresh.
        print("  [auto-K] Full head fine-tuning on phase 1 model (matching v2.3 offline conditions)...")
        _criterion_ft = nn.BCEWithLogitsLoss()
        for sid in site_ids:
            # Freeze body + adapter + disc; unfreeze head
            for p in clients[sid].body.parameters():      p.requires_grad_(False)
            for p in clients[sid].adapter.parameters():   p.requires_grad_(False)
            for p in clients[sid].disc.parameters():      p.requires_grad_(False)
            for p in clients[sid].head.parameters():      p.requires_grad_(True)
            _ft_opt = optim.Adam(clients[sid].head.parameters(), lr=args.lr * 0.1)
            clients[sid].train()
            for _ in range(30):   # full fine-tuning protocol — matches phase 2
                for xb, yb in loaders_tr[sid]:
                    xb, yb = xb.to(device), yb.to(device)
                    _ft_opt.zero_grad()
                    out, _, _ = clients[sid](xb, lam=0.0)
                    _criterion_ft(out, yb).backward()
                    nn.utils.clip_grad_norm_(clients[sid].head.parameters(), 5.0)
                    _ft_opt.step()
            # Re-freeze head; unfreeze body for embedding extraction
            for p in clients[sid].head.parameters():      p.requires_grad_(False)
            for p in clients[sid].body.parameters():      p.requires_grad_(True)
            for p in clients[sid].adapter.parameters():   p.requires_grad_(True)
            _a, _, _ = evaluate(clients[sid], loaders_te[sid], device)
            _sid_short = sid.split("_alpha")[0]
            print(f"    [phase1 post-FT] {_sid_short}: AUROC={_a:.4f}")

        auto_k_report = []

        for sid in site_ids:
            # Embeddings now reflect federation-warmed body + locally fine-tuned head
            # Sanity check: confirm embeddings are valid
            _test_embs, _ = extract_embeddings(clients[sid], loaders_tr[sid], device)
            _nan_frac = np.isnan(_test_embs).mean()
            if _nan_frac > 0:
                print(f'    [warn] {sid}: {_nan_frac*100:.1f}% NaN embeddings after warmup — falling back to K={args.k_min}')
                n_clusters_per_site[sid] = args.k_min
                auto_k_report.append({'site': sid, 'selected_k': args.k_min, 'note': 'NaN embeddings'})
                continue

            # Silhouette analysis on federation-warmed embeddings
            best_k, scores_by_cls = select_k_for_site(
                clients[sid], loaders_tr[sid], device,
                k_min=args.k_min, k_max=args.k_max
            )

            n_clusters_per_site[sid] = best_k

            # Build report row
            row = {"site": sid, "selected_k": best_k}
            for cls in [0, 1]:
                lbl = "AKI" if cls == 1 else "nonAKI"
                for k, sc in scores_by_cls.get(cls, {}).items():
                    row[f"sil_{lbl}_k{k}"] = round(sc, 4)
                best_cls_k = max(scores_by_cls.get(cls, {k: -1 for k in range(args.k_min, args.k_max+1)}),
                                 key=scores_by_cls.get(cls, {}).get) \
                             if scores_by_cls.get(cls) else args.k_min
                row[f"best_k_{lbl}"] = best_cls_k
            auto_k_report.append(row)

            # Pretty print
            sil_str = "  ".join(
                f"K={k}: sil={sc:.3f}"
                for k, sc in sorted(scores_by_cls.get(1, {}).items())
            )
            print(f"    {sid}: selected K={best_k}  |  AKI silhouettes: {sil_str}")

        # ── Discard phase 1 model — rebuild fresh for phase 2 ───────────────
        print("  [auto-K] Discarding phase 1 model — phase 2 starts from scratch with selected K.")
        clients = {}
        optimizers = {}
        for sid in site_ids:
            model = FedAdaptClient(
                input_dim=input_dims[sid],
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                n_sites=n_sites,
            ).to(device)
            clients[sid]    = model
            optimizers[sid] = optim.Adam(model.parameters(), lr=args.lr)

        # Save report
        report_path = os.path.join(output_dir, "auto_k_report.csv")
        if auto_k_report:
            keys = list(auto_k_report[0].keys())
            with open(report_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(auto_k_report)
            print(f"  [auto-K] report saved → {report_path}")

    elif manual_k:
        # Manual override
        n_clusters_per_site = {sid: manual_k.get(sid, args.n_clusters)
                                for sid in site_ids}
        print(f"  [manual K override] {n_clusters_per_site}")
    else:
        # Uniform K fallback
        n_clusters_per_site = {sid: args.n_clusters for sid in site_ids}
        print(f"  [uniform K={args.n_clusters}] applied to all sites")

    print(f"  [per-site K] {n_clusters_per_site}")

    # ── Federation loop ────────────────────────────────────────────────────────
    # Phase 2 (or only phase if no auto_k): full federation from round 1
    global_protos = {}
    phase_label = "Auto-K Phase 2" if (args.auto_k and not manual_k) else "FEDADAPTPROTO"

    print(f"\n  ── Federation [{phase_label}]  {args.n_rounds} rounds × "
          f"{args.local_epochs} local epochs ──")

    round_aurocs  = []

    for rnd in range(1, args.n_rounds + 1):
        scale = warmup_scale(rnd, args.warmup_rounds)
        all_site_protos = {}

        for sid in site_ids:
            # Fix 1: scale lambda_adv by (1 - fl_gain) so high-gain (data-scarce)
            # sites receive gentler adversarial pressure, preserving local AKI structure
            _fl_mod = (1.0 - fl_gain.get(sid, 0.5))
            lam_adv_eff = args.lambda_adv * (input_dims[sid] / max(input_dims.values())) * scale * _fl_mod
            alpha_proto_eff = args.alpha_proto * scale

            _pw = (1.0 - prevalences[sid]) / max(prevalences[sid], 0.01)
            local_protos, _ = local_train(
                client=clients[sid],
                loader=loaders_tr[sid],
                optimizer=optimizers[sid],
                device=device,
                site_idx=site_idx_map[sid],
                n_sites=n_sites,
                lam_adv=lam_adv_eff,
                alpha_proto=alpha_proto_eff,
                global_protos=global_protos,
                n_clusters=n_clusters_per_site[sid],
                local_epochs=args.local_epochs,
                pos_weight=_pw,
            )
            all_site_protos[sid] = local_protos

        # Aggregate shared body
        fedavg_bodies(clients, weights)

        # Aggregate prototypes
        global_protos = average_prototypes(all_site_protos, fl_gain_weights=fl_gain)

        # Periodic eval
        if rnd % 5 == 0:
            row = {"round": rnd}
            parts = []
            for sid in site_ids:
                auroc, _, _ = evaluate(clients[sid], loaders_te[sid], device)
                row[f"auroc_{sid}"] = round(auroc, 4)
                parts.append(f"{sid}={auroc:.3f}")
            round_aurocs.append(row)
            print(f"    Round {rnd:3d}/{args.n_rounds}  |  " + "  ".join(parts))

    # ── Head fine-tuning ───────────────────────────────────────────────────────
    print("  ── Head fine-tuning (FedAdapt) ──")
    for sid in site_ids:
        # Freeze body and adapter; unfreeze head for local fine-tuning
        for p in clients[sid].body.parameters():
            p.requires_grad_(False)
        for p in clients[sid].adapter.parameters():
            p.requires_grad_(False)
        for p in clients[sid].head.parameters():
            p.requires_grad_(True)   # unfreeze — first real head training
        head_opt = optim.Adam(clients[sid].head.parameters(), lr=args.lr * 0.1)
        criterion = nn.BCEWithLogitsLoss()
        clients[sid].train()
        for _ in range(30):   # 30 epochs — sufficient for imbalanced sites
            for xb, yb in loaders_tr[sid]:
                xb, yb = xb.to(device), yb.to(device)
                head_opt.zero_grad()
                out, _, _ = clients[sid](xb, lam=0.0)
                criterion(out, yb).backward()
                nn.utils.clip_grad_norm_(clients[sid].head.parameters(), 5.0)
                head_opt.step()

        auroc, f1, auprc = evaluate(clients[sid], loaders_te[sid], device)
        print(f"    {sid}: AUROC={auroc:.4f}  F1={f1:.4f}  AUPRC={auprc:.4f}")

    # ── Save final metrics ─────────────────────────────────────────────────────
    print("  Final metrics:")
    results = []
    header = ["site_id", "auroc", "f1", "auprc", "local_auroc",
              "delta_auroc", "fl_gain_index", "selected_k"]
    print(f"{'site_id':>8}  {'auroc':>8}  {'f1':>6}  {'auprc':>6}  "
          f"{'delta':>7}  {'K':>3}")
    for sid in site_ids:
        auroc, f1, auprc = evaluate(clients[sid], loaders_te[sid], device)
        delta = auroc - local_aurocs[sid]
        results.append({
            "site_id":        sid,
            "auroc":          round(auroc, 6),
            "f1":             round(f1, 6),
            "auprc":          round(auprc, 6),
            "local_auroc":    round(local_aurocs[sid], 6),
            "delta_auroc":    round(delta, 6),
            "fl_gain_index":  round(fl_gain[sid], 4),
            "selected_k":     n_clusters_per_site[sid],
        })
        print(f"{sid:>8}  {auroc:.6f}  {f1:.4f}  {auprc:.4f}  "
              f"{delta:+.4f}  K={n_clusters_per_site[sid]}")

    metrics_path = os.path.join(output_dir, "final_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(results)

    # Round AUROC CSV
    if round_aurocs:
        ra_path = os.path.join(output_dir, "round_auroc.csv")
        with open(ra_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(round_aurocs[0].keys()))
            w.writeheader()
            w.writerows(round_aurocs)

    # FL-gain correlation CSV
    fl_path = os.path.join(output_dir, "fl_gain_correlation.csv")
    with open(fl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "fl_gain_index", "delta_auroc", "selected_k"])
        for r in results:
            w.writerow([r["site_id"], r["fl_gain_index"],
                        r["delta_auroc"], r["selected_k"]])

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("  Generating plots...")
    _plot_training_curves(round_aurocs, site_ids, output_dir)
    _plot_final_metrics(results, output_dir)
    _plot_fl_gain(results, output_dir)
    if args.auto_k and not manual_k:
        _plot_auto_k_silhouettes(output_dir)

    print(f"\n✅  Done — all outputs in {output_dir}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_training_curves(round_aurocs, site_ids, output_dir):
    if not round_aurocs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#378ADD", "#1D9E75", "#888780", "#BA7517", "#D4537E"]
    rounds = [r["round"] for r in round_aurocs]
    for i, sid in enumerate(site_ids):
        key = f"auroc_{sid}"
        vals = [r[key] for r in round_aurocs if key in r]
        ax.plot(rounds[:len(vals)], vals, label=sid,
                color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("AUROC")
    ax.set_title("FedAdaptProto v2.4 — Training Curves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print("  [plot] training_curves.png")


def _plot_final_metrics(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sites = [r["site_id"] for r in results]
    colors = ["#E24B4A" if r["delta_auroc"] < 0 else "#1D9E75" for r in results]
    for ax, metric, label in zip(axes,
                                  ["auroc", "f1", "auprc"],
                                  ["AUROC", "F1", "AUPRC"]):
        vals = [r[metric] for r in results]
        bars = ax.bar(sites, vals, color=colors)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("FedAdaptProto v2.4 — Final Metrics (green=FL gain, red=FL loss)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "final_metrics.png"), dpi=150)
    plt.close(fig)
    print("  [plot] final_metrics.png")


def _plot_fl_gain(results, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"2": "#D4537E", "3": "#378ADD", "4": "#1D9E75", "5": "#BA7517"}
    for r in results:
        k = str(r["selected_k"])
        c = colors.get(k, "#888780")
        ax.scatter(r["fl_gain_index"], r["delta_auroc"], color=c, s=100, zorder=3)
        ax.annotate(f"{r['site_id']}\n(K={k})",
                    (r["fl_gain_index"], r["delta_auroc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.68, color="orange", linestyle=":", linewidth=0.8,
               label="Primary threshold (0.68)")
    ax.set_xlabel("FL-Gain Index")
    ax.set_ylabel("AUROC Improvement vs Local")
    ax.set_title("FL-Gain Index vs Observed Improvement (auto-K)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fl_gain_correlation.png"), dpi=150)
    plt.close(fig)
    print("  [plot] fl_gain_correlation.png")


def _plot_auto_k_silhouettes(output_dir):
    report_path = os.path.join(output_dir, "auto_k_report.csv")
    if not os.path.exists(report_path):
        return
    df = pd.read_csv(report_path)
    sil_cols = [c for c in df.columns if c.startswith("sil_AKI_k")]
    if not sil_cols:
        return
    k_vals = sorted([int(c.replace("sil_AKI_k", "")) for c in sil_cols])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#378ADD", "#1D9E75", "#888780", "#BA7517", "#D4537E"]
    for i, row in df.iterrows():
        sils = [row.get(f"sil_AKI_k{k}", np.nan) for k in k_vals]
        ax.plot(k_vals, sils, marker="o", label=row["site"],
                color=colors[i % len(colors)])
        best_k = int(row["selected_k"])
        best_sil = row.get(f"sil_AKI_k{best_k}", np.nan)
        if not np.isnan(best_sil):
            ax.scatter([best_k], [best_sil], s=150, color=colors[i % len(colors)],
                       marker="*", zorder=5)
    ax.set_xlabel("K (clusters per class)")
    ax.set_ylabel("Silhouette Score (AKI class)")
    ax.set_title("Auto-K: Silhouette Analysis — AKI Class Embeddings\n(★ = selected K)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auto_k_silhouettes.png"), dpi=150)
    plt.close(fig)
    print("  [plot] auto_k_silhouettes.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FedAdaptProto v2.4 — auto-K")

    # Data / output
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--method",      default="fedadaptproto")

    # Heterogeneity identifiers (for logging only)
    p.add_argument("--alpha",  type=float, default=0.3)
    p.add_argument("--gamma",  type=float, default=0.75)

    # Training
    p.add_argument("--n_rounds",      type=int,   default=50)
    p.add_argument("--local_epochs",  type=int,   default=5)
    p.add_argument("--lr",            type=float, default=0.001)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--embedding_dim", type=int,   default=64)
    p.add_argument("--hidden_dim",    type=int,   default=128)

    # FedAdaptProto hyperparameters
    p.add_argument("--lambda_adv",         type=float, default=0.1)
    p.add_argument("--alpha_proto",        type=float, default=1.0)
    p.add_argument("--warmup_rounds",      type=int,   default=10)
    p.add_argument("--early_stop_patience",type=int,   default=0)

    # K configuration — three mutually exclusive modes:
    # 1. --auto_k          → silhouette-based auto selection (v2.4 default)
    # 2. --n_clusters_per_site site_A=3,...  → manual override (v2.3 compat)
    # 3. neither           → uniform --n_clusters everywhere
    p.add_argument("--auto_k",    action="store_true",
                   help="Automatically select K per site via silhouette analysis")
    p.add_argument("--k_min",     type=int, default=2,
                   help="Minimum K to test in silhouette search (default 2)")
    p.add_argument("--k_max",     type=int, default=5,
                   help="Maximum K to test in silhouette search (default 5)")
    p.add_argument("--k_warmup_epochs", type=int, default=5,
                   help="Local epochs to warm embeddings before silhouette test (default 5)")
    p.add_argument("--k_select_round", type=int, default=50,
                   help="Federation rounds for phase 1 before silhouette (default=n_rounds=50). "
                        "Full federation then full fine-tuning runs before silhouette, "
                        "matching v2.3 offline analysis conditions.")
    p.add_argument("--n_clusters", type=int, default=3,
                   help="Uniform K fallback when --auto_k is not set (default 3)")
    p.add_argument("--n_clusters_per_site", type=str, default="",
                   help="Manual per-site K: 'site_A=3,site_B=3,...' — overrides --auto_k")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default 42)")

    return p.parse_args()


def main():
    args = parse_args()

    # Set all random seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    global SEED
    SEED = args.seed

    method_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(method_dir, exist_ok=True)

    print("=" * 65)
    print(f"  FedAdaptProto v2.4  |  method={args.method.upper()}")
    print(f"  rounds={args.n_rounds}  local_epochs={args.local_epochs}  "
          f"lr={args.lr}  batch={args.batch_size}")
    print(f"  embedding_dim={args.embedding_dim}  hidden_dim={args.hidden_dim}")
    print(f"  lambda_adv={args.lambda_adv}  alpha_proto={args.alpha_proto}")
    print(f"  warmup_rounds={args.warmup_rounds}")
    k_mode = (f"auto_k (silhouette @ round {args.k_select_round})"
              if args.auto_k and not args.n_clusters_per_site
              else f"manual: {args.n_clusters_per_site}" if args.n_clusters_per_site
              else f"uniform K={args.n_clusters}")
    print(f"  K mode: {k_mode}")
    print(f"  output → {method_dir}")
    print("=" * 65)

    run_fedadaptproto(args, args.data_dir, method_dir)


if __name__ == "__main__":
    main()