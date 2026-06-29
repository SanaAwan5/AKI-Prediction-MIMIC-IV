"""
silhouette_diagnostic.py
========================
Compute per-site silhouette scores over EMBEDDINGS extracted from a trained
FedAdaptProto model. Tells you, for each site and each class, how multi-modal
the learned representation is — i.e., what K the multi-cluster prototype
mechanism should adaptively choose per site.

Output:
  silhouette_scores.csv   — per (site, class, K) silhouette score
  silhouette_scores.png   — chart: silhouette vs K, per site, per class

Usage
-----
    python3 silhouette_diagnostic.py \
        --data_dir   ./phase2_sites_approach2/alpha0.3_gamma0.75 \
        --models_dir ./results_phase2_approach2/alpha0.3_gamma0.75/fedadaptproto/fedadaptproto \
        --alpha 0.3 --gamma 0.75 \
        --output_dir ./silhouette_a0.3_g0.75/

Notes
-----
- Uses the EXACT same preprocessing as fedadapt_train_approach2.py:
  site-local median imputation, site-local z-score standardisation.
- Uses the TEST split (matching train.py's split with the same seed) so the
  silhouettes are computed on data the model never saw during training.
- silhouette_score requires K >= 2; for K=1 we report None (no clustering).
- If a class has fewer than K samples at any site, that (site, class, K)
  combo is skipped (logged to stderr).
- Embedding extraction runs CPU-only — fast enough; ~30s per site for ~3k
  test patients per class. No GPU required.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# import the model — must match the training script's import path
from fedadapt_model_approach2 import FedAdaptClient


# ── preprocessing: must match fedadapt_train_approach2.py:SiteData ────────────

def load_site_features(
    csv_path: Path,
    feature_names: List[str],
    label_col: str,
    test_frac: float = 0.20,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Reproduce SiteData's preprocessing and test-split logic exactly.

    Returns: {"X_test": Tensor[N_test, D], "y_test": Tensor[N_test]}
    """
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in feature_names if c in df.columns]
    df_feat   = df[feat_cols].copy()

    # site-local median imputation (same as SiteData)
    binary_cols  = [c for c in feat_cols if df_feat[c].dropna().isin([0, 1]).all()]
    numeric_cols = [c for c in feat_cols if c not in binary_cols]
    site_medians = df_feat[numeric_cols].median()
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(site_medians)
    df_feat[binary_cols]  = df_feat[binary_cols].fillna(0)
    X = df_feat.values.astype(np.float32)

    # site-local z-score (same as SiteData)
    mu  = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-8
    X   = (X - mu) / sig

    y = df[label_col].fillna(0).values.astype(np.float32)

    # same test split as SiteData (seed=42 by default, matches train.py)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = max(1, int(len(df) * test_frac))
    test_idx = idx[:n_test]

    return {
        "X_test": torch.tensor(X[test_idx]),
        "y_test": torch.tensor(y[test_idx]),
    }


# ── embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model_path: Path,
    meta: dict,
    X_test: torch.Tensor,
) -> np.ndarray:
    """
    Load a trained FedAdaptClient checkpoint, forward X_test through it,
    return the SharedBody output (embeddings).
    """
    model = FedAdaptClient.from_metadata(meta)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    # checkpoint is the full state_dict from client.state_dict()
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    _, _, emb = model.forward_with_embedding(X_test)
    return emb.cpu().numpy()


# ── silhouette computation ────────────────────────────────────────────────────

def silhouette_for_class(
    X_emb_c: np.ndarray,
    K_max:   int,
    seed:    int,
) -> Dict[int, float]:
    """
    Returns {K: silhouette_score} for K in 2..K_max. K=1 is undefined
    for silhouette (single cluster has no separation), so we omit it.

    If fewer than K samples, that K is skipped.
    """
    out: Dict[int, float] = {}
    n = X_emb_c.shape[0]
    if n < 2:
        return out
    for K in range(2, K_max + 1):
        if n < K:
            continue
        km     = KMeans(n_clusters=K, random_state=seed, n_init=10)
        labels = km.fit_predict(X_emb_c)
        if len(set(labels)) < 2:    # k-means collapsed
            continue
        s = silhouette_score(X_emb_c, labels, metric="euclidean")
        out[K] = float(s)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=Path, required=True,
                        help="Folder with site_*_alpha{α}_gamma{γ}.csv + *_adapter_meta.json")
    parser.add_argument("--models_dir", type=Path, required=True,
                        help="Folder with trained site_*_model.pt files")
    parser.add_argument("--alpha",      type=float, required=True)
    parser.add_argument("--gamma",      type=float, required=True)
    parser.add_argument("--label_col",  type=str, default="AKI_label")
    parser.add_argument("--K_max",      type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--test_frac",  type=float, default=0.20)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    meta_paths = sorted(args.data_dir.glob("*_adapter_meta.json"))
    if not meta_paths:
        print(f"No *_adapter_meta.json files in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    for meta_path in meta_paths:
        site_id = meta_path.stem.replace("_adapter_meta", "")
        meta    = json.loads(meta_path.read_text())

        csv_path   = args.data_dir / f"{site_id}_alpha{args.alpha}_gamma{args.gamma}.csv"
        model_path = args.models_dir / f"{site_id}_model.pt"

        if not csv_path.exists():
            print(f"[skip] {csv_path} missing", file=sys.stderr); continue
        if not model_path.exists():
            print(f"[skip] {model_path} missing", file=sys.stderr); continue

        print(f"\n→ {site_id}")
        site = load_site_features(csv_path, meta["feature_names"], args.label_col,
                                   test_frac=args.test_frac, seed=args.seed)
        emb = extract_embeddings(model_path, meta, site["X_test"])
        y   = site["y_test"].numpy().astype(int)

        for class_name, class_val in [("aki", 1), ("no_aki", 0)]:
            mask = (y == class_val)
            n_c = int(mask.sum())
            emb_c = emb[mask]
            print(f"  class={class_name}: n={n_c}")
            scores = silhouette_for_class(emb_c, args.K_max, args.seed)
            for K, s in scores.items():
                rows.append({
                    "site":            site_id,
                    "class":           class_name,
                    "n_in_class":      n_c,
                    "K":               K,
                    "silhouette":      s,
                })
                print(f"    K={K}: silhouette = {s:+.4f}")

    if not rows:
        print("No silhouette rows computed.", file=sys.stderr); sys.exit(1)

    df_out = pd.DataFrame(rows)
    csv_out = args.output_dir / "silhouette_scores.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved {csv_out}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=180, sharey=True)
    sites_order = sorted(df_out["site"].unique())
    cmap = plt.get_cmap("tab10")
    site_color = {s: cmap(i) for i, s in enumerate(sites_order)}

    for ax, cls in zip(axes, ["aki", "no_aki"]):
        sub = df_out[df_out["class"] == cls]
        for site in sites_order:
            ss = sub[sub["site"] == site].sort_values("K")
            if ss.empty: continue
            ax.plot(ss["K"], ss["silhouette"], "-o",
                    color=site_color[site], label=site, linewidth=1.8, markersize=7)
        ax.set_title(f"Class = {cls}  (silhouette over EMBEDDINGS)", fontsize=11)
        ax.set_xlabel("K (number of clusters)", fontsize=10)
        ax.axhline(0.5, color="#888", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.axhline(0.25, color="#888", linewidth=0.6, linestyle=":", alpha=0.4)
        ax.grid(alpha=0.2)
        ax.set_xticks(range(2, args.K_max + 1))
    axes[0].set_ylabel("Silhouette score", fontsize=10)
    axes[0].legend(loc="best", fontsize=9, frameon=True)

    fig.suptitle(
        f"Embedding silhouette per site, per class  "
        f"(α={args.alpha}, γ={args.gamma})  —  higher = more cluster-coherent",
        fontsize=11, y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    png_out = args.output_dir / "silhouette_scores.png"
    plt.savefig(png_out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {png_out}")

    # ── recommendation: optimal K per site/class ─────────────────────────────
    print("\nOptimal K per (site, class) — silhouette argmax:")
    opt = df_out.loc[df_out.groupby(["site","class"])["silhouette"].idxmax()]
    opt = opt.sort_values(["site","class"])
    for _, r in opt.iterrows():
        print(f"  {r['site']:8s} {r['class']:6s} → K={int(r['K'])}  (silhouette={r['silhouette']:+.4f}, n={int(r['n_in_class'])})")


if __name__ == "__main__":
    main()