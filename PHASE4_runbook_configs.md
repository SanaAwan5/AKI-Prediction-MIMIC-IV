# FTL/AKI Phase 4 — Runbook (data prep + training)

---

## Script 1 — Data simulation

**`mimic_ftl_simulation_phase4_gpc_aligned.py`**

Builds the 6 real-GPC-site datasets from the MIMIC-IV cohort CSV. Every site
gets an identical baseline (`shared98_core` — 18 GPC-matched columns —
+ `gpc_dx_universal` — 70 universal ICD-9 codes) plus its own site-specific
ICD-9 codes (`dx_site_*`) layered on top.

**Prerequisite:** the cohort notebook (`AKI_Anchor_Based_Approach2_aligned_
features_FIXED.ipynb`, Sections 5B + 5C) must have been re-run against
BigQuery so the input CSV actually has the `dx_*`/`dx_site_*` columns. The
script degrades gracefully (prints a warning, doesn't crash) if they're
missing — worth checking console output for `+ N site-specific DX codes (of
N expected)` per site to confirm they landed, since a clean exit doesn't
guarantee that.

| Flag | Default | Meaning |
|---|---|---|
| `--input` | *required* | Path to the regenerated MIMIC-IV cohort CSV |
| `--label` | `AKI_label` | Label column name |
| `--alpha` | `0.5` | Dirichlet-style blend between each site's real archetype prevalence and the global rate. Lower α → closer to each site's own real rate (more non-IID); higher α → all sites converge toward the global rate (~13.3%) |
| `--gamma` | `0.75` | Covariate-shift severity applied on top of the prevalence targeting |
| `--seed` | `42` | RNG seed |
| `--embedding_dim` | `64` | Shared embedding dim written into each site's `_adapter_meta.json` (informational — doesn't affect the CSVs) |
| `--output` | `./phase4_gpc_aligned_sites/` | Output directory |
| `--sweep` | off | Runs all 5 α × 4 γ combinations (20 runs) into `output/alpha{a}_gamma{g}/` subfolders instead of one single combo |

**Single run:**
```bash
python3 mimic_ftl_simulation_phase4_gpc_aligned.py \
    --input  /path/to/aki_anchor_based_24h_lookback_aligned_features.csv \
    --alpha  0.5 \
    --gamma  0.75 \
    --output ./phase4_gpc_aligned_sites/
```

**Full sweep:**
```bash
python3 mimic_ftl_simulation_phase4_gpc_aligned.py \
    --input  /path/to/aki_anchor_based_24h_lookback_aligned_features.csv \
    --sweep \
    --output ./phase4_gpc_aligned_sites/
```
Sweep output lands in per-combo subfolders, e.g. `phase4_gpc_aligned_sites/
alpha0.1_gamma0.0/sim_KUMC_alpha0.1_gamma0.0.csv`.

**Output per run:** 6 site CSVs (`sim_KUMC`, `sim_MCW`, `sim_UIOWA`,
`sim_UPITT`, `sim_UTSW`, `sim_UofU`), `fl_gain_index_*.csv`, one
`*_adapter_meta.json` per site, plus summary/decomposition plots.

---

## Script 2 — Training

There are **two separate scripts**, not one script with version flags —
important, since they serve different purposes and one of them silently
ignores `--method`:

| | `fedadapt_train_approach2_v2_3_ftablation_taxtest.py` | `fedadapt_train_approach2_v2_5_grouptest.py` |
|---|---|---|
| Runs which method(s)? | **All 5**: fedadapt, fedadaptproto, fedavg, fedprox, scaffold — dispatches internally on `--method` | **fedadaptproto only.** `--method` only names the output subfolder — passing anything else still runs fedadaptproto training, just mislabels the folder. Confirmed by reading `main()`: it unconditionally calls `run_fedadaptproto(...)`. |
| Per-site K | **Manual only** — `--n_clusters_per_site` (or uniform `--n_clusters`). No auto-K in this script at all. | **Auto-K by default** (`--auto_k`, silhouette-based). Also accepts `--n_clusters_per_site` for v2.3-compatible manual override, or falls back to uniform `--n_clusters` if neither is set. |
| Use for | Baseline comparison (FedAdaptProto vs. FedAvg/FedProx/SCAFFOLD) | Best/current FedAdaptProto specifically |

### `fedadapt_train_approach2_v2_3_ftablation_taxtest.py` — full argument reference

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | *required* | Folder with the 6 site CSVs from Script 1 |
| `--label` | `AKI_label` | |
| `--alpha` | `0.5` | Logging only — should match the simulation run's α |
| `--gamma` | `0.75` | Logging only — should match the simulation run's γ |
| `--fl_gain_csv` | none | Path to `fl_gain_index_*.csv` from Script 1 (optional — used for weighting, if implemented) |
| `--method` | `fedadapt` | One of `fedadapt` / `fedadaptproto` / `fedavg` / `fedprox` / `scaffold` |
| `--rounds` | `50` | Federation rounds |
| `--local_epochs` | `1` | **Updated from the original `5`** — runD's confirmed free improvement, now the default in both `_ftablation.py` and `_ftablation_taxtest.py`. Pass `--local_epochs 5` to reproduce pre-runD behavior |
| `--finetune_epochs` | `30` | **Updated from `10`** — runD default. `--ft_lr_mult` (default `0.1`), `--ft_pos_weight`/`--ft_use_pos_weight` (default off), `--ft_grad_clip` (default `5.0`) are the other runD fine-tune-recipe flags, all new since this table was first written |
| `--discriminator_target` | `group` (both v2.3 and v2.5) | **New.** `site` = fixed site-identity label; `group` = per-row feature-group label. Confirmed `group` wins for BOTH v2.3 (+0.0664 vs +0.0589 for `site`) and v2.5 (+0.0197 vs +0.0035). An earlier version of this table stated an "opposite preferences" finding, based on results later found to be confounded by a fine-tune-recipe bug — that finding was retracted once corrected. |
| `--group_taxonomy` | `native` (v2.3) / `phase4_20group` (v2.5) | **New**, only relevant when `--discriminator_target group`. `phase4_20group` (16 ICD-9 chapters + 4 lab-panel groups) is the best-performing taxonomy found |
| `--group_class_weighting` | off | **New.** Inverse-frequency discriminator-loss weighting — helps genuinely imbalanced taxonomies, negligible effect on `phase4_20group` |
| `--embedding_dim` | `64` | |
| `--hidden_dim` | `128` | |
| `--lr` | `1e-3` | |
| `--batch_size` | `256` | |
| `--lambda_adv` | `0.1` | GRL max weight (FedAdapt/FedAdaptProto). Validated in Phase 3 — no retuning needed |
| `--adaptive_lambda` / `--no_adaptive_lambda` | `True` | Scale lambda_adv by site AKI prevalence |
| `--alpha_proto` | `0.5` | Prototype alignment loss weight (FedAdaptProto only). `0` disables it (reduces to plain FedAdapt) |
| `--mu` | `0.01` | Proximal weight (FedProx only) |
| `--warmup_rounds` | `10` | Rounds to linearly ramp lambda_adv/alpha_proto from 0 |
| `--early_stop_patience` | `2` | Per-round local-epoch early stop. `0` disables |
| `--track_gradient_conflict` | off | Diagnostic: saves per-round, per-site cosine similarity between local and aggregate updates |
| `--n_clusters` | `1` | Prototype clusters per class (FedAdaptProto). `K=1` = legacy single-centroid |
| `--n_clusters_per_site` | none | **Manual per-site K override** — comma-separated `sid=K` pairs. Sites not listed fall back to `--n_clusters`. ⚠️ Site IDs must match whatever your `--data_dir` actually contains |

**Phase 3's actual validated K settings (site_A..site_E naming — do NOT copy these
site IDs directly onto Phase 4 data, see warning below):**

Two flags combine: `--n_clusters` sets the default K applied to every site;
`--n_clusters_per_site` overrides specific sites on top of that default.

- Phase 3's per-site override, driven by the silhouette diagnostic (site_E's
  AKI embeddings are unimodal, peaking at K=2 — forcing K=3 there added a
  noise cluster):
  ```
  --n_clusters 3 --n_clusters_per_site "site_A=3,site_B=3,site_C=3,site_D=2,site_E=2"
  ```
- The two uniform-K sweeps that made it into Phase 3's final ranking table used
  no per-site override at all — just a single global value:
  - `fedadaptproto__kmode_uniform2` → `--n_clusters 2`
  - `fedadaptproto__kmode_uniform3` → `--n_clusters 3`

⚠️ **These site IDs (`site_A`..`site_E`) are Phase 3's 5-archetype naming and
don't carry over to Phase 4's 6 real-GPC-site data as-is** — Phase 4 uses
`sim_KUMC`/`sim_MCW`/`sim_UIOWA`/`sim_UPITT`/`sim_UTSW`/`sim_UofU`, one more
site, and no site plays site_E's specific "unimodal AKI embedding" role by
construction (that finding was specific to Phase 3's Rural archetype, not
derived for any of the 6 real sites). Before running Phase 4 with a
per-site override, the silhouette diagnostic needs to be re-run against the
actual Phase 4 site data to find which (if any) of the 6 real sites shows
the same unimodal pattern — mapping site_D/site_E's old K=2 assignment onto
two of the new site IDs by guesswork isn't grounded in anything. Uniform K
(`--n_clusters 2` or `--n_clusters 3`, no override) is the safer starting
point until that's done.
| `--test_frac` | `0.20` | |
| `--seed` | `42` | |
| `--output_dir` | `./results/` | Method subfolder created automatically under this |

**Example — full baseline comparison on one Phase 4 sweep combo**, using
uniform K=3 (Phase 3's actual headline-result setting, and the safer
starting point until the silhouette diagnostic is re-run on the real 6-site
data — see the per-site-override warning above):
```bash
for m in fedadapt fedadaptproto fedavg fedprox scaffold; do
    python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
        --data_dir    ./phase4_gpc_aligned_sites/alpha0.1_gamma0.0/ \
        --output_dir  ./results_phase4_v2_3_baselines/ \
        --method      $m \
        --n_clusters  3 \
        --alpha 0.1 --gamma 0.0
done
```
`--n_clusters` only affects `fedadaptproto`/`fedadapt` runs — harmless to
pass it for fedavg/fedprox/scaffold too, since those methods just ignore it.

### `fedadapt_train_approach2_v2_5_grouptest.py` — full argument reference

**Use this file** — it supersedes `fedadapt_train_approach2_v2_5.py`,
`_fixed.py`, `_ftablation.py`, and `_realarch.py`, all now retired. It's a
strict superset of each: real model architecture imported from
`fedadapt_model_approach2.py` (not v2.5's old separate inline classes), the
prototype-aggregation bug fixed, `torch.use_deterministic_algorithms` added
(byte-identical reruns verified), plus the `--discriminator_target` /
`--group_taxonomy` / `--group_class_weighting` flags from the full
discriminator-target investigation.

**Status, current as of the end of this investigation:** v2.5's federation
phase used to produce negative Δauroc on Phase 4 data across the board — that
specific problem (traced through several root causes: model architecture,
a prototype-aggregation bug, and a hidden `local_epochs=5`-vs-`1` confound)
is resolved. Best confirmed config now scores **+0.0197, 6/6 sites
positive** (`--discriminator_target group --group_taxonomy phase4_20group
--local_epochs 1`). It's still short of v2.3's best (+0.0664) by a real,
unexplained ~0.047 — several follow-up hypotheses to close that gap
(taxonomy merging, removing the diagnostic group, porting v2.3's
oversampling mechanism) were tried and backfired; see the disentangling
table below for the full history.

**CAUTION `--local_epochs`:** this script deliberately still defaults to
`5`, not `1` — always pass `--local_epochs 1 --n_rounds 50` explicitly. This
default was left as-is on purpose, specifically so nobody accidentally
reproduces the exact silent-mismatch confound that took a long time to find
in the first place.

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | *required* | |
| `--output_dir` | *required* | |
| `--method` | `fedadaptproto` | Cosmetic only — see table above |
| `--seed` | `42` | Vary across runs (each to its own `output_dir`) for multi-seed stability |
| `--alpha` / `--gamma` | `0.3` / `0.75` | Logging only |
| `--n_rounds` | `50` | |
| `--local_epochs` | `5` | **See caution above — always pass `1` explicitly** |
| `--lr` | `0.001` | |
| `--batch_size` | `256` | |
| `--embedding_dim` | `64` | |
| `--hidden_dim` | `128` | |
| `--baseline_seeds` | `5` | Seeds to average the shared local-only baseline over — cached in `data_dir`, computed once |
| `--baseline_epochs` | `20` | |
| `--force_baseline` | off | Retrain the cached local baseline even if one exists |
| `--lambda_adv` | `0.1` | |
| `--alpha_proto` | `1.0` | Note: different default from v2.3's `0.5` |
| `--warmup_rounds` | `10` | |
| `--early_stop_patience` | `0` | Note: disabled by default here, vs. v2.3's default of `2` |
| `--discriminator_target` | `site` | **Set to `group`** — confirmed better for v2.5 (+0.0197 vs +0.0035) |
| `--group_taxonomy` | `phase4_20group` | Only relevant when `--discriminator_target group`. Other options (`v23_original_plus_dx`, `v23_merged_plus_dx`, `v23_original_no_dx`) all underperformed this one — see disentangling table |
| `--group_class_weighting` | off | Negligible effect on `phase4_20group` specifically (isolation-tested, diff ~1e-6) — not needed for the recommended config |
| `--auto_k` | off | Silhouette-based automatic per-site K. Flag only, no value |
| `--k_min` / `--k_max` | `2` / `5` | Search range for auto-K |
| `--k_warmup_epochs` | `5` | Local epochs to warm embeddings before silhouette test |
| `--k_select_round` | `50` | Federation rounds before silhouette test runs |
| `--n_clusters` | `3` | Uniform K fallback when `--auto_k` not set |
| `--n_clusters_per_site` | `""` | Manual override, same format as v2.3 — **overrides `--auto_k` if both are set** (the script's own comment says so). Don't combine them — `--auto_k` alone is the clean way to actually test auto-K |

**Example — best confirmed v2.5 config:**
```bash
python3 fedadapt_train_approach2_v2_5_grouptest.py \
    --data_dir    ./phase4_gpc_aligned_sites/alpha0.5_gamma0.75/ \
    --output_dir  ./results_phase4_v2_5_best/ \
    --method      fedadaptproto \
    --discriminator_target group --group_taxonomy phase4_20group \
    --local_epochs 1 --n_rounds 50 \
    --alpha 0.5 --gamma 0.75 --seed 42
```

This file also has fine-tune-recipe ablation flags (`--ft_lr_mult`,
`--ft_epochs`, `--ft_pos_weight`, `--ft_grad_clip`) and the
`--no_lambda_fl_mod` flag mentioned above, all defaulting to v2.5's
original behavior — see the argument table above for what each does.

**Two things worth knowing about v2.5's auto-K, from Phase 3 experience:**
1. Despite Phase 3's ranking table labeling one config `fedadaptproto_v25__
   kmode_uniform3`, v2.5's auto-K isn't actually forced to K=3 — that label
   summarizes what auto-K *tended* to select for most sites at moderate
   heterogeneity conditions during Phase 3 development, not a fixed setting.
   If precise reproduction matters, don't read "uniform3" as literally
   `--n_clusters 3` with auto-K off — it's auto-K's typical behavior at that
   condition, and Phase 4's real 6-site data may select differently.
2. Phase 3's headline result (fedadaptproto_v25 uniform K=3, mean Δauroc
   −0.0054 ± 0.0004, multi-seed confirmed as the best of 6 methods) was
   genuinely uniform K=3 (`--n_clusters 3`, `--auto_k` off) — not the auto-K
   mode. 

---

## Parameters worth fine-tuning on Phase 4 data

**Scope** run the baseline config (uniform K=3, script
defaults otherwise) plus the Tier 1 fine-tune ablation below first.
Whether Tier 2 or Tier 3 get touched at all depends on what that baseline
actually looks like — not committing to either upfront. If a specific site
misbehaves (e.g.  early-peak-then-decay pattern, or
a low-prevalence site's calibration looks off), that's the trigger to go
look at the specific Tier 2/3 parameter most relevant to *that* symptom,
rather than sweeping everything preemptively.

Everything below was validated (or, in several cases, never validated at
all) against Phase 3's 5-archetype simulated data — feature counts 67–159,
prevalence spread 6–43%. Phase 4's real 6-site data looks meaningfully
different (features 158–219, prevalence spread 12.5–15.8%, narrower and
higher-dimensional across the board), so treat every "validated" claim
below as "validated on different data, likely a reasonable starting point,
not confirmed on this data" rather than settled.

### Tier 1 — fine-tuning

**Head fine-tune epochs, learning rate, pos_weight, and gradient clipping**
(`--finetune_epochs`, `--ft_lr_mult`, `--ft_pos_weight`/`--ft_no_pos_weight`,
`--ft_grad_clip` in `fedadapt_train_approach2_v2_3_ftablation_taxtest.py`) — this is
the step that happens **after** joint federated training of body+adapter
(50 rounds × 5 local epochs by default) finishes: the shared body gets
frozen, and each site's own `PersonalHead` trains locally for
`--finetune_epochs` more epochs to specialize the classifier on that site's
own label distribution before final evaluation.

v2.3 and v2.5 disagree on all four settings for this step, with no
documented rationale for the switch:

| | epochs | lr | pos_weight | grad clip |
|---|---|---|---|---|
| v2.3 | `--finetune_epochs` (default **10**) | base `--lr` | on | off |
| v2.5 | hardcoded **30**, not a flag | `lr × 0.1` | **off** | 5.0 |

None of these four were re-tuned for Phase 4's data specifically. The
epoch count in particular is the most likely one to actually need
retuning here, for a structural reason: Phase 4 sites have far more
head-input dimensionality than Phase 3's did in the low-feature cases
(`sim_UIOWA` at 158 features vs. Phase 3's site_D/site_E in the 60–80
range) — a head fine-tuning on a higher-dimensional embedding from a
smaller number of local samples may need more (or fewer, if it overfits
faster) epochs to converge than 10 was tuned for on Phase 3's data. This
was never swept even on Phase 3 data, let alone Phase 4's — `10` and `30`
are each one script's original author's choice, not the output of a sweep.
Recommend running the 4-combination ablation matrix already built (see
`fedadapt_train_approach2_v2_3_ftablation_taxtest.py`) on real Phase 4 site data
once it exists, watching per-site AUROC/F1/AUPRC especially on the
lowest-prevalence sites (`sim_UofU` 9.99%, `sim_KUMC` 12.91%) where
pos_weight and epoch count are most likely to interact.

**Commands** — each run needs its own `--output_dir` (a distinct tag per
config), since every run creates a `fedadaptproto/` subfolder under
whatever `--output_dir` you pass and later runs would otherwise overwrite
earlier ones:

```bash
# A — baseline (v2.3 original recipe: pos_weight on, lr×1.0, 10 epochs, no clip)
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
    --data_dir ./phase4_gpc_aligned_sites/alpha0.5_gamma0.75/ \
    --output_dir ./results_ft_ablation/runA/ --method fedadaptproto \
    --n_clusters 3 --alpha 0.5 --gamma 0.75 --seed 42

# B — pos_weight off only (isolates the class-imbalance-weighting change)
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
    --data_dir ./phase4_gpc_aligned_sites/alpha0.5_gamma0.75/ \
    --output_dir ./results_ft_ablation/runB/ --method fedadaptproto \
    --n_clusters 3 --alpha 0.5 --gamma 0.75 --seed 42 \
    --ft_no_pos_weight

# C — lr/epochs/clip changed only (isolates the optimization-schedule change)
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
    --data_dir ./phase4_gpc_aligned_sites/alpha0.5_gamma0.75/ \
    --output_dir ./results_ft_ablation/runC/ --method fedadaptproto \
    --n_clusters 3 --alpha 0.5 --gamma 0.75 --seed 42 \
    --ft_lr_mult 0.1 --finetune_epochs 30 --ft_grad_clip 5.0

# D — both together (≈ v2.5's full recipe, on v2.3's federation phase)
python3 fedadapt_train_approach2_v2_3_ftablation_taxtest.py \
    --data_dir ./phase4_gpc_aligned_sites/alpha0.5_gamma0.75/ \
    --output_dir ./results_ft_ablation/runD/ --method fedadaptproto \
    --n_clusters 3 --alpha 0.5 --gamma 0.75 --seed 42 \
    --ft_no_pos_weight --ft_lr_mult 0.1 --finetune_epochs 30 --ft_grad_clip 5.0
```

### Tier 2 — Tuning other parameters
- **`--lambda_adv` (GRL adversarial weight, default 0.1)** — two
  independent multi-seed tests (an aggregate 0.02/0.05/0.10 sweep, and a
  site_D-specific 0.10/0.20/0.30 sweep) both found 0.1 optimal, with the
  spread in both cases smaller than seed-to-seed noise. Reasonably safe to
  inherit, but both tests were run on Phase 3's archetype prevalences —
  Phase 4's narrower, more-uniform prevalence spread changes what the
  prevalence-adaptive scaling (`lambda_adv × min(1.0, prevalence/
  PREVALENCE_REF)`) actually produces per site, so the *effective* lambda
  each site sees is different even with the same base value.
- **`--n_clusters` / `--n_clusters_per_site` (K)** — see the dedicated
  section above. Phase 3's site_D=2/site_E=2 override was driven by a
  silhouette diagnostic specific to those two archetypes; nothing
  equivalent has been run on the 6 real Phase 4 sites yet. Uniform K=3 is
  the validated-elsewhere starting point; a genuine per-site override needs
  its own silhouette pass on real Phase 4 embeddings first.
- **`--warmup_rounds` (default 10)** — chosen in Phase 2 as "20% of a
  50-round run," not independently re-swept since. Untested whether that
  ratio still makes sense if `--rounds` changes for Phase 4.

### Tier 3 — Inherited defaults

- **`--rounds` (federation rounds, default 50)** and **`--local_epochs`
  (default 5)** — no dedicated sweep exists for either in the Phase 3
  findings. The one thing that *is* known: Phase 3's site_D peaked almost
  immediately (round 1–3) then decayed for the rest of the 50-round run,
  which is why per-site best-checkpoint tracking was added — it's a
  mitigation for a bad round count being possibly wasted on decay, not a
  fix for the round count itself. Worth watching Phase 4's per-round AUROC
  printouts for the same early-peak-then-decay pattern on any site before
  assuming 50 rounds is the right amount of federation for that site.
- **`--alpha_proto` (prototype alignment weight)** — note the two scripts
  don't even agree on the *default*: v2.3 defaults to `0.5`, v2.5 defaults
  to `1.0`, with no documented reason for the change. Neither value has a
  dedicated sweep behind it in the findings doc.
- **`--mu` (FedProx proximal weight, default 0.01)** — 
- **`--early_stop_patience` (default 2)** — this is the *local, per-round*
  early stop (bail out of a round's local epochs early if local-val AUROC
  stalls), distinct from the federation-level early stopping that was
  built, tested, found to have no benefit.
---

