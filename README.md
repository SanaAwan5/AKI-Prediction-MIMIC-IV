# Federated Transfer Learning for AKI Prediction on MIMIC-IV

Federated learning experiments for Acute Kidney Injury (AKI) prediction,
evaluated on two simulated multi-site cohorts derived from the same
MIMIC-IV patient population:

- **Phase 1 — clinical-archetype cohort**: 5 deliberately dissimilar
  sites (A–E) spanning a wide heterogeneity range, swept across a
  20-condition grid (5 α × 4 γ).
- **Phase 2 — GPC-aligned cohort**: 6 sites (KUMC, MCW, UIOWA, UPITT,
  UTSW, UofU) whose per-site acuity and feature-sparsity profiles are
  derived from real Greater Plains Collaborative site statistics,
  across 3 heterogeneity conditions.

Six FL strategies are compared on both: **FedAdaptProto** (v2.3 manual
*K*, and v2.5 automated *K*), **FedAdapt**, **FedAvg**, **FedProx**, and
**SCAFFOLD**.

> **Status.** All training is complete and confirmed on the corrected
> cohort; the headline numbers below are current. The three-term FL Gain
> Index is **withdrawn** and replaced by a validated one-term model — see
> [FL Gain Index](#fl-gain-index--superseded-by-a-one-term-model). Every
> gain in this repository is computed against one shared matched-effort
> local baseline via `recompute_gains.py`, for the reasons in
> [Measurement protocol](#measurement-protocol).

`run_complete.md` is the companion long-form log: every individual
command, including the historical and diagnostic runs summarized here.

---

## Cohort

Both cohorts are drawn from the identical underlying patient population,
with identical per-patient train/test assignment.

| | Value |
|---|---|
| Total patients | **114,720** |
| Train / test split | 91,776 (80%) / 22,944 (20%) |
| `subject_id` : `hadm_id` | 1:1 (one encounter per patient, last admission only) |
| Phase 1 sites | 5 × **17,000** patients (85,000 of 91,776 used) |
| Phase 2 sites | 6 × **14,000** patients (84,000 of 91,776 used) |
| Cross-site overlap | **Zero**, verified per condition via `check_overlap.py` |

`TARGET_N_PER_SITE` was established empirically, not calculated: Phase 1
fails at 18,000 (6,241 within-site duplicates at the last-processed
site, whose low prevalence target depletes the AKI-positive class);
Phase 2 fails at 15,000 (1,138-patient shortfall). Re-check both if the
cohort size changes.

---

## Results

Every gain below is computed against **one shared matched-effort local
baseline**: same architecture, same budget, same data, and the same
checkpoint-selection privilege the federated model receives, selected on the
training script's own validation split (`val_source=script`). Published
figures from the conventional protocol are shown beside them, because the
difference between the two columns is one of this repository's main results.

### Phase 1 — clinical-archetype, 20 conditions × 3 seeds

n = 300 per method (1500 observations). Source: `gains_matched_p1.csv`.

| Method | ΔAUROC | ΔAUPRC | Sites > 0 | Conventional protocol |
|---|---|---|---|---|
| **FedAdaptProto v2.3** | **−0.0029** | **−0.0099** | 43% | +0.0507 |
| FedAvg | −0.0056 | −0.0214 | 37% | +0.0481 |
| FedProx | −0.0056 | −0.0214 | 39% | +0.0480 |
| FedAdapt | −0.0072 | −0.0204 | 31% | +0.0464 |
| SCAFFOLD | −0.0087 | −0.0290 | 30% | +0.0450 |
| ALL | −0.0060 | −0.0204 | | |

Local baseline peaked at epoch 7.7 of 50.

### Phase 2 — GPC-aligned, 3 conditions × 3 seeds

n = 54 per method (324 observations). Source: `gains_matched_p2.csv`.

| Configuration / method | ΔAUROC | ΔAUPRC | Sites > 0 | Conventional |
|---|---|---|---|---|
| v2.3 *K*=2 / **FedAdaptProto** | **−0.0069** | **−0.0139** | 35% | +0.0667 |
| lepoch3 / FedAdapt | −0.0084 | −0.0213 | 30% | +0.0651 |
| lepoch3 / SCAFFOLD | −0.0122 | −0.0180 | 20% | +0.0614 |
| lepoch3 / FedAvg | −0.0132 | −0.0191 | 17% | +0.0603 |
| lepoch3 / FedProx | −0.0132 | −0.0204 | 19% | +0.0603 |
| v2.5 bestckpt / FedAdaptProto | −0.0366 | −0.0485 | 0% | +0.0176 |

Local baseline peaked at epoch 1.5 of 150.

### Two protocols, two findings

**Under the matched-baseline protocol, federation shows no absolute
benefit.** Scored against a local baseline given the same
checkpoint-selection privilege as the federated model, federation reduces
performance in both cohorts, on both metrics, at every condition tested.
The gap between the two protocols is ≈0.053 (Phase 1) and ≈0.073 (Phase 2)
— the size of the checkpoint-selection asymmetry itself, not a method
effect.

**The method ranking holds under both protocols.** Phase 1 reproduces the
conventional-protocol order down to the FedAvg/FedProx tie: FedAdaptProto >
FedAvg = FedProx > FedAdapt > SCAFFOLD. Phase 2 likewise. The architecture
claim stands in a narrower form under the matched-baseline protocol: the
domain-adversarial, personalization-head architecture **loses least to
heterogeneity**, not that federation improves on local-only training.

**v2.5 (auto-*K*) is clearly the weakest arm** — −0.0366 with 0% of sites
positive, a sharper statement than "indistinguishable from zero".

### Significance under the matched baseline

Paired tests, FedAdaptProto as reference, job-level (one mean per
condition-seed; sites within a job share one federated model and are not
independent). Source: `pvalues_matched_p{1,2}_{auroc,auprc}.csv`.

**Phase 1 (n = 60 pairs)** — all four comparisons, both metrics:

| vs | ΔAUROC diff | Wilcoxon p | ΔAUPRC diff | Wilcoxon p |
|---|---|---|---|---|
| FedAdapt | +0.0043 | 1.1e-09 | +0.0106 | 1.4e-09 |
| FedAvg | +0.0027 | 1.6e-06 | +0.0115 | 1.7e-10 |
| FedProx | +0.0027 | 1.6e-06 | +0.0116 | 7.3e-11 |
| SCAFFOLD | +0.0058 | 4.2e-11 | +0.0191 | 1.6e-11 |

**Phase 2 (n = 9 pairs)**:

| vs | ΔAUROC diff | Wilcoxon p | ΔAUPRC diff | Wilcoxon p |
|---|---|---|---|---|
| FedAdapt | +0.0016 | 0.496 | +0.0074 | 0.055 |
| FedAvg | +0.0064 | 0.0078 | +0.0052 | 0.250 |
| FedProx | +0.0063 | 0.0078 | +0.0066 | 0.039 |
| SCAFFOLD | +0.0053 | 0.0078 | +0.0041 | 0.203 |

**AUPRC is the stronger arm, not the weaker one.** On Phase 1 its effects
are 2–3× the AUROC ones with p-values orders of magnitude smaller. Phase 2's
AUPRC nulls are therefore **underpowered at n = 9**, not contradictory —
direction is consistent with Phase 1 on all four comparisons and both
metrics. Report Phase 2 as corroboration.

**One genuine cohort difference.** FedAdapt is significantly worse than
FedAdaptProto on Phase 1 (p = 4e-12) but indistinguishable on Phase 2
(p = 0.39). The two share the architecture and differ only in prototype
clustering, so clustering helps in the 5-site archetype cohort and is not
resolvable in the 6-site GPC cohort.

> **Report the means beside every p-value.** Differences of 0.003–0.019
> reaching p = 1e-25 means the measurement is precise, not that the effect
> is large. These differences are clinically immaterial, and a significant
> difference between two negative means says one method *loses less* — not
> that either improves on local-only training.

---

## Repository contents

**Cohort construction** (run first, against your own credentialed
MIMIC-IV access):
- `phase1_archetype_cohort.ipynb` → `aki_anchor_based_24h_lookback.csv`
  (94 columns, smaller lab panel)
- `phase2_gpc_aligned_cohort.ipynb` →
  `aki_anchor_based_24h_lookback_aligned_features.csv` (490 columns,
  includes BMI and the expanded GPC-aligned lab panel)

> **The master CSVs are not in this repository.** They are MIMIC-IV-derived
> patient data (PhysioNet data use agreement) and the Phase 2 file is
> ~144 MB. Generate them locally with the two notebooks above; every
> script expects them in the working directory under those exact names.

**Site simulation**
- `phase1_archetype_simulation.py`, `phase2_gpc_aligned_simulation.py`
- `run_disjoint_sites_data_gen.sh` — combined entry point for both
  cohorts (smoke test → full grid → overlap verification)
- `check_overlap.py`, `HOW_TO_CHECK_OVERLAP.txt` — cross-site overlap
  verification
- `record_train_test_numbers.py` — verifies split and cross-file
  population consistency

**Training**
- `phase1_archetype_train_v23_auprc.py`, `phase1_archetype_train_v25_auprc.py`
- `phase2_gpc_aligned_train_v23_auprc.py`, `phase2_gpc_aligned_train_v25_auprc.py`

> **Use the `_auprc` variants.** The four scripts without that suffix
> predate the val-selection patch (`grep -c VALSELECT` returns 0 on them,
> 11 on the `_auprc` variants) and carry no AUPRC columns. A baseline run
> against an unpatched script carves its own validation split, so the local
> baseline and the federated run select checkpoints on different rows.
> Every baseline run should print `val split from script`; `internal` means
> the wrong script.
- `fedadapt_model_approach2.py` — shared model definitions (`SiteInputAdapter`,
  `SharedBody`, `GRLGroupDiscriminator`, `PersonalHead`, `FedAdaptClient`),
  imported by all four
- `run_phase1_grid_v23.sh`, `run_phase1_grid_v25.sh`, `run_phase2_training.sh`

**Analysis — FL gain**
- `recompute_gains.py` — recompute every method's gain against ONE shared
  matched baseline, so methods scored against different baselines become
  comparable. Pure arithmetic, no retraining.
- `pool_site_conditions.py` — average replicate runs into (site, condition)
  observations; refuses to blend methods silently
- `flgain_sign_rank.py` — sign + ranking with leave-one-SITE-out and
  permutation nulls. `--predictors prevalence` runs the published model.
- `joining_site_report.py` — the deliverable for a prospective member:
  expected gain with a 90% LOO-residual interval and achieved coverage
- `site_predictors.py` — builds the pre-join predictors (positive count,
  feature overlap/coverage, distributional distance) leave-this-site-out
- `mlp_ceiling_test.py` — tests an architecture-independent local ceiling
  against a null calibrated at the observed rho to C_fed
- `gradient_predictor.py` — gradient alignment from `gradient_conflict.csv`
- `design_expanded_cohort.py` — Latin-hypercube cohort design preserving
  the 5 published archetypes as anchors; rejects correlated designs
- `apply_site_config_patch.py` — lets the simulator read a cohort file
- `fit_scarcity.py` — fits the `--train_frac` learning-curve sweep
  (Section 14.6); refuses a condition whose points don't share one
  protocol
- `matched_method_pvalues.py` — paired significance (t-test + Wilcoxon,
  job- and site-level) between methods against the shared matched
  baseline; `--config` guard refuses to pool a method across distinct
  algorithm configurations
- `gains_matched_p1.csv`, `gains_matched_p2.csv` — `recompute_gains.py`
  output; matched-baseline gains for every method/condition/seed/site,
  both cohorts
- `pvalues_matched_p1_auroc.csv`, `pvalues_matched_p1_auprc.csv`,
  `pvalues_matched_p2_auroc.csv`, `pvalues_matched_p2_auprc.csv` —
  `matched_method_pvalues.py` output, both metrics, both cohorts

**Analysis — protocol**
- `verify_no_test_selection.py` — AST taint check; fails if it returns
- `compute_matched_baseline.py` — the one shared local baseline
  (`--arch`, `--train_frac`, `--val_every`)
- `fix_results_naming.py` — writes a `COHORT.txt` manifest into every
  results directory (cohort, sites, methods, conditions, seeds); renames
  only the 2 directories whose name states the wrong cohort, and only if
  nothing still references the old name
- `apply_valselect_patch.py`, `apply_epochckpt_patch.py` — one-time
  patch-appliers (moves checkpoint selection off the test split; adds
  `--ckpt_every_epoch`/`--eval_every`). The `_auprc` training scripts
  already carry both patches (`grep -c VALSELECT` returns 11), so these
  two are provenance for how that happened, not something any documented
  command re-invokes — optional to keep in the repo, not required

**Analysis — superseded (do not push as part of the live pipeline)**
- `compute_flgi_correlation.py` — the withdrawn three-term index. Kept
  for traceability; see the FL Gain Index section for why two of its
  three terms are the same variable.
- `example_gpc_aligned_sites.csv`, `example_archetype_sites.csv` — worked
  inputs that reproduce the withdrawn r=0.457 and r=0.637
- `joining_site_recommendation.py` — an earlier joining-site tool built
  for the withdrawn FL-Gain-Index/scarcity-sweep approach. Not called by
  anything in this document; `joining_site_report.py` above is its
  replacement and is the one the Section 14 commands actually use.

**Documentation**
- `run_complete.md` — the full run log this README summarizes
- `BUGFIX_LOG.md` — history of real code defects found and fixed
  (KDIGO/CKD-exclusion, disjoint sampling, v2.5 checkpoint/cache bugs, the
  interrupted-grid incident); kept separate from the run log and from the
  matched-baseline protocol, which is a methodology choice, not a bug

---

## Reproducing the results

Run each command standalone — one at a time, or as a sequential
non-backgrounded block. This project has repeatedly hit silent
`--data_dir`/`--alpha`/`--gamma` misreads under concurrent execution.

```bash
# 1. Generate both cohorts' site data (includes overlap verification)
chmod +x run_disjoint_sites_data_gen.sh
./run_disjoint_sites_data_gen.sh

# 2. Phase 1 -- v2.3 + 4 baselines, 20 conditions x 3 seeds (300 jobs)
chmod +x run_phase1_grid_v23.sh
./run_phase1_grid_v23.sh

# 3. Phase 1 -- v2.5 auto-K, 20 conditions x 3 seeds (60 jobs)
chmod +x run_phase1_grid_v25.sh
./run_phase1_grid_v25.sh

# 4. Phase 2 -- all 6 methods, 3 conditions x 3 seeds (54 jobs)
chmod +x run_phase2_training.sh
./run_phase2_training.sh

# 5. FL Gain Index correlation (no training)
python3 compute_flgi_correlation.py example_gpc_aligned_sites.csv
python3 compute_flgi_correlation.py example_archetype_sites.csv
```

All three grid scripts have a **resume-skip check** — a job whose output
already exists is skipped, so an interrupted grid can be safely
restarted rather than re-run from the top. They also refuse to run
unless the training script on disk carries the expected fix signatures.

Requires `torch`, `pandas`, `numpy`, `scikit-learn`.

`local_epochs` differs by cohort: Phase 1 uses `1` (the script default),
Phase 2 uses `3`. Don't cross them — see `run_complete.md` §2 for the
sweep that settled this.

---

## FL Gain Index — superseded by a one-term model

The FL Gain Index (FLGI) predicted, per site, how much a site stands to
gain from joining federation, from three weighted terms: positive-case
scarcity, class imbalance, and feature sparsity. That formulation is
**withdrawn**. On the 25-site cohort its terms correlate with outcome
prevalence as follows:

| FLGI term | corr with prevalence | corr with dAUROC |
|---|---|---|
| `positive_case_scarcity` | **-1.000** | -0.609 |
| `class_imbalance` | **-1.000** | -0.609 |
| `feature_sparsity` | -0.014 | -0.307 |

`class_imbalance` is `1 - prevalence` by definition, and
`positive_case_scarcity` derives from `n_positive`, which equals
`33,000 x prevalence` because `n_samples` is constant across sites. Two
of the three terms are therefore the same variable entered twice under
different names, fitted with independent weights — which is why the
class-imbalance coefficient swung from `-1.508` to `+0.063` when the
local baseline was corrected. Two collinear terms with opposite-signed
weights trade off freely without changing the prediction.

`feature_sparsity` is genuinely independent of prevalence, and it does
not predict: leave-one-site-out balanced accuracy `0.500` on both
metrics and both federated methods.

### The replacement

Prevalence alone, fitted per method and per metric on 25 sites
(alpha=0.1, gamma=0.0, 3 seeds, pooled to one observation per site):

| Method | Metric | Fit | Break-even | R2 |
|---|---|---|---|---|
| FedAvg | dAUROC | `-0.0144 + 0.0432*prev` | 0.333 | 0.370 |
| FedAvg | dAUPRC | `-0.0580 + 0.1719*prev` | 0.338 | 0.707 |
| FedAdaptProto | dAUROC | `-0.0154 + 0.0460*prev` | 0.335 | 0.392 |
| FedAdaptProto | dAUPRC | `-0.0440 + 0.1330*prev` | 0.331 | 0.606 |

Four independent fits agree on the break-even prevalence to within
0.007. The AUPRC agreement is not an artifact of the precision-recall
floor: normalising by `(1 - prev)` moves the correlation only from
+0.841 to +0.835 (FedAvg) and +0.778 to +0.768 (FedAdaptProto).

### Validation

Every figure comes from a model fitted **without the site being scored**
(leave-one-site-out), since a joining site is by definition one the
model was not fitted on.

| Test | Observed | Permutation null (95th) | |
|---|---|---|---|
| Sign, dAUROC | 0.691 | 0.500 | clears |
| Sign, dAUPRC | 0.807 | 0.500 | clears |
| Ranking, Kendall tau | +0.353 | +0.167 | clears |

Balanced accuracy is reported rather than raw accuracy: a constant
predictor scores 0.500 on it regardless of class balance. The
permutation null shuffles the **target**, not site labels — with one
observation per site, shuffling site labels is a no-op.

Ranking is only testable at this cohort size. At 5 sites there are 10
pairs, tau moves in steps of 0.20, and the null reaches +0.40.

### Report intervals, not points

Leave-one-site-out residual SD is 0.0071 against a between-site spread
of 0.0082 — the noise is the same order as the quantity. Output takes
the form:

```
A site with prevalence 0.22:
  expected gain -0.005, 90% interval [-0.018, +0.005]
```

Bands come from LOO residuals rather than the fit's standard errors:
the latter assume the model is correct and describe only coefficient
uncertainty, whereas a joining site is exposed to the model being wrong
about a site it has never seen. Achieved coverage is 84% against a
nominal 90%, reported beside the interval rather than assumed. For
mid-range sites the interval spans zero and the honest output is that
the direction is unresolved.

### Controlled negative

Regenerating the identical 25 sites at `alpha=10.0` compresses the
prevalence range from 0.052-0.414 to 0.162-0.198:

| | alpha=0.1 | alpha=10.0 |
|---|---|---|
| prevalence spread | 0.1160 | 0.0116 |
| sign, balanced accuracy | 0.691 | 0.500 |
| ranking, Kendall tau | +0.353 | +0.113 |
| null 95th (tau) | +0.167 | +0.180 |
| verdict | clears both | clears neither |

Same generator, sites, pipeline and predictor. Both tests collapse to
chance when the prevalence spread is removed — which is what should
happen, and is not what a pipeline manufacturing apparent skill would
produce.

---

## Predictors tested and rejected

Reported so the one-term model is not mistaken for a lack of effort.
All scored identically: LOO balanced accuracy on the sign, against a
null that shuffles the target. Three distinct failure modes.

| Predictor | Range | Bal. acc. | Null | Outcome |
|---|---|---|---|---|
| Prevalence | 0.052-0.414 | 0.691 | 0.500 | **retained** |
| Positive count | 1,723-13,663 | — | — | r = 1.000 with prevalence |
| Feature overlap | 1.000 (constant) | — | — | no variance |
| Gradient alignment | +0.183-+0.240 | 0.504 | 0.500 | r = -0.897 with prevalence |
| Update-norm ratio | — | 0.504 | 0.500 | collinear, post-hoc only |
| Local ceiling C | 0.771-0.810 | 0.382 | 0.533 | inside the target |
| MLP ceiling | 0.766-0.818 | 0.596 | 0.566 | rho = 0.923 with C |
| Federation coverage | 0.236-1.000 | 0.500 | 0.500 | no signal |
| Distributional distance | 0.499-0.662 | 0.500 | 0.500 | no signal |

**Gradient alignment** is the mechanism by which federation helps or
hurts, measured directly (`--track_gradient_conflict`) rather than
proxied, and it relates to the gain in the expected direction: the less
a site's update resembles the aggregate, the more it gains. But it is
nearly a linear function of prevalence (`cos = 0.2354 - 0.1421*prev`,
VIF 5.13), and regressing the gain on prevalence leaves a residual that
correlates **-0.036** with alignment. It also requires the site to have
already trained inside the federation, so it cannot forecast.

**The local ceiling** sits inside the target by the `delta = F - C`
identity. `Var(C)` alone is 169% of `Var(delta)` and `r(C,F) = +0.908`:
the gain is a small difference between two large coupled quantities.
Measuring it with an independent MLP removes C from the target
algebraically but not statistically — at `rho = 0.923`, a synthetic
ceiling carrying *zero* information still shows `r ~ +0.54` with the
gain. Scored against a null built at the observed rho, the MLP ceiling
never improves on prevalence (0.691 -> 0.599 on dAUROC; unchanged at
0.807 on dAUPRC with R2 falling 0.658 -> 0.591), and its residual after
prevalence is **-0.040** — the same as gradient alignment.

Two quantities built on unrelated principles both landing at zero once
prevalence is accounted for is the evidence that prevalence is what each
was indirectly measuring.

**One measurement error worth recording**, because it would recur in any
replication: computing distributional distance over the *union* of all
features across sites gives a spuriously wide range (0.784-1.000). That
is median-imputation showing through in columns a site does not measure,
not case-mix difference. Restricting the classifier to features the site
and federation both measure gives 0.499-0.662.

---

## Measurement protocol

Two protocol findings determine how every gain in this repository should
be read; see `sections/protocol_sensitivity.tex` for the full argument.

**Checkpoint selection read the test split.** All four training scripts
updated `ckpt_best_auroc` from `evaluate_site()`, which scores `X_test`,
then restored the client to that round. Fixed by
`apply_valselect_patch.py`; `verify_no_test_selection.py` is an
AST-based taint check that fails the build if selection ever reads the
test split again. Worth 0.0032-0.0040 AUROC — smaller than feared, but
the fix is a precondition for every number above.

**Local training peaks within a few epochs — but is not data-saturated.**
Under a matched budget the local model peaks after a mean of 1.5 epochs
of 150 (GPC) and 7.7 of 50 (archetype); everything after is overfitting.
The choice of local baseline protocol (0.028 AUROC) exceeds the effect
being measured (0.013-0.026), which is why this repository fixes one
shared baseline and recomputes every method's gain against it
(`recompute_gains.py`) rather than trusting each run's own.

That early peak is an *optimisation* limit, not an information one. A
`--train_frac` sweep holding validation and test fixed shows local AUROC
rising log-linearly in data, with no plateau in any condition tested:

| Cohort | Condition | Sites | Budget | AUROC/doubling | AUPRC/doubling | R2 |
|---|---|---|---|---|---|---|
| Archetype | a=0.1 g=0 | 5 | 50 | +0.00959 | +0.01937 | 0.994 |
| Archetype | a=0.5 g=1 | 5 | 50 | +0.01141 | +0.02479 | 0.999 |
| Archetype | a=10 g=0 | 5 | 50 | +0.00985 | +0.02369 | 0.992 |
| Archetype (x25) | a=0.1 g=0 | 25 | 50 | +0.00912 | +0.02001 | 0.997 |
| GPC-aligned | a=0 g=0 | 6 | 150 | +0.01199 | +0.01726 | 0.986 |
| GPC-aligned | a=0.5 g=0.75 | 6 | 150 | +0.01143 | +0.01790 | 0.944 |
| GPC-aligned | a=1 g=1 | 6 | 150 | +0.01093 | +0.01677 | 0.992 |

Five points per row (2/5/10/25/100% of each site's training split). In
five of seven the last increment is the largest. Slopes span +0.0091 to
+0.0120 — a 1.3x range — while alpha varies 100x, site count 5x and the
matched epoch budget 3x. Since alpha abolishes the prevalence effect
entirely (the controlled negative above) while leaving the data slope
intact, the two mechanisms are separable.

The GPC-aligned cohort is the sharpest case: its local models peak at 3-13
epochs of 150, the earliest anywhere in the project, and its data slope is
the steepest of the seven. A model that stops at epoch 3 of 150 and still
improves monotonically with more data is optimisation-limited.

One doubling of a site's own data is worth **+0.0091** AUROC; federation
across 25 sites delivers **-0.0049**. Federated learning here fails to
recover what a single doubling of ordinary data would supply while
nominally providing 25x as much — a shortfall of **0.0141** AUROC that is a
lower bound on the cost of cross-site heterogeneity. We report the
single-doubling reference rather than extrapolating the fit to 25x pooling
(log2(25) = 4.64 doublings, +0.0424), which lies beyond the fitted range
and would assume pooled heterogeneous data behaves like more of the same.

Per-site slopes range 0.0050-0.0130 per doubling and correlate only
-0.167 with prevalence, so data-hunger is a genuine second axis of
site-level variation that the one-term model does not capture. It is not
usable as a joining-site predictor for the same reason gradient
alignment is not: measuring it takes four extra training runs per site.

---

## Methodology notes

**Leakage — resolved, both cohorts.** `feature_cutoff` is class-anchored:
AKI patients are cut off 24h *before* their first KDIGO-positive SCr (a
lead-time buffer with no overlap with the label-defining event); non-AKI
patients at `last_scr_time − 24h`. A subtler anchor-selection asymmetry
was also found and removed — `hours_since`/`hours_to_anchor` encoded
class-dependent monitoring-density artifacts rather than real signal
(0.62 AUROC standalone) and are dropped from the feature set.

**Baseline SCr / CKD exclusion — resolved, both cohorts.** Both notebooks
implement the standard 3-tier KDIGO baseline-SCr hierarchy: 7-day-prior
most recent → 7–365-day-prior mean → CKD history with no SCr in the past
year drops the encounter, non-CKD gets MDRD-estimated. Previously MDRD
was applied regardless of CKD status. This is what took the cohort from
163,038 to 114,720 patients.

**Cross-site overlap — resolved.** Sites previously resampled
independently from a shared pool with no mutual exclusion, producing
~35% pairwise overlap. The simulation scripts now track used
`subject_id`s across the site loop. Disjointness holds within a
condition, not across independent condition runs — `check_overlap.py`
groups by (α, γ) automatically.

**v2.5 checkpoint and cache fixes.** v2.5 originally reported strongly
negative results on both cohorts (−0.0267 GPC-aligned, −0.0945
archetype), traced to two causes: no best-checkpoint restoration in its
Phase 1 warmup stage (so Phase 2 inherited an already-overfit state),
and an uncontrolled `local_epochs` mismatch (v2.5 defaulted to 5, v2.3
used 1). Two further bugs surfaced later, found only by inspecting real
output rather than by code review: site discovery had no α/γ filtering,
so each job silently trained a ~100-site mega-federation instead of 5;
and the local-baseline cache lived at a single condition-independent
path, so every job after the first loaded a stale, cross-condition
cache. All four are fixed in the current scripts. Full detail in
`run_complete.md` §8.

**GPC-vitals exclusion (Phase 2 only).** `heart_rate`, `resp_rate`,
`temperature`, `spo2`, `oxygen_saturation`, `gcs_total` (all four
stat-variants each, 24 columns) are excluded from `feat_cols` — they
have no counterpart in real GPC production tables. `sbp`, `dbp`, `bmi`
keep all four variants; `age_at_admission` is kept as a single value.
This is baked into `SiteData.__init__` (v2.3) / `load_site()` (v2.5).

---

## Superseded files

These are earlier versions kept only for traceability. **None are part
of the pipeline and none should be run.** Several were byte-identical
duplicates of their replacements and have been untracked (they remain in
git history). One is worse than a duplicate:
`fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py` is a
*half-fixed* copy — it has the site-discovery filter fix but is missing
the condition-specific baseline-cache fix, so running it silently reuses
cross-condition-polluted caches.

| Superseded file | Replaced by |
|---|---|
| `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed.py` | — (dead end) |
| `fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_leakage_fixed_improvement.py` | `phase2_gpc_aligned_train_v23.py` |
| `fedadapt_train_approach2_v2_5_grouptest_v2_leakage_fixed.py` | — (dead end) |
| `fedadapt_train_approach2_v2_5_phase1_archetype_bestckpt_fix.py` (**half-fixed, do not run**) | `phase1_archetype_train_v25.py` |
| `fedadapt_train_approach2_v2_5_phase2_gpc_aligned_bestckpt_fix.py` | `phase2_gpc_aligned_train_v25.py` |
| `mimic_ftl_simulation_phase1_archetype_post_leakage.py` | `phase1_archetype_simulation.py` |
| `mimic_ftl_simulation_phase4_gpc_aligned_post_leakage.py` | `phase2_gpc_aligned_simulation.py` |
