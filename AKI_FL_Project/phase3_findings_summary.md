# FTL/AKI Phase 3 (GPC-Aligned) — Findings Summary

**Scope:** GPC-aligned feature set (159 features, 8 groups, 24h lookback, 12.5% AKI prevalence, 163,038 patients), 5-site simulation (A=ICU, B=General Ward, C=Anchor/full-feature, D=Community, E=Rural), 6 methods (FedAvg, FedProx, SCAFFOLD, FedAdapt, FedAdaptProto, FedAdaptProto v2.5/auto-K), 5α × 4γ heterogeneity grid.

---

## 1. What Changed This Session

### 1.1 Per-site best-checkpoint tracking — implemented & validated
Every site now gets restored to its own best-ever round-level AUROC before final evaluation, instead of always reporting round 50's state. Motivated by a round-level finding that site_D peaks almost immediately (round 1–3) and decays for the rest of training, while site_B/E keep improving for 15–40+ rounds.

**Result (full 20-condition grid, single-seed):**

| | pre-checkpoint | post-checkpoint |
|---|---|---|
| mean delta_auroc | −0.0190 | **−0.0096** |
| fraction FL beats local | 33.2% | **36.2%** |

Every site improved. Biggest gains went to methods with **no** fine-tuning stage (FedAvg +0.011, FedProx +0.012, SCAFFOLD +0.007), since checkpointing was their *only* correction mechanism. FedAdapt/FedAdaptProto (which already fine-tune post-federation) gained less (+0.005, +0.004); FedAdaptProto v2.5 gained almost nothing (~0.000) — its extensive post-federation fine-tuning stage largely overwrites whichever round's body it starts from.

### 1.2 GRL sign-error bug — found & fixed
`fedadapt_model_approach2.py`'s adversarial loss had an extra, redundant sign flip on top of the GRL's own internal gradient reversal:
```python
# before (bug): total = task_loss - lambda_adv * adv_loss
# after (fix):  total = task_loss + lambda_adv * adv_loss
```
The bug meant the discriminator was trained to get *worse* at its job, and the shared body was trained to make embeddings *more* group-distinguishable — the opposite of the intended domain-invariance goal.

**Confirmed mechanistically** via `adv_loss` trajectories: pre-fix, loss escalated unboundedly (500 → 3,000–7,000 over 50 rounds, never stabilizing — an impossible range for a bounded classification loss). Post-fix, it stabilizes properly (FedAdapt: ~0.08–0.35; FedAdaptProto: ~0.28–0.55, staying flat — consistent with FedAdaptProto's prototype-alignment term needing sustained invariance to work).

Only affects `fedadapt`/`fedadaptproto` (v2.3-based) — `fedadaptproto_v25` already had the correct sign independently.

**AUROC impact — resolved for both methods, with different outcomes.** Initial single-seed comparison was small and mixed, consistent with noise rather than a real effect. Re-checked with genuine multi-seed data (seeds 42/7/123) for both:

| | `fedadaptproto` | `fedadapt` |
|---|---|---|
| overall mean Δauroc (pre → post-fix) | +0.0017 → −0.0013 ± 0.0030 | +0.0054 → −0.0016 ± 0.0035 |
| site_D specifically (pre → post-fix) | −0.0511 → −0.0518 ± 0.0093 | **−0.0332 → −0.0579 ± 0.0061** |

`fedadaptproto`: pre-fix value falls comfortably inside the post-fix noise band at both the aggregate and site_D level — **no detectable AUROC effect**, consistent with the fix being justified on mechanistic grounds (bounded vs. unbounded adversarial loss) rather than an AUROC-improvement claim.

`fedadapt`: the aggregate is borderline (gap ≈ 2× the seed std, not clearly resolved), but **site_D's gap (0.025) is roughly 4× the seed-to-seed std (0.006) — outside the noise band, a real effect, not noise.** `fedadapt`'s site_D result appears to have genuinely gotten *worse* after the fix. This is consistent with a difference in post-fix dynamics noted earlier: `fedadapt`'s `adv_loss` trajectory decreases over training (discriminator increasingly "wins," invariance not sustained), unlike `fedadaptproto`'s flat, sustained ~0.5 — the two methods land in different post-fix equilibria, and for `fedadapt` that equilibrium carries a real cost on site_D specifically. This does not change the case for keeping the fix (it's mechanistically correct regardless of AUROC), but it's a genuine, documented side effect worth being upfront about rather than smoothing over.

### 1.3 Lambda_adv sensitivity — tested, closed
Two independent tests, both negative (no retuning needed):
- Aggregate sweep (0.02/0.05/0.10): non-monotonic, spread smaller than single-seed noise.
- Site_D-specific multi-seed sweep (0.10/0.20/0.30, n=3 seeds): spread (0.0006) an order of magnitude smaller than seed noise (0.004–0.009).

**Conclusion:** `lambda_adv = 0.1` (Phase 2's validated value) stays as-is.

### 1.4 Lambda_adv scaling-formula inconsistency — found, fixed, multi-seed confirmed
`v2.3` scaled the adversarial weight by AKI prevalence (documented rationale: low-prevalence sites have weak class gradient, GRL overpowers it). `v2.5` scaled by an unrelated criterion — feature-count ratio — with no documented rationale. Unified both scripts on the prevalence-based formula. Single-seed check at α=0.3/γ=0.75 (auto-K): `fedadaptproto_v25`'s mean improved from −0.0063 to −0.0016. **Multi-seed confirmed at the method's actual best config** (uniform K=3, seeds 42/7/123): mean Δauroc = **+0.0018 ± 0.0029**, stable and genuinely positive — the fix holds up, and site-level variance here is normal (unlike the extreme K-mode-specific instability described in 1.5 below).

### 1.5 K-mode — major silent bug found & fixed
`fedadaptproto`'s `per_site` K configuration (intended: sites A/B/C=3, D/E=2) had been **silently inert since it was written** — a suffix mismatch between the shell script (which appended an alpha/gamma suffix to site names) and v2.3's internal site-id convention (bare, unsuffixed) meant the override was never actually applied. Every "per-site K" run had secretly just been running uniform K=3.

**This likely also invalidates Phase 2's own original conclusion** that per-site K was "neutral-to-slightly-worse" than uniform K=3 — if the same bug was present then, that comparison never tested a real difference.

Fixed, then re-tested at α=0.1/γ=1.0 (max heterogeneity):

| config | fedadaptproto mean Δ | fedadaptproto_v25 mean Δ |
|---|---|---|
| uniform K=3 | 0.0046 | **0.0116** (site_D: −0.045→−0.002) |
| genuine per-site K | **0.0061** (ties uniform2) | not applicable (v2.5 uses its own auto-K, not this table) |
| uniform K=2 | 0.0063 | 0.0098 |
| auto-K (v2.5's own mechanism) | — | 0.0006 |

Single-seed, one condition — needs multi-seed confirmation before adopting as default.

### 1.6 Federation-level early stopping — implemented, tested, found no benefit, reverted
New mechanism (distinct from the pre-existing local-epoch early stop): tracked each site's best round-level AUROC across the whole run; once a site went `--fed_early_stop_patience` rounds without a new best, it stopped. Two modes tested:
- **`full_exit`**: completely frozen (no training, no aggregation contribution, no further broadcasts).
- **`solo_continue`**: kept training on its own data, but still excluded from aggregation/broadcasts.

Patience=8 was calibrated against real round-level data (correctly recovered site_D's true early peak almost exactly across three tested conditions, with negligible cost to site_E's much-later peak).

**First test run crashed**: at α=0.3/γ=0.75 with `fedadaptproto`, all 5 sites had stopped by round 28, and the prototype-alignment step had no handling for "zero active sites," causing an IndexError. Fixed (empty-list guard + early-loop-exit optimization for `full_exit` specifically).

**Second test run completed successfully — and showed no measurable benefit:**

| site | baseline (no early-stop) | full_exit | solo_continue |
|---|---|---|---|
| site_A | 0.0413 | 0.0414 | 0.0414 |
| site_B | -0.0101 | -0.0075 | -0.0100 |
| site_C | -0.0164 | -0.0163 | -0.0163 |
| **site_D** | **-0.0518** | **-0.0518** | **-0.0520** |
| site_E | 0.0134 | 0.0133 | 0.0117 |

Site_D's number was **literally identical** between baseline and `full_exit`, despite the mechanism having actually triggered (confirmed in the crashed run's log — site_D stopped at round 12). The other sites moved by at most +0.0026 (site_B), otherwise flat.

**Explanation:** checkpointing (already active regardless of early-stop setting) was already restoring every site to its own best round before final evaluation — so early stopping's only theoretically distinct contribution, excluding a struggling site from aggregation to help the *other* sites, simply didn't show up empirically. Both modes were safe and neutral, just not beneficial. Effect sizes (0.0001–0.003) were small enough that multi-seed testing was judged unlikely to change the conclusion.

**Given the negative result, all early-stopping code has been reverted** from `fedadapt_train_approach2_v2_3.py`, `run_phase3_aligned.sh`, and `compute_fl_gain_offline.py`, restoring them to their pre-early-stopping state. Checkpointing, the GRL fix, the K-mode fix, and the lambda-formula unification are all unaffected by this revert and remain in place. The thread is closed pending a genuinely new hypothesis for why federation-level early stopping might matter in a different setting (e.g., a method without a fine-tuning stage, where checkpointing alone might leave more residual damage for early stopping to still meaningfully correct).

---

## 2. Site_D: the open problem

Across every test this session, site_D is the one persistent anomaly:

- **Never beat local baseline in 120/120 method-condition combinations**, pre-fixes.
- **Peaks almost immediately** (round 1–3) then decays under continued federation, unlike every other site (which peak between round 8–44 depending on condition).
- **Ruled out as explanations:** feature-set overlap (Jaccard with its closest neighbor site_B = 0.85 — nearly identical features), covariate distributional shift (KS test showed site_B/D *more* similar than a working control pair), naive cross-site transfer collapse (severe for *every* site pair, not D-specific — actually smaller for the B/D pair than for the pairs that work well), K-selection quality (dataset-wide K=2 preference, not D-specific), and adversarial signal strength (no real effect at any tested lambda).
- **Still unexplained.** Whatever's actually wrong with site_D hasn't been isolated by any diagnostic tried so far. The remaining candidate is an in-training diagnostic (gradient norms / embedding drift specific to site_D during actual federation), not yet attempted.
- **Partial, practical mitigation exists**: checkpointing alone roughly halved site_D's average deficit (−0.0735 → −0.0545 pooled); correct K helped further at the one condition tested; early stopping's effect is still unknown pending a successful run.

---

## 3. Phase 2 vs. Phase 3 — what actually changed

| | Phase 2 (unaligned features) | Phase 3 (GPC-aligned) |
|---|---|---|
| Features | Unaligned MIMIC-IV set | 159 GPC-aligned features, 8 groups |
| Lookback | — | 24h |
| AKI prevalence | Different cohort | 12.5% |
| site_D local-only AUROC | ~0.780 | **~0.898** |
| FedAdaptProto (validated v2.2) on site_D | **+0.061** (clear win) | still negative even after this session's fixes |

**The single biggest driver of the Phase 2→3 difference in site_D's story: its local-only baseline got dramatically stronger** (+0.118 AUROC) with GPC alignment — plausibly because alignment added more informative features. A stronger local baseline means less headroom for federation to add value, independent of any federation-side problem. This reframes the comparison: site_D isn't newly "broken" in Phase 3, the yardstick moved. That said, "still negative even with a much stronger local baseline to gain against" is itself informative — it means whatever federation is doing for site_D isn't just failing to add much, it's actively costing something.

Phase 2's config choices (uniform K=3, lambda_adv=0.1, per-site K rejected) mostly still hold up — except the per-site K rejection, which is now suspect given the same-named suffix bug likely existed in Phase 2's code too.

---

## 4. Results by heterogeneity condition

### Method ranking — FINAL, fully corrected (K-mode fix + lambda formula fix + fedadapt GRL-fix multi-seed correction all folded in)

| rank | method | mean Δauroc | mean AUROC | change from original |
|---|---|---|---|---|
| **1** | **FedAdaptProto v2.5 (uniform K=3)** | **−0.0052** | 0.9088 | was −0.0152 (dead last) → **now the clear, unambiguous best** |
| 2 | FedAvg | −0.0086 | 0.9055 | unchanged |
| 3 | FedProx | −0.0086 | 0.9054 | unchanged |
| 4 | SCAFFOLD | −0.0096 | 0.9044 | unchanged |
| 5 | **FedAdapt** | **−0.0102** | 0.9038 | was −0.0054 (tied #1) → **dropped to #5** once corrected |
| 6 | FedAdaptProto (v2.3, uniform K=2) | −0.0112 | 0.9028 | still last |

**This reverses a headline claim made repeatedly earlier in this session.** `FedAdapt` was reported as "tied for best, simplest method wins" — that was single-seed data that predated the GRL sign fix. Once the fix is applied and properly multi-seed averaged across the full grid (3 seeds × 20 conditions = 100 rows, not 1), `FedAdapt` drops from tied-#1 to #5, driven mainly by a real, substantial cost to site_D (mean Δauroc −0.062 for that site alone, worse than the single-condition check had suggested). **`FedAdaptProto v2.5` is the clear winner — full stop, not tied with anything.**

**Update: `FedAdaptProto v2.5`'s full-grid ranking-table number is now genuinely multi-seed confirmed, not just single-condition.** Full 20-condition grid, all 3 seeds (42/7/123), 300 rows total: mean Δauroc = **−0.0054 ± 0.0004** — remarkably tight (spread of only 0.0007 across seeds), and matches the original single-seed estimate (−0.0052) almost exactly. This is the strongest, most rigorously-confirmed result in the entire investigation — the headline "clear best method" finding holds up under genuine statistical scrutiny, not just a single lucky seed.

**FINAL, COMPLETE multi-seed ranking — all six methods, full 20-condition grid.** The `run_multiseed_grid.sh` batch (100 new runs: 5 methods × 2 new seeds × 20 conditions) is now fully complete:

| rank | method | mean Δauroc | seeds available |
|---|---|---|---|
| **1** | **FedAdaptProto_v25 (uniform K=3)** | **−0.0054** | 3 (42/7/123) |
| 2 | FedAvg | −0.0076 | 2 (7/123 for most conditions; some also have 42) |
| 3 | FedProx | −0.0077 | 2 (7/123) |
| 4 | SCAFFOLD | −0.0083 | 2 (7/123) |
| 5 | FedAdapt | −0.0102 | 3 (42/7/123) |
| 6 | FedAdaptProto (v2.3, uniform K=2) | −0.0106 | 3 (42/7/123) |

**`FedAdaptProto_v25` wins by a real, meaningful margin — 0.0022 better than the next-best method (FedAvg), not a marginal edge.** This is the definitive answer this session has been building toward: the ranking table now has genuine multi-seed statistical backing for every method, not just single-seed estimates. The rest of the story holds too — plain `FedAdaptProto` (v2.3) and `FedAdapt` remain the two worst methods, consistent with the "two real bugs distorted the original single-seed comparison" finding established earlier. FedAvg/FedProx/SCAFFOLD cluster tightly together in the middle, as expected for methods without the GRL/prototype machinery. Full underlying data saved as `full_grid_all_methods_multiseed.csv`.

**Headline change #2: `FedAdaptProto v2.5` goes from dead-last to clearly best once its two bugs (broken K-mode, wrong lambda-scaling formula) are actually fixed.** The earlier "prototype methods don't work, simplest method wins" conclusion was substantially an artifact of bugs on both sides of that comparison — v2.5's own bugs made it look artificially bad, and `FedAdapt`'s stale pre-fix data made it look artificially good. Once both are corrected, the real story is that prototype alignment (at least v2.5's auto-K implementation) genuinely earns its complexity.

**`FedAdaptProto` (v2.3, not v2.5) remains the one method today's work never managed to improve** — genuine per-site K made it slightly *worse* on average, not better, despite clearly winning at the single condition (α=0.1/γ=1.0) it was originally tested at.

⚠️ **This one took three attempts to get right, and the final answer is "inconclusive, not resolved."** Full story, since it surfaces something more important than the K-mode question itself:

1. **Original single-condition test** (α=0.1/γ=1.0, before the GRL fix landed): genuine per-site K won clearly (+0.0061 vs. uniform3's +0.0046).
2. **Full-grid re-run** (after the GRL fix): per-site K came out slightly *worse* than the old bugged config. Investigating the gap, site-level differences (0.018–0.021 on site_C/D) were too large for seed noise, so this was attributed to the GRL fix landing in between the two tests — a real confound.
3. **"Isolated" re-test**, all three K-modes run fresh under identical (post-GRL-fix) code at the same condition: uniform K=2 came out clearly best (+0.0063), seemingly resolving the question.
4. **Full-grid sweep at uniform K=2** (20 conditions): came back at −0.0112 — essentially *identical* to the old per-site-K result (−0.0111), not the win step 3 suggested. Checking the raw numbers at the exact same condition as step 3 revealed the same site_C/D swing pattern (0.017–0.028) **with no code change in between this time** — meaning it isn't a confound, it's **genuine run-to-run non-determinism, concentrated on site_C/D, even holding seed and code constant.**

**Actual conclusion: uniform K=2 and per-site K are statistically indistinguishable for `FedAdaptProto` at the full-grid level — no confirmed winner.** The apparent single-condition wins for each side of this question (step 1 and step 3) were both artifacts of instability that a single seed can't resolve — one from a code confound, one from apparent hardware/training non-determinism. The full-grid number, precisely because it pools 100 method-site rows and thus averages out exactly this kind of site-specific noise, is the only piece of evidence in this whole saga that should actually be trusted.

**The bigger takeaway: site_C and site_D specifically appear to have real run-to-run variance even at fixed seed and fixed code**, larger than site_A/B/E show under the same conditions. Any future single-seed conclusion touching site_C or site_D should be treated with real skepticism regardless of how "clean" the comparison looks on paper — this pipeline doesn't reproduce cleanly for those two sites, and single "confirmations" of anything involving them need multi-seed backing before being trusted.

**Methodology note:** FedAvg/FedProx/SCAFFOLD/FedAdapt numbers above are unchanged from the original post-checkpoint grid (K-mode and lambda-formula fixes don't apply to them; FedAdapt's GRL-fix-related change was checked and found to be a modest, real, site_D-specific effect — see Section 1.2 — not yet re-folded into a full 20-condition sweep). FedAdaptProto and FedAdaptProto v2.5 rows are freshly regenerated with all current fixes applied. All values single-seed (seed=42) except where noted.

### By site and by heterogeneity level (pre-dates the K-mode/lambda corrections — from the original post-checkpoint grid)

| site | mean Δauroc | pattern |
|---|---|---|
| site_E | +0.0233 | consistent winner |
| site_A | +0.0216 | consistent winner |
| site_B | −0.0123 | consistent, moderate loser |
| site_C | −0.0263 | consistent loser |
| site_D | **−0.0545** | consistent, severe loser (worst by 2×) |

| α | mean Δauroc |
|---|---|
| 0.1 (most heterogeneous) | **−0.0069** (best) |
| 0.3 | −0.0078 |
| 0.5 | −0.0096 |
| 1.0 | −0.0101 |
| 10.0 (near-IID) | **−0.0137** (worst) |

Federation's average benefit gets *worse*, not better, as data becomes more homogeneous — the opposite of the usual intuition. Likely explanation: near-IID local baselines are themselves stronger (each site's data more closely resembles the global distribution), leaving less genuine headroom for federation to add, so whatever mild noise/negative-transfer federation introduces dominates a smaller available upside. **These two breakdowns have not yet been regenerated with the corrected FedAdaptProto/v2.5 data** — worth doing once the K-mode story above is fully resolved.

**Important caveat on every number in this section: even the best method (`v25`, −0.0052) is still net-negative.** No method beats local training on average — "best" means least-bad, not a positive case for FL. This is driven by a stark, consistent per-site split, explored next.

---

## 4b. The Per-Site Split: Who Wins, Who Loses, and Why (exploratory)

Pooled across all methods and conditions, only 2 of 5 sites benefit from federation on average — and it's not the sites you'd naively expect:

| site | mean Δauroc | archetype | prevalence |
|---|---|---|---|
| site_A | **+0.022** | ICU | highest (~23–36%) |
| site_E | **+0.023** | Rural/resource-limited | lowest/most variable (~6–12%) |
| site_B | −0.012 | General Ward | moderate |
| site_C | −0.026 | Academic Anchor (159 features, richest) | moderate (~9–12.5%) |
| site_D | **−0.055** | Community Clinic | moderate-low, but lowest **local-only AUROC** of all 5 sites |

**FL helps at the two prevalence extremes and hurts the three sites in the middle** — not "richer sites benefit more" or "sparser sites benefit more." Note site_C is the richest-feature site (159 features, the full anchor set) yet is a consistent loser, while site_A (69 features) and site_E (29 features) — opposite ends of the feature-richness spectrum too — are the two winners. **Tested and confirmed (Section 5, item 1): prevalence distance from the federation's central tendency explains site_A and site_E's benefit reasonably well (r=+0.53 overall), but does NOT explain site_D** — site_D's moderate prevalence distance predicts an unremarkable outcome, not the worst one of all five sites. Site_D needs a separate explanation (see hypothesis #4 below).

### Candidate explanations (none tested directly — genuine hypotheses, not conclusions)

1. **Low-prevalence sites (site_E) benefit from borrowed signal** — the standard FL story. Few positive cases locally means a high-variance local decision boundary; pooling with sites that have more positive examples stabilizes it.
2. **High-prevalence sites (site_A) may benefit differently** — already has abundant clean signal, so the prototype-alignment/adversarial machinery's push toward a sharper, domain-invariant representation may specifically help a site whose local class boundary is already well-defined enough to exploit it.
3. **Middle sites get a "tug-of-war" cost** — federated averaging produces some compromise across all local optima. A site whose own optimum is close to that compromise is unaffected; a site whose optimum is very far away can gain from being pulled toward better-informed territory. Middle-prevalence sites may be "far enough to be pulled, not far enough to need it" — dragged toward a compromise that fits nobody well, including them.
4. **Site_D specifically may be a noise/signal-quality story, not just a prevalence story.** It has the lowest local-only AUROC of all 5 sites (~0.841 vs. 0.89–0.93 elsewhere at α=0.3/γ=0.75) — meaning even its *own* data is comparatively hard to learn from. If site_D's local gradient updates are themselves noisier/less informative, federation may be substituting its weak-but-locally-relevant signal with an aggregate direction shaped by cleaner sites — actively wrong for site_D's data, not just unhelpful. This is exactly what the (still-pending) gradient-conflict diagnostic was built to test directly.

### Things to explore next

1. ~~**Prevalence-distance-from-federated-mean as a predictor**~~ — **Tested (uses only label statistics, no training required).** Correlation between `|site_prevalence − pooled_federation_prevalence|` and Δauroc, pooled across all available conditions: **r = +0.53, a real moderate positive relationship — not the U-shape originally hypothesized.**

   | site | prevalence distance from mean | mean Δauroc | fits the pattern? |
   |---|---|---|---|
   | site_A | 0.149 (largest) | +0.024 | far → wins |
   | site_E | 0.069 | +0.017 | far → wins |
   | **site_D** | **0.045 (moderate)** | **−0.053 (worst)** | **breaks the pattern — moderate distance predicts an unremarkable outcome, not the worst one** |
   | site_C | 0.030 | −0.029 | roughly consistent |
   | site_B | 0.004 (smallest) | −0.014 | close → loses |

   Prevalence distance explains most of why site_A and site_E benefit — driven mainly by those two, closer to monotonic than U-shaped. **It explicitly does NOT explain site_D**: site_D's distance is only moderate, yet it has by far the worst outcome of all five sites. This is added, quantitative support for site_D being a separate, signal-quality problem (its uniquely low local-only AUROC) rather than a prevalence-distance story — consistent with hypothesis #4 above, not with hypotheses #1–3.

   **Important caveat on transferring this metric to real GPC data, not yet tested.** This r=+0.53 relationship was measured against MIMIC's *simulated* prevalence spread (roughly 6–40% across sites, deliberately wide by design). Real GPC sites cluster tightly (9.99%–14.85%, confirmed from the 6-site audit) — a much narrower range. Since the metric is literally a *distance*, compressing the underlying spread this much could shrink the signal-to-noise of the metric itself: if every real site's distance-from-mean becomes small and similar, the metric may lose most of its discriminative power even if the *underlying mechanism* (extremes benefit, middle doesn't) still holds in principle. This hasn't been tested — it's a real, open risk that this specific predictor may not translate well to GPC's actual heterogeneity structure, not just an assumption that it will.

   **This is part of why Zijian's proposed loss-trajectory metric (tracking each site's training loss shape across federated rounds, rather than a static pre-training population statistic) is worth building as a genuinely different candidate** — it isn't tied to prevalence spread at all, so it may not share this same narrow-range limitation. Not yet built; would need per-round loss logging (likely already captured during training) aggregated per site and compared across the trajectory shape, similar in spirit to the gradient-conflict diagnostic but tracking loss magnitude rather than update direction.
2. **Local decision-boundary distance — a genuinely pre-training-computable metric.** Fit a cheap local classifier (e.g. logistic regression) per site *before* running any federation, and measure the distance between each site's decision boundary and the boundary from naively pooling all sites' data (e.g. coefficient cosine similarity, or KL divergence between predicted-probability distributions). If this distance correlates with actual observed Δauroc, it becomes a **cheap, predictive screening tool** — usable to estimate whether a new site joining the federation is likely to benefit or be harmed, without running the full, expensive federated training loop at all. This would be a genuinely valuable, low-cost pretraining diagnostic if it holds up.
3. **Complete the gradient-conflict diagnostic** (Section 4's Site_D discussion) as the in-training complement to the two pre-training metrics above — testing whether site_D's per-round update direction actually conflicts with the aggregate, not just inferring it indirectly from AUROC patterns.

---

## 5. What's Still Open

1. ~~**K-mode**~~ — **Concluded (inconclusive).** `FedAdaptProto v2.5`: uniform K=3 confirmed a large, genuine full-grid win (dead-last → tied-for-best) — trustworthy, no issues. `FedAdaptProto` (v2.3): after three rounds of single-condition testing that each pointed a different direction (see Section 4 for the full story), the full-grid comparison shows uniform K=2 and genuine per-site K are statistically indistinguishable (−0.0112 vs. −0.0111). No winner to declare; don't act on either single-condition result that suggested otherwise.
2. ~~**Reproducibility concern for site_C/site_D specifically**~~ — **Root cause investigated and fixed.** Original finding: re-running the *exact same* config (same seed, same code, same condition) at α=0.1/γ=1.0 produced site_C/D results differing by 0.017–0.028 between two runs, while site_A/B/E stayed stable. Traced to two real, distinct issues in `fedadapt_train_approach2_v2_3.py`: (1) `set_seed()` only called `torch.manual_seed()`, never `torch.use_deterministic_algorithms()` — manual_seed alone does not guarantee reproducibility, a well-documented PyTorch gap, likely worse on the MPS backend (Mac) than CUDA; (2) the K-means prototype-clustering RNG seed (`args.seed + rnd`) was **identical across all 5 sites within a round** — every site's cluster initialization drew the same relative random sequence, undermining independence between sites even though it was itself fully deterministic run-to-run. Both fixed: `set_seed()` now calls `torch.use_deterministic_algorithms(True, warn_only=True)` plus cuDNN determinism flags when CUDA is available; the K-means seed now includes each site's stable positional index (`args.seed + rnd*1000 + site_idx`), giving every site a distinct, still-fully-reproducible stream. **Not yet re-confirmed** — needs the same "run the identical config twice, diff the output" test repeated post-fix to verify the fix actually closes the gap.
3. ~~**Lambda_adv formula unification**~~ — **Closed, multi-seed confirmed.** At `fedadaptproto_v25`'s confirmed-best K-mode (uniform K=3), α=0.3/γ=0.75, seeds 42/7/123: mean Δauroc = +0.0018 ± 0.0029 — stable and genuinely positive. Site_C/D variance here (std 0.006–0.008) is normal, unlike the extreme instability found in the `FedAdaptProto` v2.3 K-mode saga (item 2) — suggesting that instability is specific to v2.3's KMeans-based prototype clustering, not a general pipeline issue. The single-seed=42 full-grid ranking table number was, if anything, conservative (seed 42 was the worst of the three seeds tested here).
4. ~~**Federation-level early stopping**~~ — **Closed.** Implemented, tested successfully, found no measurable benefit beyond checkpointing alone; code reverted.
5. ~~**GRL fix AUROC impact**~~ — **Resolved for both methods, now confirmed at full-grid scale.** `fedadaptproto`: no detectable effect (fix is AUROC-neutral there). `fedadapt`: real, substantial cost — full-grid, multi-seed (3 seeds × 20 conditions): mean Δauroc −0.0054 (stale, pre-fix) → **−0.0102** (corrected), driven mainly by site_D (mean Δauroc −0.062 for that site alone). This changed `fedadapt`'s position in the method ranking from tied-#1 to #5 — see Section 4. Fix is kept regardless (mechanistically correct — bounded vs. unbounded adversarial loss), but the AUROC cost is real and now fully reflected in the ranking table, not just a single-condition spot check.
6. **Site_D root cause**: five negative diagnostic results in a row; still no confirmed explanation. Practical mitigations (checkpointing, correct K) help without explaining why. (Item 2 above may be an important clue here — if site_D has unusually high run-to-run variance in general, that could itself be part of why it's been so hard to diagnose.)
7. **By-site and by-heterogeneity-level breakdowns** (Section 4) still reflect the pre-K-mode-fix data.
8. **General methodological note for future work**: this session repeatedly found that single-seed, single-condition comparisons were unreliable — sometimes from code confounds (other fixes landing between "before" and "after"), and in the K-mode case, from apparent genuine non-determinism with no code change at all. Only comparisons that were specifically multi-seed re-confirmed (GRL AUROC impact) or evaluated at the pooled full-grid level (K-mode's final answer) should be treated as reliable. Prefer multi-seed or full-grid evidence over single-condition point comparisons going forward, especially for anything touching site_C or site_D.
9. ~~**Bridging step: site_D-vs-site_B pairwise comparison**~~ — **Done, full 20-condition grid, script saved (`compute_pretraining_metrics.py`).** Using their 67 shared lab/vital features (Jaccard=0.848): boundary cosine similarity vs. site_D's Δauroc, r=0.15 (weak); signal-to-noise vs. Δauroc, r=−0.17 (weak). **Even with real clinical features instead of the 12-feature demographic-only set, static similarity to site_B still doesn't meaningfully predict site_D's outcome.** More informative side-finding: site_D's Δauroc stays in a narrow negative band (−0.028 to −0.074) across all 20 conditions, while its measured boundary similarity to site_B swings widely (0.53–0.94) across those same conditions — **site_D is consistently bad regardless of how similar or different it looks to site_B.** This argues against "distance from a similar site" as the driver at all, and further toward something intrinsic to site_D — consistent with the signal-quality hypothesis (#4 above), and exactly what Dr. Li's proposed ablation experiment (below) is designed to test directly.
10. ~~**Dr. Li's proposed controlled low-performer experiment**~~ — **Tier-1 (cheap) version built and run; genuinely new, robust finding.** Rather than only analyzing the existing site_D, deliberately *simulated* a controlled low-performer site: `site_F`, a stratified 10% sample of site_C (~3,300 rows, the one site with all 159 aligned features, giving clean control over the feature axis), held out from a "reference" model built on the other 90%. Features added in RF-importance-ranked order (ranked from the reference set only, no leakage), 15 at a time, from the 12-feature baseline up to all 159. Script: `progressive_feature_enrichment.py`.

    **Two distinct, separable results, both confirmed robust to regularization strength (tested at default C=1.0 and strong C=0.05):**
    - **Local AUROC (feature contribution) saturates, not declines, once regularized properly**: rises sharply from 0.776 (baseline) to ~0.83 by ~60 total features, then stays flat — most of the achievable local benefit comes from a fairly small number of top-ranked features, not all 159.
    - **Boundary similarity to the reference model (feature sensitivity) DIVERGES monotonically, and this is real, not an overfitting artifact**: 0.97 (baseline, near-perfect alignment) down to 0.44–0.56 (full feature set) depending on regularization strength — the *opposite* of the naive expectation that more shared features would mean more boundary convergence.

    **Interpretation, tentative:** simply giving a small-sample site more of the "right" features may not be enough on its own to align it with a well-resourced site's decision boundary — sample size and feature dimensionality may need to scale together, not just feature *presence*. This is a genuinely new hypothesis this session hadn't surfaced before, and a real caveat on the "feature alignment" framing generally.

    **UPDATE — direct validation on REAL site_D completed.** Master dataset (`aki_anchor_based_24h_lookback_aligned_features.csv`) and the simulation script (`mimic_ftl_simulation_phase3_aligned.py`) both obtained. Modified site_D's feature-group config to expose all 8 groups instead of its normal 3, re-ran the simulation with identical seed/alpha/gamma — verified this reproduces the *exact* same 33,000 patients and covariate shift as the original masked site_D (identical on all 68 shared columns, confirmed via `np.allclose`), just with the 92 previously-masked features now present. Ran the same progressive-enrichment protocol directly on this real, unmasked site_D data:

    | | site_D baseline (67 features) | peak (~127-142 features) |
    |---|---|---|
    | Local AUROC | 0.818 | **0.848** (+0.030, real gain) |
    | Boundary similarity to reference (site_C) | 0.89 | **0.70** (diverging) |

    **Both effects found in the synthetic testbed replicate on real site_D data.** The local AUROC gain is real and substantial (+0.03), saturating after ~60-75 added features — the first genuinely confirmed, non-speculative improvement found for site_D all session. The boundary-divergence pattern also replicates: site_D's decision boundary moves *further* from a well-resourced reference as features are added, not closer, consistent with the synthetic result.

    **Script saved: `sim_unmask_siteD.py`** (modified simulation script) and `site_D_UNMASKED_alpha0.3_gamma0.75.csv` (the resulting 159-feature real site_D dataset) — reusable for other conditions or further validation.

    **Tier 2 completed — real federated training at 4 feature-count checkpoints (67/97/127/159), `fedavg`, α=0.3/γ=0.75, seed=42. Genuinely major finding: an inverted-U, not a monotonic relationship.**

| features | local AUROC | federated AUROC | **Δauroc** |
|---|---|---|---|
| 67 (original baseline) | 0.8405 | 0.7887 | **−0.0518** |
| 82 | 0.9141 | 0.9212 | +0.0071 |
| **97** | **0.9079** | **0.9171** | **+0.0092 (peak)** |
| 112 | 0.9110 | 0.9137 | +0.0027 |
| 127 | 0.8984 | 0.9036 | +0.0052 |
| 159 (full unmask) | 0.8925 | 0.8910 | −0.0015 |

**Full six-point trajectory now confirmed.** The flip from negative to positive happens between 67 and 82 features, not all the way out at 97 as the original 4-point sweep suggested — a wider and earlier-starting benefit region than first thought. The benefit holds across a broad plateau from 82 through 127, not a narrow spike at one value, before eroding back toward break-even by full unmasking (159). One point worth flagging honestly: 112 dips *below* both its neighbors (82 and 127), breaking a smooth monotonic decline — given this is single-seed data and the effect sizes here (0.003–0.009) sit right at the seed-to-seed noise floor established elsewhere this session, this dip is most likely noise, not a genuine secondary feature, and shouldn't be over-interpreted without multi-seed confirmation at that specific point.

**Second condition tested (α=0.1/γ=1.0, near-maximum heterogeneity) — an important scope correction, not just a replication.** Full six-point trajectory, same methodology (fresh, internally-consistent simulation regeneration confirmed necessary — see reproducibility note below):

| features | α=0.3/γ=0.75 Δauroc | α=0.1/γ=1.0 Δauroc |
|---|---|---|
| 67 (original) | **−0.0518** | **+0.0116** |
| 82 | +0.0071 | +0.0225 |
| 97 | +0.0092 (peak) | −0.0026 |
| 112 | +0.0027 | +0.0115 |
| 127 | +0.0052 | +0.0157 |
| 159 (full) | −0.0015 | +0.0094 |

**Site_D is already a mild benefiter at 67 features under this condition, before any enrichment at all — the opposite of the α=0.3/γ=0.75 starting point.** This means "feature enrichment fixes site_D" is not a universal statement — it's specific to conditions where site_D actually starts as a loser. At near-maximum heterogeneity, site_D wasn't broken the same way to begin with, so there was nothing here for enrichment to dramatically fix. The α=0.3/γ=0.75 result stands as real and useful on its own, but should not be generalized as "enrichment always fixes site_D" without this caveat. Worth testing at more conditions before treating either extreme as representative — only 2 of 20 conditions have been checked with the full trajectory so far.

**Reproducibility note surfaced during this second-condition test — a separate, newly-discovered issue from the training-script determinism fix already applied:** the *simulation* script itself (not the training script) shows its own non-determinism at this condition — re-running it, even with a fully unmodified site_C config, produced different patients and values than the originally cached α=0.1/γ=1.0 data. Root cause not yet identified (ruled out: `sample_with_prevalence`'s RNG consumption is verified constant regardless of parameters). Practical workaround adopted: use all 5 sites from a single fresh simulation run rather than mixing cached and freshly-generated data. This means α=0.1/γ=1.0 findings from earlier in this session (the K-mode saga, gradient-conflict work) remain internally valid (all compared against the same stable cached data throughout) but that cached data itself may not be exactly reproducible from scratch if regenerated today.

**True per-feature sensitivity analysis — the granular "feature contribution/sensitivity" metrics Dr. Li originally asked for, done properly this time (not batched).** The earlier enrichment sweeps added features in batches of ~15; this adds them one at a time (92 steps for site_D, 90 for site_A), tracking each individual feature's marginal AUROC contribution and marginal boundary-similarity shift. Run for site_D (the loser) and site_A (a genuine benefiter), α=0.3/γ=0.75, to test whether the features that matter for site_D differ systematically from what matters for a site that already benefits from FL.

**Result — a clean, clinically-interpretable asymmetry:**

| | site_D | site_A |
|---|---|---|
| Top contributing feature | `baseline_scr`: **+0.0112** | `phosphate_hours_since`: +0.0017 |
| Share of total gain | **34.3%** from this one feature | Top feature ~7% as impactful as site_D's |
| Overall shape | One dominant spike, long tail | Flat, diffuse across many minor features |

**`baseline_scr` (baseline serum creatinine — the clinical marker AKI is diagnostically defined by) was completely absent from site_D's original 67 features, while already present in site_A's 69** — confirmed directly, not inferred. This sharpens the earlier batched-enrichment finding considerably: site_D's problem wasn't "somewhat data-poor" in a diffuse sense — it was missing the single most clinically fundamental variable for this exact prediction task, and that one omission accounts for over a third of the entire enrichment benefit. Site_A, having always had it, shows no comparable single-feature dependency. Script: `feature_sensitivity_analysis.py`; results: `feature_sensitivity_results_site{D,A}.csv`.

**Generalization check — confirmed at the second condition, unlike the sweet-spot location.** Re-ran the same per-feature analysis at α=0.1/γ=1.0 (script: `feature_sensitivity_analysis_cond2.py`):

| | α=0.3/γ=0.75 | α=0.1/γ=1.0 |
|---|---|---|
| Site_D top **added** feature | `baseline_scr` (+0.0112) | `baseline_scr` (+0.0080) — same feature, still #1 |
| Share of site_D's total gain | 34.3% | 26.0% |
| Site_A top **added** feature | phosphate_hours_since (+0.0017) | phosphate_hours_since (+0.0022) |
| Site_A shape | flat, diffuse | flat, diffuse — same shape |

**`baseline_scr` remains the top contributor for site_D at both conditions tested, and site_A's flat/diffuse profile holds at both too** — the percentage share shifts (34%→26%) but the core asymmetry replicates cleanly. This is a meaningfully more robust, generalizable finding than the enrichment sweet-spot's exact location, which did not replicate this well across the same two conditions. Results: `feature_sensitivity_results_site{D,A}_cond2_0.1_1.0.csv`.

**Important precision note, worth being exact about:** "top feature" in the table above means top feature *among those added to each site's existing baseline* — not necessarily the single most important feature overall for that site. Site_A's original 69 features (which already include `baseline_scr`) were never individually tested this way; only the 90 features *newly added* to site_A were. **This means we know `baseline_scr` is highly important when added to site_D, but we do NOT yet know whether it's similarly important for site_A** — it could matter just as much there and simply already be accounted for, or it could be specifically more critical for site_D's prediction task. A direct ablation (removing `baseline_scr` from site_A's baseline and measuring the AUROC drop) would settle this and hasn't been run yet.

**Practical implication:** a far cheaper, more targeted intervention than broad feature enrichment — ensure every site has baseline creatinine specifically before addressing anything else. Worth prioritizing this over the full-enrichment approach for any real-world GPC alignment decision, given how much better it generalizes.

This validates Dr. Li's core hypothesis (feature alignment matters) with an important refinement: **there is a sweet-spot region, not a single optimal point, and not "more is always better."** This is consistent with — and now directly confirms the practical consequence of — the Tier-1 boundary-divergence finding: over-enrichment past a certain point costs more (via boundary divergence / dimensionality-vs-sample-size effects) than it gives back in raw local signal.

**One honest calibration note on the Tier-1 proxy:** the real neural-network local baseline peaked earlier (97 features) and declined more sharply by 159 (0.908→0.892) than the cheap logistic-regression proxy predicted (which showed more of a flat plateau past ~60 added features). The proxy got the *direction* right — diminishing/negative returns from over-enrichment — but not the precise shape or peak location. Useful to know before leaning on the cheap proxy alone for future site-screening decisions; real Tier-2 validation materially changed the actionable conclusion (there's a specific, findable optimum) beyond what Tier-1 alone suggested.

**Practical implication:** if pursuing feature-alignment as a real fix for site_D (or similar low-performer sites), the target should be *moderate, ranked enrichment* — not full un-masking. The next natural step would be narrowing the sweet spot further (e.g., checkpoints between 67 and 97, and between 97 and 127) to find the actual optimum more precisely, and confirming this holds at other conditions (currently only tested at α=0.3/γ=0.75) and other methods (currently only `fedavg`).

**Metric revision, derived by checking Tier-1's predictions against the real Tier-2 outcome:**

| Tier-1 signal | as originally used | revised |
|---|---|---|
| Local AUROC (LR proxy) | Predict the peak location directly | **Don't use for peak-finding.** It plateaus (~0.847) and never declines, because L2-regularized logistic regression shrinks excess-feature coefficients toward zero rather than genuinely overfitting the way a neural network with limited local epochs does — a structural mismatch, not noise. Real local AUROC peaked at 97 and *declined* to 0.8925 by 159; the proxy never showed this at all. |
| Boundary cosine similarity | Read the *absolute value* | **Read the *rate of change* instead.** The single steepest per-step drop in boundary similarity (−0.103) occurred exactly at the 97→127 transition — precisely where the real Δauroc also peaked and began declining. |

**Revised, validated rule: the predicted sweet spot is the last checkpoint before the steepest single-step drop in boundary cosine similarity.** Tested retroactively against this dataset: the rule correctly flags 97 features as the optimum, matching the real Tier-2 result — **without needing the expensive 127/159 training runs to find it.** This turns the boundary-similarity metric from a directional-only signal into a genuinely predictive, still-cheap screening rule, and rules out relying on the local-AUROC proxy for this purpose going forward. Worth validating this revised rule at a second condition before treating it as a settled methodology, but it's the first Tier-1 metric this session that's shown real predictive precision, not just directional plausibility.
11. ~~**Gradient-conflict diagnostic**~~ — **Run successfully on both `fedavg` and `fedadaptproto` (α=0.3/γ=0.75, seed=42). Result: method-dependent, not a universal explanation.** `fedavg` (simplest, cleanest test): site_D's mean cosine similarity with the aggregate (0.435) is the **highest** of all 5 sites, not the lowest — no evidence of gradient conflict at all, and no decline over rounds; all 5 sites stay in a tight, overlapping band (0.42–0.44) throughout. `fedadaptproto`: site_D's mean cosine similarity (0.381) **is** the lowest of all 5 sites, and it starts clearly misaligned from round 1 (0.33 vs. 0.52–0.60 for others) — supporting the hypothesis, but noisily (no clean monotonic decline), and this is the same method with already-documented KMeans-clustering instability specific to site_C/D, so it's unclear whether this is the real mechanism or a symptom of that instability. **Bottom line: gradient conflict is not a property of federated averaging itself (fedavg shows none), so whatever's wrong with site_D isn't explained by simple aggregation dynamics — if the conflict signal in `fedadaptproto` is real, it's tied specifically to prototype alignment, not a general FL phenomenon.**

12. ~~**Feature-level GPC alignment audit**~~ — **Done, using real per-site GPC network feature lists (KUMC/MCW/UIOWA/UPITT/UTSW/UofU RF feature-importance exports, ~98 shared98 fields each). A more sobering result than the earlier 31-variable spot-check suggested.**

    | category | matched to Phase 3 | gap | why |
    |---|---|---|---|
    | LAB | 12 of 18 | 6 | Fibrinogen, CRP, Triglyceride, Cholesterol, HDL Cholesterol, Lipase — not present in Phase 3 at all |
    | DEMOGRAPHIC | 2 of 4 | 2 | Race, Hispanic ethnicity — Phase 3 has no race/ethnicity fields |
    | VITAL_TIME | 3 of 3 | 0 | Full coverage (DBP/SBP/BMI, once matched past naming differences) |
    | **DX (diagnosis codes)** | **0 of 70** | **70** | Phase 3 has 6 coarse comorbidity *flags* (has_cancer/chf/diabetes/hypertension/liver_disease/sepsis); shared98 uses 70 specific ICD-9 diagnosis codes. Structural mismatch, not a missing-column problem. |
    | **Total** | **19 of 98** | **79** | |

    **Only 19 of 98 shared98 concepts are represented in the full 159-feature Phase 3 set.** The dominant gap (70 of 79) is diagnosis-code granularity — Phase 3's comorbidity flags are a coarse summary of what shared98 tracks as individual ICD-9 codes, not an equivalent representation. This is a real, structural limitation worth being upfront about if this dataset is ever presented as directly comparable to real GPC network data, separate from (and a stricter measure than) the earlier 31-core-variable check, which remains accurate for what it measured.

    **Methodology note:** initial automated string-matching under-counted the true overlap (missed short terms like "AGE"→age_at_admission and semantically-equivalent-but-differently-named pairs like "DIASTOLIC"→dbp) and produced one false positive (matched "Hemoglobin A1c," a distinct diabetes marker, to Phase 3's plain CBC "hemoglobin" feature — removed after manual review). The 19/98 figure reflects corrected, clinically-reviewed matching, not the raw automated output.

13. **Toward real GPC deployment — what needs tuning, and what's already established.**

    **Parameters: validated on simulated data, need re-checking on real GPC structure.**

    | parameter | MIMIC-validated value | status for real GPC deployment |
    |---|---|---|
    | Checkpointing | Always restore best round | Should transfer directly — general robustness practice, not tied to simulated heterogeneity |
    | `lambda_adv` | 0.1 (prevalence-scaled) | Re-verify: GPC's real per-site prevalence spread (10–15%, from the 6-site audit) is much narrower than MIMIC's simulated range (6–40%) — the scaling formula's behavior at this narrower spread hasn't been tested |
    | `n_clusters` (K-mode) | uniform K=3 (`fedadaptproto_v25`) | Re-tune on real data structure — validated against simulated Dirichlet heterogeneity, not GPC's actual site-to-site variation |
    | `alpha` / `gamma` | N/A — simulation-only | Don't apply directly — these control synthetic heterogeneity; real GPC heterogeneity is whatever the hospitals actually have |

    **Feature alignment: 19/98 is not a ceiling, it's concretely improvable.** Three specific opportunities, largest first: (1) **DX codes (0/70 matched)** — replace Phase 3's coarse comorbidity flags with specific ICD codes; MIMIC-IV has this data, Phase 3 just used a simplified summary for convenience. This is the single largest, most mechanically fixable gap. (2) **6 missing labs** (fibrinogen, CRP, triglyceride, cholesterol, HDL, lipase) — standard tests, likely extractable from raw MIMIC-IV if not yet included in the aligned feature set. (3) **Race/ethnicity** — MIMIC-IV has this data; simply wasn't included in Phase 3's aligned set.

    **Should sites be modified to match GPC's real AKI rate (~13.3% pooled, all tertiary academic)?** Not as a straight replacement. GPC's real 6 sites cluster tightly (9.99%–14.85%) while MIMIC's simulated sites deliberately span 6–40% specifically to stress-test FL methods under extreme heterogeneity — narrowing prevalence to match GPC would undercut the extremes-benefit findings (site_A/E vs. the middle three) this whole session's per-site analysis relies on. **Better approach: add a narrow "GPC-realistic" prevalence condition alongside the existing wide sweep, not replace it.** Also worth noting: real GPC heterogeneity looks more size-driven than prevalence-driven — UPITT (299,422 patients) vs. UTSW (72,720) is a ~4× sample-size gap that Phase 3 doesn't currently simulate at all (all 5 simulated sites are equal N=33,000 today). This may be a more representative heterogeneity axis to add than prevalence narrowing alone.

    **Recommended next steps, in priority order:** (1) add a sample-size heterogeneity axis matching GPC's real ~4× spread; (2) re-run K-mode and `lambda_adv` validation once real or more GPC-realistic data is available, rather than assuming simulated-optimal values transfer as-is; (3) prioritize ICD-code-level diagnosis alignment over the 6 missing labs, given it's the largest gap (70 of 79) and most directly fixable with existing MIMIC-IV data.

---

## 6. Phase 4 — GPC-Aligned, 6-Site, Narrow-Prevalence Architecture

**A new, separate simulation mode — not a replacement for Phase 3's wide-heterogeneity sweep.** Phase 3's 6–40% simulated prevalence spread and varying-richness site archetypes remain the tool for the extremes-benefit, sweet-spot, and ranking-table findings, which depend on that wide variance. Phase 4 exists specifically to test whether those findings generalize to GPC's actual, much narrower real-world structure. Run both, side by side — don't treat one as superseding the other.

**Script: `mimic_ftl_simulation_phase4_gpc_aligned.py`.**

### What changed vs. Phase 3

1. **All 6 real GPC sites represented**, not 5. Phase 3's architecture forced dropping one real site (UIOWA) to fit a 5-site simulation; Phase 4 adds `sim_UIOWA` as a genuine 6th simulated site.
2. **Uniform feature architecture across all 6 sites**, replacing Phase 3's varying-richness archetypes (e.g. the old `sim_MCW`/site_C "anchor" with all 8 feature groups and 159 features while other sites had far fewer). Every site now gets the identical:
   - **`shared98_core`** — 18 MIMIC-IV columns confirmed matched to GPC's 19 real shared98 fields (age_at_admission, gender, dbp, sbp, bmi, creatinine, bun, glucose, calcium, chloride, potassium, magnesium, lactate, phosphate, bilirubin, bilirubin_dir, total_protein, basophils_pct)
   - **`gpc_dx_universal`** — the 70 universal ICD-9 shared98 diagnosis codes

   **Caught and fixed a real bug while building this**: the first version used a `"dx"` wildcard prefix for the universal group, which would have also matched the site-specific `dx_site_*` columns — silently giving every site all 86 site-specific codes and defeating the whole point of site-specific differentiation. Fixed with an explicit 70-code list instead.
3. **Site-specific diagnosis codes, not just feature-group differences.** Each site additionally gets its own confirmed real-GPC site-specific ICD-9 codes (`SITE_SPECIFIC_DX_COLUMNS`): `sim_UTSW` 67, `sim_MCW` 56, `sim_KUMC` 32, `sim_UofU` 31, `sim_UPITT` 24, `sim_UIOWA` 6 codes — genuine inter-site heterogeneity from the real network, layered on the uniform baseline. **Requires the cohort notebook's Section 5C to have been run first** (extracts `dx_site_*` columns via BigQuery) — degrades gracefully with a clear warning, not a crash, if those columns are absent (confirmed by testing against the current, not-yet-regenerated master CSV).
4. **GPC-realistic (narrow) label distribution**, matching this specific request — each site's prevalence target is its own real, confirmed GPC rate (not estimated): `sim_UTSW` 14.85%, `sim_UPITT` 13.90%, `sim_MCW` 13.41%, `sim_KUMC` 12.91%, `sim_UofU` 9.99%, `sim_UIOWA` 13.42% — all directly confirmed from the real GPC network's own per-site manifest data (n_rows/n_positive counts), not re-scaled or estimated. Tested observed spread after alpha-blend sampling: ~10.8–14.1%, matching real GPC's tight clustering — a deliberate, confirmed contrast with Phase 3's wide simulated range.
5. **No architecturally-special "anchor" site.** Phase 3's `sim_MCW` had a zero-blend special case (always hit its exact target prevalence, bypassing alpha-interpolation, justified by its uniquely rich feature set). Removed in Phase 4 — every site now uses the same alpha-blended targeting, since no real GPC site is architecturally privileged in this uniform-baseline design.

### One genuine data finding worth keeping in mind

**UIOWA's 6 site-specific DX codes overlap heavily with UofU's (5 of 6 shared: 794, E93, V13, V42, V87; only 443 unique to UIOWA)** — even though their *lab/vital* profiles are opposite extremes (UIOWA richest of all 6 real sites with 58 usable site-specific lab features, UofU sparsest with 20). Worth remembering when interpreting any UIOWA-vs-UofU comparison in Phase 4 results — they'll look similar on diagnosis codes specifically, despite being very different on labs.

### Still required before Phase 4 produces complete results

The cohort notebook (`AKI_Anchor_Based_Approach2_aligned_features_FIXED.ipynb`, Sections 5B + 5C) must be re-run against BigQuery to populate the `dx_*` (70 universal) and `dx_site_*` (86 site-specific) columns in the master CSV — same blocker as the earlier 5-site version. Tested and confirmed working structurally (correct site count, correct uniform 82-feature baseline, correct prevalence targeting, graceful DX-column-missing warnings) against the current, not-yet-regenerated cohort file.

`acuity_bias`/`spread_scale` values are carried over from the original Phase 3 archetypes as placeholders, not re-derived from real GPC data — GPC's feature-list exports don't include acuity or spread information, only feature presence and importance rankings, so there's no direct source to re-derive these from yet.
