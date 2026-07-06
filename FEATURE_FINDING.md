# Feature finding & feature selection — analysis template plan

**What this is.** The guiding + progress-tracking document for the engine's **first analysis
layer**: the `lib/analysis/` (and companion `lib/figures/`) templates that answer the two
questions a scientist actually asks of this data —

- **Univariate feature finding** — *which features differ between conditions?* Test each
  feature independently (linear model / OLS, t-test, Mann–Whitney, limma-moderated), then
  correct across features.
- **Multivariate feature selection** — *which features jointly matter?* Let a penalized or
  ensemble model pick them (elastic-net logistic & linear regression, Boruta).

It is **engine-dev planning, not user-facing** — it never ships into a project. It is the
analysis-side companion to [`QC_GAPS.md`](QC_GAPS.md) (which tracks the QC/descriptive
plots). Two source-oracle repos are mined — see **Oracle source repos** just below.

**How to use it.** Same loop as `QC_GAPS.md`: pick a method, **confirm the design with the
user** (a real-data preview before coding is the established pattern — see `id-depth`'s
history), build it to the `lib/` bar following **[`lib/AUTHORING.md`](lib/AUTHORING.md)**,
route every statistic through **[`conventions/statistics.md`](conventions/statistics.md)**,
and update the *Status* table below + `CLAUDE.md`'s *Next* section when one ships.

**Oracle source repos (absolute paths — every oracle reference below resolves to a file in
one of these).**

- **`/home/mriffle/vscode/johnson-5xFAD-lecanemab-mice-AD/`** — the primary oracle (also the
  QC oracle named in `QC_GAPS.md` / `lib/AUTHORING.md`). For this doc:
  `src/feature_finding_ols.py` (per-feature OLS **and** the limma-style empirical-Bayes
  moderation, §A.1 / §A.2) and `src/volcano_plotting.py` (the volcano, §C).
- **`/home/mriffle/vscode/manuscript-trex-phase2a/te-phase2a-pelt/`** — the **multivariate**
  oracle (introduced 2026-06-29; *not* mentioned in `lib/AUTHORING.md`, so it lives only here):
  `src/feature_finding_boruta.py` + `src/boruta_plotting.py` +
  `scripts/run_feature_finding_boruta.py` (Boruta, §B.3), and `src/classification.py` (the
  **unscanned** elastic-net / leakage-safe-classifier candidate, §B.1).

As elsewhere in the engine, `src/` is generalize-able seed code; `scripts/` is
project-specific (never template material) — read it only for the call-site wiring.

---

## The scientist's framing (the anchor — keep this in view)

These templates exist to serve common practice, not to showcase methods. From the
discussion that started this doc:

- **Finding features significantly different between conditions is usually done
  univariately.** The canonical menu: **t-test, Mann–Whitney, limma** — plus **OLS / a
  linear model**, which is the general form that *subsumes* the two-group t-test (two-group
  OLS with a single 0/1 predictor == Student's t) and adds covariate adjustment.
- **Assumptions are a scientist decision.** t-test and limma assume roughly normal,
  roughly equal-variance (log-)abundances; **Mann–Whitney is the nonparametric fallback**
  when those fail. *Which* test — and whether its assumptions hold — must be **surfaced and
  agreed before running**, the analysis-stage analogue of the Stage-2 normalization /
  imputation decisions.
- **Downstream is shared.** Whichever univariate test is chosen, *what is reported and
  visualized is the same*: a per-feature **effect + CI + p + BH-q** table, a **volcano**, a
  **p-value histogram**. So the univariate methods are one family with a swappable test, not
  four unrelated templates.
- **Multivariate methods** the user wants covered: **elastic-net logistic regression**
  (categorical outcome) and **elastic-net linear regression** (continuous outcome), and
  **Boruta** (all-relevant selection — the user is a fan and *has example runs with
  visualizations he finds informative to share*; see B.3).

Scope for now (the user's call): **feature finding / feature selection**, plus — as of
2026-07-01 — **classification as a first-class question** for the elastic-net case (§B.1).
The methods now divide by the *question asked*, not by the algorithm:

- *Which features differ between conditions?* → **univariate DE** (shipped, §A).
- *Which features are all jointly relevant?* → **Boruta** (all-relevant, §B.3).
- *Can the proteome predict the class, and how well?* → **elastic-net logistic
  classification** (§B.1) — a minimal-optimal predictor whose coefficients are reported as a
  caveated interpretation, **not** an all-relevant selection.

This reframing (the user's call — elastic net tuned for prediction gives the *minimal-optimal*
set, so selling it as "the feature finder" both overstates it and duplicates Boruta) pulls
**roadmap #3 (leakage-safe classifier)** forward as the home for elastic-net logistic. Full
predictive modelling on the *continuous* side (regression, §B.2) remains roadmap #4.

---

## Shared contract every analysis template must honor

Beyond the `lib/` bar (Dataset contract, fail-loud boundaries, `ruff` strict +
`mypy --strict`, planted-truth + real-5xFAD smoke, `__script_meta__` + manifest entry),
analysis templates are bound by `conventions/statistics.md` — the **stats-reviewer** is the
active enforcer (none of this is cleanly hook-checkable):

- **Experimental subset only.** Controls (pools / refs / blanks) are excluded; the analyzed
  set + excluded-control count are recorded in the finding's `provenance.params`.
- **No bare p.** Every significance claim reports **effect size + CI + corrected p**
  *together*. **BH/FDR** named over the stated family. **Report all tests run**, not just the
  hits (feeds the exploration log).
- **Moderated model preferred** for differential abundance over naive per-feature t-tests.
  **Canonical over esoteric** tests.
- **Batch as a covariate**, *not* ComBat pre-correction, for significance testing
  (Nygaard 2016 variance-deflation). Run the key analysis **both ways** (corrected /
  uncorrected) as a robustness check.
- **Small-n / confounded → `exploratory`.** Confounds are first-class and carried into the
  finding's caveats.
- **For any model that learns from data** (elastic-net, Boruta): **no leakage** —
  preprocessing inside CV folds, **group/subject-aware folds** matched to the generalization
  unit, a **mandatory label-shuffle null**. Class imbalance → balanced metrics + stratified
  folds (link the caveat finding).

---

## Status

| Method | Family | Source oracle | Status | Template |
|---|---|---|---|---|
| Linear model / OLS (+ covariates) | univariate | `feature_finding_ols.py` → `fit_ols_per_feature` | ✅ **Shipped** (v0.1, 2026-06-29) | `lib/analysis/differential-abundance` (`method="ols"`) |
| limma-moderated (empirical Bayes) | univariate | `feature_finding_ols.py` → `moderate_variances` | ✅ **Shipped** (v0.1) | same template; `method="moderated"` (default) |
| Welch / Student t-test | univariate | — (none) | ✅ **Shipped** (v0.1) | same family; `method="welch"` |
| Mann–Whitney U | univariate | — (none) | ✅ **Shipped** (v0.1) | same family; `method="mannwhitney"` (HL shift + rank CI) |
| Volcano plot | output viz | `volcano_plotting.py` | ✅ **Shipped** (v0.1) | `lib/figures/volcano` |
| p-value histogram | output viz | — (fresh design) | ✅ **Shipped** (v0.1) | `lib/figures/pvalue-hist` |
| Elastic-net logistic **classification** | multivariate / classification | `te-phase2a-pelt/src/classification.py` (scanned 2026-07-01) | ✅ **Shipped** (v0.1, 2026-07-06) | `lib/analysis/classification` + `lib/figures/classification` |
| Elastic-net linear regression | multivariate | — (not yet scanned) | Not started | TBD (mode of regression #4?) |
| Boruta | multivariate / selection | `te-phase2a-pelt/src/feature_finding_boruta.py` | ✅ **Shipped** (v0.1, 2026-07-06) | `lib/analysis/boruta` (`boruta_select`) |
| Boruta importance box-plot | output viz | `te-phase2a-pelt/src/boruta_plotting.py` | ✅ **Shipped** (v0.1, 2026-07-06) | `lib/figures/boruta-importance` |

**✅ Univariate layer shipped (v0.1, 2026-06-29).** The whole univariate family — `ols` /
`moderated` / `welch` / `mannwhitney` as one swappable `method=` over the settled `contrast=`
+ `covariates=[…]` API (§A.0b) — plus the **volcano** and **p-value-histogram** companions,
all to the `lib/` strict bar (ruff strict + `mypy --strict`, planted-truth + real-5xFAD smoke
reproducing the source oracle's OLS/moderated numbers exactly; 90 new tests). The **CI gap was
closed** (the oracle reported effect + p only; the template adds effect-scale CIs for every
method — §A.1/A.3/A.4). Validated on the real 5xFAD disease contrast: the top hits are the
canonical AD proteins (APP, midkine, APOE, clusterin, complement C1q), the p-value histogram is
textbook-healthy (uniform + spike-at-0, π0≈0.73). Wired into `lib/manifest.md`,
`conventions/{statistics,visualization}.md`, and `commands/stage4-explore.md`.

**Volcano label placement — ✅ DONE (volcano v0.2).** The `annotate_top=` labels were
drawn with a fixed offset and **no collision avoidance**, so a tight cluster of
co-significant hits overprinted (APP/A4 at the q-ceiling rendered as `A45xFADA4`). Fixed by
placing labels **collision-free via `textalloc`** (repelled off each other and the point
cloud, leader lines back to each point, clamped inside the axes). Chosen after a parallel
bake-off of four approaches (adjustText, textalloc, a `dynamic_range` port, and a
from-scratch repel) rendered against the A–F stress cases on real data; **textalloc** won
as the trusted library that *also* ships `py.typed` (passes `mypy --strict` with no
override) and is fast. New shipping dep (`textalloc==1.2.3`) added to `setup-env` +
`requirements-dev.txt`; a planted-dense-cluster no-overlap invariant added to the tests.
See [`VOLCANO_LABELS.md`](VOLCANO_LABELS.md) (the bake-off record).

**✅ Boruta shipped (v0.1, 2026-07-06).** The all-relevant complement to the classifier —
`lib/analysis/boruta` (`boruta_select`) + `lib/figures/boruta_importance`, strict-clean
(ruff + `mypy --strict`) + 34 tests (planted-truth classification & regression, task
inference + the multiclass path, fail-loud guards, result invariants + determinism, and a
real-5xFAD smoke). Engine settled as **BorutaPy 0.4.3** (open decision #7, option (a)) — it
runs cleanly on the current numpy 2.4 / sklearn 1.9 stack, so the historical breakage is
moot; the private-method shadow-history subclass is isolated fail-loud (`BorutaShadowHistoryError`
if the interception stops aligning with `importance_history_`). Structurally simpler than
the classifier: the shadow features are a **built-in null**, so no external null and no CV —
one run on the whole experimental matrix. Validated on the 5xFAD binary genotype contrast:
**18 Confirmed / 7 Tentative**, and the Confirmed set **contains** the classifier's validated
top coefficients (APP/A4, APOE, C1q trio, clusterin, midkine) plus the correlated neighbours
L1 had zeroed (APLP2, the full C1q trio, the testicans) — the all-relevant ⊇ minimal-optimal
containment, on real data. New pinned shipping dep `boruta==0.4.3` (setup-env + requirements-dev).
Wired into `lib/manifest.md`, `conventions/{statistics,visualization}.md`, and
`commands/stage4-explore.md`. Open decisions #7 (engine) and #8 (Boruta reuses the selection
evidence kind) are settled.

**Next:** the leakage-safe **regression** template (roadmap #4) — elastic-net **linear**
regression (§B.2, the continuous analogue of the shipped classifier) and the general
regression case. Elastic-net logistic **classification** shipped 2026-07-06
(`lib/analysis/classification`, nested-CV AUC ≈ 0.92 on the 5xFAD genotype contrast).

---

## A. Univariate feature finding

### A.0 The unifying view
All four methods are *"one test per feature, then BH across features."* They differ only in
the per-feature test and its assumptions; the **output contract and visuals are identical**
(§C). So build them as **one family with a swappable `method=`**, sharing the results table,
volcano, and p-value histogram — not four templates.

### A.0b The contrast API — ✅ SETTLED (2026-06-29)
The study-agnostic interface the whole univariate family shares — the replacement for the
source's per-study design-matrix builders. **Constrained shape, v0.1:**

```python
differential_abundance(
    dataset,                        # Dataset, experimental subset, log2
    contrast="genotype",           # the metadata column under test
    covariates=["sex", "batch"],   # nuisance columns to adjust for (batch lives here)
    reference={"genotype": "WT"},  # optional; default = sorted-first level
    method="moderated",            # ols | moderated | welch | mannwhitney
) -> DifferentialAbundanceResult   # .table sorted by q
```

Settled details:
- **Design build.** Intercept + contrast dummies + covariate columns, assembled internally
  (lift the source's `_encode_binary` / `_encode_one_hot`), handed to the OLS/moderated core
  (which already wants an intercept-first matrix). The named per-study builders do **not** ship.
- **Reference level.** Explicit per-column `reference=`; default the **sorted-first** level.
  Effect sign is *non-reference vs reference*, so it is controllable and **recorded in
  `provenance.params`**.
- **Categorical vs continuous.** Inferred from the metadata column dtype (non-numeric →
  treatment-contrast dummies; numeric → a continuous slope), with an explicit override **and a
  loud warning when a numeric column is suspiciously low-cardinality** (the `batch = 1,2,3`
  miscoded-factor trap).
- **Multi-level contrast (k>2).** Emit all **k−1** coefficients (each level vs the reference)
  as labeled row-groups; **BH across features within each coefficient** (matching the source's
  per-column BH), the family named.
- **Return shape.** Fit the **full** model; the result centers on the contrast term(s) for the
  volcano / finding, but the table also exposes the covariate terms' stats so *"report all
  tests run"* is honored and the exploration log sees them.
- **Effect + CI.** `effect = log2FC` (= the contrast coefficient on log2 data) with a CI
  (`coef ± t·se`; moderated analogue at the inflated df) — the columns the source lacks (§A.1).
- **Deferred to v0.2.** A `formula=` escape-hatch for interaction / transformed models; until
  then, interactions are a documented **project-local adaptation** (the source's
  `build_design_matrix_*_interaction` builders are the worked example).

### A.1 Linear model / OLS (+ covariates) — *source oracle exists*
The general form, and where **batch-as-covariate** lives (the spec-preferred alternative to
ComBat pre-correction).

**What the oracle gives us** (`src/feature_finding_ols.py`, read in full):
- `fit_ols_per_feature(abundances, design_matrix, covariate_names, feature_names)` —
  **vectorized closed-form per-feature OLS**: one shared `(XᵀX)⁻¹` across all features,
  coefficients/SEs/p-values mathematically identical to per-feature `statsmodels.OLS`.
  Intercept-first design (validated). Two-sided t-test p at `df = n_samples − n_params`,
  **BH-adjusted per covariate column** (`adjust_pvalues_bh`, NaN-aware). `NaN → 0` before
  the fit (mirrors the loader's "0 = not detected" policy). Singular-design guard.
- It already **carries the sufficient statistics for moderation** —
  `residual_variances`, `residual_df`, `xtx_inv_diag` — in `OLSFeatureFindingResult`.
- Generic categorical encoders worth lifting: `_encode_binary` (0/1 vs a positive level,
  raises on >2 levels) and `_encode_one_hot` (multi-level, drops a reference, names columns
  `{name}_{level}`).

**What we must ADD vs the oracle (the gap):** **confidence intervals.** The source emits
`coefficient`, `pvalue`, `pvalue_adjusted` only — but `conventions/statistics.md` *requires*
effect **+ CI** + corrected p. Derive `coef ± t_{df,0.975}·se` (and the moderated analogue
at the inflated df, §A.2). On log2 data the **effect = coefficient = log2 fold change**.

**What to generalize away:** the **per-study design-matrix builders**
(`build_design_matrix_all_experimental` and siblings — `gender_M`, `genotype_C57BL/6j`,
`treatment_Lec`, `cohort_2`, `is_not_TR1`, plus the treatment×genotype / treatment×cohort
interaction variants). These bake in one study's schema; the template must instead expose a
**study-agnostic contrast + covariates API** over `Dataset.metadata` (see Open Decision #1).
The named builders stay project-local; the generic encoders + the API ship.

### A.2 limma-moderated (empirical Bayes) — *source oracle exists*
The spec's "moderated model" — **already implemented** in the source, so this is
generalize-and-validate, not build-from-scratch.

- `moderate_variances(result)` → `fit_f_distribution_prior(sigma2, df)`: **Smyth's
  `fitFDist`** — fit a scaled inverse-χ² prior to the per-feature residual variances by
  method-of-moments on `log σ²` (using digamma/trigamma identities), giving `(s0², d0)`.
  Then `moderated_σ² = (d0·s0² + df·σ²)/(d0 + df)`, p from `t.sf` at the **inflated df
  `d0 + df`** (or `norm.sf` in the `d0 → ∞` limit). **≥50-feature guard**, immutable
  `replace`, double-moderation guard.
- **Validate numerically against R `limma::eBayes`** on a shared matrix — the moderated-t
  and prior `(s0², d0)` should match. This is the key correctness test for the template.
- **Default the template to moderated**; expose a raw-OLS toggle (small-n / debugging).

### A.3 t-test (Welch / Student) — *no source oracle*
- Student's two-group t == two-group OLS (A.1 already covers it). **Welch** (unequal
  variance) is the more defensible bare-two-group default and is **not** an OLS special case
  → a thin `scipy.stats.ttest_ind(equal_var=False)` path. Effect = mean log2 difference
  (= log2FC) + CI.
- Assumption: approximately normal log-abundances. **Surface it**; offer Mann–Whitney when
  it fails. Implement as a `method=` of the univariate family, not a separate template.

### A.4 Mann–Whitney U — *no source oracle*
- Nonparametric rank test — the fallback when normality fails / for robustness.
  `scipy.stats.mannwhitneyu`. To honor *effect + CI* (not a bare p), report a rank-based
  effect: **rank-biserial correlation** or the **Hodges–Lehmann shift** + CI.
- **Limitation to state:** no covariate adjustment. If covariates matter, the scientist
  needs the linear model (A.1), not a bare two-group rank test.

---

## B. Multivariate feature selection

**The leakage line (binds all of B).** These models learn from data, so the
`conventions/statistics.md` ML discipline applies: **preprocessing inside CV folds**,
**group/subject-aware folds** matched to the generalization unit, a **mandatory
label-shuffle null**. Selection ≠ prediction, but the same rules hold, and **stability of
the selected set across folds/resamples is the honest readout** (a feature picked once is
noise; one picked in 95% of resamples is real). **Boruta is a partial exception:** its
**shadow (permuted) features are a built-in null**, so it satisfies the null-model
requirement intrinsically; and because its output is a *selection*, not a held-out
performance estimate, the in-fold-preprocessing rule binds any *downstream performance claim*
on the selected set — not Boruta's selection run itself, which legitimately sees the whole
experimental matrix (the driver even runs it on the ComBat-corrected matrix).

### B.1 Elastic-net logistic regression — reframed as **classification** (roadmap #3 pulled forward) — ✅ SHIPPED (v0.1, 2026-07-06)

**✅ Shipped (v0.1, 2026-07-06).** `lib/analysis/classification` (`classify(...)`) + the four
result figures `lib/figures/classification`, strict-clean (ruff + `mypy --strict`) with 21 tests
(planted-truth: a separable signal is recovered at AUC 1.0 with exactly the planted features
selected; pure-noise does **not** beat the null; the outcome/binarize API, fail-loud guards,
grouping, and the figures; a real-5xFAD smoke). **Validated on the 5xFAD genotype contrast:**
nested-CV AUC ≈ 0.92, clears the shuffle null (empirical p at the floor), and the top
coefficients are the canonical AD proteins (APP, midkine, APOE, clusterin, complement C1q).
Wired into `lib/manifest.md`, `conventions/{statistics,visualization,findings}.md`, and
`commands/stage4-explore.md`; no new shipping dep (scikit-learn is already in the `setup-env`
baseline). Two source-scan findings are baked in: the modern sklearn API (`penalty=` is
deprecated → select elastic net by `l1_ratio` alone) and the leakage/null gaps the source
harness lacked (it tuned on all data, had no group-awareness and no null, and silently
`NaN→0`). The design notes below record what was built.

**The reframing (the key decision, the user's call).** Elastic net tuned for prediction yields
the **minimal-optimal** feature set — the smallest sufficient predictive basis — *not* the
all-relevant set. With correlated proteomics features (whole pathways move together) L1
arbitrarily keeps one of a cluster and zeros its neighbors, so the selected set is unstable and
is not "the important features." Selling it as "the feature finder" both overstates it and
duplicates **Boruta** (the all-relevant method, §B.3). So we frame it honestly as a
**classification** method — *can the proteome predict the class, and how well?* — with feature
coefficients reported as a **caveated interpretation** of the classifier, not as an all-relevant
selection. This makes it **roadmap #3 (leakage-safe classifier)** pulled forward, with
elastic-net logistic regression as the default estimator (settles open decision #6).

**The deliverable (settled).** Both the coefficients *and* their cross-fold stability are
essential (user, 2026-07-01), reported as three coupled pieces:

- **Performance vs a label-shuffle null — the gate.** Leakage-safe **nested CV** (tune
  `(C, l1_ratio)` in inner folds, estimate on outer folds), **group/subject-aware** folds
  (optional `groups=` metadata column matched to the generalization unit), **in-fold**
  StandardScaler, `class_weight="balanced"`. Report **balanced accuracy / ROC-AUC ± fold SD**
  against a **mandatory label-shuffle null** distribution + empirical p. **This gates the rest:**
  the coefficient report is only emitted/trusted when real performance beats the null — otherwise
  the coefficients are noise dressed as findings.
- **All-data coefficients — the point estimate.** Tune on all data, refit on all data, report
  **standardized** signed coefficients (magnitude comparable across features = importance; sign
  = direction). This is the model one would actually interpret.
- **Cross-fold stability — the trust annotation. Dedicated fixed-hyperparameter loop (design
  (b)).** Tune once, then resample (repeated stratified [group] K-fold / subsampling) at the
  *fixed* tuned `(C, l1_ratio)` so all coefficients sit at one regularization and are comparable
  (decouples stability from the hyperparameter search). Per feature report **selection
  frequency** (fraction of resamples non-zero), **sign consistency** (fraction of selecting
  resamples agreeing on sign), and the **coefficient distribution** (median + IQR).

**Interpretation caveat baked into the finding.** Correlated features → L1 flips between
redundant proteins across resamples, so a genuinely important feature can show a *low* selection
frequency simply because its correlated neighbor was picked instead — **low frequency ≠
unimportant**. The elastic net's L2 component (`l1_ratio < 1`, grouping effect) softens but does
not erase this; **Boruta is the complement** that confirms both correlated features. State this
in the finding so a low frequency isn't over-read.

**Evidence shape (settles #8).** A classification finding carries run-level **{balanced-acc /
AUC ± SD, shuffle-null distribution + empirical p}** + a per-feature table **{all-data signed
standardized coef, selection frequency, sign consistency, coef median/IQR}** — **no per-feature
q**. The findings schema + stats-reviewer must accept this "classification / selection" evidence
kind alongside the significance kind (§8).

**Figures (two, on the figure foundation).**
- **ROC ± SD vs null** — re-base the source `classification_plotting.plot_roc_curve` onto
  `figure-io` (dual-export + separate legend); overlay the shuffle-null band; **drop seaborn**.
- **Coefficient / importance plot** — top features by `|all-data standardized coef|`, **colored
  by selection frequency** (continuous → viridis, no categorical budget), so magnitude +
  stability read in one view (mirrors the Boruta importance plot). Save via `figure-io`.

**What the source oracle (`src/classification.py`) gives — and its gaps (scanned 2026-07-01).**
The pipeline scaffold (in-fold `StandardScaler` + saga LR), the neighborhood-smoothed grid
selection, and the ROC-±SD figure are the reusable bones. But it is a **performance harness, not
a selection harness** (it never extracts coefficients), and as scanned it:
- **Almost certainly isn't elastic net** — `_make_classifier_pipeline` never sets
  `penalty="elasticnet"`, so `LogisticRegression` stays at the default `penalty="l2"` and
  `l1_ratio` is silently ignored → ridge, *no sparsity*, and the `l1_ratio` grid is a no-op.
  **Verify + fix** (this is also the poster child for why the `lib/` test bar exists).
- **Tunes on the full data, then CVs on the same data** → optimistic bias (needs nested CV).
- **Folds aren't group-aware**; **no label-shuffle null**.
- **Silent `NaN → 0`** (we refuse — missingness is the Stage-2 decision), **seaborn** (we don't
  ship it), **pickle cache** (`S301` / file-format-convention — leave caching to the project
  script or JSON), study-specific label / mask helpers + hardcoded `random_state` / `n_jobs`
  (strip → consume a `Dataset`, outcome via a `contrast`-style metadata API like §A.0b).

**Next step before building:** a **real-data 5xFAD preview** (the established pre-code pattern) —
elastic-net classification of a real contrast, showing the ROC-vs-null curve and the
coefficient / selection-frequency readout — for the user to eyeball before it's built to the
`lib/` bar.

### B.2 Elastic-net linear regression (continuous outcome) — *no source oracle*
- Same machinery for a continuous target (`ElasticNetCV` / a `Pipeline`). Report selected
  features + stability + held-out R²/RMSE **vs a shuffle null**. **Overlaps roadmap #4
  (regression)** — same placement question as B.1.

### B.3 Boruta — ✅ SHIPPED (v0.1, 2026-07-06) — *source oracle* (`manuscript-trex-phase2a/te-phase2a-pelt/src/`)

**✅ Shipped.** `lib/analysis/boruta` (`boruta_select(dataset, target=, *, task=…)`) +
`lib/figures/boruta_importance` (`plot_boruta_importance` / `save_boruta_importance`),
strict-clean + 34 tests, validated on the real 5xFAD binary genotype contrast (18
Confirmed / 7 Tentative; the Confirmed set ⊇ the classifier's validated top coefficients).
Built on **BorutaPy 0.4.3** + the private-method shadow-history subclass (open decision #7,
option (a)), isolated fail-loud. Task inferred from the target dtype (numeric → regression;
categorical → **multiclass-native** classification) with a low-cardinality-numeric warning
+ a `task=` override; experimental subset in, **raises** on NaN, **warns** on non-log,
drops constants, missing-target samples dropped. Runs **once on the whole experimental
matrix** — the shadows are the built-in null, so no external null and no CV. Evidence is a
**decision + importance + shadow-null** (no per-feature q — the settled selection kind, #8).
**Ships no cache** (project-local — the file-format convention + `S301`; the oracle's pickle
does not ship). No `generalization_target` field (Boruta makes a selection, not a
performance claim). The design record below is what was built.

The user's proven implementation, read in full: `feature_finding_boruta.py` (core),
`boruta_plotting.py` (the informative viz), `scripts/run_feature_finding_boruta.py` (driver).

**Method.** All-relevant selection: real features compete against **shadow (permuted-copy)
features** over a Random Forest; each feature is **Confirmed / Tentative / Rejected** by
whether its importance beats the best shadow across iterations (binomial test, internal
Bonferroni). Gives the *all-relevant* set — every feature carrying signal — not elastic net's
*minimal-optimal* sparse set; the two **complement** each other.

**What the oracle gives us:**
- `run_boruta(X, y, target_name, feature_names, task=…)` over **`boruta.BorutaPy`** with an
  `sklearn` RF: `task="regression"` → `RandomForestRegressor`; `"classification"` →
  `RandomForestClassifier(class_weight="balanced")` (imbalance handled). **One target + task
  per call** — the clean study-agnostic API (the driver loops three: continuous dose, binary
  dose-rate, continuous time). Defaults `n_estimators="auto"`, `max_iter=100`, `alpha=0.05`,
  `perc=100`, `random_state=42`; `X` is **log2**, `NaN → 0`.
- A `_BorutaPyWithShadowHistory` subclass intercepting the **private** `_add_shadows_get_imps`
  to keep the per-iteration shadow threshold (`percentile(imp_sha, perc)` →
  `sha_max_history_`) that upstream BorutaPy throws away — needed for the plot's reference
  line. **⚠ fragile** (private API of a lightly-maintained package; see the dep note).
- `BorutaFeatureFindingResult`: `decision` (C/T/R), `ranking`, `importance` (`nanmedian` over
  the per-iteration `importance_history`, shape `(n_iter, n_features)`, NaN once a feature is
  decided), `shadow_max_history`, and the C/T/R counts.
- Results table (`build_*_results_dataframe`): `feature, decision, ranking, importance`,
  sorted ranking↑ then importance↓. **No p / q / CI** — the evidence is a *decision +
  importance*, not a significance statistic (→ evidence-shape decision #8).
- `save/load_boruta_result` cache via **pickle** (Boruta is minutes-slow, so caching pays) —
  but pickle conflicts with our file-format convention + is an `S301` smell; re-decide the
  cache, or leave caching to the project script.

**The informative visualization** (`plot_boruta_importance`) — the one to reproduce: a
**horizontal box-plot of per-iteration importance** for the top-N ranked features (box +
jittered scatter), **accepted features colored by median importance on viridis** (continuous
gradient + colorbar), **rejected gray**, and a **red dashed vertical line at the median shadow
threshold** — the visual "did this feature clear the noise floor?". Title carries the C/T/R
counts; UniProt `sp|ACC|GENE_SPECIES` labels collapse to the gene symbol. Port onto the figure
foundation: viridis importance is **continuous** (no categorical budget), save via `figure-io`
(dual-export + separate legend), keep the shadow line + counts.

**Canonical wiring (from the driver):** load → median-normalize → **ComBat batch-correct** →
filter experimental + complete-covariate → Boruta per target. Feeding Boruta the
**batch-corrected** matrix is a *legitimate ComBat-output use* (selection/ML, not a
per-feature significance test — so the Nygaard variance-deflation caveat doesn't bite here),
in contrast to the DE template's batch-as-covariate. The per-study covariate / focus-area
filtering is project-local — generalize away like the OLS design builders.

**Dependency note** (answers Open Decision #7): the proven path uses **`BorutaPy` + the
private-method subclass**. BorutaPy is lightly maintained (sklearn / numpy-deprecation
friction) and ships **no type hints**, so `mypy --strict` needs an `ignore_missing_imports`
override (and our engine venv is deliberately stub-free). Weigh: **(a)** take BorutaPy + the
subclass (matches the user's working code, fastest); **(b)** reimplement the shadow-importance
loop ourselves (full control + strict-clean, but real statistics to validate); **(c)** a
maintained fork. Lean (a) unless the strict-bar friction proves bad.

---

## C. Shared downstream outputs (the user's "downstream is similar")

The univariate family converges on three artifacts, identical across `method=`:

- **Results table** — the one artifact every univariate method emits: `feature`,
  `effect (+ CI)`, `p`, `BH-q`, `mean_abundance`, `n`; sorted by `q`. Feeds the findings
  graph **and** both plots. (Generalizes the source's
  `build_protein_results_dataframe` / `build_precursor_results_dataframe`, **plus the CI
  columns the source lacks**.)
- **Volcano plot** (`lib/figures/volcano`) — generalize `src/volcano_plotting.py` onto the
  figure foundation: route color through `okabe-ito-colors`, save via `figure-io`
  (dual-export + separate legend). Effect on x, `−log10(BH-q)` on y, three-way significance
  coloring (NS / sig-up / sig-down), **hit counts in the legend**, p-underflow floored to
  `finfo.tiny`. (Already listed in `QC_GAPS.md`.)
- **p-value histogram** (`lib/figures/pvalue-hist`) — **fresh design, no oracle**; the
  calibration diagnostic (uniform + spike-at-0 = healthy; U-shape / right-hump = unmodeled
  structure / confounding; hill-at-1 = conservative). Build **alongside** the DE template;
  overlay contrasts via registry colors; optional π0 (Storey). (Already specced in
  `QC_GAPS.md` Tier-1.)

For the multivariate methods, the shared outputs are instead **selection-frequency /
coefficient-path** plots and the **Boruta decision plots** (§B.3) — designed with the user's
examples.

---

## Open design decisions (carry these across conversations)

1. **Design / contrast API — ✅ SETTLED (2026-06-29): constrained, v0.1.** `contrast=` +
   `covariates=[…]` over metadata columns, dummy-encoded internally via the lifted encoders;
   patsy formula / interaction models deferred to v0.2 (interactions are a project-local
   adaptation of the seed, as the source's per-study interaction builders already are). Full
   contract — reference levels, categorical/continuous typing, multi-level handling, return
   shape — is in **§A.0b**.
2. **Univariate family shape — ✅ SETTLED + SHIPPED (2026-06-29): one family.** One template,
   one `differential_abundance(...)` entry, swappable `method=` (`ols`/`moderated`/`welch`/
   `mannwhitney`) sharing the `effect+CI+p+BH-q` table, the volcano, and the p-value histogram.
   `welch`/`mannwhitney` reject `covariates=` (they cannot adjust) and a continuous contrast
   (two-group only); a `k>2` factor becomes `k-1` pairwise terms for those two and `k-1`
   treatment-contrast coefficients for `ols`/`moderated`. The full table also exposes the
   covariate terms (*report all tests run*); `contrast_table` is the contrast-only deliverable.
3. **Confidence intervals — ✅ SETTLED + SHIPPED (2026-06-29).** Added for every method: `ols`
   `coef ± t·se`; `moderated` the same at the inflated (`d0+df`) df (norm in the `d0→∞` limit);
   `welch` mean-diff ± `t·se` at the Welch–Satterthwaite df; `mannwhitney` the distribution-free
   Hodges–Lehmann rank CI on the pairwise differences. `effect_label` stays honest about scale
   (log2 fold change on log2/glog2; warns + a "difference" label off-log) and the estimator.
4. **Missing values — ✅ SETTLED (2026-06-29): an upstream Stage-2 collaborative decision; the
   test consumes the resolved matrix.** The handling menu is **impute (mean / median / KNN) /
   drop (by per-feature missingness threshold) / set-to-0 / a combination** (the typical
   recipe: drop features missing in > X % of samples, then zero or impute the rest), driven by
   the shipped **missingness QC** (completeness curve + MNAR). It is decided **collaboratively
   with the scientist in Stage 2**, recorded like the normalization choice (`provenance.params`
   + the workflow preprocessing record), and **applied upstream** so feature finding receives an
   already-resolved matrix. The test template **never silently imputes / zeros.** Enforcement is
   a **Stage-4 command precondition + orchestrator checkpoint** — if the decision hasn't been
   made, *stop and make it with the scientist* (surfacing the missingness picture), **not** a
   cold runtime refuse; a template-level NaN validation error stays only as a defense-in-depth
   backstop. **✅ SHIPPED (2026-06-29):** missing-value handling is now the parameterized
   `lib/common/missing-values` template (v0.1, `Dataset → Dataset`, like `normalize`) —
   `handle_missing(max_missing_fraction=…, impute="zero"/"mean"/"median"/"knn")`, sharing the
   `missingness`/`id-depth` detection predicate, hard-refusing non-linear scale, recording
   drop/impute params for provenance (36 tests, strict-clean). This **reverses** the earlier
   "no template — by decision" call. Wired into Stage 2 (`commands/stage2-data.md`). Left-
   censored imputers (MinProb/QRILC) for strong-MNAR data deferred to v0.2 (project-local for
   now).
5. **Both-ways robustness — ✅ SETTLED (2026-06-29): orchestration, not a template feature.**
   The template runs **once per call** and stays agnostic to batch handling — batch is just a
   nuisance `covariates=` entry for the uncorrected run (the variance-deflation-safe path for
   testing). The Stage-4 command runs it twice (uncorrected-with-batch-covariate vs on the
   ComBat-corrected matrix) and overlays the two p-value histograms (`pvalue-hist` takes the
   `{label: p}` mapping). Keeping it orchestration keeps the template a clean single test and
   lets the scientist see both calibrations side by side. Wired into `commands/stage4-explore.md`.
6. **Multivariate placement — ✅ SETTLED for logistic (2026-07-01).** Elastic-net logistic is
   **the classifier template (roadmap #3), pulled forward**, with elastic net as its default
   estimator — *not* a standalone selector (it is framed as **classification**, §B.1). The
   continuous analogue (elastic-net linear, §B.2) is the same call deferred to roadmap #4.
7. **Boruta engine — ✅ SETTLED (2026-07-06): option (a), `BorutaPy 0.4.3` + the private-method
   subclass.** The historical breakage worry is moot — 0.4.3 runs cleanly on the current
   numpy 2.4 / sklearn 1.9 stack and still exposes the private `_add_shadows_get_imps` hook.
   The only friction is it ships no `py.typed`; the strict mypy config's `ignore_missing_imports`
   treats it as `Any`, so only the one **subclass-of-Any** needs a scoped `# type: ignore[misc]`
   (the import needs none). The private-method reliance is isolated **fail-loud**
   (`BorutaShadowHistoryError` if the captured shadow history stops aligning with
   `importance_history_`), and the dep is **pinned** (`boruta==0.4.3`). **Cache — no pickle in
   the template:** caching a minutes-slow result is a **project-local** concern (the file-format
   convention reserves pickle for non-LLM consumers; a pickle of an arbitrary result is an
   `S301` smell), so the oracle's `save/load_boruta_result` do **not** ship.
8. **Selection-method evidence shape — ✅ SETTLED for classification (2026-07-01); Boruta to
   confirm reuse.** Significance methods (§A) emit effect + CI + p + BH-q; the **classification
   / selection** kind emits a different shape. For elastic-net classification (§B.1): run-level
   **{balanced-acc / AUC ± SD, shuffle-null distribution + empirical p}** + per-feature
   **{all-data signed standardized coef, selection frequency, sign consistency, coef
   median/IQR}** — **no per-feature q**. **✅ CONFIRMED for Boruta (2026-07-06):** Boruta (§B.3)
   reuses this same "classification / selection" evidence kind — its per-feature **decision
   (Confirmed / Tentative / Rejected) + importance** and the run-level **shadow-null +
   C/T/R counts** slot into the run-level + per-feature split, **no per-feature q**. The kind
   is in `conventions/findings.md` §2.3 and accepted by the stats-reviewer
   (`conventions/statistics.md`).

---

## Scope boundaries (deliberately *not* general templates)

- **Per-study design-matrix builders stay project-local.** A user's metadata schema is
  unknowable a priori (same rationale as control detection / caveat findings). The template
  ships the **generic encoders + the contrast/covariate API**, never named builders.
- **Enrichment / pathway / GSEA** is downstream of feature finding and a large scope of its
  own (gene-set DBs, ID mapping, within-set multiple testing) — out of scope here (also noted
  in `QC_GAPS.md`).
- **Full predictive modelling** (ROC, calibration, held-out performance *as the deliverable*)
  is roadmap #3/#4. This doc covers those models' **feature-selection** use only — keep the
  line explicit.
