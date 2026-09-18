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
| **XGBoost** (gradient-boosted-tree) **classification** | multivariate / classification | `te-phase2a-pelt/src/classification_xgboost.py` + `classification_xgboost_plotting.py` | ✅ **Shipped** (v0.1, 2026-07-06) | `lib/analysis/classification-xgboost` (`classify_xgboost`) + `lib/figures/classification_xgboost` |
| **Linear SVM** (soft-margin `SVC(kernel="linear")`, `C` tuned; hard margin = the large-`C` limit) **classification** | multivariate / classification | — (fresh design; `lib/analysis/classification` is the shape oracle — no SVM in the source project) | ✅ **Shipped** (v0.1, 2026-09-14; **v0.3** — within-unit null) | `lib/analysis/classification-svm` (`classify_svm`) + `lib/figures/classification_svm` |
| **Shrinkage LDA** (Ledoit-Wolf shrinkage linear discriminant, dual/Woodbury form; tuning-free) **classification** | multivariate / classification | — (fresh design; the sklearn `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` recipe is the numerics oracle, pinned at machine precision; `classification-svm` is the shape oracle) | ✅ **Shipped** (v0.1, 2026-09-17) | `lib/analysis/classification-lda` (`classify_lda`) + `lib/figures/classification_lda` (three figures) |
| Elastic-net linear regression | multivariate | `te-phase2a-pelt/src/regression.py` + `regression_plotting.py` | ✅ **Shipped** (v0.1, 2026-07-06) | `lib/analysis/regression` (`regress`) + `lib/figures/regression` |
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

**✅ Regression shipped (v0.1, 2026-07-06).** The leakage-safe **regression** template
(roadmap #4) — elastic-net **linear** regression (§B.2, the continuous analogue of the
classifier) — `lib/analysis/regression` (`regress`) + the four figures
`lib/figures/regression`, built to the **same shape** as the classifier (the user's steer:
nested CV [R²/RMSE/MAE], in-fold scaling, group-aware-when-repeats, all-data coefficients +
stability loop, opt-in **target**-shuffle null). Oracles `regression.py` +
`regression_plotting.py` gave the ElasticNet bones + scatter/heatmap; the oracle's
leakage/null/stability/grouping gaps, `NaN→0`, seaborn, pickle cache, and dose-specific
clipping were all fixed/stripped. New leakage-safe **`feature_list=`** prior-restriction
param (the user's steer). Preview-first + real-data smoke on **trex time-since-exposure**
(dose is null even in the manuscript, R²=−0.018; time is the strong signal — manuscript
R²=0.90, honest nested-CV R²=0.77, clears the null p=0.005). 22 tests; no new shipping dep.
**With this the whole analysis layer (univariate DE + classification + Boruta + regression)
ships.**

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

**➕ Non-linear sibling — XGBoost classification — ✅ SHIPPED (v0.1, 2026-07-06).** A **parallel
template** `lib/analysis/classification-xgboost` (`classify_xgboost`) + `lib/figures/classification_xgboost`,
built as a **self-contained sibling** of the elastic-net classifier (the `regression`-precedent
pattern — own `XGBClassificationResult`, own figure module, the label/CV/null/stability
scaffolding duplicated so the seed stands alone) for when the class boundary is **non-linear or
interaction-driven**. Same leakage-safe shape (nested CV tuning `max_depth × learning_rate`
in-fold, opt-in label-shuffle null gating exploratory↔validated, fixed-hyperparameter stability
loop). **Four deliberate divergences from the linear classifier**, all documented in-code: (1)
**no standardization and no scale warning** — trees split on per-feature thresholds, so they are
scale-invariant; (2) imbalance via **fold-local `scale_pos_weight = n_neg/n_pos`** (not
`class_weight`); (3) **unsigned gain importances** (a magnitude, no direction → **no sign
consistency**, and the importance figure has no zero line / starts at 0 — the one figure that
genuinely diverges, closer to the Boruta importance plot); (4) a **2-D tuning grid** (other tree
knobs held-constant scalars) to keep nested CV tractable *and* reuse the 2-D heatmap. Sourced from
`te-phase2a-pelt/src/classification_xgboost.py` (+ its plotting) but **stripped** of the source's
GPU detection / device resolution / file-lock / joblib scaffolding / **7-D corner plot** / pickle
cache — none of which matched our established classifier design. **NaN still raises** (native NaN
routing noted as a deliberate non-use for auditability). New pinned shipping dep `xgboost==3.3.0`
(ships `py.typed`, so `mypy --strict` clean, no override). 23 tests (planted-truth invariants +
determinism, guards incl. the **no-scale-warning** divergence, grouping, four figures, a real-5xFAD
smoke). **Validated on the 5xFAD genotype contrast:** nested-CV AUC **0.904 ± 0.072**, observed
0.898 vs shuffle-null mean 0.511 (**p = 0.0099**), and the top gain importances are the canonical
AD proteins (APOE, APP/A4, complement C1q, GPC1) — matching the elastic-net classifier's validated
hits, with the correlated APP neighbour APLP2 surfacing lower down. The 5xFAD preview lives in
`testdata/5xFAD/_classification_xgboost_preview/`. Evidence reuses the classification/selection kind
(§8), with the per-feature estimate an **unsigned importance** rather than a signed coefficient
(`conventions/findings.md` amended). Wired into `lib/manifest.md`, `conventions/{statistics,
visualization,findings}.md`, and `commands/stage4-explore.md` (offered *alongside* the linear model).

**➕ Second linear sibling — linear-SVM classification — ✅ SHIPPED (v0.1, 2026-09-14).** A
**parallel template** `lib/analysis/classification-svm` (`classify_svm`) + `lib/figures/classification_svm`,
built as a **self-contained sibling** of the elastic-net classifier (the XGBoost precedent — own
`SVMClassificationResult`, own figure module, the label/CV/null/stability scaffolding duplicated so the
seed stands alone) because a max-margin linear classifier is inherently interpretable (signed weights,
support vectors) and suits separable small-*n*/high-*p* proteomics. The user's protocol, verbatim in
the code: `Pipeline(StandardScaler → SVC(kernel="linear", class_weight="balanced"))` so scaling is
learned only in the training fold; `RepeatedStratifiedKFold` outer / `StratifiedKFold` + `GridSearchCV`
inner on **ROC AUC**; the outer test fold scored only by **`decision_function`** margins (no
probability calibration — `roc_auc_scorer` resolves to `decision_function` for an SVC pipeline, verified
in the installed sklearn); per-fold AUC summarized across repeats, plus a **repeat-level pooled-OOF AUC**;
and **identical outer splits across classifier types** so comparisons are **paired**. Four user decisions
(2026-09-14): **soft-margin with `C` tuned** (a true hard-margin SVM has no slack — it is the
**large-`C` limit** of the grid; `c_grid=(1e6,)` pins it; the inner search still runs over the one value); the stability read is the
**top-k membership frequency** (fraction of stability resamples in which the feature ranks in the top *k*
by |weight|; `top_k=20`) because every weight of a dense model is non-zero and *selection frequency* is
meaningless; **fold identity recorded + verified** — every classifier fold record now carries
`repeat`/`fold`/`test_indices` (`classification` and `classification-xgboost` bumped to **v0.2** with
back-compatible defaults; `result-io` **0.2** now applies a dataclass default for a field missing from an
older cache instead of raising, so pre-0.2 cached results still load — +2 tests) and
`lib/tests/test_classification_splits.py` asserts the three templates' `_make_cv`/`_split` copies and
their public entry points produce identical `(repeat, fold, test_indices)` sequences for one seed;
real data re-copied from the source project into git-ignored `testdata/5xFAD/`. **Five deliberate
divergences from the elastic-net classifier**, documented in-code: (1) **dense, not sparse** — top-k
frequency, the table holds *every* feature, ridge-like weight sharing across correlated features;
(2) **scores are margins** — `y_score` not `y_prob`, balanced accuracy at margin 0, no
`probability=True`; (3) **a 1-D `C` grid → a tuning curve, not a heatmap**, with a **tolerance-aware
smallest-`C` tie-break** (separable data plateau above the hard-margin threshold, so ties are the normal
case) applied by **one rule in both the outer loop (`GridSearchCV(refit=False)` + `_select_c`) and the
all-data fit** — the elastic net lets GridSearchCV's own argmax pick inside the outer loop and applies
`select` only to the all-data fit, a latent inconsistency not inherited; (4) **per-repeat AUCs** — the
mean of a repeat's fold AUCs (primary; averages back to `cv_auc`) and the pooled-OOF AUC (supplementary,
with the caveat that fold models share no score scale, after Forman & Scholz 2010); plus the per-fold
tuned `C` on each fold record (a tuning-stability read); (5) **support-vector counts** (total + per
class). `SVC` needs no `random_state` (libsvm's SMO is deterministic; the seed only feeds unused Platt
scaling) — a determinism test pins it. **The C-grid finding (from the preview):** on the 8,828-protein
5xFAD matrix the inner-CV curve was **flat across the whole first-draft grid (1e-3 … 100)** — the
kernel scale of standardized data is ~p, so the hard-margin plateau begins around `C ≈ 1e-3` here and
the knee scales roughly with **1/n_features**; a low-C probe showed the soft-margin regime at
1e-5 … 1e-3 (AUC 0.70 → 0.89, monotone into the plateau; 47/52 samples are support vectors on it). The
default grid now reaches **1e-5 … 100** (14 half-decades) and a new **`CGridEdgeWarning`** fires when
the selected `C` is a grid edge (the grid did not bracket the optimum — extend it, don't report the
edge as "tuned"); the C-curve figure shows the knee. **No new shipping dep** (scikit-learn). **38
tests** (planted-truth incl. the dense-table/top-k invariants, determinism, margins-not-probabilities,
fold identity + per-repeat AUCs, single-C hard-margin pin, tie-break, hand-computed top-k semantics,
grid-edge warning, guards incl. class < n_splits and bad grids, grouping incl. the grouped null path,
four figures, feature-list restriction + title note, a real-5xFAD smoke) + 2 splits + 2 result-io.
**Validated on the 5xFAD genotype contrast (defaults, seed 0):** nested-CV AUC **0.796 ± 0.174**
(per-repeat 0.74–0.89), all-data `C = 1e-3` (the plateau start), 47 support vectors, observed 0.802 vs
shuffle-null mean 0.492 (**p = 0.005**, 200 permutations), and the top weights are the canonical AD
proteins — **APOE**, TICN1, **APP/A4 + the 5xFAD human APP transgene**, midkine, clusterin, TICN2,
cystatin F, GPC1 (top-20 frequency ≥ 0.86, sign consistency 1.0), with PLGT2 (+) and LAG3 (−) beside
them — **10 of the elastic net's top 15**. **The paired comparison is the honest headline:** on the *same 25 outer folds* the
elastic net (reduced grid) scores **0.905 ± 0.139** and beats the SVM on **17 / 25 folds** (tied 4,
SVM better 4; mean paired difference −0.11, Wilcoxon p = 0.002) — on this contrast the sparse model
wins, which is exactly the kind of statement the shared splits exist to license. Two more preview
reads (from the same protocol run on the source project's *current*, post-FASTA-update 8,809-protein
snapshot, archived under the preview's `newdata/`): with the first-draft grid floor at 1e-3 the SVM
scored 0.820 ± 0.156 vs 0.772 ± 0.176 on the wide grid, so **widening the grid downward cost ~0.05
AUC** — 4 of 25 outer folds tuned to `C ≤ 1e-4` (fold AUC 0.39–0.75) because ~33-sample inner folds
are noisy over 14 grid points; `select="smoothed"` (plateau-seeking) recovers part of it (0.786) and
is the setting to prefer when the curve has a long plateau. Preview under
`testdata/5xFAD/_classification_svm_preview/` (`summary.md` + `paired_fold_auc.csv` on the
documented 8,829-protein testdata snapshot; `newdata/` the current-snapshot runs incl. the
first-grid `summary.grid1e-3.md`). Evidence reuses the classification/selection kind (§8) with the per-feature
estimate a signed weight read through top-k frequency (`conventions/findings.md` amended). Wired into
`lib/manifest.md`, `conventions/{statistics,visualization,findings,results-cache,enforcement-map}.md`,
`commands/stage4-explore.md` (offered *beside* the elastic net; compare paired), `agents/{statistician,
figure-generator}.md`, `templates/{project-CLAUDE,finding}.md`, `lib/README.md`.

**v0.2 (2026-09-14) — the tuning-stability diagnostic, from a second real-data check.** On a second
dataset (MagNet-EV plasma proteomics, ADD vs HCN+PDCN+PDD, **40 samples x 2,343 proteins**, git-ignored
`testdata/MagNet-EV-ADD/` with `compare_classifiers.py`) the verdict **reversed**: SVM nested-CV AUC
**0.963 ± 0.067** vs elastic net **0.870 ± 0.180**, the SVM better on 11 / 25 shared folds and never
worse (Wilcoxon p = 0.003), both clearing the null (p = 0.002 / 0.006). The user's own nested-CV SVM
script reported 0.977 on the same files; reproducing it verbatim and ablating one difference at a time
showed the gap is **the C-grid floor** (their grid's smallest value, 2^-10 ≈ 1e-3, sits exactly at this
dataset's plateau start, so it cannot under-tune; the template's 1e-5 floor lets ~30-sample inner folds,
whose AUC is coarse and ties in most folds, pick a sub-plateau `C` in a share of outer folds: 0.977 →
0.950 from the grid alone) plus **seed noise** (their protocol at seed 0 x 5 repeats: 0.950; the
template at seed 1: elastic net 0.927, SVM 0.970). `class_weight` and the inner fold count are neutral.
A largest-C (hard-margin) tie-break was tried and is **not** a consistent fix (+0.02 at one seed, −0.01
at another), so the smallest-C rule and the 1e-5 floor stand (user decisions, 2026-09-14, with `top_k=20`
and the sibling v0.2 edits). What shipped instead is a **diagnostic**: the result records the all-data
**plateau start** `C` (`plateau_start_c`, the unsmoothed smallest-C-within-tolerance) and
**`n_folds_sub_plateau`**, the outer folds that tuned strictly below it; a **`TuningNoiseWarning`** fires
past 25 % of folds (5xFAD: 11 / 25 → warns; MagNet: 4 / 25 → quiet), and the C-curve figure draws the
**per-fold picks as stacked ticks** with a dotted plateau-start line. Both fields are defaulted, so v0.1
caches reload with `None`. +2 tests. Also recorded for any finding on MagNet: the **cohort confound**
(all ADD are Kerchner, all PDCN are Poston) and the coarse per-fold AUC (two positives per test fold).
A four-seed sweep settles the seed question: nested AUC over seeds 1–4 averages **elastic net 0.933**
(0.917–0.950) vs **SVM 0.967** (0.960–0.973) — seed 0's elastic-net 0.870 was an unlucky draw, the
user's 0.92 sits at the multi-seed mean, and the paired SVM advantage holds at every seed.
**Tie-break bake-off (2026-09-14) — the `tie_break` option is *not* added.** The open question from v0.2
(smallest-`C` vs largest-`C` on a tied plateau) was measured **paired on the same inner searches**: per
outer fold the inner grid search ran once, then both rules refit and scored the same held-out fold
(default grid, 5×5 outer CV, `select` ∈ {best, smoothed}; MagNet-EV-ADD seeds 0–4, 5xFAD seeds 0–2 —
`testdata/MagNet-EV-ADD/tiebreak_bakeoff.csv`, `testdata/5xFAD/_classification_svm_preview/tiebreak_bakeoff.{py,csv}`).
Result: the held-out AUC is **identical in 90–99 % of folds** (MagNet, best: 111/125 ties, largest better
8 / worse 6, Wilcoxon p = 0.74; smoothed: 119/125 ties, p = 0.78; 5xFAD, best: 71/75 ties, 1 / 3;
smoothed: 74/75 ties), mean AUC differs by ≤ 0.003, and the per-seed swings (−0.013 … +0.020)
reproduce the v0.2 anecdote — it was seed noise. Mechanism: the two rules pick a different `C` in most
folds (92/125, 43/75) but on a plateau the solutions rank identically; where they differ the largest
rule is the **grid top** in 85/92 and 43/43 folds, i.e. it is not a tie-break but "hard margin at
whatever `C` the grid tops out at", which `c_grid=(1e6,)` already pins. Adding the option would also
have broken two shipped pieces: `_warn_if_grid_edge`'s upper-edge message ("still rising") is false on
every plateau (the all-data largest pick hits the top edge in 6 of 8 seeds), and `n_folds_sub_plateau`
drops 34 → 12 (MagNet) without the tuning noise changing, miscalibrating the 25 % `TuningNoiseWarning`.
**Decision (user, 2026-09-14): no `tie_break` option; `smallest` stands** (`conventions/statistics.md`
records it in one sentence). What would reopen it: a paired multi-seed advantage consistent in sign
across ≥ 2 datasets that exceeds seed noise (|Δ| > ~0.03, Wilcoxon p < 0.05 over ≥ 100 paired folds) —
the bake-off script is that harness.
**Independent review (Fable subagent, 2026-09-14) → all findings fixed before release:** (B1) the
class-size guard was one fold too lenient — the inner CV runs on the outer *training* fold, so a
minority class of 5–6 at `n_splits=5` passed the guard and crashed deep inside with an all-NaN
tuning surface; now `min_class − ceil(min_class/n_splits) >= n_splits` with a message that says so, in
**all three** classifier templates (the elastic net had no such guard and silently refit on a NaN
grid); (B2) `select="smoothed"` could pick a `C` whose own inner fit failed (NaN smoothed into a
neighbour mean) — non-finite raw scores are never candidates now; (R1) the edge and noise warnings
could both fire when the curve was still rising at the top of the grid — the noise warning is
suppressed at the upper edge; (R2) the group-level null silently *rounded* a unit carrying both
labels (changing the permuted class balance) — a mixed unit now raises, up-front when `run_null` is
grouped, in all three templates; (R3) the fixed-`C` null is now stated in the result docstring and
every classifier ROC annotates the p with the fixed-hyperparameter observed AUC it belongs to (not
the nested AUC beside it); (R4) the default-aware loader now emits a `ResultSchemaWarning` naming the
defaulted fields, so a manifest missing the feature-list counts can no longer silently drop the
mandatory title caveat (`conventions/results-cache.md`); (R5) exact ties at rank *k* are all members
of the top-k (a duplicated protein row no longer gets an arbitrary 1.0 vs 0.0 split). Nits: `select`
is validated, the fold-pick ticks are height-capped and the legend uses `loc="best"`, the
`plateau_start_c`-under-smoothing semantics and the serial grouped-null path are documented, and the
"null ≈3.5× the run" claim in four docs now reads 3.5× (elastic net) / ≈9× (SVM). +7 tests (45 in
the SVM file: the nested class-size boundary, failed-fit selection, upper-edge-only warning, `select`
validation, a hand-built sub-plateau count under smoothing, tie membership, and grouped-fold
integrity + the mixed-unit refusal).
**Second Fable round (same day) — nothing blocking; all nine fixes verified; six risks + five doc
inaccuracies, all addressed:** the edge suppression now keys on the unsmoothed plateau start (a helper,
`_should_warn_tuning_noise`, unit-tested in place of a seed-pinned integration test); a single-class
outer test fold now **raises** in all three classifier templates instead of yielding a silent
`cv_auc = nan`; the loader's `ResultSchemaWarning` fires **once** per reload with dotted paths
(`fold_predictions[].repeat`), not once per fold record; the nested class-size guard documents its
leniency under grouping (still fail-loud); the cost figures use one convention (null ≈3.5× an
elastic-net run, ≈9× an SVM run; the stability loop ≈1% / ≈3%); "skips tuning" corrected (the
inner search still runs over a single-element grid). **User decision (at the time):** the group-level null's
one-label-per-unit requirement was **documented as a limitation** — a batch-grouped design
(`generalization_target="batches"`) had no label-shuffle null and stayed `exploratory`; a within-unit
permutation option was deferred (46 tests in the SVM file) — and then built the same day, next.

**Within-unit null — ✅ BUILT (v0.3 of all three classifier templates, 2026-09-14).** The deferred
option above shipped as **`null_permutation: "units" | "within_units"`** on `classify` /
`classify_xgboost` / `classify_svm`. `"units"` (default) is the prior behaviour — one label per unit, a
mixed unit refused (the message now names the remedy). `"within_units"` shuffles labels **inside each
unit, preserving every unit's class counts** — the restricted permutation for exchangeable blocks
(Anderson & ter Braak 2003; Winkler et al. 2015), testing H0: label ⊥ proteome | batch, i.e. *does the
proteome predict the label beyond batch* — the batch-held-out question. Two facts settled it: **(a)**
`StratifiedGroupKFold` assigns units to folds from their class-count vectors, so under within-unit
permutation the grouped folds are **byte-identical** to the observed run (verified 50/50; a unit-level
permutation moves them 50/50) — every null draw is scored on the observed folds and only the labels
move (pinned by `test_within_unit_permutation_keeps_grouped_folds_fixed`); **(b)** on real 5xFAD
(`groups="Cohort"` — two cohorts, each holding both genotypes — cohort-held-out 2-fold, fixed
`C = 1e-3`) the unit scheme refuses while the within-cohort null gives observed AUC **0.843** vs null
**0.500 ± 0.098** (max 0.751), **p = 0.005** (200 permutations, 6 s; the slow smoke
`test_smoke_5xfad_cohort_held_out_within_unit_null` pins it at 100). Guards, all up-front before the CV:
`within_units` needs grouped CV (`groups=None` or the singleton fallback → raise, naming the column);
every unit single-class → raise (the shuffle would be the identity and p = 1 silently); fewer distinct
arrangements (`∏ C(n_u, k_u)`, an exact Python int — an int64 product overflows on six 20-sample
batches) than `n_permutations` → `NullPermutationWarning` (p resolves only to ~1/count). The scheme
actually applied is recorded on the result (`null_permutation`: `samples` / `units` / `within_units`;
`None` when no null, and on a pre-0.3 cache via the loader's `ResultSchemaWarning`), named in
`plot_null`'s title (figure modules bumped), and enters the cache fingerprint. The CV blocks are
untouched (the identical-splits invariant holds); `test_classification_splits.py` now also pins the
null helpers **byte-identical** across the three templates (`inspect.getsource`) and identical draws
under both schemes. Assumption, documented in-code: samples within a unit are exchangeable — a nested
subject-within-batch design needs a multi-level block permutation (not offered). +5 tests per template
file (+1 mirrored mixed-unit refusal in the elastic-net and XGBoost files, which had none), +3 splits,
+1 result-io, +1 slow smoke. Wired into `conventions/{statistics,results-cache}.md`,
`commands/stage4-explore.md` (the generalization-target answer now also fixes the scheme),
`agents/{statistician,stats-reviewer}.md`, `skills/statistical-analysis/SKILL.md`, `lib/manifest.md`.

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

**➕ Third linear sibling — shrinkage-LDA classification — ✅ SHIPPED (v0.1, 2026-09-17).** A
**parallel template** `lib/analysis/classification-lda` (`classify_lda`) +
`lib/figures/classification_lda`, built as a **self-contained sibling** of the elastic-net and SVM
classifiers (own `LDAClassificationResult`, own figure module, the label/CV/null/stability
scaffolding copied byte-identically so the seed stands alone and `test_classification_splits.py` now
pins **four** templates to identical outer splits + byte-identical null helpers). **Why:** a
*tuning-free* dense linear model (Ledoit-Wolf sets the shrinkage analytically — no grid, so none of
the SVM's grid-bracketing / tuning-noise failure modes), **calibrated log-odds** scores, a
**covariance-adjusted** reading (the direction is `S⁻¹Δμ` — which proteins separate the classes
after accounting for co-variation), and a null cheap enough to run in the same sitting. **The build
was gated on scale:** scikit-learn's own solver forms the p×p shrunk covariance (26 s per fit at p =
2,000; unfinished after 10 min at p = 6,186; 3.2 GB per fit at the user's routine 20,000) — the
template solves it in the **dual (Woodbury) form** (`diag(a) + UᵀU`, an n×n solve; the Ledoit-Wolf
intensity from the Gram matrix), the one place the engine ships **in-module numerics** rather than a
wrapper, justified by an identity the test suite pins to sklearn at ~1e-10 (shrinkage intensity,
coefficients, intercepts, scores, probabilities; binary + the k-class math kept for v0.2). Two
sklearn subtleties reproduced exactly: the per-class target is `μ·diag(scale²)` with μ the mean
*standardized* variance (exactly 1 unless a feature is constant within the class), and **two rows
per class give β ≡ 0** (the centered block is ±v), so λ = 0 and the pooled covariance is singular —
the class-size guard therefore requires ≥ 3 rows per class inside every training fold (the SVM's
nested guard is replaced, not copied). **Four user decisions (2026-09-17):** binary-only v0.1
(multiclass is a v0.2 extension; the design's four-class MagNet run reached macro OvR AUC 0.957);
`run_null=False` stays the default for API symmetry, with Stage 4 running the cheap null immediately
after the first pass; the LDA is a **third sibling on its own triggers** (SVM tuning noise
unresolvable at small *n*; a covariance-adjusted reading; the null needed now), each alternative
compared **paired against the elastic net, never all-pairs** — the *Choosing between the two linear
classifiers* rule was generalized to *Choosing among the linear classifiers* (reference = elastic
net; reading pre-stated among three; alternative cleanly fitted — SVM tuning warnings resolved / LDA
`ShrinkageSaturationWarning` acknowledged; every AUC reported); and **three figures**, not four (the
first template with no hyperparameter figure; the shrinkage λ pair is annotated on the ROC instead).
**Six documented divergences** in-code: nothing to tune; dense weights → top-k membership (the SVM
convention); calibrated log-odds (class-frequency priors, no `class_weight`); the per-fold +
all-data shrinkage diagnostic with `ShrinkageSaturationWarning` at λ ≥ 0.95 (the weights are then ≈
a standardized mean difference); fold identity + per-repeat AUCs; no `n_jobs`. **Preview evidence
(same folds as the shipped templates, seed 0; `testdata/*/_classification_lda_preview/`):** 5xFAD
genotype LDA **0.848 ± 0.140** (EN 0.905, SVM 0.796 — LDA > SVM at all three seeds; vs EN 5/13/7), λ
= 0.47 / 0.24, null p = 0.001 in 139 s incl. 1,000 permutations, top-20 = APP + the human transgene,
midkine, APOE, TICN1/2, clusterin, GPC1, the C1q trio, SPON1 with 0.83 mean fold-membership and
17/20 overlap with the elastic net; MagNet-EV-ADD LDA **0.950 ± 0.108** (SVM 0.963 — a tie across
seeds; vs EN 0.870: 10/0/15, corrected-t p 0.34 — the correction is conservative at J = 25, which is
why sign-across-seeds is the rule), λ = 0.25 / 0.50, null p = 0.001 in 41 s, but weights unstable
(0.45 membership) — the membership read is what stops a dense list being mistaken for a stable
feature set. A synthetic 100 × 20,000 fit: 40 ms. **52 tests + 1 splits extension + 1 result-io
round trip**; CI now type-checks `lib/analysis` too (a pre-existing gap). Wired into
`lib/manifest.md`,
`conventions/{statistics,visualization,findings,results-cache,enforcement-map}.md`,
`commands/stage4-explore.md` (offered on its triggers; compared paired against the elastic net),
`agents/{statistician,stats-reviewer,figure-generator}.md`, `skills/statistical-analysis/SKILL.md`,
`templates/{project-CLAUDE,finding}.md`, `spec/05`, `lib/README.md`.

### B.2 Elastic-net linear regression (continuous outcome) — ✅ SHIPPED (v0.1, 2026-07-06) — *source oracle* (`te-phase2a-pelt/src/regression.py` + `regression_plotting.py`)

**✅ Shipped.** `lib/analysis/regression` (`regress(dataset, outcome=, *, groups=,
generalization_target=, feature_list=, run_null=, …)`) + the four result figures
`lib/figures/regression` (predicted-vs-observed / target-shuffle-null / coefficient-
selection-frequency / alpha×l1 heatmap), strict-clean (ruff + `mypy --strict`) + 22 tests
(planted-truth linear recovery, pure-noise-vs-null, continuous API, `feature_list`, guards,
grouping, four figures, a real-trex smoke). **Built as the continuous-outcome twin of the
classifier (§B.1), the same shape** (the user's steer — stay consistent with how the
classifier does CV and reports features/coefficients): nested CV (in-fold `StandardScaler`,
**R²/RMSE/MAE**), all-data standardized coefficients, a fixed-hyperparameter stability loop
(selection frequency + sign consistency), and an **opt-in target-shuffle null** (the
exploratory↔validated gate). Numeric `outcome=` used directly (a categorical outcome
**raises** → use `classify`; non-finite-outcome rows dropped); group-aware CV only when
`groups` repeats (group-level target permutation); `generalization_target` recorded.

**What the oracle gave — and its gaps (the same the classifier fixed).** `regression.py`'s
`tune_regression_hyperparameters` (von-Neumann-smoothed grid) + `evaluate_regression`
(repeated K-fold, in-fold scaler) + `regression_plotting.py`'s scatter + grid heatmap are
the reusable bones. But it **tunes on all data then CVs on the same data** (optimistic
leakage → nested CV), has **no group-awareness, no shuffle null, no coefficient
extraction/stability**, silently `NaN→0` (we **raise**), uses **seaborn** (re-implemented on
matplotlib), a **pickle cache** (`S301` / file-format convention — dropped, project-local),
and study-specific `pred_floor/pred_ceiling` clipping (stripped). All fixed/stripped, exactly
as the classifier did.

**New `feature_list=` (the user's steer, 2026-07-06).** Optional prior-knowledge feature
restriction — leakage-safe (**must be outcome-independent**), matched/unmatched counts
recorded in provenance; mirrored into the flow (`commands/stage4-explore.md`) and
`conventions/statistics.md`, and kept in mind for the classifier. See memory
`feature-list-restriction-ml-templates`.

**Preview-first eyeball-gate finding.** Trex **dose** is genuinely **null** even in the
published manuscript (R²=−0.018, MAE 27 cGy) — the preview reproduced it, and a curated
feature list couldn't sharpen an absent signal. Pivoted to **time-since-exposure** (the
manuscript's strong signal, R²=0.90): honest nested-CV R²=0.77 on the 144-sample irradiated
subset, clears the target-shuffle null (p=0.005). New git-ignored `testdata/trex/` (5xFAD has
no continuous outcome). Preview record in `testdata/trex/_regression_preview/`. **No new
shipping dep** (scikit-learn already baselined). Evidence reuses the classification/selection
kind (§8; R²/RMSE/MAE vs the target-shuffle null + the per-feature stability read).

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

**Tuning guidance (encoded 2026-07-13).** The default run (`max_iter=100`, `alpha=0.05`,
`perc=100`) is a starting point, not a fixed contract — the template docstring, `commands/stage4-explore.md`,
and `conventions/statistics.md` all carry the same steer: **too few** Confirmed → raise
`max_iter` (~`200`, so borderline Tentatives resolve) and/or lower `perc` toward `95`/`90`
(a more lenient shadow floor — the threshold is `percentile(imp_sha, perc)`); **too many**
for a tractable downstream analysis → lower `alpha` to `0.01` (a stricter accept/reject bar).
Any non-default is a recorded choice (enters `provenance.params` + the cache fingerprint).

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
- **Enrichment / pathway** ORA is downstream of feature finding — **now shipped as its own
  layer:** `lib/analysis/enrichment` (GO/KEGG over-representation via g:Profiler) + the three
  `lib/figures/enrichment` figures. It consumes a query (a DE hit list, Boruta-confirmed,
  classifier-selected, a cluster) + the detected-proteome background, so it is a new family,
  not a `method=` of this template. Full plan/record in [`ENRICHMENT.md`](ENRICHMENT.md).
  **GSEA** (rank-based, no cutoff) remains a possible later template (a different method
  family — fgsea-style), and other enrichment services / offline gene-set DBs stay out of
  scope for now.
- **Full predictive modelling** (ROC, calibration, held-out performance *as the deliverable*)
  is roadmap #3/#4. This doc covers those models' **feature-selection** use only — keep the
  line explicit.
