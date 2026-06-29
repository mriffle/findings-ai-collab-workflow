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

Scope for now (the user's call): **feature finding / feature selection only.** Full
predictive modelling (calibration, ROC, held-out performance as the deliverable) is roadmap
items #3/#4 — this doc covers those models' *selection* use, and keeps the line explicit.

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

| Method | Family | Source oracle | Status | Template (planned) |
|---|---|---|---|---|
| Linear model / OLS (+ covariates) | univariate | `feature_finding_ols.py` → `fit_ols_per_feature` | Not started | `lib/analysis/differential-abundance` |
| limma-moderated (empirical Bayes) | univariate | `feature_finding_ols.py` → `moderate_variances` | Not started | (same template; moderated default) |
| Welch / Student t-test | univariate | — (none) | Not started | (same family; `method=` param) |
| Mann–Whitney U | univariate | — (none) | Not started | (same family; `method=` param) |
| Elastic-net logistic regression | multivariate | — (not yet scanned) | Not started | TBD (mode of classifier #3?) |
| Elastic-net linear regression | multivariate | — (not yet scanned) | Not started | TBD (mode of regression #4?) |
| Boruta | multivariate | `te-phase2a-pelt/src/feature_finding_boruta.py` | Not started | TBD (`lib/analysis/boruta`) |
| Volcano plot | output viz | `volcano_plotting.py` | Not started | `lib/figures/volcano` |
| p-value histogram | output viz | — (fresh design) | Not started | `lib/figures/pvalue-hist` |
| Boruta importance box-plot | output viz | `te-phase2a-pelt/src/boruta_plotting.py` | Not started | `lib/figures/boruta-importance` |

**Recommended starting order:** the univariate **linear-model template (OLS + moderated)** —
the spec's core — shipped with its **volcano** + **p-value-histogram** companions; then the
**t-test / Mann–Whitney** variants as parameters of the *same* family; then the
**multivariate** selection methods (elastic-net, then Boruta).

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

### B.1 Elastic-net logistic regression (categorical outcome) — *candidate oracle to scan*
*(`te-phase2a-pelt/src/classification.py` is imported by the Boruta driver and almost
certainly holds the user's leakage-safe classifier / selection code — scan it when we reach B.1.)*
- L1/L2-penalized logistic regression; the **non-zero coefficients are the selected
  features**. `sklearn` `LogisticRegression(penalty="elasticnet", solver="saga")` inside a
  `Pipeline`, `(C, l1_ratio)` tuned by nested / group-aware CV.
- Report: **selection frequency** across resamples (stability selection), signed
  coefficients, CV performance **vs the label-shuffle null**. Imbalance → balanced metrics +
  stratified folds (link the caveat).
- **Overlaps roadmap #3 (leakage-safe classifier).** Open Decision #6: a *mode* of the
  classifier template, or a standalone selection template reusing its CV harness?

### B.2 Elastic-net linear regression (continuous outcome) — *no source oracle*
- Same machinery for a continuous target (`ElasticNetCV` / a `Pipeline`). Report selected
  features + stability + held-out R²/RMSE **vs a shuffle null**. **Overlaps roadmap #4
  (regression)** — same placement question as B.1.

### B.3 Boruta — *source oracle exists* (`manuscript-trex-phase2a/te-phase2a-pelt/src/`)
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
2. **Univariate family shape.** One template with `method=` (`ols` / `moderated` / `welch` /
   `mannwhitney`) sharing the table + plots, vs separate templates. *Leaning one family* —
   matches "downstream is similar."
3. **Confidence intervals.** Must be added (source lacks them). Confirm effect-scale
   (log2FC) + CI columns in the results table for every `method=`.
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
5. **Both-ways robustness.** Corrected-vs-uncorrected run: orchestration (the Stage-4 command
   runs the template twice) vs a built-in template feature.
6. **Multivariate placement.** Elastic-net selection as a *mode* of the classifier (#3) /
   regression (#4) templates, vs standalone selection templates reusing their CV harness.
7. **Boruta engine.** The oracle uses **`BorutaPy` + a private-method subclass**; BorutaPy is
   lightly maintained + untyped (mypy-strict friction, stub-free venv). Take it as-is (a),
   reimplement the shadow loop (b), or a maintained fork (c)? *Leaning (a).* Cache: the oracle
   pickles — re-decide vs our file-format convention. (See §B.3.)
8. **Selection-method evidence shape.** Significance methods (§A) emit effect + CI + p + BH-q;
   **selection methods (§B) emit a different shape** — Boruta: decision (Confirmed / Tentative
   / Rejected) + importance + shadow-null; elastic-net: signed coefficient + selection
   frequency. The "no bare p / effect + CI" rule is written for *significance* tests. **Decide
   how the findings evidence schema + stats-reviewer accommodate a "selection" evidence kind**
   so a Boruta / elastic-net finding is first-class, not force-fit into a p-value mold.
   (Touches `conventions/findings.md` + `conventions/statistics.md`.)

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
