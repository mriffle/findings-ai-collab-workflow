# QC plot gaps — candidate `lib/figures/` templates

**What this is.** A pickup list of canonical quantitative-DIA-proteomics QC/descriptive
visualizations the engine does *not* yet ship, with enough design context to build each in
a future session. It came out of a deliberate "what are we missing" review (2026-06-29).
It is **engine-dev planning**, not user-facing — it never ships into a project.

**How to use it.** Pick one, confirm the design with the user (a real-data *preview*
before coding is the established pattern — see `id-depth`'s history), then build it to the
`lib/` bar following **[`lib/AUTHORING.md`](lib/AUTHORING.md)**. Update the *Status* table
below and `CLAUDE.md`'s *Next* section when one ships.

Mine the source project `johnson-5xFAD-lecanemab-mice-AD` (`src/`) as an oracle where one
exists; note that some of these have **no** source oracle and were designed fresh.

## Shared contract every new plot must honor

These are not optional — they are what makes a template a sound seed (`lib/AUTHORING.md`,
`conventions/{coding,visualization}.md`):

- **`Dataset` in.** Consume the standard container (abundances `(n_samples, n_features)` +
  `feature_names` + `feature_metadata` + row-aligned `metadata`, tagged with a `scale`).
  Multi-state/multi-level plots take an ordered `{label: Dataset}` (the cv / abundance-
  boxplot / id-depth pattern).
- **Scale is load-bearing.** Read `Dataset.scale` and either **refuse** (CV / id-depth
  style, when the statistic is undefined off-scale) or **warn** (PCA / correlation /
  abundance-boxplot style, when it is merely suboptimal). State the choice in-code.
- **Colors through the registry.** Route categorical color via `figures.colors`
  (`okabe-ito-colors`) so a value keeps its color across figures; the **>8-category guard**
  comes for free. Continuous → a perceptually-uniform map (viridis).
- **Save through `figures.figure_io.save_figure`.** Dual-export (`<base>.{svg,png}`) +
  a **separate legend image** (`<base>.legend.{svg,png}`); never bake the legend onto the
  data. `publication_style()` for the shared rcParams. Never leak a figure on an error path.
- **Controls rendered separately** from experimental samples, *unless* the figure labels
  the class and the cross-class comparison is the deliverable (the documented exception —
  `sample-correlation`'s Sample Type stripe, `id-depth`'s bar color).
- **Fail loud, strict-clean.** Validate-and-raise at every boundary; `ruff` strict +
  `mypy --strict`; planted-truth + real-5xFAD smoke (skip when data absent).

## Status

| Plot | Tier | Status | Template |
|---|---|---|---|
| Identification depth (IDs per run) | 1 | **Shipped** (2026-06-29) | `lib/figures/id-depth` |
| Data completeness / missingness | 1 | **Shipped** (2026-06-29) | `lib/figures/missingness` |
| RLE — relative log expression | 1 | Not started | — |
| p-value histogram | 1 | **Shipped** (2026-06-29) | `lib/figures/pvalue-hist` (with the DE template) |
| MA plot (Bland–Altman) | 2 | Not started | — |
| Dynamic range / rank-abundance | 2 | **Shipped** (2026-06-29) | `lib/figures/dynamic-range` |
| Variance components (PVCA-style) | 2 | Not started | — |
| Top-features results heatmap | 2 | Not started | — (Stage 4, after DE) |

Recommended order: **RLE** (the rest of Tier 1 — the p-value histogram shipped with the DE
template), then Tier 2 as the analysis templates land.

---

## Tier 1 — canonical, clearly missing, fit the `Dataset` contract

### Identification depth — **SHIPPED** (`lib/figures/id-depth`)
Detected features (finite & `> min_intensity`) per run, one stacked panel per feature
level (protein over precursor), bars in acquisition order; the first-look QC. Kept here
for completeness — see the manifest and `commands/stage3-loaders.md`.

### RLE — Relative Log Expression
- **What.** Per-sample box plots of each feature's deviation from its **across-sample
  median**: for feature *i* compute `med_i = median_j(x_ij)`, then box the distribution of
  `x_ij − med_i` for each sample *j*. One box per sample, centered on zero.
- **Why it's canonical.** Subtracting the per-feature median removes most biological
  signal, so RLE is far more sensitive to *technical* / normalization problems than the
  raw abundance box plot: well-normalized samples sit as flat boxes centered on 0 with
  similar IQR; a shifted median or inflated IQR flags a bad sample or residual batch
  effect. The standard normalization/batch QC plot (Gandolfo & Speed 2018). It is the
  direct visual answer to "did normalization work?", complementing `cv-plot` (per-feature
  dispersion) and `abundance-boxplot` (per-sample absolute level).
- **Scale.** Read on a **log** scale — the deviation `x_ij − med_i` is a log-ratio (log
  fold change vs the median sample). **Warn** on a non-log scale (like abundance-boxplot;
  on linear the deviations are additive and dominated by abundant features). Compute on the
  normalized log matrix.
- **Canonical use.** One panel per processing state (**raw → normalized → batch-corrected**)
  so you *see* normalization center/flatten the boxes and batch correction remove residual
  shift — the same stacked `{label: Dataset}` shape as abundance-boxplot.
- **Reuse / implementation.** This is essentially `abundance-boxplot` plotting
  *deviations-from-feature-median* instead of raw values, centered at 0. **Open design
  question for the user: extend `abundance-boxplot` with an RLE mode, or ship a sibling
  template** (`rle.py`). A sibling keeps each template single-purpose; a mode shares the
  stacked-box layout/legend code. Lean sibling unless the user prefers the mode.
- **Controls.** Rendered separately (own call), like abundance-boxplot / cv.
- **Wiring.** Stage 3 QC report (`commands/stage3-loaders.md`, `conventions/{visualization,
  statistics}.md`).

### Data completeness / missingness — **SHIPPED** (`lib/figures/missingness`)
Built as a two-panel figure (per recommended scope below): a **completeness curve**
(features retained vs required detection fraction, overlaid per sample class) + an **MNAR
diagnostic** (per-feature detection rate vs mean log2 abundance, hexbin + binned-median
trend + Pearson r). Per-sample missingness was deliberately left to `id-depth`; the
features×samples missingness *map* stays write-from-scratch (the design notes below were
the build spec). Refuses non-linear scale; wired into the Stage 3 QC report and the
Stage-2 imputation decision. Original design notes retained for reference:

- **What.** A small family of views: **(a)** missingness fraction per sample and per
  feature; **(b)** a feature-detection/completeness curve — e.g. "# features detected in
  ≥ k samples" or the distribution of per-feature detection rate; **(c)**
  **missingness-vs-abundance** (per-feature detection rate vs mean log-abundance of its
  detected values) to diagnose **MNAR** — do low-abundance features go missing more?
- **Why it's canonical.** DIA missingness is structured, and the pattern (MCAR vs MNAR)
  determines the right imputation. This is the picture that should *drive* the Stage-2
  imputation decision (leave 0 / impute mean·median·KNN / drop), which is currently made
  blind.
- **Scale.** Missingness/detection is a presence property on the **raw** matrix —
  refuse/compute on linear (same "detected = finite & `> min_intensity`" definition as
  `id-depth`; reuse it). The MNAR view needs an abundance axis (mean log-abundance of the
  detected values per feature).
- **Reuse / implementation.** Registry (color by sample class), `figure_io`. Likely several
  sub-figures or a multi-panel figure — decide with the user whether it's one template with
  a few functions or split. Relationship to `id-depth`: id-depth is the per-sample *count*;
  this is the per-feature / cohort view + the MNAR diagnostic. Complementary, not redundant.
- **Wiring.** Stage 3 QC report **and** an explicit feed into the Stage-2 imputation choice
  (`commands/stage2-data.md` "Preprocessing decisions to confirm").

### p-value histogram — ✅ Shipped (`lib/figures/pvalue-hist`, v0.1, 2026-06-29)
Shipped with the differential-abundance template (see `FEATURE_FINDING.md`). `plot_pvalue_histogram`
takes a `{label: p-array}` mapping (or a single array) or reads a `DifferentialAbundanceResult`
via `pvalue_histogram_from_result`; the uniform null is drawn at density 1.0, Storey's π0 is
shown per distribution, and values outside `[0,1]` are refused (the *passed q by mistake* slip).
The original design notes are kept below for reference.

- **What.** Histogram of the per-feature p-values from a differential-abundance contrast.
  A well-calibrated test → roughly **uniform on [0,1] with a spike near 0** (the true
  positives). Diagnostic shapes: a **U-shape / left-and-right hump** = anti-conservative or
  unmodeled structure (e.g. unmodeled batch/confounding); a **hill near 1** = conservative /
  mis-specified null; a clean **spike-at-0-on-uniform** = healthy.
- **Why it's canonical.** The standard test-calibration diagnostic, and the cheapest insurance
  that the model is right before trusting the volcano. Catches a misspecified model that a
  volcano alone hides.
- **Scale.** Operates on **p-values, not abundances** — scale-agnostic. Input is a 1D array
  of p-values (or the DE result object); can overlay multiple contrasts (registry colors).
  Optionally annotate a π0 estimate (Storey) and the expected-uniform line.
- **Reuse / implementation.** Minimal — `figure_io` + a histogram. **Build it alongside the
  moderated differential-abundance template** (roadmap analysis item #2); it is an
  analysis-output QC, driven by `lib/analysis` output, living in `lib/figures`.
- **Wiring.** Beside the volcano in the Stage 4 analysis/reporting path.

---

## Tier 2 — canonical, additive, more specialized

### MA plot (Bland–Altman)
- **What.** Per-sample `M` (log-ratio vs a pseudo-reference — the per-feature median sample)
  vs `A` (mean log-abundance), with a loess/lowess trend per sample. Faceted over samples or
  a representative subset.
- **Why.** Reveals **intensity-dependent** normalization bias — does the log-ratio drift
  with abundance? A tilted MA cloud means a global median scale won't fix it (motivates
  loess / quantile / VSN). The classic complement to RLE: RLE emphasizes per-sample
  centering/spread, MA emphasizes the abundance-*dependence* of the bias.
- **Scale.** Log (both axes are log quantities). Reference = per-feature median across
  samples.
- **Reuse.** `figure_io`, per-sample faceting. Some purpose-overlap with RLE — build RLE
  first; add MA if intensity-dependent bias turns out to matter.

### Dynamic range / rank-abundance — **SHIPPED** (`lib/figures/dynamic-range`)
Built as a whole-cohort median + IQR band (default) with optional leader-labelled
`highlight_features` (and registry-colored `highlight_groups`), plus a per-class
`class_by` overlay. The **annotation lifecycle** the design discussion settled — QC plain,
annotated downstream with domain targets / DE hits via the same `highlight_features` param —
is wired into Stage 3 (plain) + Stage 4 (re-render). Refuses non-linear scale. Original
design notes retained for reference:

- **What.** Log abundance vs descending rank — one line per sample, or a median line with a
  cohort band. Optionally annotate known high-abundance / contaminant features.
- **Why.** The standard proteomics depth/coverage plot: how many orders of magnitude are
  quantified and whether a few features dominate the signal. (Per-sample **TIC / total
  summed intensity** is a related one-number loading check — cheap to add as a companion.)
- **Scale.** Log y (abundance), on the raw or normalized matrix.
- **Reuse.** `figure_io`, registry (color by sample class).

### Variance components (PVCA-style)
- **What.** For each known factor (batch, group, sex, run-order, …), the fraction of total
  variance it explains. PVCA proper: PCA → fit a random-effects model on the top PCs
  weighted by their eigenvalues → bar chart of variance-explained per factor + residual.
- **Why.** Turns `batch-correct-combat`'s `assess_batch_confounding` from "*is* batch
  confounded" into "*how big* is the batch effect vs biology" — the quantitative input to
  "is ComBat's variance-deflation cost worth it, or model batch as a covariate?"
- **Scale.** Normalized log; reuses PCA (`pca.compute_pca`) + a mixed-effects fit
  (statsmodels) or a simpler per-factor-R²-from-PC-regression proxy / variancePartition
  style.
- **Reuse.** `pca` compute, `figure_io`, registry.
- **Note.** Heavier than the others (needs the random-effects fit done carefully). Pairs
  with the batch-confounding assessment already shipped.

### Top-features results heatmap
- **What.** Clustered heatmap (features × samples) of the top-N significant / most-variable
  features, z-scored per feature for display, with sample annotation stripes (group, batch,
  sex).
- **Why.** The canonical **results** figure — do the significant features separate the
  groups and cluster sensibly?
- **Scale.** Normalized log, z-scored per feature for the color map.
- **Reuse.** Heavily reuses `sample-correlation`'s machinery (scipy clustering + matplotlib
  heatmap + annotation stripes + separate legend) — factor out the shared clustering/stripe
  helpers if it's the second consumer.
- **Note.** This is a **Stage 4 results** plot, not Stage-3 QC. Build after the DE template.

---

## Scope boundaries — deliberately *not* general `lib/` templates

- **Contaminant fraction per sample** (% of total intensity from contaminants — keratin,
  albumin, trypsin, the cRAP set). A real QC metric — the 5xFAD feature ids are even
  prefixed `crapola_crap|…`, so it's directly computable *there* — but it needs a
  contaminant annotation that is **study/database-specific**. Belongs as a Stage-2 prose
  decision or a thin project-local script, **not** a general template (a user's contaminant
  tagging is unknowable a priori — same rationale as caveat findings / control detection).
- **Chromatographic / RT-stability / mass-accuracy / peak-width QC.** Genuinely canonical
  for DIA, but these come from the **raw files or the search-engine report**
  (DIA-NN / Spectronaut / Skyline), not the quant matrix — outside the `Dataset` contract.
  Explicitly out of scope for a quant-matrix workflow; a separate ingestion path if a user
  has the search-engine QC export.
- **Enrichment / pathway analysis (GO / Reactome / GSEA).** Canonical downstream *analysis*,
  but a large scope of its own (gene-set databases, ID mapping, within-set multiple
  testing). A future analysis area, not a QC plot.
