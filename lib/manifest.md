---
schema_version: 1
generated: 2026-06-23
---

# lib/ template registry

Derived index of the vetted **template scripts** in `lib/` — source of truth is each template's
`__script_meta__` header (regenerate by scanning them, like the findings manifest). Templates are
*seeds*: copied into a project's `scripts/` and adapted (`conventions/script-registry.md`). A
project records which template + version it seeded from in its findings' `provenance.seeded_from`.

**The contract is the data structure, not the file format.** A loader template encodes the
*shape* of the in-memory object the rest of the templates consume (e.g. `Dataset`). A study whose
files don't match a template's input may need its own loader — written and tested for that data,
with the template as a guide — but it should return the **same structure** so downstream templates
(analysis, figures) compose unchanged.

| Template | Version | Path | Kind | Provides | Description |
|----------|---------|------|------|----------|-------------|
| wide-data-loader | 0.4 | lib/common/data_loading.py | module | `Dataset`, `Scale`, `LOG_SCALES`, `ReplicateCollapse`, `load_wide_data`, `load_precursor_data` | Verified loader for wide feature×sample omics matrices + a sample-metadata table: orientation/pairing checks, optional technical-replicate collapse, precursor charge-state collapse, zero-preserving, fail-loud. Study-agnostic (column names are arguments); returns the standard `Dataset` contract, tagged with a recorded `scale` (`linear`/`log2`/`log10`/`ln`/`glog2`/`zscore`/`ratio`). |
| normalize | 0.3 | lib/common/normalize.py | module | `NormalizationMethod`, `normalize`, `log2_transform`, `median_center` | Verified `Dataset`→`Dataset` normalization (pronoms median / MAD / VSN), `log2(x+1)` transform, and log-domain `median_center`. Scale-tag guarded against the double-log trap (refuses non-linear input), shape-preserving, fail-loud, returns an independent (non-aliased) `Dataset`. Study-agnostic; sets the output `scale`. Requires `pronoms`. |
| batch-correct-combat | 0.2 | lib/common/batch_correct.py | module | `BatchConfoundingWarning`, `BatchPassthroughWarning`, `ConfoundingReport`, `assess_batch_confounding`, `combat_correct` | Verified ComBat batch correction on a `Dataset` — **batch label only** (no covariate preserved, a deliberate anti-cheating choice), log-scale guarded, near-constant-feature passthrough (warned, with a finite-output backstop), `sample_mask`, fail-loud, returns an independent `Dataset`. Ships `assess_batch_confounding` (bias-corrected Cramér's V) which warns — graded (perfect or strong) — when the batch axis is confounded with a covariate of interest. Requires `pycombat`. |
| okabe-ito-colors | 0.2 | lib/figures/colors.py | module | `BACKGROUND_COLOR`, `Palette`, `CategoricalPaletteExceededError`, `DEFAULT_REGISTRY_PATH`, `load_registry`, `load_palette`, `assign_colors`, `get_color` | Project color registry: reads/extends `state/color_registry.json` so a given `(category, value)` gets the same Okabe-Ito color in every figure. Deterministic next-unused-color assignment, persisted; gray background labels outside the palette (no slot consumed); **raises `CategoricalPaletteExceededError` past 8 categories** (the >8-category guard). Study-agnostic; fail-loud. |
| figure-io | 0.3 | lib/figures/figure_io.py | module | `PUBLICATION_RCPARAMS`, `FigureArtifacts`, `publication_style`, `save_figure` | Figure save helpers enforcing the visualization conventions: **dual export** (SVG vector + 300-DPI PNG) plus an optional companion legend figure exported as `<base>.legend.{svg,png}` (kept out of the plot so it cannot overlap the data), and a shared publication matplotlib style. Study-agnostic; fail-loud. |
| pca-plot | 0.3 | lib/figures/pca.py | module | `PCAScaleWarning`, `PCAResult`, `PCAPlot`, `compute_pca`, `plot_pca`, `save_pca` | PCA scatter figures from a `Dataset`: PC1/PC2 + PC3/PC4 panels with marginal KDE (categorical, Kruskal-Wallis/Mann-Whitney p) or regression (continuous, slope+p). Per-feature standardized PCA; categorical colors from the project registry with the >8-category guard; greyable reference samples; warns on non-log scale; dual-export plus a separate legend image (swatches/colorbar). Uses `common.data_loading`, `figures.colors`, `figures.figure_io`. Study-agnostic; fail-loud. |
