# Convention — Visualization

*Spec source: doc 06. Figures are evidence — a wrong or misleading figure propagates as confidently as a wrong number. Governing principles: **accuracy is paramount**, and **a figure is a regenerable artifact, not a hand-made image.** Enforced by the **figure-reviewer**, now backed by the shipped `lib/figures/` machinery — `figure-io` (dual export + separate legend + publication style) and `okabe-ito-colors` (the color registry + the >8-category raising guard) — which carries these defaults mechanically for any project figure script that routes through it. The QC/descriptive plot templates built on it have shipped and are wired into the Stage 3 QC report (`id-depth`, `missingness`, `dynamic-range`, `abundance-boxplot`, `cv-plot`, `sample-correlation`, `pca-plot` — see *QC & exploration figures*), as have the first **analysis-result** plot templates — the **`volcano`** and **`pvalue-hist`** companions of the differential-abundance template, and the **four classifier result plots** — ROC-vs-null, coefficient/selection-frequency, and the hyperparameter heatmap (`lib/figures/classification`) — of the classification template (see *Analysis-result figures*). The regression result plots remain phase-E work.*

## Accuracy & review

- Every generated figure is **reviewed as a rendered PNG**, not merely as the code that produced it. Code can be correct and the render still wrong (clipped labels, misleading axis, wrong color mapping, overplotting). The **figure-reviewer** inspects the actual render.
- Figure generation and review are a **generator/reviewer subagent pair**.
- A figure is not accepted until its render passes review.

## Output formats — dual export

Every visualization is saved in **both**:

- **SVG** — vector master, for editing and publication.
- **PNG at 300 DPI** — raster, the review and embedding target.

Both go to `figures/`. In matplotlib: `fig.savefig(base + ".svg")` and `fig.savefig(base + ".png", dpi=300)`. The finding's `figures` entry points at both plus the legend image.

## Legends as separate images

Render the **legend as its own image** (`figures/<name>.legend.svg` + `figures/<name>.legend.png`) alongside the figure rather than baking it into the plot. A legend drawn inside the axes routinely overlaps the data; rendering it as a standalone swatch key (categorical) or colorbar (continuous) keeps the figure clean and lets publication workflows place the legend separately. The figure's free-text caption lives in the finding's `figures[].caption`, so the legend artifact is purely the visual key. (`lib/figures/figure_io.save_figure` dual-exports a companion legend figure to `<name>.legend.{svg,png}`; `lib/figures/pca.save_pca` builds the swatch/colorbar legend.)

**Exception — a keyed legend that provably clears the data.** A few figures keep their legend *on-axes* where it cannot overlap the plot: the ROC curve's chance / mean / ±SD key in the empty lower-right corner (`lib/figures/classification.plot_roc`), and value-scale colorbars placed beside the axes (the `sample-correlation` r-colorbar; the coefficient plot's selection-frequency bar). These are the documented exceptions — the reviewer treats them as conforming, precisely because the key doesn't collide with the data the way an in-axes categorical legend would.

## Publication-ready defaults

Figures default to publication quality: legible font sizes **at print scale**, no chartjunk, clear axis labels **with units**, appropriate aspect ratios, consistent typography. The `lib/` figure templates encode these defaults (a shared matplotlib style, imported by the project's figure scripts) so each figure starts from them rather than re-specifying them.

## Descriptive & cohort figures

The Stage 1 metadata characterization (the distribution of every variable, pairwise crosstabs, the cohort "Table 1") produces **first-class figures**, subject to every rule here — dual export, Okabe–Ito via the registry, the ≤8-category guard, render review. They are both publication deliverables and the lens that exposes class imbalance and confounding; the consequential ones are recorded as caveat findings (`conventions/statistics.md`; `conventions/findings.md` §2.6). Color the categorical design variables (sex, group, batch) through `state/color_registry.json` so a given level keeps its color from the very first cohort plot through every downstream figure.

**Control samples are rendered separately from experimental samples** in QC and descriptive figures — their own panels or visibly distinct, never silently merged into the experimental distributions (a pool's tight cluster or a blank's empty profile would otherwise distort the very spread the plot is meant to show). The experimental/control split is the binary one settled in Stage 1 (`conventions/statistics.md`); because pools and references read differently (technical reproducibility/drift vs cross-batch anchoring), a QC figure may further distinguish the control subtypes.

**Four deliberate exceptions** — the **sample-correlation heatmap** (`lib/figures/sample-correlation`), the **identification-depth bar chart** (`lib/figures/id-depth`), the **missingness completeness curve** (`lib/figures/missingness`), and the **dynamic-range per-class overlay** (`lib/figures/dynamic-range` with `class_by`): there, controls are shown *with* the experimental samples, distinguished by a **label** (a correlation annotation stripe; a depth-bar color; a curve color), because the cross-class comparison (do pools sit apart from biology / sit high and tight on depth / run more complete / cover the same dynamic range? do references bridge batches?) *is* the deliverable — splitting them apart would destroy it. These are the only QC figures that mix the two classes, and only because they label them; the reviewer treats them as conforming.

## QC & exploration figures

The Stage 3 QC report (and the Stage 4 exploration seeds carried from it) is built on seven shipped `lib/figures/` templates, each routing colors through the registry and dual-exporting with a separate legend. **Scale is load-bearing and differs per plot:**

- **`id-depth`** — detected-features-per-run bar charts, one stacked panel per feature level (protein over precursor), bars drawn in **acquisition order** so a failing/low-ID run or a drifting block stands out; the first-look QC of how deep each run went. Bars colored by an optional categorical column (e.g. sample class) via the registry + >8 guard; a per-panel **reference-median line** over the experimental subset. Computed on the **raw linear** matrix — the template **hard-refuses a non-linear scale** (`IdDepthScaleError`: a detection count is a raw-data property). Controls shown with experimentals, distinguished by bar color (the exception above).
- **`missingness`** — the per-feature complement to `id-depth`: a **completeness curve** (features retained vs the required detection fraction, optionally overlaid per sample class) plus an **MNAR diagnostic** (per-feature detection rate vs mean log2 abundance, as a hexbin + binned-median trend + Pearson r). A positive trend = low-abundance features are left-censored (MNAR), which drives the imputation-method choice. Computed on the **raw linear** matrix — **hard-refuses a non-linear scale** (`MissingnessScaleError`). The hexbin density colorbar stays on the figure; the completeness overlay shows controls with experimentals (the exception above).
- **`dynamic-range`** — features ranked by abundance vs log2 abundance (rank-abundance), showing the quantified **dynamic range** (orders of magnitude) and any hyper-abundant dominance (e.g. albumin / contaminants at the top). Whole-cohort **median + IQR band** (run on the experimental subset, like CV/PCA), or a per-class independently-ranked overlay (`class_by` — the labeled exception above). Optional **`highlight_features`** mark named proteins of interest with leader-line labels — **empty at QC, populated downstream** (domain targets / DE hits; see Stage 4), turning the QC curve into a results figure. Computed on the **raw linear** matrix — **hard-refuses a non-linear scale** (`DynamicRangeScaleError`).
- **`abundance-boxplot`** — one stacked per-sample box-plot panel per processing state (raw → normalized → batch-corrected); the per-feature counterpart to the CV plot, read as a per-sample view of whether normalization flattens the per-sample medians and batch correction removes residual shift. Boxes are colored by **position** (a sequential / ordinal encoding — no categorical budget consumed), so ordering the samples by acquisition order upstream surfaces run-order drift. On the **normalized log** matrix (**warns** on a non-log scale); per-panel y-labels are derived from the `scale` tag. Controls rendered separately (own call).
- **`cv-plot`** — per-feature CV (std/mean) distributions, overlaid across normalization states (unnormalized → normalized → batch-corrected) for the **experimental subset**, with pools on a **separate** technical-reproducibility panel. Computed on the **linear** matrix — the template **hard-refuses a non-linear scale** (`CVScaleError`).
- **`sample-correlation`** — hierarchically-clustered sample×sample heatmap with metadata annotation stripes; the design sanity check. On the **normalized log** matrix (Pearson **warns** on a non-log scale; Spearman is scale-invariant). Controls shown with experimentals via the Sample Type stripe (the exception above).
- **`pca-plot`** — PC1/PC2 + PC3/PC4 with marginal tests, colored by the batch axis (reference samples greyable). On the **normalized log** matrix (**warns** on a non-log scale).

In Stage 3 these are descriptive sanity checks (no findings); in Stage 4, PCA and sample-correlation recur as exploration seeds (e.g. PCA colored by the variable of interest). The figure-reviewer confirms the scale matches the plot and that controls are handled per the rules above.

## Analysis-result figures

The **differential-abundance** template (`lib/analysis/differential-abundance`) ships with two result figures on the same foundation (registry colors + dual-export + separate legend). These are **Stage-4** figures — they visualize a test result, not raw-data QC — so they run on whatever the test ran on (normalized log data):

- **`volcano`** — per-feature **effect (log2 fold change)** on x vs **−log10(BH q)** on y, three-way significance coloring (NS / up / down) with the **hit counts in the legend**; the `q` underflow is floored so an extreme point still plots. Up/down are two registry colors (a `Significance` category) and **NS is the gray background** (no palette slot); an optional `effect_threshold` adds the fold-change gate. An optional `annotate_top` labels the most-significant hits **collision-free** (via `textalloc`: labels repelled off each other and the data, leader lines back to each point, clamped inside the axes — a dense cluster of co-significant hits stays legible). Read it off a `DifferentialAbundanceResult` via `volcano_from_result` (pick the contrast term for a `k>2` factor).
- **`pvalue-hist`** — the calibration diagnostic on the **raw** p-values (not q): uniform body + a spike at 0 is healthy; a U-shape / hump near 1 signals unmodeled structure or confounding; a hill toward 1 is conservative. Overlay the both-ways (corrected/uncorrected) or method pair, colored from the Okabe-Ito palette; Storey's π0 is shown per distribution. Its overlay labels are figure-local (methods, batch-handling, real-vs-null), so colors are **not persisted to the registry by default** (`persist_colors=False`) — unlike the QC plots' fixed vocabularies — which keeps independent figures from accumulating palette-tail colors or polluting the registry with non-biological labels. It **refuses** values outside `[0, 1]` (the *passed q by mistake* slip).

The **classification** template (`lib/analysis/classification`) ships **four** result figures (`lib/figures/classification`), also Stage-4 (they visualize a fitted model, on normalized log data):

- **`plot_roc`** — the mean ROC across outer nested-CV folds with a **±1 SD band** and a chance diagonal; balanced accuracy, average precision, and per-class N annotated. Its legend sits **on-axes** (the documented exception above).
- **`plot_null`** — the **label-shuffle null** AUC histogram with the observed AUC marked and the empirical p. **Conditional:** it renders only when the null was run (`run_null=True`) and **raises** otherwise (a classification finding without the null is `exploratory`).
- **`plot_coefficients`** — the **top-N selected features** by |all-data standardized coefficient|, each a diamond at its final coefficient over its **resample IQR**, colored by **selection frequency** (viridis); a line at 0 separates the classes. **Selected-only** — features zeroed in the final model do not appear — so it shares a *skeleton* with the future Boruta importance plot, not an identical look. The viridis colorbar sits beside the axes (no separate legend).
- **`plot_hyperparameter_heatmap`** — the C × l1_ratio tuning surface (mean inner-CV AUC) with the selected cell boxed; a diagnostic.

## Color — palette and the registry

- **Palette: Okabe–Ito** (color-blind-friendly) as the standard categorical palette. Figures should remain interpretable for color-vision-deficient viewers and, where feasible, in grayscale.
- **Standardized category colors.** A given categorical value uses the **same color in every figure** — if male/female are colored, they keep their colors everywhere; likewise every categorical label.
- **The color registry.** The mapping lives in `state/color_registry.json` (schema: `templates/color_registry.json`). **Every plotting script reads it**, so consistency is mechanical, not remembered. Universal defaults (e.g. sex) ship with the plugin and are seeded at `init`; project-specific categories (treatment arms, cell lines, timepoints) are added once `METADATA.md` exists, with `scope: project`. New categorical values are assigned colors deterministically from `_palette.colors`.

## The >8-category problem

Okabe–Ito provides eight distinguishable colors. Beyond eight categories, adding more colors is the wrong move — a 12-color categorical palette is unreadable regardless of palette. **The rule: color encodes at most eight categories; beyond that, change the encoding strategy.** In rough order of preference:

1. **Faceting / small multiples** — split into panels rather than cramming categories into one legend.
2. **A second channel** — combine color with shape or linetype to extend distinguishability modestly.
3. **Group the long tail** — collapse minor categories into an explicit "other."
4. **Position/sequential encodings** — for ordinal/numeric categories, use position or a sequential scale instead of categorical color.

The shared color module (`lib/figures/colors.py`, `okabe-ito-colors`) includes a **guard that raises (`CategoricalPaletteExceededError`) when a category would exceed eight colors, requiring the script to choose an explicit strategy** rather than silently recycling colors. *(This guard has shipped — it is the deterministic enforcer of the rule. The figure-reviewer still confirms the script routes its colors through the registry, since a reviewer cannot reliably see a silently-recycled color in a script that bypasses it.)*

## Figure provenance

Every figure records — and the finding that uses it pins — the producing **script (path + commit)**, the **data version**, and **parameters** (finding `provenance`). Because figures are regenerable from this, the staleness machinery (doc 03.8) covers them: if the data version or script changes, figures built on the old version are flagged for re-generation.

## Enforcement

| Rule | Enforced by |
|---|---|
| Render reviewed (PNG), not just code | **Figure-reviewer** |
| Dual export (SVG + 300 DPI PNG) + separate legend image present | **Figure-reviewer** (+ `figure-io.save_figure` dual-exports the figure and a companion `<name>.legend.{svg,png}` legend image) |
| Okabe–Ito; category colors from the registry; consistency | **Figure-reviewer** (+ `okabe-ito-colors` reads/extends `state/color_registry.json`) |
| ≤8 categorical colors; explicit strategy beyond | **Figure-reviewer** (+ `okabe-ito-colors` raises `CategoricalPaletteExceededError` past 8) |
| Control samples rendered separately from experimental in QC/descriptive figures (exceptions: the `sample-correlation` heatmap, the `id-depth` bar chart, the `missingness` completeness curve, and the `dynamic-range` per-class overlay, which label them — a stripe / bar / curve color) | **Figure-reviewer** |
| QC plot scale correct (id-depth/missingness/dynamic-range/CV linear; PCA/correlation log) | **Figure-reviewer** (+ `id-depth`/`missingness`/`dynamic-range`/`cv-plot` refuse non-linear; `pca-plot`/`sample-correlation` warn on non-log) |
| Analysis-result figures consistent with the test (volcano effect/q + hit counts; p-value histogram on raw p, not q) | **Figure-reviewer** (+ `volcano`/`pvalue-hist` route colors through the registry; `pvalue-hist` refuses values outside [0,1]) |
| Classification figures consistent with the result (ROC ±SD, legend on-axes; null figure present only when the null ran; coefficient plot selected-only + colored by selection frequency; heatmap selected cell boxed) | **Figure-reviewer** (+ `lib/figures/classification`; `plot_null` raises without a null) |
| Figure provenance pinned; staleness tracked | findings-manager (staleness) + finding `provenance` |
