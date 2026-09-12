# 06 — Visualizations

Figures are evidence. They carry findings into talks and reports, and a wrong or misleading figure propagates as confidently as a wrong number. The governing principles: **accuracy is paramount**, and **a figure is a regenerable artifact, not a hand-made image.**

## 6.0 Show, don't tell — figures are first-class evidence

*(Added after implementation, by user decision — the strengthening of the inline-figures rule of doc 03.)*

- **A claim that can be shown is shown.** Figures are how a claim is made, not decoration on it. For every claim a finding makes about the data, the question "what figure shows this?" is asked, and where a figure can be made it **is** made — commissioned as part of recording the claim (doc 03; `conventions/findings.md` §2.4). A finding whose claims could have been shown and were not is incomplete.
- **The figure shows; the text explains.** The image carries the evidence; the finding's (or report's) prose carries the reading — what is plotted, where to look, what it establishes (`conventions/findings.md` §9). An unexplained figure leaves the reader guessing; an explanation baked onto the canvas is illegible at print scale and drifts out of sync with the text it duplicates.
- **The annotation budget** (§6.4) follows from this: only what a reader needs in order to *read* the figure goes on the canvas.

Coverage is a **judgment call and has no hook** — "could this claim have been shown?" is not decidable from a tool-use event. The findings-manager runs the claim→figure coverage pass and reports uncovered-but-showable claims; the figure-reviewer enforces the annotation budget; the report-reviewer checks a shown claim stays shown (`conventions/enforcement-map.md`).

## 6.1 Accuracy and review

- Every generated figure is **reviewed as a rendered PNG**, not merely as the code that produced it. Code can be correct and the render still wrong (clipped labels, misleading axis, wrong color mapping, overplotting). The figure reviewer (doc 04) inspects the actual render. This mirrors the correctness charter's rule: verify the artifact, not just the script.
- Figure generation and review are a **generator/reviewer subagent pair** (doc 04).
- A figure is not accepted until its render passes review.
- **Control samples are rendered separately from experimental samples** in QC and descriptive figures (their own panels or visibly distinct), never silently pooled into the experimental distributions — a pool's tight cluster or a blank's empty profile would otherwise distort the very spread the plot exists to show. The experimental/control split is the one settled in Stage 1 (doc 02.1, doc 05.3). Because control *subtypes* read differently (pooled-QC = technical reproducibility, references = cross-batch anchoring), a QC figure may give **each control type its own distinction** — the PCA class coloring uses a separate color per control type, and the CV plot renders one figure per control type. *(Five deliberate, documented exceptions in the implementation: the `sample-correlation` heatmap, the `id-depth` bar chart, the `missingness` completeness curve, the `dynamic-range` per-class overlay, and the `pca-plot` sample-class coloring show controls together with experimentals — labeled by a stripe / bar / curve / point color — because the cross-class comparison is the deliverable; see `conventions/visualization.md`.)*
- **QC figures depict the processing state and let the scientist see each step's effect.** Every QC figure labels whether it shows **raw**, **normalized**, or **batch-corrected** data (and the scale). Plots that read an *effect of processing* — PCA, CV, abundance box plots — are rendered as the **raw → normalized → batch-corrected** series so the scientist can judge whether each step helped (e.g. do controls cluster more tightly, does CV drop). The batch-corrected state is shown **only when batch correction is applicable** (a real batch axis with ≥ 2 batches) and is a **preview** — batch-label-only, to display the effect for sign-off — *not* the committed testing decision, which is made in Stage 4 (doc 02.4, doc 05.3).

## 6.2 Output formats

Every visualization is saved in **both**:

- **SVG** (vector, for editing and publication), and
- **PNG at 300 DPI** (raster, for review and embedding).

Both are written under `figures/`. The PNG is the review and embedding target; the SVG is the editable master.

**`figures/` is structured, not flat** *(added after implementation, by user decision)*. A study renders dozens to hundreds of image files, so the directory path describes the figure: `figures/metadata/<family>/` (Stage 1 cohort characterization), `figures/qc/<family>/` (the Stage 3 QC report), `figures/analysis/<family>/<label>/` (Stage 4 onward — `<label>` the contrast / outcome / result). The top level is decided by **the stage that commissions the figure** (so a PCA rendered as QC in Stage 3 lands in `qc/pca/`, and the same template re-rendered for biology in Stage 4 lands in `analysis/pca/<label>/`), the family is the plot/analysis template, processing state stays in the file stem (most QC figures compare states inside one figure), a finding-attached figure's stem starts with its finding id, and the tree never exceeds three levels. The legend image sits beside its figure. The orchestrator names the target directory in every figure dispatch; the reviewers check it, and the findings guard backstops it in projects that opted in via `state/workflow.json` (`figures_layout`), leaving projects initialized before the layout untouched (`conventions/visualization.md`, *Where figures live*).

## 6.3 Legends

Render **legends as separate images** alongside each figure (`<name>.legend.{svg,png}`), rather than baking them into the plot, where they routinely overlap the data. This keeps figures clean, supports publication workflows where legends are placed separately, and makes the figure's encoding explicit and reviewable. *(As implemented, by user decision: the legend is an image, not a `.legend.md` document — the free-text caption lives in the finding's `figures[].caption`.)*

**Separate does not mean optional.** A legend is essential to interpreting its figure, so wherever a figure is embedded — a finding's body, a report — its legend image is **embedded directly beneath it** (doc 03, the show-don't-tell pattern), never cited as a path the reader must open. A few figures keep a small key on-axes where it provably clears the data (`conventions/visualization.md` lists the exceptions); those have no legend image and embed none.

## 6.4 Publication-ready defaults

Figures default to publication quality: legible font sizes at print scale, no chartjunk, clear axis labels with units, appropriate aspect ratios, and consistent typography. The visualization library (`lib/`, doc 04) encodes these defaults so every figure inherits them.

**The annotation budget — do not embed long descriptions in a figure.** A figure carries only the annotation a reader needs in order to *read* it: axis labels with units, tick labels, a short title naming the comparison (and the processing state + scale where it matters), terse load-bearing numbers (N, effect, p/q, hit counts, a threshold's value), mandatory caveat markers, and direct point/series labels where they beat a legend. **Off the canvas:** paragraphs of any kind (interpretation, "what this shows", methods narrative, conclusions), a duplicated caption (it lives once, in the finding's `figures[].caption`), the legend (a separate image, §6.3), and anything legible only past print scale. *A sentence someone could say about the figure belongs in the text; a label the eye needs while looking at the figure belongs on the figure.* The figure-reviewer fails a render carrying explanatory prose.

## 6.5 Color — palette and the category registry

- **Palette: Okabe–Ito** (color-blind-friendly) as the standard categorical palette. Figures should remain interpretable for color-vision-deficient viewers and, where feasible, in grayscale.
- **Standardized category colors.** A given categorical value must use the **same color in every figure**. If male/female are colored, male and female keep their colors everywhere; the same holds for every categorical label.
- **The color registry.** The mapping is stored in `state/color_registry.json`, a machine-readable file every plotting script reads, so consistency is mechanical rather than remembered.

### Registry structure

A JSON object mapping a category dimension to value→color assignments, with provenance for whether a mapping is a universal default or project-specific:

```json
{
  "sex":      { "scope": "universal", "values": { "male": "#0072B2", "female": "#D55E00" } },
  "treatment":{ "scope": "project",   "values": { "control": "#009E73", "drug_A": "#CC79A7", "drug_B": "#E69F00" } }
}
```

- **Universal defaults** (e.g. sex) ship with the plugin and are seeded into every project.
- **Project-specific categories** (treatment arms, cell lines, timepoints) are only knowable after `METADATA.md` exists (doc 02), so the registry is **extended per project** once metadata is understood. Implementation must distinguish the two scopes.

## 6.6 The >8-category problem

Okabe–Ito provides eight distinguishable colors. Beyond eight categories, **adding more colors is the wrong move** — a 12-color categorical palette is unreadable regardless of which palette it comes from. The rule: **color encodes at most eight categories; beyond that, change the encoding strategy.** Options, in rough order of preference:

1. **Faceting / small multiples** — split into panels rather than cramming categories into one legend.
2. **A second channel** — combine color with shape or linetype to extend distinguishability modestly.
3. **Group the long tail** — collapse minor categories into an explicit "other."
4. **Position/sequential encodings** — where the category is ordinal or numeric, use position or a sequential scale instead of categorical color.

The library should detect when a plot is about to exceed eight categorical colors and require the script to choose an explicit strategy rather than silently recycling colors.

## 6.7 Figure provenance

Every figure records, and the finding that uses it pins, the producing script (path + commit), the data version, and parameters (doc 03 provenance). Because figures are regenerable from this, the staleness machinery (doc 03.8) covers them too: if the data version or script changes, figures built on the old version are flagged.
