# 06 — Visualizations

Figures are evidence. They carry findings into talks and reports, and a wrong or misleading figure propagates as confidently as a wrong number. The governing principles: **accuracy is paramount**, and **a figure is a regenerable artifact, not a hand-made image.**

## 6.1 Accuracy and review

- Every generated figure is **reviewed as a rendered PNG**, not merely as the code that produced it. Code can be correct and the render still wrong (clipped labels, misleading axis, wrong color mapping, overplotting). The figure reviewer (doc 04) inspects the actual render. This mirrors the correctness charter's rule: verify the artifact, not just the script.
- Figure generation and review are a **generator/reviewer subagent pair** (doc 04).
- A figure is not accepted until its render passes review.
- **Control samples are rendered separately from experimental samples** in QC and descriptive figures (their own panels or visibly distinct), never silently pooled into the experimental distributions — a pool's tight cluster or a blank's empty profile would otherwise distort the very spread the plot exists to show. The experimental/control split is the one settled in Stage 1 (doc 02.1, doc 05.3). *(Four deliberate, documented exceptions in the implementation: the `sample-correlation` heatmap, the `id-depth` bar chart, the `missingness` completeness curve, and the `dynamic-range` per-class overlay show controls together with experimentals — labeled by a stripe / bar / curve color — because the cross-class comparison is the deliverable; see `conventions/visualization.md`.)*

## 6.2 Output formats

Every visualization is saved in **both**:

- **SVG** (vector, for editing and publication), and
- **PNG at 300 DPI** (raster, for review and embedding).

Both are written to `figures/`. The PNG is the review and embedding target; the SVG is the editable master.

## 6.3 Legends

Render **legends as separate documents** alongside each figure, rather than only baking them into the image. This keeps figures clean, supports publication workflows where legends are typeset separately, and makes the figure's encoding explicit and reviewable.

## 6.4 Publication-ready defaults

Figures default to publication quality: legible font sizes at print scale, no chartjunk, clear axis labels with units, appropriate aspect ratios, and consistent typography. The visualization library (`lib/`, doc 04) encodes these defaults so every figure inherits them.

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
