# Convention — Visualization

*Spec source: doc 06. Figures are evidence — a wrong or misleading figure propagates as confidently as a wrong number. Governing principles: **accuracy is paramount**, and **a figure is a regenerable artifact, not a hand-made image.** Enforced by the **figure-reviewer**, now backed by the shipped `lib/figures/` machinery — `figure-io` (dual export + separate legend + publication style) and `okabe-ito-colors` (the color registry + the >8-category raising guard) — which carries these defaults mechanically for any project figure script that routes through it. The QC/descriptive and analysis plot templates that build on it remain phase-E work.*

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

## Publication-ready defaults

Figures default to publication quality: legible font sizes **at print scale**, no chartjunk, clear axis labels **with units**, appropriate aspect ratios, consistent typography. The `lib/` figure templates encode these defaults (a shared matplotlib style, imported by the project's figure scripts) so each figure starts from them rather than re-specifying them.

## Descriptive & cohort figures

The Stage 1 metadata characterization (the distribution of every variable, pairwise crosstabs, the cohort "Table 1") produces **first-class figures**, subject to every rule here — dual export, Okabe–Ito via the registry, the ≤8-category guard, render review. They are both publication deliverables and the lens that exposes class imbalance and confounding; the consequential ones are recorded as caveat findings (`conventions/statistics.md`; `conventions/findings.md` §2.6). Color the categorical design variables (sex, group, batch) through `state/color_registry.json` so a given level keeps its color from the very first cohort plot through every downstream figure.

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
| Figure provenance pinned; staleness tracked | findings-manager (staleness) + finding `provenance` |
