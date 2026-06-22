# Convention — Visualization

*Spec source: doc 06. Figures are evidence — a wrong or misleading figure propagates as confidently as a wrong number. Governing principles: **accuracy is paramount**, and **a figure is a regenerable artifact, not a hand-made image.** Implemented by the `lib/` visualization machinery; enforced by the figure-reviewer and a dual-export check.*

## Accuracy & review

- Every generated figure is **reviewed as a rendered PNG**, not merely as the code that produced it. Code can be correct and the render still wrong (clipped labels, misleading axis, wrong color mapping, overplotting). The **figure-reviewer** inspects the actual render.
- Figure generation and review are a **generator/reviewer subagent pair**.
- A figure is not accepted until its render passes review.

## Output formats — dual export

Every visualization is saved in **both**:

- **SVG** — vector master, for editing and publication.
- **PNG at 300 DPI** — raster, the review and embedding target.

Both go to `figures/`. In matplotlib: `fig.savefig(base + ".svg")` and `fig.savefig(base + ".png", dpi=300)`. The finding's `figures` entry points at both plus the legend doc.

## Legends as separate documents

Render **legends as separate documents** (`figures/<name>.legend.md`) alongside each figure rather than only baking them into the image. This keeps figures clean, supports publication workflows where legends are typeset separately, and makes the figure's encoding explicit and reviewable.

## Publication-ready defaults

Figures default to publication quality: legible font sizes **at print scale**, no chartjunk, clear axis labels **with units**, appropriate aspect ratios, consistent typography. The `lib/` visualization library encodes these defaults (a shared matplotlib style) so every figure inherits them rather than re-specifying them.

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

The `lib/` plotting machinery **detects when a plot is about to exceed eight categorical colors and requires the script to choose an explicit strategy** rather than silently recycling colors.

## Figure provenance

Every figure records — and the finding that uses it pins — the producing **script (path + commit)**, the **data version**, and **parameters** (finding `provenance`). Because figures are regenerable from this, the staleness machinery (doc 03.8) covers them: if the data version or script changes, figures built on the old version are flagged for re-generation.

## Enforcement

| Rule | Enforced by |
|---|---|
| Render reviewed (PNG), not just code | **Figure-reviewer** |
| Dual export (SVG + 300 DPI PNG) + separate legend present | **Figure-reviewer** (+ optional hook checking the trio exists) |
| Okabe–Ito; category colors from the registry; consistency | **Figure-reviewer** + `lib/` reads `state/color_registry.json` |
| ≤8 categorical colors; explicit strategy beyond | `lib/` (raises) + **figure-reviewer** |
| Figure provenance pinned; staleness tracked | findings-manager (staleness) + finding `provenance` |
