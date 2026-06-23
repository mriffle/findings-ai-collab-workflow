---
name: figure-generation
description: >-
  How to render a publication-ready figure in a Findings Workflow project: read
  the color registry, apply the publication defaults, dual-export (SVG + 300 DPI
  PNG), render a separate legend image, and handle the >8-category problem.
  Use whenever generating a figure for a finding or report.
---

# Figure generation procedure

Authoritative rules: `conventions/visualization.md`. A figure is a **regenerable artifact**, produced by a parameterized script — never a hand-made image.

## Read the color registry first

Load `state/color_registry.json`. For every categorical dimension you plot, use the color the registry assigns that value — so a value keeps the **same color in every figure**. For a new categorical value not yet in the registry, assign the next unused color from `_palette.colors` (Okabe–Ito) and add it (with `scope: project`) so it stays consistent thereafter. Never pick ad-hoc colors.

## Publication defaults

Seed from the relevant `lib/` figure template (`${CLAUDE_PLUGIN_ROOT}/lib/`), or reuse the project's existing script for this plot type (one script per task). The templates encode these defaults and import a shared figure/style module. If working without one, apply: legible font sizes at print scale, no chartjunk, axis labels **with units**, an appropriate aspect ratio, consistent typography, and the **Okabe–Ito** palette (color-blind-safe; aim for grayscale-interpretable too).

## The >8-category rule

Color encodes **at most eight** categories. If you're about to exceed eight, do **not** add colors — pick an explicit strategy and note it in the figure caption:

1. **Facet / small multiples** — separate panels;
2. **A second channel** — color + shape or linetype;
3. **Group the long tail** — collapse minor categories into "other";
4. **Position / sequential** — for ordinal/numeric categories.

## Dual export + separate legend image

Write the figure and its legend to `figures/` with a shared base name. Render the **legend as its own figure** — a swatch key (categorical) or a colorbar (continuous) — and keep it **out of the plot**, because a legend baked into the axes routinely overlaps the data:

```python
fig.savefig(f"figures/{base}.svg")                   # vector master
fig.savefig(f"figures/{base}.png", dpi=300)          # review + embed target
legend_fig.savefig(f"figures/{base}.legend.svg")     # legend vector master
legend_fig.savefig(f"figures/{base}.legend.png", dpi=300)  # legend image
```
The `lib/figures/figure_io.save_figure` helper does both exports (pass `legend_fig=`). The figure's textual caption — what each axis/series/color encodes, units, n, and any grouping/strategy applied — goes in the finding's `figures[].caption`, not a separate document.

## Provenance

The producing script (path + commit), the data version, and parameters are recorded so the figure is regenerable and staleness-tracked.

## Handoff

Render, then route the **PNG** to the **figure-reviewer**. The figure is accepted only after the render passes review.
