# lib/

The vetted, tested **standard library** the analysis agents prefer over generating fresh code (spec docs 04.3, 05, 06). Python only.

Two halves:

- **Statistical boilerplate** — differential abundance (limma/MSstats-style moderated models), nonparametric tests, feature selection, leakage-safe cross-validated classifiers, regression, dimensionality reduction.
- **Visualization** — standard descriptive/QC plots and the publication-ready figure machinery (dual SVG + 300 DPI PNG export, separate legend docs, the Okabe–Ito palette, the color registry, the >8-category strategy).

**Why a shared library:** models get test assumptions and missingness handling wrong in ways that look fine. A wrong default here is wrong in *every* project, so `lib/` is held to the maximum (typed, tested, linted, seeds recorded), is scientifically reviewed, and is **version-recorded** — every finding pins the `lib/` version that produced its numbers (doc 05).

Referenced from agents/hooks via `${CLAUDE_PLUGIN_ROOT}/lib/...`.
