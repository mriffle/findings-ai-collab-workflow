# lib/

A library of vetted, tested **template scripts** for the analyses and figures the workflow uses (spec docs 04.3, 05, 06). Python only.

**The model: templates, not a called library.** These scripts are *seeds*. When an analysis or figure is needed, the orchestrator **copies the relevant template into the user's project `scripts/scratch/` on demand**, adapts it for that study's data and question, and it flows through the code/stats/figure reviewer gates into `scripts/promoted/`. The finding then links to that **project-local promoted script** — pinned by its commit + the project lockfile, so a finding is fully regenerable from the project itself, independent of which plugin version is installed. Central template updates do **not** silently change anyone's local processing (by design).

Two halves:

- **Statistical / analysis templates** — differential abundance (limma/MSstats-style moderated models), nonparametric tests, feature selection, leakage-safe cross-validated classifiers, regression, dimensionality reduction, and tested data loaders.
- **Visualization templates** — descriptive/QC plots and the publication-ready figure machinery (dual SVG + 300 DPI PNG export, separate legend docs, the Okabe–Ito palette, the color registry, the >8-category guard).

**Why templates (and how rigor is kept):** models get test assumptions and missingness handling wrong in ways that look fine, so starting from a vetted example that already encodes the correct, leakage-safe, convention-following approach beats writing from scratch. Because each project owns its adapted copy, a fix here does **not** auto-propagate, and a careless edit could reintroduce a hazard — so rigor is kept by: (a) the **reviewer gates** on every adapted script before it can back a finding; (b) writing templates so the **dangerous structure is hard to break** (preprocessing stays inside CV folds, the figure guards fire, etc.) and clearly marked as such; and (c) recording **template lineage** in each finding's `provenance.seeded_from` (which template + version it was adapted from), so derived scripts can be flagged when a template is later corrected.

Each template is held to the maximum (typed, tested, linted, seeds recorded) so it is a sound starting point. Templates are referenced for copying via `${CLAUDE_PLUGIN_ROOT}/lib/...`.

**Status — building this (phase E):** no templates ship yet; `lib/` is the remaining build work. The build plan, the constraints already decided, and the open design questions to settle first are in the engine `CLAUDE.md` → *Next: building `lib/`*. Governance for the project-local scripts these templates seed is in `conventions/script-registry.md`.
