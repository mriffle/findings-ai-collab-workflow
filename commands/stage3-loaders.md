---
name: stage3-loaders
description: "Stage 3 — Loaders, pairing, QC [INTEGRITY GATE]. Build and verify tested data+metadata loaders that pair every sample exactly, verify the loaded data against the source, produce QC, and pass the integrity gate. Unlocks all analysis."
---

# Stage 3 — Loaders, pairing, QC  **[INTEGRITY GATE]**

**Precondition — prior stage:** `state/workflow.json` shows `data_done: true`. If not, run `stage2-data` first and say so.

**Precondition — Python environment (re-verify).** Python execution began back at Stage 1 (`stage1-metadata`), so the environment hard gate lives there. But this is the **integrity gate** — too important to trust an inherited or stale environment — so **re-verify** a working Python ≥ 3.11 before any loader work (do not trust `state/workflow.json` `environment.configured` — check the interpreter): prefer the project venv (`./.venv/bin/python` on Unix, `.\.venv\Scripts\python.exe` on Windows), else a project-local/`PATH` `uv`, else a system `python3`/`python`/`py -3`, and confirm it reports ≥ 3.11. If none is usable, **stop and run `setup-env`** (which detects, transparently asks consent, and installs Python ≥ 3.11 *into the project* if approved). If the scientist previously declined the project-local install, tell them to install Python ≥ 3.11 themselves and re-run. Do not start loader work without a verified interpreter.

This is the workflow's **hardest precondition and most important stage**. Correctness is upstream of everything: a silent loader error is common-mode and defeats independent validation (the verifier reads the same data through the same loader, so both sides agree on a falsehood). **No Stage 4 analysis may begin until this gate passes** — the `stage4-explore` command precondition + orchestrator behavior enforce it (and `guard_findings.py` blocks any finding claiming sign-off / `validated` before the gate); do not try to work around it.

Standing rule for this stage and the whole workflow: **assume nothing, verify everything, fail loud.** A loader that silently drops rows, coerces a type, or mismatches a sample is worse than one that crashes.

## Build robust, tested loaders

Build loaders (in `scripts/`, seeding from the `lib/` loader template where one exists — otherwise write to `conventions/coding.md` + `conventions/correctness.md`; `lib/` templates arrive in phase E) that load **both the data and the metadata** and **correctly pair every sample with its metadata description**. Hold this code to the maximum (doc 05): type hints + a type checker, lint/format, seeds recorded, logging over print, fail-loud on shape mismatch or silent NA coercion. Dispatch the **coder** and **code-reviewer** agents for this where available; otherwise write and self-review against `conventions/` directly.

## Data loading is two obligations — both must pass

**A. Test the loader.**
- Unit tests with hand-verified fixtures.
- Property/invariant tests: loading preserves source counts; no value appears that wasn't in the source.
- A **planted-truth fixture**: synthetic data with a known effect the pipeline must recover.
- Edge cases: empty, all-missing, single sample, duplicate IDs, ties.

**B. Verify the loaded data on the real file.**
- Counts reconcile (rows/cols vs source).
- Random-cell **spot reconciliation** against the raw source.
- Orientation **confirmed, not assumed**.
- Dtypes explicit and correct (no silent string↔numeric coercion).
- Value ranges plausible.
- Identifier integrity (no truncation/reformatting — watch the spreadsheet-corruption traps).
- Missing-value encoding made explicit.
- Transformation/normalization state confirmed, and the loader **records the matching `scale`** on its `Dataset` (`linear`/`log2`/`glog2`/`zscore`, per Stage 2) — this is what lets the normalization/batch templates refuse scale-incorrect steps.
- **Sample↔metadata pairing complete and exact** — every sample matched once, no orphans/duplicates, counts reconcile on both sides. (Fuzzy matches are a **human checkpoint**: have the scientist confirm the join resolution.)
- **Experimental/control classification carried and verified** — control samples (pools, references, standards, blanks) are real samples and pair like any other; carry the Stage-1 experimental-vs-control label onto the loaded samples and verify the per-class counts reconcile, so the **experimental subset** the analysis will run on is well-defined and certified at this gate (not re-decided ad hoc downstream).

Where feasible, **derive critical quantities two independent ways and reconcile** — especially the data read itself, here at the data boundary, before any analysis.

## QC report

Produce descriptive QC so collaborators can trust the data, and save a QC report under `reports/`. The backbone is seven shipped `lib/figures/` templates — **seed each into `scripts/scratch/` and adapt**; routing through them carries the registry colors, the dual-export, and the separate legend image for free. The optional features×samples **missingness map** (a clustered presence heatmap) has no template — add it write-from-scratch if a structural view is useful (the `missingness` template covers the per-feature completeness + MNAR views). These are **descriptive sanity checks, not findings** — findings are Stage 4.

**Normalization is applied here for the first time** (the first point analysis normalization touches the data): seed from the `lib/` `normalize` template, use the method confirmed in Stage 2 (median by default), and let the `scale` tag travel with the `Dataset` so each plot gets the scale it needs.

**Batch correction is *previewed* here too — but only when the design has a real batch axis with ≥ 2 batches** (one plate/batch has nothing to correct: skip it and omit every batch-corrected panel below). The preview is **batch-label-only** (`batch-correct-combat`; never hand ComBat the covariate of interest) and exists only to *show the effect* of correction — does control/sample clustering tighten, does CV drop — for the scientist to weigh at sign-off. It is **not** the committed analysis decision (whether to correct for *testing*, run both-ways with batch-as-covariate); that stays in Stage 4.

**Label the processing state on every QC figure.** Each figure must state whether it shows **raw**, **normalized**, or **batch-corrected** data, and on what scale — via panel labels on the multi-state plots (CV, abundance box plots) and the title/caption on the single-state plots — so a reader never has to guess which processing stage a figure depicts.

**The required scale differs per plot — respect it or the templates refuse/mislead:**

- **Identification depth (`id-depth`) — on the RAW LINEAR matrix.** Detected features (finite & > 0) per run, one stacked panel per feature level — **protein over precursor** — bars drawn in **acquisition order** (loader `order_by`) so a failing/low-ID run or a drifting block stands out as a short bar even when surviving intensities look normal. The first-look QC: how deep did each run go. Color the bars by the **sample-class** column (experimental vs pooled-QC / reference) so controls and experimentals are distinguishable in one figure; draw a **reference-median line over the experimental subset** (`reference_mask`) so a dropout reads against where the real samples sit. **Hard-refuses a non-linear scale** (`IdDepthScaleError`) — detection is a raw-data property, so feed it the unnormalized matrix (before the normalization step below). Controls are shown *with* the experimental samples here (distinguished by bar color), the same labeled exception as sample-correlation.
- **Missingness / completeness (`missingness`) — on the RAW LINEAR matrix.** The per-feature complement to `id-depth`: a two-panel figure of (1) a **completeness curve** — features retained vs the required detection fraction, optionally overlaid per sample class (pools should sit higher) — and (2) an **MNAR diagnostic** — per-feature detection rate vs mean log2 abundance (hexbin + binned-median trend + Pearson r). A positive trend means low-abundance features are left-censored (MNAR), which **drives the imputation-method choice** (left-censored MinProb/QRILC vs mean/median/KNN — the Stage-2 decision, now made on evidence). **Hard-refuses a non-linear scale** (`MissingnessScaleError`) — run it on the unnormalized matrix. Controls shown with experimentals via the completeness-curve color (the same labeled exception).
- **Dynamic range / rank-abundance (`dynamic-range`) — on the RAW LINEAR matrix.** Features ranked by abundance vs log2 abundance: the quantified dynamic range (orders of magnitude) and whether a few hyper-abundant features dominate (albumin / contaminants pile at the top). Run the whole-cohort **median + IQR band** on the **experimental subset** (subset upstream, like CV/PCA), and use `class_by` for the cross-class comparison (do pools cover the same range? — the labeled exception). At QC leave `highlight_features` empty (or mark known **contaminants**); the named **proteins of interest** are a **downstream** annotation (Stage 4 — domain targets / DE hits), which re-renders this same plot as a results figure. **Hard-refuses a non-linear scale** (`DynamicRangeScaleError`).
- **Abundance box plots (`abundance-boxplot`) — on the NORMALIZED LOG matrix.** One stacked panel per processing state — **raw → normalized → batch-corrected** (omit the batch-corrected panel when no correction was previewed — single batch) — each a box per sample over its feature abundances, so you read the effect of each step top-to-bottom: raw per-sample medians wander (loading / depth), a good normalization flattens them to a common level, batch correction removes residual batch shift. Order the samples by **acquisition order** upstream (loader `order_by`); boxes are colored by position so run-order drift shows as a left-to-right gradient the normalized panel should erase. **Warns on a non-log scale** (a few abundant features stretch every box on linear); per-panel y-labels come from the `scale` tag. Render controls on a **separate** call/figure (like CV/PCA). Optional annotation stripes (e.g. batch, sample type) via the registry.
- **CV distributions (`cv-plot`) — on the LINEAR matrix.** Overlay per-feature CV (std/mean) across the processing states **raw → normalized → batch-corrected** — a drop in median CV is the evidence a step reduced nuisance variance. Two things to get right:
  - **Scale (the template hard-refuses log — `CVScaleError`).** Raw and median-normalized data are already linear. The **batch-corrected** matrix is log (ComBat runs on log), so **de-log it with `normalize.to_linear`** before adding its curve — the batch-corrected CV is read on the strictly-positive de-logged values (a documented approximation: ComBat optimizes on the log scale, we read CV on its linear image). Omit the batch-corrected curve when no correction was previewed (single batch). If the scientist chose a **non-linear normalization** (MAD/VSN → zscore/glog2), the normalized state isn't CV-able either — overlay raw alone, or skip CV.
  - **Experimental, then each control type separately.** Render the **experimental** subset as one overlay (raw → normalized → batch-corrected). Then render **one CV figure per control type** (pooled-QC, reference, standard, …), each overlaying the same states — control CV is a distinct reading (pool CV = technical-reproducibility / instrument precision) and different control types are not comparable, so they never share an axis with each other or with the experimental overlay (controls separate).
- **Sample correlation (`sample-correlation`) — on the NORMALIZED LOG matrix.** A hierarchically-clustered sample×sample heatmap: the design sanity check (do experimental samples cohere, do controls self-cluster). **Controls are shown *with* the experimental samples here, distinguished by a Sample Type annotation stripe** — a deliberate exception to "render controls separately", because the cross-class clustering is the deliverable. Add annotation stripes for the design factors (group, sex, batch). Default **Pearson**, which **warns on a non-log scale** (abundant-feature dominance drags r down) — so use the log matrix, or **Spearman** (rank/scale-invariant) to compare on linear.
- **PCA (`pca-plot`) — on the LOG matrix.** PC1/PC2 + PC3/PC4 with marginal tests; **warns on a non-log scale**. Produce it as a **processing-state series** and with **two colorings**:
  - **State series — `raw → normalized → batch-corrected`.** One PCA per state (raw = log2 of the *unnormalized* matrix; normalized = log2 of the normalized matrix; batch-corrected = the ComBat output, already log) so the scientist reads the *effect* of each step — do the **control samples cluster more tightly** after normalization and again after batch correction, does a batch-driven split collapse. Omit the batch-corrected panel when no correction was previewed (single batch).
  - **Colorings.** Color one series by the **batch axis** to make batch structure visible (this is what the state series is read against). Color another by **sample class with a distinct color per control type** — experimental, pooled-QC, reference, standard, … as *separate* registry categories (the >8-category guard applies), **not** a single control-vs-experimental split — so each control kind's placement in the projection is legible. Grey out reference samples only where a plain background is wanted; for the class coloring we *want* the control colors visible.

Then run `assess_batch_confounding` (the `batch-correct-combat` template) against the covariates of interest — surface any batch↔biology confound to the scientist as part of QC sign-off; if a confound or severe imbalance surfaced here (or in the Stage 1 characterization) is not already captured as a caveat finding, have the findings-manager record it (`kind: caveat`) now. The batch correction shown above is a **QC preview only** — batch-label-only, to *display the effect* of correction so it informs sign-off. The **committed** decision — whether to correct for *significance testing*, run **both-ways** with **batch-as-covariate** — belongs to Stage 4, not here; a confounded batch axis makes even the preview suggestive-not-decisive (note it in the caveat). (Detailed control-rendering rules: `conventions/visualization.md`.)

## The integrity gate — pass criteria

The gate passes only when **all** hold:

1. Loader tests pass (unit, property, planted-truth, edge cases).
2. The loaded data is verified against the source (every item in obligation B).
3. Sample↔metadata pairing is complete and exact.
4. **The scientist signs off.** Present the verification results and QC — including the experimental/control classification (the analysis subset) and any confounds or imbalances surfaced (now recorded as caveat findings); get explicit sign-off.

## On pass

Update `state/workflow.json` `integrity_gate`:
```json
{ "passed": true, "signed_off_by": "<scientist>", "date": "<YYYY-MM-DD>", "data_version": "<stamp>", "qc_report": "reports/<qc-report>.md" }
```
Set `current_stage: 4`, bump `updated`. Record the certified `data_version` — if the dataset later changes, this gate is invalidated and must be re-run.

**Settle the metadata caveats.** The caveat findings recorded in Stage 1 (`kind: caveat`) rest on the sample↔metadata pairing this gate just certified — dispatch the findings-manager to set their `integrity_signoff: true` for this `data_version`, so the cohort's imbalances and confounds carry into Stage 4 analysis and Stage 6 reporting as trustworthy, attachable caveats.

Then tell the scientist: **the integrity gate has passed; Stage 4 exploration is now unlocked** (`stage4-explore`). Findings recorded from here may set `integrity_signoff: true` for this `data_version`.

## On fail

Do not flip the gate. Report exactly which checks failed and what must be fixed. Analysis stays blocked.
