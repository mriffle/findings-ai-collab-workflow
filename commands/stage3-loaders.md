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

**Normalization is applied here for the first time** (the first point analysis normalization touches the data): seed from the `lib/` `normalize` template, use the method confirmed in Stage 2 (median by default), and let the `scale` tag travel with the `Dataset` so each plot gets the scale it needs. **The required scale differs per plot — respect it or the templates refuse/mislead:**

- **Identification depth (`id-depth`) — on the RAW LINEAR matrix.** Detected features (finite & > 0) per run, one stacked panel per feature level — **protein over precursor** — bars drawn in **acquisition order** (loader `order_by`) so a failing/low-ID run or a drifting block stands out as a short bar even when surviving intensities look normal. The first-look QC: how deep did each run go. Color the bars by the **sample-class** column (experimental vs pooled-QC / reference) so controls and experimentals are distinguishable in one figure; draw a **reference-median line over the experimental subset** (`reference_mask`) so a dropout reads against where the real samples sit. **Hard-refuses a non-linear scale** (`IdDepthScaleError`) — detection is a raw-data property, so feed it the unnormalized matrix (before the normalization step below). Controls are shown *with* the experimental samples here (distinguished by bar color), the same labeled exception as sample-correlation.
- **Missingness / completeness (`missingness`) — on the RAW LINEAR matrix.** The per-feature complement to `id-depth`: a two-panel figure of (1) a **completeness curve** — features retained vs the required detection fraction, optionally overlaid per sample class (pools should sit higher) — and (2) an **MNAR diagnostic** — per-feature detection rate vs mean log2 abundance (hexbin + binned-median trend + Pearson r). A positive trend means low-abundance features are left-censored (MNAR), which **drives the imputation-method choice** (left-censored MinProb/QRILC vs mean/median/KNN — the Stage-2 decision, now made on evidence). **Hard-refuses a non-linear scale** (`MissingnessScaleError`) — run it on the unnormalized matrix. Controls shown with experimentals via the completeness-curve color (the same labeled exception).
- **Dynamic range / rank-abundance (`dynamic-range`) — on the RAW LINEAR matrix.** Features ranked by abundance vs log2 abundance: the quantified dynamic range (orders of magnitude) and whether a few hyper-abundant features dominate (albumin / contaminants pile at the top). Run the whole-cohort **median + IQR band** on the **experimental subset** (subset upstream, like CV/PCA), and use `class_by` for the cross-class comparison (do pools cover the same range? — the labeled exception). At QC leave `highlight_features` empty (or mark known **contaminants**); the named **proteins of interest** are a **downstream** annotation (Stage 4 — domain targets / DE hits), which re-renders this same plot as a results figure. **Hard-refuses a non-linear scale** (`DynamicRangeScaleError`).
- **Abundance box plots (`abundance-boxplot`) — on the NORMALIZED LOG matrix.** One stacked panel per processing state — **raw → normalized → batch-corrected** — each a box per sample over its feature abundances, so you read the effect of each step top-to-bottom: raw per-sample medians wander (loading / depth), a good normalization flattens them to a common level, batch correction removes residual batch shift. Order the samples by **acquisition order** upstream (loader `order_by`); boxes are colored by position so run-order drift shows as a left-to-right gradient the normalized panel should erase. **Warns on a non-log scale** (a few abundant features stretch every box on linear); per-panel y-labels come from the `scale` tag. Render controls on a **separate** call/figure (like CV/PCA). Optional annotation stripes (e.g. batch, sample type) via the registry.
- **CV distributions (`cv-plot`) — on the LINEAR matrix.** Overlay per-feature CV (std/mean) across an ordered set of states. Canonical QC use: a **normalization-state comparison on the experimental subset** — unnormalized → normalized (→ batch-corrected, if previewed) — where a drop in median CV is the evidence normalization reduced nuisance variance. CV is ill-defined on a log scale and the template **hard-refuses it** (`CVScaleError`), so feed it the linear data. Render a **separate pool/control CV panel** — pool CV is the technical-reproducibility / instrument-precision readout, a distinct reading kept off the experimental overlay (controls separate).
- **Sample correlation (`sample-correlation`) — on the NORMALIZED LOG matrix.** A hierarchically-clustered sample×sample heatmap: the design sanity check (do experimental samples cohere, do controls self-cluster). **Controls are shown *with* the experimental samples here, distinguished by a Sample Type annotation stripe** — a deliberate exception to "render controls separately", because the cross-class clustering is the deliverable. Add annotation stripes for the design factors (group, sex, batch). Default **Pearson**, which **warns on a non-log scale** (abundant-feature dominance drags r down) — so use the log matrix, or **Spearman** (rank/scale-invariant) to compare on linear.
- **PCA (`pca-plot`) — on the NORMALIZED LOG matrix.** PC1/PC2 + PC3/PC4 with marginal tests. **Color by the batch axis** to make batch structure visible (grey out reference samples); it **warns on a non-log scale**.

Then run `assess_batch_confounding` (the `batch-correct-combat` template) against the covariates of interest — surface any batch↔biology confound to the scientist as part of QC sign-off; if a confound or severe imbalance surfaced here (or in the Stage 1 characterization) is not already captured as a caveat finding, have the findings-manager record it (`kind: caveat`) now. Do **not** batch-correct here; correction (batch-label-only, both-ways reporting) is a Stage 4 analysis decision. (Detailed control-rendering rules: `conventions/visualization.md`.)

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
