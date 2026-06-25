---
name: stage3-loaders
description: "Stage 3 — Loaders, pairing, QC [INTEGRITY GATE]. Build and verify tested data+metadata loaders that pair every sample exactly, verify the loaded data against the source, produce QC, and pass the integrity gate. Unlocks all analysis."
---

# Stage 3 — Loaders, pairing, QC  **[INTEGRITY GATE]**

**Precondition — prior stage:** `state/workflow.json` shows `data_done: true`. If not, run `stage2-data` first and say so.

**Precondition — Python environment (hard gate).** This is the first stage that *executes* Python, so a usable interpreter must exist before any loader work. **Live-verify** a working Python ≥ 3.11 (do not trust `state/workflow.json` `environment.configured` — check the interpreter): prefer the project venv (`./.venv/bin/python` on Unix, `.\.venv\Scripts\python.exe` on Windows), else a project-local/`PATH` `uv`, else a system `python3`/`python`/`py -3`, and confirm it reports ≥ 3.11. If none is usable, **stop and run `setup-env`** (which detects, transparently asks consent, and installs Python ≥ 3.11 *into the project* if approved). If the scientist previously declined the project-local install, tell them to install Python ≥ 3.11 themselves and re-run. Do not start loader work without a verified interpreter.

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

Where feasible, **derive critical quantities two independent ways and reconcile** — especially the data read itself, here at the data boundary, before any analysis.

## QC report

Produce descriptive QC so collaborators can trust the data: abundance boxplots across samples, PCA, missingness maps, correlation structure (figures dual-exported with legends, colored via the registry). Save a QC report under `reports/`.

QC plots (boxplots, PCA) are normally read on **normalized, log-scale** data, so this is the first point the analysis normalization is applied: seed from the `lib/` `normalize` template, use the method confirmed in Stage 2 (median by default), and let the `scale` tag travel with the `Dataset`. **Color PCA/QC by the batch axis** to make batch structure visible, and run `assess_batch_confounding` (the `batch-correct-combat` template) against the covariates of interest — surface any batch↔biology confound to the scientist as part of QC sign-off; if a confound or severe imbalance surfaced here (or in the Stage 1 characterization) is not already captured as a caveat finding, have the findings-manager record it (`kind: caveat`) now. Do **not** batch-correct here; correction (batch-label-only, both-ways reporting) is a Stage 4 analysis decision.

## The integrity gate — pass criteria

The gate passes only when **all** hold:

1. Loader tests pass (unit, property, planted-truth, edge cases).
2. The loaded data is verified against the source (every item in obligation B).
3. Sample↔metadata pairing is complete and exact.
4. **The scientist signs off.** Present the verification results and QC — including any confounds or imbalances surfaced (now recorded as caveat findings); get explicit sign-off.

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
