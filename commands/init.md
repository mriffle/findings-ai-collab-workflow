---
name: init
description: Scaffold a Findings Workflow project in the current directory — create the standard layout, seed the color registry, initialize the findings graph, and write the project CLAUDE.md.
argument-hint: "[project directory, default: current directory]"
---

# Initialize a Findings Workflow project

Scaffold the project structure for a new study in **the target directory** (`$ARGUMENTS` if given, otherwise the current working directory). The engine lives in the installed plugin; this step creates only the per-study data/derived-state layout and the project's standing instructions.

**Be idempotent and safe:** never overwrite or delete an existing file. If a file already exists, skip it and report it as "kept." At the end, print a summary of what was created vs kept, and the next step.

Plugin templates are referenced under `${CLAUDE_PLUGIN_ROOT}/templates/`.

## Steps

1. **Create the directory tree** (use `mkdir -p`; harmless if they exist):
   ```
   data/                  scripts/scratch/      results/      findings/
   state/                 scripts/promoted/     figures/      research/
                                                              reports/
   ```

2. **Seed `state/color_registry.json`** — if absent, copy `${CLAUDE_PLUGIN_ROOT}/templates/color_registry.json`. This carries the universal Okabe–Ito defaults; project-specific categories are added later, once `state/METADATA.md` exists. If present, keep it.

3. **Write the project `CLAUDE.md`** — if absent, copy `${CLAUDE_PLUGIN_ROOT}/templates/project-CLAUDE.md` to `./CLAUDE.md`. This is the mechanism by which the workflow's standing behavior reaches the orchestrator every session (the plugin cannot auto-inject it). If a `CLAUDE.md` already exists, **do not overwrite it** — instead show the user the template path and offer to merge the Findings Workflow section in.

4. **Initialize the findings graph** — if `findings/manifest.md` is absent, create it (format: `conventions/manifest.md`) as a Markdown file: a YAML frontmatter block then an empty findings table.
   ```markdown
   ---
   schema_version: 1
   generated: <today's date, YYYY-MM-DD>
   next_id: 1
   engine_version: 0.1.0
   ---

   # Findings manifest

   Derived index of the findings graph — regenerable from the finding files. One row per finding.

   | ID | Slug | Title | Status | Phase | Entities | Relationships | Updated | Data version |
   |----|------|-------|--------|-------|----------|---------------|---------|--------------|
   ```
   The findings-manager is its only writer. If `findings/exploration-log.md` is absent, create it with a heading:
   ```
   # Exploration log

   Append-only record of what was looked at and discarded — the multiplicity context that informs each finding's caveats (doc 03.6). One dated entry per exploratory thread.
   ```
   Also seed two empty Markdown indexes (formats: `conventions/script-registry.md`, `conventions/research-corpus.md`), each a YAML frontmatter block (`schema_version: 1`, `generated: <today>`) + a heading + an empty table:
   - `scripts/manifest.md` — heading `# Script registry`; table header `| Task | Path | Kind | Status | Provides | Uses | Seeded from | Description |`.
   - `research/manifest.md` — heading `# Research manifest`; table header `| Slug | Topic | Type | Status | Entities | Refs | Updated |`.

5. **Initialize the workflow state** — if `state/workflow.json` is absent, create it (schema: `conventions/workflow-state.md`):
   ```json
   { "schema_version": "1", "current_stage": 0, "science_done": false, "metadata_done": false, "data_done": false, "integrity_gate": { "passed": false, "signed_off_by": null, "date": null, "data_version": null, "qc_report": null }, "environment": { "mode": null, "python_min": "3.11", "interpreter": null, "configured": false, "declined": false, "updated": null }, "updated": "<today's date, YYYY-MM-DD>" }
   ```
   This is the single source of truth for pipeline position; the stage commands update it, and `guard_findings.py` reads `integrity_gate.passed` from it to gate finding writes (analysis ordering itself is enforced by the `stage4-explore` command precondition + orchestrator behavior).

6. **Mark `data/` read-only by convention** — if `data/README.md` is absent, create it stating: raw data here is immutable and read-only (a hook blocks writes); place the dataset and its metadata file here; everything in `results/` and `figures/` is regenerated from this plus a script.

## Offer to set up the Python environment (optional)

After scaffolding, briefly check whether this project has a usable Python (≥ 3.11) and offer `setup-env`: *"This workflow runs analysis in Python ≥ 3.11. I can set up a project-local environment now (or later) with `setup-env` — it detects an existing Python and, only if needed, asks before installing one **into this project**."* Do not run any installer here and do not block on this — `setup-env` owns the detection, the consent prompt, and the install. Python execution begins at **Stage 1** (the metadata examination runs validity checks, the cohort characterization, and confounding statistics in code), so the environment should be in place before Stage 1 — it can be set up now or any time before then.

## After scaffolding

Report the created/kept summary, then tell the scientist the workflow's first step:

> **Stage 0 — State the science.** Describe the study: the domain, what is being examined and why, the experimental design, and the scientific goals. I'll capture it to `state/PROJECT.md`, and that framing shapes everything that follows.

Do **not** proceed to analyze any data. The ordering rule is absolute: understand before analyzing, verify the read before exploring.
