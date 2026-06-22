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

4. **Initialize the findings graph** — if `findings/manifest.json` is absent, create it as:
   ```json
   { "schema_version": "1", "lib_version": "0.1.0", "generated": "<today's date, YYYY-MM-DD>", "next_id": 1, "findings": [] }
   ```
   If `findings/exploration-log.md` is absent, create it with a heading:
   ```
   # Exploration log

   Append-only record of what was looked at and discarded — the multiplicity context that informs each finding's caveats (doc 03.6). One dated entry per exploratory thread.
   ```

5. **Initialize the workflow state** — if `state/workflow.json` is absent, create it (schema: `conventions/workflow-state.md`):
   ```json
   { "schema_version": "1", "current_stage": 0, "science_done": false, "metadata_done": false, "data_done": false, "integrity_gate": { "passed": false, "signed_off_by": null, "date": null, "data_version": null, "qc_report": null }, "updated": "<today's date, YYYY-MM-DD>" }
   ```
   This is the single source of truth for pipeline position; the stage commands update it and the integrity-gate hook reads `integrity_gate.passed` from it.

6. **Mark `data/` read-only by convention** — if `data/README.md` is absent, create it stating: raw data here is immutable and read-only (a hook blocks writes); place the dataset and its metadata file here; everything in `results/` and `figures/` is regenerated from this plus a script.

## After scaffolding

Report the created/kept summary, then tell the scientist the workflow's first step:

> **Stage 0 — State the science.** Describe the study: the domain, what is being examined and why, the experimental design, and the scientific goals. I'll capture it to `state/PROJECT.md`, and that framing shapes everything that follows.

Do **not** proceed to analyze any data. The ordering rule is absolute: understand before analyzing, verify the read before exploring.
