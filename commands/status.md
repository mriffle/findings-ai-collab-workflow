---
name: status
description: "Show where the project stands in the Findings Workflow — pipeline position, integrity-gate state, and a breakdown of findings by status and phase."
---

# Workflow status

Render a concise dashboard of the project's position. Read the state files directly; do not guess.

## Gather

1. **Pipeline position** — read `state/workflow.json`: `current_stage`, the `*_done` flags, and `integrity_gate` (`passed`, `signed_off_by`, `date`, `data_version`).
2. **Findings breakdown** — read `findings/manifest.md` (the Markdown table): total findings, counts by `status` (candidate / under_exploration / validated / invalidated / superseded / closed), by `phase` (exploratory / confirmatory), and by `kind` (discovery / caveat — surface the cohort caveats separately). Note any findings flagged for re-review or staleness if recorded.
3. **State files present** — note whether `state/PROJECT.md`, `state/METADATA.md`, `state/DATA_DESCRIPTION.md` exist.
4. **Python environment** — read `state/workflow.json` `environment` (`mode`, `python_min`, `configured`, `declined`). For an accurate readout, quickly confirm the interpreter actually works (`./.venv/bin/python --version`, or system Python) rather than trusting the flag alone.

If `state/workflow.json` is absent, tell the scientist the project isn't initialized and to run `init`.

## Render

Show the staged pipeline with the current position and gate marked, e.g.:

```
Findings Workflow — status

  ✓ Stage 0  State the science          state/PROJECT.md
  ✓ Stage 1  Understand the metadata    state/METADATA.md
  ✓ Stage 2  Understand the data        state/DATA_DESCRIPTION.md
  ⛔ Stage 3  Loaders + QC  [GATE]        integrity_gate: NOT PASSED  ← analysis blocked here
    Stage 4  Explore ⇄ record           (locked until the gate passes)
    Stage 5  Independent validation
    Stage 6  Reporting

  Env: Python 3.11 (project .venv) ✓        ← or: "not set up — run setup-env"
  Findings: 0
```

Show the environment line near the pipeline: the verified interpreter version and where it lives (project `.venv`, or system), or — if no usable Python ≥ 3.11 — flag it and point to `setup-env` (this is the Stage 1 precondition — the metadata examination is the first stage that runs code).

When the gate has passed, show it as passed with the signer/date/data_version, mark Stage 4+ available, and print the findings breakdown:

```
  Findings: 17 total
    candidate 9 · under_exploration 4 · validated 3 · invalidated 1
    phase: exploratory 14 · confirmatory 3
```

End with the **single most useful next action** for the current position (e.g. "Next: run `stage3-loaders` to build and verify the loaders and pass the integrity gate").
