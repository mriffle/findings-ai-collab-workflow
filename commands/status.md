---
name: status
description: "Show where the project stands in the Findings Workflow — pipeline position, integrity-gate state, and a breakdown of findings by status and phase."
---

# Workflow status

Render a concise dashboard of the project's position. Read the state files directly; do not guess.

## Gather

1. **Pipeline position** — read `state/workflow.json`: `current_stage`, the `*_done` flags, and `integrity_gate` (`passed`, `signed_off_by`, `date`, `data_version`).
2. **Findings breakdown** — read `findings/manifest.md` (the Markdown table): total findings, counts by `status` (candidate / under_exploration / validated / invalidated / superseded / closed) and by `phase` (exploratory / confirmatory). Note any findings flagged for re-review or staleness if recorded.
3. **State files present** — note whether `state/PROJECT.md`, `state/METADATA.md`, `state/DATA_DESCRIPTION.md` exist.

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

  Findings: 0
```

When the gate has passed, show it as passed with the signer/date/data_version, mark Stage 4+ available, and print the findings breakdown:

```
  Findings: 17 total
    candidate 9 · under_exploration 4 · validated 3 · invalidated 1
    phase: exploratory 14 · confirmatory 3
```

End with the **single most useful next action** for the current position (e.g. "Next: run `stage3-loaders` to build and verify the loaders and pass the integrity gate").
