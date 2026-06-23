# Convention — Workflow State (`state/workflow.json`)

*The cross-cutting progress + gate contract for a project. Spec source: doc 02 (stages, gates, checkpoints). Written by the stage commands, read by the `status` command, and **read by `guard_findings.py`** to gate finding writes. One file, one source of truth for "where are we in the pipeline."*

## Why this file exists

The workflow has a hard ordering rule and a load-bearing gate (doc 02). Rather than infer progress from the incidental presence of files, the project records its position explicitly in `state/workflow.json`. `guard_findings.py` reads exactly one field from it (`integrity_gate.passed`) to block any finding that claims `integrity_signoff: true` / `status: validated` before the gate, so that side is enforced deterministically. The broader *no-Stage-4-analysis-before-the-gate* ordering is carried by the `stage4-explore` command precondition + orchestrator behavior — **not** a hook, because a single tool-use event can't cleanly tell exploratory analysis from legitimate Stage 3 loader/QC work (see `conventions/enforcement-map.md`).

## Schema

```jsonc
{
  "schema_version": "1",
  "current_stage": 0,            // highest stage reached: 0..6
  "science_done": false,         // Stage 0 → state/PROJECT.md written
  "metadata_done": false,        // Stage 1 → state/METADATA.md written + scientist confirmed
  "data_done": false,            // Stage 2 → state/DATA_DESCRIPTION.md written
  "integrity_gate": {            // Stage 3 — the hard precondition for any analysis
    "passed": false,             // ← the field guard_findings.py checks (finding-write gate)
    "signed_off_by": null,       // who signed off (the scientist)
    "date": null,                // YYYY-MM-DD of sign-off
    "data_version": null,        // the data_version the gate certifies
    "qc_report": null            // path to the QC report backing the gate
  },
  "environment": {               // the project Python environment (managed by setup-env)
    "mode": null,                // "project-uv" | "system" | null (undecided)
    "python_min": "3.11",        // the interpreter floor the Stage 3 gate enforces
    "interpreter": null,         // path to the venv python, or "system"
    "configured": false,         // a usable env (>= python_min) has been verified
    "declined": false,           // scientist declined the project-local install
    "updated": null              // YYYY-MM-DD of the last environment change
  },
  "updated": "YYYY-MM-DD"
}
```

## Rules

- **Seeded by `init`** with everything false / null and `current_stage: 0`.
- **Each stage command updates it** when its work advances: set the stage's `*_done` flag *if it has one* (Stages 0–2 do; Stage 3 flips `integrity_gate.passed`; Stages 4–6 are continuous loops with **no** `*_done` flag), **raise `current_stage`** to the stage's number (highest reached — monotonic, so the `status` dashboard advances through validation and reporting), and bump `updated`. A stage command must **refuse to run** if its preconditions (prior flags) are not met — defense in depth alongside the hooks.
- **`integrity_gate.passed` flips to `true` only inside Stage 3**, and only after the full integrity-gate checklist passes (doc 05) *and* the scientist signs off. It records the certified `data_version`. If the dataset changes (a new `data_version`), the gate is no longer valid: reset `passed` to `false` and re-run Stage 3.
- **`guard_findings.py`** blocks a finding write that claims `integrity_signoff: true` / `status: validated` when this file is absent or `integrity_gate.passed` is not `true` (absent ⇒ treated as not passed). It does **not** block analysis tool-calls; *no analysis before the gate* is carried by the `stage4-explore` command precondition + orchestrator behavior.
- A finding's `integrity_signoff` (conventions/findings.md) may be `true` only while `integrity_gate.passed` is `true` for the finding's `data_version`.
- **The `environment` block is written by `setup-env`** and is **advisory**: it records the chosen Python provisioning (`mode`), the floor (`python_min`), and whether the scientist declined a project-local install. It is **not** read by any hook. The Stage 3 command live-verifies a working interpreter ≥ `python_min` rather than trusting `configured`, so a stale flag can never unlock analysis on a broken env.

## Hook read (illustrative — `guard_findings.py`, on a finding write)

The guard reads exactly one field with stdlib `json` (no `jq`, no `bash` — so it runs identically on Windows/macOS/Linux):

```python
import json, sys
state = json.load(open("state/workflow.json", encoding="utf-8"))
passed = state.get("integrity_gate", {}).get("passed") is True
if claims_signoff_or_validated and not passed:
    sys.exit(2)  # a finding may not claim integrity_signoff/validated yet (doc 02.3)
```
