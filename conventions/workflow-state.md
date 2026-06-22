# Convention — Workflow State (`state/workflow.json`)

*The cross-cutting progress + gate contract for a project. Spec source: doc 02 (stages, gates, checkpoints). Written by the stage commands, read by the `status` command, and **checked by the integrity-gate hook** (Phase D). One file, one source of truth for "where are we in the pipeline."*

## Why this file exists

The workflow has a hard ordering rule and a load-bearing gate (doc 02). Rather than infer progress from the incidental presence of files, the project records its position explicitly in `state/workflow.json`. The integrity-gate hook reads exactly one field from it (`integrity_gate.passed`) to decide whether Stage 4 analysis is allowed, so the gate is enforced deterministically, not by trusting memory.

## Schema

```jsonc
{
  "schema_version": "1",
  "current_stage": 0,            // highest stage reached: 0..6
  "science_done": false,         // Stage 0 → state/PROJECT.md written
  "metadata_done": false,        // Stage 1 → state/METADATA.md written + scientist confirmed
  "data_done": false,            // Stage 2 → state/DATA_DESCRIPTION.md written
  "integrity_gate": {            // Stage 3 — the hard precondition for any analysis
    "passed": false,             // ← the single field the integrity-gate hook checks
    "signed_off_by": null,       // who signed off (the scientist)
    "date": null,                // YYYY-MM-DD of sign-off
    "data_version": null,        // the data_version the gate certifies
    "qc_report": null            // path to the QC report backing the gate
  },
  "updated": "YYYY-MM-DD"
}
```

## Rules

- **Seeded by `init`** with everything false / null and `current_stage: 0`.
- **Each stage command updates it** when its work completes (set the stage's flag, raise `current_stage`, bump `updated`). A stage command must **refuse to run** if its preconditions (prior flags) are not met — defense in depth alongside the hooks.
- **`integrity_gate.passed` flips to `true` only inside Stage 3**, and only after the full integrity-gate checklist passes (doc 05) *and* the scientist signs off. It records the certified `data_version`. If the dataset changes (a new `data_version`), the gate is no longer valid: reset `passed` to `false` and re-run Stage 3.
- **The integrity-gate hook** (Phase D) blocks analysis tool-calls when this file is absent or `integrity_gate.passed` is not `true`. Absent ⇒ treated as not passed.
- A finding's `integrity_signoff` (conventions/findings.md) may be `true` only while `integrity_gate.passed` is `true` for the finding's `data_version`.

## Hook read (illustrative)

```bash
passed=$(jq -r '.integrity_gate.passed // false' state/workflow.json 2>/dev/null)
[ "$passed" = "true" ] || { echo "Integrity gate not passed — Stage 4 analysis is blocked (doc 02.3)." >&2; exit 2; }
```
