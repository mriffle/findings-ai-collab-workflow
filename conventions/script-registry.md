# Convention — The Script Registry

*How a project's analysis code is kept single-sourced and DRY. Source of truth: a metadata header in each script. Derived index: `scripts/manifest.md`. Owned by the coder; enforced by the code-reviewer. Spec basis: the repository conventions (doc 05.1) + the project's one-script-per-task and reuse rules.*

## Two rules this enforces

1. **One script per task.** Once the project has a script for an analysis operation (PCA, differential abundance, a QC plot, a classifier…), that is **the** script for it. We never want two scripts doing the same thing drifting apart. Variations are handled by **parameters to the one script**, not by new scripts. (A `lib/` template is a central seed, not a project copy — it doesn't count.)
2. **Reuse shared code.** A function implemented once is used everywhere. Shared logic lives in **one module** and is **imported**, never copy-pasted into multiple scripts.

## What "a task" is

A **task** is an analysis *operation*, not a specific run of it. "PCA plot" is one task → one parameterized script (which data, which coloring are params). "Differential abundance" is one task → one script parameterized by contrast — **not** one script per contrast. If you're tempted to write `pca_v2.py` or `de_drugB.py`, you instead **parameterize and reuse the existing script**.

## Project script structure

Project scripts (in `scripts/`, flowing `scratch/` → `promoted/`) come in two kinds:

- **`analysis`** — an entry-point script for one task. Exactly one per task. Produces results/figures. Imports shared modules.
- **`module`** — shared library code for the project (verified loaders, normalization, the figure/color machinery). Provides reusable functions; imported by analysis scripts. Lives in a project package, e.g. `scripts/promoted/common/`.

**Invariant:** a **promoted** script may only import other **promoted** modules — so a finding's computation never depends on unreviewed code. (The finding links to the promoted analysis entry-point; the modules it imports must be promoted too.)

## Source of truth: the per-script header

Every project script declares its own metadata in a module-level `__script_meta__` dict — this is the source of truth (consistent with "code is the source of truth"):

```python
__script_meta__ = {
    "task": "pca",                     # analysis operation key; unique among analysis scripts. null for a module.
    "kind": "analysis",                # "analysis" | "module"
    "provides": [],                    # exported reusable symbols (for modules); [] for analysis scripts
    "uses": ["common.io", "common.figures"],  # project modules this script imports
    "seeded_from": {"template": "pca-plot", "version": "0.3"},  # lib/ template lineage; null if from scratch
    "description": "PCA scatter of samples, colored from the registry.",
}
```

## Derived index: `scripts/manifest.md`

A Markdown file (YAML frontmatter + one table), **regenerated from the per-script headers** — same model as the findings manifest. It is the at-a-glance "what scripts exist, what they provide/use" view, written/read by LLM agents (so Markdown, not JSON — see the format convention in the engine `CLAUDE.md`).

```markdown
---
schema_version: 1
generated: 2026-06-22
---

# Script registry

Derived index of project scripts — source of truth is each script's `__script_meta__` header. One analysis script per task; shared code lives in modules and is imported, never duplicated.

| Task | Path | Kind | Status | Provides | Uses | Seeded from | Description |
|------|------|------|--------|----------|------|-------------|-------------|
| pca | scripts/promoted/pca.py | analysis | promoted | — | common.io, common.figures | pca-plot@0.3 | PCA scatter colored from the registry |
| differential-abundance | scripts/promoted/de.py | analysis | promoted | — | common.io, common.stats | de-moderated@0.2 | Moderated-model DE, parameterized by contrast |
| — | scripts/promoted/common/io.py | module | promoted | load_data, pair_metadata | — | loader@0.1 | Verified data+metadata loader |
| — | scripts/promoted/common/figures.py | module | promoted | save_dual, apply_style, color_for, guard_categories | — | figure-base@0.1 | Dual export, color registry, >8-category guard |
```

Use `—` for empty cells. The registry omits nothing heavy (scripts are few); it's a flat table.

## Governance

- **Coder** (owns the registry): **before writing anything**, consult `scripts/manifest.md` —
  - is there already an `analysis` script for this task? → **reuse/extend it** (add a parameter; do not create a second);
  - need a function? → check `Provides`; **import it**. If it's reusable and not yet shared, add it to a `module` (don't inline-duplicate).
  Then set the script's `__script_meta__` header and update `scripts/manifest.md`.
- **Code-reviewer** (enforces, before promotion): no duplicate task; no reimplementation of a symbol already in a module's `provides`; shared code factored into a module; the **promoted-imports-only** invariant holds; and the registry row matches the script's header. FAIL on any violation — a duplicate or copy-pasted function is not promotable.
- **Regeneration:** rebuild `scripts/manifest.md` by scanning the `__script_meta__` headers, exactly as the findings manifest is rebuilt from finding files.

## Relationship to findings

A finding's `provenance.script.path` points at the **promoted analysis entry-point** for the operation that produced its numbers (plus `params`). Many findings may come from the same script with different params — that's reuse working as intended; each finding pins the script commit + its params + `seeded_from` lineage.
