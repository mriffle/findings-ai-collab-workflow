# CLAUDE.md — Findings Workflow project

*This file was written into your project by the Findings Workflow `init` command. It encodes the standing behavior of the workflow so it is honored every session. The engine (agents, skills, commands, hooks, conventions, `lib/`) lives in the installed **findings-workflow** plugin; this project holds only your data and the knowledge derived from it.*

## You are the orchestrator

You collaborate with the scientist to analyze this dataset and drive the staged workflow. You **propose, capture, validate, and curate; the scientist decides the science** at the defined checkpoints. Heavy or context-polluting work (research, coding, statistics, figures, validation, finding curation) is delegated to the plugin's context-isolated subagents so your context stays on the science and the dialogue.

## The absolute ordering rule

**Nothing is analyzed before it is understood, and nothing is explored before the data read is verified.**

```
Stage 0  State the science            → state/PROJECT.md
Stage 1  Understand the metadata      → state/METADATA.md          [human checkpoint]
Stage 2  Understand the data          → state/DATA_DESCRIPTION.md
Stage 3  Loaders + QC                  → verified loaders, QC       [INTEGRITY GATE]
Stage 4  Explore  ⇄  Record findings  → findings/                   (the heart)
Stage 5  Independent validation       → validated findings
Stage 6  Reporting                     → reports/
```

**No Stage 4 analysis may begin until the integrity gate passes** (loaders tested and verified against source; sample↔metadata pairing complete and exact; scientist signs off). The `stage4-explore` command enforces this as a hard precondition, and a hook (`guard_findings.py`) blocks any finding that claims sign-off or `validated` status before the gate.

**Driving the workflow.** Each stage has a command (invoked as `/findings-workflow:<name>`): `stage0-science`, `stage1-metadata`, `stage2-data`, `stage3-loaders`, `stage4-explore`, `stage5-validate`, `stage6-report`. Run `status` any time to see the pipeline position, the integrity-gate state, and the findings breakdown. Each stage command refuses to run until its preconditions are met; `state/workflow.json` is the single source of truth for progress and the gate (never hand-edit it loosely — the stage commands maintain it). The scientist can also just talk to you — the commands structure the work, but the collaboration drives it.

## Always-on: record findings as they emerge

During exploration, **every substantive insight is captured as a finding the moment it emerges** — automatically, with a brief non-disruptive notice ("recorded as finding 0042"). Dispatch the **findings-manager** to create/update findings and the manifest; never hand-edit `findings/manifest.md`.

- **What counts:** a tangible, specific, evidence-bearing observation about the data or its biology — if it has an effect, a statistic, or a concrete claim someone might later cite, record it.
- **Caveat findings are first-class.** Class imbalances, covariate skews, and confounds found while characterizing the cohort (Stage 1) are recorded as caveat findings (`kind: caveat`) — the workflow's durable memory of the gotchas that bias interpretation. They are consulted in Stage 4 (a confounded covariate enters the model; an imbalance dictates balanced metrics/stratified folds) and rendered as limitations in Stage 6, attached to the discoveries they qualify via `relates_to`. (Schema: the plugin's `conventions/findings.md` §2.6.)
- **Bias toward capturing too much.** Capture is cheap and low-bar (`candidate`); rigor is applied at promotion. Clutter is cheaper than lost insight.
- **Promotion to `validated` is never silent** and requires independent validation + the scientist's acceptance.

The schema, status machine, edge ontology, phase semantics, and the `validated` bar are defined in the plugin's `conventions/findings.md`. The `validated` bar is **independent re-derivation (blinded analytic replication) + the phase bar**; data replication is required only to claim `confirmatory`.

## The non-negotiables

1. **Code is the source of truth; conversation is ephemeral.** A finding is a script + a data version + a result, regenerable by anyone later. Figures and result tables are caches; the script is the artifact.
2. **Correctness is foundational and upstream of everything.** A silent loader error is common-mode and defeats validation (the verifier reads the same data through the same loader). Data fidelity is established before any analysis, not checked at the validation gate.
3. **No finding is validated without independent validation** — a fresh, history-starved verifier, a task derived mechanically with the answer stripped, and a concordance criterion fixed before it runs.
4. **Skepticism lives in the gates, calibrated by phase** — generous during exploration (capture freely), ruthless at promotion.
5. **Multiplicity is made honest** — exploratory vs confirmatory is marked on every finding; the exploration log records what was tried and discarded.

## Filesystem discipline

- **`data/` is read-only.** Raw inputs are immutable (enforced by a hook). Everything in `results/` and `figures/` is **regenerable** from raw data + a script and is never hand-edited.
- **Scratch vs promoted scripts.** Exploration spawns throwaways in `scripts/scratch/`. Reviewed, tested, typed, linted scripts live in `scripts/promoted/`. **A finding may link only to a promoted script.**
- **Defined homes:** project state in `state/`, findings + manifest in `findings/`, research in `research/`, reports in `reports/`, regenerable outputs in `results/` and `figures/`.
- **Figures** are dual-exported (SVG + 300 DPI PNG) with a separate legend image (rendered as its own figure so it never overlaps the plot), use the Okabe–Ito palette via `state/color_registry.json`, and are reviewed as the rendered PNG.
- **Python lives in the project.** Analysis runs on **Python ≥ 3.11** in a **project-local** environment (`./.venv`), with zero global footprint. Run `setup-env` to establish it — it detects an existing Python and, only if needed, asks before installing one **into this project** (the `uv` binary and interpreter under `./.uv`). Run code via `./.venv/bin/python …` (`.\.venv\Scripts\python …` on Windows); the gate tools (`ruff`, `mypy`, `pytest`, `hypothesis`) live in the env. `.uv/` and `.venv/` are git-ignored; `pyproject.toml`, `uv.lock`, and `.python-version` are committed (and feed `provenance.environment`). **Stage 3 will not start without a verified interpreter** — if there isn't one, run `setup-env` first.

## Project state files (rehydrate from these every session)

| File | Holds |
|---|---|
| `state/PROJECT.md` | Domain, design, scientific goals (Stage 0). |
| `state/METADATA.md` | Verified column meanings, design, confounds/imbalances, the experimental/control sample classification, join key; data-version stamped (Stage 1). |
| `state/DATA_DESCRIPTION.md` | Orientation, shape, transform/normalization + the recorded `scale`, missingness semantics, the confirmed preprocessing decisions (normalization method; batch axis + any confound), issues; data-version stamped (Stage 2). |
| `state/color_registry.json` | Category→color map; universal defaults seeded, extended per project once metadata is understood. |

These are canonical, regenerable, version-stamped references — never hand-edited into inconsistency with the data.

## Statistical & correctness conventions (enforced by reviewer agents + hooks)

No bare p-values (always effect size + CI + a **named** multiple-testing correction); report all tests run; **biological analysis runs on the experimental samples only** (control pools/references are identified in Stage 1, excluded from every contrast, and the exclusion is recorded in provenance); prefer canonical/moderated models for differential abundance; normalization is a recorded choice (median by default; respect the `scale` tag — never double-log); **batch correction is batch-label-only** (never hand ComBat the covariate of interest) and the key analysis is reported **corrected and uncorrected** (survival-of-correction is the test under confounding); **no data leakage** (learn preprocessing inside CV folds); match CV to the generalization target; mandate a label-shuffle null for classifiers; treat small-n as exploratory. Assumptions are hypotheses — test them in code and record the result. Full text in the plugin's `conventions/`.
