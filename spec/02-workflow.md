# 02 — The Workflow

The workflow is a staged pipeline. Its ordering rule is absolute: **nothing is analyzed before it is understood, and nothing is explored before the data read is verified.** Each stage writes durable project state that later stages and fresh agents rehydrate from, so the workflow survives context loss and subagent boundaries.

```
Stage 0  State the science            → state/PROJECT.md
Stage 1  Understand the metadata      → state/METADATA.md
Stage 2  Understand the data          → state/DATA_DESCRIPTION.md
Stage 3  Loaders + QC  [INTEGRITY GATE]→ verified loaders, QC report
Stage 4  Explore  ⇄  Record findings  → findings/ (the loop; the heart)
Stage 5  Independent validation       → validated findings
Stage 6  Reporting                    → reports/
```

Stages 0–3 are largely sequential and gated. Stage 4 is an open loop. Stage 5 runs continuously as candidates mature. Stage 6 runs on demand.

## 2.0 Stage 0 — State the science

The scientist describes the research up front: the domain, what is being examined and why, the experimental design, and the scientific goals — what we want to find out. The agent writes this to `state/PROJECT.md`. This shapes what "understanding the data" even means and what findings are relevant.

Guard: stating goals first improves relevance but invites motivated reasoning. The skepticism gates (principle 6) and independent validation (doc 03) exist partly to counter this; the orchestrator must not let stated hopes bias what it reports.

## 2.1 Stage 1 — Understand the metadata

The agent locates the metadata file (asking the scientist where it is), then:

- examines structure, columns, value domains;
- infers the meaning of each column from names, values, and domains;
- opens an interaction to validate that understanding with the scientist;
- checks value validity (types, ranges, allowed sets, uniqueness where expected);
- infers relationships that *should* hold if its understanding is correct, and **tests them as hypotheses** (principle from doc 05) — including confound detection: is the variable of interest aliased with batch, run order, or another factor?
- generates thorough descriptive plots and tables of the metadata (per doc 06).

**Output — `state/METADATA.md`.** A verified, human- and agent-readable description containing: every column with its inferred meaning and validated type/domain; the experimental design it encodes; detected relationships and confounds; the join key to the data matrix; and a data-version stamp. This file is the canonical reference for what the experiment is. It is *generated from* verified understanding, never hand-asserted, and carries the data-version it describes so it cannot silently drift.

## 2.2 Stage 2 — Understand the data

For the sample-by-feature matrix the agent:

- determines orientation (samples vs features), shape, and identifier formats;
- examines feature names, sample names, and values;
- determines transformation state (log vs linear), normalization state, and missing-value encoding (see the domain traps in doc 05);
- characterizes missingness structure, contaminants, decoys, and duplicates;
- runs tests against its current understanding and surfaces problems.

**Output — `state/DATA_DESCRIPTION.md`.** A verified description containing: orientation and shape; feature and sample identifier schemes; transformation/normalization state; missing-value semantics; contaminant/decoy handling decisions; known data-quality issues; and the data-version stamp. Same regeneration and stamping discipline as `METADATA.md`.

## 2.3 Stage 3 — Loaders, pairing, QC  **[INTEGRITY GATE]**

The agent builds robust, tested loaders that load both the data and the metadata and **correctly pair every sample with its metadata description**. It then verifies the loaded data against the source and produces descriptive QC: abundance boxplots across samples, PCA, missingness maps, correlation structure.

This stage ends at the **integrity gate**, the workflow's hardest precondition (full requirements in doc 05). The gate passes only when:

- loader tests pass (unit, property, planted-truth, edge cases);
- the loaded data is verified against the source (counts reconcile, spot reconciliation, dtypes, ranges, identifier integrity, missing-value semantics, orientation);
- sample↔metadata pairing is complete and exact (every sample matched once, no orphans/duplicates, counts reconcile on both sides);
- the scientist signs off.

**No Stage 4 analysis may begin until the integrity gate passes**, not by trusting the agent to remember. *(Implementation divergence: as built, this ordering is carried by the `stage4-explore` command precondition + orchestrator behavior, and the finding-write side is hook-enforced via `guard_findings.sh`. It is **not** a standalone analysis-blocking hook — a single tool-use event can't cleanly separate exploratory analysis from legitimate Stage 3 loader/QC work. See `conventions/enforcement-map.md`.)*

## 2.4 Stage 4 — Explore ⇄ record findings (the heart)

With understanding established and the read verified, the loop begins. The agent runs boilerplate analysis from the vetted library (t-tests, Mann–Whitney, feature finding, classifiers, regression — all from `lib/`, doc 04/05) to spark discussion. The scientist and agent go back and forth: a plot looks interesting, a protein group invites a question, the scientist points at a heatmap cluster. The agent investigates by composing or writing appropriate analysis (held to the code-rigor conventions), and **every substantive insight is captured as a finding** the moment it emerges (doc 03 defines the object, the trigger policy, candidate-vs-promoted, and silent-vs-confirmed recording).

Skepticism here is calibrated to *generous* (principle 6): capture freely as candidates; the rigor is applied at promotion, not at every breath.

## 2.5 Stage 5 — Independent validation

Candidate findings are promoted toward *validated* through the independent-validation gate (doc 03): a clean-context verifier, a task derived mechanically from the finding's structured fields with the answer stripped, and a concordance criterion fixed before the verifier runs. What "validated" requires is defined by the status machine (doc 03).

## 2.6 Stage 6 — Reporting

Validated findings (and vetted research) are compiled into reports, which are projections of the findings graph rather than fresh write-ups (doc 07). Two modes: an exhaustive QC/data-quality report and a selective, disseminable research report that supports downstream manuscript preparation.

## 2.7 Project state files — contract

| File | Written after | Contains | Discipline |
|---|---|---|---|
| `state/PROJECT.md` | Stage 0 | Domain, design, scientific goals | Updated as understanding deepens |
| `state/METADATA.md` | Stage 1 | Verified column meanings, design, confounds, join key | Generated from verified understanding; data-version stamped |
| `state/DATA_DESCRIPTION.md` | Stage 2 | Orientation, shape, transform/normalization, missingness semantics, issues | Generated from verified understanding; data-version stamped |
| `state/color_registry.json` | Stage 1+ | Category→color map (doc 06) | Seeded with universal defaults, extended per project |

All state files are canonical references that any fresh agent reads to rehydrate understanding. They are regenerable, never hand-edited into inconsistency with the data, and version-stamped so drift is detectable.

## 2.8 Gates and human checkpoints

| Point | Type | Condition / decision |
|---|---|---|
| End of Stage 1 | Human checkpoint | Scientist confirms metadata understanding |
| Sample↔metadata pairing | Human checkpoint | Scientist confirms join resolution (esp. fuzzy matches) |
| Integrity gate (end Stage 3) | Command precondition + hook (finding writes) + human sign-off | Loaders verified; no analysis before pass |
| Finding promotion | Gate (validation) + human | Independent validation cleared; scientist accepts |
| Report finding-selection | Human checkpoint | Which findings the report is about (doc 07) |
| Code promotion | Hook | Tests/types/lint pass before a script leaves `scratch/` |

Gates that can be made deterministic are implemented as hooks (doc 08); the rest are explicit interaction points the orchestrator must honor.
