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

**Environment precondition.** Stage 1 is the **first stage that executes Python** — the validity checks, the cross-column hypothesis tests, the cohort characterization (distributions, crosstabs, the Table 1), and the confounding statistics below all run in code, which is what makes the understanding *verified* rather than asserted (Stage 0 is a pure interview). So a usable Python ≥ 3.11 must exist before it runs. The `setup-env` command establishes a **project-local** environment — detecting an existing Python or, only with the scientist's explicit consent, installing one *into the project* (zero global footprint). It is **live-verified as a Stage 1 precondition**, and **re-verified at the Stage 3 integrity gate** (doc 05.2).

The agent locates the metadata file (asking the scientist where it is), then:

- examines structure, columns, value domains;
- infers the meaning of each column from names, values, and domains;
- **identifies control samples** (pooled QC, reference/bridge channels, standards, blanks) from an explicit sample-role column or, failing that, naming-convention / no-group inference, and classifies every sample **experimental vs control** — controls are excluded from downstream biological analysis (doc 05.3) and viewed separately in QC (doc 06);
- opens an interaction to validate that understanding with the scientist — **including the experimental/control split and the rule deriving it**;
- checks value validity (types, ranges, allowed sets, uniqueness where expected);
- infers relationships that *should* hold if its understanding is correct, and **tests them as hypotheses** (principle from doc 05) — including confound detection: is the variable of interest aliased with batch, run order, or another factor?
- generates thorough descriptive plots and tables of the metadata — the distribution of every variable, pairwise cross-tabulations (variable of interest against covariates and batch), and a cohort summary table — to characterize the cohort and **expose class imbalance, covariate skew, and confounding** (per doc 06; metrics in doc 05.3);
- records each **material imbalance, skew, or confound as a caveat finding** (`kind: caveat`, doc 03.1) — the durable memory of the gotchas that constrain downstream claims, consulted in Stage 4 and carried into the report.

**Output — `state/METADATA.md`.** A verified, human- and agent-readable description containing: every column with its inferred meaning and validated type/domain; the experimental design it encodes; the **experimental/control sample classification and the rule deriving it** (the experimental subset is the analysis set); detected relationships, imbalances, and confounds (with the caveat findings recorded for the material ones); the join key to the data matrix; and a data-version stamp. This file is the canonical reference for what the experiment is. It is *generated from* verified understanding, never hand-asserted, and carries the data-version it describes so it cannot silently drift.

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

**Environment precondition (re-verify).** The Python environment gate lives at Stage 1 (§2.1), where execution begins. Because the integrity gate must not trust an inherited or stale interpreter, the working Python ≥ 3.11 is **re-verified here** before any loader work — floor **Python ≥ 3.11**, established by `setup-env` (doc 05.2).

This stage ends at the **integrity gate**, the workflow's hardest precondition (full requirements in doc 05). The gate passes only when:

- loader tests pass (unit, property, planted-truth, edge cases);
- the loaded data is verified against the source (counts reconcile, spot reconciliation, dtypes, ranges, identifier integrity, missing-value semantics, orientation);
- sample↔metadata pairing is complete and exact (every sample matched once, no orphans/duplicates, counts reconcile on both sides);
- the experimental/control classification (Stage 1) is carried onto the loaded samples and certified, so the experimental subset analysis runs on is well-defined (doc 05.3);
- the scientist signs off.

Metadata **caveat findings** recorded in Stage 1 (doc 03.1) have their `integrity_signoff` set at this gate, which certifies the sample↔metadata pairing they rest on.

**No Stage 4 analysis may begin until the integrity gate passes**, not by trusting the agent to remember. *(Implementation divergence: as built, this ordering is carried by the `stage4-explore` command precondition + orchestrator behavior, and the finding-write side is hook-enforced via `guard_findings.py`. It is **not** a standalone analysis-blocking hook — a single tool-use event can't cleanly separate exploratory analysis from legitimate Stage 3 loader/QC work. See `conventions/enforcement-map.md`.)*

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
| End of Stage 1 | Human checkpoint | Scientist confirms metadata understanding (incl. surfaced imbalances/confounds) |
| End of Stage 1 | Human checkpoint | Scientist confirms the experimental/control sample split + the rule deriving it |
| Sample↔metadata pairing | Human checkpoint | Scientist confirms join resolution (esp. fuzzy matches) |
| Integrity gate (end Stage 3) | Command precondition + hook (finding writes) + human sign-off | Loaders verified; no analysis before pass |
| Finding promotion | Gate (validation) + human | Independent validation cleared; scientist accepts |
| Report finding-selection | Human checkpoint | Which findings the report is about (doc 07) |
| Code promotion | Hook | Tests/types/lint pass before a script leaves `scratch/` |
| **Every stage boundary** | Human checkpoint | The scientist starts the next stage — the orchestrator suggests it and stops |

Gates that can be made deterministic are implemented as hooks (doc 08); the rest are explicit interaction points the orchestrator must honor.

**Stages advance on the scientist's word** *(added after implementation, by user decision).* The staged order says which stage may come next; it never licenses the orchestrator to *start* it. Each stage ends by reporting what was done, naming what comes next, and **stopping** — the scientist begins the next stage by running its command, asking in words, or having given a standing instruction to continue. Two things explicitly do **not** count as permission: the orchestrator's own judgment that the previous stage looks complete, and a **content sign-off** — a scientist confirming the metadata interpretation (the Stage 1 checkpoint) has agreed about the metadata, not asked for Stage 2 to begin; they are different questions. Writing `<stage>_done` and raising `current_stage` is bookkeeping that records the next stage is *available*, not a decision to enter it. The rule governs **boundaries**, not every action: within a stage the work is carried through without asking permission at each step. This makes every stage boundary a human checkpoint above, and is the complement of *Leaving a next step* — the orchestrator suggests, the scientist decides.

**Leaving a next step** *(added after implementation, by user decision).* The scientist is never left wondering what to do now: a substantive response ends with one concrete, named next step — the stage command, the analysis, the finding to validate, the figure to commission — and each stage command closes by naming what follows. **Stage 4 is the deliberate exception**, and it is a *hard* one: it is an open loop that only the scientist closes, so the orchestrator suggests freely *within* it (the next analysis, a finding to record, a figure that would show a claim, `stage5-validate` on a matured candidate — validation runs continuously, so it is not an exit) but **never volunteers that exploration is over**: it does not raise reporting, does not observe that the findings look complete, does not ask whether the scientist is ready to wrap up. Exploration has no end the agent can judge; nudging toward closure is the motivated-reasoning pressure the phase-calibrated skepticism exists to resist; and the report's finding-selection is a human checkpoint above. Asked directly, the orchestrator answers plainly; signalled, it follows. The rule forbids raising it, not discussing it.

**How a checkpoint is asked** *(added after implementation, by user decision).* Every question to the scientist — at a checkpoint or in ordinary conversation — follows one rule: **one question at a time**, never a form dump, because a batch gets a single merged reply that silently drops half of it. Where the answer is a **choice among known options** (the normalization method, whether to pay for a shuffle null, the report mode), the options are **offered** — via the `AskUserQuestion` tool where available — with the **recommended option first and the reason given**; where the answer is genuinely open-ended (Stage 0's elicitation), it is asked plainly in prose. And nothing is asked that `state/`, the data, or the manifests already answer. A checkpoint bundled into a multi-part question is not a checkpoint: the scientist's "yes" must attach to exactly one thing. The standing rule lives in the project `CLAUDE.md` (*How to ask*), materialized from `templates/project-CLAUDE.md`; the stage commands point at it where they ask.
