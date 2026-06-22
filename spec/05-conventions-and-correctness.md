# 05 — Conventions and Correctness

Conventions are an **enforcement spec**: a convention is only real if something checks it. Where a check can be made deterministic it is a **hook** (doc 08); otherwise it is a **reviewer-agent check**. The enforcement map (5.5) assigns each rule.

## 5.1 Repository / filesystem conventions

- **Immutable inputs vs regenerable outputs, made physical.** Raw data (`data/`) is read-only. Everything in `results/` and `figures/` is reproducible from raw data plus a script and is never hand-edited. A figure is a cache; the script is the artifact.
- **Scratch vs promoted scripts.** Exploration spawns throwaways in `scripts/scratch/`. Reviewed, tested scripts live in `scripts/promoted/`. **A finding may link only to a promoted script.**
- **Defined homes.** Project state in `state/`, findings and manifest in `findings/`, research in `research/`, reports in `reports/` (see README layout).

## 5.2 Coding conventions

- **Python only.**
- **Python ≥ 3.11, in a project-local environment.** The floor is 3.11. The interpreter and virtualenv live **inside the project** with **zero global footprint** — setup never modifies the user's `PATH`, shell profile, or system Python. The `setup-env` command detects an existing suitable interpreter and, only when none is found, **transparently asks consent** before downloading `uv` + a standalone Python *into the project* (project-local install dir, no PATH changes). Declining is allowed; the scientist then supplies Python ≥ 3.11 and analysis stays blocked until one exists. This precondition is **live-verified at Stage 3** (the first stage that executes Python), not assumed.
- **Locked environments.** Pinned versions in a lockfile; the environment is recorded alongside any finding, so computational reproduction is meaningful.
- **Seeds set and recorded** everywhere stochastic (numpy, sklearn, etc.).
- **Non-interactive, parameterized scripts.** Scripts run end-to-end from config, not hard-coded paths. Notebooks, if used for exploration, are disposable; the canonical path is always a script (out-of-order cell execution is a reproducibility landmine).
- **Logging over print;** data handling **fails loud** on shape mismatch or silent NA coercion.
- **Maximum testing, typing, linting** (see 5.4) — required before any script is promoted.

## 5.3 Statistical conventions

- **No bare p-values.** Always report an effect size and a confidence interval alongside a corrected p-value.
- **Always apply and name a multiple-testing correction** (BH/FDR for omics).
- **Report all tests run,** not only the significant ones (feeds the exploration log, doc 03.6).
- **Canonical over esoteric.** Prefer widely used, explainable tests. For differential abundance prefer a moderated linear model (limma / MSstats) over naive per-feature t-tests.
- **No data leakage.** Every preprocessing step that learns from data — normalization, imputation, feature selection, scaling — happens **inside** each cross-validation fold, never on the full dataset first.
- **Match cross-validation to the generalization target.** Define the unit of generalization (new samples? new patients? new batches?) and structure folds accordingly (group/subject-wise folds when samples cluster within a patient).
- **Mandate a label-shuffle null** for classifiers: if performance does not collapse under randomized labels, there is leakage.
- **Be honest about power.** Treat small-n results as `exploratory` by default (doc 03.6).

## 5.4 Correctness and data integrity (the charter)

Correctness is paramount and **upstream of every other safeguard.** Independent validation, the statistical conventions, and the review agents all assume the data were read correctly; if that fails, none of them help.

### Why this comes first — the common-mode argument

A data-loading error is silent and common-mode. A broken loader does not crash; it produces plausible, wrong numbers. Every finding built on it is then confidently false, and the independent verifier **cannot catch it**: the verifier re-derives the analysis but reads the same data through the same loader, so it reproduces the same wrong input and the two agree on a falsehood. Data fidelity is therefore a precondition validation *assumes*, established before any analysis — not something validation checks.

**Standing rule for the whole workflow: assume nothing, verify everything, fail loud.** A loader that silently drops rows, coerces a type, or mismatches a sample is worse than one that crashes.

### Data loading is two obligations

**A. Test the loader.** Unit tests with hand-verified fixtures; property/invariant tests (loading preserves source counts; no value appears that wasn't in the source); a **planted-truth fixture** (synthetic data with a known effect the pipeline must recover); edge cases (empty, all-missing, single sample, duplicate IDs, ties).

**B. Verify the loaded data on the real file.** Counts reconcile (rows/cols vs source); random-cell spot reconciliation against the raw source; orientation confirmed not assumed; dtypes explicit and correct (no silent string↔numeric coercion); value ranges plausible; identifier integrity (no truncation/reformatting); missing-value encoding made explicit; transformation/normalization state confirmed; **sample↔metadata pairing complete and exact** (every sample matched once, no orphans/duplicates, counts reconcile both sides).

Loading is not "done" until both pass and the scientist signs off (the integrity gate, doc 02.3).

### Domain-specific fidelity traps (proteomics)

- **Spreadsheet identifier corruption:** gene symbols turned into dates (SEPT/MARCH/DEC families), accessions in scientific notation, stripped leading zeros. Documented in a large fraction of published genomics supplements; assume it happened until proven otherwise.
- **Missing-value semantics:** `0`, `NA`, `NaN`, empty string, and tool tokens like "Filtered" are **not** interchangeable. Conflating a true zero with missing-not-at-random is catastrophic and changes every downstream statistic. Identify and handle deliberately.
- **Scale confusion:** linear vs log mistaken for each other.
- **Contaminants/decoys:** `CON__`/`REV__` rows included or excluded by explicit decision, never by accident.
- **Protein groups/ambiguity:** semicolon-delimited members handled by explicit policy.
- **Mechanical parsing hazards:** locale/decimal separators, multi-row/merged headers, embedded metadata, duplicated/inconsistently named replicate columns.

### Assumptions are hypotheses

Every inference about structure, meaning, or relationships is a hypothesis tested with code, and the test and its result are recorded. Nothing proceeds on an unchecked assumption.

### Analysis code is held to the maximum

Before promotion (and before a finding may link to it): maximum unit testing including edge cases; property-based tests for invariants (CV folds partition without overlap; a normalization preserves shape; a transform is invertible where claimed); **type hints throughout with a type checker in the loop** (mypy/pyright); **linting and formatting** (ruff or equivalent); seeds set/recorded; planted-truth checks where applicable. Code that has not passed tests, types, and lint is `scratch` and a finding may not link to it.

### Double-check critical quantities

Where feasible, derive key numbers two independent ways and reconcile. Because loader errors are common-mode, double-checking the *data read itself* lives here, at the data boundary, before any analysis — not at the verifier.

## 5.5 Enforcement map

| Rule | Enforced by |
|---|---|
| Usable Python ≥ 3.11 project env before analysis | **Command precondition** (`stage3-loaders` live-verifies) + `setup-env` |
| No analysis before integrity gate passes | **Command precondition** + orchestrator behavior (hook gates finding writes) + human sign-off |
| Script not promoted until tests/types/lint pass | **Hook** |
| Raw data read-only; outputs regenerable | **Hook** (block writes to `data/`) |
| Finding links only to a promoted script | **Hook** / findings-manager check |
| Record-the-finding during exploration | **Hook** + orchestrator behavior (CLAUDE.md) |
| No bare p; correction named; effect+CI present | Stats reviewer |
| No leakage; CV matched to target; label-shuffle null | Stats reviewer |
| Canonical tests; moderated models for DE | Stats reviewer |
| Loader test + load verification complete | Code reviewer + hook on gate |
| Every reference exists and supports its claim | Research reviewer |
| Figure rendered, reviewed, dual-exported (doc 06) | Figure reviewer + hook |

Hooks fire deterministically on events and do not depend on the model remembering — which is exactly why the highest-stakes gates are hooks rather than reviewer judgment.
