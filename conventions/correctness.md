# Convention — Correctness & Data Integrity (the charter)

*Spec source: doc 05.4. This is the most important convention in the system. It is **upstream of every other safeguard**: independent validation, the statistical conventions, and the review agents all assume the data were read correctly. If that fails, none of them help. Operationalized by the **integrity gate** (`stage3-loaders`) and the read-only-data hook.*

## Why this comes first — the common-mode argument

A data-loading error is **silent and common-mode**. A broken loader does not crash; it produces plausible, wrong numbers. Every finding built on it is then confidently false, and **the independent verifier cannot catch it**: the verifier re-derives the analysis but reads the same data through the same loader, reproduces the same wrong input, and the two agree on a falsehood. Data fidelity is therefore a precondition validation *assumes*, established before any analysis — **not** something the validation gate checks.

## The standing rule

**Assume nothing, verify everything, fail loud.** A loader that silently drops rows, coerces a type, or mismatches a sample is worse than one that crashes.

## Data loading is two obligations — both required (the integrity gate)

**A. Test the loader.**
- Unit tests with hand-verified fixtures.
- Property/invariant tests: loading preserves source counts; no value appears that wasn't in the source.
- A **planted-truth fixture**: synthetic data with a known effect the pipeline must recover.
- Edge cases: empty, all-missing, single sample, duplicate IDs, ties.

**B. Verify the loaded data on the real file.**
- Counts reconcile (rows/cols vs source).
- Random-cell **spot reconciliation** against the raw source.
- Orientation **confirmed, not assumed**.
- Dtypes explicit and correct (no silent string↔numeric coercion).
- Value ranges plausible.
- Identifier integrity (no truncation/reformatting).
- Missing-value encoding made explicit.
- Transformation/normalization state confirmed.
- **Sample↔metadata pairing complete and exact** — every sample matched once, no orphans/duplicates, counts reconcile on both sides (fuzzy matches are a human checkpoint).

Loading is not "done" until both pass and the scientist signs off.

## Domain-specific fidelity traps (proteomics)

- **Spreadsheet identifier corruption:** gene symbols turned into dates (SEPT/MARCH/DEC families), accessions in scientific notation, stripped leading zeros. Documented in a large fraction of published genomics supplements — **assume it happened until proven otherwise.**
- **Missing-value semantics:** `0`, `NA`, `NaN`, empty string, and tool tokens like `"Filtered"` are **not** interchangeable. Conflating a true zero with missing-not-at-random is catastrophic and changes every downstream statistic. Identify and handle deliberately.
- **Scale confusion:** linear vs log mistaken for each other.
- **Contaminants/decoys:** `CON__`/`REV__` rows included or excluded by **explicit decision**, never by accident.
- **Protein groups/ambiguity:** semicolon-delimited members handled by explicit policy.
- **Mechanical parsing hazards:** locale/decimal separators, multi-row/merged headers, embedded metadata, duplicated/inconsistently named replicate columns.

## Assumptions are hypotheses

Every inference about structure, meaning, or relationships is a hypothesis **tested with code**, with the test and its result recorded. Nothing proceeds on an unchecked assumption (this is why Stage 1 and Stage 2 test their inferences rather than asserting them).

## Analysis code is held to the maximum

See `conventions/coding.md`. Before promotion (and before a finding may link to a script): maximum unit testing including edge cases; property-based tests for invariants; type hints throughout with a type checker; linting and formatting; seeds set/recorded; planted-truth checks where applicable.

## Double-check critical quantities

Where feasible, **derive key numbers two independent ways and reconcile.** Because loader errors are common-mode, double-checking the **data read itself** lives here, at the data boundary, before any analysis — not at the verifier.

## Enforcement

| Rule | Enforced by |
|---|---|
| No analysis before the integrity gate passes | **Hook** (`guard_findings.sh` on integrity_signoff/validated; reads `state/workflow.json`) + orchestrator behavior (stage ordering) + human sign-off |
| Raw data read-only | **Hook** (`guard_readonly_data.sh`) |
| Loader tested + load verified + pairing exact | **Code-reviewer** + the Stage 3 integrity-gate checklist + human sign-off |
| Assumptions tested in code, results recorded | Orchestrator behavior (Stages 1–2) + code/stats reviewers |
