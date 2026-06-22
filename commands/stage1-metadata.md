---
name: stage1-metadata
description: "Stage 1 — Understand the metadata. Infer and validate the meaning of every column, test the relationships that should hold, detect confounds, and write state/METADATA.md."
---

# Stage 1 — Understand the metadata

**Precondition:** `state/workflow.json` shows `science_done: true`. If not, run `stage0-science` first and say so.

The goal is a *verified* understanding of the experiment's metadata — generated from evidence, never hand-asserted.

## Do this

1. **Locate the metadata file.** Ask the scientist where it is (typically in `data/`).
2. **Examine** its structure, columns, and value domains.
3. **Infer the meaning of each column** from names, values, and domains.
4. **Validate that understanding with the scientist** — this is a **human checkpoint** (doc 02.8). Present your inferred column meanings and design interpretation; get confirmation or correction before writing the file.
5. **Check value validity** — types, ranges, allowed sets, uniqueness where expected. Fail loud on anything inconsistent.
6. **Treat your inferences as hypotheses and test them in code** (doc 05): infer relationships that *should* hold if your understanding is correct, then test them. Crucially, do **confound detection** — is the variable of interest aliased with batch, run order, or another factor? A confounded design changes what every downstream finding can claim.
7. **Generate thorough descriptive plots and tables** of the metadata (per the visualization conventions — dual export, Okabe–Ito via the color registry).
8. **Extend the color registry.** Now that the design is understood, add project-specific categorical dimensions (treatment arms, cell lines, timepoints) to `state/color_registry.json` with `scope: project`, so every figure colors them consistently from the start.

## Output — `state/METADATA.md`

A verified, human- and agent-readable description containing:

- every column with its inferred meaning and validated type/domain;
- the experimental design it encodes;
- detected relationships and **confounds**;
- the **join key** to the data matrix (how samples in metadata map to columns/rows in the data);
- a **data-version stamp** (so the file cannot silently drift from the data it describes).

This is the canonical reference for *what the experiment is*. Regenerate it from verified understanding; never edit it into inconsistency with the data.

## Then

- Update `state/workflow.json`: `metadata_done: true`, `current_stage: 2`, bump `updated`.
- Next: **Stage 2 — Understand the data** (`stage2-data`).
