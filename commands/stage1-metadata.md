---
name: stage1-metadata
description: "Stage 1 — Understand the metadata. Infer and validate the meaning of every column, test the relationships that should hold, detect confounds, and write state/METADATA.md."
---

# Stage 1 — Understand the metadata

**Precondition — prior stage:** `state/workflow.json` shows `science_done: true`. If not, run `stage0-science` first and say so.

**Precondition — Python environment (hard gate).** This is the **first stage that executes Python**: the type/range/uniqueness validity checks (step 6), the cross-column invariant tests (step 7), the cohort characterization — distributions, crosstabs, the Table 1 (step 8), and the bias-corrected Cramér's V confounding statistics (step 9) all run in code, not by eye — that is what makes this understanding *verified* rather than hand-asserted. (Stage 0 was a pure interview; execution begins here.) So a usable interpreter must exist before any metadata examination. **Live-verify** a working Python ≥ 3.11 (do not trust `state/workflow.json` `environment.configured` — check the interpreter): prefer the project venv (`./.venv/bin/python` on Unix, `.\.venv\Scripts\python.exe` on Windows), else a project-local/`PATH` `uv`, else a system `python3`/`python`/`py -3`, and confirm it reports ≥ 3.11. If none is usable, **stop and run `setup-env`** (which detects, transparently asks consent, and installs Python ≥ 3.11 *into the project* if approved). If the scientist previously declined the project-local install, tell them to install Python ≥ 3.11 themselves and re-run. Do not start metadata examination without a verified interpreter.

The goal is a *verified* understanding of the experiment's metadata — generated from evidence, never hand-asserted.

## Do this

1. **Locate the metadata file.** Ask the scientist where it is (typically in `data/`).
2. **Examine** its structure, columns, and value domains.
3. **Infer the meaning of each column** from names, values, and domains.
4. **Identify control samples — do not assume every sample is experimental.** Studies routinely include non-biological **control samples** alongside the experimental ones: **pooled QC samples** (a pool run repeatedly to track technical reproducibility and drift), **reference / bridge** channels (a common standard for cross-batch normalization), **standards/calibrants** (iRT, HeLa, BSA), and **blanks**. These are handled differently downstream — viewed *separately* in QC and **excluded from the biological analysis** (Stage 4 runs on the experimental subset only) — so they must be found now. Look **specifically**: prefer an **explicit metadata column** that encodes sample role (names like `sample type`, `type`, `class`, `category`, `role`, or a `group` column carrying a control level); failing that, **infer** from sample-naming conventions (`Pool`/`Pooled`/`QC`/`Ref`/`Reference`/`Bridge`/`Std`/`Standard`/`Blank`/`IS`) or from samples that carry no biological-group assignment. Classify every sample **experimental vs control** (binary), and note the control *subtype* in prose where known (pool / reference / standard / blank), since QC reads pools and references differently. Because a misclassification silently corrupts the analysis set in **both** directions, any inferred split is confirmed at the human checkpoint below. A control-related *limitation* — no technical-QC control exists (drift can't be assessed), or controls are unevenly distributed across batches — is itself recordable as a caveat finding (`kind: caveat`); reuse that mechanism (step 9), don't add new machinery.
5. **Validate that understanding with the scientist** — this is a **human checkpoint** (doc 02.8). Present your inferred column meanings, the design interpretation, **and the proposed experimental/control split (with the column or rule it rests on)**; get confirmation or correction before writing the file. Walk it **one question at a time** rather than as a single wall of inferences to ratify, and where a point is a genuine choice (which column encodes the role; whether an ambiguous sample is a control) offer the options with **`AskUserQuestion`**, recommended first (*How to ask*, project `CLAUDE.md`).
6. **Check value validity** — types, ranges, allowed sets, uniqueness where expected. Fail loud on anything inconsistent.
7. **Treat your inferences as hypotheses and test them in code** (doc 05): infer relationships that *should* hold if your understanding is correct, then test them — including whether the variable of interest is independent of batch, run order, and the key covariates (confounding; quantified and, where material, recorded in step 9). Assumptions are hypotheses, not assertions.
8. **Characterize the cohort — thorough descriptive plots and tables.** Generate, per the visualization conventions (dual export, Okabe–Ito via the color registry): the **distribution of every metadata variable** (sample counts per categorical level; summaries/histograms for continuous variables like age); **pairwise cross-tabulations** of the variables that matter — the variable of interest against each covariate and against batch/run-order (e.g. sex × group, age × sex, group × batch); and a publication-ready **cohort summary table ("Table 1")** broken down by the primary grouping. These are deliverables in their own right (papers, talks) *and* the lens for the next step. See `conventions/statistics.md` (descriptive characterization) for what to compute.
9. **Hunt for imbalance, bias, and confounding — and record the material ones as findings.** The point of the characterization is not just figures: it is to surface **class imbalance** (a grouping dominated by one level, severely unequal arm sizes), **covariate skew** (age/sex/etc. distributed unevenly across the contrast), and **confounding** (the variable of interest aliased with batch, run order, sex, or another factor — quantify with bias-corrected Cramér's V). A confounded or imbalanced design changes what every downstream finding can claim. **For each material gotcha, dispatch the findings-manager to record a caveat finding** (`kind: caveat`, `status: candidate`, `phase: exploratory`, `integrity_signoff: false`) capturing the evidence — the crosstab, the arm sizes, the confounding statistic — and the interpretive risk. **Show it, don't just state it:** the characterization has already rendered the crosstab / distribution that exposes the gotcha, so hand that figure to the manager with its producing script + input, caption, and reading, so the caveat *shows* the imbalance rather than asserting it; if the right figure doesn't exist yet, dispatch a `figure-generator` for it (`conventions/findings.md` §2.4). The threshold is judgment, not a rule (suggestive cutoffs in `conventions/statistics.md`): record what would change a downstream analysis or its interpretation, not every histogram. These candidates have their `integrity_signoff` set at the integrity gate (Stage 3), which certifies the pairing they rest on.
10. **Extend the color registry.** Now that the design is understood, add project-specific categorical dimensions (treatment arms, cell lines, timepoints) to `state/color_registry.json` with `scope: project`, so every figure colors them consistently from the start.

## Output — `state/METADATA.md`

A verified, human- and agent-readable description containing:

- every column with its inferred meaning and validated type/domain;
- the experimental design it encodes;
- the **experimental-vs-control sample classification** — the column or rule used to derive it, the per-class counts (and control subtypes where known: pool / reference / standard / blank), and that the **experimental subset is the analysis set** (controls are excluded from biological analysis and viewed separately in QC);
- detected relationships, **class imbalances, covariate skews, and confounds** — each material one with a pointer to the caveat finding (`kind: caveat`) recorded for it;
- the **join key** to the data matrix (how samples in metadata map to columns/rows in the data);
- a **data-version stamp** (so the file cannot silently drift from the data it describes).

This is the canonical reference for *what the experiment is*. Regenerate it from verified understanding; never edit it into inconsistency with the data.

## Then

- Update `state/workflow.json`: `metadata_done: true`, `current_stage: 2`, bump `updated`.
- Next: **Stage 2 — Understand the data** (`stage2-data`).

**Then stop.** Naming the next stage is where your turn ends — the scientist starts it, by running the command or asking you to. Recording `current_stage` is bookkeeping, not permission (project `CLAUDE.md`, *Stages advance on the scientist's word, not yours*). Note the scientist's confirmation at the checkpoint above was about the *metadata*, not about starting Stage 2.
