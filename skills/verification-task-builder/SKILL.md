---
name: verification-task-builder
description: >-
  Mechanically construct a blind verification task for the verifier agent from a
  finding's structured fields, with the answer (evidence and verdict) stripped,
  and pre-specify the concordance criterion before the verifier runs. Use this
  whenever preparing to validate a finding (analytic or data replication), so the
  agent that knows the result never hand-writes a prompt that could leak it.
---

# Verification-task builder

This procedure turns a finding into (a) a **blind task** for the `verifier` agent and (b) a **pre-specified concordance criterion** held by the dispatcher. It exists to make independent validation contamination-resistant: the task is derived *mechanically from structured fields*, not free-prosed by someone who knows the answer (doc 03.5).

Schema reference: `conventions/findings.md` (§2 fields, §4 validation, §5 phase).

## Inputs

- The target finding file `findings/<id>-<slug>.md`.
- The validation **mode**: `analytic_replication` (same data, lower bar) or `data_replication` (held-out/orthogonal data, required to claim `confirmatory`).

## Step 1 — Strip the answer

From the finding's frontmatter, **remove and set aside** (the verifier must never see these):

- `verdict` — the plain-language bottom line.
- `evidence` — every recorded number (this is the answer).
- `validation` — prior validation notes.
- `figures` — **the whole block, plus the body's inline images and their readings.** A figure *is* the answer rendered: a volcano shows the direction and the hit count, an ROC shows the performance, and the four-part body pattern puts a plain-language **reading** right beside each image (`conventions/findings.md` §9). Never pass a figure path, caption, or reading to the verifier, and never point it at this finding's `figures/<NNNN>-*` artifacts — it renders its own figures if it wants them.

Also treat `summary` as **answer-bearing**: it states the claim and often its direction/magnitude. **Do not pass `summary` to the verifier.** Derive the question from the structured comparison instead (Step 2).

## Step 2 — Build the neutral question

Construct the question from the *structure* of the analysis, phrased so it reveals **what to measure** but never **what was found**:

- **Comparison/contrast** — from `provenance.params` (e.g. `contrast: drug_A_vs_control`). State it as a neutral comparison, not a directional claim.
- **Feature(s)** — from `entities` (canonical IDs) and/or the params (e.g. the protein, gene, or feature set under test).
- **Quantities to report** — the same metric family the finding used (e.g. "report the effect size, its CI, and a BH-corrected p-value"), so results are comparable — **without** stating the recorded values.
- **Method** — ideally **omit** (let the verifier choose the canonical approach; method divergence is informative). Include it only if the question is meaningless without it, and even then never include the result.

> **The cardinal rule:** the question may say *"What is the effect of drug_A vs control on TP53 (P04637)? Report log2FC, its 95% CI, and a BH-corrected p-value."* It may **never** say *"Confirm that drug_A upregulates TP53"* or mention any recorded number. Direction and magnitude are the answer.

## Step 3 — Set the data scope (and enforce disjointness for data replication)

- **`analytic_replication`** — same dataset as the finding. This is the lower bar; a finding validated only this way stays `phase: exploratory`.
- **`data_replication`** — name the **held-out split or orthogonal dataset**, and verify it is **disjoint** from the data that generated the hypothesis (the hard rule, doc 03.6: generate-set ≠ validate-set). Pass the split/dataset identifier explicitly in the task. Only a passing data replication permits `phase: confirmatory`.

## Step 4 — Pre-specify the concordance criterion (before the verifier runs)

Decide, **now**, what counts as concordant — never after seeing the verifier's result (that would reintroduce forking paths at the validation stage). Choose a criterion appropriate to the metric and record it verbatim in the finding's `validation.<mode>.concordance_criterion`:

- **Effect-size metrics (log2FC, mean diff, etc.):** same **sign**, magnitude within a stated tolerance, and significance agreement (e.g. `p_adjusted < α` on both sides). Example: *"same sign AND |log2FC| within 0.3 AND p_adj < 0.05."*
- **Classifier performance (AUC, accuracy):** within a stated delta **and** above the chance/label-shuffle null. Example: *"AUC within 0.05 AND exceeds the shuffled-label null."*
- **Correlation/association:** same sign, |r| within a tolerance, significance agreement.
- **Counts/proportions:** within a stated relative tolerance.

The criterion must be phrased so that **handing it to the verifier would not reveal the recorded value.** If a tolerance like "within 0.3 of recorded" would leak the number, keep the criterion with the dispatcher and pass the verifier only the neutral "report log2FC and CI" instruction; apply the criterion afterward.

Write the criterion into the finding *before* dispatching, so the record proves it was pre-specified.

## Step 5 — Emit the artifacts

Produce two clearly separated outputs:

1. **Blind verifier task** (the prompt to dispatch to the `verifier` agent): mode, neutral question, feature/entity scope, data scope, quantities to report, optional method. Contains **no** verdict, evidence, summary, or recorded number.
2. **Held concordance record** (kept by the dispatcher / findings-manager): the mode, the pre-specified criterion, and the recorded `evidence` to compare against once the verifier returns.

## Step 6 — After the verifier returns

The dispatcher (orchestrator / findings-manager), **not the verifier**, compares the verifier's independent result against the recorded evidence using the pre-specified criterion, then asks the findings-manager to record `validation.<mode>` (`status`, `by`, `date`, `concordance_criterion`, `concordance_result`). Recall the full `validated` bar (conventions/findings.md §4): `validated` needs `integrity_signoff`, computational reproduction, **blinded analytic replication**, the phase bar, **and** human acceptance — promotion is never silent.
