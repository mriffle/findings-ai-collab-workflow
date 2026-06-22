---
name: stage6-report
description: "Stage 6 — Reporting. Compile findings into a report that is a projection of the findings graph (not a fresh write-up). Two modes: an exhaustive QC report or a selective research report."
argument-hint: "[qc | research]"
---

# Stage 6 — Reporting

A report is **not** a fresh act of writing. Because findings already carry evidence, figures, methods, caveats, references, and status, a report is a **projection of selected findings into prose** (doc 07). Writers select, order, and add connective tissue; they do not invent content.

## Pick the mode (`$ARGUMENTS`)

- **`qc`** — QC / data-quality report: exhaustive and descriptive, so collaborators can trust the data. Produced largely from Stage 3 outputs and the project state files.
- **`research`** — research report: selective and narrative; a **disseminable artifact that supports downstream manuscript preparation** (not a manuscript itself). The primary deliverable. Findings are deliberately left out to tell a focused story.

The selection discipline is **inverted** between modes (completeness vs focus) — confirm the mode before writing.

## Process

1. **Select + outline** — choose which findings the report is about and in what order. This is a scientific judgment and a **human checkpoint** (doc 02.8): the scientist decides. For a research report, default to `validated` findings; an `exploratory` finding may appear only with its phase and caveats intact.
2. **Dispatch per section** — hand each logical section to a writer/reviewer pair with a **shared spec** (glossary, defined abbreviations, the agreed narrative) so parallel writers don't drift or repeat each other. (Writer/report-reviewer agents arrive with the reporting subsystem; until then, draft and self-review against `conventions/` and doc 07.)
3. **Write / review / iterate** until each section passes review.
4. **Coherence / editor pass** — real editing, not concatenation: resolve repetition, terminology drift, cross-references.
5. **Assemble** the final markdown under `reports/`. Write the **abstract last**.
6. **Optional PDF** via pandoc + a CSL citation style.

## Structure

Title · Abstract/Summary · Methods · Results · Discussion · References.

- **Results** = selected findings rendered into narrative.
- **Methods** = the union of those findings' methods + their pinned script and environment provenance (comes essentially for free).
- **Figures** = the exact artifacts the findings already point to.
- **References** = the findings' fact-checked references + software/environment citations, deduped at compile time.

## Integrity rules (hard — the report reviewer enforces these)

- **Claim-source checking.** A writer may assert only what traces back to a finding or a vetted research finding. Every sentence in Results and Discussion maps to a finding id, or it is flagged.
- **Caveats and status propagate.** An `exploratory` finding must never be written up with the confidence of a `validated` one — that re-hides the multiplicity the whole system worked to expose. Status and caveats survive into the prose.
- **No invented references.** References come only from the (fact-checked) findings and the environment; the reviewer re-checks any that appear.

## Workflow state

The first time a report is compiled, raise `state/workflow.json` `current_stage` to **6** (highest stage reached) and bump `updated`. Reporting is a loop with no per-stage `*_done` flag; raising `current_stage` keeps `status` from stalling at an earlier stage.
