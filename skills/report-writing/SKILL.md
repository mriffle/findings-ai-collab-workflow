---
name: report-writing
description: >-
  How to assemble a report in a Findings Workflow project as a projection of the
  findings graph: structure, the shared-spec discipline for parallel writers, the
  per-section write/review loop, and the coherence/assembly pass. Use when running
  Stage 6 (the stage6-report command) or drafting/assembling report sections.
---

# Report-writing procedure

Authoritative rules: `conventions/reporting.md`. A report is a **projection of selected findings into prose** — writers select, order, and connect; they never invent content. Methods, figures, and references come essentially for free from the findings' provenance.

## Confirm the mode first

- **QC / data-quality** — exhaustive, descriptive; completeness so collaborators trust the data; built largely from Stage 3 outputs + state files.
- **Research** — selective, narrative; a focused disseminable artifact (not a manuscript). Findings deliberately omitted to tell the story.

The selection discipline is inverted between them — everyone writing must know which one.

## Structure

`Title · Abstract/Summary · Methods · Results · Discussion · References`

- **Results** = selected findings rendered into narrative, each sentence carrying its finding id.
- **Methods** = union of those findings' methods + pinned script/environment provenance.
- **Figures** = the exact artifacts the findings already point to (svg/png + legend), **embedded inline and explained** — show, don't tell: a claim the source finding illustrated stays illustrated, and each figure keeps its caption plus a one-or-two-sentence reading in the prose (`conventions/findings.md` §9).
- **References** = the findings' fact-checked references + software/environment citations, deduped and styled at compile time.

## Process (matches the stage6-report command)

1. **Select + outline** — the scientist chooses which findings the report is about and their order (**human checkpoint**). Record the chosen finding ids.
2. **Build the shared spec** — a glossary, defined abbreviations, the agreed narrative arc, the mode, and per-section finding assignments. This is what keeps parallel writers from drifting in terminology or repeating each other. Distribute it to every writer.
3. **Dispatch per section** — one `writer` + `report-reviewer` pass per logical section, each with the shared spec and its assigned findings.
4. **Write / review / iterate** — until each section passes the report-reviewer (claim-source, status/caveat propagation, no invented references).
5. **Coherence / editor pass** — the report-reviewer stitches: resolve repetition, terminology drift, and cross-references. This is real editing, not concatenation.
6. **Assemble** the final markdown under `reports/`. **Write the abstract last**, since it summarizes the assembled whole.
7. **Optional PDF** — pandoc + a CSL citation style for dissemination.

## Integrity (non-negotiable)

Every Results/Discussion sentence maps to a finding id; `exploratory` findings keep exploratory confidence and their caveats; no reference appears that isn't in a source finding. These are enforced by the report-reviewer — write to them from the start rather than fixing them at review.
