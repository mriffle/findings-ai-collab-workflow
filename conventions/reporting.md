# Convention — Reporting

*Spec source: doc 07. A report is **not** a fresh act of writing — it is a **projection of selected nodes of the findings graph** into prose. Enforced by the **report-reviewer/editor**. The full reporting subsystem (writer/reviewer agents, report skills) is built in the reporting phase; this convention is the contract they implement.*

## A report is a projection of the findings graph

Because findings already carry evidence, figures, methods, caveats, references, and status:

- **Results** = selected findings rendered into narrative.
- **Methods** = the union of those findings' methods + their pinned script and environment provenance (comes essentially for free).
- **Figures** = the exact artifacts the findings already point to (dual-exported, with legends).
- **References** = the findings' fact-checked references + software/environment citations, deduped and styled at compile time.

Writers **select, order, and add connective tissue; they do not invent content.** The report inherits the findings' rigor and traceability.

## Two modes — same machinery, opposite selection norms

- **QC / data-quality report** — exhaustive and descriptive; the point is completeness, so collaborators can trust the data. Produced largely from Stage 3 outputs and the project state files.
- **Research report** — selective and narrative; a **disseminable artifact that supports downstream manuscript preparation**, not a manuscript itself. The primary deliverable. Findings are deliberately left out to tell a focused story.

A writing agent must **know which mode it is producing**, because the selection discipline is inverted between them.

## Structure

Title · Abstract/Summary · Methods · Results · Discussion · References.

Methods are driven by the scripts and the methods recorded in findings. Results are the selected findings with their supporting evidence, figures, and tables. Discussion interprets them. **Write the abstract last**, even though it sits first.

## Process

1. **Select + outline** — choose which findings the report is about and in what order. A scientific judgment, not automatable: a **human checkpoint** (doc 02.8).
2. **Dispatch per section** — hand each logical section to a writer/reviewer pair with a **shared spec** (glossary, defined abbreviations, the agreed narrative) so parallel writers don't drift in terminology or repeat each other.
3. **Write / review / iterate** until each section passes review.
4. **Coherence / editor pass** — real editing, not concatenation: resolve repetition, terminology drift, cross-references.
5. **Assemble** the final markdown under `reports/`.
6. **Optional PDF** via pandoc + a CSL citation style.

## Integrity rules (hard — the report-reviewer enforces)

Reporting is where an LLM most easily overstates, so:

- **Claim-source checking.** A writer may assert only what traces back to a finding or a vetted research finding. **Every sentence in Results and Discussion maps to a finding id**, or it is flagged.
- **Caveats and status propagate.** An `exploratory` finding must **never** be written up with the confidence of a `validated` one — that quietly re-hides the multiplicity the whole system worked to expose. Status and caveats from the source findings must survive into the prose. **Caveat findings** (`kind: caveat`, conventions/findings.md §2.6) — the cohort's class imbalances, skews, and confounds — are rendered into the report's Discussion (its Limitations) and attached to each discovery they qualify via its `relates_to` edge; a confounded or imbalanced result must carry its caveat wherever it is stated.
- **No invented references.** References come only from the (fact-checked) findings and the environment; the reviewer re-checks any that appear.

## Enforcement

| Rule | Enforced by |
|---|---|
| Every Results/Discussion claim maps to a finding id | **Report-reviewer** |
| Status + caveats propagate (exploratory ≠ validated confidence); `kind: caveat` findings rendered as Limitations | **Report-reviewer** |
| No invented references; all re-checked | **Report-reviewer** (+ research-reviewer's prior fact-check) |
| Figures are the findings' existing artifacts | **Report-reviewer** |
