# 07 — Reporting

## 7.1 A report is a projection of the findings graph

Reports are **not** a fresh act of writing. Because findings already carry evidence, figures, methods, caveats, references, and status (doc 03), a report is a *projection* of selected nodes of the findings graph into prose:

- **Results** = selected findings rendered into narrative.
- **Methods** = the union of those findings' methods plus their pinned script and environment provenance.
- **Figures** = the exact artifacts the findings already point to (doc 06).
- **References** = the findings' references (literature, fact-checked per doc 04) plus software/environment citations (doc 05), with style and dedup applied at compile time.

Writers select, order, and add connective tissue; they do not invent content. The report thereby inherits the findings' rigor and traceability, and the Methods/reproducibility material comes essentially for free from provenance.

## 7.2 Two report modes

Same machinery, opposite selection norms:

- **QC / data-quality report** — exhaustive and descriptive. The point is completeness, so collaborators can trust the data. Produced largely from Stage 3 outputs and the project state files.
- **Research report** — selective and narrative; a **disseminable artifact that supports downstream manuscript preparation** rather than a manuscript itself. Findings are deliberately left out to tell a focused story. This is the primary report deliverable.

A writing agent must know which mode it is producing, because the selection discipline is inverted between them.

## 7.3 Structure

Title · Abstract/Summary · Methods · Results · Discussion · References.

Methods are driven by the scripts and the methods recorded in the findings. Results are the selected findings with their supporting evidence, figures, and tables. Discussion interprets them — what they mean, why they are interesting.

## 7.4 Process

1. **Select + outline** — choose which findings the report is about and in what order. This is a scientific judgment, not an automatable step, and is a **human checkpoint** (doc 02.8). It is effectively a finding-selection step.
2. **Dispatch per unit** — hand each logical section to a writer/reviewer pair (doc 04), with a **shared spec** (glossary, defined abbreviations, the agreed narrative) so parallel writers don't drift in terminology or repeat each other.
3. **Write / review / iterate** — until each section passes review.
4. **Coherence / editor pass** — the stitch step is real editing, not concatenation: resolve repetition, terminology drift, and cross-references.
5. **Assemble** the final markdown.
6. **Optional PDF** for dissemination, via pandoc plus a CSL citation style.

Write the abstract last, even though it sits first, since it summarizes the assembled whole.

## 7.5 Integrity rules

Reporting is where an LLM most easily overstates, so these are hard rules enforced by the report reviewer:

- **Claim-source checking.** A writer may assert only what traces back to a finding or a vetted research finding. Every sentence in Results and Discussion maps to a finding ID, or it is flagged.
- **Caveats and status propagate.** An `exploratory` finding must never be written up with the confidence of a `validated` one. If it is, the report has quietly re-hidden the multiplicity the whole system worked to expose (doc 03.6). Status and caveats from the source findings must survive into the prose.
- **No invented references.** References come only from the (fact-checked) findings and the environment; the reviewer re-checks any that appear.
