---
# Report skeleton — a projection of the findings graph (spec doc 07).
# Stored in reports/<slug>.md. Works for both modes; set `mode` and apply the
# matching selection discipline (QC = exhaustive; research = selective).

title: "<report title>"
mode: research                 # research | qc
created: <YYYY-MM-DD>
updated: <YYYY-MM-DD>

# The findings this report projects (the human-selected set, in order). The report
# asserts nothing that doesn't trace to one of these (or a vetted research-finding).
finding_ids: []                # e.g. [42, 7, 13]
research_ids: []               # vetted research-findings cited

# Shared spec for parallel writers — keeps terminology consistent and avoids drift/repeat.
glossary: {}                   # term -> definition
abbreviations: {}              # abbr -> expansion
---

# <title>

<!--
  Integrity rules (the report-reviewer enforces):
  - Every sentence in Results and Discussion maps to a finding id — carry it inline, e.g. [F0042].
  - exploratory findings keep exploratory confidence + their caveats; never write them as validated.
  - No reference appears that isn't in a source finding (+ software/environment citations).
  Write the Abstract LAST.
-->

## Abstract
<Written last. Faithfully summarizes the assembled whole; over-claims nothing.>

## Methods
<The union of the selected findings' methods + their pinned scripts and environment
provenance (data version, template lineage, locked environment). Assembled from provenance,
not reinvented. Software/tools cited with versions.>

## Results
<The selected findings rendered into narrative, in the chosen order, each claim carrying
its finding id [F00NN]. Supporting evidence, figures (the findings' existing svg/png +
legend), and tables. For a QC report: exhaustive and descriptive. For a research report:
selective and focused.>

## Discussion
<What the results mean and why they are interesting — interpretation that still traces to
findings (and vetted research-findings for background). Status and caveats from the source
findings are preserved; exploratory results are framed as hypothesis-generating.>

## References
<The source findings' fact-checked references + software/environment citations, deduped and
styled. No reference that isn't present in a source finding.>
