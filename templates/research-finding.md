---
# Research-finding document — external knowledge, structured like a finding but for the literature/tools.
# Spec: doc 04.4. Stored in research/<slug>.md. A research finding without verified references is NOT accepted.

topic: "<what was researched>"          # (required)
type: general                            # protein | gene | disease | pathway | publication | software | general
created: <YYYY-MM-DD>                     # (required)
updated: <YYYY-MM-DD>                     # (required)

status: draft                            # draft | reviewed  (reviewed = passed the research-reviewer)
reviewed_by: null                        # set by the research-reviewer on acceptance
reviewed_date: null

# Normalized domain-entity references this research is about (same scheme as findings; canonical IDs).
entities:
  - { db: uniprot, id: "<accession>", label: "<symbol>" }

# (required, non-empty) Every external claim traces to a reference here. Each is fact-checked by the
# research-reviewer: it must EXIST and SUPPORT the claim attributed to it. Hallucinated citations are
# the known failure mode this section + review exist to stop.
references:
  - id: "<doi:... | pmid:... | url:... | software-name@version>"
    type: doi                            # doi | pmid | url | software
    claim: "<the specific claim this source supports>"
    verified: false                      # set true by the research-reviewer
    verified_by: null
---

# Research: <topic>

## Summary
<The bottom line in a few sentences. Every interpretive statement here is backed by an entry in `references`.>

## Detailed findings
<The thorough write-up of what was learned, organized for reuse by the librarian and by findings that cite it. Each non-trivial claim carries an inline citation to a `references` entry. Distinguish established fact from contested/uncertain claims.>

## Methods / sources consulted
<How this was researched — which databases/tools/papers, what was searched, what was verified (e.g. UniProt accession confirmed; limma source inspected). Lets the research-reviewer retrace the work.>

## References
<Rendered from `references`: specific papers (with DOI/PMID), specific web sources (with URLs), and software/tools (with versions). No general "see the literature" hand-waving — specific, checkable sources only.>
