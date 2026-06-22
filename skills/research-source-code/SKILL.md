---
name: research-source-code
description: >-
  How to ground a methodological claim in what a tool ACTUALLY computes by reading
  its source code and documentation, rather than trusting a methods-section
  description. Use for claims about how software like DIA-NN, limma, MSstats, or a
  search engine processes data. Output feeds a research-finding.
---

# Researching what a tool actually computes

Methods sections — and prompts, and memory — routinely misdescribe what software actually does. This procedure grounds methodological claims in the **source and official documentation**, which is independently valuable: a finding's interpretation often hinges on what a tool's number really means.

## Procedure

1. **Identify the exact tool and version.** Behavior changes across versions; pin the version (from the locked environment where the tool is part of the pipeline).
2. **Go to the source/docs of record.** The official repository, release docs, and the function/module implementing the step in question. Prefer reading the **implementation** over secondary descriptions.
3. **Trace the specific computation.** What transformation/normalization/model does it apply? What are the defaults? What does a given output column actually represent (e.g. what "intensity", "q-value", "LFQ", or a moderated statistic means in *this* tool)? What is silently filtered or imputed?
4. **Note defaults and silent behavior.** Defaults that change results, silent filtering/imputation, and assumptions baked into the tool are exactly what gets misreported — capture them.
5. **Verify by citation to source.** Cite the specific file/function/release-doc (a URL or repo path + version), not a general "the tool does X."

## Output

A research-finding (`templates/research-finding.md`) stating what the tool computes for the step in question, with defaults and caveats, each claim cited to the source/docs (and the version). Software references use `type: software` with the version. It goes to the research-reviewer, which verifies the sources resolve and support the claims. If you cannot confirm a behavior from the source/docs, record it as **unverified** rather than asserting it.
