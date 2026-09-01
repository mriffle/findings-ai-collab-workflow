---
# Finding document — schema defined in conventions/findings.md.
# Frontmatter is machine-readable; the body below is the human narrative.
# Fields marked (required) must be present. Replace every <placeholder>.

id: <NNNN>                              # (required) int, assigned in order by the findings-manager
title: "<human label>"                  # (required)
status: candidate                       # (required) candidate | under_exploration | validated | invalidated | superseded | closed
phase: exploratory                      # (required) exploratory | confirmatory
kind: discovery                         # (optional) discovery (default) | caveat — caveat = a dataset/design caveat (class imbalance, confound, cohort skew) that constrains downstream claims; see conventions/findings.md §2.6
created: <YYYY-MM-DD>                    # (required)
updated: <YYYY-MM-DD>                    # (required)

summary: "<one or two sentence claim>"  # (required)
verdict: "<current bottom line, plain terms>"   # (required) — stripped before blind validation

# Normalized domain-entity references (required when the finding is about identifiable entities).
# Reference by canonical ID, never free text. db ∈ uniprot|hgnc|ensembl|reactome|go|mondo|...
entities:
  - { db: uniprot, id: "<accession>", label: "<symbol>" }

# Typed, directed links to other findings. type ∈ supports|refines|contradicts|supersedes|closes|relates_to
relationships: []
  # - { type: supports, target: <id>, note: "<why>" }

# (required) How to regenerate the numbers. Before `validated`, script.path must be under scripts/promoted/.
provenance:
  data_version: "<sha256:... or version stamp>"
  script: { path: "scripts/scratch/<file>.py", commit: "<short-sha>" }
  params: {}                            # analysis params incl. preprocessing — e.g. { sample_set: "experimental", n_experimental: 40, n_controls_excluded: 6, normalization: "median", log2: true, batch_correct: { method: "combat", batch_column: "<batch column>", covariate_preserved: null } }. Record the analyzed sample set (controls excluded — see conventions/statistics.md); the normalization method; for batch correction, record the batch column and that no covariate was preserved (batch-label-only).
  environment: "<lockfile ref, e.g. env/uv.lock@commit>"
  seeded_from: null                     # { template: "<lib template>", version: "<ver>" } if adapted from a lib/ template (e.g. wide-data-loader, normalize, batch-correct-combat); null if from scratch
  seed: null                            # required where anything stochastic ran
  result_id: null                       # for a finding built on a cached CPU-heavy result (classification/xgboost/regression/boruta): the result-io fingerprint it was rendered from (conventions/results-cache.md). Pins the exact cached result; null otherwise.

# (required) The numbers. No bare p-values: effect size + CI + corrected p with the correction named.
evidence:
  - metric: "<e.g. log2FC>"
    value: <number>
    ci: [<low>, <high>]
    p_value: <raw or null>
    p_adjusted: <corrected or null>
    correction: "<e.g. BH>"
    test: "<e.g. limma moderated t>"
    n: <sample size>
    note: "<what this measures>"

# Regenerable figure artifacts (dual export + separate legend image, per doc 06).
# SHOW, DON'T TELL: every claim this finding makes about the data that CAN be shown gets a
# figure — commissioned with the claim, not gathered afterwards. List every such figure here
# AND embed it inline in the body AND explain it in the prose (see ## Evidence). Each entry
# carries its OWN producing script + input (per-figure provenance) so that figure is
# regenerable on its own.
figures: []
  # - { png: "figures/<NNNN>-<name>.png", svg: "figures/<NNNN>-<name>.svg",
  #     legend_png: "figures/<NNNN>-<name>.legend.png", legend_svg: "figures/<NNNN>-<name>.legend.svg",
  #     caption: "<free-text caption>",
  #     script: { path: "scripts/promoted/<figure_script>.py", commit: "<short-sha>" },
  #     data_version: "<sha256:... or version stamp>",
  #     result_id: null,          # the result-io fingerprint if rendered from a cached result (classification/xgboost/regression/boruta); null otherwise
  #     params: {} }              # optional render params

# Required for any background/interpretive claim. Each is fact-checked by the research-reviewer.
references: []
  # - { id: "doi:<...>", type: doi, claim: "<what it supports>", verified: false }

# Set by the validation subsystem. Analytic replication (blinded) is required for `validated`.
validation:
  computational_reproduction: { status: not_attempted }
  analytic_replication:       { status: not_attempted }
  data_replication:           { status: not_attempted }   # required only to claim `confirmatory`

# (required) true only if the data cleared the integrity gate (doc 05). Must be true before `validated`.
integrity_signoff: false
---

# <title>

## Summary
<One or two sentences: what was observed and the claim it supports.>

## Verdict
<The current bottom line in plain terms. Honest about confidence and phase.>

## Evidence
<The numbers behind the claim — effect sizes, intervals, corrected p-values — shown with inline figures/tables. Mirrors the `evidence` frontmatter in narrative form.>

<**Show, don't tell.** Every claim here that *can* be shown carries a figure: ask "what figure shows this?" and commission it with the claim. Embed each one inline, right where it is discussed — the reader must never have to go find it. Every figure embedded here is also listed in the `figures` frontmatter (and vice-versa). Write each in **four parts**: the claim → the figure → its caption + provenance pointer → **the reading** (how to see the claim in the picture). The reading is required; a caption is a label, not an explanation. Keep the words here, not on the canvas (`conventions/visualization.md`, *The annotation budget*).>

<!-- Example (delete if the finding has no figures):

<The claim, in prose — what the figure below is being shown to establish.>

![<caption — what each axis/series/color encodes, units, n>](figures/<NNNN>-<name>.png)

*Figure 1. <caption>. Produced by `scripts/promoted/<figure_script>.py` (<commit>) from data `<data_version>`[, result `<result_id>`]. Legend: `figures/<NNNN>-<name>.legend.png`.*

<The reading — one or two sentences: what is plotted, where to look, and what that establishes for the claim.>
-->


## Methods / how to produce
<Enough to regenerate: the promoted script invoked, its parameters, the data version, and the environment. Point at `provenance`. A reader (or a blinded verifier) could reproduce the numbers from this section + the pinned code.>

## Discussion
<What it means and why it is interesting. Interpretive/background claims here require `references`.>

## Caveats
<Confounds, assumptions, and — for `exploratory` findings — the multiplicity context: what else was tested, what the forking-paths exposure is. This survives into any report; it must not be quietly dropped.>

## Follow-ups
<Open questions and next analyses this finding suggests.>

## Related findings
<Narrative companion to the `relationships` edges: how this connects to other findings.>

## References
<Rendered from `references`. Literature with identifiers; software/tools with versions. Every entry fact-checked.>
