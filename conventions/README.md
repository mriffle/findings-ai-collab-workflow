# conventions/

The convention/correctness specs that reviewer agents and hooks enforce (spec docs 05, 06, 07). A convention here is only real if the enforcement map (doc 05.5) assigns it to a hook or a reviewer agent.

Documents (all present):

- **`findings.md`** — the finding object: frontmatter schema, the status state machine, the `validated` bar, the edge ontology, exploratory-vs-confirmatory phase, the recording trigger policy (doc 03).
- **`manifest.md`** — the `findings/manifest.json` derived-index schema (doc 03.7).
- **`workflow-state.md`** — the `state/workflow.json` progress + integrity-gate contract (doc 02).
- **`coding.md`** — Python, locked environments, seeds, non-interactive parameterized scripts, fail-loud data handling, max testing/typing/linting; names tool choices (uv, pytest, mypy, ruff) (doc 05.2).
- **`statistics.md`** — no bare p-values, named multiple-testing correction (BH/FDR), report all tests, canonical/moderated models, no leakage, group CV matched to target, label-shuffle null, power honesty (doc 05.3).
- **`correctness.md`** — the charter: the common-mode argument, the two loader obligations, domain fidelity traps, assumptions-as-hypotheses, double-checking critical quantities (doc 05.4).
- **`visualization.md`** — accuracy, dual export, legends, Okabe–Ito + color registry, the >8-category rule, figure provenance (doc 06).
- **`reporting.md`** — report-as-projection, the two modes, claim-source checking, status/caveat propagation (doc 07).
- **`enforcement-map.md`** — the master index: every rule → its hook / reviewer agent / orchestrator behavior / human checkpoint (doc 05.5).
