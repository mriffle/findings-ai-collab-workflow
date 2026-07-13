# 03 — Findings and the Knowledge Graph

This is the centerpiece. Everything else exists to make the objects defined here trustworthy, durable, and useful.

## 3.1 The finding as a first-class object

A finding is a structured, uniquely numbered document capturing one substantive insight, recorded the moment it emerges from the scientist–agent exchange. It is simultaneously human-readable (a markdown narrative with inline figures) and machine-readable (a YAML frontmatter block that the manifest and graph tooling consume). The durable artifact is never the conversation; it is this document plus the pinned, regenerable code that produced its numbers.

### File and naming

- One file per finding: `findings/NNNN-slug.md`, where `NNNN` is a zero-padded unique integer assigned in order.
- A machine-readable **frontmatter** block (YAML) carries the structured fields; the **body** carries the human narrative.

### Finding schema (frontmatter)

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | int | yes | Unique, ordered |
| `title` | string | yes | Human label |
| `status` | enum | yes | See state machine (3.2) |
| `created` | date | yes | First recorded |
| `updated` | date | yes | Last modified |
| `summary` | string | yes | One–two sentence claim |
| `verdict` | string | yes | Current bottom line in plain terms |
| `phase` | enum | yes | `exploratory` or `confirmatory` (3.6) |
| `kind` | enum | no | `discovery` (default) or `caveat` — a dataset/design caveat (imbalance, confound, cohort skew) vs an analysis result (conventions/findings.md §2.6) |
| `entities` | list[id] | yes if applicable | Normalized entity references (3.4): UniProt/HGNC/Reactome/MONDO/etc. |
| `relationships` | list[edge] | no | Typed links to other findings (3.3) |
| `provenance` | object | yes | `data_version` (hash), `script` (path+commit), `params`, `environment` (lock ref) |
| `evidence` | list | yes | Effect sizes, statistics, intervals, corrected p-values (*as built:* a list of measurement objects — see `conventions/findings.md` §2.3) |
| `figures` | list[figure] | conditional | SVG/PNG artifacts (doc 06), regenerable. **Required whenever a relevant figure exists** — every such figure is embedded inline in the body *and* listed here with its own producing script + input (per-figure provenance; conventions/findings.md §2.4) |
| `references` | list[ref] | conditional | Required for any background/interpretive claim (doc 04 invariant) |
| `validation` | object | no | Which validation senses cleared, by whom/what, concordance result |
| `integrity_signoff` | bool | yes | Data passed the integrity gate (doc 05) |

**Most findings are discoveries** (analysis results). A `kind: caveat` finding instead records a structural property of the dataset or design — a class imbalance, confound, or cohort skew — that constrains what downstream results can claim. Caveat findings use the same schema and lifecycle; they surface in Stage 1, carry descriptive evidence, and have `integrity_signoff` set at the integrity gate (3.9; conventions/findings.md §2.6).

### Finding body (sections)

`Summary` · `Verdict` · `Evidence` (numbers, with inline figures/tables) · `Methods / how to produce` (sufficient to regenerate; references the promoted script) · `Discussion` (meaning, why interesting) · `Caveats` (confounds, assumptions, multiplicity context) · `Follow-ups` · `Related findings` · `References`.

Every figure relevant to the finding is **embedded inline** in the body (normally in Evidence), shown with its caption and a one-line pointer to the producing script + input — the finding is a comprehensive, standalone artifact, never a pointer to figures a reader must go find (conventions/findings.md §2.4).

## 3.2 Status — the lifecycle state machine

Status is a position in a state machine with evidentiary bars on transitions, not a free-text label. States:

- `candidate` — captured during exploration; low bar; may be incomplete.
- `under_exploration` — being actively investigated; evidence accumulating.
- `validated` — cleared the independent-validation gate **and** the bar for its phase.
- `invalidated` — failed validation or contradicted by stronger evidence.
- `superseded` — replaced by a later finding that refines or subsumes it.
- `closed` — retired (e.g. merged, withdrawn, no longer relevant).

Transition rules (minimum bar):

- `candidate → under_exploration`: someone is actively pursuing it.
- `under_exploration → validated`: passes independent validation (3.5) **and** the phase bar (3.6). The data used to *generate* a hypothesis may not be the data that *validates* it.
- `→ invalidated`: validation fails, or a `contradicts` edge from a stronger finding resolves against it.
- `→ superseded`: a `supersedes` edge from another finding is asserted and accepted.

The exact bar for `validated` is the project's definition of rigor and is the single most important parameter to fix before implementation; the recommended default combines independent re-derivation with the phase bar.

## 3.3 Relationships — the edge ontology

Findings reference one another through a **controlled vocabulary** of directed edge types. Implement exactly these (extend only deliberately):

- `supports` — A provides evidence consistent with B.
- `refines` — A narrows or sharpens B without contradicting it.
- `contradicts` — A's evidence opposes B's. Triggers reconciliation (one may move to `invalidated`).
- `supersedes` — A replaces B; B moves to `superseded`.
- `closes` — A resolves an open question raised by B.
- `relates_to` — generic association (use sparingly; prefer a specific type).

Edges are directed and recorded in the source finding's frontmatter. The findings manager maintains reverse-edge consistency.

## 3.4 The findings graph and the knowledge graph

There are **two layers**, and conflating them is a design error:

1. **Claim layer (the findings graph).** Findings as nodes; the edges above as relationships. This is an *argumentation / evidence* graph — how our claims relate to each other. This is the primary artifact.
2. **Entity layer (the knowledge graph).** The domain entities a finding refers to — proteins, genes, pathways, diseases — which exist in the world independently. When findings reference these by **normalized identifier** (UniProt accession, HGNC symbol, Reactome ID, MONDO ID, etc.) rather than free text, the corpus becomes queryable: "all validated findings involving the proteasome," "which findings contradict each other," "the most connected entities." Entity normalization is the single step that elevates a pile of linked notes into something that legitimately reasons like a knowledge graph.

**Terminology guidance:** default to "findings graph" — it is precise and over-promises nothing. Use "knowledge graph" deliberately, for the integrated findings-plus-entities layer, with a one-line definition. Serialize as markdown + a manifest; note that it *can be projected* into a formal graph representation (or RDF, à la nanopublications) for querying, without requiring a graph database in the core tool.

Normalization ties directly to the references invariant (doc 04) and the identifier-integrity discipline (doc 05): entities are referenced by canonical ID, and those IDs are verified.

## 3.5 Independent validation

No finding is `validated` without re-derivation by a process independent of the one that produced it. Three senses, certifying different claims (the project requires independent validation as a principle; which combination defines `validated` is set in 3.2):

- **Computational reproduction** — re-run the exact code on the exact data; expect identical numbers. Catches reporting bugs; says nothing about truth. Cheap; should be automatic given pinned provenance.
- **Analytic replication (the blinded one)** — an independent agent, told the *question* but not the *answer* and ideally not the *method*, reaches its own result. Guards against forking-paths fragility.
- **Data replication** — the effect survives in held-out samples or an orthogonal dataset not used to generate the hypothesis. The strongest bar; the only one that addresses overfitting.

**Mechanism (avoiding contamination):**

- Use a fresh subagent with **no conversation history** — the clean context is itself the blind. It never saw the back-and-forth that generated the excitement.
- Derive the verification task **mechanically from the finding's structured fields** (comparison, feature, question), with the answer fields (`evidence`, `verdict`) programmatically stripped. The agent that knows the result never writes a free-prose prompt that could leak it.
- You cannot blind the verifier to the *question* (it must know what to check); you blind it to the *answer* and ideally the *method*.
- **Pre-specify the concordance criterion before the verifier runs** (same sign past the same threshold, effect within an interval, etc.). Deciding "close enough" after seeing the result reintroduces forking paths at the validation stage.
- **Common-mode caveat:** the verifier typically reads the same data through the same loader, so a loader bug is reproduced on both sides and they agree on a falsehood. Validation therefore *assumes* the integrity gate (doc 05); it does not substitute for it.

Record the outcome in the finding's `validation` object: which senses cleared, by what, and the concordance result.

## 3.6 Exploratory vs confirmatory (multiplicity honesty)

Every finding carries a `phase`. The hard rule: **the data used to generate a hypothesis cannot be the data used to validate it.** Exploratory findings are hypothesis-generating and are held to a lower bar but are explicitly marked; confirmatory findings clear a held-out or orthogonal check. This is how the forking-paths problem is made visible rather than buried. The system also keeps an **exploration log** — what was looked at and discarded — so the multiplicity context that informs a finding's `caveats` is auditable, not lost.

## 3.7 The manifest

`findings/manifest.md` is the graph index — compact enough to reason over as findings grow into the hundreds, with full findings pulled on demand. *(Implementation divergence: as built this is Markdown — `findings/manifest.md`, not the `.json` this section originally sketched; the file-format convention reserves JSON for files a non-LLM parser consumes. See `conventions/manifest.md`.)* Per finding it holds: `id`, `title`, `status`, `phase`, `entities`, `relationships`, `updated`, and `data_version`. It is the structure the findings-manager queries to judge novelty, detect relationships, and run consistency checks. It is regenerable from the finding files (the files are the source of truth; the manifest is a derived index).

## 3.8 The findings manager (subagent)

A dedicated subagent owns the findings graph (full contract in doc 04). Responsibilities:

- assign IDs and write/update finding files and the manifest;
- on each new finding, judge whether it is novel or relates to existing findings, and whether it validates/contradicts/supersedes anything on record;
- maintain graph consistency: reverse edges, and **cascade on invalidation/supersession** — when a finding falls, detect dependents (incoming `supports`/`refines`/`closes` edges) and flag them for re-review;
- detect **staleness**: when `data_version` or a linked script's commit changes, flag findings generated against the older version for re-verification.

## 3.9 Recording trigger policy

Findings are recorded automatically, but "automatic" needs a policy (fix before implementation):

- **What counts:** a tangible, specific, evidence-bearing observation about the data or its biology — not every remark. Heuristic: if it has an effect, a statistic, or a concrete claim that someone might later cite, record it. This includes **caveat findings**: a class imbalance, covariate skew, or confound found while characterizing the metadata (Stage 1) is exactly such an observation when it would change how a downstream result is analyzed or interpreted — record it as `kind: caveat` (its `integrity_signoff` is set at the integrity gate, which certifies the pairing it rests on).
- **Candidate vs promoted:** capture is cheap and low-bar (candidate); rigor is applied at promotion. Bias toward capturing too much rather than too little — clutter is cheaper than lost insight, and the manager can merge/close duplicates.
- **Silent vs confirmed:** default to recording with a brief, non-disruptive notice to the scientist ("recorded as finding 0042"), so the exploration flow is not broken, while keeping the scientist aware. Promotion to `validated` is never silent.
