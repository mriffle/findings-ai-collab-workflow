# Convention — The Finding Object

*Authoritative schema for findings. Spec source: doc 03. The finding template (`templates/finding.md`), the manifest schema (`conventions/manifest.md`), the findings-manager agent, and the validation subsystem all consume this document. When in doubt, this file defines the shape; the spec defines the intent.*

A **finding** is a structured, uniquely numbered markdown document capturing one substantive insight, recorded the moment it emerges from the scientist–agent exchange. It is simultaneously:

- **human-readable** — a markdown narrative with inline figures (the body), and
- **machine-readable** — a YAML frontmatter block the manifest and graph tooling consume.

The durable artifact is never the conversation. It is this document **plus the pinned, regenerable code that produced its numbers.**

## 1. File and naming

- One file per finding: `findings/NNNN-slug.md`.
- `NNNN` is a zero-padded (4-digit minimum) unique integer assigned **in order** by the findings-manager. `slug` is a short kebab-case label derived from the title.
- The frontmatter carries structured fields; the body carries the human narrative.
- The finding files are the **source of truth**; `findings/manifest.md` is a derived index regenerable from them (doc 03.7; format in `conventions/manifest.md`).

## 2. Frontmatter schema

All findings carry this YAML frontmatter. `R` = required, `C` = conditional, `O` = optional.

| Field | Req | Type | Notes |
|---|---|---|---|
| `id` | R | int | Unique, ordered. Matches `NNNN` in the filename. |
| `title` | R | string | Human label. |
| `status` | R | enum | One of the six states in §3. |
| `phase` | R | enum | `exploratory` or `confirmatory` (§5). |
| `kind` | O | enum | `discovery` (default) or `caveat` — a dataset/design caveat vs an analysis result (§2.6). |
| `created` | R | date | `YYYY-MM-DD`, first recorded. |
| `updated` | R | date | `YYYY-MM-DD`, last modified. |
| `summary` | R | string | One–two sentence claim. |
| `verdict` | R | string | Current bottom line, plain terms. **Stripped before blind validation (§4).** |
| `entities` | C | list[entity] | Normalized domain-entity references (§2.1). Required when the finding is about identifiable entities. |
| `relationships` | O | list[edge] | Typed links to other findings (§6). |
| `provenance` | R | object | How to regenerate the numbers (§2.2). |
| `evidence` | R | list[measurement] | Effect sizes, intervals, corrected p-values (§2.3). **Stripped before blind validation (§4).** |
| `figures` | O | list[figure] | Regenerable figure artifacts (§2.4). |
| `references` | C | list[reference] | Required for any background/interpretive claim (§2.5; doc 04.5). |
| `validation` | O | object | Which validation senses cleared, by what, with what concordance (§4). |
| `integrity_signoff` | R | bool | `true` only if the data cleared the integrity gate (doc 05). A finding may not be `validated` while this is `false`. |

### 2.1 `entities` — normalized references

Reference domain entities by **canonical identifier**, never free text. This is the single step that turns linked notes into a queryable knowledge graph (doc 03.4).

```yaml
entities:
  - { db: uniprot,  id: "P04637",     label: "TP53" }
  - { db: hgnc,     id: "HGNC:11998", label: "TP53" }
  - { db: reactome, id: "R-HSA-69541", label: "Stabilization of p53" }
  - { db: mondo,    id: "MONDO:0007254", label: "breast cancer" }
  - { db: go,       id: "GO:0006915", label: "apoptotic process" }
```

- `db` — controlled namespace: `uniprot`, `hgnc`, `ensembl`, `reactome`, `go`, `mondo`, `pubchem`, `chebi`, … (extend deliberately).
- `id` — the canonical accession in that namespace. IDs are **verified** (doc 05 identifier-integrity); an unverifiable ID is a defect.
- `label` — a human label for readability; the `id` is authoritative.

### 2.2 `provenance` — regeneration contract

Every finding pins enough to re-run to the same numbers (principle 2).

```yaml
provenance:
  data_version: "sha256:9f86d08…"          # hash / version stamp of the pinned dataset
  script: { path: "scripts/promoted/de_treatment.py", commit: "abc1234" }
  params: { sample_set: "experimental", contrast: "drug_A_vs_control", fdr: 0.05, min_obs: 3 }
  environment: "env/uv.lock@abc1234"        # reference to the locked environment
  seeded_from: { template: "de-moderated", version: "0.3" }  # the lib/ template this script was adapted from (lineage); null if written from scratch
  seed: 12345                               # required where anything stochastic ran
```

- `script.path` **must be under `scripts/promoted/`** before the finding can be `validated` — a finding may link only to a promoted (reviewed, tested) script (doc 05.1). A `candidate` may temporarily reference a scratch script, but promotion to `validated` requires a promoted path + commit.
- `data_version` is the hash/stamp the staleness machinery compares against (§7, doc 03.8).
- `seeded_from` records **template lineage** — which `lib/` template (and its version) this project-local script was adapted from. It is provenance/attribution, **not** a runtime dependency: the finding is regenerable from the promoted script itself. Lineage lets the findings-manager flag derived scripts for re-review when a `lib/` template is later corrected. `null` for a script written from scratch. The fields map to the source template's `__script_meta__` `template:` block — `seeded_from.template` = its `name`, `seeded_from.version` = its `version`.
- `params` records the **analyzed sample set** for any biological contrast. Control samples (pools/references/standards/blanks) are excluded from the experimental subset that analysis runs on (`conventions/statistics.md`), so record `sample_set` (e.g. `"experimental"`) with the experimental-n and the count of controls excluded — making the exclusion explicit and the filter reproducible.

### 2.3 `evidence` — the numbers

A **list** of measurements. The statistical conventions (doc 05.3) forbid a bare p-value: each significance claim carries an effect size, an interval, and a **corrected** p-value with the correction named.

```yaml
evidence:
  - metric: "log2FC"          # effect size
    value: 1.84
    ci: [1.21, 2.47]          # confidence interval for the effect
    p_value: 0.0003           # raw
    p_adjusted: 0.011         # corrected
    correction: "BH"          # correction method, named
    test: "limma moderated t"
    n: 24                      # sample size behind this measurement
    note: "drug_A vs control, protein P04637"
```

`metric`, `value`, and (for any significance claim) `p_adjusted` + `correction` + a `ci` are expected. Descriptive findings without a hypothesis test may carry effect/estimate measurements without p-values.

**Classification / regression / selection evidence (a distinct shape).** A finding from a predictive-model or selection template — `lib/analysis/classification`, `lib/analysis/classification-xgboost`, `lib/analysis/regression`, and Boruta — does **not** fit the effect + CI + corrected-p mold, and forcing it to would be dishonest. Its evidence is two coupled parts, both recorded:

- **Run-level performance vs a null.** The cross-validated metric with its fold spread — a **classifier's** balanced accuracy / ROC-AUC ± SD (plus average precision under imbalance), a **regressor's** held-out R² / RMSE / MAE ± SD — *and* the **shuffle null** it is tested against (label-shuffle for a classifier, target-shuffle for a regressor; the null distribution's summary + the **empirical p**). Record the generalization target (`"on unseen samples/individuals/batches"`) and the CV design. **Without the null the finding is `exploratory`** — the coefficients/importances are not licensed; running it is what makes the finding eligible for `validated` (`conventions/statistics.md`).
- **Per-feature selection evidence.** For each selected feature, the **all-data point estimate** with its cross-resample **stability read** — its **selection frequency** and the estimate's resample spread. The estimate's form follows the estimator: a **signed standardized coefficient** (with **sign consistency**) for a linear model (elastic-net classifier/regressor), or an **unsigned gain importance** (no sign, so no sign consistency) for a **gradient-boosted-tree** classifier (`classification-xgboost`) — a magnitude that says a feature was *useful for splitting*, not a direction. Either way this is a *selection*, not a per-feature significance test — there is **no per-feature q**, and the *minimal-optimal* caveat (a low selection frequency ≠ unimportant; correlated features) is carried in the finding's `caveats`.

The "no bare p / effect + CI" rule (§2.3) governs *significance* tests; this selection kind satisfies honesty differently — a performance claim earned against an explicit null, plus a per-feature **stability** read rather than a naked coefficient. The stats-reviewer accepts it on those terms.

### 2.4 `figures`

```yaml
figures:
  - png:        "figures/0042-volcano.png"        # 300 DPI raster; review + embed target
    svg:        "figures/0042-volcano.svg"        # vector master
    legend_png: "figures/0042-volcano.legend.png" # legend as a separate IMAGE (doc 06.3)
    legend_svg: "figures/0042-volcano.legend.svg" # legend vector master
    caption:    "Volcano plot of drug_A vs control."  # free-text caption
```

Figures are caches of a script (doc 06). They are covered by the staleness machinery: if `data_version` or the producing script's commit changes, figures built on the old version are flagged.

### 2.5 `references`

Required wherever an external-knowledge or interpretive claim is made (the references invariant, doc 04.5). Every reference is fact-checked by the research-reviewer — that it **exists** and that it **supports the claim** attributed to it.

```yaml
references:
  - id: "doi:10.1038/s41586-020-2649-2"
    type: doi                 # doi | pmid | url | software
    claim: "TP53 stabilization triggers apoptosis under genotoxic stress."
    verified: true            # set by the research-reviewer
    verified_by: "research-reviewer"
```

Software/tool citations (with versions) are drawn for free from the locked environment (doc 05) and recorded the same way (`type: software`).

### 2.6 `kind` — discovery vs caveat

Most findings are **discoveries**: an analysis result about the data or its biology (`kind: discovery`, the default — omit it and it is assumed). A second, equally first-class kind records a **caveat**: a structural property of the dataset or experimental design that constrains what downstream results can claim — a class imbalance, a confound, batch structure, or a cohort skew (e.g. "treatment is 80% male vs 30% control; sex is aliased with the contrast"). Set `kind: caveat` on these. They are how the workflow *remembers* the gotchas that bias interpretation instead of letting them evaporate into prose.

Caveat findings use the **same schema and lifecycle** as discoveries, with three characteristic shapes:

- **Recorded early, certified at the gate.** Metadata caveats surface in Stage 1 (`stage1-metadata`), before the integrity gate, so they are written `candidate` with `integrity_signoff: false`. Their `integrity_signoff` is set `true` at the integrity gate (Stage 3) — the gate certifies the sample↔metadata pairing the caveat rests on (doc 05; `commands/stage3-loaders.md`).
- **Descriptive evidence.** A caveat's `evidence` is typically descriptive — group sizes, a contingency table, a confounding statistic (bias-corrected Cramér's V) — and carries no hypothesis test, which §2.3 already permits. `phase` is normally `exploratory` (a caveat describes the cohort as it is; it makes no held-out claim).
- **Attached to what it qualifies.** When a later discovery is affected by a caveat, the discovery links to it with a `relates_to` edge (§6). That edge is how the caveat travels: the analysis stage consults open caveats (a confounded covariate enters the model; an imbalance dictates balanced metrics), and the report renders them as the discovery's limitations rather than burying them.

**What earns a caveat finding** (the trigger): a metadata observation becomes a finding when it would **change how a downstream result is analyzed or interpreted** — a severe group imbalance, a real confound, a notable covariate skew. A clean, balanced distribution does not need one; the full descriptive characterization lives in `state/METADATA.md` and the QC report (`conventions/statistics.md`). Record the consequential ones, not every histogram.

## 3. Status — the lifecycle state machine

`status` is a position in a state machine with evidentiary bars on transitions, **not** a free-text label.

```
                ┌─────────────┐
   record ─────▶│  candidate  │
                └──────┬──────┘
                       │ someone actively pursues it
                       ▼
              ┌───────────────────┐
              │ under_exploration │
              └─────────┬─────────┘
                        │ INDEPENDENT VALIDATION + PHASE BAR (§4, §5)
                        ▼
                  ┌───────────┐
                  │ validated │
                  └───────────┘

  any state ──▶ invalidated   (validation fails, or a `contradicts` edge from a stronger finding resolves against it)
  any state ──▶ superseded     (a `supersedes` edge from another finding is asserted and accepted)
  any state ──▶ closed          (retired: merged, withdrawn, no longer relevant)
```

States:

- **`candidate`** — captured during exploration; low bar; may be incomplete.
- **`under_exploration`** — actively investigated; evidence accumulating.
- **`validated`** — cleared the bar in §4 **and** the phase bar in §5.
- **`invalidated`** — failed validation, or contradicted by stronger evidence.
- **`superseded`** — replaced by a later finding that refines or subsumes it.
- **`closed`** — retired.

The findings-manager owns transitions and refuses any that don't meet the bar.

## 4. The `validated` bar (this project's definition of rigor)

> **Fixed decision** (doc 03.2 calls this the single most important parameter): **`validated` = independent re-derivation (blinded analytic replication) + the phase bar.** Data replication is encouraged and recorded when available, but is mandatory only when claiming `confirmatory` (§5).

A finding may transition `under_exploration → validated` only when **all** of the following hold:

1. **`integrity_signoff: true`** — the underlying data cleared the integrity gate (doc 05). This is a precondition validation *assumes*; it does not substitute for it (the common-mode loader argument).
2. **Computational reproduction passed** — re-running the exact promoted script on the exact pinned data reproduces the recorded numbers. Cheap; automatic given provenance. Necessary but **not** sufficient (it is not independent — same code, same data).
3. **Analytic replication passed (the independent, blinded sense)** — a fresh-context verifier, told the *question* but not the *answer* (and ideally not the *method*), reaches a result **concordant under a criterion pre-specified before the verifier ran** (§4.1). This is the independent re-derivation that principle 5 requires.
4. **The phase bar (§5) is satisfied** — `phase` is set honestly, and if `phase: confirmatory` then data replication has passed.

The three validation senses (doc 03.5) are recorded in the `validation` object:

```yaml
validation:
  computational_reproduction:
    status: passed            # not_attempted | in_progress | passed | failed
    by: "verifier"
    date: 2026-06-22
  analytic_replication:        # the blinded one — REQUIRED for `validated`
    status: passed
    by: "verifier (clean context, no history)"
    date: 2026-06-22
    concordance_criterion: "same sign AND |log2FC| within 0.3 AND p_adj < 0.05"
    concordance_result: "verifier log2FC 1.79 vs recorded 1.84; same sign; p_adj 0.009 → concordant"
  data_replication:            # required only to claim `confirmatory`
    status: not_attempted
    dataset: null              # held-out split or orthogonal dataset, when attempted
```

### 4.1 Mechanism — avoiding contamination

- **Fresh subagent with no conversation history.** The clean context *is* the blind — it never saw the back-and-forth that generated the excitement.
- **The verification task is derived mechanically from the structured fields** (comparison, feature, question) with the answer fields (`evidence`, `verdict`) **programmatically stripped**. The agent that knows the result never writes a free-prose prompt that could leak it.
- The verifier cannot be blinded to the **question** (it must know what to check); it is blinded to the **answer** and ideally the **method**.
- **Pre-specify the concordance criterion before the verifier runs.** Deciding "close enough" after seeing the result reintroduces forking paths at the validation stage.
- **Common-mode caveat:** the verifier typically reads the same data through the same loader, so a loader bug is reproduced on both sides. Validation therefore *assumes* the integrity gate; it does not replace it.

## 5. Phase — exploratory vs confirmatory (multiplicity honesty)

Every finding carries a `phase`. The hard rule (doc 03.6): **the data used to *generate* a hypothesis cannot be the data used to *validate* it.**

- **`exploratory`** — hypothesis-generating; its effect was derived on the same data it was observed in. May reach `validated` (re-derived + reproduced + blinded-replicated), but **stays marked `exploratory`**, signalling that the forking-paths risk is not yet retired. Its `caveats` must state the multiplicity context. An exploratory finding must never be written up with the confidence of a confirmatory one (doc 07.5).
- **`confirmatory`** — the effect survived a check on **held-out samples or an orthogonal dataset** not used to generate the hypothesis. To set `phase: confirmatory`, `validation.data_replication.status` must be `passed` with the replicating dataset recorded.

The **exploration log** (`findings/exploration-log.md` in the project) records what was looked at and discarded, so the multiplicity context informing a finding's `caveats` is auditable rather than lost.

## 6. Relationships — the edge ontology

Findings reference one another through a **controlled vocabulary** of directed edge types. Implement exactly these (extend only deliberately):

| Type | Meaning | Side effect |
|---|---|---|
| `supports` | A provides evidence consistent with B. | — |
| `refines` | A narrows/sharpens B without contradicting it. | — |
| `contradicts` | A's evidence opposes B's. | Triggers reconciliation; the weaker may move to `invalidated`. |
| `supersedes` | A replaces B. | B moves to `superseded`. |
| `closes` | A resolves an open question raised by B. | — |
| `relates_to` | Generic association — including a discovery's link to a **caveat finding** (§2.6) that qualifies it. | Prefer a specific type elsewhere; the discovery→caveat link is an endorsed use. |

```yaml
relationships:
  - { type: supports,  target: 12, note: "Same direction in the orthogonal cohort." }
  - { type: refines,   target: 7,  note: "Restricts the effect to the high-grade subset." }
```

Edges are **directed** and recorded in the **source** finding's frontmatter. The findings-manager maintains **reverse-edge consistency** and **cascades on invalidation/supersession** (§7).

## 7. Lifecycle integrity (owned by the findings-manager)

- **Reverse edges** are kept consistent: an edge A→B is queryable from B.
- **Cascade on fall:** when a finding moves to `invalidated`/`superseded`, dependents (incoming `supports`/`refines`/`closes` edges) are detected and **flagged for re-review**.
- **Staleness:** when a finding's `data_version` or a linked script's commit changes, the finding (and its figures) are flagged for re-verification — a `validated` finding built on a now-stale version is no longer trusted on autopilot.

## 8. Recording trigger policy (doc 03.9)

Findings are recorded **automatically** during exploration, governed by:

- **What counts** — a tangible, specific, evidence-bearing observation about the data or its biology, not every remark. **Heuristic:** if it has an effect, a statistic, or a concrete claim someone might later cite, record it. This includes **caveat findings** (§2.6): a class imbalance, covariate skew, or confound found while characterizing the metadata (Stage 1) is exactly such an observation when it would change a downstream analysis or its interpretation — record it the moment the metadata is understood, not only once analysis is underway.
- **Candidate vs promoted** — capture is cheap and low-bar (`candidate`); rigor is applied at promotion. **Bias toward capturing too much** — clutter is cheaper than lost insight, and the findings-manager can merge/close duplicates.
- **Silent vs confirmed** — default to recording with a **brief, non-disruptive notice** to the scientist ("recorded as finding 0042"), so exploration flow is not broken while the scientist stays aware. **Promotion to `validated` is never silent.**

## 9. Body sections

The body is the human narrative. Section order:

`Summary` · `Verdict` · `Evidence` (numbers, with inline figures/tables) · `Methods / how to produce` (sufficient to regenerate; references the promoted script) · `Discussion` (meaning, why interesting) · `Caveats` (confounds, assumptions, multiplicity context) · `Follow-ups` · `Related findings` · `References`.
