# Convention — The Findings Manifest

*Authoritative format for `findings/manifest.md`. Spec source: doc 03.7 (which sketched a JSON index; we use Markdown — see "Why Markdown" below). Consumed by the findings-manager (and read by the `status` command and reporting).*

## What it is

`findings/manifest.md` is the **graph index** — a compact projection of the findings graph that is cheap to load and reason over as findings grow into the hundreds, with full finding documents pulled on demand.

> **The finding files are the source of truth; the manifest is a *derived* index.** It is regenerable in full by scanning `findings/NNNN-slug.md` frontmatter. If the manifest and a finding file disagree, the finding file wins and the manifest is rebuilt. Never hand-edit the manifest into inconsistency with the files.

It is the structure the findings-manager queries to judge **novelty**, detect **relationships**, and run **consistency checks**.

## Why Markdown (not JSON)

This index is written and read by **LLM agents** (the findings-manager maintains it; the `status` command and writers read it) — it has no non-LLM parser. Markdown + a YAML frontmatter block is more robust for an LLM to emit correctly (no brace/comma/quote fragility), supports comments/notes, and is human-glanceable. JSON in this project is reserved for the files a non-LLM parser actually consumes (`state/workflow.json` → the Python hook; `state/color_registry.json` → Python plotting). See the format convention in the engine `CLAUDE.md`.

## Format

A YAML frontmatter block for the manifest's own metadata, then one Markdown table with one row per finding:

```markdown
---
schema_version: 1
generated: 2026-06-22      # date last regenerated (YYYY-MM-DD)
next_id: 43                # next finding id to assign (monotonic; see IDs)
engine_version: 0.1.0      # plugin version that last wrote it
---

# Findings manifest

Derived index of the findings graph — regenerable from the finding files (the files are the source of truth). One row per finding.

| ID | Slug | Title | Status | Phase | Kind | Entities | Relationships | Updated | Data version |
|----|------|-------|--------|-------|------|----------|---------------|---------|--------------|
| 42 | drug-a-upregulates-tp53 | Drug A upregulates TP53 | validated | exploratory | discovery | uniprot:P04637, hgnc:HGNC:11998 | supports:12, refines:7 | 2026-06-22 | sha256:9f86d08… |
| 43 | tp53-targets-enriched | p53 targets enriched in responders | under_exploration | exploratory | discovery | reactome:R-HSA-69541 | supports:42 | 2026-06-22 | sha256:9f86d08… |
| 7  | sex-confounded-with-group | Sex confounded with treatment group | candidate | exploratory | caveat | — | — | 2026-06-22 | sha256:9f86d08… |
```

### Columns (the doc 03.7 field set, plus `Slug` for file resolution and `Kind` for caveat queries)

| Column | From the finding | Purpose |
|---|---|---|
| `ID` | `id` | Identity; filename resolution with `Slug`. |
| `Slug` | filename | Resolve `findings/<id>-<slug>.md` without a directory scan — also how the findings-manager builds and repairs **cross-reference links** between finding bodies (`conventions/findings.md` §2.7). A slug change breaks inbound links, so a rename must repair them. |
| `Title` | `title` | Human label in listings. |
| `Status` | `status` | Filter/queries; consistency checks. |
| `Phase` | `phase` | Multiplicity-honest filtering (exploratory vs confirmatory). |
| `Kind` | `kind` | Separate dataset/design **caveats** from discoveries — lets reporting pull the Limitations set and queries exclude caveats (`conventions/findings.md` §2.6). Defaults to `discovery`; shown explicitly. |
| `Entities` | `entities` | Knowledge-graph queries. Compact `db:id` list, comma-separated (labels live in the file). |
| `Relationships` | `relationships` | Graph edges. Compact `type:target` list, comma-separated. |
| `Updated` | `updated` | Recency. |
| `Data version` | `provenance.data_version` | Staleness detection (doc 03.8). |

Use `—` for an empty cell. The manifest deliberately **omits** the heavy fields (`evidence`, `verdict`, `provenance` detail, body) — those are pulled from the file on demand. Keeping it lean is what lets it scale.

## IDs

- `next_id` (frontmatter) is the single source for the next id; the findings-manager assigns it, then increments. IDs are **never reused**, even after a finding is `closed`/`invalidated` — this keeps edge targets stable.
- The findings-manager is the only writer of `next_id`; concurrent finding creation is serialized through it (it owns the graph, doc 04.2).

## Reverse edges

Edges are stored **once**, in the source finding's `relationships` (and mirrored in the `Relationships` column). Reverse lookups ("what points at finding 7?") are **computed** by scanning the column for `…:7` — not stored separately. At hundreds of findings this scan is trivial. The findings-manager guarantees every edge target resolves to an existing finding (no dangling edges).

## Queries the manifest is built to answer

- **Novelty / dedup** — does a new observation overlap an existing finding's entities + claim?
- **Relationship detection** — which existing findings does a new one support/refine/contradict/supersede?
- **Knowledge-graph slices** — all `validated` findings involving a given entity; the most-connected entities; mutually `contradicting` findings.
- **Consistency** — dangling edge targets, reverse-edge agreement, status-vs-edge invariants (e.g. a `superseded` finding has an incoming `supersedes`).
- **Staleness** — findings whose `Data version` no longer matches the current dataset stamp, or whose linked script commit moved (doc 03.8).

## Regeneration

The manifest is rebuilt from the finding files at any time: scan each `findings/NNNN-*.md`, read frontmatter, project the columns above into the table, recompute `next_id = max(id)+1`. A rebuild is the recovery path whenever the manifest is suspected stale or corrupt.
