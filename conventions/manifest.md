# Convention — The Findings Manifest

*Authoritative schema for `findings/manifest.json`. Spec source: doc 03.7. Consumed by the findings-manager and any graph/report tooling.*

## What it is

`findings/manifest.json` is the **graph index** — a compact projection of the findings graph that is cheap to load and reason over as findings grow into the hundreds, with full finding documents pulled on demand.

> **The finding files are the source of truth; the manifest is a *derived* index.** It is regenerable in full by scanning `findings/NNNN-slug.md` frontmatter. If the manifest and a finding file disagree, the finding file wins and the manifest is rebuilt. Never hand-edit the manifest into inconsistency with the files.

It is the structure the findings-manager queries to judge **novelty**, detect **relationships**, and run **consistency checks**.

## Schema

```jsonc
{
  "schema_version": "1",          // manifest format version
  "lib_version": "0.1.0",          // engine version that last wrote it
  "generated": "2026-06-22",       // date the manifest was last regenerated (YYYY-MM-DD)
  "next_id": 43,                   // next finding id to assign (monotonic; see §IDs)
  "findings": [
    {
      "id": 42,                                    // int, matches the file's NNNN
      "slug": "drug-a-upregulates-tp53",           // the file slug (filename = NNNN-slug.md)
      "title": "Drug A upregulates TP53",
      "status": "validated",                       // §3 of conventions/findings.md
      "phase": "exploratory",                      // exploratory | confirmatory
      "entities": [                                 // normalized refs, db+id only (labels live in the file)
        { "db": "uniprot", "id": "P04637" },
        { "db": "hgnc",    "id": "HGNC:11998" }
      ],
      "relationships": [                            // forward, directed edges (source = this finding)
        { "type": "supports", "target": 12 },
        { "type": "refines",  "target": 7 }
      ],
      "updated": "2026-06-22",                      // mirrors the file's `updated`
      "data_version": "sha256:9f86d08…"            // mirrors provenance.data_version; drives staleness
    }
  ]
}
```

### Per-finding fields (exactly the doc 03.7 set, plus `slug` for file resolution)

| Field | From the finding | Purpose |
|---|---|---|
| `id` | `id` | Identity; filename resolution with `slug`. |
| `slug` | filename | Resolve `findings/<id>-<slug>.md` without a directory scan. |
| `title` | `title` | Human label in listings. |
| `status` | `status` | Filter/queries; consistency checks. |
| `phase` | `phase` | Multiplicity-honest filtering (exploratory vs confirmatory). |
| `entities` | `entities` (db+id) | Knowledge-graph queries ("all findings involving the proteasome"). |
| `relationships` | `relationships` | Graph edges; reverse-edge and cascade reasoning. |
| `updated` | `updated` | Recency. |
| `data_version` | `provenance.data_version` | Staleness detection (doc 03.8). |

The manifest deliberately **omits** the heavy fields (`evidence`, `verdict`, `provenance` detail, body) — those are pulled from the file on demand. Keeping it lean is what lets it scale.

## IDs

- `next_id` is the single source for the next id; the findings-manager assigns it, then increments. IDs are **never reused**, even after a finding is `closed`/`invalidated` — this keeps edge targets stable.
- The findings-manager is the only writer of `next_id`; concurrent finding creation is serialized through it (it owns the graph, doc 04.2).

## Reverse edges

Edges are stored **once**, in the source finding's `relationships` (and mirrored here). Reverse lookups ("what points at finding 7?") are **computed** by scanning `findings[].relationships` for `target == 7` — not stored as a separate field. At hundreds of findings this scan is trivial; storing reverse edges would create a second copy to keep consistent. The findings-manager guarantees that every edge target resolves to an existing finding (no dangling edges).

## Queries the manifest is built to answer

- **Novelty / dedup** — does a new observation overlap an existing finding's entities + claim?
- **Relationship detection** — which existing findings does a new one support/refine/contradict/supersede?
- **Knowledge-graph slices** — all `validated` findings involving a given entity; the most-connected entities; mutually `contradicting` findings.
- **Consistency** — dangling edge targets, reverse-edge agreement, status-vs-edge invariants (e.g. a `superseded` finding has an incoming `supersedes`).
- **Staleness** — findings whose `data_version` no longer matches the current dataset stamp, or whose linked script commit moved (doc 03.8).

## Regeneration

The manifest is rebuilt from the finding files at any time: scan each `findings/NNNN-*.md`, read frontmatter, project the fields above, recompute `next_id = max(id)+1`. A rebuild is the recovery path whenever the manifest is suspected stale or corrupt.
