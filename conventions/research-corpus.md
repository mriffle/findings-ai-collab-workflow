# Convention — The Research Corpus Manifest

*The index of the research corpus. Source of truth: the research-finding files (`research/<slug>.md`). Derived index: `research/manifest.md`, managed by the **librarian**. Spec basis: doc 04.4–04.5.*

## What it is

`research/manifest.md` is the librarian's at-a-glance index of the research corpus — which research-findings exist, what they cover, and whether they are accepted. It lets the librarian judge coverage and avoid dispatching redundant research without opening every file.

> **The research-finding files are the source of truth; this manifest is a *derived* index** (same model as the findings manifest), regenerable by scanning `research/<slug>.md` frontmatter. **Only `reviewed` entries count as corpus** — a `draft` is not yet trustworthy knowledge (it hasn't cleared the research-reviewer's reference check).

Markdown, not JSON: it is written/read by LLM agents (the librarian), so it follows the project's format convention (Markdown for LLM/human-maintained indexes; JSON only where a non-LLM parser consumes — see the engine `CLAUDE.md`).

## Format

```markdown
---
schema_version: 1
generated: 2026-06-22
---

# Research manifest

Index of the research corpus, managed by the librarian. Only `reviewed` entries count as corpus. Source of truth: the `research/<slug>.md` files. One row per research-finding.

| Slug | Topic | Type | Status | Entities | Refs | Updated |
|------|-------|------|--------|----------|------|---------|
| tp53-function | TP53 function & regulation | protein | reviewed | uniprot:P04637, hgnc:HGNC:11998 | 3 | 2026-06-22 |
| limma-ebayes | What limma's eBayes actually computes | software | reviewed | — | 2 | 2026-06-22 |
```

### Columns (projected from each research-finding's frontmatter)

| Column | From | Purpose |
|---|---|---|
| `Slug` | filename | Resolve `research/<slug>.md`. |
| `Topic` | `topic` | Human label. |
| `Type` | `type` | `protein`/`gene`/`disease`/`pathway`/`publication`/`software`/`general`. |
| `Status` | `status` | `draft` or `reviewed`. **Only `reviewed` is corpus.** |
| `Entities` | `entities` | Coverage by canonical entity (compact `db:id`, comma-separated). |
| `Refs` | `references` | Count of (verified) references. |
| `Updated` | `updated` | Recency. |

Use `—` for empty cells.

## Lifecycle

1. The **researcher** writes a research-finding (`research/<slug>.md`, `status: draft`).
2. The **research-reviewer** verifies factual accuracy and **every reference** and returns ACCEPT/REVISE. It holds no write access — generator/reviewer separation, so it never edits the artifact it reviews.
3. On the reviewer's ACCEPT, the **librarian** — the sole writer of corpus state — applies the verdict to the research-finding file (`status → reviewed`, accepted references marked `verified`, and stamps `reviewed_by`/`reviewed_date` as review provenance — file-level only, not projected into the manifest) and registers/updates its row in `research/manifest.md`, marking it `reviewed`. The librarian regenerates the manifest from the files whenever it is suspected stale.

The librarian answers questions only from `reviewed` corpus, always carrying the references (the references invariant, doc 04.5).
