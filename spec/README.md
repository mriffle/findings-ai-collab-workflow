# The Findings Workflow — Specification Suite

*A collaborative, provenance-tracked paradigm for AI-assisted scientific data analysis, distributed as a Claude Code plugin.*

**Status:** detailed pre-implementation specification. This suite supersedes and expands the earlier design draft and correctness charter; it is intended to be complete enough that any engineer or agent can implement the system from it.

---

## What this is, in one paragraph

The Findings Workflow is a formalized way for a scientist and an AI agent (Claude Code) to analyze scientific data together, in which the durable output is not a chat transcript but a curated, interconnected, independently validated **findings graph**. The workflow is shipped as a Claude Code plugin: a repository of agent definitions, skills, slash commands, hooks, conventions, and a vetted analysis-script library that a user installs with a single command and runs against their own dataset. Proteomics is the proving ground; the design is discipline-agnostic.

## The core idea (read this first)

Today an AI assistant is used as a *transient analyst*: insights surface in conversation and evaporate, exploratory tests pile up untracked, results can't be regenerated, and speculative and solid claims blur together in the write-up. This project inverts that. As the scientist and the agent explore, **every substantive insight is automatically captured as a structured, regenerable, reviewable finding** that pins its inputs, records its caveats and lifecycle status, links into a graph of related findings, and is promoted to *validated* only after independent re-derivation. The unit of value shifts from a disposable answer to a governed body of knowledge. That automated, in-the-moment, rigor-bound capture of findings is the heart of the project.

## How to read this suite

The documents are ordered from "why" to "how." An implementer can read them in order; a contributor working on one concern can jump straight to the relevant file.

| # | Document | Concern |
|---|---|---|
| — | [`README.md`](README.md) | This index, the elevator pitch, the repo map |
| 01 | [`01-overview-and-principles.md`](01-overview-and-principles.md) | What we're doing and why; design principles; novelty; glossary |
| 02 | [`02-workflow.md`](02-workflow.md) | The staged process; project state files; gates; human checkpoints |
| 03 | [`03-findings-and-knowledge-graph.md`](03-findings-and-knowledge-graph.md) | The finding object, status machine, the findings/knowledge graph, manifest, validation — the centerpiece |
| 04 | [`04-agents-skills-and-research.md`](04-agents-skills-and-research.md) | Agent roster and contracts; skills; the research subsystem; the references invariant |
| 05 | [`05-conventions-and-correctness.md`](05-conventions-and-correctness.md) | Repository, coding, and statistical conventions; the correctness & data-integrity charter; enforcement map |
| 06 | [`06-visualizations.md`](06-visualizations.md) | Figure accuracy, dual export, the color registry, the >8-category strategy |
| 07 | [`07-reporting.md`](07-reporting.md) | Reports as projections of the findings graph; QC vs research reports |
| 08 | [`08-packaging-and-distribution.md`](08-packaging-and-distribution.md) | Plugin packaging, one-command install, hooks, plugin-vs-project-state boundary |

## The two-repository mental model

A persistent architectural boundary runs through the whole spec (see doc 08):

- **The plugin repository** ships the *engine*: agents, skills, commands, hooks, conventions, the analysis-script library, and universal defaults (e.g. the base color palette). It is versioned and shared.
- **The user's project** holds the *data and derived state*: the dataset, `METADATA.md`, `DATA_DESCRIPTION.md`, the findings graph, the per-project color registry, results, and figures. It is generated per dataset.

The plugin is the engine; the project is the data. Keep this line crisp when implementing.

## Indicative plugin + project layout

```
# ── PLUGIN REPOSITORY (shipped, versioned) ───────────────
findings-workflow/
├── .claude-plugin/
│   ├── plugin.json            # plugin manifest
│   └── marketplace.json       # marketplace manifest (this repo is its own marketplace)
├── CLAUDE.md                  # workflow enforcement at project scope
├── commands/                  # slash commands: workflow entry points
├── agents/                    # subagent definitions (doc 04)
├── skills/                    # research / stats / viz / finding / report skills (doc 04)
├── hooks/                     # deterministic enforcement gates (doc 05, 08)
├── lib/                       # vetted boilerplate stats + visualization library
├── conventions/               # coding, statistical, correctness, visualization specs
└── templates/                 # finding, research-finding, report, color-registry templates

# ── USER PROJECT (generated per dataset) ─────────────────
my-study/
├── data/                      # raw data (read-only)
├── state/
│   ├── PROJECT.md             # research goals / framing (stage 0)
│   ├── METADATA.md            # verified metadata description (stage 1)
│   ├── DATA_DESCRIPTION.md    # verified data description (stage 2)
│   └── color_registry.json    # category → color map (doc 06)
├── scripts/
│   ├── scratch/               # exploratory, disposable
│   └── promoted/              # reviewed, reproducible; findings link here
├── results/                   # regenerable CSVs
├── figures/                   # regenerable SVG + 300 DPI PNG + legend docs
├── findings/
│   ├── manifest.md            # findings graph index (Markdown — see conventions/manifest.md)
│   └── NNNN-slug.md           # individual finding documents
├── research/                  # research-finding documents (with references)
└── reports/                   # QC reports and research reports
```

## The non-negotiables

If everything else is forgotten, these survive:

1. **Code is the source of truth; conversation is ephemeral.** Every finding is regenerable from pinned data + code + parameters + environment.
2. **Correctness is foundational and upstream of everything.** Data-loading fidelity is tested *and* verified before any analysis, because a silent read error is a common-mode failure that defeats downstream validation.
3. **No finding is validated without independent validation.**
4. **Skepticism lives in the gates, calibrated by phase** — generous in exploration, ruthless at promotion.
5. **A convention is only real if something checks it** — preferably a deterministic hook.
