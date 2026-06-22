# The Findings Workflow

*A collaborative, provenance-tracked paradigm for AI-assisted scientific data analysis, distributed as a Claude Code plugin.*

Today an AI assistant is used as a **transient analyst**: insights surface in conversation and evaporate, exploratory tests pile up untracked, results can't be regenerated, and speculative and solid claims blur together. The Findings Workflow inverts this. As a scientist and Claude Code explore a dataset together, **every substantive insight is captured as a structured, regenerable, independently validated finding** that pins its inputs, records its caveats and lifecycle status, and links into a queryable **findings graph**. The durable output is not a chat transcript — it is a curated body of knowledge.

Proteomics is the proving ground; the design is discipline-agnostic.

> **Status:** early implementation. The full design lives in [`spec/`](spec/README.md) (docs 01–08). This README is the entry point; the spec is the source of truth for intent.

## The two-repository model

A persistent boundary runs through the whole system:

- **This plugin repository ships the *engine*** — agents, skills, slash commands, hooks, conventions, a vetted analysis/visualization library, and universal defaults (e.g. the base color palette). Versioned and shared; identical for every user.
- **Your project holds the *data and derived state*** — the dataset, project-state files, the findings graph, the per-project color registry, results, figures, and reports. Generated per dataset; unique to your study.

The plugin is the engine; your project is the data. The plugin never carries a study's data, and a study never re-implements the engine.

## Installation

```text
# 1. Add this repository as a plugin marketplace
/plugin marketplace add mriffle/findings-ai-collab-workflow

# 2. Install the plugin
/plugin install findings-workflow@findings-workflow
```

Then, in the working directory for your study, run the init command (see `commands/`) to scaffold the project structure and project-scoped instructions.

## How it works (the staged workflow)

```
Stage 0  State the science            → state/PROJECT.md
Stage 1  Understand the metadata      → state/METADATA.md
Stage 2  Understand the data          → state/DATA_DESCRIPTION.md
Stage 3  Loaders + QC  [INTEGRITY GATE]→ verified loaders, QC report
Stage 4  Explore  ⇄  Record findings  → findings/        (the heart)
Stage 5  Independent validation       → validated findings
Stage 6  Reporting                    → reports/
```

The ordering rule is absolute: **nothing is analyzed before it is understood, and nothing is explored before the data read is verified.** See [`spec/02-workflow.md`](spec/02-workflow.md).

## The non-negotiables

1. **Code is the source of truth; conversation is ephemeral.** Every finding is regenerable from pinned data + code + parameters + environment.
2. **Correctness is foundational and upstream of everything.** Data-loading fidelity is tested *and* verified before any analysis — a silent read error is a common-mode failure that defeats downstream validation.
3. **No finding is validated without independent validation.**
4. **Skepticism lives in the gates, calibrated by phase** — generous in exploration, ruthless at promotion.
5. **A convention is only real if something checks it** — preferably a deterministic hook.

## Security notice

This is a Claude Code plugin. **Installed plugins run inside your Claude Code session: they can read and edit files in your repository, run shell commands, and call external services.** Review the source before installing, and treat plugin updates as code you are accepting into your environment. Distribution-via-plugin is the adoption/reproducibility mechanism, distinct from the scientific method the workflow encodes.

## Repository layout (the engine)

```
findings-workflow/
├── .claude-plugin/        # plugin.json + marketplace.json (this repo is its own marketplace)
├── CLAUDE.md              # engine-development guidance (applies when working ON the plugin)
├── commands/              # slash commands: workflow entry points
├── agents/                # context-isolated subagents (spec doc 04)
├── skills/                # research / stats / viz / finding / report procedures (doc 04)
├── hooks/                 # deterministic enforcement gates (docs 05, 08)
├── lib/                   # vetted, tested analysis + visualization template scripts (seed project-local scripts)
├── conventions/           # coding, statistical, correctness, visualization specs (docs 05, 06)
├── templates/             # finding, research-finding, report, color-registry templates
└── spec/                  # the design specification suite (docs 01–08)
```

## License

[Apache-2.0](LICENSE).
