# The Findings Workflow

[![CI](https://github.com/mriffle/findings-ai-collab-workflow/actions/workflows/ci.yml/badge.svg)](https://github.com/mriffle/findings-ai-collab-workflow/actions/workflows/ci.yml)

*A collaborative, provenance-tracked paradigm for AI-assisted scientific data analysis, distributed as a Claude Code plugin.*

Today an AI assistant is used as a **transient analyst**: insights surface in conversation and evaporate, exploratory tests pile up untracked, results can't be regenerated, and speculative and solid claims blur together. The Findings Workflow inverts this. As a scientist and Claude Code explore a dataset together, **every substantive insight is captured as a structured, regenerable, independently validated finding** that pins its inputs, records its caveats and lifecycle status, and links into a queryable **findings graph**. The durable output is not a chat transcript — it is a curated body of knowledge.

Proteomics is the proving ground; the design is discipline-agnostic.

> **Status:** early implementation. The full design lives in [`spec/`](spec/README.md) (docs 01–08). This README is the entry point; the spec is the source of truth for intent.

## The two-repository model

A persistent boundary runs through the whole system:

- **This plugin repository ships the *engine*** — agents, skills, slash commands, hooks, conventions, vetted analysis/visualization template scripts (seeding project-local analysis code), and universal defaults (e.g. the base color palette). Versioned and shared; identical for every user.
- **Your project holds the *data and derived state*** — the dataset, project-state files, the findings graph, the per-project color registry, results, figures, and reports. Generated per dataset; unique to your study.

The plugin is the engine; your project is the data. The plugin never carries a study's data, and a study never re-implements the engine.

## Installation

```text
# 1. Add this repository as a plugin marketplace
/plugin marketplace add mriffle/findings-ai-collab-workflow

# 2. Install the plugin
/plugin install findings-workflow@findings-workflow

# 3. Apply the changes: run /reload-plugins, or restart Claude Code
```

**Apply the changes after installing.** The install writes the plugin to disk, but a running session picks it up only after `/reload-plugins` — or a full restart. If the slash commands (`/findings-workflow:init`, etc.) don't appear after `/reload-plugins`, quit and relaunch Claude Code; a fresh session reliably registers them. Confirm by typing `/findings-workflow:` and checking that autocomplete lists the commands.

**Scope matters.** A plugin installed at **local** scope is active only in the project directory it was installed for — if you install it while pointed at project A, its commands will *not* appear in project B (they'll read as "Unknown command" no matter how many times you reload). Launch Claude Code from the study directory you installed it for, or install at **user** scope to make it available everywhere.

Then, in the working directory for your study, run the init command (see `commands/`) to scaffold the project structure and project-scoped instructions. Analysis runs on **Python ≥ 3.11**; the `setup-env` command establishes a **project-local** environment (detecting an existing Python, or — with your consent — installing one *into the project* via `uv`, with no changes to your system).

### Updating

To pull a newer version of the plugin:

```text
# 1. Refresh the marketplace catalog so Claude Code sees the latest version
/plugin marketplace update findings-workflow

# 2. Re-install to upgrade to the latest version
/plugin install findings-workflow@findings-workflow

# 3. Apply the changes: run /reload-plugins, or restart Claude Code
```

As with a fresh install, run `/reload-plugins` (or restart) so the updated commands take effect. The `marketplace update` step is required first — without it, the install may not see a newly published version because the local catalog is cached. You can also manage updates interactively via the `/plugin` menu (**Marketplaces** → select → **Update**).

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

**Restarting between stages is safe — and can help.** Progress lives in `state/`, `findings/`, and `state/workflow.json`, not in the conversation, so restarting Claude Code never loses work — a fresh session re-reads that state and resumes where you left off (`status` reorients it). A long session eventually compacts its context (lossy); a fresh one re-reads the durable state losslessly, so a restart at a stage boundary can *improve* downstream results. Watch your context indicator: on **Pro (~200k)** consider restarting after **Stage 1** and **Stage 3**; on **Max** plans (up to ~1M) you likely won't need to. (In Stage 4, record substantive insights as findings before restarting — findings survive, in-flight discussion doesn't.)

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
