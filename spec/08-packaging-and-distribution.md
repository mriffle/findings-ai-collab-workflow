# 08 — Packaging and Distribution

A first-class goal: a user sets this up with a single Claude Code command, installed straight from the GitHub repository. The Claude Code **plugin** system is the intended mechanism, and the project maps onto it almost one-to-one.

> **Version caveat.** The plugin system evolves quickly; exact manifest fields, hook event names, and minimum versions shift between releases. Treat the shapes below as the design intent and **verify against the current official Claude Code plugin documentation before building the manifests.** Reference scaffolding exists (Anthropic's official plugins repo has an example plugin; community plugin-development templates provide scaffolding commands).

## 8.1 The model

- A **plugin** packages skills, agents, slash commands, hooks, and MCP/LSP servers into one installable unit.
- A **marketplace** is a git repository (or local directory) containing a marketplace manifest. **This project's GitHub repo is its own marketplace** — no separate hosting needed.
- A user installs in two steps: add the marketplace (`/plugin marketplace add owner/repo`), then install the plugin (`/plugin install <name>@<marketplace>`). Marketplaces can be added from `owner/repo` GitHub format or any git URL.

## 8.2 Component → plugin mapping

| Project component | Plugin home |
|---|---|
| Subagents (doc 04): findings manager, verifier, researcher, librarian, reviewers, coder, statistician, figure agents, writers | `agents/` |
| Skills (doc 04): publication/protein/source-code research; stats boilerplate guidance; visualization; finding template; verification-task builder; report skills | `skills/` |
| Workflow entry points (the staged process, doc 02) | `commands/` (slash commands) |
| Deterministic gates (doc 05.5): integrity-gate finding-write check, promotion gate, read-only raw data | `hooks/` |
| Vetted analysis + visualization library | `lib/` (bundled scripts) |
| Conventions and specs (docs 05, 06, 07) | `conventions/` + `CLAUDE.md` |
| Templates: finding, research-finding, report, color registry | `templates/` |

## 8.3 Hooks for deterministic enforcement

The principle "a convention is only real if something checks it" (doc 01) gets its teeth here. Plugin **skills are model-invoked** (loaded when relevant), which is fine for procedures but unreliable for must-always-fire rules. **Hooks fire deterministically on events**, so the highest-stakes gates are implemented as hooks rather than relying on the model to remember:

- **No analysis before the integrity gate passes** (doc 02.3 / 05).
- **A script cannot be promoted until tests, types, and lint pass** (doc 05).
- **Raw data is read-only** (block writes to `data/`).
- **A finding links only to a promoted script** (doc 03 / 05).
- **Record-the-finding** behavior during exploration (paired with `CLAUDE.md`).

These map onto Claude Code's tool-use and session lifecycle events; the exact event names are version-dependent (see caveat). Community plugins already use hooks this way (e.g. test-first gates that block writes lacking a failing test, destructive-command blockers), which confirms the pattern is supported.

## 8.4 Manifest shapes (design intent — verify before building)

The plugin manifest (`.claude-plugin/plugin.json`) declares metadata and the components it ships; optional directories (`commands/`, `agents/`, `skills/`, hooks, `.mcp.json`) are discovered by the standard layout. The marketplace manifest (`.claude-plugin/marketplace.json`) lists this repo's plugin(s) with name, source, and description. Keep the manifests minimal and let the directory layout (README) carry the structure.

## 8.5 The plugin-vs-project-state boundary (architectural)

This boundary is load-bearing and must stay crisp:

- **The plugin ships the engine** — workflow, agents, skills, commands, hooks, conventions, the `lib/` analysis library, and **universal defaults** (e.g. the base Okabe–Ito color registry). Versioned and shared; identical across all users.
- **The user's project holds the data and derived state** — the dataset, `state/PROJECT.md`, `state/METADATA.md`, `state/DATA_DESCRIPTION.md`, the project color registry extension, the findings graph and manifest, research, results, figures, and reports. Generated per dataset; unique to the study.

Plugin skills live in a namespaced cache and coexist with any project-level `.claude/skills/`, so the separation is clean: the engine never carries a study's data, and a study never re-implements the engine.

## 8.6 CLAUDE.md

A project-scoped `CLAUDE.md` (shipped by the plugin, applied in the user's project) encodes the workflow's standing behavior — the staged ordering, the always-on findings-recording instruction, and the references to conventions — so the orchestrator honors them every session. Deterministic must-fire rules are backed by hooks (8.3); `CLAUDE.md` carries the behaviors that are guidance rather than hard gates.

## 8.7 Versioning and security

- **Versioning.** The plugin is versioned; `lib/` versions are recorded in findings (doc 05) so a finding's computation is pinned to the engine version that produced it.
- **Security.** Plugins run inside the session and can read or edit the repo, run shell commands, and call external services. The README must state this plainly and advise reviewing the source before installing — and, for the paper, frame distribution-via-plugin as the adoption/reproducibility story, distinct from the scientific novelty (doc 01.5).
