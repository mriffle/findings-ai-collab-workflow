# The Findings Workflow

[![CI](https://github.com/mriffle/findings-ai-collab-workflow/actions/workflows/ci.yml/badge.svg)](https://github.com/mriffle/findings-ai-collab-workflow/actions/workflows/ci.yml)

*A collaborative, provenance-tracked way to do AI-assisted scientific data analysis — packaged as a Claude Code plugin.*

## What this is

Normally an AI assistant is a **transient analyst**: insights surface in conversation and evaporate, exploratory tests pile up untracked, results can't be regenerated, and speculative and solid claims blur together. The Findings Workflow inverts that. As you and Claude Code explore a dataset together, **every substantive insight is captured as a structured, regenerable, independently validated finding** — one that pins its exact inputs, records its caveats and lifecycle status, and links into a queryable findings graph. What you keep at the end is not a chat transcript; it's a curated, reproducible body of knowledge.

Proteomics is the proving ground, but the design is discipline-agnostic. You bring a dataset; the plugin walks you and Claude through understanding it, verifying it loads correctly, exploring it, and recording validated findings.

**New here? The whole point is that you don't need to memorize any of this.** Install the plugin, run one command, and Claude leads the way. If you ever get stuck or aren't sure what to do next, just ask Claude — "what's next?" — and it will know.

---

## Installation

The plugin installs from inside Claude Code using its built-in `/plugin` commands. Type these at the Claude Code prompt (they are slash commands, not shell commands).

```text
# 1. Add this repository as a plugin marketplace
/plugin marketplace add mriffle/findings-ai-collab-workflow

# 2. Install the plugin
/plugin install findings-workflow@findings-workflow

# 3. Restart Claude Code so the plugin loads
```

### ⚠️ Restart Claude Code after installing

**This step is required.** Installing writes the plugin to disk, but a running session does not pick it up until it restarts. Fully **quit and relaunch Claude Code** after installing — a fresh session reliably registers the plugin's commands. (You can also try `/reload-plugins` to apply it without a restart, but if the commands don't show up, a full restart always fixes it.)

**Confirm it worked:** at the prompt, type `/findings-workflow:` and pause. Autocomplete should list the workflow commands (`init`, `setup-env`, `stage0-science`, and so on). If you see them, you're ready to go.

### Choosing an installation scope: user-wide vs. project-specific

When you install, Claude Code asks *where* to install the plugin. This matters, so here's the tradeoff:

| Scope | Where it's active | Choose this if… |
|---|---|---|
| **User** (global) | Every project you open in Claude Code | You want the workflow available everywhere, or you'll analyze several different studies. Simplest for most people. |
| **Local** (project) | Only the one project directory you installed it from | You want the plugin tied to a single study, or pinned to a specific version per project. |

The important gotcha: **a plugin installed at *local* scope only works in the exact project directory you installed it for.** If you install it while pointed at study A and then open study B, its commands won't appear in B — they'll read as "Unknown command" no matter how many times you reload. If that happens, either launch Claude Code from the directory you installed it in, or reinstall at **user** scope to make it available everywhere.

If you're unsure, **choose user scope** — it's the least surprising.

---

## Getting started

Once the plugin is installed and Claude Code has restarted:

1. **Open Claude Code in the folder where your study will live** (create an empty folder for it if needed). This becomes your *project* — it holds your data, findings, results, and reports. (The plugin is the reusable engine; your project folder is your study's data and results. The plugin never stores your data, and your project never re-implements the engine.)

2. **Run the init command** to scaffold the project:

   ```text
   /findings-workflow:init
   ```

   This creates the standard folder layout, seeds the project's configuration, and writes a project `CLAUDE.md` that teaches Claude how to run the workflow in this folder. It's safe to run — it never overwrites files you already have.

3. **Follow Claude's lead.** After `init`, Claude tells you the very first step (describing your study). From there you move through the stages below. At any point you can type:

   ```text
   /findings-workflow:status
   ```

   to see where you are, and — again — **if you're ever unsure what to do next, just ask Claude.** It reads the project's state and knows the next move.

> **A note on Python.** Analysis runs on **Python ≥ 3.11**. The `/findings-workflow:setup-env` command sets up a **project-local** Python environment for you — it detects an existing Python, and only if needed asks your permission before installing one *into your project folder* (nothing is changed on your system). Claude will offer this at the right time; you don't need to do it manually up front.

---

## The workflow, stage by stage

The workflow is a sequence of stages. Each stage is a slash command you run (`/findings-workflow:<name>`). You generally go in order — the guiding rule is absolute: **nothing is analyzed before it is understood, and nothing is explored before the data read is verified.** Claude enforces this for you.

You don't have to remember the command names. Run `init`, then let Claude prompt you into each stage. This table is here for reference.

| Run this | Stage | What it does |
|---|---|---|
| `/findings-workflow:init` | setup | Scaffolds the project folder and writes the project instructions. **Start here.** |
| `/findings-workflow:setup-env` | setup | Sets up a project-local Python ≥ 3.11 environment (Claude offers this when it's needed). |
| `/findings-workflow:stage0-science` | 0 | **State the science.** Claude interviews you about the study — the domain, the design, what you're examining and why — and captures it. This framing shapes everything downstream. |
| `/findings-workflow:stage1-metadata` | 1 | **Understand the metadata.** Works out what every sample-metadata column means, tests the relationships that should hold, and flags confounds. Includes a checkpoint where you confirm it got things right. |
| `/findings-workflow:stage2-data` | 2 | **Understand the data.** Determines the data matrix's orientation, its transformation/normalization state, and how missing values behave; characterizes missingness, contaminants, and duplicates. |
| `/findings-workflow:stage3-loaders` | 3 | **Loaders + QC — the integrity gate.** Builds and *verifies* tested loaders that pair every sample correctly and match the source data, then runs quality control. Passing this gate is what unlocks analysis. |
| `/findings-workflow:stage4-explore` | 4 | **Explore and record findings — the heart of the workflow.** You and Claude explore the data together, and every substantive insight is captured as a finding the moment it emerges. (Refuses to run until the integrity gate has passed.) |
| `/findings-workflow:stage5-validate` | 5 | **Independent validation.** Promotes a candidate finding toward *validated* by re-deriving it blindly and checking the results agree. Run it on a finding, e.g. `/findings-workflow:stage5-validate 42`. |
| `/findings-workflow:stage6-report` | 6 | **Reporting.** Compiles your findings into a report — a projection of the findings graph, not a fresh write-up. Two modes: `qc` or `research`. |
| `/findings-workflow:status` | anytime | Shows where the project stands: pipeline position, integrity-gate state, and a breakdown of findings by status and phase. |

### Restarting between stages is safe — and can even help

Your progress lives in files on disk (in `state/` and `findings/`), **not** in the conversation. So restarting Claude Code never loses work — a fresh session re-reads that state and picks up where you left off (run `status` to reorient it). In fact, because a long conversation eventually compresses its own context (losing some detail), starting a fresh session at a stage boundary re-reads the durable state cleanly and can *improve* later results.

A rule of thumb by plan: on **Pro (~200k context)**, consider restarting after **Stage 1** and again after **Stage 3**; on **Max** plans you likely won't need to. One caution: while exploring in Stage 4, record insights as findings before restarting — findings survive, but in-flight discussion doesn't.

---

## Getting help & giving feedback

**Feedback, suggestions, and issue reports are very welcome and genuinely appreciated** — they're how this gets better.

- **Questions or feedback by email:** [mriffle@uw.edu](mailto:mriffle@uw.edu)
- **Bug reports & feature requests:** open a GitHub issue at [github.com/mriffle/findings-ai-collab-workflow/issues](https://github.com/mriffle/findings-ai-collab-workflow/issues)
- **Stuck mid-workflow?** Just ask Claude what to do next. It reads your project's state and knows the next step — you don't have to memorize the stages or command names.

---

## Updating the plugin

> **Re-installing does *not* upgrade it.** If you run `/plugin install findings-workflow@findings-workflow` again, Claude Code just answers *"already installed — use `/plugin` to manage existing plugins"* and pulls nothing. There is also no `/plugin update` slash command. Use one of the two routes below instead.

### Option 1 — Update from inside Claude Code

1. Run `/plugin`
2. Go to the **Installed** tab
3. Select **findings-workflow** and press **Enter**
4. Choose **Update now**

### Option 2 — Update from your shell

This one is a **shell command**, typed in a terminal — not a slash command:

```bash
claude plugin update findings-workflow@findings-workflow
```

It refreshes this repository's marketplace catalog for you first, so no separate `/plugin marketplace update` step is needed. If you're already current it tells you so: `findings-workflow is already at the latest version (0.2.0)`.

**If you installed at *local* or *project* scope**, note that this command targets your **user**-scope install by default. Pass the scope explicitly and run it from the project directory you installed into:

```bash
claude plugin update findings-workflow@findings-workflow --scope local
```

(`claude plugin list` shows every install and its scope — handy if you're not sure which one you have.)

### ⚠️ Update between studies, not in the middle of one

Updating swaps out the engine your project is running on, so do it at a natural boundary — before you start a study, or between stages — rather than partway through an analysis. Claude Code leaves this to you: auto-update is **off by default** for marketplaces like this one, and we recommend leaving it off for exactly this reason.

After updating, **quit and relaunch Claude Code** — the running session keeps using the version it loaded at launch. (`/reload-plugins` often applies it without a restart, but if anything looks stale, a full restart always fixes it.)

---

## Good to know

### The two-repository model

A persistent boundary runs through the whole system:

- **This plugin repository is the *engine*** — the commands, agents, conventions, and vetted analysis/visualization templates. It's versioned, shared, and identical for every user.
- **Your project folder holds the *data and results*** — your dataset, project state, the findings graph, results, figures, and reports. It's unique to your study.

The plugin is the engine; your project is the data. The plugin never carries a study's data, and a study never re-implements the engine.

### The principles behind it

1. **Code is the source of truth; conversation is ephemeral.** Every finding is regenerable from pinned data + code + parameters + environment.
2. **Correctness is foundational and upstream of everything.** Data-loading fidelity is tested *and* verified before any analysis — a silent read error is a common-mode failure that defeats downstream validation.
3. **No finding is validated without independent validation.**
4. **Skepticism lives in the gates, calibrated by phase** — generous in exploration, ruthless at promotion.
5. **A convention is only real if something checks it** — preferably an automated check.

### Security notice

This is a Claude Code plugin. **Installed plugins run inside your Claude Code session: they can read and edit files in your project, run shell commands, and call external services.** Review the source before installing, and treat plugin updates as code you are accepting into your environment.

---

## For contributors and the curious

The full design specification lives in [`spec/`](spec/README.md) (docs 01–08) — it's the source of truth for intent. The repository is organized as its own Claude Code plugin *and* marketplace:

```
findings-workflow/
├── .claude-plugin/        # plugin.json + marketplace.json (this repo is its own marketplace)
├── CLAUDE.md              # engine-development guidance (applies when working ON the plugin)
├── commands/              # slash commands: the workflow entry points you run
├── agents/                # context-isolated subagents the workflow dispatches
├── skills/                # research / stats / viz / finding / report procedures
├── hooks/                 # automated enforcement gates
├── lib/                   # vetted, tested analysis + visualization template scripts
├── conventions/           # coding, statistical, correctness, visualization specs
├── templates/             # finding, research-finding, report, color-registry templates
└── spec/                  # the design specification suite (docs 01–08)
```

If you're working *on* the plugin itself, start with [`CLAUDE.md`](CLAUDE.md) and [`spec/`](spec/README.md).

## License

[Apache-2.0](LICENSE).
</content>
</invoke>
