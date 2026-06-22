# commands/

Slash commands that are the **workflow entry points** (spec doc 02). Each is an `.md` file; when the plugin is installed they are invoked as `/findings-workflow:<name>`.

Commands (invoked as `/findings-workflow:<name>`):

| Command | Stage | Purpose |
|---|---|---|
| `init` | — | Scaffold a user project + write the project `CLAUDE.md` + seed registry/manifest/workflow state. |
| `stage0-science` | 0 | State the science → `state/PROJECT.md`. |
| `stage1-metadata` | 1 | Understand + validate the metadata → `state/METADATA.md` (human checkpoint). |
| `stage2-data` | 2 | Understand the data matrix → `state/DATA_DESCRIPTION.md`. |
| `stage3-loaders` | 3 | Loaders + pairing + QC; the **integrity gate**. Unlocks analysis. |
| `stage4-explore` | 4 | Explore ⇄ record findings (refuses until the gate passes). |
| `stage5-validate` | 5 | Independent validation of a finding (blind verifier + concordance). |
| `stage6-report` | 6 | Compile a report as a projection of the findings graph. |
| `status` | — | Show pipeline position, gate state, and findings breakdown. |

Commands orchestrate; the heavy, context-isolated work happens in `agents/` invoking `skills/`. The **orchestrator is the main session** (governed by the project `CLAUDE.md`), not a subagent — these commands structure how the scientist drives it through the stages.
