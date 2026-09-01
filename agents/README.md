# agents/

Context-isolated subagents (spec doc 04). Each is an `.md` file: YAML frontmatter (`name`, `description`, optional `tools`/`model`) + a markdown body that becomes the agent's system prompt.

Context isolation is the point: research, code, statistics, and figure work happen in subagents so the orchestrator's context stays focused on the science and the dialogue.

**Generator/reviewer pairing** is used wherever an artifact's correctness matters; the **verifier** is deliberately starved of conversation history (doc 03.5). The **orchestrator** is the main session (governed by the project `CLAUDE.md`), not a file here.

Roster (doc 04.2):

| Agent | Built | Model | Role |
|---|---|---|---|
| `findings-manager` | ✅ | default | Owns the findings graph + manifest. |
| `verifier` | ✅ | default | Blind, history-starved independent validation. |
| `coder` | ✅ | **sonnet** | Write Python scripts/loaders (→ scratch). |
| `code-reviewer` | ✅ | default | Run tests/types/lint + data-handling review; gates promotion. |
| `statistician` | ✅ | default | Perform analysis via `lib/`, obey stats conventions. |
| `stats-reviewer` | ✅ | default | Check analysis against statistical conventions. |
| `figure-generator` | ✅ | default | Render publication figures (dual export + legend). |
| `figure-reviewer` | ✅ | default | Review the rendered PNG for accuracy/standards. |
| `researcher` | ✅ | **sonnet** | Research one bounded topic → research-finding. |
| `librarian` | ✅ | default | Control the research corpus; scope/dispatch research. |
| `research-reviewer` | ✅ | default | Fact-check research; verify every reference. |
| `writer` | ✅ | default | Draft a report section by projecting selected findings. |
| `report-reviewer` | ✅ | default | Claim-source check + status/caveat propagation + coherence pass. |

**Model column.** `default` = no `model` in the frontmatter, so Claude Code resolves it through its own order (per-invocation parameter → the `CLAUDE_CODE_SUBAGENT_MODEL` env var → the main conversation's model). A **pinned** agent names an alias (`model: sonnet`). The pattern: pin the **generator** of a generator/reviewer pair, leave the **reviewer** on the default — the reviewer is the gate, and the gates are where the engine spends its capability budget.
