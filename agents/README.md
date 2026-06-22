# agents/

Context-isolated subagents (spec doc 04). Each is an `.md` file: YAML frontmatter (`name`, `description`, optional `tools`/`model`) + a markdown body that becomes the agent's system prompt.

Context isolation is the point: research, code, statistics, and figure work happen in subagents so the orchestrator's context stays focused on the science and the dialogue.

**Generator/reviewer pairing** is used wherever an artifact's correctness matters; the **verifier** is deliberately starved of conversation history (doc 03.5). The **orchestrator** is the main session (governed by the project `CLAUDE.md`), not a file here.

Roster (doc 04.2):

| Agent | Built | Role |
|---|---|---|
| `findings-manager` | ✅ | Owns the findings graph + manifest. |
| `verifier` | ✅ | Blind, history-starved independent validation. |
| `coder` | ✅ | Write Python scripts/loaders (→ scratch). |
| `code-reviewer` | ✅ | Run tests/types/lint + data-handling review; gates promotion. |
| `statistician` | ✅ | Perform analysis via `lib/`, obey stats conventions. |
| `stats-reviewer` | ✅ | Check analysis against statistical conventions. |
| `figure-generator` | ✅ | Render publication figures (dual export + legend). |
| `figure-reviewer` | ✅ | Review the rendered PNG for accuracy/standards. |
| `researcher` | ✅ | Research one bounded topic → research-finding. |
| `librarian` | ✅ | Control the research corpus; scope/dispatch research. |
| `research-reviewer` | ✅ | Fact-check research; verify every reference. |
| `writer` | ✅ | Draft a report section by projecting selected findings. |
| `report-reviewer` | ✅ | Claim-source check + status/caveat propagation + coherence pass. |
