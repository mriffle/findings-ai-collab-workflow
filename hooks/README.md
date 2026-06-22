# hooks/

Deterministic enforcement gates (spec docs 05.5, 08.3). Hook configuration lives in `hooks/hooks.json`; any scripts it runs live alongside and are referenced with `${CLAUDE_PLUGIN_ROOT}`.

"A convention is only real if something checks it." Hooks fire deterministically on events, so the highest-stakes gates are hooks rather than model memory:

- **The integrity gate's finding-write side** — no `integrity_signoff` / `validated` before the gate passes (doc 02.3 / 05). The *no-analysis-before-the-gate* ordering itself is a command precondition + orchestrator behavior, not a hook.
- **A script cannot be promoted until tests, types, and lint pass** (doc 05).
- **Raw data is read-only** — block writes to `data/`.
- **A finding links only to a promoted script** (doc 03 / 05).

Hooks block via exit code 2 (reason on stderr) or a JSON `permissionDecision: "deny"` on stdout for `PreToolUse`.

## Built hooks

`hooks.json` wires three guard scripts (invoked as `bash ${CLAUDE_PLUGIN_ROOT}/hooks/<script>`):

| Script | Events | Enforces |
|---|---|---|
| `guard_readonly_data.sh` | PreToolUse `Write\|Edit`, `Bash` | Raw `data/` is read-only (Write/Edit blocked under `data/`; best-effort Bash block). |
| `guard_findings.sh` | PreToolUse `Write\|Edit` | No `integrity_signoff: true` / `status: validated` on a finding before the integrity gate passes; a `validated` finding may link only to `scripts/promoted/`. |
| `guard_promotion.sh` | PreToolUse `Write` (blocks), PostToolUse `Write\|Edit` (warns) | A `scripts/promoted/*.py` must pass `ruff` + `mypy` (where available/configured). Resolves both from the **project `./.venv` first**, then `PATH` (the zero-footprint `setup-env` installs them into the venv, not globally). Tests are verified by the code-reviewer, not in-hook. |

**Design invariants** (see `conventions/enforcement-map.md`): all guards **scope to initialized projects** (`state/workflow.json` present) so they never touch unrelated repos, and **fail open** if `jq`/`ruff`/`mypy` are missing so they never wedge a session over their own tooling. The "is this exploratory analysis?" judgment is deliberately left to command preconditions + orchestrator behavior — it can't be cleanly decided from one tool-use event. All three are tested against synthetic events.
