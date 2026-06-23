# hooks/

Deterministic enforcement gates (spec docs 05.5, 08.3). Hook configuration lives in `hooks/hooks.json`; the guard scripts live alongside and are referenced with `${CLAUDE_PLUGIN_ROOT}`.

"A convention is only real if something checks it." Hooks fire deterministically on events, so the highest-stakes gates are hooks rather than model memory:

- **The integrity gate's finding-write side** — no `integrity_signoff` / `validated` before the gate passes (doc 02.3 / 05). The *no-analysis-before-the-gate* ordering itself is a command precondition + orchestrator behavior, not a hook.
- **A script cannot be promoted until tests, types, and lint pass** (doc 05).
- **Raw data is read-only** — block writes to `data/`.
- **A finding links only to a promoted script** (doc 03 / 05).

Hooks block via exit code 2 (reason on stderr); any other non-zero exit (and a failure to even spawn the interpreter) is a non-blocking error, so the gates **fail open** rather than wedge a session.

## The guards are Python (cross-platform)

The guards are **Python**, not bash. The earlier bash versions were invoked as `bash "${CLAUDE_PLUGIN_ROOT}"/hooks/<script>.sh` and depended on `bash` **and** `jq` being on `PATH`. Native Windows (now a first-class Claude Code platform) has neither, so on Windows a `command`-type hook is handed to PowerShell, `bash` isn't found, the hook exits non-zero-non-2, and **every guard silently failed open** — exactly the platform where the enforcement layer should still hold. Python is already a hard dependency of the workflow (`setup-env` bootstraps Python ≥ 3.11), and stdlib `json` removes the `jq` dependency entirely, so the guards now run unchanged on Windows, macOS, and Linux.

The guards are **stdlib-only to run** (so any Python ≥ 3.11 executes them); only `guard_promotion.py` spawns external tools (`ruff`/`mypy`), which it resolves from the project `./.venv` first, then `PATH`.

### How `hooks.json` invokes them

Each guard is wired in **exec form** (`command` + `args`, no shell — no quoting/`PATH`-name fragility), with the **project's own interpreter** as the command and the guard script as the argument. Because a single exec command can't branch on platform, each guard is listed **twice** — once per OS venv layout:

```json
{ "type": "command", "command": "${CLAUDE_PROJECT_DIR}/.venv/bin/python",        "args": ["${CLAUDE_PLUGIN_ROOT}/hooks/guard_findings.py"] }
{ "type": "command", "command": "${CLAUDE_PROJECT_DIR}/.venv/Scripts/python.exe", "args": ["${CLAUDE_PLUGIN_ROOT}/hooks/guard_findings.py"] }
```

The two interpreter paths are **mutually exclusive per platform** (`bin/python` exists only on Unix, `Scripts/python.exe` only on Windows). All hooks under a matcher run in parallel, so exactly **one** spawns and runs the guard; the other fails to spawn and fails open (non-blocking) — no double-execution. Using `${CLAUDE_PROJECT_DIR}/.venv/...` (the interpreter `setup-env` guarantees, ≥ 3.11, at a known path) removes any dependence on a *system* Python's name (`python` vs `python3` vs the Windows `py` launcher) — the cross-platform fragility we're eliminating.

### Files

| File | Role |
|---|---|
| `guard_readonly_data.py` | PreToolUse `Write\|Edit`, `Bash` — raw `data/` is read-only (Write/Edit blocked under `data/`; best-effort Bash block). |
| `guard_findings.py` | PreToolUse `Write\|Edit` — no `integrity_signoff: true` / `status: validated` on a finding before the integrity gate passes; a `validated` finding may link only to `scripts/promoted/`. |
| `guard_promotion.py` | PreToolUse `Write` (blocks), PostToolUse `Write\|Edit` (warns) — a `scripts/promoted/*.py` must pass `ruff` + `mypy` (where available/configured), resolved from `./.venv` first then `PATH`. Tests are verified by the code-reviewer, not in-hook. |
| `_hooklib.py` | Shared helpers (event parse, project scope, fail-open exits). Imported by the guards — not itself a hook. |
| `test_guards.py` | Stdlib-only synthetic-event tests: `python3 hooks/test_guards.py`. |

**Design invariants** (see `conventions/enforcement-map.md`): all guards **scope to initialized projects** (`state/workflow.json` present) so they never touch unrelated repos, and **fail open** — on a malformed event, a missing interpreter, or (for promotion) missing `ruff`/`mypy` — so they never wedge a session over their own tooling. Block (exit 2) only on a genuine, clearly-decidable violation. The "is this exploratory analysis?" judgment is deliberately left to command preconditions + orchestrator behavior — it can't be cleanly decided from one tool-use event.

**Honest note on engagement window.** Because the guards run via the project venv interpreter, they engage once the project env exists (after `setup-env`, the Stage-3 precondition); before that the interpreter path doesn't resolve and the hooks fail open. The gated artifacts — findings and promoted scripts — belong to Stage 3+ anyway; the only reduction in coverage is the read-only-`data/` guard during the pre-env stages, recovered the moment `setup-env` runs. This mirrors the prior design, where the promotion guard already depended on `./.venv` tooling.

**Testing.** Run `python3 hooks/test_guards.py` after any guard change (stdlib-only, no deps; feeds each guard synthetic PreToolUse/PostToolUse events and asserts block/allow). The guard scripts are also kept `ruff`-clean and `mypy --strict`-clean.
