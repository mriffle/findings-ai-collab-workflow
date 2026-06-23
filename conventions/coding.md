# Convention — Coding

*Spec source: doc 05.2 + the "analysis code held to the maximum" rule in doc 05.4. These are the rules the code-reviewer and the promotion hook enforce. Authoritative tool choices are named here so enforcement is concrete.*

## Language & environment

- **Python only.** All analysis, loaders, and figure code are Python.
- **Python ≥ 3.11.** This is the floor the workflow targets and the Stage 3 gate enforces; older interpreters are "not suitable." Pin the exact version in `.python-version`.
- **Project-local environment, zero global footprint.** The interpreter and virtualenv live **inside the project** (`./.venv`; a project-local `uv` under `./.uv` when bootstrapped), never in a shared/user location, and setup must not modify the user's `PATH` or shell profile. The `setup-env` command establishes this: it detects an existing suitable Python and, only if none is found, **transparently asks consent** before downloading `uv` + a standalone Python *into the project* (flags: `UV_UNMANAGED_INSTALL`, `UV_PYTHON_INSTALL_DIR`). If the scientist declines, they provide Python ≥ 3.11 themselves; Stage 3 stays blocked until a usable interpreter exists. `.uv/` and `.venv/` are git-ignored (regenerable); `pyproject.toml`, `uv.lock`, and `.python-version` are committed.
- **Tooling is invoked from `./.venv`.** The quality tools the gates depend on — `ruff`, `mypy` (promotion hook) and `pytest`, `hypothesis` (code-reviewer) — are installed into the project env, not globally. Hooks and reviewers resolve them from `./.venv/bin` (`.venv\Scripts` on Windows) before `PATH`. If they aren't in the env the promotion gate degrades to a no-op, so `setup-env` installs them as dev dependencies.
- **Locked environments.** Pin exact versions in a lockfile committed to the project. Recommended: **`uv`** (`uv.lock`); `pip-tools` (`requirements.txt` with hashes) or a conda lock are acceptable. The environment is **recorded alongside every finding** (`provenance.environment`) so computational reproduction is meaningful, and software/tool citations (with versions) are drawn from it for free.
- **Seeds set and recorded** everywhere stochastic — `numpy`, `random`, `scikit-learn` (`random_state=`), any framework RNG. The seed is recorded in the finding (`provenance.seed`). A result that can't be reproduced because the seed wasn't pinned is a defect.

## Script discipline

- **Non-interactive and parameterized.** Scripts run end-to-end from configuration (argparse or a config file), never from hard-coded paths or interactive state. A reader can re-run a script with its recorded params and get the recorded numbers.
- **Notebooks are disposable.** Exploration may use notebooks, but the **canonical path is always a script** — out-of-order cell execution is a reproducibility landmine. A finding never links to a notebook.
- **Logging over `print`.** Use the `logging` module. Scripts log their inputs, params, seed, and key shapes.
- **Fail loud.** Data handling asserts its expectations: explicit dtype checks, shape assertions, and **no silent NA coercion**. A script that silently drops rows or coerces a type is worse than one that crashes. Validate-and-raise at every boundary.

## Held to the maximum before promotion (doc 05.4)

A script may move from `scripts/scratch/` to `scripts/promoted/` — and a finding may link to it — **only** when all of these pass:

- **Testing (`pytest`):** unit tests with hand-verified fixtures; **property/invariant tests** (`hypothesis` where useful — e.g. CV folds partition without overlap; a normalization preserves shape; a transform is invertible where claimed); a **planted-truth** check where applicable (synthetic data with a known effect the code must recover); **edge cases** (empty, all-missing, single sample, duplicate IDs, ties).
- **Typing — strict (`mypy --strict`):** **type hints throughout**, and **`mypy` in strict mode** (`[tool.mypy] strict = true`, equivalent to `mypy --strict`) **clean** on the file — no untyped or partially-typed defs, no implicit `Any`, no `Any`-returns. A `# type: ignore` is permitted **only** scoped to a specific error code with a one-line justification (`# type: ignore[no-any-return]  — …`); a blanket `# type: ignore` fails review. The *only* sanctioned relaxation is `ignore_missing_imports` for third-party scientific libraries that ship no type stubs.
- **Linting & formatting — strict (`ruff`):** **`ruff check` clean against an explicitly-configured strict rule set** — **not** ruff's minimal defaults; at minimum `E, F, W, I, B, UP, SIM, C4, PD, NPY, RUF` — and **`ruff format`** applied. A `# noqa` is permitted **only** rule-scoped and justified (`# noqa: E501 — …`); a bare `# noqa` fails review.

Code that has not passed tests, **strict** type-checking, and **strict** lint is `scratch`, and a finding may not link to it.

## Enforcement

| Rule | Enforced by |
|---|---|
| Usable Python ≥ 3.11 project env before any analysis | **Command precondition** (`stage3-loaders` live-verifies) + `setup-env` |
| Raw data read-only; outputs regenerable | **Hook** (`guard_readonly_data.sh`) |
| Script not promoted until **strict** lint + types pass | **Hook** (`guard_promotion.sh`: `ruff check` strict rule set + `mypy` strict, via the project config) |
| Script not promoted until **tests** pass | **Code-reviewer** agent (tests aren't run in-hook) |
| Finding links only to a promoted script | **Hook** (`guard_findings.sh`) + findings-manager |
| Seeds/locked-env/logging/fail-loud/parameterization | **Code-reviewer** agent (against this doc) |

The hook enforces the fast, safe, deterministic checks (lint, types, read-only data, promoted-link). Test-passing and the qualitative rules (fail-loud, parameterization, seeds recorded) are verified by the code-reviewer, because running arbitrary tests synchronously inside a tool-use hook is slow and unsafe. See `conventions/enforcement-map.md`.
