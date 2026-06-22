# Convention — Coding

*Spec source: doc 05.2 + the "analysis code held to the maximum" rule in doc 05.4. These are the rules the code-reviewer and the promotion hook enforce. Authoritative tool choices are named here so enforcement is concrete.*

## Language & environment

- **Python only.** All analysis, loaders, and figure code are Python.
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
- **Typing:** **type hints throughout** with a type checker in the loop (**`mypy`** or `pyright`), clean on the file.
- **Linting/formatting:** **`ruff check`** clean and **`ruff format`** applied (or an equivalent).

Code that has not passed tests, types, and lint is `scratch`, and a finding may not link to it.

## Enforcement

| Rule | Enforced by |
|---|---|
| Raw data read-only; outputs regenerable | **Hook** (`guard_readonly_data.sh`) |
| Script not promoted until lint + types pass | **Hook** (`guard_promotion.sh`: ruff + mypy) |
| Script not promoted until **tests** pass | **Code-reviewer** agent (tests aren't run in-hook) |
| Finding links only to a promoted script | **Hook** (`guard_findings.sh`) + findings-manager |
| Seeds/locked-env/logging/fail-loud/parameterization | **Code-reviewer** agent (against this doc) |

The hook enforces the fast, safe, deterministic checks (lint, types, read-only data, promoted-link). Test-passing and the qualitative rules (fail-loud, parameterization, seeds recorded) are verified by the code-reviewer, because running arbitrary tests synchronously inside a tool-use hook is slow and unsafe. See `conventions/enforcement-map.md`.
