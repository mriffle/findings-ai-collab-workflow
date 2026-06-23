---
name: setup-env
description: Ensure this project has a usable, project-local Python environment (>= 3.11). Detects existing Python; if none is suitable, transparently offers to download uv and install Python into this project (zero global footprint), then creates the project .venv, lockfile, and baseline tooling.
---

# Set up the project Python environment

The workflow runs all analysis, loaders, and figures in **Python** (`conventions/coding.md`). This command makes sure this project has a usable interpreter and a locked, project-local environment **before** any Python is executed (Stage 3 onward). It is **idempotent** — safe to re-run; it never overwrites a working setup.

**The floor is Python ≥ 3.11** (`conventions/coding.md`). Anything older is treated as "not suitable."

**Two principles govern this command:**
- **Project-local, zero global footprint.** When we install anything, it goes *into this project* — the `uv` binary under `./.uv/bin`, the interpreter under `./.uv/python`, the virtualenv under `./.venv`. We do **not** modify the user's `PATH`, shell profile, or any system location. Removing the project removes everything.
- **Transparent consent before any download.** We never download software silently. If — and only if — a download is required, we explain what `uv` is, why we want it, and exactly what will happen, then ask the scientist to approve. If they decline, they install Python ≥ 3.11 themselves and we proceed once it is present.

## Step 1 — Detect what already exists (no side effects)

Check, in order, and stop at the first that satisfies the floor:

1. **An existing project env.** If `./.venv` exists and `./.venv/bin/python` (Unix) or `./.venv/Scripts/python.exe` (Windows) reports **≥ 3.11**, the project is already set up. Report it as **kept**, run Step 5 (baseline tooling) idempotently, and stop.
2. **A usable `uv`.** Is `./.uv/bin/uv` (project-local) present, or `uv` on `PATH`? Note it — we can reuse it instead of downloading it.
3. **A usable system Python ≥ 3.11.** Probe `python3 --version`, then `python --version`, then (Windows) `py -3 --version`. Parse the `MAJOR.MINOR` and compare to `3.11`.

From this, determine **what (if anything) must be downloaded**:
- the **`uv` binary** — needed only if no usable `uv` was found in (2);
- a **Python interpreter** — needed only if no usable Python ≥ 3.11 was found in (3) (uv will fetch one).

If **nothing needs downloading** (a suitable `uv` and a suitable Python already exist), skip Step 2 and go straight to Step 3 with a brief notice of what you're reusing.

## Step 2 — Transparent consent (only when a download is required)

Tell the scientist, in plain terms, before doing anything:

> Your environment doesn't have a Python that meets this workflow's floor (≥ 3.11)[, and `uv` isn't installed]. I can set one up **inside this project** without touching anything else on your system. Here's exactly what that involves:
>
> - **What `uv` is:** a fast, widely-used Python package & version manager (from Astral). It can download a self-contained Python interpreter and manage a locked dependency set — which is what makes every finding computationally reproducible.
> - **What I'll download:** the `uv` binary (~a few tens of MB) [and a standalone Python 3.11 interpreter (~a few hundred MB)].
> - **Where it goes:** entirely under this project — `./.uv/bin` (uv), `./.uv/python` (interpreter), `./.venv` (the virtualenv). I will set the install flags that **prevent any change to your PATH, shell profile, or system Python**, and disable uv self-updates.
> - **To undo it later:** delete `./.uv`, `./.venv`, `./.python-version`, `./uv.lock` — nothing is left elsewhere.
> - **Source:** the official one-line installer from https://github.com/astral-sh/uv.
>
> Shall I proceed? If you'd rather install Python ≥ 3.11 yourself, decline and I'll continue once it's available.

Wait for an explicit answer.

**If the scientist declines:** do **not** install anything. Record the decline (Step 6, `mode: "system"`, `declined: true`), and tell them: *"No problem — install Python ≥ 3.11 yourself (e.g. python.org, your OS package manager, pyenv, or conda), then re-run `setup-env`. Stage 3 stays blocked until a usable interpreter is available."* Stop here.

**If the scientist approves:** continue to Step 3.

## Step 3 — Install (project-local) and create the environment

Detect the platform (you know it from your environment). Use an **absolute** project path for the env-var values (`$PWD` on Unix, the resolved project path on Windows). Run from the project root.

**3a. Install `uv` into the project (only if it must be downloaded).** The key flag is `UV_UNMANAGED_INSTALL`, which installs uv to the given directory **and** suppresses all PATH/profile/env modification and self-update (`UV_INSTALL_DIR` + `UV_NO_MODIFY_PATH` are the lower-level equivalents).

- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_UNMANAGED_INSTALL="$PWD/.uv/bin" sh
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:UV_UNMANAGED_INSTALL = "$PWD\.uv\bin"
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

The project-local uv is invoked by its explicit path (`./.uv/bin/uv` on Unix, `.\.uv\bin\uv.exe` on Windows). If a usable `uv` already existed, use that instead.

**3b. Pin the interpreter into the project.** Set `UV_PYTHON_INSTALL_DIR` so any interpreter uv downloads lands **inside the project**, then pin and create the venv. Keep `UV_PYTHON_INSTALL_DIR` set on **every** uv invocation that may resolve or change the interpreter:

- **macOS / Linux:**
  ```bash
  export UV_PYTHON_INSTALL_DIR="$PWD/.uv/python"
  ./.uv/bin/uv python pin 3.11          # writes ./.python-version
  ./.uv/bin/uv venv --python 3.11       # creates ./.venv (downloads a 3.11 into .uv/python only if needed)
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:UV_PYTHON_INSTALL_DIR = "$PWD\.uv\python"
  .\.uv\bin\uv.exe python pin 3.11
  .\.uv\bin\uv.exe venv --python 3.11
  ```

`uv venv` reuses a suitable already-present interpreter if one exists and only downloads when it must — so an approved install with a fine system Python still avoids an unnecessary interpreter download.

**Fail open with a clear message.** If a download fails (no network, corporate proxy / SSL interception, or — on Windows — execution-policy blocking the script), do **not** leave a half-installed state: report the specific error and fall back to the decline path (Step 2), telling the scientist to install Python ≥ 3.11 themselves and re-run. Clean up an empty `./.uv` if nothing was installed.

## Step 4 — Establish the locked dependency set

If `./pyproject.toml` is absent, create a minimal one (do not overwrite an existing one):

```toml
[project]
name = "findings-workflow-study"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = []

[tool.mypy]
python_version = "3.11"
strict = true                    # full strict type checking — the promotion type-gate has real teeth
ignore_missing_imports = true    # scientific deps often ship no stubs; don't fail the gate on those

[tool.ruff]
target-version = "py311"

[tool.ruff.lint]
# Strict, explicit rule set (conventions/coding.md) — NOT ruff's minimal defaults.
select = ["E", "F", "W", "I", "B", "UP", "SIM", "C4", "PD", "NPY", "RUF"]
```

Seed the `[tool.mypy]` **and** `[tool.ruff]` blocks above — **the strictness lives in this config**, and the promotion hook (`ruff check` + `mypy`) reads it. Without them both gates are near-no-ops: bare `mypy` lets untyped functions through, and bare `ruff` checks only a minimal default rule set. `strict = true` gives the type-gate full teeth (untyped/partially-typed defs, implicit `Any`, and unused ignores all become errors), while `ignore_missing_imports = true` keeps it failing on *real* type errors rather than on unstubbed scientific libraries. The `select` set is the strict lint bar (`conventions/coding.md`); tune per-rule *ignores* per study if a rule genuinely doesn't fit, but do not weaken the set below this baseline.

Then add the baseline with uv (this resolves, installs into `./.venv`, and writes `./uv.lock` — the lockfile recorded in every finding's `provenance.environment`). Invoke uv with `UV_PYTHON_INSTALL_DIR` set (as in 3b):

```bash
# Quality tooling the gates depend on (dev deps):
./.uv/bin/uv add --dev ruff mypy pytest hypothesis
# Recommended scientific baseline (tune per study):
./.uv/bin/uv add numpy pandas scipy scikit-learn matplotlib statsmodels
# Normalization the lib/ templates seed from (pronoms: median / MAD / VSN normalizers).
# The normalize template is version-sensitive (MADNormalizer's scaling default and the
# VSN engine API), so pin it; only add when the study normalizes (omics intensities):
./.uv/bin/uv add 'pronoms==0.4.0'
# ComBat for the batch-correction template (pycombat exposes `Combat`); add only when
# the study has a batch axis to correct:
./.uv/bin/uv add 'pycombat==0.20'
```

The **dev tooling is not optional**: `ruff` + `mypy` power the promotion hook and `pytest` + `hypothesis` power the code-reviewer's test check. Without them in `./.venv`, the promotion gate degrades to a no-op. The scientific baseline is a sensible default — add/remove packages as the study needs (`uv add` / `uv remove`), and the lockfile updates.

## Step 5 — `.gitignore` the env (idempotent)

Ensure the project `./.gitignore` contains these entries (create the file if absent; append any missing lines; never duplicate):

```gitignore
# Findings Workflow — project-local Python environment (regenerable; not committed)
.uv/
.venv/
__pycache__/
*.pyc
```

The interpreter and venv are **derived state** — regenerable from `pyproject.toml` + `uv.lock` + `.python-version`, which **are** committed. Do not commit `.uv/` or `.venv/`.

## Step 6 — Record the environment in workflow state

Update the `environment` block in `state/workflow.json` (schema: `conventions/workflow-state.md`) and bump `updated`:

```json
"environment": {
  "mode": "project-uv",
  "python_min": "3.11",
  "interpreter": ".venv/bin/python",
  "configured": true,
  "declined": false,
  "updated": "<YYYY-MM-DD>"
}
```

- `mode`: `"project-uv"` when we created a project-local env; `"system"` when reusing a suitable system Python (or when the scientist declined and will provide their own).
- `interpreter`: the venv python path (`.venv/Scripts/python.exe` on Windows), or `"system"` if relying on a system interpreter without a project venv.
- `configured`: `true` only once a usable env (≥ `python_min`) has actually been verified; `false` if the scientist declined and no suitable Python is present yet.
- `declined`: `true` if the scientist declined the project-local install.

This block is **advisory state**; the Stage 3 gate still *live-verifies* a working interpreter (it never trusts a stale flag).

## Step 7 — Report

Print a created-vs-kept summary and how to use the environment in this project:

- **Run Python / scripts:** use the project venv directly — `./.venv/bin/python …` (Unix) or `.\.venv\Scripts\python …` (Windows). Equivalently `./.uv/bin/uv run python …` (with `UV_PYTHON_INSTALL_DIR` set) ensures the env is synced first.
- **Change dependencies:** `UV_PYTHON_INSTALL_DIR="$PWD/.uv/python" ./.uv/bin/uv add|remove <pkg>` (re-locks automatically).
- **The committed reproducibility artifacts** are `pyproject.toml`, `uv.lock`, and `.python-version`; `.uv/` and `.venv/` are git-ignored and rebuilt with `./.uv/bin/uv sync`.

End by pointing to the next workflow step (e.g. *"Environment ready. Continue the workflow; Python execution unlocks at Stage 3 — run `stage3-loaders` when you reach it."*).
