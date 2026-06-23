#!/usr/bin/env python3
"""Findings Workflow hook — promotion gate for analysis scripts (spec doc 05).

A script under ``scripts/promoted/`` must pass static checks before it is
accepted, because a finding may link only to a promoted script. This hook
enforces the deterministic, fast, side-effect-free part of the bar:
  - ruff  (lint, strict rule set) — blocks on failure when available
  - mypy  (strict type checking)  — blocks on failure when available AND configured
TESTS (unit/property/planted-truth/edge cases) are also required by convention
but are verified by the code-reviewer agent, not run here (running arbitrary
tests synchronously in a hook is slow and unsafe). See
conventions/enforcement-map.md.

STRICTNESS is driven by the PROJECT config (pyproject [tool.mypy] strict=true and
[tool.ruff.lint].select), which setup-env seeds. mypy is run from the project dir
so it loads that config; ruff is pointed at it explicitly (--config) so the strict
rule set applies even when checking the PreToolUse temp copy (ruff resolves config
from the file's own directory, not cwd).

PreToolUse(Write): validate the incoming content on a temp file; exit 2 blocks.
PostToolUse(Write|Edit): re-validate the on-disk file; warn only (write happened).

Scope + fail-open behavior match the other guards. Pure stdlib to run (no
``bash``/``jq``); resolves ruff/mypy from the project ./.venv first, then PATH —
so it runs unchanged on Windows, macOS, and Linux.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

from _hooklib import load_event, project_cwd, require_initialized, tool_input


def resolve_tool(name: str, cwd: str) -> str | None:
    """Resolve a tool from the project-local venv first, then PATH.

    The zero-global-footprint setup (setup-env) installs ruff/mypy into ./.venv
    rather than on PATH, so prefer that; fall back to PATH. Returns a runnable
    path, or None if unavailable (caller skips the check — fail open).
    """
    candidates = (
        os.path.join(cwd, ".venv", "bin", name),
        os.path.join(cwd, ".venv", "Scripts", f"{name}.exe"),
        os.path.join(cwd, ".venv", "Scripts", name),
    )
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which(name)


def ruff_config(cwd: str) -> str | None:
    """The project's ruff config, following ruff's own precedence
    (.ruff.toml > ruff.toml > pyproject.toml). Returns None if none present."""
    cfg: str | None = None
    for fname in ("pyproject.toml", "ruff.toml", ".ruff.toml"):
        path = os.path.join(cwd, fname)
        if os.path.isfile(path):
            cfg = path  # last existing wins, matching the precedence order
    return cfg


def run_checks(target: str, cwd: str) -> tuple[bool, str]:
    """Run the available static checks on ``target``.

    Returns (ok, messages). ok is False if any *available* check fails; a missing
    tool is skipped (fail open) and noted in the messages, never a failure.
    """
    failed = False
    msgs: list[str] = []

    ruff_bin = resolve_tool("ruff", cwd)
    if ruff_bin:
        cfg = ruff_config(cwd)
        cmd = [ruff_bin, "check"]
        if cfg:
            cmd += ["--config", cfg]
        cmd.append(target)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failed = True
            msgs.append("[ruff]\n" + (proc.stdout + proc.stderr).strip())
    else:
        msgs.append(
            "[ruff] not installed — lint check skipped "
            "(install ruff, e.g. via setup-env, to enforce)"
        )

    mypy_bin = resolve_tool("mypy", cwd)
    has_cfg = any(
        os.path.isfile(os.path.join(cwd, f))
        for f in ("pyproject.toml", "mypy.ini", "setup.cfg")
    )
    if mypy_bin and has_cfg:
        # Run from the project dir so mypy loads [tool.mypy] (strict=true).
        proc = subprocess.run(
            [mypy_bin, target], capture_output=True, text=True, cwd=cwd
        )
        if proc.returncode != 0:
            failed = True
            msgs.append("[mypy]\n" + (proc.stdout + proc.stderr).strip())
    else:
        msgs.append("[mypy] not installed or no type-check config — type check skipped")

    return (not failed), "\n".join(msgs)


def is_promoted_py(fp: str) -> bool:
    norm = fp.replace("\\", "/")
    return "scripts/promoted/" in norm and norm.endswith(".py")


def main() -> None:
    event = load_event()
    cwd = project_cwd(event)
    require_initialized(cwd)

    ti = tool_input(event)
    fp = ti.get("file_path") or ""
    if not fp or not is_promoted_py(fp):
        return

    fpath = fp if os.path.isabs(fp) else os.path.join(cwd, fp)
    event_name = event.get("hook_event_name") or ""

    if event_name == "PreToolUse":
        # Only Write carries full content; Edit is handled at PostToolUse.
        content = ti.get("content") or ""
        if not content:
            return
        # Materialize byte-faithfully (no newline translation) so the temp copy
        # matches what lands on disk — otherwise a stripped trailing newline trips
        # a spurious W292 under the strict ruleset.
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as tmp:
            tmp.write(content.encode("utf-8"))
            tmp_name = tmp.name
        try:
            ok, msgs = run_checks(tmp_name, cwd)
        finally:
            os.unlink(tmp_name)
        if not ok:
            print(
                f"Blocked: {os.path.basename(fp)} does not pass the promotion "
                "checks; a finding may not link to it until it does (doc 05). Fix "
                "the issues below or keep the script in scripts/scratch/.\nTests are "
                f"still required and are verified by the code-reviewer.\n{msgs}",
                file=sys.stderr,
            )
            sys.exit(2)
    else:
        # PostToolUse — file is on disk; cannot block, so warn.
        if os.path.isfile(fpath):
            ok, msgs = run_checks(fpath, cwd)
            if not ok:
                print(
                    f"Warning: promoted script {os.path.basename(fp)} currently "
                    "fails the promotion checks; it is not validly promoted and "
                    f"findings must not link to it until fixed (doc 05).\n{msgs}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
