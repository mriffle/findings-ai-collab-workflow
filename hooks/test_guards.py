#!/usr/bin/env python3
"""Synthetic-event tests for the Findings Workflow hook guards.

Stdlib-only and self-contained: run with any Python >= 3.11, no pytest/deps —

    python3 hooks/test_guards.py

Each case feeds a guard a synthetic PreToolUse/PostToolUse JSON event on stdin
(exactly as Claude Code does) and asserts the exit code (2 = block, 0 = allow)
and, where relevant, the stderr message. Exits non-zero if any case fails.

These replace the ad-hoc manual checks the bash guards relied on; the Python
guards are exercised here on every change.
"""
# ruff: noqa: E501  — test cases favor readable inline event/content literals over wrapping.

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PASS = 0
FAIL = 0


def run_guard(script: str, event: dict[str, object]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, script)],
        input=json.dumps(event),
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stderr


def check(name: str, result: tuple[int, str], want_rc: int, want_sub: str = "") -> None:
    global PASS, FAIL
    got_rc, stderr = result
    ok = got_rc == want_rc and (want_sub in stderr if want_sub else True)
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}: rc={got_rc} (want {want_rc}) stderr={stderr!r}")


def init_project(initialized: bool = True, gate_passed: bool = False) -> str:
    proj = tempfile.mkdtemp(prefix="fw_hook_test_")
    if initialized:
        os.makedirs(os.path.join(proj, "state"), exist_ok=True)
        with open(os.path.join(proj, "state", "workflow.json"), "w", encoding="utf-8") as fh:
            json.dump({"integrity_gate": {"passed": gate_passed}}, fh)
    return proj


def write_fake_tool(proj: str, name: str, exit_code: int) -> None:
    """A POSIX fake ruff/mypy in the project .venv so the promotion checks are
    deterministic without the real tools. Callers guard on os.name == 'posix'."""
    bindir = os.path.join(proj, ".venv", "bin")
    os.makedirs(bindir, exist_ok=True)
    path = os.path.join(bindir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"#!/bin/sh\nexit {exit_code}\n")
    os.chmod(path, 0o755)


# --------------------------------------------------------------------------- #
# guard_readonly_data.py
# --------------------------------------------------------------------------- #
def _ro(proj: str, tool: str, ti: dict[str, object]) -> tuple[int, str]:
    return run_guard("guard_readonly_data.py", {"cwd": proj, "tool_name": tool, "tool_input": ti})


def test_readonly() -> None:
    print("guard_readonly_data.py")
    proj = init_project()

    check("Write under data/ blocked", _ro(proj, "Write", {"file_path": "data/raw.csv"}),
          2, "read-only")
    check("Write nested under data/ blocked", _ro(proj, "Write", {"file_path": "data/sub/x.csv"}),
          2)
    check("Write absolute under data/ blocked",
          _ro(proj, "Write", {"file_path": os.path.join(proj, "data", "x.csv")}), 2)
    check("traversal into data/ blocked", _ro(proj, "Write", {"file_path": "sub/../data/x.csv"}),
          2)
    check("Write to results/ allowed", _ro(proj, "Write", {"file_path": "results/out.csv"}), 0)
    check("Edit under data/ blocked", _ro(proj, "Edit", {"file_path": "data/raw.csv"}), 2)
    check("data-lookalike (database/) allowed", _ro(proj, "Write", {"file_path": "database/x"}), 0)
    check("Bash redirect into data/ blocked", _ro(proj, "Bash", {"command": "echo x > data/y"}), 2)
    check("Bash rm in data/ blocked", _ro(proj, "Bash", {"command": "rm data/foo"}), 2)
    check("Bash read of data/ allowed", _ro(proj, "Bash", {"command": "cat data/x"}), 0)

    # Scope + fail-open
    uninit = init_project(initialized=False)
    check("uninitialized project: no-op", _ro(uninit, "Write", {"file_path": "data/raw.csv"}), 0)
    check("empty event: fail open", run_guard("guard_readonly_data.py", {}), 0)
    proc = subprocess.run([sys.executable, os.path.join(HERE, "guard_readonly_data.py")],
                          input="{not json", text=True, capture_output=True)
    check("malformed JSON: fail open", (proc.returncode, proc.stderr), 0)


# --------------------------------------------------------------------------- #
# guard_findings.py
# --------------------------------------------------------------------------- #
def _find(proj: str, fp: str, content: str) -> tuple[int, str]:
    return run_guard("guard_findings.py", {
        "cwd": proj, "tool_name": "Write",
        "tool_input": {"file_path": fp, "content": content}})


def test_findings() -> None:
    print("guard_findings.py")
    open_proj = init_project(gate_passed=False)
    passed_proj = init_project(gate_passed=True)

    check("validated before gate blocked",
          _find(open_proj, "findings/0001-x.md", "---\nstatus: validated\n---\n"),
          2, "integrity gate")
    check("signoff before gate blocked",
          _find(open_proj, "findings/0001-x.md", "---\nintegrity_signoff: true\n---\n"), 2)
    check("quoted signoff before gate blocked",
          _find(open_proj, "findings/0001-x.md", '---\nintegrity_signoff: "true"\n---\n'), 2)
    check("exploratory before gate allowed",
          _find(open_proj, "findings/0001-x.md", "---\nstatus: exploratory\n---\n"), 0)
    check("validated after gate allowed",
          _find(passed_proj, "findings/0001-x.md", "---\nstatus: validated\n---\n"), 0)
    check("validated + scratch link blocked",
          _find(passed_proj, "findings/0001-x.md",
                "---\nstatus: validated\nscript:\n  path: scripts/scratch/x.py\n---\n"),
          2, "promoted script")
    check("validated + promoted link allowed",
          _find(passed_proj, "findings/0001-x.md",
                "---\nstatus: validated\nscript:\n  path: scripts/promoted/x.py\n---\n"), 0)
    check("validated + prose scratch mention allowed",
          _find(passed_proj, "findings/0001-x.md",
                "---\nstatus: validated\n---\nCaveat: old scripts/scratch/x.py was discarded.\n"),
          0)
    check("non-finding file (manifest) ignored",
          _find(open_proj, "findings/manifest.md", "---\nstatus: validated\n---\n"), 0)
    check("non-findings path ignored",
          _find(open_proj, "reports/0001-x.md", "---\nstatus: validated\n---\n"), 0)


# --------------------------------------------------------------------------- #
# guard_promotion.py
# --------------------------------------------------------------------------- #
CLEAN_PY = "def add(a: int, b: int) -> int:\n    return a + b\n"


def _promo(proj: str, event_name: str, fp: str, content: str = "") -> tuple[int, str]:
    return run_guard("guard_promotion.py", {
        "cwd": proj, "hook_event_name": event_name, "tool_name": "Write",
        "tool_input": {"file_path": fp, "content": content}})


def _seed_promo_project(ruff_rc: int, mypy_rc: int) -> str:
    proj = init_project()
    write_fake_tool(proj, "ruff", ruff_rc)
    write_fake_tool(proj, "mypy", mypy_rc)
    with open(os.path.join(proj, "pyproject.toml"), "w", encoding="utf-8") as fh:
        fh.write("[tool.mypy]\nstrict = true\n")
    return proj


def test_promotion() -> None:
    print("guard_promotion.py")
    proj = init_project()

    check("non-promoted (scratch) ignored",
          _promo(proj, "PreToolUse", "scripts/scratch/x.py", "import nonsense !!!"), 0)
    check("non-py promoted file ignored",
          _promo(proj, "PreToolUse", "scripts/promoted/notes.md", "whatever"), 0)
    check("clean content allowed",
          _promo(proj, "PreToolUse", "scripts/promoted/ok.py", CLEAN_PY), 0)

    if os.name == "posix":
        blk = _seed_promo_project(ruff_rc=1, mypy_rc=0)
        check("failing ruff blocks (PreToolUse)",
              _promo(blk, "PreToolUse", "scripts/promoted/bad.py", CLEAN_PY),
              2, "promotion checks")

        ok = _seed_promo_project(ruff_rc=0, mypy_rc=0)
        check("passing tools allowed (PreToolUse)",
              _promo(ok, "PreToolUse", "scripts/promoted/good.py", CLEAN_PY), 0)

        # PostToolUse warns (exit 0) on an on-disk file failing the checks.
        post = _seed_promo_project(ruff_rc=1, mypy_rc=0)
        os.makedirs(os.path.join(post, "scripts", "promoted"), exist_ok=True)
        with open(os.path.join(post, "scripts", "promoted", "p.py"), "w", encoding="utf-8") as fh:
            fh.write(CLEAN_PY)
        check("PostToolUse warns not blocks",
              _promo(post, "PostToolUse", "scripts/promoted/p.py"), 0, "Warning")


def main() -> int:
    test_readonly()
    test_findings()
    test_promotion()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
