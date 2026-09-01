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
import shutil
import subprocess
import sys
import tempfile

import guard_promotion

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


def _find_edit(proj: str, fp: str, new_string: str) -> tuple[int, str]:
    return run_guard("guard_findings.py", {
        "cwd": proj, "tool_name": "Edit",
        "tool_input": {"file_path": fp, "new_string": new_string}})


# Whole-finding fixtures for the figure-embed backstop.
_FIG_FM = (
    '---\nid: 42\nstatus: candidate\nfigures:\n'
    '  - { png: "figures/0042-volcano.png", svg: "figures/0042-volcano.svg",'
    ' legend_png: "figures/0042-volcano.legend.png", caption: "V" }\n---\n'
)
_FIG_EMBEDDED = _FIG_FM + "\n# T\n\n## Evidence\n![Volcano](figures/0042-volcano.png)\n"
_FIG_MISSING = _FIG_FM + "\n# T\n\n## Evidence\nNumbers, but no figure image.\n"
_FIG_LEGEND_ONLY = _FIG_FM + "\n# T\n\n## Evidence\n![key](figures/0042-volcano.legend.png)\n"
_FIG_RELPATH = _FIG_FM + "\n# T\n\n## Evidence\n![Volcano](./figures/0042-volcano.png)\n"
_FIG_EMPTY_LIST = "---\nid: 42\nstatus: candidate\nfigures: []\n---\n\n# T\n\n## Evidence\nNo figures.\n"
_FIG_FM_ONLY = _FIG_FM  # frontmatter, empty body → fail open (write in progress)

# Fixtures for the embedded⇒listed backstop (invariant 4): a shown figure must
# carry its own provenance, i.e. appear in the `figures` frontmatter.
_FIG_UNLISTED = (
    _FIG_FM + "\n# T\n\n## Evidence\n![Volcano](figures/0042-volcano.png)\n"
    "![PCA](figures/0042-pca.png)\n"
)
_FIG_UNLISTED_HTML = (
    _FIG_FM + "\n# T\n\n## Evidence\n![Volcano](figures/0042-volcano.png)\n"
    '<img src="figures/0042-pca.png" alt="PCA">\n'
)
_FIG_EMBED_NO_LIST = (
    "---\nid: 42\nstatus: candidate\nfigures: []\n---\n"
    "\n# T\n\n## Evidence\n![PCA](figures/0042-pca.png)\n"
)
_FIG_WITH_LEGEND = (
    _FIG_FM + "\n# T\n\n## Evidence\n![Volcano](figures/0042-volcano.png)\n"
    "![key](figures/0042-volcano.legend.png)\n"
)
_FIG_MD_TITLE = (
    _FIG_FM + '\n# T\n\n## Evidence\n![Volcano](figures/0042-volcano.png "Volcano")\n'
)
_FIG_EXTERNAL_IMG = (
    _FIG_FM + "\n# T\n\n## Evidence\n![Volcano](figures/0042-volcano.png)\n"
    "![schematic](https://example.org/s.png)\n"
)


# Fixtures for the cross-reference link backstop (invariant 5): a finding that
# names another finding links to it (conventions/findings.md §2.7).
_XREF_FM = '---\nid: 42\nstatus: candidate\nfigures: []\n---\n'
_XREF_LINKED = _XREF_FM + "\n# T\n\n## Related findings\nRefines [finding 0031](0031-sex-confound.md).\n"
_XREF_UNLINKED = _XREF_FM + "\n# T\n\n## Related findings\nThis refines finding 0031.\n"
_XREF_BARE_NUMBERS = _XREF_FM + "\n# T\n\n## Evidence\nCollected in 2026; n=1998 precursors.\n"
_XREF_OWN_ID = _XREF_FM + "\n# T\n\n## Summary\nThis is finding 0042, recorded today.\n"
_XREF_ALT_TEXT = _XREF_FM + "\n# T\n\n## Caveats\nQualified by [the batch caveat (0007)](0007-batch-skew.md).\n"
_XREF_MIXED = (
    _XREF_FM + "\n# T\n\n## Related findings\nSee [finding 0031](0031-x.md) "
    "and finding 0055.\n"
)
_XREF_WRONG_TARGET = _XREF_FM + "\n# T\n\n## Related findings\nSee [finding 0031](0099-other.md).\n"


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

    # Figure-embed backstop (invariant 3).
    check("listed figure not embedded blocked",
          _find(open_proj, "findings/0042-v.md", _FIG_MISSING), 2, "inline")
    check("listed figure embedded allowed",
          _find(open_proj, "findings/0042-v.md", _FIG_EMBEDDED), 0)
    check("only the legend embedded still blocks (main png required)",
          _find(open_proj, "findings/0042-v.md", _FIG_LEGEND_ONLY), 2, "inline")
    check("embedded via ./ relative path allowed (basename match)",
          _find(open_proj, "findings/0042-v.md", _FIG_RELPATH), 0)
    check("empty figures list allowed",
          _find(open_proj, "findings/0042-v.md", _FIG_EMPTY_LIST), 0)
    check("frontmatter-only write (empty body) fails open",
          _find(open_proj, "findings/0042-v.md", _FIG_FM_ONLY), 0)
    check("Edit fragment (no delimiters) fails open",
          _find_edit(open_proj, "findings/0042-v.md",
                     '  - { png: "figures/0042-volcano.png" }\n'), 0)
    check("missing-figure block still fires after the gate passes",
          _find(passed_proj, "findings/0042-v.md", _FIG_MISSING), 2, "inline")

    # Embedded⇒listed backstop (invariant 4): a shown figure carries its provenance.
    check("embedded but unlisted figure blocked",
          _find(open_proj, "findings/0042-v.md", _FIG_UNLISTED), 2, "but not listed")
    check("embedded but unlisted figure blocked (HTML img)",
          _find(open_proj, "findings/0042-v.md", _FIG_UNLISTED_HTML), 2, "but not listed")
    check("embedded figure with an empty figures list blocked",
          _find(open_proj, "findings/0042-v.md", _FIG_EMBED_NO_LIST), 2, "but not listed")
    check("embedded legend image needs no figures entry",
          _find(open_proj, "findings/0042-v.md", _FIG_WITH_LEGEND), 0)
    check("markdown image title stripped before matching",
          _find(open_proj, "findings/0042-v.md", _FIG_MD_TITLE), 0)
    check("image outside figures/ ignored",
          _find(open_proj, "findings/0042-v.md", _FIG_EXTERNAL_IMG), 0)
    check("Edit fragment with an unlisted image fails open",
          _find_edit(open_proj, "findings/0042-v.md",
                     "![PCA](figures/0042-pca.png)\n"), 0)
    check("unlisted-figure block still fires after the gate passes",
          _find(passed_proj, "findings/0042-v.md", _FIG_UNLISTED), 2, "but not listed")

    # Cross-reference link backstop (invariant 5).
    check("linked finding mention allowed",
          _find(open_proj, "findings/0042-v.md", _XREF_LINKED), 0)
    check("unlinked finding mention blocked",
          _find(open_proj, "findings/0042-v.md", _XREF_UNLINKED), 2, "must be a link")
    check("bare 4-digit numbers (year, n) are not citations",
          _find(open_proj, "findings/0042-v.md", _XREF_BARE_NUMBERS), 0)
    check("a finding may name its own id unlinked",
          _find(open_proj, "findings/0042-v.md", _XREF_OWN_ID), 0)
    check("link text without the word 'finding' still counts",
          _find(open_proj, "findings/0042-v.md", _XREF_ALT_TEXT), 0)
    check("one linked, one unlinked -> blocks on the unlinked",
          _find(open_proj, "findings/0042-v.md", _XREF_MIXED), 2, "0055")
    check("link pointing at the wrong finding file blocked",
          _find(open_proj, "findings/0042-v.md", _XREF_WRONG_TARGET), 2, "must be a link")
    check("Edit fragment with an unlinked mention fails open",
          _find_edit(open_proj, "findings/0042-v.md", "This refines finding 0031.\n"), 0)


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

    # Cross-platform: when a real ruff is available (e.g. in CI on every OS,
    # incl. Windows), a lint-failing script must block — this exercises the actual
    # tool-execution path, not a POSIX-only fake. `import os` (unused) trips F401.
    if shutil.which("ruff"):
        rp = init_project()
        check("real ruff blocks a lint-failing script",
              _promo(rp, "PreToolUse", "scripts/promoted/bad.py", "import os\n"),
              2, "promotion checks")

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


def test_resolve_tool() -> None:
    """The OS-specific venv tool resolution (the Windows `.venv/Scripts/*.exe`
    branch is the one the bash guard couldn't have on Windows)."""
    print("guard_promotion.resolve_tool (project venv layout)")
    proj = init_project()
    rel = ("bin", "ruff") if os.name == "posix" else ("Scripts", "ruff.exe")
    full = os.path.join(proj, ".venv", *rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write("")
    if os.name == "posix":
        os.chmod(full, 0o755)
    got = guard_promotion.resolve_tool("ruff", proj)
    check("resolves the OS-specific .venv tool path",
          (0 if got == full else 1, f"got={got!r} want={full!r}"), 0)


def main() -> int:
    test_readonly()
    test_findings()
    test_promotion()
    test_resolve_tool()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
