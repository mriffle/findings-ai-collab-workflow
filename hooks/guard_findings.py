#!/usr/bin/env python3
"""Findings Workflow hook — integrity-gate + promoted-script guard on findings.

Spec: docs 02.3, 03, 05. Enforces two invariants when a finding file
(``findings/NNNN-*.md``) is written or edited:
  1. A finding may not claim ``integrity_signoff: true`` or ``status: validated``
     before the integrity gate has passed
     (``state/workflow.json .integrity_gate.passed == true``).
  2. A ``validated`` finding may link only to a promoted script
     (``scripts/promoted/``), never ``scripts/scratch/``.

This is a deterministic backstop; the findings-manager is the authoritative
enforcer and applies the full ``validated`` bar. Scope + fail-open behavior match
the other guards. Pure stdlib (no ``bash``/``jq``) so it runs unchanged on
Windows, macOS, and Linux.

Reads the PreToolUse JSON event on stdin; exit 2 + stderr blocks the tool call.
"""

from __future__ import annotations

import json
import os
import re

from _hooklib import block, load_event, project_cwd, require_initialized, tool_input

# A finding basename: NNNN-*.md (four leading digits).
_FINDING_BASENAME = re.compile(r"^[0-9]{4}.*\.md$")

# Allow an optional single/double quote before the value so a YAML-legal quoted
# scalar (integrity_signoff: "true") can't slip past the gate.
_M = re.MULTILINE
_CLAIMS_SIGNOFF = re.compile(r'^[ \t]*integrity_signoff:[ \t]*["\']?true\b', _M)
_CLAIMS_VALIDATED = re.compile(r'^[ \t]*status:[ \t]*["\']?validated\b', _M)

# Scope the scratch-link check to an actual YAML script-path value — a `path:` key
# line (block form) or an inline `script: { path: ... }` mapping — so a mere prose
# mention of scripts/scratch/ (e.g. in a Caveat) doesn't trigger a false block.
_SCRATCH_PATH_LINE = re.compile(r'^[ \t]*path:[ \t]*["\']?scripts/scratch/', _M)
_SCRATCH_INLINE = re.compile(r"script:[ \t]*\{[^}]*scripts/scratch/")


def gate_passed(cwd: str) -> bool:
    """Whether the integrity gate has passed per state/workflow.json.

    Any read/parse failure yields False (treat the gate as not passed) — the
    guard's job is to prevent a premature sign-off, so an undeterminable state is
    the safe (blocking) side, mirroring the original bash guard.
    """
    try:
        with open(os.path.join(cwd, "state", "workflow.json"), encoding="utf-8") as fh:
            state = json.load(fh)
        return state.get("integrity_gate", {}).get("passed") is True
    except Exception:
        return False


def main() -> None:
    event = load_event()
    cwd = project_cwd(event)
    require_initialized(cwd)

    ti = tool_input(event)
    fp = ti.get("file_path") or ""
    if not fp:
        return

    # Only finding documents: a path containing findings/ and a basename NNNN-*.md.
    norm = fp.replace("\\", "/")
    if "findings/" not in norm:
        return
    base = norm.rsplit("/", 1)[-1]
    if not _FINDING_BASENAME.match(base):
        return

    # Content being written (Write: .content; Edit: .new_string).
    content = ti.get("content")
    if not content:
        content = ti.get("new_string") or ""
    if not content:
        return

    claims_signoff = bool(_CLAIMS_SIGNOFF.search(content))
    claims_validated = bool(_CLAIMS_VALIDATED.search(content))

    if (claims_signoff or claims_validated) and not gate_passed(cwd):
        block(
            "Blocked: a finding cannot claim integrity_signoff: true or status: "
            "validated before the integrity gate passes (state/workflow.json "
            ".integrity_gate.passed is not true). Complete stage3-loaders and obtain "
            "sign-off first (docs 02.3, 05)."
        )

    if claims_validated and (
        _SCRATCH_PATH_LINE.search(content) or _SCRATCH_INLINE.search(content)
    ):
        block(
            "Blocked: a validated finding may link only to a promoted script "
            "(scripts/promoted/), not scripts/scratch/ (docs 03, 05). Promote the "
            "script and re-point provenance.script.path before validating."
        )


if __name__ == "__main__":
    main()
