"""Shared helpers for the Findings Workflow hook guards.

The guards (``guard_readonly_data.py``, ``guard_findings.py``,
``guard_promotion.py``) are Python so they run unchanged on Windows, macOS, and
Linux — the previous bash versions required ``bash`` + ``jq`` on PATH, which
native Windows lacks, so they silently failed open there (see hooks/README.md).
Python is already a hard dependency of the workflow (``setup-env`` bootstraps
>= 3.11), and stdlib ``json`` removes the ``jq`` dependency entirely.

This module is not itself a hook; it is imported by the guard scripts. The
sibling import works because Python prepends a script's own directory to
``sys.path`` when it is run as ``python /path/to/hooks/guard_x.py``.

Two invariants every guard preserves (conventions/enforcement-map.md):
  * **Scope to initialized projects** — engage only when ``state/workflow.json``
    is present, so the plugin never touches unrelated repositories.
  * **Fail open** — never wedge a session over the guard's own tooling or a
    malformed event; block (exit 2) only on a genuine, clearly-decidable
    violation.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def load_event() -> dict[str, Any]:
    """Read and parse the hook JSON event from stdin.

    Fail open (exit 0) if it cannot be read or parsed — a guard must never
    wedge a session over a malformed event.
    """
    try:
        obj = json.loads(sys.stdin.read())
    except Exception:
        sys.exit(0)
    return obj if isinstance(obj, dict) else {}


def project_cwd(event: dict[str, Any]) -> str:
    """The project directory the tool call targets (event ``cwd``, else PWD)."""
    cwd = event.get("cwd")
    return cwd if isinstance(cwd, str) and cwd else os.getcwd()


def require_initialized(cwd: str) -> None:
    """Engage only inside an initialized Findings Workflow project.

    Exits 0 (does nothing) when ``state/workflow.json`` is absent, so the
    guards never interfere with unrelated repositories.
    """
    if not os.path.isfile(os.path.join(cwd, "state", "workflow.json")):
        sys.exit(0)


def tool_input(event: dict[str, Any]) -> dict[str, Any]:
    """The ``tool_input`` mapping for the call, or an empty dict."""
    ti = event.get("tool_input")
    return ti if isinstance(ti, dict) else {}


def block(message: str) -> None:
    """Block the tool call: reason to stderr, exit 2 (the blocking exit code)."""
    print(message, file=sys.stderr)
    sys.exit(2)


def warn(message: str) -> None:
    """Emit a non-blocking warning to stderr (caller still exits 0)."""
    print(message, file=sys.stderr)
