#!/usr/bin/env python3
"""Findings Workflow hook — raw data is read-only (spec doc 05.1).

Blocks Write/Edit to anything under the project's ``data/`` directory, and makes
a best-effort attempt to block obvious Bash writes into ``data/``.

Scope: only engages inside an initialized Findings Workflow project
(``state/workflow.json`` present), so it never interferes with unrelated
repositories. Fails OPEN (allows the action) if the event is unparseable — the
gate is defense-in-depth alongside orchestrator behavior, and should never wedge
a session over its own tooling.

Cross-platform: pure stdlib (no ``bash``/``jq``), so it runs unchanged on
Windows, macOS, and Linux. Reads the PreToolUse JSON event on stdin; exit 2 +
stderr message blocks the tool call.
"""

from __future__ import annotations

import os
import re

from _hooklib import block, load_event, project_cwd, require_initialized, tool_input

# Redirects into data/, or rm/mv/cp/tee/truncate/dd/sed -i into data/ (best-effort).
_BASH_WRITE = re.compile(
    r"(>>?|tee\s+(-a\s+)?)\s*\.?/?data/"
    r"|(\b(rm|mv|cp|truncate|dd)\b|sed\s+-i)[^|]*\s\.?/?data/"
)


def is_under_data(fp: str, cwd: str) -> bool:
    """True if ``fp`` — after lexical normalization, relative paths anchored to
    ``cwd`` — is ``<cwd>/data`` or under it.

    Lexical only (``os.path.normpath`` collapses ``.``/``..``/``//`` without any
    filesystem access), so a traversal like ``sub/../data/raw.csv`` can't slip
    past and symlinks are deliberately not resolved — matching the original
    bash guard. ``normcase`` makes the comparison correct on case-insensitive
    Windows while staying a no-op on POSIX.
    """
    cand = fp if os.path.isabs(fp) else os.path.join(cwd, fp)
    cand = os.path.normcase(os.path.normpath(cand))
    data_dir = os.path.normcase(os.path.normpath(os.path.join(cwd, "data")))
    return cand == data_dir or cand.startswith(data_dir + os.sep)


def main() -> None:
    event = load_event()
    cwd = project_cwd(event)
    require_initialized(cwd)

    tool = event.get("tool_name") or ""
    ti = tool_input(event)

    if tool in ("Write", "Edit"):
        fp = ti.get("file_path") or ""
        if fp and is_under_data(fp, cwd):
            block(
                "Blocked: data/ holds immutable raw inputs and is read-only in the "
                "Findings Workflow (doc 05.1). Outputs belong in results/ or figures/, "
                "which are regenerated from data + a script."
            )
    elif tool == "Bash":
        cmd = ti.get("command") or ""
        if cmd and _BASH_WRITE.search(cmd):
            block(
                "Blocked (best-effort): this command appears to write or modify the "
                "read-only data/ directory (doc 05.1). If this is a false positive, "
                "perform the write outside data/."
            )


if __name__ == "__main__":
    main()
