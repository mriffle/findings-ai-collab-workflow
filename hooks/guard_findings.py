#!/usr/bin/env python3
"""Findings Workflow hook — integrity-gate + promoted-script + figure-embed guard.

Spec: docs 02.3, 03, 05, 06. Enforces five invariants when a finding file
(``findings/NNNN-*.md``) is written or edited:
  1. A finding may not claim ``integrity_signoff: true`` or ``status: validated``
     before the integrity gate has passed
     (``state/workflow.json .integrity_gate.passed == true``).
  2. A ``validated`` finding may link only to a promoted script
     (``scripts/promoted/``), never ``scripts/scratch/``.
  3. On a *complete* finding write (frontmatter + a non-empty body), every figure
     listed in the ``figures`` frontmatter must be embedded as an inline image in
     the body — a finding is a standalone artifact and the reader must never have
     to track down a figure it lists (conventions/findings.md §2.4). This check
     fails open on an Edit fragment that doesn't carry the whole document and on
     an empty ``figures`` list.
  4. The converse: every inline body image pointing under ``figures/`` must be
     listed in the ``figures`` frontmatter, so a figure the finding *shows*
     always carries its own producing script + input (per-figure provenance;
     conventions/findings.md §2.4). Legend images (``<base>.legend.png``) are
     exempt — they are a figure's key, carried by its entry's ``legend_png``.
     Same fail-open scope as invariant 3.

  5. Every mention of another finding in the body is a **link**: a ``finding NNNN``
     reference must sit inside a markdown link (conventions/findings.md §2.7), so
     a reader never has to hunt for a cited finding. Deliberately narrow — only a
     4-digit id directly preceded by the word ``finding``/``findings`` counts, so
     a bare number (a year, an n) can never trigger it. A finding's own id is
     exempt (a document may name itself). Same fail-open scope as 3 and 4.

Neither figure check can judge whether a *showable claim* was left unillustrated,
or whether an embedded figure was actually explained in the prose — those are the
findings-manager's judgment calls ("show, don't tell", conventions/findings.md
§2.4 coverage + §9 the reading).

This is a deterministic backstop; the findings-manager is the authoritative
enforcer and applies the full ``validated`` bar + figure completeness. Scope +
fail-open behavior match the other guards. Pure stdlib (no ``bash``/``jq``) so it
runs unchanged on Windows, macOS, and Linux.

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

# Figure-embed check. A figure PNG listed in the `figures` frontmatter: the
# (?<!\w) lookbehind excludes `legend_png:` (and any other `*_png:` key), so only
# the main figure raster — the inline-embed target — is required in the body.
_OPEN_FM = re.compile(r"^\s*---[ \t]*\n")
_CLOSE_FM = re.compile(r"\n---[ \t]*(?:\n|$)")
_FIGURE_PNG = re.compile(r'(?<!\w)png:[ \t]*["\']?(figures/[^"\'\s,}\]]+\.png)')
# Inline image targets in the body: markdown `![alt](target)` and HTML `<img src=…>`.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_HTML_IMG = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)

# Cross-reference check (invariant 5). A mention is the word finding/findings
# followed by a 4-digit id — narrow on purpose, so a bare 4-digit number (a year,
# an n, a count) is never mistaken for a citation.
_FINDING_MENTION = re.compile(r"\bfindings?\s+#?([0-9]{4})\b", re.IGNORECASE)
# Markdown links in the body: `[text](target)`. A mention counts as linked when it
# falls inside the *text* of a link whose target names that id's file.
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
# The finding's own id, so a document may name itself without linking.
_OWN_ID = re.compile(r"^[ \t]*id:[ \t]*[\"\']?([0-9]+)", _M)


def _split_frontmatter(content: str) -> tuple[str | None, str]:
    """Split a finding into (frontmatter, body), or (None, "") if it isn't a
    complete document (no opening/closing `---` delimiter line) — an Edit
    fragment fails open here."""
    m_open = _OPEN_FM.match(content)
    if not m_open:
        return None, ""
    rest = content[m_open.end() :]
    m_close = _CLOSE_FM.search(rest)
    if not m_close:
        return None, ""
    return rest[: m_close.start()], rest[m_close.end() :]


def _image_targets(body: str) -> list[str]:
    """Inline image targets in the body — markdown ``![alt](target)`` and HTML
    ``<img src=…>`` — with a markdown title (``(path "title")``) and angle
    brackets (``(<path>)``) stripped, so the bare path is compared."""
    targets = []
    for raw in _MD_IMAGE.findall(body) + _HTML_IMG.findall(body):
        stripped = raw.strip()
        if not stripped:
            continue
        target = stripped.split()[0].strip("<>")
        if target:
            targets.append(target)
    return targets


def unembedded_figures(content: str) -> list[str]:
    """PNG paths listed in the `figures` frontmatter but not embedded inline in
    the body. Empty (no violation) unless `content` is a complete finding
    document with a non-empty body and a non-empty `figures` list.

    A listed path counts as embedded when it — or its basename — appears inside
    any inline image target in the body (lenient over `./figures/…` vs `figures/…`
    so only a genuinely missing figure blocks)."""
    frontmatter, body = _split_frontmatter(content)
    if frontmatter is None or not body.strip():
        return []
    listed = _FIGURE_PNG.findall(frontmatter)
    if not listed:
        return []
    targets = _image_targets(body)
    missing = []
    for png in listed:
        base = png.rsplit("/", 1)[-1]
        if not any(png in t or base in t for t in targets):
            missing.append(png)
    return missing


def unlisted_figures(content: str) -> list[str]:
    """Inline body images under ``figures/`` that are *not* listed in the
    ``figures`` frontmatter — a shown figure with no per-figure provenance
    (conventions/findings.md §2.4).

    Empty (no violation) unless ``content`` is a complete finding document with a
    non-empty body. Legend images (``<base>.legend.png``) are excluded: they are a
    figure's key, carried by its entry's ``legend_png``, not figures in their own
    right. Matching is basename-lenient, mirroring ``unembedded_figures``.
    """
    frontmatter, body = _split_frontmatter(content)
    if frontmatter is None or not body.strip():
        return []
    listed_bases = {p.rsplit("/", 1)[-1] for p in _FIGURE_PNG.findall(frontmatter)}
    unlisted = []
    for target in _image_targets(body):
        norm = target.replace("\\", "/")
        if "figures/" not in norm:
            continue
        base = norm.rsplit("/", 1)[-1]
        if ".legend." in base:
            continue
        if base not in listed_bases and target not in unlisted:
            unlisted.append(target)
    return unlisted


def unlinked_mentions(content: str) -> list[str]:
    """Ids mentioned as ``finding NNNN`` in the body but not linked to that finding.

    A finding that names another finding links to it, so a reader never has to hunt
    for the one being cited (conventions/findings.md §2.7). A mention counts as
    linked when it sits inside the text of a markdown link whose target names the
    same id (``[finding 0031](0031-sex-confounded-with-group.md)``).

    Empty (no violation) unless ``content`` is a complete finding document with a
    non-empty body. The document's **own** id is exempt — a finding may name itself.
    The pattern requires the literal word ``finding``/``findings`` before the id, so a
    bare 4-digit number (a year, a sample count) never trips it.
    """
    frontmatter, body = _split_frontmatter(content)
    if frontmatter is None or not body.strip():
        return []

    own = _OWN_ID.search(frontmatter)
    own_id = f"{int(own.group(1)):04d}" if own else None

    # Ids that appear inside the text of a link pointing at that id's file.
    linked: set[str] = set()
    for text, target in _MD_LINK.findall(body):
        base = target.replace("\\", "/").rsplit("/", 1)[-1]
        for mention in _FINDING_MENTION.findall(text):
            if base.startswith(mention):
                linked.add(mention)
        # `[the batch caveat (0007)](0007-batch-skew.md)` — the link text may carry
        # the bare id without the word, so credit the target's own id too.
        if len(base) >= 4 and base[:4].isdigit() and base[:4] in text.replace(",", " "):
            linked.add(base[:4])

    missing: list[str] = []
    for mention in _FINDING_MENTION.findall(body):
        if mention == own_id or mention in linked or mention in missing:
            continue
        missing.append(mention)
    return missing


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

    missing = unembedded_figures(content)
    if missing:
        block(
            "Blocked: a finding must embed every figure it lists inline in the body, "
            "so the reader never has to track one down (conventions/findings.md §2.4). "
            "Listed in the `figures` frontmatter but not shown as an inline image in "
            "the body: " + ", ".join(missing) + ". Add ![caption](<png>) where each is "
            "discussed (with its producing script + input), or drop it from `figures`."
        )

    unlisted = unlisted_figures(content)
    if unlisted:
        block(
            "Blocked: a figure a finding shows must also be listed in the `figures` "
            "frontmatter, so it carries its own producing script + input "
            "(conventions/findings.md §2.4). Embedded inline in the body but not "
            "listed: " + ", ".join(unlisted) + ". Add a `figures` entry for each "
            "(png/svg/legend_png + caption + script/data_version/result_id), or "
            "remove the image."
        )

    unlinked = unlinked_mentions(content)
    if unlinked:
        block(
            "Blocked: every mention of another finding must be a link, so a reader "
            "never has to hunt for the finding being cited "
            "(conventions/findings.md §2.7). Mentioned but not linked: "
            + ", ".join(f"finding {m}" for m in unlinked)
            + ". Link each as [finding <NNNN>](<NNNN>-<slug>.md) — the target is the "
            "sibling filename, resolved from the manifest's ID + Slug columns."
        )


if __name__ == "__main__":
    main()
