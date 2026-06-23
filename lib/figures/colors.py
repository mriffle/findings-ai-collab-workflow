"""Project color registry: consistent Okabe-Ito colors for categorical figures.

TEMPLATE (lib/) — a *seed* for a project's ``common`` color module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

Why this exists (conventions/visualization.md):
  * **One palette — Okabe-Ito** (colorblind-safe). The eight palette colors live in the
    registry's ``_palette.colors``; this module never hardcodes them, so changing the
    palette is a registry edit, not a code edit.
  * **A category value gets the same color in every figure.** ``Genotype=5xFAD`` must
    look the same in every plot it appears in. The mapping is persisted to
    ``state/color_registry.json`` so consistency is *mechanical*, not remembered —
    every plotting script reads (and extends) the one file.
  * **Color encodes at most eight categories.** Beyond eight, more colors is the wrong
    move; the encoding strategy must change (facet, a second channel, group the long
    tail, or a position/sequential scale). :func:`assign_colors` therefore **raises**
    rather than recycle or invent a ninth color — the deterministic enforcer of the
    >8-category rule that the figure-reviewer otherwise has to catch by eye.

Registry schema (``state/color_registry.json``; see ``templates/color_registry.json``)::

    {
      "_palette": {"name": "Okabe-Ito", "colors": [<=8 hex>], "max_categorical": 8},
      "sex": {"scope": "universal", "values": {"male": "#0072B2", "female": "#D55E00"}},
      <category>: {"scope": "universal"|"project", "values": {<value>: <hex>}}
    }

Keys starting with ``_`` are reserved metadata; every other top-level key is a category
dimension. New ``(category, value)`` pairs are assigned the next unused palette color
for that category and **written back** so the next script sees the same assignment.

This module makes NO study-specific decisions: which metadata column maps to which
category namespace is the caller's choice. Background ("reference") values are colored a
neutral gray *outside* the palette — they do not consume a palette slot and do not count
toward the eight-color budget.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

__script_meta__: dict[str, object] = {
    "template": {"name": "okabe-ito-colors", "version": "0.2"},
    "kind": "module",
    "provides": [
        "BACKGROUND_COLOR",
        "Palette",
        "CategoricalPaletteExceededError",
        "DEFAULT_REGISTRY_PATH",
        "load_registry",
        "load_palette",
        "assign_colors",
        "get_color",
    ],
    "uses": [],
    "seeded_from": None,
    "description": (
        "Project color registry: reads/extends state/color_registry.json so a given "
        "(category, value) gets the same Okabe-Ito color in every figure. "
        "Deterministic next-unused-color assignment, persisted; gray background "
        "labels outside the palette; raises CategoricalPaletteExceededError past 8 "
        "categories (the >8-category guard). Study-agnostic; fail-loud."
    ),
}

# Project location of the registry (relative to the project root the script runs from).
# Tests and ad-hoc renders pass an explicit path.
DEFAULT_REGISTRY_PATH: Path = Path("state") / "color_registry.json"

# Neutral light gray for "background" / reference labels. Deliberately *outside* the
# Okabe-Ito palette so it never collides with an assigned categorical color and never
# consumes one of the eight slots.
BACKGROUND_COLOR = "#d0d0d0"

# Process-local lock making concurrent assign_colors() calls within one Python process
# safe. There is NO cross-process lock: running plot scripts in parallel against the
# same registry file is not supported (the last writer wins).
_lock = Lock()


class CategoricalPaletteExceededError(ValueError):
    """Raised when a category needs more colors than the palette allows (>8).

    The fix is never "add a color" — it is to change the encoding strategy
    (conventions/visualization.md): facet into small multiples, add a second channel
    (shape/linetype), group the long tail into an explicit "other", or use a
    position/sequential encoding for ordinal data.
    """


@dataclass(frozen=True)
class Palette:
    """The categorical palette read from the registry's ``_palette`` block.

    Attributes
    ----------
    name:
        Palette name (e.g. ``"Okabe-Ito"``), for provenance/legends.
    colors:
        The ordered hex colors. Assignment walks this list; ``max_categorical`` caps how
        many of them may be used per category.
    max_categorical:
        The hard ceiling on categorical colors (the >8 rule's threshold; normally 8).
    """

    name: str
    colors: tuple[str, ...]
    max_categorical: int


# --------------------------------------------------------------------------- #
# Registry IO (fail loud at every boundary)
# --------------------------------------------------------------------------- #


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Read the color registry JSON; return ``{}`` if the file does not exist.

    Raises a clear error if the file exists but is not a JSON object — a corrupt
    registry must fail loud, not silently reset color assignments.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        loaded: Any = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Color registry {path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Color registry {path} must be a JSON object, got {type(loaded).__name__}."
        )
    return loaded


def load_palette(path: str | Path = DEFAULT_REGISTRY_PATH) -> Palette:
    """Read and validate the registry ``_palette`` block; return a :class:`Palette`.

    Raises (fail loud) if ``_palette`` is missing or malformed, if any color is not a
    string, or if there are fewer than two colors. The palette is load-bearing for every
    figure, so a missing/broken one is an error, not a silent default.
    """
    registry = load_registry(path)
    palette = registry.get("_palette")
    if not isinstance(palette, dict):
        raise ValueError(
            f"Color registry {Path(path)} has no '_palette' block; cannot assign "
            f"colors. Seed it from templates/color_registry.json."
        )
    raw_colors = palette.get("colors")
    if not isinstance(raw_colors, list) or not all(
        isinstance(c, str) for c in raw_colors
    ):
        raise ValueError(
            f"'_palette.colors' in {Path(path)} must be a list of hex strings; "
            f"got {raw_colors!r}."
        )
    colors = tuple(str(c) for c in raw_colors)
    if len(colors) < 2:
        raise ValueError(
            f"'_palette.colors' in {Path(path)} needs >= 2 colors; got {len(colors)}."
        )
    max_categorical_raw = palette.get("max_categorical", len(colors))
    if not isinstance(max_categorical_raw, int) or isinstance(
        max_categorical_raw, bool
    ):
        raise ValueError(
            f"'_palette.max_categorical' in {Path(path)} must be an int; "
            f"got {max_categorical_raw!r}."
        )
    if max_categorical_raw < 1:
        raise ValueError(
            f"'_palette.max_categorical' in {Path(path)} must be >= 1; "
            f"got {max_categorical_raw}."
        )
    # Clamp to the number of *distinct* colors — the true assignable capacity. A palette
    # with duplicate entries cannot color more categories than it has unique colors.
    max_categorical = min(int(max_categorical_raw), len(set(colors)))
    name = palette.get("name")
    return Palette(
        name=str(name) if isinstance(name, str) else "categorical",
        colors=colors,
        max_categorical=max_categorical,
    )


def _category_values(registry: dict[str, Any], category: str) -> dict[str, str]:
    """Return a validated ``{value: hex}`` map for ``category`` (empty if absent)."""
    entry = registry.get(category)
    if entry is None:
        return {}
    if not isinstance(entry, dict):
        raise ValueError(
            f"Registry entry for category {category!r} must be an object with a "
            f"'values' map; got {type(entry).__name__}."
        )
    values = entry.get("values", {})
    if not isinstance(values, dict):
        raise ValueError(
            f"Registry entry {category!r} has a non-object 'values'; got "
            f"{type(values).__name__}."
        )
    cleaned: dict[str, str] = {}
    for key, color in values.items():
        if not isinstance(color, str):
            raise ValueError(
                f"Registry color for {category!r}/{key!r} is not a string: {color!r}."
            )
        cleaned[str(key)] = color
    return cleaned


def _save_registry(path: Path, registry: dict[str, Any]) -> None:
    """Persist the registry atomically so a torn write can't corrupt the source file.

    Writes to a temp file in the same directory, then ``os.replace`` swaps it in (atomic
    on a single filesystem). A crash mid-write leaves the previous registry intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(registry, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _next_color(palette: Palette, used: set[str]) -> str:
    """First palette color not already used in this category (palette order)."""
    for color in palette.colors:
        if color not in used:
            return color
    # Unreachable while the caller enforces the max_categorical budget first, but kept
    # as a hard backstop so palette exhaustion can never silently wrap around.
    raise CategoricalPaletteExceededError(
        f"Palette {palette.name!r} has only {len(palette.colors)} colors; all are used."
    )


# --------------------------------------------------------------------------- #
# Public assignment
# --------------------------------------------------------------------------- #


def assign_colors(
    category: str,
    values: object,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    background_values: object = (),
    scope: str = "project",
    persist: bool = True,
) -> dict[str, str]:
    """Map each distinct value to a hex color, consistent across figures and runs.

    Foreground (non-background) values reuse their registry color if present, else are
    assigned the next unused palette color for ``category`` and (when ``persist``)
    written back to ``registry_path``. Background values are mapped to
    :data:`BACKGROUND_COLOR` — a neutral gray *outside* the palette, using no slot.

    Parameters
    ----------
    category:
        Registry namespace (e.g. ``"Genotype"``). Two equal values in different
        categories get colors independently.
    values:
        Iterable of categorical values (coerced to ``str``; ``1`` and ``"1"`` are one
        value). Order of *first appearance* determines new-color assignment order.
    registry_path:
        The registry JSON. Defaults to ``state/color_registry.json``; tests pass a tmp
        file. Must contain a ``_palette`` block (see :func:`load_palette`).
    background_values:
        Values to color :data:`BACKGROUND_COLOR` instead of a palette color. They are
        NOT persisted and do NOT count toward the eight-color budget.
    scope:
        ``"scope"`` recorded when this call first creates ``category`` in the registry
        (``"project"`` for study-specific dimensions discovered at plot time;
        ``"universal"`` ships pre-seeded). Ignored if the category already exists.
    persist:
        Write newly assigned colors back to the registry (default ``True``). Pass
        ``False`` for a throwaway render that must not mutate the shared file.

    Returns
    -------
    dict[str, str]
        ``{str(value): hex}`` for every distinct input value (foreground + background).

    Raises
    ------
    CategoricalPaletteExceededError
        If the category's foreground values exceed ``_palette.max_categorical`` (the
        >8-category rule). Nothing is persisted in that case.
    """
    if not category or category.startswith("_"):
        raise ValueError(
            f"category {category!r} is invalid: it must be non-empty and not start "
            f"with '_' ('_'-prefixed keys like '_palette' are reserved registry "
            f"metadata, never category dimensions)."
        )
    background = {str(v) for v in _as_iterable(background_values)}
    # Distinct foreground values in first-appearance order.
    foreground: list[str] = []
    seen: set[str] = set()
    result: dict[str, str] = {}
    for raw in _as_iterable(values):
        value = str(raw)
        if value in background:
            result[value] = BACKGROUND_COLOR
            continue
        if value not in seen:
            seen.add(value)
            foreground.append(value)
    # Background values that never appeared in ``values`` are still mapped, so the
    # legend can describe them.
    for value in background:
        result.setdefault(value, BACKGROUND_COLOR)

    with _lock:
        registry = load_registry(registry_path)
        palette = load_palette(registry_path)
        existing = _category_values(registry, category)

        # Distinct foreground colors this category would hold after the call.
        distinct_after = set(existing).union(foreground)
        if len(distinct_after) > palette.max_categorical:
            raise CategoricalPaletteExceededError(
                f"Category {category!r} would need {len(distinct_after)} categorical "
                f"colors but the palette allows at most {palette.max_categorical}. "
                f"Color encodes <= {palette.max_categorical} categories — change the "
                f"encoding strategy instead of adding colors: facet into small "
                f"multiples, add a second channel (shape/linetype), group the long "
                f"tail into an explicit 'other', or use a position/sequential scale "
                f"for ordinal data (conventions/visualization.md)."
            )

        used = set(existing.values())
        new_assignments: dict[str, str] = {}
        for value in foreground:
            if value in existing:
                result[value] = existing[value]
                continue
            color = _next_color(palette, used)
            used.add(color)
            new_assignments[value] = color
            result[value] = color

        if persist and new_assignments:
            entry = registry.get(category)
            if not isinstance(entry, dict):
                entry = {"scope": scope, "values": {}}
            entry_values = entry.get("values")
            if not isinstance(entry_values, dict):
                entry_values = {}
            entry_values.update(new_assignments)
            entry["values"] = entry_values
            entry.setdefault("scope", scope)
            registry[category] = entry
            _save_registry(Path(registry_path), registry)

    return result


def get_color(
    category: str,
    value: object,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist: bool = True,
) -> str:
    """Return the hex color for a single ``(category, value)`` (assigning if needed)."""
    return assign_colors(
        category, [value], registry_path=registry_path, persist=persist
    )[str(value)]


def _as_iterable(values: object) -> list[object]:
    """Coerce ``values`` to a concrete list, treating a bare string as one value.

    A lone ``str`` is iterable character-by-character; treating it as a sequence of
    values would silently explode ``"WT"`` into ``"W"``/``"T"``. So a ``str`` (or
    non-iterable) becomes a single-element list.
    """
    if isinstance(values, str):
        return [values]
    if isinstance(values, Iterable):
        return list(values)
    return [values]
