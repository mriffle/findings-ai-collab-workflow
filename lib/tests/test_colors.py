"""Tests for the color-registry template (lib/figures/colors.py).

Layers:
  * unit — an isolated tmp registry seeded with the canonical ``_palette`` block, with
    planted-truth checks (first value -> palette[0], persistence, category independence)
    and the load-bearing guards (the >8-category raise; gray backgrounds that consume no
    slot; fail-loud on a missing/corrupt registry);
  * seed — the shipped templates/color_registry.json parses and its universal ``sex``
    colors resolve (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from figures import colors as col

# The canonical 8-color Okabe-Ito palette, in the shipped registry's order.
_PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """An isolated registry seeded with the canonical palette (no categories yet)."""
    path = tmp_path / "color_registry.json"
    path.write_text(
        json.dumps(
            {
                "_palette": {
                    "name": "Okabe-Ito",
                    "colors": _PALETTE,
                    "max_categorical": 8,
                }
            }
        )
    )
    return path


# --------------------------------------------------------------------------- #
# Unit — assignment + persistence (planted truth)
# --------------------------------------------------------------------------- #


def test_first_value_gets_first_palette_color(registry: Path) -> None:
    assert col.get_color("Genotype", "5xFAD", registry_path=registry) == _PALETTE[0]


def test_distinct_values_walk_the_palette(registry: Path) -> None:
    out = col.assign_colors("Genotype", ["5xFAD", "WT", "C57"], registry_path=registry)
    assert out == {"5xFAD": _PALETTE[0], "WT": _PALETTE[1], "C57": _PALETTE[2]}


def test_repeat_value_returns_same_color(registry: Path) -> None:
    first = col.get_color("Genotype", "5xFAD", registry_path=registry)
    second = col.get_color("Genotype", "5xFAD", registry_path=registry)
    assert first == second


def test_input_order_is_preserved_in_result(registry: Path) -> None:
    out = col.assign_colors("G", ["a", "b", "a", "c"], registry_path=registry)
    assert out["a"] == _PALETTE[0]
    assert out["b"] == _PALETTE[1]
    assert out["c"] == _PALETTE[2]


def test_assignment_persists_across_calls(registry: Path) -> None:
    col.assign_colors("Genotype", ["A", "B"], registry_path=registry)
    # A new call in a different order must not reassign.
    again = col.assign_colors("Genotype", ["B", "A", "C"], registry_path=registry)
    assert again["A"] == _PALETTE[0]
    assert again["B"] == _PALETTE[1]
    assert again["C"] == _PALETTE[2]


def test_categories_are_independent(registry: Path) -> None:
    geno = col.get_color("Genotype", "x", registry_path=registry)
    treat = col.get_color("Treatment", "y", registry_path=registry)
    # Each namespace starts from palette[0].
    assert geno == _PALETTE[0]
    assert treat == _PALETTE[0]
    assert col.get_color("Genotype", "z", registry_path=registry) == _PALETTE[1]


def test_int_and_str_values_share_a_key(registry: Path) -> None:
    c1 = col.get_color("Cohort", 1, registry_path=registry)
    c2 = col.get_color("Cohort", "1", registry_path=registry)
    assert c1 == c2


def test_written_registry_has_values_and_scope(registry: Path) -> None:
    col.assign_colors(
        "Genotype", ["5xFAD", "WT"], registry_path=registry, scope="project"
    )
    stored = json.loads(registry.read_text())
    assert stored["Genotype"]["values"] == {"5xFAD": _PALETTE[0], "WT": _PALETTE[1]}
    assert stored["Genotype"]["scope"] == "project"
    # The palette block is preserved untouched.
    assert stored["_palette"]["colors"] == _PALETTE


def test_loads_existing_assignment(registry: Path) -> None:
    stored = json.loads(registry.read_text())
    stored["Genotype"] = {"scope": "project", "values": {"5xFAD": _PALETTE[5]}}
    registry.write_text(json.dumps(stored))
    assert col.get_color("Genotype", "5xFAD", registry_path=registry) == _PALETTE[5]


def test_persist_false_does_not_write(registry: Path) -> None:
    col.assign_colors("Genotype", ["A"], registry_path=registry, persist=False)
    assert "Genotype" not in json.loads(registry.read_text())


# --------------------------------------------------------------------------- #
# Unit — background labels (gray, no slot, not persisted)
# --------------------------------------------------------------------------- #


def test_background_value_is_gray_and_not_persisted(registry: Path) -> None:
    out = col.assign_colors(
        "ST", ["ref", "a", "b"], background_values=["ref"], registry_path=registry
    )
    assert out["ref"] == col.BACKGROUND_COLOR
    # Foreground still starts at palette[0] — the background does not consume a slot.
    assert out["a"] == _PALETTE[0]
    assert out["b"] == _PALETTE[1]
    # And the background label is not written to the registry.
    assert "ref" not in json.loads(registry.read_text())["ST"]["values"]


def test_background_only_value_still_mapped(registry: Path) -> None:
    out = col.assign_colors("ST", [], background_values=["ref"], registry_path=registry)
    assert out == {"ref": col.BACKGROUND_COLOR}


# --------------------------------------------------------------------------- #
# Unit — the >8-category guard and fail-loud edges
# --------------------------------------------------------------------------- #


def test_more_than_eight_categories_raises(registry: Path) -> None:
    values = [f"v{i}" for i in range(9)]  # 9 > max_categorical (8)
    with pytest.raises(
        col.CategoricalPaletteExceededError, match="change the encoding"
    ):
        col.assign_colors("Many", values, registry_path=registry)


def test_exactly_eight_categories_is_allowed(registry: Path) -> None:
    values = [f"v{i}" for i in range(8)]
    out = col.assign_colors("Eight", values, registry_path=registry)
    assert len(set(out.values())) == 8


def test_overflow_does_not_persist_partial(registry: Path) -> None:
    values = [f"v{i}" for i in range(9)]
    with pytest.raises(col.CategoricalPaletteExceededError):
        col.assign_colors("Many", values, registry_path=registry)
    assert "Many" not in json.loads(registry.read_text())


def test_missing_palette_block_raises(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"sex": {"scope": "universal", "values": {}}}))
    with pytest.raises(ValueError, match="no '_palette' block"):
        col.assign_colors("X", ["a"], registry_path=path)


def test_corrupt_registry_raises(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        col.load_registry(path)


def test_load_palette_rejects_single_color(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"_palette": {"colors": ["#000000"]}}))
    with pytest.raises(ValueError, match=">= 2 colors"):
        col.load_palette(path)


def test_load_palette_rejects_max_categorical_below_one(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    path.write_text(
        json.dumps({"_palette": {"colors": _PALETTE, "max_categorical": 0}})
    )
    with pytest.raises(ValueError, match="must be >= 1"):
        col.load_palette(path)


def test_load_palette_clamps_to_distinct_colors(tmp_path: Path) -> None:
    """A duplicate-containing palette can only color as many categories as it has
    distinct colors, so max_categorical is clamped to that."""
    path = tmp_path / "reg.json"
    dup = ["#E69F00", "#E69F00", "#56B4E9"]  # 3 entries, 2 distinct
    path.write_text(json.dumps({"_palette": {"colors": dup, "max_categorical": 3}}))
    assert col.load_palette(path).max_categorical == 2


def test_reserved_underscore_category_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="reserved registry metadata"):
        col.assign_colors("_palette", ["x"], registry_path=registry)


def test_empty_category_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        col.assign_colors("", ["x"], registry_path=registry)


# --------------------------------------------------------------------------- #
# Seed — the shipped registry template (read-only)
# --------------------------------------------------------------------------- #

_SHIPPED = Path(__file__).resolve().parents[2] / "templates" / "color_registry.json"


def test_shipped_registry_palette_and_sex() -> None:
    """The plugin's seed registry parses and its universal sex colors resolve."""
    palette = col.load_palette(_SHIPPED)
    assert palette.max_categorical == 8
    assert len(palette.colors) == 8
    out = col.assign_colors(
        "sex", ["male", "female"], registry_path=_SHIPPED, persist=False
    )
    assert out == {"male": "#0072B2", "female": "#D55E00"}
