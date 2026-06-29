"""Tests for the volcano template (lib/figures/volcano.py).

Layers:
  * unit/core — planted up/down/NS counts, the q-underflow floor, NaN-q drop, the
    fold-change gate, registry coloring (up/down palette, NS gray background), the
    separate legend with hit counts, top-hit annotation, and the fail-loud guards;
  * unit/bridge — volcano_from_result term selection (default sole term, explicit
    choice, the multi-term ambiguity + unknown-term errors);
  * io — save writes the figure + separate legend image; a bad base_name closes both.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from analysis import differential_abundance as da
from common import data_loading as dl
from figures import colors as col
from figures import volcano as vol
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

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
# Core — counts, coloring, guards
# --------------------------------------------------------------------------- #
def test_planted_up_down_ns_counts(registry: Path) -> None:
    effect = np.array([2.0, -2.0, 0.1, 3.0])
    q = np.array([0.001, 0.001, 0.9, 0.2])
    plot = vol.plot_volcano(effect, q, registry_path=registry)
    assert plot.counts == vol.VolcanoCounts(up=1, down=1, ns=2)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_q_underflow_floored_to_finite() -> None:
    nlq = vol._safe_neg_log10(np.array([0.0, 1e-300, np.nan]))
    assert np.isfinite(nlq[0]) and nlq[0] > 300
    assert np.isfinite(nlq[1])
    assert np.isnan(nlq[2])


def test_nan_q_excluded_from_counts(registry: Path) -> None:
    effect = np.array([2.0, -2.0, 1.0])
    q = np.array([0.001, 0.001, np.nan])
    plot = vol.plot_volcano(effect, q, registry_path=registry)
    assert plot.counts == vol.VolcanoCounts(up=1, down=1, ns=0)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_effect_threshold_gates_hits(registry: Path) -> None:
    effect = np.array([0.1, 2.0])
    q = np.array([0.001, 0.001])
    plot = vol.plot_volcano(effect, q, effect_threshold=1.0, registry_path=registry)
    assert plot.counts == vol.VolcanoCounts(up=1, down=0, ns=1)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_colors_up_down_palette_ns_background(registry: Path) -> None:
    plot = vol.plot_volcano(
        np.array([2.0, -2.0]), np.array([0.001, 0.001]), registry_path=registry
    )
    assert plot.color_map["up"] == _PALETTE[0]
    assert plot.color_map["down"] == _PALETTE[1]
    assert plot.color_map["NS"] == col.BACKGROUND_COLOR
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_separate_with_counts(registry: Path) -> None:
    plot = vol.plot_volcano(
        np.array([2.0, -2.0, 0.1]),
        np.array([0.001, 0.001, 0.9]),
        direction_labels=("higher in X", "higher in Y"),
        registry_path=registry,
    )
    assert isinstance(plot.legend_figure, Figure)
    assert plot.legend_figure is not plot.figure
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    texts = {t.get_text() for t in legend.get_texts()}
    assert "higher in X (n=1)" in texts
    assert "higher in Y (n=1)" in texts
    assert "NS (n=1)" in texts
    # The main axes carries no baked legend.
    assert plot.figure.axes[0].get_legend() is None
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_swatch_color_matches_registry(registry: Path) -> None:
    plot = vol.plot_volcano(
        np.array([2.0, -2.0]), np.array([0.001, 0.001]), registry_path=registry
    )
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    handle = legend.legend_handles[0]  # "up" entry, first
    assert isinstance(handle, Line2D)
    assert mcolors.to_hex(handle.get_markerfacecolor()).lower() == _PALETTE[0].lower()
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_annotate_top_labels_present(registry: Path) -> None:
    effect = np.array([3.0, 2.0, 0.1])
    q = np.array([1e-9, 1e-3, 0.9])
    names = np.array(["BIG", "MID", "NS"])
    plot = vol.plot_volcano(
        effect, q, labels=names, annotate_top=1, registry_path=registry
    )
    texts = [t.get_text() for t in plot.figure.axes[0].texts]
    assert "BIG" in texts
    assert "MID" not in texts  # only top-1 annotated
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_shape_mismatch_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="same shape"):
        vol.plot_volcano(np.array([1.0, 2.0]), np.array([0.1]), registry_path=registry)


def test_bad_fdr_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="fdr"):
        vol.plot_volcano(
            np.array([1.0]), np.array([0.1]), fdr=1.5, registry_path=registry
        )


def test_annotate_top_without_labels_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="requires labels"):
        vol.plot_volcano(
            np.array([1.0]), np.array([0.1]), annotate_top=1, registry_path=registry
        )


def test_error_path_closes_figure(tmp_path: Path) -> None:
    """A registry with no _palette raises inside the try; the figure must not leak."""
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="_palette"):
        vol.plot_volcano(np.array([1.0]), np.array([0.1]), registry_path=bad)
    assert set(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# Bridge — volcano_from_result
# --------------------------------------------------------------------------- #
def _result(levels: list[str]) -> da.DifferentialAbundanceResult:
    rng = np.random.default_rng(0)
    n = len(levels)
    ab = rng.standard_normal((n, 40)) + 10.0
    meta = pd.DataFrame({"grp": levels}, index=[f"s{i}" for i in range(n)])
    ds = dl.Dataset(
        abundances=ab,
        feature_names=np.array([f"F{j}" for j in range(40)], dtype=str),
        feature_metadata=pd.DataFrame({"feature": [f"F{j}" for j in range(40)]}),
        metadata=meta,
        scale="log2",
    )
    return da.differential_abundance(ds, "grp", method="ols")


def test_from_result_default_sole_term(registry: Path) -> None:
    res = _result(["A", "A", "A", "B", "B", "B"])
    plot = vol.volcano_from_result(res, registry_path=registry)
    assert plot.counts.up + plot.counts.down + plot.counts.ns == 40
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_from_result_multiterm_requires_choice(registry: Path) -> None:
    res = _result(["A", "A", "B", "B", "C", "C"])
    with pytest.raises(ValueError, match="contrast terms"):
        vol.volcano_from_result(res, registry_path=registry)
    plot = vol.volcano_from_result(res, term="grp[C vs A]", registry_path=registry)
    assert plot.counts.up + plot.counts.down + plot.counts.ns == 40
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_from_result_unknown_term_raises(registry: Path) -> None:
    res = _result(["A", "A", "A", "B", "B", "B"])
    with pytest.raises(ValueError, match="not a contrast term"):
        vol.volcano_from_result(res, term="grp[Z vs A]", registry_path=registry)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def test_save_writes_figure_and_legend(registry: Path, tmp_path: Path) -> None:
    arts = vol.save_volcano(
        np.array([2.0, -2.0]),
        np.array([0.001, 0.001]),
        tmp_path,
        "volcano_grp",
        registry_path=registry,
    )
    assert arts.svg.exists() and arts.png.exists()
    assert arts.legend_svg is not None and arts.legend_svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()


def test_save_bad_base_name_closes_figures(registry: Path, tmp_path: Path) -> None:
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="bare filename stem"):
        vol.save_volcano(
            np.array([2.0]),
            np.array([0.001]),
            tmp_path,
            "bad/name",
            registry_path=registry,
        )
    assert set(plt.get_fignums()) == before
