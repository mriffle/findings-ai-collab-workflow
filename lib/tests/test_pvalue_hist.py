"""Tests for the p-value-histogram template (lib/figures/pvalue_hist.py).

Layers:
  * unit/pi0 — planted Storey π0, the clamp-to-1, and the empty -> NaN rule;
  * unit/plot — single (bar) vs overlay (step) structure, the uniform null line, the π0
    line for a single distribution, registry coloring + the >8 guard, the legend with n
    and π0, NaN drop, and the out-of-[0,1] (passed-q) refuse;
  * unit/bridge — pvalue_histogram_from_result term selection;
  * io — save writes figure + separate legend; a bad base_name closes both.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from analysis import differential_abundance as da
from common import data_loading as dl
from figures import colors as col
from figures import pvalue_hist as ph
from matplotlib.figure import Figure

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
# estimate_pi0 — planted
# --------------------------------------------------------------------------- #
def test_pi0_planted() -> None:
    """20 p-values, 10 above lam=0.5 -> pi0 = 10 / (20 * 0.5) = 1.0 (clamped)."""
    p = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
    assert ph.estimate_pi0(p, lam=0.5) == pytest.approx(1.0)


def test_pi0_half_null() -> None:
    """5 of 20 above 0.5 -> 5 / (20*0.5) = 0.5."""
    p = np.concatenate([np.full(15, 0.1), np.full(5, 0.9)])
    assert ph.estimate_pi0(p, lam=0.5) == pytest.approx(0.5)


def test_pi0_empty_is_nan() -> None:
    assert np.isnan(ph.estimate_pi0(np.array([np.nan, np.nan])))


def test_pi0_bad_lam_raises() -> None:
    with pytest.raises(ValueError, match="lam"):
        ph.estimate_pi0(np.array([0.1, 0.2]), lam=1.0)


# --------------------------------------------------------------------------- #
# Plot — structure, guards
# --------------------------------------------------------------------------- #
def _p(n: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, n)


def test_single_distribution_bar_and_uniform_line(registry: Path) -> None:
    plot = ph.plot_pvalue_histogram(_p(), registry_path=registry)
    assert isinstance(plot, ph.PValueHistogramPlot)
    ax = plot.figure.axes[0]
    # one bar container (a single filled histogram)
    assert len(ax.containers) == 1
    # uniform null line at y=1.0
    assert any(
        np.asarray(line.get_ydata())[0] == pytest.approx(1.0)
        and line.get_linestyle() == "--"
        for line in ax.lines
    )
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_single_pi0_line_drawn(registry: Path) -> None:
    p = np.concatenate([np.full(50, 0.01), _p(150, seed=1)])
    plot = ph.plot_pvalue_histogram(p, show_pi0=True, registry_path=registry)
    ax = plot.figure.axes[0]
    pi0 = plot.result.pi0["p-values"]
    assert any(
        np.asarray(line.get_ydata())[0] == pytest.approx(pi0)
        and line.get_linestyle() == ":"
        for line in ax.lines
    )
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_overlay_uses_step_and_two_colors(registry: Path) -> None:
    plot = ph.plot_pvalue_histogram(
        {"OLS": _p(seed=1), "moderated": _p(seed=2)}, registry_path=registry
    )
    assert plot.color_map == {"OLS": _PALETTE[0], "moderated": _PALETTE[1]}
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_has_n_and_pi0(registry: Path) -> None:
    plot = ph.plot_pvalue_histogram({"OLS": _p(100)}, registry_path=registry)
    assert isinstance(plot.legend_figure, Figure)
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    text = next(t.get_text() for t in legend.get_texts())
    assert "n=100" in text and "π0" in text
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_does_not_persist_colors_by_default(registry: Path) -> None:
    """Figure-local labels: the default must not write them to the registry."""
    plot = ph.plot_pvalue_histogram({"ols": _p(50), "moderated": _p(50, 1)},
                                    registry_path=registry)
    # Colors are still assigned from the palette head...
    assert plot.color_map == {"ols": _PALETTE[0], "moderated": _PALETTE[1]}
    # ...but nothing was written back.
    assert "PValueDistribution" not in json.loads(registry.read_text())
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_persist_true_writes_to_registry(registry: Path) -> None:
    plot = ph.plot_pvalue_histogram(
        {"real": _p(50)}, category="NullCheck",
        persist_colors=True, registry_path=registry,
    )
    assert "NullCheck" in json.loads(registry.read_text())
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_out_of_range_pvalues_raise(registry: Path) -> None:
    """Passing q-values (or anything outside [0,1]) must fail loud."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ph.plot_pvalue_histogram(np.array([0.1, 1.4, 0.3]), registry_path=registry)


def test_nan_pvalues_dropped(registry: Path) -> None:
    p = np.array([0.1, np.nan, 0.3, 0.9])
    plot = ph.plot_pvalue_histogram(p, registry_path=registry)
    assert plot.result.counts["p-values"] == 3
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_empty_mapping_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        ph.plot_pvalue_histogram({}, registry_path=registry)


def test_bad_n_bins_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="n_bins"):
        ph.plot_pvalue_histogram(_p(), n_bins=0, registry_path=registry)


def test_more_than_eight_distributions_raises(registry: Path) -> None:
    data = {f"d{i}": _p(50, seed=i) for i in range(9)}
    with pytest.raises(col.CategoricalPaletteExceededError):
        ph.plot_pvalue_histogram(data, registry_path=registry)


def test_error_path_closes_figure(registry: Path) -> None:
    data = {f"d{i}": _p(50, seed=i) for i in range(9)}
    before = set(plt.get_fignums())
    with pytest.raises(col.CategoricalPaletteExceededError):
        ph.plot_pvalue_histogram(data, registry_path=registry)
    assert set(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# Bridge — from_result
# --------------------------------------------------------------------------- #
def _result(levels: list[str]) -> da.DifferentialAbundanceResult:
    rng = np.random.default_rng(0)
    n = len(levels)
    ab = rng.standard_normal((n, 60)) + 10.0
    meta = pd.DataFrame({"grp": levels}, index=[f"s{i}" for i in range(n)])
    ds = dl.Dataset(
        abundances=ab,
        feature_names=np.array([f"F{j}" for j in range(60)], dtype=str),
        feature_metadata=pd.DataFrame({"feature": [f"F{j}" for j in range(60)]}),
        metadata=meta,
        scale="log2",
    )
    return da.differential_abundance(ds, "grp", method="ols")


def test_from_result_all_terms_by_default(registry: Path) -> None:
    res = _result(["A", "A", "B", "B", "C", "C"])
    plot = ph.pvalue_histogram_from_result(res, registry_path=registry)
    assert set(plot.result.counts.keys()) == {"grp[B vs A]", "grp[C vs A]"}
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_from_result_subset_terms(registry: Path) -> None:
    res = _result(["A", "A", "B", "B", "C", "C"])
    plot = ph.pvalue_histogram_from_result(
        res, terms=["grp[B vs A]"], registry_path=registry
    )
    assert set(plot.result.counts.keys()) == {"grp[B vs A]"}
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_from_result_unknown_term_raises(registry: Path) -> None:
    res = _result(["A", "A", "B", "B", "C", "C"])
    with pytest.raises(ValueError, match="not contrast terms"):
        ph.pvalue_histogram_from_result(
            res, terms=["grp[Z vs A]"], registry_path=registry
        )


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def test_save_writes_figure_and_legend(registry: Path, tmp_path: Path) -> None:
    arts = ph.save_pvalue_histogram(_p(), tmp_path, "pvals", registry_path=registry)
    assert arts.svg.exists() and arts.png.exists()
    assert arts.legend_svg is not None and arts.legend_svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()


def test_save_bad_base_name_closes_figures(registry: Path, tmp_path: Path) -> None:
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="bare filename stem"):
        ph.save_pvalue_histogram(_p(), tmp_path, "bad/name", registry_path=registry)
    assert set(plt.get_fignums()) == before
