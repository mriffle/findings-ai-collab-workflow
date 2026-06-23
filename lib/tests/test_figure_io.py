"""Tests for the figure-IO template (lib/figures/figure_io.py).

Verify *our* wiring, not matplotlib: the main figure is dual-exported (SVG + PNG), an
optional companion legend figure lands at ``<base>.legend.{svg,png}``, bad inputs fail
loud, and the publication style is applied within the context and restored on exit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from figures import figure_io
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


@pytest.fixture
def fig() -> Figure:
    """A trivial main figure to save."""
    figure, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return figure


@pytest.fixture
def legend_fig() -> Figure:
    """A trivial companion legend figure."""
    figure, ax = plt.subplots()
    ax.axis("off")
    ax.legend([Line2D([0], [0], marker="o", linestyle="")], ["A"], loc="center")
    return figure


# --------------------------------------------------------------------------- #
# save_figure — main figure + optional legend figure
# --------------------------------------------------------------------------- #


def test_writes_svg_and_png(fig: Figure, tmp_path: Path) -> None:
    arts = figure_io.save_figure(fig, tmp_path, "myplot")
    assert arts.svg == tmp_path / "myplot.svg"
    assert arts.png == tmp_path / "myplot.png"
    assert arts.svg.exists() and arts.png.exists()


def test_no_legend_fig_means_no_legend_files(fig: Figure, tmp_path: Path) -> None:
    arts = figure_io.save_figure(fig, tmp_path, "p")
    assert arts.legend_svg is None
    assert arts.legend_png is None
    assert not (tmp_path / "p.legend.svg").exists()


def test_legend_fig_dual_exported(
    fig: Figure, legend_fig: Figure, tmp_path: Path
) -> None:
    arts = figure_io.save_figure(fig, tmp_path, "p", legend_fig=legend_fig)
    assert arts.legend_svg == tmp_path / "p.legend.svg"
    assert arts.legend_png == tmp_path / "p.legend.png"
    assert arts.legend_svg.exists() and arts.legend_png.exists()
    # The main figure is still its own pair (legend kept out of it).
    assert (tmp_path / "p.svg").exists() and (tmp_path / "p.png").exists()


def test_output_dir_created(fig: Figure, tmp_path: Path) -> None:
    nested = tmp_path / "figures" / "pca"
    arts = figure_io.save_figure(fig, nested, "p")
    assert arts.png.exists()


def test_closes_both_figures_by_default(
    fig: Figure, legend_fig: Figure, tmp_path: Path
) -> None:
    main_num, leg_num = fig.number, legend_fig.number
    figure_io.save_figure(fig, tmp_path, "p", legend_fig=legend_fig)
    assert not plt.fignum_exists(main_num)
    assert not plt.fignum_exists(leg_num)


def test_close_false_keeps_figures(
    fig: Figure, legend_fig: Figure, tmp_path: Path
) -> None:
    main_num, leg_num = fig.number, legend_fig.number
    figure_io.save_figure(fig, tmp_path, "p", legend_fig=legend_fig, close=False)
    assert plt.fignum_exists(main_num)
    assert plt.fignum_exists(leg_num)
    plt.close(fig)
    plt.close(legend_fig)


# --------------------------------------------------------------------------- #
# save_figure — fail loud
# --------------------------------------------------------------------------- #


def test_base_name_with_path_separator_raises_and_closes(
    fig: Figure, tmp_path: Path
) -> None:
    num = fig.number
    with pytest.raises(ValueError, match="bare filename stem"):
        figure_io.save_figure(fig, tmp_path, "sub/p")
    assert not plt.fignum_exists(num)  # close=True closes even on a bad-arg raise


def test_backslash_base_name_rejected_cross_platform(
    fig: Figure, tmp_path: Path
) -> None:
    # A backslash must be rejected on every platform (these seeds may run on Windows).
    num = fig.number
    with pytest.raises(ValueError, match="bare filename stem"):
        figure_io.save_figure(fig, tmp_path, "a\\b")
    assert not plt.fignum_exists(num)


def test_dotdot_base_name_rejected(fig: Figure, tmp_path: Path) -> None:
    num = fig.number
    with pytest.raises(ValueError, match="bare filename stem"):
        figure_io.save_figure(fig, tmp_path, "..")
    assert not plt.fignum_exists(num)


def test_nonpositive_dpi_raises_and_closes(fig: Figure, tmp_path: Path) -> None:
    num = fig.number
    with pytest.raises(ValueError, match="dpi must be positive"):
        figure_io.save_figure(fig, tmp_path, "p", dpi=0)
    assert not plt.fignum_exists(num)


def test_savefig_error_still_closes_figures(
    fig: Figure, legend_fig: Figure, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If savefig raises (e.g. disk full), close=True must still free the figures."""
    main_num, leg_num = fig.number, legend_fig.number

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(fig, "savefig", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        figure_io.save_figure(fig, tmp_path, "p", legend_fig=legend_fig)
    assert not plt.fignum_exists(main_num)
    assert not plt.fignum_exists(leg_num)


def test_close_false_keeps_figure_even_on_error(fig: Figure, tmp_path: Path) -> None:
    num = fig.number
    with pytest.raises(ValueError):
        figure_io.save_figure(fig, tmp_path, "bad/name", close=False)
    assert plt.fignum_exists(num)  # caller owns it
    plt.close(fig)


# --------------------------------------------------------------------------- #
# publication_style — scoped rcParam override
# --------------------------------------------------------------------------- #


def test_publication_style_applies_and_restores() -> None:
    matplotlib.rcParams["axes.spines.top"] = True
    before = matplotlib.rcParams["axes.spines.top"]
    with figure_io.publication_style():
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["savefig.dpi"] == figure_io.DEFAULT_DPI
    assert matplotlib.rcParams["axes.spines.top"] == before


def test_publication_style_restores_on_exception() -> None:
    matplotlib.rcParams["font.size"] = 11
    with pytest.raises(RuntimeError), figure_io.publication_style():
        raise RuntimeError("boom")
    assert matplotlib.rcParams["font.size"] == 11
