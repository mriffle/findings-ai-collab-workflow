"""Tests for the PCA-plot template (lib/figures/pca.py).

Layers:
  * unit/compute — planted-truth on the PCA itself (a rank-1 dataset must put almost all
    variance on PC1) plus shape/flag invariants and the finite/feasibility guards;
  * unit/plot — the figure structure (the 8-axis layout), registry coloring, the
    separate legend figure (swatches/colorbar), background greying, and the guards
    (missing/NaN color column, the >8-category overflow, the non-log scale warning);
  * smoke — real 5xFAD proteins (git-ignored) reproducing the captured
    explained-variance oracle; skips cleanly when the data is absent.
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
from common import data_loading as dl
from figures import colors as col
from figures import pca
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

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
    """Isolated registry seeded with the canonical palette."""
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


def _dataset(
    abundances: np.ndarray,
    metadata: dict[str, object],
    scale: dl.Scale = "log2",
) -> dl.Dataset:
    """Build a Dataset around an abundance matrix and a metadata-column dict."""
    n_samples, n_features = abundances.shape
    feature_names = np.array([f"F{j}" for j in range(n_features)], dtype=str)
    meta = pd.DataFrame(metadata, index=[f"s{i}" for i in range(n_samples)])
    return dl.Dataset(
        abundances=np.asarray(abundances, dtype=float),
        feature_names=feature_names,
        feature_metadata=pd.DataFrame({"protein": feature_names}),
        metadata=meta,
        scale=scale,
    )


def _grouped(n_per: int = 10, n_features: int = 20) -> dl.Dataset:
    """A 3-group dataset (A/B/C) on the log2 scale for plot tests."""
    rng = np.random.default_rng(42)
    n = 3 * n_per
    ab = rng.standard_normal((n, n_features)) + 5.0
    labels = np.array(["A"] * n_per + ["B"] * n_per + ["C"] * n_per)
    runorder = np.arange(n, dtype=float)
    return _dataset(ab, {"Group": labels, "RunOrder": runorder})


def _first_facecolor_hex(figure: Figure) -> str:
    """Lowercase hex of the first scatter collection's facecolor on the first panel."""
    rgba = np.asarray(figure.get_axes()[0].collections[0].get_facecolor())[0]
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    return mcolors.to_hex((r, g, b)).lower()


# --------------------------------------------------------------------------- #
# compute_pca — planted truth + invariants
# --------------------------------------------------------------------------- #


def test_compute_pca_rank_one_loads_pc1() -> None:
    """A rank-1 dataset (each feature an affine image of one factor) -> PC1 ~ all."""
    rng = np.random.default_rng(0)
    factor = rng.standard_normal(40)
    loadings = rng.standard_normal(15)
    ab = np.outer(factor, loadings) + 1e-3 * rng.standard_normal((40, 15))
    result = pca.compute_pca(_dataset(ab, {"x": np.arange(40)}))
    assert result.scores.shape == (40, 4)
    assert result.explained_variance_ratio.shape == (4,)
    assert result.explained_variance_ratio[0] > 0.95
    assert result.standardized is True


def test_compute_pca_ratios_descending_and_bounded() -> None:
    ds = _grouped()
    ratio = pca.compute_pca(ds).explained_variance_ratio
    assert np.all(np.diff(ratio) <= 1e-12)  # non-increasing
    assert 0.0 <= float(ratio.sum()) <= 1.0 + 1e-9


def test_compute_pca_without_standardize_flag() -> None:
    ds = _grouped()
    assert pca.compute_pca(ds, standardize=False).standardized is False


def test_compute_pca_rejects_non_finite() -> None:
    ab = np.ones((5, 6))
    ab[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pca.compute_pca(_dataset(ab, {"x": np.arange(5)}))


def test_compute_pca_rejects_too_many_components() -> None:
    ds = _dataset(
        np.random.default_rng(0).standard_normal((3, 10)), {"x": np.arange(3)}
    )
    with pytest.raises(ValueError, match="must be between 1 and"):
        pca.compute_pca(ds, n_components=4)


def test_compute_pca_rejects_single_sample() -> None:
    ds = _dataset(np.ones((1, 10)), {"x": [0]})
    with pytest.raises(ValueError, match="needs >= 2 samples"):
        pca.compute_pca(ds, n_components=1)


# --------------------------------------------------------------------------- #
# plot_pca — categorical
# --------------------------------------------------------------------------- #


def test_plot_categorical_layout_and_colors(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(
        ds, "Group", category="Group", feature_type="protein", registry_path=registry
    )
    # 2 scatter + 4 marginal + 2 title strips = 8 axes (the source layout).
    assert len(plot.figure.get_axes()) == 8
    assert plot.color_map == {"A": _PALETTE[0], "B": _PALETTE[1], "C": _PALETTE[2]}
    # First drawn group (sorted: A) sits in the first scatter collection at palette[0].
    assert _first_facecolor_hex(plot.figure) == _PALETTE[0].lower()
    plt.close(plot.figure)


def test_plot_title_and_feature_type(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(
        ds, "Group", feature_type="precursor", title="My PCA", registry_path=registry
    )
    assert plot.figure.get_suptitle() == "My PCA"
    texts = [t.get_text() for ax in plot.figure.get_axes() for t in ax.texts]
    assert any("precursor" in t for t in texts)
    plt.close(plot.figure)


def test_legend_figure_is_separate_with_group_swatches(registry: Path) -> None:
    """The legend is its own figure (not in the plot) with one swatch per group."""
    ds = _grouped()
    plot = pca.plot_pca(ds, "Group", legend_title="Group", registry_path=registry)
    assert isinstance(plot.legend_figure, Figure)
    assert plot.legend_figure is not plot.figure
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"A", "B", "C"}
    assert legend.get_title().get_text() == "Group"
    # The main plot carries no legend of its own (kept clean / no overlap).
    assert all(ax.get_legend() is None for ax in plot.figure.get_axes())
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_figure_swatch_colors_match_registry(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(ds, "Group", registry_path=registry)
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    by_label = dict(
        zip(
            [t.get_text() for t in legend.get_texts()],
            legend.legend_handles,
            strict=True,
        )
    )
    handle = by_label["A"]
    assert isinstance(handle, Line2D)
    assert mcolors.to_hex(handle.get_markerfacecolor()).lower() == _PALETTE[0].lower()
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_two_group_legend_has_two_swatches(registry: Path) -> None:
    rng = np.random.default_rng(1)
    ab = rng.standard_normal((20, 12)) + 5.0
    ds = _dataset(ab, {"Group": np.array(["A"] * 10 + ["B"] * 10)})
    plot = pca.plot_pca(ds, "Group", registry_path=registry)
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"A", "B"}
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_background_greyed_and_not_persisted(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(
        ds, "Group", category="ST", background_values=["A"], registry_path=registry
    )
    assert plot.color_map is not None
    assert plot.color_map["A"] == col.BACKGROUND_COLOR
    assert plot.color_map["B"] == _PALETTE[0]  # foreground does not skip a slot
    # Background drawn first (underneath): first collection is gray.
    assert _first_facecolor_hex(plot.figure) == col.BACKGROUND_COLOR.lower()
    assert "A" not in json.loads(registry.read_text())["ST"]["values"]
    plt.close(plot.figure)


def test_show_marginal_stats_toggle(registry: Path) -> None:
    ds = _grouped()
    on = pca.plot_pca(ds, "Group", registry_path=registry, show_marginal_stats=True)
    on_texts = [t.get_text() for ax in on.figure.get_axes() for t in ax.texts]
    assert any("p =" in t for t in on_texts)
    plt.close(on.figure)

    off = pca.plot_pca(ds, "Group", registry_path=registry, show_marginal_stats=False)
    off_texts = [t.get_text() for ax in off.figure.get_axes() for t in ax.texts]
    assert all("p =" not in t for t in off_texts)
    plt.close(off.figure)


# --------------------------------------------------------------------------- #
# plot_pca — continuous
# --------------------------------------------------------------------------- #


def test_plot_continuous_no_color_map_and_colorbar_legend(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(
        ds, "RunOrder", continuous=True, colormap="viridis", registry_path=registry
    )
    assert plot.color_map is None
    assert len(plot.figure.get_axes()) == 8
    # The continuous legend is a standalone colorbar figure (its own axes/cax).
    assert isinstance(plot.legend_figure, Figure)
    assert plot.legend_figure is not plot.figure
    assert len(plot.legend_figure.axes) >= 1
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_continuous_marginal_annotation_is_pearson_r(registry: Path) -> None:
    """The continuous annotation reports Pearson r + p per PC (not the unit-bound
    slope), and the numbers are scipy's pearsonr of the variable against the PC."""
    ds = _grouped()
    plot = pca.plot_pca(ds, "RunOrder", continuous=True, registry_path=registry)
    stats_line = plot.figure.get_axes()[6].texts[1].get_text()
    assert stats_line.startswith("Pearson:")
    assert "slope" not in stats_line
    run = ds.metadata["RunOrder"].to_numpy(dtype=float)
    r, p = stats.pearsonr(plot.result.scores[:, 0], run)
    assert f"PC1 r = {r:.2f}, p = {pca._format_p(float(p))}" in stats_line
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


# --------------------------------------------------------------------------- #
# plot_pca — fail loud + the scale warning
# --------------------------------------------------------------------------- #


def test_missing_color_column_raises(registry: Path) -> None:
    ds = _grouped()
    with pytest.raises(ValueError, match="not a metadata column"):
        pca.plot_pca(ds, "Nope", registry_path=registry)


def test_categorical_missing_label_raises(registry: Path) -> None:
    ab = np.random.default_rng(0).standard_normal((6, 8)) + 5.0
    labels = np.array(["A", "A", "B", "B", None, "B"], dtype=object)
    ds = _dataset(ab, {"Group": labels})
    with pytest.raises(ValueError, match="missing value"):
        pca.plot_pca(ds, "Group", registry_path=registry)


def test_more_than_eight_groups_raises(registry: Path) -> None:
    rng = np.random.default_rng(0)
    ab = rng.standard_normal((18, 10)) + 5.0
    labels = np.array([f"g{i}" for i in range(9) for _ in range(2)])
    ds = _dataset(ab, {"Group": labels})
    with pytest.raises(col.CategoricalPaletteExceededError):
        pca.plot_pca(ds, "Group", registry_path=registry)


def test_non_log_scale_warns(registry: Path) -> None:
    rng = np.random.default_rng(0)
    ab = rng.uniform(1.0, 1000.0, size=(30, 20))
    ds = _dataset(ab, {"Group": np.array(["A"] * 15 + ["B"] * 15)}, scale="linear")
    with pytest.warns(pca.PCAScaleWarning, match="not log-ish"):
        plot = pca.plot_pca(ds, "Group", registry_path=registry)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_error_path_closes_figure(registry: Path) -> None:
    """A guard that raises after the figure is built must not leak it (the >8 path)."""
    rng = np.random.default_rng(0)
    ab = rng.standard_normal((18, 10)) + 5.0
    ds = _dataset(
        ab, {"Group": np.array([f"g{i}" for i in range(9) for _ in range(2)])}
    )
    before = set(plt.get_fignums())
    with pytest.raises(col.CategoricalPaletteExceededError):
        pca.plot_pca(ds, "Group", registry_path=registry)
    assert set(plt.get_fignums()) == before


def test_missing_label_error_closes_figure(registry: Path) -> None:
    ab = np.random.default_rng(0).standard_normal((6, 8)) + 5.0
    ds = _dataset(
        ab, {"Group": np.array(["A", "A", "B", "B", None, "B"], dtype=object)}
    )
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="missing value"):
        pca.plot_pca(ds, "Group", registry_path=registry)
    assert set(plt.get_fignums()) == before


def test_save_pca_bad_base_name_closes_both_figures(
    registry: Path, tmp_path: Path
) -> None:
    ds = _grouped()
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="bare filename stem"):
        pca.save_pca(ds, "Group", tmp_path, "bad/name", registry_path=registry)
    assert set(plt.get_fignums()) == before  # main + legend both closed


# --------------------------------------------------------------------------- #
# save_pca — figure + separate legend image
# --------------------------------------------------------------------------- #


def test_save_pca_writes_figure_and_legend_image(
    registry: Path, tmp_path: Path
) -> None:
    ds = _grouped()
    arts = pca.save_pca(
        ds, "Group", tmp_path, "pca_by_group", category="Group", registry_path=registry
    )
    assert arts.svg.exists() and arts.png.exists()
    assert arts.legend_svg is not None and arts.legend_svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
    assert arts.legend_svg == tmp_path / "pca_by_group.legend.svg"


# --------------------------------------------------------------------------- #
# Continuous wiring, arg threading, and the marginal-regression CI band
# --------------------------------------------------------------------------- #


def test_continuous_scatter_colored_by_variable(registry: Path) -> None:
    """The main scatter's color array IS color_by, and the colorbar spans its range."""
    ds = _grouped()
    plot = pca.plot_pca(ds, "RunOrder", continuous=True, registry_path=registry)
    expected = ds.metadata["RunOrder"].to_numpy(dtype=float)
    arr = plot.figure.get_axes()[0].collections[0].get_array()
    assert arr is not None
    np.testing.assert_allclose(np.asarray(arr, dtype=float), expected)
    # The colorbar legend's axis spans [min, max] of the variable.
    lo, hi = plot.legend_figure.axes[0].get_xlim()
    assert lo == pytest.approx(float(expected.min()), abs=1e-6)
    assert hi == pytest.approx(float(expected.max()), abs=1e-6)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_continuous_show_marginal_stats_false(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(
        ds,
        "RunOrder",
        continuous=True,
        show_marginal_stats=False,
        registry_path=registry,
    )
    texts = [t.get_text() for ax in plot.figure.get_axes() for t in ax.texts]
    assert all("r =" not in t and "Pearson" not in t for t in texts)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_persist_colors_false_does_not_write(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(
        ds, "Group", category="NewCat", persist_colors=False, registry_path=registry
    )
    assert "NewCat" not in json.loads(registry.read_text())
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_standardize_false_threads_through(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(ds, "Group", standardize=False, registry_path=registry)
    assert plot.result.standardized is False
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_title_defaults_to_color_by(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(ds, "Group", registry_path=registry)
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "Group"
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_degenerate_pc_skips_regression_marginal(registry: Path) -> None:
    """A ~0-variance trailing PC must not draw a regression on numerical noise (no
    guard before the fix gave slopes like 6e+15): its marginal stays empty and its
    statistic reads n/a."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((5, 3))
    ab = np.hstack([base, base @ rng.standard_normal((3, 7))])  # rank 3 -> PC4 ~ 0
    ds = _dataset(ab, {"RunOrder": np.arange(5.0)})
    plot = pca.plot_pca(ds, "RunOrder", continuous=True, registry_path=registry)
    ax2_right = plot.figure.get_axes()[5]
    assert not ax2_right.lines and not ax2_right.collections
    assert "PC4 n/a" in plot.figure.get_axes()[7].texts[1].get_text()
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_regression_ci_band_is_mean_ci_not_slope_se(registry: Path) -> None:
    """The 95% band must be the mean-response CI (residual SE * t), not the slope SE *
    1.96 (which understated it by ~1/sqrt(Sxx))."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(25)
    y = 1.5 * x + rng.standard_normal(25)
    fig, ax = plt.subplots()
    pca._plot_regression(x, y, ax, plt.get_cmap("viridis"), vertical=False)

    n = 25
    dof = n - 2
    lr = stats.linregress(x, y)
    resid = y - (lr.intercept + lr.slope * x)
    s = float(np.sqrt(float(np.sum(resid**2)) / dof))
    correct_hw = float(stats.t.ppf(0.975, dof)) * s * np.sqrt(1.0 / n)  # at x = mean
    buggy_hw = float(lr.stderr) * np.sqrt(1.0 / n) * 1.96

    # Extract the band half-width at x ~= mean from the fill_between polygon.
    fills = [c for c in ax.collections if type(c).__name__ != "PathCollection"]
    verts = np.asarray(fills[-1].get_paths()[0].vertices, dtype=float)
    order = np.argsort(np.abs(verts[:, 0] - float(x.mean())))
    drawn_hw = abs(float(verts[order[0], 1]) - float(verts[order[1], 1])) / 2.0

    assert drawn_hw == pytest.approx(correct_hw, rel=0.05)
    assert drawn_hw > 2.0 * buggy_hw  # decisively wider than the old slope-SE band
    plt.close(fig)


def test_random_state_param_accepted(registry: Path) -> None:
    """random_state is plumbed through (recordable); full SVD makes it inert here."""
    ds = _grouped()
    a = pca.compute_pca(ds, random_state=0)
    b = pca.compute_pca(ds, random_state=999)
    np.testing.assert_allclose(a.explained_variance_ratio, b.explained_variance_ratio)


# --------------------------------------------------------------------------- #
# Marginal statistics — reference exclusion, p formatting, KDE extent, placement
# --------------------------------------------------------------------------- #


def _stats_lines(figure: Figure) -> list[str]:
    """The statistics line under each panel title (PC1/PC2, then PC3/PC4)."""
    axes = figure.get_axes()
    return [axes[6].texts[1].get_text(), axes[7].texts[1].get_text()]


@pytest.mark.parametrize(
    ("p_value", "expected"),
    [
        (0.5, "0.500"),
        (0.001, "0.001"),
        (0.000999, "1.0e-03"),
        (3.0e-11, "3.0e-11"),
        (0.0, "0.0e+00"),
        (float("nan"), "n/a"),
    ],
)
def test_format_p(p_value: float, expected: str) -> None:
    """A tiny p is never flattened to 0.0000; an untestable one reads n/a."""
    assert pca._format_p(p_value) == expected


def test_strong_separation_p_not_rounded_to_zero(registry: Path) -> None:
    rng = np.random.default_rng(0)
    ab = rng.standard_normal((60, 50))
    ab[:30, :20] += 3.0
    ds = _dataset(ab, {"G": np.array(["A"] * 30 + ["B"] * 30)})
    plot = pca.plot_pca(ds, "G", registry_path=registry)
    line = _stats_lines(plot.figure)[0]
    s = plot.result.scores
    p = float(stats.mannwhitneyu(s[:30, 0], s[30:, 0]).pvalue)
    assert p < 1e-4
    assert f"PC1 p = {pca._format_p(p)}" in line
    assert "0.0000" not in line
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_background_samples_excluded_from_group_test(registry: Path) -> None:
    """Greyed reference samples must not drive the p: A and B are drawn identically,
    a far-off pool is greyed, and the reported p is the A-vs-B test alone (before the
    fix the pool made it ~3e-4 against an A-vs-B p of ~0.9)."""
    rng = np.random.default_rng(0)
    rng.standard_normal((60, 50))  # keep the draw identical to the review probe
    ab = rng.standard_normal((66, 50))
    ab[60:, :] += 4.0
    labels = np.array(["A"] * 30 + ["B"] * 30 + ["Pool"] * 6)
    ds = _dataset(ab, {"G": labels})
    plot = pca.plot_pca(ds, "G", background_values=["Pool"], registry_path=registry)
    s = plot.result.scores
    lines = _stats_lines(plot.figure)
    assert lines[0].startswith("Mann-Whitney U:")  # 2 foreground groups, not 3
    for pc in range(4):
        p = float(stats.mannwhitneyu(s[:30, pc], s[30:60, pc]).pvalue)
        assert f"PC{pc + 1} p = {pca._format_p(p)}" in lines[pc // 2]
    p_with_pool = float(stats.kruskal(s[:30, 0], s[30:60, 0], s[60:, 0]).pvalue)
    assert p_with_pool < 1e-3  # what the old annotation reported
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


@pytest.mark.parametrize(
    "make_background",
    [
        lambda: ["Pool"],
        lambda: np.array(["Pool"]),
        lambda: pd.Series(["Pool"]),
        lambda: pd.Index(["Pool"]),
        lambda: (v for v in ["Pool"]),
        lambda: "Pool",
    ],
    ids=["list", "ndarray", "series", "index", "generator", "bare-str"],
)
def test_background_exclusion_accepts_any_iterable(
    registry: Path, make_background: object
) -> None:
    """However the caller spells background_values (e.g. ``df[...].unique()`` is an
    ndarray), the greyed samples are greyed, drawn underneath, AND left out of the
    test — the registry and the plot must agree on which values are background."""
    rng = np.random.default_rng(0)
    ab = rng.standard_normal((66, 50))
    ab[60:, :] += 4.0
    labels = np.array(["A"] * 30 + ["B"] * 30 + ["Pool"] * 6)
    ds = _dataset(ab, {"G": labels})
    assert callable(make_background)
    plot = pca.plot_pca(
        ds, "G", background_values=make_background(), registry_path=registry
    )
    assert plot.color_map is not None
    assert plot.color_map["Pool"] == col.BACKGROUND_COLOR
    assert _first_facecolor_hex(plot.figure) == col.BACKGROUND_COLOR.lower()
    s = plot.result.scores
    p = float(stats.mannwhitneyu(s[:30, 0], s[30:60, 0]).pvalue)
    line = _stats_lines(plot.figure)[0]
    assert line.startswith("Mann-Whitney U:")
    assert f"PC1 p = {pca._format_p(p)}" in line
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_single_foreground_group_has_no_test(registry: Path) -> None:
    rng = np.random.default_rng(2)
    ab = rng.standard_normal((20, 12))
    ds = _dataset(ab, {"G": np.array(["exp"] * 16 + ["pool"] * 4)})
    plot = pca.plot_pca(ds, "G", background_values=["pool"], registry_path=registry)
    for line in _stats_lines(plot.figure):
        assert line.count("p = n/a") == 2
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_kde_spans_the_scatter_view_and_leaves_it_unchanged(registry: Path) -> None:
    """Each KDE is evaluated across the scatter's full view (its tails run to the
    frame, not stopping at the extreme sample); drawing it does not move the view."""
    ds = _grouped()
    plot = pca.plot_pca(ds, "Group", registry_path=registry)
    axes = plot.figure.get_axes()
    ax1, ax2, ax1_top, ax1_right = axes[0], axes[1], axes[2], axes[3]
    s = plot.result.scores
    lo, hi = ax1.get_xlim()
    assert lo < float(s[:, 0].min()) and hi > float(s[:, 0].max())
    for line in ax1_top.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        assert x[0] == pytest.approx(lo) and x[-1] == pytest.approx(hi)
    ylo, yhi = ax1.get_ylim()
    for line in ax1_right.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        assert y[0] == pytest.approx(ylo) and y[-1] == pytest.approx(yhi)
    # The view is pinned (not autoscaled past the curves) on both panels.
    assert not ax1.get_autoscalex_on() and not ax1.get_autoscaley_on()
    assert not ax2.get_autoscalex_on() and not ax2.get_autoscaley_on()
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_density_marginals_share_a_zero_baseline(registry: Path) -> None:
    ds = _grouped()
    plot = pca.plot_pca(ds, "Group", registry_path=registry)
    axes = plot.figure.get_axes()
    assert axes[2].get_ylim()[0] == 0.0 and axes[4].get_ylim()[0] == 0.0  # top
    assert axes[3].get_xlim()[0] == 0.0 and axes[5].get_xlim()[0] == 0.0  # right
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


@pytest.mark.parametrize("continuous", [False, True])
def test_stats_live_in_title_strips_not_marginals(
    registry: Path, continuous: bool
) -> None:
    """No statistic is drawn inside the narrow marginal axes (where it overlapped the
    curves and overflowed the figure edge); each title strip carries one stats line."""
    ds = _grouped()
    color_by = "RunOrder" if continuous else "Group"
    plot = pca.plot_pca(ds, color_by, continuous=continuous, registry_path=registry)
    axes = plot.figure.get_axes()
    assert all(not ax.texts for ax in axes[2:6])
    assert [len(axes[i].texts) for i in (6, 7)] == [2, 2]
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


# --------------------------------------------------------------------------- #
# Smoke — real 5xFAD proteins (git-ignored)
# --------------------------------------------------------------------------- #

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)


@_skip_no_data
def test_smoke_pca_real_proteins(tmp_path: Path) -> None:
    """median+log2 PCA of the real proteins matches the captured variance oracle."""
    from common import normalize as norm

    registry = tmp_path / "color_registry.json"
    registry.write_text(
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
    ds = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=dl.ReplicateCollapse("Sample ID", "Technical Replicate"),
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    logged = norm.log2_transform(norm.normalize(ds, "median"))
    assert logged.abundances.shape == (61, 8829)

    result = pca.compute_pca(logged)
    variance_pct = result.explained_variance_ratio * 100.0
    # Captured oracle — pin all four (a wiring regression could keep PC1 ~right while
    # drifting the rest).
    np.testing.assert_allclose(variance_pct, [31.07, 14.45, 8.88, 5.09], atol=0.3)

    plot = pca.plot_pca(
        logged,
        "SampleType",
        category="SampleType",
        feature_type="protein",
        registry_path=registry,
    )
    assert len(plot.figure.get_axes()) == 8
    assert plot.color_map is not None
    assert "brain" in plot.color_map and "unknown" in plot.color_map
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    assert "brain" in {t.get_text() for t in legend.get_texts()}
    plt.close(plot.figure)
    plt.close(plot.legend_figure)
