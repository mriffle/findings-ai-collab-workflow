"""Tests for the CV-plot template (lib/figures/cv.py).

Layers:
  * unit/compute — planted-truth on ``compute_cv`` (hand-computed CVs, the mean<=0 ->
    NaN rule) plus the linear-scale hard-refuse and the few-samples guard;
  * unit/plot — the single-axes figure structure, registry coloring, the separate
    swatch legend, median annotations + toggle, shared-bin x-limit, and the guards
    (empty input, feature mismatch, >8-category overflow, figure-leak on error);
  * smoke — real 5xFAD proteins (git-ignored) reproducing the captured median-CV oracle
    (unnormalized vs median-normalized); skips cleanly when the data is absent.
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
from figures import cv
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
    *,
    scale: dl.Scale = "linear",
    feature_names: np.ndarray | None = None,
) -> dl.Dataset:
    """Build a linear-scale Dataset around an abundance matrix (samples x features)."""
    n_samples, n_features = abundances.shape
    names = (
        feature_names
        if feature_names is not None
        else np.array([f"F{j}" for j in range(n_features)], dtype=str)
    )
    meta = pd.DataFrame(
        {"sample": [f"s{i}" for i in range(n_samples)]},
        index=[f"s{i}" for i in range(n_samples)],
    )
    return dl.Dataset(
        abundances=np.asarray(abundances, dtype=float),
        feature_names=names,
        feature_metadata=pd.DataFrame({"protein": names}),
        metadata=meta,
        scale=scale,
    )


def _two_states(n_samples: int = 12, n_features: int = 40) -> dict[str, dl.Dataset]:
    """Two linear states (Unnormalized/Normalized) over the same features."""
    rng = np.random.default_rng(7)
    base = rng.lognormal(mean=8.0, sigma=1.0, size=(n_samples, n_features))
    tighter = base / base.mean(axis=1, keepdims=True) * base.mean()
    return {
        "Unnormalized": _dataset(base),
        "Normalized": _dataset(tighter),
    }


# --------------------------------------------------------------------------- #
# compute_cv — planted truth + guards
# --------------------------------------------------------------------------- #


def test_compute_cv_planted_values() -> None:
    """Hand-computed CVs: [1,2,3]->0.5, constant->0, all-zero mean<=0 ->NaN."""
    ab = np.array(
        [
            [1.0, 5.0, 0.0],
            [2.0, 5.0, 0.0],
            [3.0, 5.0, 0.0],
        ]
    )
    cvs = cv.compute_cv(_dataset(ab))
    assert cvs[0] == pytest.approx(0.5)  # mean 2, std(ddof=1) 1
    assert cvs[1] == pytest.approx(0.0)  # constant column
    assert np.isnan(cvs[2])  # mean == 0 -> undefined


def test_compute_cv_negative_mean_is_nan() -> None:
    """A mean<=0 column is NaN, not a (nonsensical) negative CV."""
    ab = np.array([[-1.0, 2.0], [-3.0, 2.0], [1.0, 2.0]])  # col0 mean = -1
    cvs = cv.compute_cv(_dataset(ab))
    assert np.isnan(cvs[0])


def test_compute_cv_refuses_non_linear_scale() -> None:
    ab = np.random.default_rng(0).standard_normal((5, 4)) + 10.0
    with pytest.raises(cv.CVScaleError, match="linear-scale"):
        cv.compute_cv(_dataset(ab, scale="log2"))


def test_compute_cv_refuses_too_few_samples() -> None:
    with pytest.raises(ValueError, match=">= 2 samples"):
        cv.compute_cv(_dataset(np.array([[1.0, 2.0, 3.0]])))


# --------------------------------------------------------------------------- #
# plot_cv_distribution — structure, colors, legend, annotations
# --------------------------------------------------------------------------- #


def test_plot_single_axes_and_one_hist_per_state(registry: Path) -> None:
    plot = cv.plot_cv_distribution(_two_states(), registry_path=registry)
    assert isinstance(plot, cv.CVPlot)
    assert len(plot.figure.get_axes()) == 1  # one overlay axes (legend is separate)
    assert len(plot.figure.axes[0].containers) == 2  # one histogram per state
    # The main axes carries no baked legend (kept clean / no overlap).
    assert plot.figure.axes[0].get_legend() is None
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_colors_from_registry(registry: Path) -> None:
    plot = cv.plot_cv_distribution(_two_states(), registry_path=registry)
    assert plot.color_map == {
        "Unnormalized": _PALETTE[0],
        "Normalized": _PALETTE[1],
    }
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_is_separate_figure_with_state_swatches(registry: Path) -> None:
    plot = cv.plot_cv_distribution(
        _two_states(), legend_title="State", registry_path=registry
    )
    assert isinstance(plot.legend_figure, Figure)
    assert plot.legend_figure is not plot.figure
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"Unnormalized", "Normalized"}
    assert legend.get_title().get_text() == "State"
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_swatch_color_matches_registry(registry: Path) -> None:
    plot = cv.plot_cv_distribution(_two_states(), registry_path=registry)
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    by_label = dict(
        zip(
            [t.get_text() for t in legend.get_texts()],
            legend.legend_handles,
            strict=True,
        )
    )
    handle = by_label["Unnormalized"]
    assert isinstance(handle, Line2D)
    assert mcolors.to_hex(handle.get_color()).lower() == _PALETTE[0].lower()
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_title_defaults_to_category(registry: Path) -> None:
    plot = cv.plot_cv_distribution(
        _two_states(), category="NormState", registry_path=registry
    )
    legend = plot.legend_figure.axes[0].get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "NormState"
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_median_annotations_present_and_match_result(registry: Path) -> None:
    plot = cv.plot_cv_distribution(_two_states(), registry_path=registry)
    texts = [t.get_text() for t in plot.figure.axes[0].texts]
    assert sum("median=" in t for t in texts) == 2
    # The annotated value matches the computed median (3 d.p.).
    med = plot.result.medians["Unnormalized"]
    assert any(f"median={med:.3f}" in t for t in texts)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_show_medians_false_suppresses_lines_and_text(registry: Path) -> None:
    plot = cv.plot_cv_distribution(
        _two_states(), show_medians=False, registry_path=registry
    )
    texts = [t.get_text() for t in plot.figure.axes[0].texts]
    assert all("median=" not in t for t in texts)
    # No dashed median axvlines either (KDE lines are solid).
    assert all(line.get_linestyle() != "--" for line in plot.figure.axes[0].lines)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_max_cv_caps_x_axis(registry: Path) -> None:
    # max_cv inside the data's CV range, so each capped histogram still has content
    # (capping below every CV empties a histogram -> matplotlib density /0 warning).
    plot = cv.plot_cv_distribution(_two_states(), max_cv=2.0, registry_path=registry)
    lo, hi = plot.figure.axes[0].get_xlim()
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(2.0)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_single_state_allowed(registry: Path) -> None:
    states = {"Normalized": _two_states()["Normalized"]}
    plot = cv.plot_cv_distribution(states, registry_path=registry)
    assert len(plot.figure.axes[0].containers) == 1
    assert plot.color_map == {"Normalized": _PALETTE[0]}
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_persist_colors_false_does_not_write(registry: Path) -> None:
    plot = cv.plot_cv_distribution(
        _two_states(),
        category="Ephemeral",
        persist_colors=False,
        registry_path=registry,
    )
    assert "Ephemeral" not in json.loads(registry.read_text())
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


# --------------------------------------------------------------------------- #
# plot_cv_distribution — fail loud + figure-leak safety
# --------------------------------------------------------------------------- #


def test_empty_datasets_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        cv.plot_cv_distribution({}, registry_path=registry)


def test_bad_n_bins_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="n_bins"):
        cv.plot_cv_distribution(_two_states(), n_bins=0, registry_path=registry)


def test_bad_max_cv_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="max_cv"):
        cv.plot_cv_distribution(_two_states(), max_cv=0.0, registry_path=registry)


def test_feature_mismatch_raises(registry: Path) -> None:
    rng = np.random.default_rng(0)
    a = _dataset(rng.standard_normal((6, 4)) + 10.0)
    b = _dataset(
        rng.standard_normal((6, 4)) + 10.0,
        feature_names=np.array(["X0", "X1", "X2", "X3"], dtype=str),
    )
    with pytest.raises(ValueError, match="different feature_names"):
        cv.plot_cv_distribution({"A": a, "B": b}, registry_path=registry)


def test_non_linear_state_refused(registry: Path) -> None:
    states = _two_states()
    states["Logged"] = _dataset(
        np.random.default_rng(0).standard_normal((12, 40)) + 5.0, scale="log2"
    )
    with pytest.raises(cv.CVScaleError):
        cv.plot_cv_distribution(states, registry_path=registry)


def _nine_states() -> dict[str, dl.Dataset]:
    """Nine linear states over identical features (to trip the >8 color guard)."""
    ds = _two_states()["Normalized"]
    return {f"state{i}": ds for i in range(9)}


def test_more_than_eight_states_raises(registry: Path) -> None:
    with pytest.raises(col.CategoricalPaletteExceededError):
        cv.plot_cv_distribution(_nine_states(), registry_path=registry)


def test_error_path_closes_figure(registry: Path) -> None:
    """The >8 guard fires after the figure is built; it must not leak the figure."""
    before = set(plt.get_fignums())
    with pytest.raises(col.CategoricalPaletteExceededError):
        cv.plot_cv_distribution(_nine_states(), registry_path=registry)
    assert set(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# save_cv — figure + separate legend image
# --------------------------------------------------------------------------- #


def test_save_cv_writes_figure_and_legend_image(registry: Path, tmp_path: Path) -> None:
    arts = cv.save_cv(_two_states(), tmp_path, "cv_by_state", registry_path=registry)
    assert arts.svg.exists() and arts.png.exists()
    assert arts.legend_svg is not None and arts.legend_svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
    assert arts.legend_svg == tmp_path / "cv_by_state.legend.svg"


def test_save_cv_bad_base_name_closes_both_figures(
    registry: Path, tmp_path: Path
) -> None:
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="bare filename stem"):
        cv.save_cv(_two_states(), tmp_path, "bad/name", registry_path=registry)
    assert set(plt.get_fignums()) == before  # main + legend both closed


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
def test_smoke_cv_real_proteins(registry: Path, tmp_path: Path) -> None:
    """median normalization tightens the protein CV; pin the captured oracle medians."""
    from common import normalize as norm

    raw = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=dl.ReplicateCollapse("Sample ID", "Technical Replicate"),
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    normed = norm.normalize(raw, "median")
    assert raw.scale == "linear" and normed.scale == "linear"

    states = {"Unnormalized": raw, "Normalized": normed}
    plot = cv.plot_cv_distribution(
        states, feature_type="protein", registry_path=registry
    )
    medians = plot.result.medians
    # Captured oracle (compute_cv on the real matrix).
    assert medians["Unnormalized"] == pytest.approx(0.8078, abs=0.01)
    assert medians["Normalized"] == pytest.approx(0.6040, abs=0.01)
    # The QC truth: normalization reduces the median CV.
    assert medians["Normalized"] < medians["Unnormalized"]

    assert len(plot.figure.axes[0].containers) == 2
    plt.close(plot.figure)
    plt.close(plot.legend_figure)

    arts = cv.save_cv(states, tmp_path, "proteins_cv", registry_path=registry)
    assert arts.png.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
