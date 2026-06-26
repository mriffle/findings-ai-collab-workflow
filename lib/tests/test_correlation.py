"""Tests for the sample-correlation template (lib/figures/correlation.py).

Layers:
  * unit/compute — planted-truth on ``compute_correlation`` (an exact 2-block matrix
    whose within-block samples are perfectly correlated and across-block perfectly
    anti-correlated), the symmetry/unit-diagonal invariants, clustering that recovers
    the blocks, method dispatch, and the guards (bad method, few samples, non-finite,
    constant sample, duplicate ids, bad label column);
  * unit/plot — the layout axes (dendrogram presence by ``cluster``, one stripe per
    annotation, the colorbar), registry coloring + the returned color maps, the separate
    legend figure (present with annotations, ``None`` without), the >8-category
    overflow, the Pearson/linear scale warning (and Spearman / log2 silence), the
    annotation guards (missing / overlapping / NaN column), and the figure-leak guard;
  * smoke — real 5xFAD proteins (git-ignored) reproducing the captured Pearson oracle;
    skips cleanly when the data is absent.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from common import data_loading as dl
from figures import colors as col
from figures import correlation as corr

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
    metadata: dict[str, object] | None = None,
    *,
    scale: dl.Scale = "linear",
    index: list[str] | None = None,
) -> dl.Dataset:
    """Build a Dataset around an abundance matrix (+ optional metadata columns)."""
    n_samples, n_features = abundances.shape
    feature_names = np.array([f"F{j}" for j in range(n_features)], dtype=str)
    idx = index if index is not None else [f"s{i}" for i in range(n_samples)]
    meta = pd.DataFrame(metadata if metadata is not None else {}, index=idx)
    return dl.Dataset(
        abundances=np.asarray(abundances, dtype=float),
        feature_names=feature_names,
        feature_metadata=pd.DataFrame({"protein": feature_names}),
        metadata=meta,
        scale=scale,
    )


def _two_block(n_features: int = 20) -> np.ndarray:
    """Four samples: {0,1} correlated, {2,3} correlated, the two blocks anti-correlated.

    Each sample is an affine function of a shared feature axis, so Pearson is exactly
    +/-1: ``corr(a*x+b, c*x+d) = sign(a*c)``.
    """
    base = np.arange(n_features, dtype=float)
    s0 = base
    s1 = 2.0 * base + 3.0  # corr(s0, s1) = +1
    s2 = 5.0 - base  # corr(s0, s2) = -1
    s3 = 10.0 - 2.0 * base  # corr(s2, s3) = +1
    return np.vstack([s0, s1, s2, s3])


# --------------------------------------------------------------------------- #
# compute_correlation — planted truth + invariants
# --------------------------------------------------------------------------- #


def test_compute_planted_block_matrix() -> None:
    """The exact +/-1 block structure, symmetry, and unit diagonal."""
    ds = _dataset(_two_block())
    res = corr.compute_correlation(ds, method="pearson", cluster=False)
    m = res.matrix
    assert m.shape == (4, 4)
    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m, m.T)  # symmetric
    assert m[0, 1] == pytest.approx(1.0)
    assert m[2, 3] == pytest.approx(1.0)
    assert m[0, 2] == pytest.approx(-1.0)
    assert m[1, 3] == pytest.approx(-1.0)
    # cluster=False -> identity order
    assert np.array_equal(res.order, np.arange(4))


def test_compute_clustering_recovers_blocks() -> None:
    """Average-linkage order keeps the two correlated blocks contiguous."""
    res = corr.compute_correlation(
        _dataset(_two_block()), method="pearson", cluster=True
    )
    order = res.order.tolist()
    # {0,1} adjacent and {2,3} adjacent, in either block order.
    assert abs(order.index(0) - order.index(1)) == 1
    assert abs(order.index(2) - order.index(3)) == 1


def test_compute_spearman_dispatch() -> None:
    """Spearman returns a valid (symmetric, unit-diagonal) matrix; monotone -> +/-1."""
    res = corr.compute_correlation(_dataset(_two_block()), method="spearman")
    m = res.matrix
    assert np.allclose(np.diag(m), 1.0)
    assert m[0, 1] == pytest.approx(1.0)
    assert m[0, 2] == pytest.approx(-1.0)


def test_compute_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        corr.compute_correlation(_dataset(_two_block()), method="kendall")


def test_compute_rejects_one_sample() -> None:
    with pytest.raises(ValueError, match=">= 2 samples"):
        corr.compute_correlation(_dataset(np.arange(5.0).reshape(1, 5)))


def test_compute_rejects_non_finite() -> None:
    ab = _two_block().copy()
    ab[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        corr.compute_correlation(_dataset(ab))


def test_compute_rejects_constant_sample() -> None:
    """A sample with no spread across features is undefined -> fail loud, named."""
    ab = _two_block()
    ab[1, :] = 7.0  # constant row
    with pytest.raises(ValueError, match="constant across all features"):
        corr.compute_correlation(_dataset(ab, index=["a", "b", "c", "d"]))


def test_compute_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        corr.compute_correlation(_dataset(_two_block(), index=["a", "a", "b", "c"]))


def test_compute_rejects_bad_label_column() -> None:
    with pytest.raises(ValueError, match="sample_label_column"):
        corr.compute_correlation(_dataset(_two_block()), sample_label_column="nope")


# --------------------------------------------------------------------------- #
# plot — layout, color, legend, guards
# --------------------------------------------------------------------------- #


def _annotated(n: int = 6, n_features: int = 15) -> dl.Dataset:
    """A small dataset with two categorical + one continuous annotation column."""
    rng = np.random.default_rng(1)
    ab = rng.standard_normal((n, n_features)) + 5.0
    half = n // 2
    return _dataset(
        ab,
        {
            "Group": np.array(["A"] * half + ["B"] * (n - half)),
            "Sex": np.array((["F", "M"] * n)[:n]),
            "RunOrder": np.arange(n, dtype=float),
        },
        scale="log2",
    )


def test_plot_layout_axes_with_cluster(registry: Path) -> None:
    """cluster=True: dendrogram + one stripe per annotation + heatmap + colorbar."""
    ds = _annotated()
    plot = corr.plot_sample_correlation(
        ds,
        categorical_annotations=["Group", "Sex"],
        continuous_annotations=["RunOrder"],
        registry_path=registry,
    )
    # dendrogram(1) + stripes(3) + heatmap(1) + colorbar(1) = 6
    assert len(plot.figure.axes) == 6
    assert plot.legend_figure is not None
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_no_dendrogram_when_unclustered(registry: Path) -> None:
    """cluster=False drops the dendrogram row (stripes + heatmap + colorbar only)."""
    ds = _annotated()
    plot = corr.plot_sample_correlation(
        ds,
        categorical_annotations=["Group"],
        cluster=False,
        registry_path=registry,
    )
    # stripes(1) + heatmap(1) + colorbar(1) = 3
    assert len(plot.figure.axes) == 3
    assert np.array_equal(plot.result.order, np.arange(ds.abundances.shape[0]))
    plt.close(plot.figure)
    if plot.legend_figure is not None:
        plt.close(plot.legend_figure)


def test_plot_color_maps_from_registry(registry: Path) -> None:
    """Categorical annotations get registry colors; the maps come back per dimension."""
    plot = corr.plot_sample_correlation(
        _annotated(),
        categorical_annotations=["Group", "Sex"],
        registry_path=registry,
    )
    assert set(plot.color_maps) == {"Group", "Sex"}
    assert set(plot.color_maps["Group"]) == {"A", "B"}
    for hexval in plot.color_maps["Group"].values():
        assert hexval in _PALETTE
    # The registry persisted the new dimensions.
    saved = json.loads(registry.read_text())
    assert "Group" in saved and "Sex" in saved
    plt.close(plot.figure)
    if plot.legend_figure is not None:
        plt.close(plot.legend_figure)


def test_plot_no_annotations_has_no_legend(registry: Path) -> None:
    """With no annotation columns there is no legend figure (and none is saved)."""
    plot = corr.plot_sample_correlation(_annotated(), registry_path=registry)
    assert plot.legend_figure is None
    assert plot.color_maps == {}
    plt.close(plot.figure)


def test_plot_over_eight_categories_raises_and_closes(registry: Path) -> None:
    """A 9-value categorical annotation trips the >8 guard and leaks no figure."""
    rng = np.random.default_rng(2)
    ab = rng.standard_normal((9, 12)) + 5.0
    ds = _dataset(ab, {"Many": np.array([f"g{i}" for i in range(9)])}, scale="log2")
    before = set(plt.get_fignums())
    with pytest.raises(col.CategoricalPaletteExceededError):
        corr.plot_sample_correlation(
            ds, categorical_annotations=["Many"], registry_path=registry
        )
    assert set(plt.get_fignums()) == before  # the orphaned figure was closed


def test_plot_pearson_linear_warns(registry: Path) -> None:
    """Pearson on a linear scale warns about abundant-feature dominance."""
    ds = _annotated()
    ds.scale = "linear"
    with pytest.warns(corr.CorrelationScaleWarning):
        plot = corr.plot_sample_correlation(
            ds, categorical_annotations=["Group"], registry_path=registry
        )
    plt.close(plot.figure)
    if plot.legend_figure is not None:
        plt.close(plot.legend_figure)


def test_plot_pearson_log_and_spearman_do_not_warn(registry: Path) -> None:
    """No warning for Pearson on log2, nor for Spearman on any scale."""
    ds_log = _annotated()  # scale="log2"
    ds_lin = _annotated()
    ds_lin.scale = "linear"
    with warnings.catch_warnings():
        warnings.simplefilter("error", corr.CorrelationScaleWarning)
        p1 = corr.plot_sample_correlation(
            ds_log, categorical_annotations=["Group"], registry_path=registry
        )
        p2 = corr.plot_sample_correlation(
            ds_lin,
            categorical_annotations=["Group"],
            method="spearman",
            registry_path=registry,
        )
    for p in (p1, p2):
        plt.close(p.figure)
        if p.legend_figure is not None:
            plt.close(p.legend_figure)


def test_plot_rejects_missing_annotation(registry: Path) -> None:
    with pytest.raises(ValueError, match="not a metadata column"):
        corr.plot_sample_correlation(
            _annotated(), categorical_annotations=["Nope"], registry_path=registry
        )


def test_plot_rejects_overlapping_annotation(registry: Path) -> None:
    with pytest.raises(ValueError, match="both"):
        corr.plot_sample_correlation(
            _annotated(),
            categorical_annotations=["RunOrder"],
            continuous_annotations=["RunOrder"],
            registry_path=registry,
        )


def test_plot_rejects_missing_values_in_annotation(registry: Path) -> None:
    ab = np.random.default_rng(3).standard_normal((4, 10)) + 5.0
    ds = _dataset(
        ab, {"Group": np.array(["A", "B", None, "B"], dtype=object)}, scale="log2"
    )
    with pytest.raises(ValueError, match="missing value"):
        corr.plot_sample_correlation(
            ds, categorical_annotations=["Group"], registry_path=registry
        )


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #


def test_save_writes_dual_export_plus_legend(registry: Path, tmp_path: Path) -> None:
    arts = corr.save_sample_correlation(
        _annotated(),
        tmp_path,
        "sample_corr",
        categorical_annotations=["Group"],
        continuous_annotations=["RunOrder"],
        registry_path=registry,
    )
    assert arts.svg.exists() and arts.png.exists()
    assert arts.legend_svg is not None and arts.legend_svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
    assert set(plt.get_fignums()) == set()  # save closed both figures


def test_save_without_annotations_writes_no_legend(
    registry: Path, tmp_path: Path
) -> None:
    arts = corr.save_sample_correlation(
        _annotated(), tmp_path, "bare_corr", registry_path=registry
    )
    assert arts.png.exists()
    assert arts.legend_png is None and arts.legend_svg is None


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
def test_smoke_correlation_real_proteins(registry: Path, tmp_path: Path) -> None:
    """Reproduce the captured Pearson (linear) oracle on the real 5xFAD proteins."""
    raw = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=dl.ReplicateCollapse("Sample ID", "Technical Replicate"),
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    res = corr.compute_correlation(raw, method="pearson", cluster=True)
    assert res.matrix.shape == (61, 61)
    off = res.matrix[~np.eye(61, dtype=bool)]
    # Captured oracle (pandas .corr on the real linear matrix).
    assert off.min() == pytest.approx(0.0906, abs=0.005)
    assert float(np.median(off)) == pytest.approx(0.8958, abs=0.005)
    assert off.mean() == pytest.approx(0.8227, abs=0.005)
    assert res.matrix[0, 1] == pytest.approx(0.3418, abs=0.005)

    # Pearson on the linear scale warns; the figure + legend still save.
    with pytest.warns(corr.CorrelationScaleWarning):
        arts = corr.save_sample_correlation(
            raw,
            tmp_path,
            "proteins_correlation",
            categorical_annotations=["SampleType", "Genotype"],
            continuous_annotations=["RunOrder"],
            feature_type="protein",
            registry_path=registry,
        )
    assert arts.png.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
