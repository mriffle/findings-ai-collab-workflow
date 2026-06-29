"""Tests for the missingness template (lib/figures/missingness.py).

Layers:
  * unit/compute — planted-truth on ``compute_completeness`` (hand-computed feature /
    sample detection rates, mean-log2-abundance, the MNAR-correlation sign), plus the
    non-linear scale refuse and shape/row-alignment guards;
  * unit/plot — the two-panel structure, per-class completeness coloring via the
    registry + the separate legend, the single-curve (uncolored) path, and the guards
    (bad color_by / missing values, >8-category overflow, figure-leak on error);
  * smoke — real 5xFAD proteins (git-ignored) reproducing the captured oracle: 8829
    features, 6162 complete, MNAR Pearson r ~ 0.524 (low-abundance features
    left-censored), max per-sample missing ~ 0.218. Skips cleanly when data is absent.
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
from common import data_loading as dl
from figures import colors as col
from figures import missingness as ms

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
    """Isolated registry seeded with the canonical 8-color palette."""
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
    sample_ids: list[str] | None = None,
    meta: dict[str, list[object]] | None = None,
) -> dl.Dataset:
    """Build a linear-scale Dataset (missingness refuses non-linear) around a matrix."""
    n_samples, n_features = abundances.shape
    ids = sample_ids if sample_ids is not None else [f"s{i}" for i in range(n_samples)]
    columns = {} if meta is None else dict(meta)
    metadata = pd.DataFrame(columns, index=pd.Index(ids, name="sample"))
    names = np.array([f"F{j}" for j in range(n_features)], dtype=str)
    return dl.Dataset(
        abundances=np.asarray(abundances, dtype=float),
        feature_names=names,
        feature_metadata=pd.DataFrame({"protein": names}),
        metadata=metadata,
        scale=scale,
    )


# --------------------------------------------------------------------------- #
# compute_completeness — planted truth + guards
# --------------------------------------------------------------------------- #


def test_compute_planted_rates_and_abundance() -> None:
    """Hand-computed detection rates + mean log2 abundance of detected values."""
    ab = np.array(
        [
            [10.0, 0.0, 5.0, 0.0],  # s0 detects F0,F2
            [20.0, 0.0, 0.0, 8.0],  # s1 detects F0,F3
            [30.0, 4.0, 0.0, 0.0],  # s2 detects F0,F1
        ]
    )
    res = ms.compute_completeness(_dataset(ab))
    # feature detection rate: F0 in all 3; F1,F2,F3 in 1 of 3
    assert res.feature_detection_rate.tolist() == pytest.approx(
        [1.0, 1 / 3, 1 / 3, 1 / 3]
    )
    # each sample detects 2 of 4 features
    assert res.sample_detection_rate.tolist() == pytest.approx([0.5, 0.5, 0.5])
    # mean log2 abundance of detected values
    f0 = float(np.mean([np.log2(10.0), np.log2(20.0), np.log2(30.0)]))
    assert res.feature_mean_log_abundance.tolist() == pytest.approx(
        [f0, np.log2(4.0), np.log2(5.0), np.log2(8.0)]
    )


def test_compute_mnar_correlation_sign() -> None:
    """A planted left-censoring pattern yields a positive MNAR correlation."""
    # 4 features, increasing abundance left->right; low-abundance features missing more.
    ab = np.array(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [0.0, 10.0, 100.0, 1000.0],
            [0.0, 0.0, 100.0, 1000.0],
            [0.0, 0.0, 0.0, 1000.0],
        ]
    )
    res = ms.compute_completeness(_dataset(ab))
    # detection rate rises with feature index (abundance) -> positive corr
    assert res.feature_detection_rate.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert res.mnar_correlation > 0.9


def test_compute_never_detected_feature_is_nan_abundance() -> None:
    ab = np.array([[0.0, 5.0], [0.0, 6.0]])  # F0 never detected
    res = ms.compute_completeness(_dataset(ab))
    assert res.feature_detection_rate.tolist() == pytest.approx([0.0, 1.0])
    assert np.isnan(res.feature_mean_log_abundance[0])
    assert res.feature_mean_log_abundance[1] == pytest.approx(
        np.mean([np.log2(5.0), np.log2(6.0)])
    )


def test_compute_refuses_non_linear_scale() -> None:
    ab = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ms.MissingnessScaleError, match="linear"):
        ms.compute_completeness(_dataset(ab, scale="log2"))


def test_compute_requires_2d() -> None:
    bad = dl.Dataset(
        abundances=np.array([1.0, 2.0, 3.0]),
        feature_names=np.array(["F0", "F1", "F2"], dtype=str),
        feature_metadata=pd.DataFrame({"protein": ["F0", "F1", "F2"]}),
        metadata=pd.DataFrame(index=pd.Index(["s0"], name="sample")),
        scale="linear",
    )
    with pytest.raises(ValueError, match="2D"):
        ms.compute_completeness(bad)


def test_compute_row_misalignment_raises() -> None:
    ds = _dataset(np.array([[1.0, 0.0], [1.0, 1.0]]), sample_ids=["s0", "s1"])
    object.__setattr__(
        ds, "metadata", pd.DataFrame(index=pd.Index(["only"], name="sample"))
    )
    with pytest.raises(ValueError, match="not row-aligned"):
        ms.compute_completeness(ds)


# --------------------------------------------------------------------------- #
# plot_missingness — structure, color, legend
# --------------------------------------------------------------------------- #


def _class_dataset() -> dl.Dataset:
    """Six samples, two classes, increasing-abundance features (a clean MNAR slope)."""
    rng = np.random.default_rng(0)
    base = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    rows = []
    for _ in range(6):
        keep = rng.random(base.size) < (np.arange(base.size) + 1) / base.size
        rows.append(np.where(keep, base, 0.0))
    ab = np.vstack(rows)
    ab[:, -1] = base[-1]  # the top feature is always detected
    return _dataset(
        ab,
        sample_ids=[f"s{i}" for i in range(6)],
        meta={"klass": ["exp", "exp", "exp", "exp", "ctrl", "ctrl"]},
    )


def test_plot_two_panels_and_result(registry: Path) -> None:
    plot = ms.plot_missingness(
        _class_dataset(), color_by="klass", registry_path=registry
    )
    # two panels + the hexbin density colorbar
    assert len(plot.figure.get_axes()) == 3
    assert plot.result.feature_detection_rate.shape == (5,)
    plt.close(plot.figure)
    assert plot.legend_figure is not None
    plt.close(plot.legend_figure)


def test_plot_colors_classes_via_registry(registry: Path) -> None:
    plot = ms.plot_missingness(
        _class_dataset(), color_by="klass", registry_path=registry
    )
    assert set(plot.color_map) == {"exp", "ctrl"}
    assert plot.color_map["exp"] in _PALETTE and plot.color_map["ctrl"] in _PALETTE
    assert plot.color_map["exp"] != plot.color_map["ctrl"]
    saved = col.load_registry(registry)
    assert "klass" in saved
    plt.close(plot.figure)
    assert plot.legend_figure is not None
    plt.close(plot.legend_figure)


def test_plot_uncolored_single_curve_no_legend(registry: Path) -> None:
    plot = ms.plot_missingness(_class_dataset(), registry_path=registry)
    assert plot.color_map == {}
    assert plot.legend_figure is None
    plt.close(plot.figure)


def test_plot_bad_color_by_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="not a metadata column"):
        ms.plot_missingness(_class_dataset(), color_by="nope", registry_path=registry)


def test_plot_color_by_missing_values_raises(registry: Path) -> None:
    ds = _dataset(
        np.array([[1.0, 0.0], [1.0, 1.0]]),
        sample_ids=["s0", "s1"],
        meta={"klass": ["exp", None]},
    )
    with pytest.raises(ValueError, match="missing value"):
        ms.plot_missingness(ds, color_by="klass", registry_path=registry)


def test_plot_non_linear_raises_no_leak(registry: Path) -> None:
    ds = _dataset(np.array([[1.0, 0.0], [1.0, 1.0]]), scale="log2")
    before = set(plt.get_fignums())
    with pytest.raises(ms.MissingnessScaleError):
        ms.plot_missingness(ds, registry_path=registry)
    assert set(plt.get_fignums()) == before


def test_plot_too_many_classes_raises_no_leak(registry: Path) -> None:
    ids = [f"s{i}" for i in range(9)]
    ab = np.ones((9, 3))
    ds = _dataset(ab, sample_ids=ids, meta={"klass": [f"g{i}" for i in range(9)]})
    before = set(plt.get_fignums())
    with pytest.raises(col.CategoricalPaletteExceededError):
        ms.plot_missingness(ds, color_by="klass", registry_path=registry)
    assert set(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# save_missingness
# --------------------------------------------------------------------------- #


def test_save_dual_export_with_legend(registry: Path, tmp_path: Path) -> None:
    arts = ms.save_missingness(
        _class_dataset(),
        tmp_path,
        "missingness",
        color_by="klass",
        registry_path=registry,
    )
    assert arts.png.exists() and arts.svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
    assert not plt.get_fignums()  # both figures closed by save_figure


def test_save_uncolored_writes_no_legend(registry: Path, tmp_path: Path) -> None:
    arts = ms.save_missingness(
        _class_dataset(), tmp_path, "miss_plain", registry_path=registry
    )
    assert arts.png.exists()
    assert arts.legend_png is None and arts.legend_svg is None


# --------------------------------------------------------------------------- #
# Smoke — real 5xFAD (git-ignored)
# --------------------------------------------------------------------------- #

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)


@_skip_no_data
def test_smoke_missingness_real_proteins(registry: Path, tmp_path: Path) -> None:
    """8829 features, 6162 complete, MNAR r~0.524, max per-sample missing ~0.218."""
    prot = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        order_by="RunOrder",
        numeric_columns=("RunOrder", "Technical Replicate"),
    )
    res = ms.compute_completeness(prot)
    assert res.feature_detection_rate.shape == (8829,)
    assert int((res.feature_detection_rate == 1).sum()) == 6162
    # MNAR present: low-abundance features detected less.
    assert res.mnar_correlation == pytest.approx(0.524, abs=0.005)
    assert float((1.0 - res.sample_detection_rate).max()) == pytest.approx(
        0.218, abs=0.001
    )

    plot = ms.plot_missingness(
        prot, color_by="SampleType", title="5xFAD completeness", registry_path=registry
    )
    assert plot.legend_figure is not None
    plt.close(plot.figure)
    plt.close(plot.legend_figure)

    arts = ms.save_missingness(
        prot,
        tmp_path,
        "proteins_missingness",
        color_by="SampleType",
        registry_path=registry,
    )
    assert arts.png.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
