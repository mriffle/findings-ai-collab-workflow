"""Tests for the dynamic-range template (lib/figures/dynamic_range.py).

Layers:
  * unit/compute — planted-truth on ``compute_dynamic_range`` (hand-computed ranking,
    log2 median / IQR, dynamic-range orders, never-detected exclusion), the non-linear
    scale refuse, and shape guards;
  * unit/plot — whole-cohort default, single-color + grouped highlights (+ the separate
    legend), per-class overlay, and every guard (class_by vs highlights exclusion,
    unknown / never-detected highlight, incomplete groups, >8 overflow, figure-leak);
  * smoke — real 5xFAD proteins (git-ignored) reproducing the oracle: 8829 features, 7.2
    orders of dynamic range, albumin contaminants at ranks 1-2, the 5xFAD APP transgene
    at rank 432. Skips cleanly when the data is absent.
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
from figures import dynamic_range as dr

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
    feature_names: list[str] | None = None,
    meta: dict[str, list[object]] | None = None,
) -> dl.Dataset:
    """Build a linear-scale Dataset (dynamic-range refuses non-linear)."""
    n_samples, n_features = abundances.shape
    names = feature_names or [f"F{j}" for j in range(n_features)]
    ids = [f"s{i}" for i in range(n_samples)]
    columns = {} if meta is None else dict(meta)
    metadata = pd.DataFrame(columns, index=pd.Index(ids, name="sample"))
    return dl.Dataset(
        abundances=np.asarray(abundances, dtype=float),
        feature_names=np.array(names, dtype=str),
        feature_metadata=pd.DataFrame({"protein": names}),
        metadata=metadata,
        scale=scale,
    )


# --------------------------------------------------------------------------- #
# compute_dynamic_range — planted truth + guards
# --------------------------------------------------------------------------- #


def test_compute_planted_ranking() -> None:
    """Rank by median of detected; hand-computed log2 median / IQR / orders."""
    ab = np.array([[10.0, 0.0, 5.0], [20.0, 0.0, 0.0], [30.0, 4.0, 0.0]])
    # F0 detected [10,20,30] median 20; F1 [4] median 4; F2 [5] median 5 -> F0>F2>F1
    res = dr.compute_dynamic_range(_dataset(ab))
    assert res.feature_names_ranked.tolist() == ["F0", "F2", "F1"]
    assert res.log2_median.tolist() == pytest.approx(
        [np.log2(20.0), np.log2(5.0), np.log2(4.0)]
    )
    assert res.log2_q25[0] == pytest.approx(np.log2(15.0))  # q25 of [10,20,30]
    assert res.log2_q75[0] == pytest.approx(np.log2(25.0))
    assert res.dynamic_range_orders == pytest.approx(np.log10(20.0 / 4.0))
    assert res.n_features_total == 3 and res.n_features_detected == 3
    assert res.ranks.tolist() == [1, 2, 3]


def test_compute_excludes_never_detected() -> None:
    """An all-zero feature has no abundance and is left off the curve."""
    ab = np.array([[10.0, 0.0, 5.0], [20.0, 0.0, 7.0]])  # F1 never detected
    res = dr.compute_dynamic_range(_dataset(ab))
    assert res.n_features_total == 3 and res.n_features_detected == 2
    assert "F1" not in res.feature_names_ranked.tolist()


def test_compute_refuses_non_linear_scale() -> None:
    with pytest.raises(dr.DynamicRangeScaleError, match="linear"):
        dr.compute_dynamic_range(
            _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]), scale="log2")
        )


def test_compute_requires_2d() -> None:
    bad = dl.Dataset(
        abundances=np.array([1.0, 2.0, 3.0]),
        feature_names=np.array(["F0", "F1", "F2"], dtype=str),
        feature_metadata=pd.DataFrame({"protein": ["F0", "F1", "F2"]}),
        metadata=pd.DataFrame(index=pd.Index(["s0"], name="sample")),
        scale="linear",
    )
    with pytest.raises(ValueError, match="2D"):
        dr.compute_dynamic_range(bad)


def test_compute_all_undetected_raises() -> None:
    with pytest.raises(ValueError, match="no feature is detected"):
        dr.compute_dynamic_range(_dataset(np.zeros((3, 4))))


# --------------------------------------------------------------------------- #
# plot_dynamic_range — modes, highlights, guards
# --------------------------------------------------------------------------- #


def _demo() -> dl.Dataset:
    """Eight features of decreasing abundance over six samples, two classes."""
    base = np.array([1e6, 3e5, 1e5, 3e4, 1e4, 3e3, 1e3, 3e2])
    rng = np.random.default_rng(0)
    ab = base[None, :] * rng.uniform(0.8, 1.2, size=(6, base.size))
    return _dataset(ab, meta={"klass": ["exp", "exp", "exp", "exp", "ctrl", "ctrl"]})


def test_plot_whole_cohort_default(registry: Path) -> None:
    plot = dr.plot_dynamic_range(_demo(), registry_path=registry)
    assert plot.color_map == {}
    assert plot.legend_figure is None
    assert plot.result.n_features_detected == 8
    plt.close(plot.figure)


def test_plot_single_color_highlights(registry: Path) -> None:
    plot = dr.plot_dynamic_range(
        _demo(), highlight_features={"F0": "top", "F5": "low"}, registry_path=registry
    )
    assert plot.color_map == {}  # no groups -> single color, no legend
    assert plot.legend_figure is None
    plt.close(plot.figure)


def test_plot_grouped_highlights_build_legend(registry: Path) -> None:
    plot = dr.plot_dynamic_range(
        _demo(),
        highlight_features={"F0": "albumin", "F6": "APP"},
        highlight_groups={"F0": "contaminant", "F6": "of interest"},
        registry_path=registry,
    )
    assert set(plot.color_map) == {"contaminant", "of interest"}
    assert plot.legend_figure is not None
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_highlight_unknown_feature_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="not in the Dataset"):
        dr.plot_dynamic_range(
            _demo(), highlight_features={"NOPE": "x"}, registry_path=registry
        )


def test_plot_highlight_never_detected_raises(registry: Path) -> None:
    ab = np.array([[10.0, 0.0], [20.0, 0.0]])  # F1 never detected
    ds = _dataset(ab)
    with pytest.raises(ValueError, match="never detected"):
        dr.plot_dynamic_range(
            ds, highlight_features={"F1": "x"}, registry_path=registry
        )


def test_plot_incomplete_highlight_groups_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="missing group"):
        dr.plot_dynamic_range(
            _demo(),
            highlight_features={"F0": "a", "F1": "b"},
            highlight_groups={"F0": "g"},  # F1 ungrouped
            registry_path=registry,
        )


def test_plot_class_by_overlay(registry: Path) -> None:
    plot = dr.plot_dynamic_range(_demo(), class_by="klass", registry_path=registry)
    assert set(plot.color_map) == {"exp", "ctrl"}
    assert plot.legend_figure is not None
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_class_by_and_highlights_mutually_exclusive(registry: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        dr.plot_dynamic_range(
            _demo(),
            class_by="klass",
            highlight_features={"F0": "x"},
            registry_path=registry,
        )


def test_plot_class_by_missing_column_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="not a metadata column"):
        dr.plot_dynamic_range(_demo(), class_by="nope", registry_path=registry)


def test_plot_too_many_classes_raises_no_leak(registry: Path) -> None:
    ab = np.ones((9, 4)) * np.array([1e6, 1e5, 1e4, 1e3])
    ds = _dataset(ab, meta={"klass": [f"g{i}" for i in range(9)]})
    before = set(plt.get_fignums())
    with pytest.raises(col.CategoricalPaletteExceededError):
        dr.plot_dynamic_range(ds, class_by="klass", registry_path=registry)
    assert set(plt.get_fignums()) == before


def test_plot_non_linear_raises_no_leak(registry: Path) -> None:
    ds = _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]), scale="log2")
    before = set(plt.get_fignums())
    with pytest.raises(dr.DynamicRangeScaleError):
        dr.plot_dynamic_range(ds, registry_path=registry)
    assert set(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# save_dynamic_range
# --------------------------------------------------------------------------- #


def test_save_per_class_dual_export_with_legend(registry: Path, tmp_path: Path) -> None:
    arts = dr.save_dynamic_range(
        _demo(), tmp_path, "dynrange", class_by="klass", registry_path=registry
    )
    assert arts.png.exists() and arts.svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
    assert not plt.get_fignums()


def test_save_plain_writes_no_legend(registry: Path, tmp_path: Path) -> None:
    arts = dr.save_dynamic_range(
        _demo(), tmp_path, "dynrange_plain", registry_path=registry
    )
    assert arts.png.exists()
    assert arts.legend_png is None


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
def test_smoke_dynamic_range_real_proteins(registry: Path, tmp_path: Path) -> None:
    """8829 features, 7.2 orders, albumin at ranks 1-2, APP transgene at rank 432."""
    prot = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        order_by="RunOrder",
        numeric_columns=("RunOrder", "Technical Replicate"),
    )
    res = dr.compute_dynamic_range(prot)
    assert res.n_features_detected == 8829
    assert res.dynamic_range_orders == pytest.approx(7.2, abs=0.1)
    assert res.feature_names_ranked[0] == "crapola_crap|ALBU_BOVIN|ALBU_BOVIN"
    assert res.feature_names_ranked[1] == "crapola_crap|ALBU_HUMAN|ALBU_HUMAN"
    ranked = res.feature_names_ranked.tolist()
    assert ranked.index("sp|P05067|5xFADA4_HUMAN") + 1 == 432

    plot = dr.plot_dynamic_range(
        prot,
        highlight_features={
            "crapola_crap|ALBU_BOVIN|ALBU_BOVIN": "Albumin",
            "sp|P05067|5xFADA4_HUMAN": "APP (5xFAD)",
        },
        highlight_groups={
            "crapola_crap|ALBU_BOVIN|ALBU_BOVIN": "contaminant",
            "sp|P05067|5xFADA4_HUMAN": "of interest",
        },
        title="5xFAD dynamic range",
        registry_path=registry,
    )
    assert plot.legend_figure is not None
    plt.close(plot.figure)
    plt.close(plot.legend_figure)

    arts = dr.save_dynamic_range(
        prot, tmp_path, "proteins_dynrange", registry_path=registry
    )
    assert arts.png.exists()
