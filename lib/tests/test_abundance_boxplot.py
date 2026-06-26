"""Tests for the abundance-boxplot template (lib/figures/abundance_boxplot.py).

Layers:
  * unit/compute — planted-truth on ``compute_sample_medians`` (hand-computed per-sample
    medians, NaN handling) plus the 2D guard;
  * unit/plot — the stacked-panel structure (one panel per state), per-sample boxes,
    position coloring, annotation stripes + registry coloring, the separate legend, the
    non-log scale warning, and the guards (empty input, sample mismatch, empty sample,
    bad annotation, >8-category overflow, figure-leak on error);
  * smoke — real 5xFAD proteins (git-ignored) reproducing the captured oracle: median
    normalization collapses the per-sample medians to a single value (raw medians spread
    ~1.04, normalized spread ~0); skips cleanly when the data is absent.
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
from figures import abundance_boxplot as ab
from figures import colors as col
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
    scale: dl.Scale = "log2",
    sample_ids: list[str] | None = None,
    meta: dict[str, list[object]] | None = None,
) -> dl.Dataset:
    """Build a Dataset (default log2, so no scale warning) around a matrix."""
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


def _three_states(n_samples: int = 10, n_features: int = 30) -> dict[str, dl.Dataset]:
    """Three log2 states (raw/normalized/batch-corrected) over the same samples."""
    rng = np.random.default_rng(3)
    ids = [f"s{i}" for i in range(n_samples)]
    meta: dict[str, list[object]] = {
        "batch": ["A" if i % 2 else "B" for i in range(n_samples)],
        "RunOrder": list(range(n_samples)),
    }
    raw = rng.normal(20.0, 2.0, size=(n_samples, n_features))
    normed = raw - raw.mean(axis=1, keepdims=True) + 20.0
    corrected = normed + rng.normal(0.0, 0.1, size=(n_samples, n_features))
    return {
        "Raw": _dataset(raw, sample_ids=ids, meta=meta),
        "Normalized": _dataset(normed, sample_ids=ids, meta=meta),
        "Batch-corrected": _dataset(corrected, sample_ids=ids, meta=meta),
    }


# --------------------------------------------------------------------------- #
# compute_sample_medians — planted truth + guards
# --------------------------------------------------------------------------- #


def test_compute_sample_medians_planted_values() -> None:
    """Hand-computed per-sample medians (median over features, per row)."""
    ab_ = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [5.0, 5.0, 5.0]])
    med = ab.compute_sample_medians(_dataset(ab_))
    assert med.tolist() == [2.0, 20.0, 5.0]


def test_compute_sample_medians_ignores_nan() -> None:
    """A NaN feature is dropped before the per-sample median (here -> 2.0)."""
    ab_ = np.array([[1.0, np.nan, 3.0], [4.0, 6.0, np.nan]])
    med = ab.compute_sample_medians(_dataset(ab_))
    assert med[0] == pytest.approx(2.0)
    assert med[1] == pytest.approx(5.0)


def test_compute_sample_medians_all_nan_is_nan() -> None:
    ab_ = np.array([[np.nan, np.nan], [1.0, 3.0]])
    med = ab.compute_sample_medians(_dataset(ab_))
    assert np.isnan(med[0])
    assert med[1] == pytest.approx(2.0)


def test_compute_sample_medians_requires_2d() -> None:
    ds = _dataset(np.array([[1.0, 2.0]]))
    ds.abundances = np.array([1.0, 2.0, 3.0])  # break the contract on purpose
    with pytest.raises(ValueError, match="2D"):
        ab.compute_sample_medians(ds)


# --------------------------------------------------------------------------- #
# plot_abundance_boxplots — structure, colors, legend, annotations
# --------------------------------------------------------------------------- #


def test_one_panel_per_state(registry: Path) -> None:
    plot = ab.plot_abundance_boxplots(_three_states(), registry_path=registry)
    assert isinstance(plot, ab.AbundanceBoxplotPlot)
    # Three panels, no annotation stripes -> three axes on the main figure.
    assert len(plot.figure.get_axes()) == 3
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_one_box_per_sample_per_panel(registry: Path) -> None:
    plot = ab.plot_abundance_boxplots(
        _three_states(n_samples=8), registry_path=registry
    )
    for panel in plot.figure.get_axes():
        assert len(panel.patches) == 8  # one box (PathPatch) per sample
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_boxes_colored_by_position(registry: Path) -> None:
    """First box takes the colormap's low end, last box the high end."""
    plot = ab.plot_abundance_boxplots(
        _three_states(n_samples=6), box_colormap="cool", registry_path=registry
    )
    panel = plot.figure.get_axes()[0]
    cmap = plt.get_cmap("cool")
    assert np.allclose(panel.patches[0].get_facecolor(), cmap(0.0))
    assert np.allclose(panel.patches[-1].get_facecolor(), cmap(1.0))
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_single_state_allowed(registry: Path) -> None:
    states = {"Normalized": _three_states()["Normalized"]}
    plot = ab.plot_abundance_boxplots(states, registry_path=registry)
    assert len(plot.figure.get_axes()) == 1
    assert list(plot.result.medians) == ["Normalized"]
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_result_records_medians_and_scales(registry: Path) -> None:
    plot = ab.plot_abundance_boxplots(_three_states(), registry_path=registry)
    assert set(plot.result.medians) == {"Raw", "Normalized", "Batch-corrected"}
    assert plot.result.medians["Raw"].shape == (10,)
    assert plot.result.scales == {
        "Raw": "log2",
        "Normalized": "log2",
        "Batch-corrected": "log2",
    }
    assert plot.result.sample_ids.tolist() == [f"s{i}" for i in range(10)]
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_ylabel_derived_from_scale(registry: Path) -> None:
    states = {
        "Raw": _dataset(np.random.default_rng(0).normal(20, 1, (6, 8)), scale="log2"),
        "VSN": _dataset(np.random.default_rng(1).normal(0, 1, (6, 8)), scale="glog2"),
    }
    plot = ab.plot_abundance_boxplots(
        states, feature_type="protein", registry_path=registry
    )
    panels = plot.figure.get_axes()
    assert panels[0].get_ylabel() == "log2 protein abundance"
    assert panels[1].get_ylabel() == "protein abundance (glog2 / VSN)"
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_categorical_annotation_stripe_and_colors(registry: Path) -> None:
    plot = ab.plot_abundance_boxplots(
        _three_states(),
        categorical_annotations=["batch"],
        registry_path=registry,
    )
    # 3 panels + 1 stripe axes.
    assert len(plot.figure.get_axes()) == 4
    assert "batch" in plot.color_maps
    # batch A/B took the first two palette colors (first-appearance order: s0 -> B).
    assert set(plot.color_maps["batch"]) == {"A", "B"}
    assert plot.color_maps["batch"]["B"] == _PALETTE[0]
    assert plot.color_maps["batch"]["A"] == _PALETTE[1]
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_continuous_annotation_stripe(registry: Path) -> None:
    plot = ab.plot_abundance_boxplots(
        _three_states(),
        continuous_annotations=["RunOrder"],
        registry_path=registry,
    )
    assert len(plot.figure.get_axes()) == 4  # 3 panels + 1 continuous stripe
    assert plot.color_maps == {}  # continuous dims are not registry-colored
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_legend_is_separate_figure(registry: Path) -> None:
    plot = ab.plot_abundance_boxplots(
        _three_states(), categorical_annotations=["batch"], registry_path=registry
    )
    assert isinstance(plot.legend_figure, Figure)
    assert plot.legend_figure is not plot.figure
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_persist_colors_false_does_not_write(registry: Path) -> None:
    plot = ab.plot_abundance_boxplots(
        _three_states(),
        categorical_annotations=["batch"],
        persist_colors=False,
        registry_path=registry,
    )
    assert "batch" not in json.loads(registry.read_text())
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


# --------------------------------------------------------------------------- #
# scale warning (warn, not refuse)
# --------------------------------------------------------------------------- #


def test_linear_scale_warns_but_still_plots(registry: Path) -> None:
    states = {
        "Raw (linear)": _dataset(
            np.random.default_rng(0).lognormal(8.0, 1.0, (8, 20)), scale="linear"
        )
    }
    with pytest.warns(ab.AbundanceBoxplotScaleWarning, match="non-log scale"):
        plot = ab.plot_abundance_boxplots(states, registry_path=registry)
    assert len(plot.figure.get_axes()) == 1  # produced anyway
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


# --------------------------------------------------------------------------- #
# fail loud + figure-leak safety
# --------------------------------------------------------------------------- #


def test_empty_datasets_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        ab.plot_abundance_boxplots({}, registry_path=registry)


def test_sample_mismatch_raises(registry: Path) -> None:
    rng = np.random.default_rng(0)
    a = _dataset(rng.normal(20, 1, (6, 8)), sample_ids=[f"s{i}" for i in range(6)])
    b = _dataset(rng.normal(20, 1, (6, 8)), sample_ids=[f"x{i}" for i in range(6)])
    with pytest.raises(ValueError, match="different samples"):
        ab.plot_abundance_boxplots({"A": a, "B": b}, registry_path=registry)


def test_empty_sample_raises(registry: Path) -> None:
    ab_ = np.random.default_rng(0).normal(20, 1, (4, 8))
    ab_[2, :] = np.nan  # sample s2 has no finite features
    with pytest.raises(ValueError, match="no finite feature"):
        ab.plot_abundance_boxplots({"Raw": _dataset(ab_)}, registry_path=registry)


def test_missing_annotation_column_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="not a metadata column"):
        ab.plot_abundance_boxplots(
            _three_states(), categorical_annotations=["nope"], registry_path=registry
        )


def test_annotation_in_both_lists_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="both"):
        ab.plot_abundance_boxplots(
            _three_states(),
            categorical_annotations=["batch"],
            continuous_annotations=["batch"],
            registry_path=registry,
        )


def _states_with_nine_batches() -> dict[str, dl.Dataset]:
    """One state whose batch annotation has nine distinct values (>8 guard)."""
    n = 9
    ids = [f"s{i}" for i in range(n)]
    meta: dict[str, list[object]] = {"batch9": [f"b{i}" for i in range(n)]}
    ab_ = np.random.default_rng(0).normal(20, 1, (n, 12))
    return {"Raw": _dataset(ab_, sample_ids=ids, meta=meta)}


def test_more_than_eight_categories_raises(registry: Path) -> None:
    with pytest.raises(col.CategoricalPaletteExceededError):
        ab.plot_abundance_boxplots(
            _states_with_nine_batches(),
            categorical_annotations=["batch9"],
            registry_path=registry,
        )


def test_error_path_closes_figure(registry: Path) -> None:
    """The >8 guard fires after the figure is built; it must not leak the figure."""
    before = set(plt.get_fignums())
    with pytest.raises(col.CategoricalPaletteExceededError):
        ab.plot_abundance_boxplots(
            _states_with_nine_batches(),
            categorical_annotations=["batch9"],
            registry_path=registry,
        )
    assert set(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# save_abundance_boxplots — figure + separate legend image
# --------------------------------------------------------------------------- #


def test_save_writes_figure_and_legend_image(registry: Path, tmp_path: Path) -> None:
    arts = ab.save_abundance_boxplots(
        _three_states(),
        tmp_path,
        "abundance_by_state",
        categorical_annotations=["batch"],
        registry_path=registry,
    )
    assert arts.svg.exists() and arts.png.exists()
    assert arts.legend_svg is not None and arts.legend_svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
    assert arts.legend_svg == tmp_path / "abundance_by_state.legend.svg"


def test_save_bad_base_name_closes_both_figures(registry: Path, tmp_path: Path) -> None:
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="bare filename stem"):
        ab.save_abundance_boxplots(
            _three_states(), tmp_path, "bad/name", registry_path=registry
        )
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
def test_smoke_abundance_boxplot_real_proteins(registry: Path, tmp_path: Path) -> None:
    """Median normalization collapses the per-sample medians (captured oracle)."""
    from common import normalize as norm

    raw = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=dl.ReplicateCollapse("Sample ID", "Technical Replicate"),
        order_by="RunOrder",
        numeric_columns=("RunOrder", "Technical Replicate"),
    )
    raw_log = norm.log2_transform(raw)
    normed = norm.log2_transform(norm.normalize(raw, "median"))
    assert raw_log.scale == "log2" and normed.scale == "log2"

    states = {"Raw (log2)": raw_log, "Median-normalized": normed}
    plot = ab.plot_abundance_boxplots(
        states, feature_type="protein", registry_path=registry
    )

    med_raw = plot.result.medians["Raw (log2)"]
    med_norm = plot.result.medians["Median-normalized"]
    # Oracle: raw per-sample medians wander (std ~1.04); median normalization collapses
    # them to a single common value (std ~0, every sample at ~20.6033).
    assert float(np.nanstd(med_raw)) == pytest.approx(1.0424, abs=0.02)
    assert float(np.nanstd(med_norm)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.nanmin(med_norm)) == pytest.approx(20.6033, abs=0.01)
    # The QC truth: normalization flattens the per-sample medians.
    assert float(np.nanstd(med_norm)) < float(np.nanstd(med_raw))

    assert len(plot.figure.get_axes()) == 2
    plt.close(plot.figure)
    plt.close(plot.legend_figure)

    arts = ab.save_abundance_boxplots(
        states,
        tmp_path,
        "proteins_abundance_boxplot",
        categorical_annotations=["SampleType"],
        registry_path=registry,
    )
    assert arts.png.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
