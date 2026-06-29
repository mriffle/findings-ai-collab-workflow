"""Tests for the identification-depth template (lib/figures/id_depth.py).

Layers:
  * unit/compute — planted-truth on ``compute_detection_counts`` (hand-computed detected
    counts, the ``min_intensity`` threshold, NaN/zero handling), plus the non-linear
    scale refuse and the 2D guard;
  * unit/plot — the stacked-panel structure (one panel per level), bar coloring via the
    registry, the reference-median line (all-sample vs subset), the uncolored path
    (no legend), and the guards (empty input, sample mismatch, bad ``color_by`` /
    missing values, bad ``reference_mask``, >8-category overflow, figure-leak on error);
  * smoke — real 5xFAD (git-ignored) reproducing the captured oracle: 90 runs, the
    experimental-protein median depth is 8502 and a low run sinks to 6904; the slow
    variant adds the precursor panel (exp-median 86712, min 49944). Skips cleanly when
    the data is absent.
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
from figures import id_depth as idp

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
    """Build a linear-scale Dataset (id-depth refuses non-linear) around a matrix."""
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
# compute_detection_counts — planted truth + guards
# --------------------------------------------------------------------------- #


def test_compute_counts_planted_values() -> None:
    """Hand-counted detections (finite & > 0): zeros and NaN are not detected."""
    ab = np.array(
        [
            [5.0, 0.0, 3.0, np.nan],  # detected: 5, 3            -> 2
            [0.0, 0.0, 0.0, 2.0],  # detected: 2                  -> 1
            [1.0, 1.0, 1.0, 1.0],  # detected: all four           -> 4
        ]
    )
    counts = idp.compute_detection_counts(_dataset(ab))
    assert counts.tolist() == [2, 1, 4]
    assert counts.dtype == np.dtype(int)


def test_compute_counts_threshold() -> None:
    """``min_intensity`` raises the detection floor (strictly greater than)."""
    ab = np.array([[5.0, 0.0, 3.0, np.nan], [0.0, 0.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    counts = idp.compute_detection_counts(_dataset(ab), min_intensity=1.0)
    assert counts.tolist() == [2, 1, 0]  # only 5,3 / 2 / nothing exceed 1.0


def test_compute_counts_refuses_non_linear_scale() -> None:
    ab = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(idp.IdDepthScaleError, match="linear-scale"):
        idp.compute_detection_counts(_dataset(ab, scale="log2"))


def test_compute_counts_requires_2d() -> None:
    bad = dl.Dataset(
        abundances=np.array([1.0, 2.0, 3.0]),
        feature_names=np.array(["F0", "F1", "F2"], dtype=str),
        feature_metadata=pd.DataFrame({"protein": ["F0", "F1", "F2"]}),
        metadata=pd.DataFrame(index=pd.Index(["s0"], name="sample")),
        scale="linear",
    )
    with pytest.raises(ValueError, match="2D"):
        idp.compute_detection_counts(bad)


# --------------------------------------------------------------------------- #
# _compute_result — reference median (all samples vs a subset)
# --------------------------------------------------------------------------- #


def test_reference_median_all_vs_subset() -> None:
    """The reference line is the all-sample median, or a subset's when masked."""
    ab = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],  # 2 detected
            [1.0, 0.0, 0.0, 0.0],  # 1 detected
            [1.0, 1.0, 1.0, 1.0],  # 4 detected
        ]
    )
    datasets = {"Protein": _dataset(ab)}

    plot_all = idp.plot_id_depth(datasets, show_reference_line=False)
    assert plot_all.result.counts["Protein"].tolist() == [2, 1, 4]
    assert plot_all.result.reference_median["Protein"] == pytest.approx(
        2.0
    )  # median of all
    assert plot_all.result.reference_is_subset is False
    plt.close(plot_all.figure)

    mask = np.array([True, False, True])  # keep samples 0 (2) and 2 (4) -> median 3
    plot_sub = idp.plot_id_depth(
        datasets, reference_mask=mask, show_reference_line=False
    )
    assert plot_sub.result.reference_median["Protein"] == pytest.approx(3.0)
    assert plot_sub.result.reference_is_subset is True
    plt.close(plot_sub.figure)


# --------------------------------------------------------------------------- #
# plot_id_depth — structure, color, legend
# --------------------------------------------------------------------------- #


def _two_levels() -> dict[str, dl.Dataset]:
    """Two levels (different feature counts) over the same four samples."""
    ids = ["s0", "s1", "s2", "s3"]
    meta: dict[str, list[object]] = {"klass": ["exp", "exp", "ctrl", "exp"]}
    prot = _dataset(
        np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        sample_ids=ids,
        meta=meta,
    )
    prec = _dataset(
        np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        sample_ids=ids,
        meta=meta,
    )
    return {"Protein": prot, "Precursor": prec}


def test_plot_stacks_one_panel_per_level(registry: Path) -> None:
    plot = idp.plot_id_depth(_two_levels(), registry_path=registry)
    assert len(plot.figure.get_axes()) == 2
    assert list(plot.result.counts.keys()) == ["Protein", "Precursor"]
    plt.close(plot.figure)


def test_plot_colors_bars_via_registry(registry: Path) -> None:
    """``color_by`` colors bars through the registry and builds a swatch legend."""
    plot = idp.plot_id_depth(_two_levels(), color_by="klass", registry_path=registry)
    assert set(plot.color_map) == {"exp", "ctrl"}
    assert plot.color_map["exp"] in _PALETTE and plot.color_map["ctrl"] in _PALETTE
    assert plot.color_map["exp"] != plot.color_map["ctrl"]
    assert plot.legend_figure is not None
    # Persisted to the registry under the color_by namespace.
    saved = col.load_registry(registry)
    assert "klass" in saved
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_plot_uncolored_has_no_legend(registry: Path) -> None:
    """No ``color_by`` -> uniform bars, empty color map, and no legend figure."""
    plot = idp.plot_id_depth(_two_levels(), registry_path=registry)
    assert plot.color_map == {}
    assert plot.legend_figure is None
    plt.close(plot.figure)


def test_plot_single_level_allowed(registry: Path) -> None:
    one = {"Protein": _two_levels()["Protein"]}
    plot = idp.plot_id_depth(one, color_by="klass", registry_path=registry)
    assert len(plot.figure.get_axes()) == 1
    plt.close(plot.figure)
    assert plot.legend_figure is not None
    plt.close(plot.legend_figure)


# --------------------------------------------------------------------------- #
# plot_id_depth — guards (fail loud, no figure leak)
# --------------------------------------------------------------------------- #


def test_plot_empty_datasets_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        idp.plot_id_depth({})


def test_plot_sample_mismatch_raises(registry: Path) -> None:
    a = _dataset(np.array([[1.0, 0.0], [1.0, 1.0]]), sample_ids=["s0", "s1"])
    b = _dataset(np.array([[1.0, 0.0], [1.0, 1.0]]), sample_ids=["s0", "sX"])
    with pytest.raises(ValueError, match="different samples"):
        idp.plot_id_depth({"Protein": a, "Precursor": b}, registry_path=registry)


def test_plot_non_linear_scale_raises_no_leak(registry: Path) -> None:
    ds = _dataset(np.array([[1.0, 0.0], [1.0, 1.0]]), scale="log2")
    before = set(plt.get_fignums())
    with pytest.raises(idp.IdDepthScaleError):
        idp.plot_id_depth({"Protein": ds}, registry_path=registry)
    assert set(plt.get_fignums()) == before  # computed before the figure exists


def test_plot_bad_color_by_column_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="not a metadata column"):
        idp.plot_id_depth(_two_levels(), color_by="nope", registry_path=registry)


def test_plot_color_by_missing_values_raises(registry: Path) -> None:
    ds = _dataset(
        np.array([[1.0, 0.0], [1.0, 1.0]]),
        sample_ids=["s0", "s1"],
        meta={"klass": ["exp", None]},
    )
    with pytest.raises(ValueError, match="missing value"):
        idp.plot_id_depth({"Protein": ds}, color_by="klass", registry_path=registry)


def test_plot_bad_reference_mask_length_raises(registry: Path) -> None:
    with pytest.raises(ValueError, match="length n_samples"):
        idp.plot_id_depth(
            _two_levels(),
            reference_mask=np.array([True, False]),
            registry_path=registry,
        )


def test_plot_too_many_categories_raises_no_leak(registry: Path) -> None:
    """>8 color_by categories trip the registry guard with no leaked figure."""
    ids = [f"s{i}" for i in range(9)]
    ab = np.ones((9, 3))
    ds = _dataset(ab, sample_ids=ids, meta={"klass": [f"g{i}" for i in range(9)]})
    before = set(plt.get_fignums())
    with pytest.raises(col.CategoricalPaletteExceededError):
        idp.plot_id_depth({"Protein": ds}, color_by="klass", registry_path=registry)
    assert set(plt.get_fignums()) == before  # figure closed on the error path


# --------------------------------------------------------------------------- #
# save_id_depth
# --------------------------------------------------------------------------- #


def test_save_id_depth_dual_export_with_legend(registry: Path, tmp_path: Path) -> None:
    arts = idp.save_id_depth(
        _two_levels(), tmp_path, "id_depth", color_by="klass", registry_path=registry
    )
    assert arts.png.exists() and arts.svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()
    assert arts.legend_svg is not None and arts.legend_svg.exists()
    assert not plt.get_fignums()  # both figures closed by save_figure


def test_save_id_depth_uncolored_writes_no_legend(
    registry: Path, tmp_path: Path
) -> None:
    arts = idp.save_id_depth(
        _two_levels(), tmp_path, "id_depth_plain", registry_path=registry
    )
    assert arts.png.exists()
    assert arts.legend_png is None and arts.legend_svg is None


# --------------------------------------------------------------------------- #
# Smoke — real 5xFAD (git-ignored)
# --------------------------------------------------------------------------- #

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_PREC = _TESTDATA / "data" / "precursors_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)


def _load_proteins() -> dl.Dataset:
    return dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        order_by="RunOrder",
        numeric_columns=("RunOrder", "Technical Replicate"),
    )


@_skip_no_data
def test_smoke_id_depth_real_proteins(registry: Path, tmp_path: Path) -> None:
    """90 runs; experimental protein-depth median 8502, a low run sinks to 6904."""
    prot = _load_proteins()
    counts = idp.compute_detection_counts(prot)
    assert counts.shape == (90,)
    assert int(counts.min()) == 6904  # the low run this plot exists to surface
    assert int(counts.max()) == 8701

    exp_mask = prot.metadata["SampleType"].to_numpy().astype(str) == "unknown"
    assert int(exp_mask.sum()) == 74
    plot = idp.plot_id_depth(
        {"Protein": prot},
        color_by="SampleType",
        reference_mask=exp_mask,
        registry_path=registry,
    )
    # Oracle: experimental-sample median depth.
    assert plot.result.reference_median["Protein"] == pytest.approx(8502.0)
    assert plot.result.reference_is_subset is True
    plt.close(plot.figure)
    assert plot.legend_figure is not None
    plt.close(plot.legend_figure)

    arts = idp.save_id_depth(
        {"Protein": prot},
        tmp_path,
        "proteins_id_depth",
        color_by="SampleType",
        reference_mask=exp_mask,
        registry_path=registry,
    )
    assert arts.png.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()


@pytest.mark.slow
@_skip_no_data
def test_smoke_id_depth_protein_and_precursor(registry: Path, tmp_path: Path) -> None:
    """Two-panel protein+precursor depth over the shared 90 runs (precursor oracle)."""
    if not _PREC.exists():
        pytest.skip("precursor testdata not present")
    prot = _load_proteins()
    prec = dl.load_wide_data(
        _PREC,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        id_columns=("protein", "modifiedSequence", "precursorCharge"),
        order_by="RunOrder",
        numeric_columns=("RunOrder", "Technical Replicate"),
        # Precursor rows are unique by (protein, sequence, charge); the single-column
        # feature-id guard would false-trip on the repeated protein. A detection count
        # is per-row, so row-id uniqueness is irrelevant here.
        require_unique_features=False,
    )
    prec_counts = idp.compute_detection_counts(prec)
    assert prec_counts.shape == (90,)
    assert int(prec_counts.min()) == 49944
    assert int(prec_counts.max()) == 92201

    exp_mask = prot.metadata["SampleType"].to_numpy().astype(str) == "unknown"
    plot = idp.plot_id_depth(
        {"Protein": prot, "Precursor": prec},
        color_by="SampleType",
        reference_mask=exp_mask,
        registry_path=registry,
    )
    assert len(plot.figure.get_axes()) == 2
    assert plot.result.reference_median["Precursor"] == pytest.approx(86712.0)
    plt.close(plot.figure)
    if plot.legend_figure is not None:
        plt.close(plot.legend_figure)
