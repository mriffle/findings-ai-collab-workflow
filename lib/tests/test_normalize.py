"""Tests for the normalization template (lib/common/normalize.py).

Two layers:
  * unit — hand-built synthetic fixtures with planted-truth checks (the median identity,
    the MAD per-row-median-is-0 invariant, log2(x+1) on zeros) and the scale-tag guards
    that prevent the double-log trap;
  * smoke — the real 5xFAD data under testdata/ (git-ignored); skips cleanly if absent.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common import data_loading as dl
from common import normalize as norm


def _ds(abundances: object, scale: dl.Scale = "linear") -> dl.Dataset:
    """Build a minimal Dataset around an abundance matrix for normalization tests."""
    ab = np.asarray(abundances, dtype=float)
    n_samples, n_features = ab.shape
    feature_names = np.array([f"F{j}" for j in range(n_features)], dtype=str)
    feature_metadata = pd.DataFrame({"protein": feature_names})
    metadata = pd.DataFrame({"sample": [f"s{i}" for i in range(n_samples)]}).set_index(
        "sample"
    )
    return dl.Dataset(
        abundances=ab,
        feature_names=feature_names,
        feature_metadata=feature_metadata,
        metadata=metadata,
        scale=scale,
    )


# --------------------------------------------------------------------------- #
# Unit — median (planted truth + the defining invariant)
# --------------------------------------------------------------------------- #


def test_median_planted_truth() -> None:
    """Two samples with medians 2 and 20; mean-of-medians 11. Each row -> row/median*11,
    so both collapse to [5.5, 11, 16.5]."""
    ds = _ds([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = norm.normalize(ds, "median")
    expected = np.array([[5.5, 11.0, 16.5], [5.5, 11.0, 16.5]])
    np.testing.assert_allclose(out.abundances, expected)
    assert out.scale == "linear"  # median is linear in -> linear out


def test_median_equalizes_row_medians() -> None:
    """Defining property: after median normalization every sample shares one median."""
    rng = np.random.default_rng(0)
    ds = _ds(rng.uniform(1.0, 1000.0, size=(5, 40)))
    out = norm.normalize(ds, "median")
    row_medians = np.median(out.abundances, axis=1)
    np.testing.assert_allclose(row_medians, row_medians[0])


def test_normalize_preserves_shape_and_passes_metadata_through() -> None:
    ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = norm.normalize(ds, "median")
    assert out.abundances.shape == ds.abundances.shape
    assert list(out.feature_names) == list(ds.feature_names)
    pd.testing.assert_frame_equal(out.metadata, ds.metadata)
    pd.testing.assert_frame_equal(out.feature_metadata, ds.feature_metadata)


# --------------------------------------------------------------------------- #
# Unit — mad and vsn (output scale + invariants)
# --------------------------------------------------------------------------- #


def test_mad_centers_each_row_and_tags_zscore() -> None:
    rng = np.random.default_rng(1)
    ds = _ds(rng.uniform(0.0, 500.0, size=(4, 60)))
    out = norm.normalize(ds, "mad")
    assert out.scale == "zscore"
    # robust z-score: median of each normalized row is exactly 0
    np.testing.assert_allclose(np.median(out.abundances, axis=1), 0.0, atol=1e-9)


def test_mad_emits_no_deprecation_warning() -> None:
    """We pass scale_to_sigma=True explicitly, so pronoms must not warn."""
    ds = _ds(np.arange(1.0, 41.0).reshape(4, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        norm.normalize(ds, "mad")


def test_vsn_tags_glog2_and_preserves_shape() -> None:
    rng = np.random.default_rng(2)
    ds = _ds(rng.uniform(1.0, 1000.0, size=(6, 50)))
    out = norm.normalize(ds, "vsn")
    assert out.scale == "glog2"
    assert out.abundances.shape == ds.abundances.shape
    assert np.isfinite(out.abundances).all()


# --------------------------------------------------------------------------- #
# Unit — log2 transform
# --------------------------------------------------------------------------- #


def test_log2_planted_truth_zeros_preserved() -> None:
    ds = _ds([[0.0, 1.0, 3.0, 7.0]])
    out = norm.log2_transform(ds)
    np.testing.assert_allclose(out.abundances, [[0.0, 1.0, 2.0, 3.0]])
    assert out.scale == "log2"


def test_median_then_log2_chains_to_log2_scale() -> None:
    ds = _ds([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = norm.log2_transform(norm.normalize(ds, "median"))
    assert out.scale == "log2"


def test_median_center_planted_truth() -> None:
    """Log-domain median normalization: subtract each sample's median; scale kept."""
    ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], scale="log2")
    out = norm.median_center(ds)
    np.testing.assert_allclose(out.abundances, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    assert out.scale == "log2"


def test_median_center_refuses_linear() -> None:
    ds = _ds([[1.0, 2.0, 3.0]])  # linear
    with pytest.raises(ValueError, match="requires a log scale"):
        norm.median_center(ds)


def test_normalize_returns_independent_dataset() -> None:
    """Output must be independent: mutating its metadata must not touch the input."""
    ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = norm.normalize(ds, "median")
    assert out.metadata is not ds.metadata
    out.metadata["injected"] = 1
    assert "injected" not in ds.metadata.columns


# --------------------------------------------------------------------------- #
# Unit — the scale guards (double-log trap) and fail-loud edges
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["median", "mad", "vsn"])
def test_normalize_refuses_non_linear_input(method: norm.NormalizationMethod) -> None:
    ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], scale="log2")
    with pytest.raises(ValueError, match="requires linear-scale"):
        norm.normalize(ds, method)


def test_log2_refuses_non_linear_input() -> None:
    ds = _ds([[1.0, 2.0, 3.0]], scale="glog2")
    with pytest.raises(ValueError, match="requires linear-scale"):
        norm.log2_transform(ds)


def test_log2_then_normalize_is_blocked() -> None:
    """The real chain that would double-log: log2 first, then a log-producing method."""
    ds = _ds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logged = norm.log2_transform(ds)
    with pytest.raises(ValueError, match="requires linear-scale"):
        norm.normalize(logged, "mad")


def test_normalize_refuses_nan() -> None:
    ds = _ds([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="non-finite"):
        norm.normalize(ds, "median")


def test_log2_refuses_negative() -> None:
    ds = _ds([[1.0, -2.0, 3.0]])
    with pytest.raises(ValueError, match="non-negative"):
        norm.log2_transform(ds)


def test_unknown_method_raises() -> None:
    ds = _ds([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="Unknown normalization method"):
        norm.normalize(ds, "zscore")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Smoke — real 5xFAD data (git-ignored)
# --------------------------------------------------------------------------- #

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"
_COLLAPSE = dl.ReplicateCollapse("Sample ID", "Technical Replicate")
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)


def _load_real() -> dl.Dataset:
    return dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=_COLLAPSE,
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )


@_skip_no_data
def test_smoke_median_then_log2_real_data() -> None:
    ds = _load_real()
    med = norm.normalize(ds, "median")
    assert med.scale == "linear"
    assert med.abundances.shape == ds.abundances.shape
    # every sample shares one median after median normalization
    row_medians = np.median(med.abundances, axis=1)
    np.testing.assert_allclose(row_medians, row_medians[0])
    logged = norm.log2_transform(med)
    assert logged.scale == "log2"
    assert np.isfinite(logged.abundances).all()
    assert int((logged.abundances == 0).sum()) > 0  # not-detected zeros stay 0


@_skip_no_data
def test_smoke_mad_real_data() -> None:
    ds = _load_real()
    out = norm.normalize(ds, "mad")
    assert out.scale == "zscore"
    np.testing.assert_allclose(np.median(out.abundances, axis=1), 0.0, atol=1e-9)


@_skip_no_data
@pytest.mark.slow
def test_smoke_vsn_real_data_subset() -> None:
    """VSN is iterative; exercise the real-data path on a feature subset for speed."""
    ds = _load_real()
    subset = dl.Dataset(
        abundances=ds.abundances[:, :500],
        feature_names=ds.feature_names[:500],
        feature_metadata=ds.feature_metadata.iloc[:500].reset_index(drop=True),
        metadata=ds.metadata,
        scale="linear",
    )
    out = norm.normalize(subset, "vsn")
    assert out.scale == "glog2"
    assert out.abundances.shape == subset.abundances.shape
    assert np.isfinite(out.abundances).all()
