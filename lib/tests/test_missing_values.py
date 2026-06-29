"""Tests for the missing-value template (lib/common/missing_values.py).

Two layers:
  * unit — hand-built synthetic fixtures with planted-truth checks (the per-feature
    mean/median fills, the zero fill, the shared "0 or NaN = missing" predicate, the
    missing-fraction filter) plus the fail-loud guards (scale, parameters, fully-missing
    features) and the KNN wiring (verified against sklearn directly, not re-proven);
  * smoke — the real 5xFAD data under testdata/ (git-ignored); skips cleanly if absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common import data_loading as dl
from common import missing_values as mv
from sklearn.impute import KNNImputer


def _ds(
    abundances: object,
    *,
    feature_names: list[str] | None = None,
    scale: dl.Scale = "linear",
) -> dl.Dataset:
    """Build a minimal aligned Dataset around an abundance matrix for these tests."""
    ab = np.asarray(abundances, dtype=float)
    n_samples, n_features = ab.shape
    names = (
        feature_names
        if feature_names is not None
        else [f"F{j}" for j in range(n_features)]
    )
    fn = np.array(names, dtype=str)
    feature_metadata = pd.DataFrame({"protein": fn})
    metadata = pd.DataFrame({"sample": [f"s{i}" for i in range(n_samples)]}).set_index(
        "sample"
    )
    return dl.Dataset(
        abundances=ab,
        feature_names=fn,
        feature_metadata=feature_metadata,
        metadata=metadata,
        scale=scale,
    )


# --------------------------------------------------------------------------- #
# Unit — feature filtering (planted truth)
# --------------------------------------------------------------------------- #


def test_filter_drops_high_missing_features() -> None:
    """F1 is all-missing (frac 1.0), F2 missing 1/3; max 0.5 keeps F0 + F2, drops F1."""
    ab = [[1.0, 0.0, 5.0], [2.0, 0.0, 0.0], [3.0, 0.0, 7.0]]
    ds = _ds(ab, feature_names=["F0", "F1", "F2"])
    res = mv.handle_missing(ds, max_missing_fraction=0.5)
    assert list(res.dataset.feature_names) == ["F0", "F2"]
    assert list(res.dropped_features) == ["F1"]
    assert list(res.dataset.feature_metadata["protein"]) == ["F0", "F2"]
    assert res.n_features_in == 3
    assert res.n_features_out == 2
    assert res.n_values_imputed == 0  # impute is None
    np.testing.assert_allclose(
        res.dataset.abundances, [[1.0, 5.0], [2.0, 0.0], [3.0, 7.0]]
    )


def test_filter_none_keeps_all_features() -> None:
    ds = _ds([[1.0, 0.0], [2.0, 3.0]])
    res = mv.handle_missing(ds)
    assert res.n_features_out == 2
    assert list(res.dropped_features) == []


def test_filter_then_zero_combination() -> None:
    """The canonical recipe: drop features missing in > 50 %, then zero the rest."""
    ab = [[1.0, 0.0, 5.0], [2.0, 0.0, 0.0], [3.0, 0.0, 7.0]]
    ds = _ds(ab, feature_names=["F0", "F1", "F2"])
    res = mv.handle_missing(ds, max_missing_fraction=0.5, impute="zero")
    assert list(res.dataset.feature_names) == ["F0", "F2"]
    np.testing.assert_allclose(
        res.dataset.abundances, [[1.0, 5.0], [2.0, 0.0], [3.0, 7.0]]
    )
    assert np.isfinite(res.dataset.abundances).all()
    assert res.n_values_imputed == 1  # F2's single not-detected entry, zeroed


# --------------------------------------------------------------------------- #
# Unit — imputation (planted truth)
# --------------------------------------------------------------------------- #


def test_zero_impute_planted() -> None:
    """NaN and a literal 0 both count as missing and become 0."""
    ds = _ds([[1.0, np.nan, 5.0], [2.0, 0.0, 8.0]])
    res = mv.handle_missing(ds, impute="zero")
    np.testing.assert_allclose(
        res.dataset.abundances, [[1.0, 0.0, 5.0], [2.0, 0.0, 8.0]]
    )
    assert res.n_values_imputed == 2
    assert np.isfinite(res.dataset.abundances).all()


def test_mean_impute_planted() -> None:
    """F0 detected = [4, 8] -> mean 6 fills the missing 0."""
    ds = _ds([[4.0, 10.0], [0.0, 20.0], [8.0, 30.0]])
    res = mv.handle_missing(ds, impute="mean")
    np.testing.assert_allclose(
        res.dataset.abundances, [[4.0, 10.0], [6.0, 20.0], [8.0, 30.0]]
    )
    assert res.n_values_imputed == 1
    assert res.dataset.scale == "linear"


def test_median_impute_planted() -> None:
    """F0 detected = [2, 4, 10] -> median 4 fills the missing 0."""
    ds = _ds([[2.0, 1.0], [4.0, 1.0], [0.0, 1.0], [10.0, 1.0]])
    res = mv.handle_missing(ds, impute="median")
    np.testing.assert_allclose(
        res.dataset.abundances, [[2.0, 1.0], [4.0, 1.0], [4.0, 1.0], [10.0, 1.0]]
    )
    assert res.n_values_imputed == 1


def test_knn_impute_planted_nearest() -> None:
    """n_neighbors=1: row 1's missing F2 takes row 0's value (identical on F0, F1)."""
    ds = _ds([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [9.0, 9.0, 9.0]])
    res = mv.handle_missing(ds, impute="knn", knn_neighbors=1)
    np.testing.assert_allclose(
        res.dataset.abundances, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0]]
    )
    assert res.n_values_imputed == 1


def test_knn_impute_matches_sklearn_directly() -> None:
    """Wiring check: missing -> NaN, then KNNImputer; we don't re-prove the library."""
    rng = np.random.default_rng(7)
    ab = rng.uniform(1.0, 100.0, size=(8, 12))
    ab[1, 3] = 0.0
    ab[4, 7] = np.nan
    ab[6, 0] = 0.0
    ds = _ds(ab)
    res = mv.handle_missing(ds, impute="knn", knn_neighbors=3)

    missing = ~(np.isfinite(ab) & (ab > 0.0))
    expected = KNNImputer(n_neighbors=3).fit_transform(np.where(missing, np.nan, ab))
    np.testing.assert_allclose(res.dataset.abundances, expected)
    assert res.n_values_imputed == 3


def test_impute_none_leaves_missing_untouched() -> None:
    ds = _ds([[1.0, np.nan], [2.0, 3.0]])
    res = mv.handle_missing(ds)
    assert np.isnan(res.dataset.abundances[0, 1])
    assert res.n_values_imputed == 0


@pytest.mark.parametrize("method", ["zero", "mean", "median", "knn"])
def test_detected_values_pass_through(method: mv.ImputeMethod) -> None:
    """Every method leaves the detected (finite, > 0) values exactly as they were."""
    rng = np.random.default_rng(3)
    ab = rng.uniform(1.0, 100.0, size=(5, 6))
    ab[2, 3] = 0.0  # one not-detected entry
    ds = _ds(ab)
    res = mv.handle_missing(ds, impute=method)
    detected = ab > 0.0
    np.testing.assert_allclose(res.dataset.abundances[detected], ab[detected])


def test_min_intensity_threshold_marks_low_values_missing() -> None:
    """With min_intensity=5, values <= 5 are not detected and get imputed."""
    ds = _ds([[3.0, 10.0], [6.0, 2.0]])
    res = mv.handle_missing(ds, impute="zero", min_intensity=5.0)
    np.testing.assert_allclose(res.dataset.abundances, [[0.0, 10.0], [6.0, 0.0]])
    assert res.n_values_imputed == 2


# --------------------------------------------------------------------------- #
# Unit — invariants
# --------------------------------------------------------------------------- #


def test_scale_stays_linear() -> None:
    ds = _ds([[1.0, 0.0], [2.0, 3.0]])
    assert mv.handle_missing(ds, impute="zero").dataset.scale == "linear"


def test_returns_independent_dataset() -> None:
    """Output shares no mutable state: mutating it must not touch the input."""
    ds = _ds([[1.0, 2.0], [3.0, 4.0]])
    res = mv.handle_missing(ds, impute="zero")
    assert res.dataset.metadata is not ds.metadata
    res.dataset.metadata["injected"] = 1
    assert "injected" not in ds.metadata.columns
    res.dataset.abundances[0, 0] = 999.0
    assert ds.abundances[0, 0] == 1.0


def test_report_records_parameters() -> None:
    ab = [[1.0, 0.0, 5.0], [2.0, 0.0, 0.0], [3.0, 0.0, 7.0]]
    ds = _ds(ab, feature_names=["F0", "F1", "F2"])
    res = mv.handle_missing(
        ds, max_missing_fraction=0.5, impute="zero", min_intensity=0.0
    )
    assert res.max_missing_fraction == 0.5
    assert res.impute == "zero"
    assert res.min_intensity == 0.0
    assert list(res.dropped_features) == ["F1"]


# --------------------------------------------------------------------------- #
# Unit — fail-loud guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale", ["log2", "glog2", "zscore"])
def test_refuses_non_linear_scale(scale: dl.Scale) -> None:
    ds = _ds([[1.0, 2.0], [3.0, 4.0]], scale=scale)
    with pytest.raises(mv.MissingValueScaleError, match="linear-scale"):
        mv.handle_missing(ds, impute="zero")


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_bad_max_missing_fraction(bad: float) -> None:
    ds = _ds([[1.0, 2.0]])
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        mv.handle_missing(ds, max_missing_fraction=bad)


def test_unknown_impute_method_raises() -> None:
    ds = _ds([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Unknown impute method"):
        mv.handle_missing(ds, impute="bogus")  # type: ignore[arg-type]


def test_knn_neighbors_must_be_positive() -> None:
    ds = _ds([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="knn_neighbors must be"):
        mv.handle_missing(ds, impute="knn", knn_neighbors=0)


@pytest.mark.parametrize("method", ["mean", "median", "knn"])
def test_fully_missing_feature_refused(method: mv.ImputeMethod) -> None:
    """A kept feature with no detected value cannot be mean/median/KNN imputed."""
    ds = _ds([[1.0, 0.0], [2.0, 0.0]])  # F1 entirely missing
    with pytest.raises(ValueError, match="no detected"):
        mv.handle_missing(ds, impute=method)


def test_fully_missing_feature_allowed_for_zero() -> None:
    ds = _ds([[1.0, 0.0], [2.0, 0.0]])
    res = mv.handle_missing(ds, impute="zero")
    np.testing.assert_allclose(res.dataset.abundances, [[1.0, 0.0], [2.0, 0.0]])
    assert res.n_features_out == 2


def test_drop_all_features_raises() -> None:
    ds = _ds([[1.0, 0.0], [0.0, 2.0]])  # each feature missing in half the samples
    with pytest.raises(ValueError, match="drops every feature"):
        mv.handle_missing(ds, max_missing_fraction=0.0)


def test_non_2d_abundances_raises() -> None:
    ds = dl.Dataset(
        abundances=np.array([1.0, 2.0, 3.0]),
        feature_names=np.array(["F0", "F1", "F2"]),
        feature_metadata=pd.DataFrame({"protein": ["F0", "F1", "F2"]}),
        metadata=pd.DataFrame({"sample": ["s0"]}).set_index("sample"),
        scale="linear",
    )
    with pytest.raises(ValueError, match="must be 2D"):
        mv.handle_missing(ds)


def test_empty_dataset_raises() -> None:
    ds = _ds(np.empty((0, 2)))
    with pytest.raises(ValueError, match="empty"):
        mv.handle_missing(ds)


def test_feature_name_misalignment_raises() -> None:
    ds = dl.Dataset(
        abundances=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        feature_names=np.array(["F0", "F1"]),  # 2, but 3 features
        feature_metadata=pd.DataFrame({"protein": ["F0", "F1", "F2"]}),
        metadata=pd.DataFrame({"sample": ["s0", "s1"]}).set_index("sample"),
        scale="linear",
    )
    with pytest.raises(ValueError, match="feature_names has"):
        mv.handle_missing(ds)


def test_feature_metadata_misalignment_raises() -> None:
    ds = dl.Dataset(
        abundances=np.array([[1.0, 2.0, 3.0]]),
        feature_names=np.array(["F0", "F1", "F2"]),
        feature_metadata=pd.DataFrame({"protein": ["F0", "F1"]}),  # 2, but 3 features
        metadata=pd.DataFrame({"sample": ["s0"]}).set_index("sample"),
        scale="linear",
    )
    with pytest.raises(ValueError, match="feature_metadata has"):
        mv.handle_missing(ds)


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
def test_smoke_filter_zero_real_data() -> None:
    ds = _load_real()
    assert ds.abundances.shape == (61, 8829)
    res = mv.handle_missing(ds, max_missing_fraction=0.5, impute="zero")
    assert res.n_features_in == 8829
    assert res.n_features_out == 8602
    assert len(res.dropped_features) == 227
    assert res.n_values_imputed == 15814
    assert res.dataset.scale == "linear"
    assert np.isfinite(res.dataset.abundances).all()
    assert int((res.dataset.abundances == 0).sum()) == 15814  # not-detected stay 0


@_skip_no_data
def test_smoke_mean_impute_real_data() -> None:
    ds = _load_real()
    res = mv.handle_missing(ds, max_missing_fraction=0.5, impute="mean")
    assert res.n_features_out == 8602
    assert res.n_values_imputed == 15814
    assert np.isfinite(res.dataset.abundances).all()
    assert bool(
        (res.dataset.abundances > 0).all()
    )  # mean of detected values is positive


@_skip_no_data
def test_smoke_strict_filter_real_data() -> None:
    ds = _load_real()
    res = mv.handle_missing(ds, max_missing_fraction=0.0)
    assert res.n_features_out == 6275
    missing = ~(np.isfinite(res.dataset.abundances) & (res.dataset.abundances > 0.0))
    assert (
        int(missing.sum()) == 0
    )  # strict filter leaves no missing among kept features
