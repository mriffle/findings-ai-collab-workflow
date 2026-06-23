"""Tests for the batch-correction template (lib/common/batch_correct.py).

Two layers:
  * unit — synthetic fixtures with a planted additive batch effect (ComBat must remove
    it), the near-constant-feature passthrough (and that it doesn't poison the rest),
    the independent-Dataset contract, the scale/finite/batch-size/NaN guards, the
    sample_mask passthrough, and the confounding assessment (perfect / strong / none);
  * smoke — real 5xFAD data (git-ignored): Cohort batch vs Treatment, correction reduces
    between-cohort separation. Skips cleanly when absent.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common import batch_correct as bc
from common import data_loading as dl
from common import normalize as norm


def _ds(
    abundances: object,
    batch: Sequence[str] | None = None,
    scale: dl.Scale = "log2",
    meta_cols: dict[str, Sequence[str]] | None = None,
) -> dl.Dataset:
    """Build a minimal Dataset for batch-correction tests (default log2 scale)."""
    ab = np.asarray(abundances, dtype=float)
    n_samples, n_features = ab.shape
    feature_names = np.array([f"F{j}" for j in range(n_features)], dtype=str)
    feature_metadata = pd.DataFrame({"protein": feature_names})
    data: dict[str, list[str]] = {"sample": [f"s{i}" for i in range(n_samples)]}
    if batch is not None:
        data["batch"] = list(batch)
    if meta_cols is not None:
        for key, values in meta_cols.items():
            data[key] = list(values)
    metadata = pd.DataFrame(data).set_index("sample")
    return dl.Dataset(
        abundances=ab,
        feature_names=feature_names,
        feature_metadata=feature_metadata,
        metadata=metadata,
        scale=scale,
    )


def _two_batch_with_shift(shift: float, seed: int = 0) -> tuple[dl.Dataset, float]:
    """Two batches of 5 samples; batch B is base + ``shift``. Returns (ds, pre_gap)."""
    rng = np.random.default_rng(seed)
    n_features = 30
    base = rng.uniform(5.0, 10.0, size=n_features)
    a = base + rng.normal(0.0, 0.3, size=(5, n_features))
    b = base + shift + rng.normal(0.0, 0.3, size=(5, n_features))
    ab = np.vstack([a, b])
    batch = ["A"] * 5 + ["B"] * 5
    pre_gap = float(np.abs(a.mean(axis=0) - b.mean(axis=0)).mean())
    return _ds(ab, batch=batch), pre_gap


# --------------------------------------------------------------------------- #
# Unit — correction removes a planted batch effect
# --------------------------------------------------------------------------- #


def test_combat_removes_planted_batch_shift() -> None:
    ds, pre_gap = _two_batch_with_shift(shift=3.0)
    out = bc.combat_correct(ds, "batch")
    post_gap = float(
        np.abs(out.abundances[:5].mean(axis=0) - out.abundances[5:].mean(axis=0)).mean()
    )
    assert pre_gap > 2.5  # the planted shift is ~3
    assert post_gap < pre_gap / 5  # ComBat removes most of it


def test_combat_preserves_shape_scale_and_metadata() -> None:
    ds, _ = _two_batch_with_shift(shift=2.0)
    out = bc.combat_correct(ds, "batch")
    assert out.abundances.shape == ds.abundances.shape
    assert out.scale == "log2"  # correction does not change the scale
    assert list(out.feature_names) == list(ds.feature_names)
    pd.testing.assert_frame_equal(out.metadata, ds.metadata)


def test_globally_constant_feature_passes_through_and_does_not_poison() -> None:
    """A globally-constant feature can't be standardized; it is passed through (warned)
    and must NOT NaN the other features (ComBat's pooled EB step otherwise would)."""
    ds, _ = _two_batch_with_shift(shift=3.0)
    ab = ds.abundances.copy()
    ab[:, 0] = 7.0  # feature 0 constant across ALL samples -> pooled variance 0
    ds = _ds(ab, batch=list(ds.metadata["batch"]))
    with pytest.warns(bc.BatchPassthroughWarning, match="1 feature"):
        out = bc.combat_correct(ds, "batch")
    np.testing.assert_array_equal(out.abundances[:, 0], ab[:, 0])  # passed through
    assert np.isfinite(out.abundances).all()  # the rest were not poisoned
    assert not np.allclose(out.abundances[:, 1], ab[:, 1])  # other features corrected


def test_feature_constant_within_one_batch_is_corrected() -> None:
    """Constant within one batch but variable overall -> nonzero pooled variance, so it
    IS corrected. The old per-batch-variance filter wrongly excluded these."""
    ds, _ = _two_batch_with_shift(shift=3.0)
    ab = ds.abundances.copy()
    ab[:5, 0] = 7.0  # constant within batch A only; varies across the full matrix
    ds = _ds(ab, batch=list(ds.metadata["batch"]))
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # pycombat emits a benign per-batch divide warning
        out = bc.combat_correct(ds, "batch")
    assert not np.allclose(
        out.abundances[:, 0], ab[:, 0]
    )  # corrected, not passed through
    assert np.isfinite(out.abundances).all()


def test_combat_returns_independent_dataset() -> None:
    """Output must be independent: mutating its metadata must not touch the input."""
    ds, _ = _two_batch_with_shift(shift=2.0)
    out = bc.combat_correct(ds, "batch")
    assert out.metadata is not ds.metadata
    out.metadata["injected"] = 1
    assert "injected" not in ds.metadata.columns


# --------------------------------------------------------------------------- #
# Unit — sample_mask passthrough
# --------------------------------------------------------------------------- #


def test_sample_mask_passes_excluded_rows_through() -> None:
    rng = np.random.default_rng(1)
    base = rng.uniform(5.0, 10.0, size=20)
    exp = np.vstack(
        [
            base + rng.normal(0, 0.3, size=(2, 20)),
            base + 3.0 + rng.normal(0, 0.3, size=(2, 20)),
        ]
    )
    qc = rng.uniform(0.0, 1.0, size=(2, 20))
    ab = np.vstack([exp, qc])
    ds = _ds(ab, batch=["A", "A", "B", "B", "na", "na"])
    mask = np.array([True, True, True, True, False, False])
    out = bc.combat_correct(ds, "batch", sample_mask=mask)
    np.testing.assert_array_equal(out.abundances[4:], ab[4:])  # QC rows untouched
    assert not np.allclose(out.abundances[:4], ab[:4])  # experimental rows corrected


def test_sample_mask_wrong_shape_raises() -> None:
    ds, _ = _two_batch_with_shift(shift=2.0)
    with pytest.raises(ValueError, match="sample_mask shape"):
        bc.combat_correct(ds, "batch", sample_mask=np.array([True, False]))


# --------------------------------------------------------------------------- #
# Unit — fail-loud guards
# --------------------------------------------------------------------------- #


def test_refuses_linear_scale() -> None:
    ds, _ = _two_batch_with_shift(shift=2.0)
    ds = dl.Dataset(
        abundances=ds.abundances,
        feature_names=ds.feature_names,
        feature_metadata=ds.feature_metadata,
        metadata=ds.metadata,
        scale="linear",
    )
    with pytest.raises(ValueError, match="requires a log-ish scale"):
        bc.combat_correct(ds, "batch")


def test_combat_accepts_extended_log_scales() -> None:
    """log10 / ln / glog2 are valid ComBat inputs via the shared LOG_SCALES taxonomy."""
    scales: tuple[dl.Scale, ...] = ("log10", "ln", "glog2")
    for sc in scales:
        base, _ = _two_batch_with_shift(shift=2.0)
        ds = _ds(base.abundances, batch=list(base.metadata["batch"]), scale=sc)
        out = bc.combat_correct(ds, "batch")
        assert out.scale == sc


def test_refuses_non_finite() -> None:
    ds, _ = _two_batch_with_shift(shift=2.0)
    ab = ds.abundances.copy()
    ab[0, 0] = np.nan
    ds = _ds(ab, batch=list(ds.metadata["batch"]))
    with pytest.raises(ValueError, match="non-finite"):
        bc.combat_correct(ds, "batch")


def test_refuses_single_batch() -> None:
    rng = np.random.default_rng(2)
    ds = _ds(rng.uniform(5, 10, size=(4, 10)), batch=["A", "A", "A", "A"])
    with pytest.raises(ValueError, match="2 distinct batches"):
        bc.combat_correct(ds, "batch")


def test_refuses_singleton_batch() -> None:
    rng = np.random.default_rng(3)
    ds = _ds(rng.uniform(5, 10, size=(3, 10)), batch=["A", "A", "B"])
    with pytest.raises(ValueError, match="2 per batch"):
        bc.combat_correct(ds, "batch")


def test_missing_batch_column_raises() -> None:
    ds, _ = _two_batch_with_shift(shift=2.0)
    with pytest.raises(ValueError, match="not a metadata column"):
        bc.combat_correct(ds, "no_such_column")


# --------------------------------------------------------------------------- #
# Unit — confounding assessment
# --------------------------------------------------------------------------- #


def test_assess_flags_perfect_confounding_and_warns() -> None:
    ds = _ds(
        np.zeros((4, 5)),
        batch=["1", "1", "2", "2"],
        meta_cols={"treatment": ["Lec", "Lec", "ISO", "ISO"]},
    )
    with pytest.warns(bc.BatchConfoundingWarning, match="PERFECTLY confounded"):
        reports = bc.assess_batch_confounding(ds, "batch", ["treatment"])
    assert len(reports) == 1
    assert reports[0].perfectly_confounded
    assert reports[0].cramers_v == pytest.approx(1.0)
    assert int(reports[0].crosstab.to_numpy().sum()) == 4


def test_assess_independent_does_not_warn() -> None:
    ds = _ds(
        np.zeros((4, 5)),
        batch=["1", "1", "2", "2"],
        meta_cols={"treatment": ["Lec", "ISO", "Lec", "ISO"]},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", bc.BatchConfoundingWarning)
        reports = bc.assess_batch_confounding(ds, "batch", ["treatment"])
    assert not reports[0].perfectly_confounded
    assert reports[0].cramers_v == pytest.approx(0.0)


def test_assess_strong_but_not_perfect_confounding_warns() -> None:
    """Graded warning: near-perfect (not perfect) confounding above threshold warns."""
    batch = ["1"] * 10 + ["2"] * 10
    # batch1: all A; batch2: one A, nine B -> [[10,0],[1,9]], strong but not perfect.
    treat = ["A"] * 10 + ["A"] + ["B"] * 9
    ds = _ds(np.zeros((20, 5)), batch=batch, meta_cols={"treatment": treat})
    with pytest.warns(bc.BatchConfoundingWarning, match="STRONGLY"):
        reports = bc.assess_batch_confounding(ds, "batch", ["treatment"])
    assert not reports[0].perfectly_confounded
    assert reports[0].cramers_v >= 0.5


def test_assess_raises_on_nan_covariate() -> None:
    """The safety check must fail loud on a missing covariate, not silently drop it."""
    meta = pd.DataFrame(
        {
            "sample": ["s0", "s1", "s2", "s3"],
            "batch": ["1", "1", "2", "2"],
            "treatment": ["Lec", "Lec", "ISO", None],
        }
    ).set_index("sample")
    ds = dl.Dataset(
        abundances=np.zeros((4, 5)),
        feature_names=np.array([f"F{j}" for j in range(5)], dtype=str),
        feature_metadata=pd.DataFrame({"protein": [f"F{j}" for j in range(5)]}),
        metadata=meta,
        scale="log2",
    )
    with pytest.raises(ValueError, match="missing value"):
        bc.assess_batch_confounding(ds, "batch", ["treatment"])


def test_assess_reports_one_per_covariate() -> None:
    ds = _ds(
        np.zeros((4, 5)),
        batch=["1", "1", "2", "2"],
        meta_cols={
            "treatment": ["Lec", "ISO", "Lec", "ISO"],
            "genotype": ["5xFAD", "5xFAD", "WT", "WT"],
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bc.BatchConfoundingWarning)
        reports = bc.assess_batch_confounding(ds, "batch", ["treatment", "genotype"])
    assert [r.covariate_column for r in reports] == ["treatment", "genotype"]
    assert not reports[0].perfectly_confounded  # treatment varies within batch
    assert reports[1].perfectly_confounded  # genotype nested in batch


def test_assess_missing_column_raises() -> None:
    ds = _ds(np.zeros((4, 5)), batch=["1", "1", "2", "2"])
    with pytest.raises(ValueError, match="not a metadata column"):
        bc.assess_batch_confounding(ds, "batch", ["nope"])


# --------------------------------------------------------------------------- #
# Smoke — real 5xFAD data (git-ignored): Cohort batch, confounded with Treatment
# --------------------------------------------------------------------------- #

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"
_COLLAPSE = dl.ReplicateCollapse("Sample ID", "Technical Replicate")
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)


def _real_experimental_log2() -> dl.Dataset:
    """Load real data, median-normalize + log2, keep Cohort 1/2 (experimental)."""
    ds = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=_COLLAPSE,
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    logged = norm.log2_transform(norm.normalize(ds, "median"))
    mask = logged.metadata["Cohort"].isin(["1", "2"]).to_numpy()
    return dl.Dataset(
        abundances=logged.abundances[mask],
        feature_names=logged.feature_names,
        feature_metadata=logged.feature_metadata,
        metadata=logged.metadata[mask],
        scale=logged.scale,
    )


@_skip_no_data
def test_smoke_assess_cohort_vs_treatment() -> None:
    ds = _real_experimental_log2()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bc.BatchConfoundingWarning)
        reports = bc.assess_batch_confounding(ds, "Cohort", ["Treatment"])
    assert len(reports) == 1
    assert 0.0 <= reports[0].cramers_v <= 1.0
    assert int(reports[0].crosstab.to_numpy().sum()) == ds.abundances.shape[0]


@_skip_no_data
@pytest.mark.slow
def test_smoke_combat_reduces_cohort_separation() -> None:
    ds = _real_experimental_log2()
    cohort = ds.metadata["Cohort"].to_numpy().astype(str)
    a, b = ds.abundances[cohort == "1"], ds.abundances[cohort == "2"]
    pre_gap = float(np.abs(a.mean(axis=0) - b.mean(axis=0)).mean())
    with warnings.catch_warnings():
        # the experimental subset has a globally-constant feature (passthrough expected)
        warnings.simplefilter("ignore", bc.BatchPassthroughWarning)
        out = bc.combat_correct(ds, "Cohort")
    assert out.abundances.shape == ds.abundances.shape
    assert out.scale == "log2"
    assert np.isfinite(out.abundances).all()
    oa, ob = out.abundances[cohort == "1"], out.abundances[cohort == "2"]
    post_gap = float(np.abs(oa.mean(axis=0) - ob.mean(axis=0)).mean())
    assert post_gap < pre_gap  # correction reduces between-cohort separation
