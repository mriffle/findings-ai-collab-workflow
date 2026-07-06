"""Tests for the Boruta all-relevant selection template.

Layers (conventions/coding.md + lib/AUTHORING.md):
  * planted-truth — a separable classification signal and a linear regression signal
    must be recovered (planted features Confirmed and above the shadow floor; pure-noise
    features Rejected);
  * task inference (numeric -> regression; categorical -> classification), the
    low-cardinality-numeric warning, the explicit override, and the multiclass path;
  * fail-loud guards (NaN, missing target, bad params, too-few samples, degenerate
    target) and the constant-feature drop / non-log-scale warning;
  * result invariants (decision domain, history alignment, sorted table) + determinism;
  * real-5xFAD smoke — binary genotype recovers the canonical AD proteins (skips if
    absent).

We test *our wrapper* — arguments threaded, guards fire, the private shadow-history
capture aligns, the Dataset contract honored — not BorutaPy's published statistics.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analysis import boruta as bor
from common import data_loading as dl

# Fast Boruta settings for the synthetic tests (real defaults are max_iter=100).
_FAST_ITER = 40
_GUARD_ITER = 8


def _planted_classification(
    n: int = 44,
    p: int = 50,
    n_signal: int = 4,
    effect: float = 3.0,
    seed: int = 0,
    scale: dl.Scale = "log2",
) -> dl.Dataset:
    """A balanced 2-class Dataset with signal planted (higher in B) in n_signal cols."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(size=(n, p))
    x[:, :n_signal] += y[:, None] * effect
    names = np.array([f"F{i}" for i in range(p)])
    meta = pd.DataFrame(
        {
            "grp": np.where(y == 1, "B", "A"),
            "level": y.astype(float),  # numeric 0/1 (low-cardinality regression trap)
            "score": x[:, 0] + rng.normal(scale=0.1, size=n),  # continuous target
        }
    )
    return dl.Dataset(
        abundances=x,
        feature_names=names,
        feature_metadata=pd.DataFrame({"feature": names}),
        metadata=meta,
        scale=scale,
    )


@pytest.fixture(scope="module")
def clf_result() -> bor.BorutaResult:
    """A recovered planted-classification result, reused across tests."""
    ds = _planted_classification(seed=0)
    return bor.boruta_select(ds, "grp", max_iter=_FAST_ITER, random_state=0)


# --------------------------------------------------------------------------- #
# Planted truth
# --------------------------------------------------------------------------- #
def test_planted_classification_recovers_signal(clf_result: bor.BorutaResult) -> None:
    res = clf_result
    assert res.task == "classification"
    planted = {f"F{i}" for i in range(4)}
    confirmed = set(res.confirmed_features.tolist())
    assert planted <= confirmed  # every planted feature is Confirmed
    # the planted features clear the shadow noise floor
    idx = {name: i for i, name in enumerate(res.feature_names)}
    for name in planted:
        assert res.importance[idx[name]] > res.shadow_threshold


def test_planted_classification_rejects_noise(clf_result: bor.BorutaResult) -> None:
    # the 46 pure-noise features are overwhelmingly Rejected
    assert clf_result.n_rejected >= clf_result.n_features - 12
    assert clf_result.n_confirmed >= 4


def test_planted_regression_recovers_signal() -> None:
    ds = _planted_classification(seed=1)
    res = bor.boruta_select(ds, "score", max_iter=_FAST_ITER, random_state=0)
    assert res.task == "regression"
    assert res.classes is None
    # score is built from feature F0, so F0 must be selected (Confirmed or Tentative)
    decided = dict(zip(res.feature_names, res.decision, strict=True))
    assert decided["F0"] in ("Confirmed", "Tentative")


# --------------------------------------------------------------------------- #
# Task inference / override / multiclass
# --------------------------------------------------------------------------- #
def test_task_inferred_classification_from_categorical() -> None:
    ds = _planted_classification(seed=2)
    res = bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER, random_state=0)
    assert res.task == "classification"
    assert res.classes == ("A", "B")


def test_low_cardinality_numeric_warns_and_can_override() -> None:
    ds = _planted_classification(seed=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg = bor.boruta_select(ds, "level", max_iter=_GUARD_ITER, random_state=0)
    assert reg.task == "regression"
    assert any(issubclass(w.category, bor.LowCardinalityTargetWarning) for w in caught)
    # the override runs it as classification without the warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", bor.LowCardinalityTargetWarning)
        clf = bor.boruta_select(
            ds, "level", task="classification", max_iter=_GUARD_ITER, random_state=0
        )
    assert clf.task == "classification"
    assert clf.classes is not None and len(clf.classes) == 2


def test_multiclass_classification_runs() -> None:
    rng = np.random.default_rng(3)
    n, p = 45, 30
    codes = np.array([0, 1, 2] * 15)
    x = rng.normal(size=(n, p))
    x[:, 0] += codes * 1.5  # a little 3-way signal so the run is not degenerate
    names = np.array([f"F{i}" for i in range(p)])
    ds = dl.Dataset(
        abundances=x,
        feature_names=names,
        feature_metadata=pd.DataFrame({"feature": names}),
        metadata=pd.DataFrame({"trio": np.array(["A", "B", "C"] * 15)}),
        scale="log2",
    )
    res = bor.boruta_select(ds, "trio", max_iter=_GUARD_ITER, random_state=0)
    assert res.task == "classification"
    assert res.classes == ("A", "B", "C")
    assert res.n_confirmed + res.n_tentative + res.n_rejected == res.n_features


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_nan_abundances_raise() -> None:
    ds = _planted_classification(seed=0)
    ds.abundances[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER)


def test_non_log_scale_warns() -> None:
    ds = _planted_classification(seed=0, scale="linear")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER, random_state=0)
    assert any(issubclass(w.category, bor.BorutaScaleWarning) for w in caught)


def test_constant_features_dropped() -> None:
    ds = _planted_classification(seed=0)
    ds.abundances[:, -1] = 5.0  # a constant feature
    ds.abundances[:, -2] = 0.0  # an all-zero feature
    res = bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER, random_state=0)
    assert res.n_dropped_constant == 2
    assert res.n_features == ds.abundances.shape[1] - 2
    assert "F49" not in set(res.feature_names.tolist())


def test_all_constant_features_raise() -> None:
    ds = _planted_classification(seed=0)
    ds.abundances[:] = 1.0
    with pytest.raises(ValueError, match="non-constant"):
        bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER)


def test_missing_target_samples_dropped() -> None:
    ds = _planted_classification(seed=0)
    ds.metadata.loc[0:2, "grp"] = np.nan  # 3 samples with no class
    res = bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER, random_state=0)
    assert res.n_dropped_missing_target == 3
    assert res.n_samples == ds.abundances.shape[0] - 3


def test_target_not_in_metadata_raises() -> None:
    ds = _planted_classification(seed=0)
    with pytest.raises(ValueError, match="not in metadata"):
        bor.boruta_select(ds, "nope", max_iter=_GUARD_ITER)


def test_single_class_target_raises() -> None:
    ds = _planted_classification(seed=0)
    ds.metadata["grp"] = "A"  # one class only
    with pytest.raises(ValueError, match="at least 2"):
        bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER)


def test_class_with_one_sample_raises() -> None:
    ds = _planted_classification(seed=0)
    ds.metadata.loc[0, "grp"] = "solo"  # a singleton class
    with pytest.raises(ValueError, match="< 2 samples"):
        bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER)


def test_constant_regression_target_raises() -> None:
    ds = _planted_classification(seed=0)
    ds.metadata["flat"] = 7.0
    with pytest.raises(ValueError, match="constant"):
        bor.boruta_select(ds, "flat", max_iter=_GUARD_ITER)


def test_too_few_samples_raises() -> None:
    ds = _planted_classification(n=4, p=10, n_signal=2, seed=0)
    with pytest.raises(ValueError, match="at least"):
        bor.boruta_select(ds, "grp", max_iter=_GUARD_ITER)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"perc": 0.0}, "perc"),
        ({"perc": 150.0}, "perc"),
        ({"task": "cluster"}, "task"),
    ],
)
def test_bad_params_raise(kwargs: dict[str, object], match: str) -> None:
    ds = _planted_classification(seed=0)
    with pytest.raises(ValueError, match=match):
        bor.boruta_select(ds, "grp", **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Result invariants + determinism
# --------------------------------------------------------------------------- #
def test_result_invariants(clf_result: bor.BorutaResult) -> None:
    res = clf_result
    n = res.n_features
    assert res.decision.shape == (n,)
    assert res.ranking.shape == (n,)
    assert res.importance.shape == (n,)
    assert res.importance_history.shape == (res.n_iter, n)
    assert res.shadow_max_history.shape == (res.n_iter,)
    assert set(res.decision.tolist()) <= {"Confirmed", "Tentative", "Rejected"}
    assert res.n_confirmed + res.n_tentative + res.n_rejected == n
    assert res.shadow_threshold == pytest.approx(
        float(np.median(res.shadow_max_history))
    )


def test_table_sorted_and_confirmed_view(clf_result: bor.BorutaResult) -> None:
    table = clf_result.table
    assert list(table.columns) == ["feature", "decision", "ranking", "importance"]
    # ranking non-decreasing (primary sort key)
    assert bool(
        (table["ranking"].to_numpy()[1:] >= table["ranking"].to_numpy()[:-1]).all()
    )
    # confirmed_features == the Confirmed rows, in table order
    conf = table.loc[table["decision"] == "Confirmed", "feature"].tolist()
    assert clf_result.confirmed_features.tolist() == conf


def test_determinism() -> None:
    ds = _planted_classification(seed=5)
    a = bor.boruta_select(ds, "grp", max_iter=_FAST_ITER, random_state=7)
    b = bor.boruta_select(ds, "grp", max_iter=_FAST_ITER, random_state=7)
    assert a.decision.tolist() == b.decision.tolist()
    assert np.array_equal(a.ranking, b.ranking)


# --------------------------------------------------------------------------- #
# Real-data smoke (skips cleanly when testdata/5xFAD is absent)
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[2]
_PROT = _ROOT / "testdata" / "5xFAD" / "data" / "proteins_wide_unnormalized.tsv"
_META = _ROOT / "testdata" / "5xFAD" / "metadata" / "Replicates_5xFAD.csv"


@pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()), reason="5xFAD testdata absent"
)
def test_smoke_5xfad_binary_genotype_recovers_ad_proteins() -> None:
    from common import normalize as norm

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
    mask = logged.metadata["Genotype"].to_numpy() != "na"
    meta = logged.metadata.loc[mask].reset_index(drop=True).copy()
    meta["AD"] = np.where(meta["Genotype"].to_numpy() == "5xFAD", "5xFAD", "nonAD")
    import dataclasses

    exp = dataclasses.replace(
        logged, abundances=logged.abundances[mask, :], metadata=meta
    )
    res = bor.boruta_select(exp, "AD", max_iter=100, random_state=42)
    # oracle (validated in the preview): 18 Confirmed, 7 Tentative on this contrast
    assert res.n_confirmed == 18
    assert res.n_tentative == 7
    assert res.n_confirmed + res.n_tentative + res.n_rejected == res.n_features

    def gene(name: str) -> str:
        parts = str(name).split("|")
        return parts[2].replace("_MOUSE", "") if len(parts) == 3 else str(name)

    confirmed = {gene(n) for n in res.confirmed_features}
    # the canonical AD proteins (also the classifier's validated top coefficients)
    for canon in ("A4", "APOE", "CLUS", "MK", "C1QA", "C1QB", "C1QC"):
        assert canon in confirmed
