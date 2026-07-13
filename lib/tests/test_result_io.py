"""Tests for the analysis-result on-disk round-trip + cache identity (result_io.py).

* round-trip — a frozen dataclass with every supported field kind (DataFrame,
  numeric + object ndarray, nested dataclass, list of dataclasses, dict, tuple,
  optional-set + optional-None, scalars) must reload identically;
* edge-case arrays — NaN/inf, an empty object array, a 2-D object array round-trip;
* real end-to-end — each of the four CPU-heavy results (classification, regression,
  xgboost, boruta) is run on tiny data, round-tripped, and its **figures render from
  the reloaded result** (the reason the cache exists);
* identity — result_fingerprint is deterministic, input-sensitive, and stable across
  invocations (a pinned golden value); two independent runs of the same operation hit
  the same cache slot, and a different operation does not;
* cache — save/load/read_meta round-trips under <root>/<analysis>/<fingerprint>/;
* fail-loud — non-dataclass, missing manifest, bad version, non-JSON params raise.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from analysis import boruta as bor
from analysis import classification as clf
from analysis import classification_xgboost as xgb
from analysis import regression as reg
from analysis import result_io as rio
from common import data_loading as dl
from figures import boruta_importance as borfig
from figures import classification as clffig
from figures import classification_xgboost as xgbfig
from figures import regression as regfig
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _planted(n: int = 32, p: int = 30, n_signal: int = 4, seed: int = 0) -> dl.Dataset:
    """A tiny balanced 2-class Dataset with a numeric ``dose`` twin, signal planted."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(size=(n, p))
    x[:, :n_signal] += y[:, None] * 2.0
    names = np.array([f"F{i}" for i in range(p)])
    meta = pd.DataFrame(
        {"grp": np.where(y == 1, "B", "A"), "dose": y.astype(float) * 50.0}
    )
    return dl.Dataset(
        abundances=x,
        feature_names=names,
        feature_metadata=pd.DataFrame({"feature": names}),
        metadata=meta,
        scale="log2",
    )


@dataclass(frozen=True)
class _Inner:
    a: np.ndarray
    b: float


@dataclass(frozen=True)
class _Outer:
    frame: pd.DataFrame
    numeric: np.ndarray
    strings: np.ndarray
    inners: list[_Inner]
    one_inner: _Inner
    mapping: dict[str, float]
    grid: tuple[float, ...]
    label: str
    count: int
    flag: bool
    maybe: str | None
    opt_arr: np.ndarray | None
    opt_none: np.ndarray | None


def _outer(count: int = 3) -> _Outer:
    return _Outer(
        frame=pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}),
        numeric=np.array([[1.0, 2.0], [3.0, 4.0]]),
        strings=np.array(["P1", "P2", "P3"], dtype=object),
        inners=[
            _Inner(a=np.array([1.0, 2.0]), b=0.5),
            _Inner(a=np.array([9.0]), b=1.5),
        ],
        one_inner=_Inner(a=np.array([7.0]), b=2.5),
        mapping={"alpha": 0.1, "beta": 0.9},
        grid=(0.1, 1.0, 10.0),
        label="run",
        count=count,
        flag=True,
        maybe=None,
        opt_arr=np.array([0.5, 0.6]),
        opt_none=None,
    )


@dataclass(frozen=True)
class _Arrays:
    nan_inf: np.ndarray
    empty_obj: np.ndarray
    obj_2d: np.ndarray


def test_edge_case_arrays_round_trip(tmp_path: Path) -> None:
    """NaN/inf numeric, an empty object array, and a 2-D object array survive."""
    original = _Arrays(
        nan_inf=np.array([1.0, np.nan, np.inf, -np.inf, 0.0]),
        empty_obj=np.empty(0, dtype=object),
        obj_2d=np.array([["a", "bb"], ["ccc", "d"]], dtype=object),
    )
    rio.save_result(original, tmp_path / "arrs")
    out = rio.load_result(tmp_path / "arrs", _Arrays)
    np.testing.assert_array_equal(out.nan_inf, original.nan_inf)  # nan/inf preserved
    assert out.empty_obj.shape == (0,)
    np.testing.assert_array_equal(out.obj_2d, original.obj_2d)


def test_round_trip_every_field_kind(tmp_path: Path) -> None:
    original = _outer()
    rio.save_result(original, tmp_path / "outer")
    out = rio.load_result(tmp_path / "outer", _Outer)

    pd.testing.assert_frame_equal(out.frame, original.frame)
    np.testing.assert_allclose(out.numeric, original.numeric)
    np.testing.assert_array_equal(out.strings, original.strings)
    assert [i.b for i in out.inners] == [0.5, 1.5]
    np.testing.assert_allclose(out.inners[0].a, [1.0, 2.0])
    np.testing.assert_allclose(out.one_inner.a, [7.0])
    assert out.mapping == {"alpha": 0.1, "beta": 0.9}
    assert out.grid == (0.1, 1.0, 10.0) and isinstance(out.grid, tuple)
    assert out.label == "run" and out.count == 3 and out.flag is True
    assert out.maybe is None
    assert out.opt_arr is not None
    np.testing.assert_allclose(out.opt_arr, np.array([0.5, 0.6]))
    assert out.opt_none is None


def test_real_classification_result_round_trips(tmp_path: Path) -> None:
    result = clf.ClassificationResult(
        coefficients=pd.DataFrame(
            {
                "feature": ["P1"],
                "coef": [0.5],
                "abs_coef": [0.5],
                "selection_frequency": [1.0],
                "sign_consistency": [1.0],
                "coef_median": [0.5],
                "coef_q25": [0.4],
                "coef_q75": [0.6],
            }
        ),
        fold_predictions=[
            clf.FoldPrediction(
                y_true=np.array([0, 1, 0]), y_prob=np.array([0.1, 0.9, 0.2])
            )
        ],
        cv_auc=0.9,
        cv_auc_sd=0.05,
        cv_balanced_accuracy=0.85,
        cv_average_precision=0.88,
        best_c=1.0,
        best_l1_ratio=0.5,
        grid_scores=np.array([[0.8, 0.9], [0.7, 0.85]]),
        c_grid=(0.1, 1.0),
        l1_grid=(0.2, 0.5),
        outcome="genotype",
        positive_label="high",
        negative_label="low",
        generalization_target="samples",
        grouped=False,
        groups_column=None,
        n_samples=30,
        n_positive=15,
        n_negative=15,
        n_features=100,
        n_dropped_constant=2,
        n_dropped_unassigned=0,
        random_state=42,
        null_aucs=np.array([0.5, 0.52, 0.48]),
        observed_auc=0.9,
        null_p=0.01,
        feature_names=np.array(["P1", "P2"], dtype=object),
    )
    rio.save_result(result, tmp_path / "clf")
    out = rio.load_result(tmp_path / "clf", clf.ClassificationResult)

    assert out.cv_auc == 0.9
    assert out.generalization_target == "samples"
    assert out.groups_column is None
    pd.testing.assert_frame_equal(out.coefficients, result.coefficients)
    np.testing.assert_array_equal(out.fold_predictions[0].y_prob, [0.1, 0.9, 0.2])
    assert out.null_aucs is not None
    np.testing.assert_allclose(out.null_aucs, np.array([0.5, 0.52, 0.48]))
    np.testing.assert_array_equal(
        out.feature_names, np.array(["P1", "P2"], dtype=object)
    )
    assert out.validated_eligible is True  # property derived from null_p


# --------------------------------------------------------------------------- #
# Real end-to-end: run each CPU-heavy analysis on tiny data, round-trip it, and
# render its figures FROM THE RELOADED result (the reason the cache exists).
# --------------------------------------------------------------------------- #
def _assert_renders(*figs: Figure) -> None:
    for fig in figs:
        assert isinstance(fig, Figure)
        plt.close(fig)


def test_classification_result_cache_then_figures(tmp_path: Path) -> None:
    result = clf.classify(
        _planted(),
        "grp",
        run_null=True,
        n_permutations=8,
        null_repeats=1,
        random_state=0,
        c_grid=[0.1, 1.0],
        l1_ratios=[0.5, 1.0],
        n_repeats=2,
        stability_repeats=3,
        n_jobs=1,
        max_iter=2000,
        tol=1e-3,
    )
    rio.save_result(result, tmp_path / "clf")
    out = rio.load_result(tmp_path / "clf", clf.ClassificationResult)
    assert out.cv_auc == result.cv_auc
    assert out.null_p == result.null_p
    pd.testing.assert_frame_equal(out.coefficients, result.coefficients)
    np.testing.assert_allclose(out.grid_scores, result.grid_scores)
    assert len(out.fold_predictions) == len(result.fold_predictions)
    _assert_renders(
        clffig.plot_roc(out),
        clffig.plot_null(out),
        clffig.plot_coefficients(out),
        clffig.plot_hyperparameter_heatmap(out),
    )


def test_regression_result_cache_then_figures(tmp_path: Path) -> None:
    result = reg.regress(
        _planted(),
        "dose",
        run_null=True,
        n_permutations=8,
        null_repeats=1,
        random_state=0,
        alpha_grid=[0.05, 0.1, 0.5],
        l1_ratios=[0.5, 1.0],
        n_repeats=2,
        stability_repeats=3,
        n_jobs=1,
        max_iter=5000,
        tol=1e-3,
    )
    rio.save_result(result, tmp_path / "reg")
    out = rio.load_result(tmp_path / "reg", reg.RegressionResult)
    assert out.cv_r2 == result.cv_r2
    assert out.null_p == result.null_p
    pd.testing.assert_frame_equal(out.coefficients, result.coefficients)
    np.testing.assert_allclose(out.grid_scores, result.grid_scores)
    _assert_renders(
        regfig.plot_predicted_vs_observed(out),
        regfig.plot_null(out),
        regfig.plot_coefficients(out),
        regfig.plot_hyperparameter_heatmap(out),
    )


def test_xgboost_result_cache_then_figures(tmp_path: Path) -> None:
    result = xgb.classify_xgboost(
        _planted(),
        "grp",
        run_null=True,
        n_permutations=8,
        null_repeats=1,
        random_state=0,
        max_depth_grid=[2, 3],
        learning_rate_grid=[0.1, 0.3],
        n_estimators=40,
        n_repeats=2,
        stability_repeats=3,
        n_jobs=1,
    )
    rio.save_result(result, tmp_path / "xgb")
    out = rio.load_result(tmp_path / "xgb", xgb.XGBClassificationResult)
    assert out.cv_auc == result.cv_auc
    assert out.best_params == result.best_params  # dict[str, float] round-trip
    assert out.null_p == result.null_p
    pd.testing.assert_frame_equal(out.importances, result.importances)
    _assert_renders(
        xgbfig.plot_roc(out),
        xgbfig.plot_null(out),
        xgbfig.plot_importance(out),
        xgbfig.plot_hyperparameter_heatmap(out),
    )


def test_boruta_result_cache_then_figure(tmp_path: Path) -> None:
    result = bor.boruta_select(_planted(), "grp", max_iter=40, random_state=0)
    rio.save_result(result, tmp_path / "bor")
    out = rio.load_result(tmp_path / "bor", bor.BorutaResult)
    np.testing.assert_array_equal(out.decision, result.decision)  # object str array
    np.testing.assert_allclose(out.importance, result.importance)
    assert out.classes == result.classes  # tuple[str, ...] | None
    assert out.task == result.task
    _assert_renders(borfig.plot_boruta_importance(out))


def test_fingerprint_is_stable_across_invocations() -> None:
    # sha256 over canonical JSON → the same id in any process/version (not Python's
    # per-process salted hash). Pinned so a change to the identity scheme is caught.
    assert (
        rio.result_fingerprint(
            analysis="classification",
            data_version="v1",
            params={"outcome": "grp", "run_null": True},
            seed=0,
        )
        == "bdc844a59060"
    )


def test_feature_list_participates_in_the_fingerprint() -> None:
    # feature_list rides in `params`, so it is part of the cache identity: a different
    # list -> a different entry, and a restricted run differs from the whole-proteome
    # one. (Callers record a canonical sorted/deduped list so equal sets collide.)
    def fp(feature_list: object) -> str:
        return rio.result_fingerprint(
            analysis="classification",
            data_version="v1",
            params={"outcome": "grp", "feature_list": feature_list},
            seed=0,
        )

    assert fp(["P1", "P2"]) == fp(["P1", "P2"])
    assert fp(["P1", "P2"]) != fp(["P1", "P3"])
    assert fp(["P1", "P2"]) != fp(None)


def test_independent_runs_hit_the_same_cache_slot(tmp_path: Path) -> None:
    root = tmp_path / "results"
    params: dict[str, object] = {"outcome": "grp", "run_null": False}
    # Run 1 computes + caches.
    m1 = rio.save_cached_result(
        _outer(),
        cache_root=root,
        analysis="classification",
        data_version="v1",
        params=params,
        seed=0,
    )
    # Run 2 — a separate invocation of the *same operation* — recomputes only the
    # fingerprint from the same declared inputs, and finds Run 1's slot (→ reuse).
    fp2 = rio.result_fingerprint(
        analysis="classification", data_version="v1", params=params, seed=0
    )
    assert fp2 == m1.fingerprint
    assert (root / "classification" / fp2).is_dir()
    reused = rio.load_cached_result(
        _Outer, cache_root=root, analysis="classification", fingerprint=fp2
    )
    assert reused.count == 3  # Run 1's cached result, found + loaded by Run 2

    # Identity is keyed by inputs, not result content: a different result, same inputs
    # → the same slot.
    m2 = rio.save_cached_result(
        _outer(count=99),
        cache_root=root,
        analysis="classification",
        data_version="v1",
        params=params,
        seed=0,
    )
    assert m2.fingerprint == m1.fingerprint

    # A genuinely different operation → a different slot (no false cache hit).
    m3 = rio.save_cached_result(
        _outer(),
        cache_root=root,
        analysis="classification",
        data_version="v1",
        params={"outcome": "dose", "run_null": False},
        seed=0,
    )
    assert m3.fingerprint != m1.fingerprint


def test_fingerprint_is_deterministic_and_input_sensitive() -> None:
    def fp(
        *, data_version: str = "v1", params: dict[str, object], seed: int = 7
    ) -> str:
        return rio.result_fingerprint(
            analysis="classification",
            data_version=data_version,
            params=params,
            seed=seed,
        )

    ref = fp(params={"outcome": "g"})
    assert fp(params={"outcome": "g"}) == ref
    # key order in params must not matter
    assert fp(params={"a": 1, "b": 2}) == fp(params={"b": 2, "a": 1})
    # any input change changes the fingerprint
    assert fp(params={"outcome": "g"}, seed=8) != ref
    assert fp(params={"outcome": "h"}) != ref
    assert fp(params={"outcome": "g"}, data_version="v2") != ref


def test_fingerprint_rejects_non_json_params() -> None:
    with pytest.raises(TypeError):
        rio.result_fingerprint(
            analysis="c", data_version="v1", params={"bad": {1, 2, 3}}, seed=1
        )


def test_cache_round_trip_and_meta(tmp_path: Path) -> None:
    meta = rio.save_cached_result(
        _outer(),
        cache_root=tmp_path / "results",
        analysis="demo",
        data_version="v1",
        params={"outcome": "g", "run_null": True},
        seed=7,
        label="my run",
        created="2026-07-13T00:00:00",
    )
    expected_dir = tmp_path / "results" / "demo" / meta.fingerprint
    assert (expected_dir / "meta.json").is_file()
    assert meta.fingerprint == rio.result_fingerprint(
        analysis="demo",
        data_version="v1",
        params={"outcome": "g", "run_null": True},
        seed=7,
    )

    out = rio.load_cached_result(
        _Outer,
        cache_root=tmp_path / "results",
        analysis="demo",
        fingerprint=meta.fingerprint,
    )
    assert out.label == "run"

    read = rio.read_result_meta(expected_dir)
    assert read.label == "my run"
    assert read.created == "2026-07-13T00:00:00"
    assert read.params == {"outcome": "g", "run_null": True}


def test_save_rejects_non_dataclass(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="dataclass"):
        rio.save_result({"not": "a dataclass"}, tmp_path / "x")


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a save_result directory"):
        rio.load_result(tmp_path / "nope", _Outer)


def test_load_bad_format_version_raises(tmp_path: Path) -> None:
    rio.save_result(_outer(), tmp_path / "outer")
    manifest = tmp_path / "outer" / "_result.json"
    payload = json.loads(manifest.read_text())
    payload["format_version"] = 999
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="format version"):
        rio.load_result(tmp_path / "outer", _Outer)
