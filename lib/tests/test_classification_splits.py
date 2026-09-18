"""The identical-outer-splits guarantee across the classifier templates.

A comparison between two classifier types is **paired** (per outer fold) only if both
were evaluated on the same outer train/test partitions. The four classifier templates
(`classification`, `classification-xgboost`, `classification-svm`,
`classification-lda`) each carry a
byte-identical copy of the CV construction (`_make_cv` / `_split`), so the same
``(n_splits, n_repeats, random_state)`` on the same analyzed samples yields the same
splits — this file turns that implicit "same seed" into a checked invariant:

  * unit — each module's ``_make_cv`` + ``_split`` produce identical index sequences
    on one synthetic ``y`` (ungrouped and grouped), guarding the copies against drift;
  * end-to-end — the four public entry points on one tiny dataset + seed record the
    same ``(repeat, fold, test_indices)`` sequence in their ``fold_predictions``.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from analysis import classification as clf
from analysis import classification_lda as lda
from analysis import classification_svm as svm
from analysis import classification_xgboost as xgb
from analysis import regression as reg
from common import data_loading as dl
from sklearn.exceptions import ConvergenceWarning

_MODULES: dict[str, Any] = {"elastic-net": clf, "xgboost": xgb, "svm": svm, "lda": lda}


def _splits(module: Any, grouped: bool) -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(7)
    n = 30
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(size=(n, 3))
    groups = np.array([f"g{i // 2}" for i in range(n)]) if grouped else None
    cv = module._make_cv(3, 2, grouped, 11)
    return [
        (train.tolist(), test.tolist())
        for train, test in module._split(cv, x, y, groups)
    ]


def test_cv_construction_identical_across_templates() -> None:
    for grouped in (False, True):
        reference = _splits(clf, grouped)
        assert len(reference) == 6  # 3 folds x 2 repeats
        for name, module in _MODULES.items():
            assert _splits(module, grouped) == reference, (name, grouped)


def _planted(n: int = 30, p: int = 20, n_signal: int = 3, seed: int = 0) -> dl.Dataset:
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(size=(n, p))
    x[:, :n_signal] += y[:, None] * 2.0
    names = np.array([f"F{i}" for i in range(p)])
    return dl.Dataset(
        abundances=x,
        feature_names=names,
        feature_metadata=pd.DataFrame({"feature": names}),
        metadata=pd.DataFrame({"grp": np.where(y == 1, "B", "A")}),
        scale="log2",
    )


def _fold_identity(result: Any) -> list[tuple[int, int, tuple[int, ...]]]:
    return [
        (f.repeat, f.fold, tuple(int(i) for i in f.test_indices))
        for f in result.fold_predictions
    ]


def test_outer_splits_identical_across_classifiers() -> None:
    ds = _planted()
    common: dict[str, Any] = {
        "n_splits": 3,
        "n_repeats": 2,
        "stability_repeats": 1,
        "n_jobs": 1,
        "random_state": 0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        a = clf.classify(
            ds, "grp", c_grid=[1.0], l1_ratios=[0.5], max_iter=2000, tol=1e-3, **common
        )
    b = xgb.classify_xgboost(
        ds,
        "grp",
        max_depth_grid=[2],
        learning_rate_grid=[0.3],
        n_estimators=20,
        **common,
    )
    c = svm.classify_svm(ds, "grp", c_grid=[1.0], top_k=5, **common)
    lda_common = {
        k: v for k, v in common.items() if k != "n_jobs"
    }  # nothing to parallelize
    d = lda.classify_lda(ds, "grp", top_k=5, **lda_common)
    ident = _fold_identity(a)
    assert len(ident) == 6
    assert ident == _fold_identity(b) == _fold_identity(c) == _fold_identity(d)
    # the same held-out labels, fold for fold -> a paired per-fold comparison is valid
    for fa, fb, fc, fd in zip(
        a.fold_predictions,
        b.fold_predictions,
        c.fold_predictions,
        d.fold_predictions,
        strict=True,
    ):
        np.testing.assert_array_equal(fa.y_true, fb.y_true)
        np.testing.assert_array_equal(fa.y_true, fc.y_true)
        np.testing.assert_array_equal(fa.y_true, fd.y_true)


# --------------------------------------------------------------------------- #
# The null block: byte-identical helpers, identical draws, and the fold-invariance
# property of the within-unit scheme
# --------------------------------------------------------------------------- #
_NULL_HELPERS = (
    "_permute_labels",
    "_unit_labels",
    "_permute_within_units",
    "_within_unit_arrangements",
)

# The label / grouping / CV scaffolding the four classifiers also carry as deliberate
# byte-identical copies (self-contained seeds). A drift in one would make "the same
# analyzed sample set on the same folds" untrue across a paired comparison.
_SHARED_HELPERS = (
    "_is_numeric",
    "_labels_and_missing",
    "_resolve_labels",
    "_resolve_already_binary",
    "_resolve_threshold",
    "_resolve_level_map",
    "_resolve_grouping",
    "_make_cv",
    "_split",
    "_RepeatedStratifiedGroupKFold",
    *_NULL_HELPERS,
)


def test_null_helpers_byte_identical_across_templates() -> None:
    # the four templates carry deliberate copies (self-contained seeds); a drift in one
    # would make "the same null" untrue across a paired comparison
    for name in _NULL_HELPERS:
        sources = {k: inspect.getsource(getattr(m, name)) for k, m in _MODULES.items()}
        assert len(set(sources.values())) == 1, name


def test_shared_helpers_byte_identical_across_templates() -> None:
    for name in _SHARED_HELPERS:
        sources = {k: inspect.getsource(getattr(m, name)) for k, m in _MODULES.items()}
        assert len(set(sources.values())) == 1, name


# --------------------------------------------------------------------------- #
# Missing metadata values: an outcome NA is dropped (never a level), a groups NA
# raises (never a unit). pandas >= 3 keeps NA as a float through astype(str), so the
# old ``labels != "nan"`` sentinel silently stopped working.
# --------------------------------------------------------------------------- #
_MISSING_VALUES: tuple[object, ...] = (None, np.nan, pd.NA)


@pytest.mark.parametrize("module", list(_MODULES.values()), ids=list(_MODULES))
@pytest.mark.parametrize("missing", _MISSING_VALUES, ids=("None", "nan", "NA"))
def test_missing_outcome_value_is_dropped(module: Any, missing: object) -> None:
    for dtype in (object, "string"):
        series = pd.Series(["A", "B", missing, "A", "B", "A"], dtype=dtype)
        labels = module._resolve_labels(series, None, None, "grp")
        assert labels.keep.tolist() == [True, True, False, True, True, True]
        assert labels.y.tolist() == [0, 1, 0, 1, 0]
        assert (labels.positive_label, labels.negative_label) == ("B", "A")
        # LevelMap: a missing value is unassigned, so it is dropped the same way
        lm = module._resolve_labels(
            series, module.LevelMap(positive=("B",), negative=("A",)), None, "grp"
        )
        assert lm.keep.tolist() == labels.keep.tolist()
        assert lm.y.tolist() == labels.y.tolist()


@pytest.mark.parametrize(
    "module",
    [*_MODULES.values(), reg],
    ids=[*_MODULES, "regression"],
)
def test_missing_group_value_raises(module: Any) -> None:
    metadata = pd.DataFrame({"unit": ["a", "a", "b", "b", None, None, "c", "c"]})
    keep = np.ones(len(metadata), dtype=bool)
    with pytest.raises(ValueError, match=r"missing value.*positions \[4, 5\]"):
        module._resolve_grouping("unit", metadata, keep)
    # a missing unit on a sample the binarize rule already dropped is not an error
    keep[4:6] = False
    assert module._resolve_grouping("unit", metadata, keep) is not None


def _batch_design() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    n = 60
    groups = np.repeat([f"b{i}" for i in range(6)], 10)
    y = np.zeros(n, dtype=int)
    for b, k in enumerate([7, 5, 2, 6, 4, 5]):  # mixed, unbalanced batches
        y[b * 10 : b * 10 + k] = 1
    return rng.normal(size=(n, 3)), y, groups


def test_null_draws_identical_across_templates() -> None:
    _, y, groups = _batch_design()
    pure_groups = np.array([f"u{i // 2}" for i in range(len(y))])
    y_pure = np.repeat([0, 1] * 15, 2)
    for scheme in ("units", "within_units"):
        for g, yy in ((None, y), (groups, y), (pure_groups, y_pure)):
            if scheme == "units" and g is groups:
                continue  # mixed units are refused under "units" (tested per file)
            draws = [
                m._permute_labels(yy, g, np.random.default_rng(5), scheme).tolist()
                for m in _MODULES.values()
            ]
            assert all(d == draws[0] for d in draws[1:]), (scheme, g is None)


def test_within_unit_permutation_keeps_grouped_folds_fixed() -> None:
    # StratifiedGroupKFold assigns units to folds from their class-count vectors, which
    # within-unit permutation preserves — so every null draw is scored on the observed
    # folds and only the labels move; a unit-level permutation moves the folds
    x, y, groups = _batch_design()
    cv = svm._make_cv(3, 3, True, 11)
    reference = [(a.tolist(), b.tolist()) for a, b in svm._split(cv, x, y, groups)]
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_perm = svm._permute_labels(y, groups, rng, "within_units")
        assert [
            (a.tolist(), b.tolist()) for a, b in svm._split(cv, x, y_perm, groups)
        ] == reference
    pure_groups = np.array([f"u{i // 2}" for i in range(len(y))])
    y_pure = np.repeat([0, 1] * 15, 2)
    ref_pure = [
        (a.tolist(), b.tolist()) for a, b in svm._split(cv, x, y_pure, pure_groups)
    ]
    moved = 0
    for _ in range(20):
        y_perm = svm._permute_labels(y_pure, pure_groups, rng, "units")
        cur = [
            (a.tolist(), b.tolist()) for a, b in svm._split(cv, x, y_perm, pure_groups)
        ]
        moved += cur != ref_pure
    assert moved > 0


# --------------------------------------------------------------------------- #
# The tuning block (v0.4): the inner tuning CV is grouped whenever the outer CV is,
# and the tuning table is indexed by the recorded parameter values, never by position
# --------------------------------------------------------------------------- #
_TUNED: dict[str, Any] = {
    "elastic-net": clf,
    "svm": svm,
    "xgboost": xgb,
    "regression": reg,
}
_TUNING_HELPERS = (
    "_make_inner_cv",
    "_check_inner_folds",
    "_check_class_sizes_for_nested_cv",
)
_GRID_HELPERS = ("_grid_table",)
_GRID_2D: dict[str, Any] = {k: m for k, m in _TUNED.items() if k != "svm"}


def test_tuning_helpers_byte_identical_across_classifiers() -> None:
    for name in _TUNING_HELPERS:
        sources = {
            k: inspect.getsource(getattr(m, name))
            for k, m in _TUNED.items()
            if k != "regression"  # its splitters are KFold / GroupKFold
        }
        assert len(set(sources.values())) == 1, name
    for name in _GRID_HELPERS:
        sources = {k: inspect.getsource(getattr(m, name)) for k, m in _GRID_2D.items()}
        assert len(set(sources.values())) == 1, name


class _RecordingSplitter:
    """Wraps a real inner splitter; asserts every split keeps every unit intact."""

    def __init__(self, inner: Any, expect_groups: bool) -> None:
        self.inner = inner
        self.expect_groups = expect_groups
        self.n_split_calls = 0

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return int(self.inner.get_n_splits(X, y, groups))

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Any:
        self.n_split_calls += 1
        assert (groups is not None) == self.expect_groups
        for train, test in self.inner.split(X, y, groups):
            if groups is not None:
                assert not set(groups[train]) & set(groups[test])
            yield train, test


def _replicated_units(p: int = 40) -> dl.Dataset:
    """20 units x 3 near-identical replicates, one class per unit, no class signal.

    A row-level inner split memorizes the twin replicate (tuning AUC 1.0 on pure
    noise); a grouped inner split cannot.
    """
    rng = np.random.default_rng(0)
    unit = np.repeat(np.arange(20), 3)
    x = rng.normal(0, 2.0, (20, p))[unit] + rng.normal(0, 0.01, (60, p))
    names = np.array([f"F{i}" for i in range(p)])
    return dl.Dataset(
        abundances=x,
        feature_names=names,
        feature_metadata=pd.DataFrame({"feature": names}),
        metadata=pd.DataFrame(
            {
                "grp": np.where(unit % 2 == 0, "A", "B"),
                "animal": [f"a{u}" for u in unit],
                "t": rng.normal(size=20)[unit],
            }
        ),
        scale="log2",
    )


def _run_tuned(module: Any, ds: dl.Dataset, **extra: Any) -> Any:
    common: dict[str, Any] = {
        "groups": "animal",
        "generalization_target": "individuals",
        "n_splits": 3,
        "n_repeats": 1,
        "stability_repeats": 1,
        "n_jobs": 1,
        "random_state": 0,
    }
    common.update(extra)
    if module is clf:
        return clf.classify(
            ds,
            "grp",
            c_grid=[0.1, 1.0],
            l1_ratios=[0.5, 1.0],
            max_iter=2000,
            tol=1e-3,
            **common,
        )
    if module is svm:
        return svm.classify_svm(ds, "grp", c_grid=[0.1, 1.0], top_k=5, **common)
    if module is lda:
        common.pop("n_jobs")  # nothing to parallelize
        return lda.classify_lda(ds, "grp", top_k=5, **common)
    if module is xgb:
        return xgb.classify_xgboost(
            ds,
            "grp",
            max_depth_grid=[2, 3],
            learning_rate_grid=[0.1, 0.3],
            n_estimators=20,
            **common,
        )
    return reg.regress(
        ds,
        "t",
        alpha_grid=[0.1, 1.0],
        l1_ratios=[0.5, 1.0],
        max_iter=2000,
        tol=1e-3,
        **common,
    )


@pytest.mark.parametrize("module", list(_TUNED.values()), ids=list(_TUNED))
def test_inner_cv_never_splits_a_unit(module: Any, monkeypatch: Any) -> None:
    ds = _replicated_units()
    real = module._make_inner_cv
    made: list[_RecordingSplitter] = []

    def recording(grouped: bool, n_splits: int, random_state: int) -> Any:
        rec = _RecordingSplitter(real(grouped, n_splits, random_state), grouped)
        made.append(rec)
        return rec

    monkeypatch.setattr(module, "_make_inner_cv", recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        res = _run_tuned(module, ds)
    # one inner splitter per outer training fold + one for the all-data fit
    assert len(made) == 3 + 1
    assert all(m.expect_groups and m.n_split_calls >= 1 for m in made)
    assert res.inner_cv_grouped is True
    # ungrouped run: no groups reach the inner splitter, and the result says so
    made.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        res = _run_tuned(module, ds, groups=None, generalization_target="samples")
    assert len(made) == 4
    assert not any(m.expect_groups for m in made)
    assert res.inner_cv_grouped is False


@pytest.mark.parametrize("module", list(_TUNED.values()), ids=list(_TUNED))
def test_grouped_inner_cv_does_not_leak_replicates(module: Any) -> None:
    ds = _replicated_units()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        res = _run_tuned(module, ds)
    fixed = float(np.nanmax(res.grid_scores))
    # the same all-data search with a row-level inner split: the twin replicate is in
    # the inner training half, so the tuning surface reads near-perfect on pure noise
    grouped_kind = type(module._make_inner_cv(True, 3, 0))
    row_level = module._make_inner_cv(False, 3, 0)
    from sklearn.model_selection import GridSearchCV

    x = ds.abundances
    if module is reg:
        y = ds.metadata["t"].to_numpy(dtype=float)
        est = module._build_pipeline(0.1, 0.5, 2000, 1e-3, 0)
        grid: dict[str, list[Any]] = {
            "en__alpha": [0.1, 1.0],
            "en__l1_ratio": [0.5, 1.0],
        }
        scoring = "r2"
    else:
        y = (ds.metadata["grp"] == "B").to_numpy().astype(int)
        if module is clf:
            est = module._build_pipeline(None, 0.5, 2000, 1e-3, 0)
            grid = {"lr__C": [0.1, 1.0], "lr__l1_ratio": [0.5, 1.0]}
        elif module is svm:
            est = module._build_pipeline(0.1, 1e-3, -1)
            grid = {"svm__C": [0.1, 1.0]}
        else:
            defaults = {
                k: v.default
                for k, v in inspect.signature(
                    module.classify_xgboost
                ).parameters.items()
            }
            defaults.update(
                max_depth_grid=(2, 3),
                learning_rate_grid=(0.1, 0.3),
                n_estimators=20,
                n_jobs=1,
                random_state=0,
            )
            cfg = module._Config(
                **{f.name: defaults[f.name] for f in dataclasses.fields(module._Config)}
            )
            est = module._make_xgb(2, 0.1, module._scale_pos_weight(y), cfg, n_jobs=1)
            grid = {"max_depth": [2, 3], "learning_rate": [0.1, 0.3]}
        scoring = "roc_auc"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        leaked = float(
            np.nanmax(
                GridSearchCV(est, grid, cv=row_level, scoring=scoring, n_jobs=1)
                .fit(x, y)
                .cv_results_["mean_test_score"]
            )
        )
    assert grouped_kind is not type(row_level)
    if module is reg:
        assert leaked > fixed + 0.3, (fixed, leaked)  # R² on noise: leaked > 0 > fixed
    else:
        assert leaked > 0.95, leaked  # the leak the fix removes
        assert fixed < 0.8, fixed


def test_grid_table_is_index_based() -> None:
    rows, cols = (0.1, 1.0, 10.0), (0.25, 0.5)
    # ParameterGrid order for keys ("param_a", "param_b"): a outer, b fastest
    cells = [(r, c) for r in rows for c in cols]
    scores = [float(i) / 10 for i in range(len(cells))]
    expected = np.asarray(scores).reshape(3, 2)

    def results(order: list[int]) -> dict[str, Any]:
        return {
            "mean_test_score": np.asarray([scores[i] for i in order]),
            "param_a": np.asarray([cells[i][0] for i in order]),
            "param_b": np.asarray([cells[i][1] for i in order]),
        }

    natural = list(range(len(cells)))
    shuffled = [5, 0, 3, 1, 4, 2]
    for module in _GRID_2D.values():
        for order in (natural, shuffled):
            table = module._grid_table(results(order), "param_a", rows, "param_b", cols)
            np.testing.assert_array_equal(table, expected)
        with pytest.raises(RuntimeError, match="twice"):
            module._grid_table(
                results([0, 0, 2, 3, 4, 5]), "param_a", rows, "param_b", cols
            )
        with pytest.raises(RuntimeError, match="unevaluated"):
            module._grid_table(results(natural[:-1]), "param_a", rows, "param_b", cols)
        with pytest.raises(RuntimeError, match="not in the grid"):
            module._grid_table(
                results(natural), "param_a", (0.1, 1.0, 7.0), "param_b", cols
            )


def test_select_cell_masks_failed_cells_and_refuses_all_nan() -> None:
    grid = np.array([[np.nan, 0.9], [0.9, 0.5]])
    # the old smoothed rule averaged the NaN cell's neighbours to 0.9 and picked it;
    # a cell whose own fit failed is never a candidate: hand-smoothed surface is
    # [[nan, 0.7], [0.7, 0.767]] -> (1, 1)
    for module, rows in ((clf, (1.0, 2.0)), (xgb, (1, 2)), (reg, (1.0, 2.0))):
        assert module._select_cell(grid, rows, (0.5, 1.0), "smoothed") == (rows[1], 1.0)
        assert module._select_cell(grid, rows, (0.5, 1.0), "best") == (rows[0], 1.0)
        with pytest.raises(ValueError, match="all non-finite"):
            module._select_cell(np.full((2, 2), np.nan), rows, (0.5, 1.0), "best")


@pytest.mark.parametrize("module", list(_GRID_2D.values()), ids=list(_GRID_2D))
def test_smoothed_selection_used_in_outer_folds(module: Any, monkeypatch: Any) -> None:
    ds = _replicated_units()
    real = module._select_cell
    seen: list[str] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(args[-1] if args else kwargs["select"])
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_select_cell", spy)
    extra: dict[str, Any] = {
        "select": "smoothed",
        "groups": None,
        "generalization_target": "samples",
    }
    if module is clf:
        extra["l1_ratios"] = [0.25, 0.5, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        _run_tuned(module, ds, **extra) if module is not clf else clf.classify(
            ds,
            "grp",
            c_grid=[0.1, 1.0],
            l1_ratios=[0.25, 0.5, 1.0],
            max_iter=2000,
            tol=1e-3,
            n_splits=3,
            n_repeats=1,
            stability_repeats=1,
            n_jobs=1,
            random_state=0,
            select="smoothed",
        )
    # one selection per outer fold + one for the all-data fit, every one smoothed
    assert seen == ["smoothed"] * 4


@pytest.mark.parametrize("module", list(_GRID_2D.values()), ids=list(_GRID_2D))
def test_select_is_validated(module: Any) -> None:
    ds = _replicated_units()
    with pytest.raises(ValueError, match="select must be"):
        _run_tuned(
            module, ds, groups=None, generalization_target="samples", select="smooth"
        )
    with pytest.raises(ValueError, match="at least 3 cells"):
        if module is clf:
            clf.classify(
                ds,
                "grp",
                c_grid=[0.1, 1.0],
                l1_ratios=[0.5],
                select="smoothed",
                n_jobs=1,
            )
        elif module is xgb:
            xgb.classify_xgboost(
                ds,
                "grp",
                max_depth_grid=[2],
                learning_rate_grid=[0.1, 0.3],
                select="smoothed",
                n_jobs=1,
            )
        else:
            reg.regress(
                ds,
                "t",
                alpha_grid=[0.1, 1.0],
                l1_ratios=[0.5],
                select="smoothed",
                n_jobs=1,
            )


@pytest.mark.parametrize("module", list(_TUNED.values()), ids=list(_TUNED))
def test_result_records_tuning_metric(module: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        res = _run_tuned(
            module, _replicated_units(), groups=None, generalization_target="samples"
        )
    assert res.tuning_metric == ("r2" if module is reg else "roc_auc")
    assert res.inner_cv_grouped is False


@pytest.mark.parametrize("module", [clf, xgb], ids=["elastic-net", "xgboost"])
def test_grid_scores_pinned_to_search_params(module: Any) -> None:
    """Every table cell equals the inner-CV score of the parameter pair it is labelled
    with — an asymmetric, non-monotone grid, so a transposed/scrambled reshape cannot
    pass by accident (the XGBoost bug: ParameterGrid enumerates learning_rate outer)."""
    from sklearn.model_selection import cross_val_score

    ds = _planted()
    y = (ds.metadata["grp"] == "B").to_numpy().astype(int)
    inner = module._make_inner_cv(False, 3, 0)
    res: Any
    if module is clf:
        rows, cols = (0.05, 1.0), (1.0, 0.25, 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            res = clf.classify(
                ds,
                "grp",
                c_grid=rows,
                l1_ratios=cols,
                n_splits=3,
                n_repeats=1,
                stability_repeats=1,
                n_jobs=1,
                max_iter=2000,
                tol=1e-3,
            )

        def estimator(r: float, c: float) -> Any:
            return clf._build_pipeline(r, c, 2000, 1e-3, 0)
    else:
        rows, cols = (1, 6), (0.3, 0.01, 0.1)
        res = xgb.classify_xgboost(
            ds,
            "grp",
            max_depth_grid=rows,
            learning_rate_grid=cols,
            n_estimators=20,
            n_splits=3,
            n_repeats=1,
            stability_repeats=1,
            n_jobs=1,
        )
        defaults = {
            k: v.default
            for k, v in inspect.signature(xgb.classify_xgboost).parameters.items()
        }
        defaults.update(
            max_depth_grid=rows,
            learning_rate_grid=cols,
            n_estimators=20,
            n_jobs=1,
            random_state=0,
        )
        cfg = xgb._Config(
            **{f.name: defaults[f.name] for f in dataclasses.fields(xgb._Config)}
        )
        spw = xgb._scale_pos_weight(y)

        def estimator(r: float, c: float) -> Any:
            return xgb._make_xgb(int(r), c, spw, cfg, n_jobs=1)

    assert res.grid_scores.shape == (len(rows), len(cols))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                cell = float(
                    np.mean(
                        cross_val_score(
                            estimator(r, c),
                            ds.abundances,
                            y,
                            cv=inner,
                            scoring="roc_auc",
                        )
                    )
                )
                assert abs(res.grid_scores[i, j] - cell) < 1e-9, (r, c)
    # and the selected cell is the table's first maximum (row-major)
    ij = np.unravel_index(int(np.nanargmax(res.grid_scores)), res.grid_scores.shape)
    i, j = int(ij[0]), int(ij[1])
    if module is clf:
        assert (res.best_c, res.best_l1_ratio) == (rows[i], cols[j])
    else:
        assert (res.best_params["max_depth"], res.best_params["learning_rate"]) == (
            rows[i],
            cols[j],
        )


# --------------------------------------------------------------------------- #
# The null path (v0.4): a fold the scorer cannot score raises, a non-finite score
# never becomes a permutation p, and the counts are validated up front
# --------------------------------------------------------------------------- #
_FIVE: dict[str, Any] = {**_MODULES, "regression": reg}
_ENTRY: dict[str, Any] = {
    "elastic-net": clf.classify,
    "xgboost": xgb.classify_xgboost,
    "svm": svm.classify_svm,
    "lda": lda.classify_lda,
    "regression": reg.regress,
}


def _config(module: Any, **over: Any) -> Any:
    """A template's private ``_Config`` from its public defaults (+ overrides)."""
    entry = _ENTRY[next(k for k, m in _FIVE.items() if m is module)]
    defaults = {k: v.default for k, v in inspect.signature(entry).parameters.items()}
    alias = {"l1_grid": "l1_ratios"}
    values = {}
    for f in dataclasses.fields(module._Config):
        src = over.get(
            f.name, defaults.get(f.name, defaults.get(alias.get(f.name, "")))
        )
        values[f.name] = tuple(src) if isinstance(src, list) else src
    return module._Config(**values)


def _two_pure_units() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 8))
    y = np.repeat([0, 1], 6)
    groups = np.repeat(["u0", "u1"], 6)
    return x, y, groups


@pytest.mark.parametrize("module", list(_FIVE.values()), ids=list(_FIVE))
def test_null_scorer_refuses_a_fold_it_cannot_score(module: Any) -> None:
    """Two single-class units under 2-fold grouped CV: each null test fold holds one
    class. scikit-learn scores that as NaN with a warning; the scorer raises instead
    (a NaN observed AUC would have read as p = 1 / (n_permutations + 1))."""
    x, y, groups = _two_pure_units()
    cfg = _config(module, n_splits=2, null_repeats=1, n_jobs=1, random_state=0)
    call: Callable[[], float]
    if module is clf:
        call = lambda: clf._fixed_cv_auc(x, y, groups, 1.0, 0.5, cfg)  # noqa: E731
    elif module is svm:
        call = lambda: svm._fixed_cv_auc(x, y, groups, 1.0, cfg)  # noqa: E731
    elif module is xgb:
        call = lambda: xgb._fixed_cv_auc(x, y, groups, 2, 0.3, cfg)  # noqa: E731
    elif module is lda:
        call = lambda: lda._cv_auc(x, y, groups, cfg)  # noqa: E731
    else:
        yy = y.astype(float)
        call = lambda: reg._fixed_cv_r2(x, yy, groups, 0.1, 0.5, cfg)  # noqa: E731
    with pytest.raises(ValueError, match="null-CV test fold holds a"):
        call()


@pytest.mark.parametrize("module", list(_FIVE.values()), ids=list(_FIVE))
def test_non_finite_null_score_raises(module: Any, monkeypatch: Any) -> None:
    """A NaN from the scorer must not become a permutation p."""
    scorer = (
        "_cv_auc"
        if module is lda
        else ("_fixed_cv_r2" if module is reg else "_fixed_cv_auc")
    )
    monkeypatch.setattr(module, scorer, lambda *a, **k: float("nan"))
    ds = _replicated_units()
    with pytest.raises(ValueError, match="not finite"):
        _run_tuned(
            module,
            ds,
            groups=None,
            generalization_target="samples",
            run_null=True,
            n_permutations=2,
            null_repeats=1,
        )


@pytest.mark.parametrize("module", list(_FIVE.values()), ids=list(_FIVE))
def test_cv_counts_are_validated(module: Any) -> None:
    ds = _replicated_units()
    base: dict[str, Any] = {"groups": None, "generalization_target": "samples"}
    for bad in (
        {"n_permutations": 0},
        {"null_repeats": 0},
        {"n_splits": 1},
        {"n_repeats": 0},
        {"stability_repeats": 0},
    ):
        with pytest.raises(ValueError, match="must be >= "):
            _run_tuned(module, ds, **base, **bad)
