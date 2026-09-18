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

import inspect
import warnings
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
