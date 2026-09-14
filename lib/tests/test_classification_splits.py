"""The identical-outer-splits guarantee across the classifier templates.

A comparison between two classifier types is **paired** (per outer fold) only if both
were evaluated on the same outer train/test partitions. The three classifier templates
(`classification`, `classification-xgboost`, `classification-svm`) each carry a
byte-identical copy of the CV construction (`_make_cv` / `_split`), so the same
``(n_splits, n_repeats, random_state)`` on the same analyzed samples yields the same
splits — this file turns that implicit "same seed" into a checked invariant:

  * unit — each module's ``_make_cv`` + ``_split`` produce identical index sequences
    on one synthetic ``y`` (ungrouped and grouped), guarding the copies against drift;
  * end-to-end — the three public entry points on one tiny dataset + seed record the
    same ``(repeat, fold, test_indices)`` sequence in their ``fold_predictions``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from analysis import classification as clf
from analysis import classification_svm as svm
from analysis import classification_xgboost as xgb
from common import data_loading as dl
from sklearn.exceptions import ConvergenceWarning

_MODULES: dict[str, Any] = {"elastic-net": clf, "xgboost": xgb, "svm": svm}


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
    ident = _fold_identity(a)
    assert len(ident) == 6
    assert ident == _fold_identity(b) == _fold_identity(c)
    # the same held-out labels, fold for fold -> a paired per-fold comparison is valid
    for fa, fb, fc in zip(
        a.fold_predictions, b.fold_predictions, c.fold_predictions, strict=True
    ):
        np.testing.assert_array_equal(fa.y_true, fb.y_true)
        np.testing.assert_array_equal(fa.y_true, fc.y_true)
