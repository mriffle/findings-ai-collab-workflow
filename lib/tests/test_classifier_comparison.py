"""Tests for the paired classifier-comparison template + its figures.

Layers (conventions/coding.md + lib/AUTHORING.md):
  * planted-truth — stub results built on the classifier templates' own
    ``_make_cv`` / ``_split`` folds (so fold identity is the real thing) with scores
    engineered to give hand-computable per-fold AUCs: the reference ranks every fold
    perfectly (AUC 1.0) and the alternative has exactly ``k`` discordant (positive,
    negative) pairs per fold (AUC ``1 - k / (n_pos · n_neg)``);
  * the four real result classes pair (ENxSVM, ENxLDA, ENxXGB) — ``y_prob`` and
    ``y_score`` both read;
  * fail-loud identity checks, each with its own message; ties with tolerance;
  * the corrected resampled t pinned to hand-computed numbers (incl. the uneven-fold
    mean-of-ratios ≠ ratio-of-means case) + the zero-variance definition;
  * the indicative Wilcoxon only over non-zero differences (``None`` below five);
  * the reported ``cv_auc`` is recorded but never trusted;
  * the seed rule (consistent sign / a flip is a tie / input guards);
  * a ``result-io`` round-trip of both result classes; the two figures render and
    dual-export with a legend image, close on an error path, and keep the summary
    strip under a custom title.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from analysis import classification as clf
from analysis import classification_lda as lda
from analysis import classification_svm as svm
from analysis import classification_xgboost as xgb
from analysis import classifier_comparison as cc
from analysis.result_io import load_result, save_result
from figures import classifier_comparison as ccfig
from matplotlib.figure import Figure
from scipy import stats

Kind = str  # "en" | "svm" | "lda" | "xgb"

_N_SAMPLES = 40
_N_SPLITS = 5
_N_REPEATS = 2
_EPS = 1e-3


# --------------------------------------------------------------------------- #
# Stub construction on the templates' real folds
# --------------------------------------------------------------------------- #
def _labels(n_samples: int = _N_SAMPLES) -> np.ndarray:
    return np.array([0, 1] * (n_samples // 2), dtype=int)


def _real_folds(
    seed: int, n_samples: int = _N_SAMPLES
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Outer splits exactly as the classifier templates build them."""
    y = _labels(n_samples)
    x = np.zeros((n_samples, 1))
    cv = clf._make_cv(_N_SPLITS, _N_REPEATS, False, seed)
    return clf._split(cv, x, y, None)


def _perfect_scores(y_true: np.ndarray) -> np.ndarray:
    """Strictly rank-separated scores: positives 1 + j·eps, negatives j·eps."""
    scores = np.empty(len(y_true), dtype=float)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    scores[pos] = 1.0 + np.arange(len(pos)) * _EPS
    scores[neg] = np.arange(len(neg)) * _EPS
    return scores


def _degraded_scores(y_true: np.ndarray, k: int) -> np.ndarray:
    """Exactly ``k`` discordant pairs: ``k`` positives dropped just beneath the
    top-ranked negative (above every other negative), so each is discordant with
    that one negative only — AUC = 1 - k / (n_pos · n_neg), hand-computable.
    """
    scores = _perfect_scores(y_true)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if k > len(pos):
        raise ValueError("k exceeds the number of positives in the fold")
    top_neg = float(scores[neg].max())
    for j in range(k):
        scores[pos[j]] = top_neg - _EPS / (j + 3)  # still above the other negatives
    return scores


def _fold_objects(
    kind: Kind,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    scores_for: Any,
    *,
    repeat_override: int | None = None,
) -> list[Any]:
    folds: list[Any] = []
    for i, (_, test) in enumerate(splits):
        y_true = y[test]
        score = scores_for(i, y_true)
        repeat = i // _N_SPLITS if repeat_override is None else repeat_override
        fold = i % _N_SPLITS
        idx = np.asarray(test, dtype=int).copy()
        if kind == "en":
            folds.append(clf.FoldPrediction(y_true, score, repeat, fold, idx))
        elif kind == "xgb":
            folds.append(xgb.FoldPrediction(y_true, score, repeat, fold, idx))
        elif kind == "svm":
            folds.append(svm.FoldPrediction(y_true, score, repeat, fold, idx, 1.0))
        elif kind == "lda":
            folds.append(lda.FoldPrediction(y_true, score, repeat, fold, idx, 0.1, 0.1))
        else:
            raise ValueError(kind)
    return folds


def _stub_result(
    kind: Kind,
    folds: list[Any],
    *,
    seed: int,
    n_samples: int = _N_SAMPLES,
    cv_auc: float | None = None,
    outcome: str = "grp",
    positive_label: str = "B",
    negative_label: str = "A",
) -> Any:
    """A REAL result dataclass of the given template with minimal dummy fields."""
    if cv_auc is None:
        cv_auc = float(cc.fold_aucs(_Holder(folds)).mean())
    common: dict[str, Any] = {
        "fold_predictions": folds,
        "cv_auc": cv_auc,
        "cv_auc_sd": 0.0,
        "cv_balanced_accuracy": 0.0,
        "cv_average_precision": 0.0,
        "outcome": outcome,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "generalization_target": "samples",
        "grouped": False,
        "groups_column": None,
        "n_samples": n_samples,
        "n_positive": n_samples // 2,
        "n_negative": n_samples // 2,
        "n_features": 3,
        "n_dropped_constant": 0,
        "n_dropped_unassigned": 0,
        "random_state": seed,
    }
    table = pd.DataFrame({"feature": ["f0"], "coef": [1.0]})
    if kind == "en":
        return clf.ClassificationResult(
            coefficients=table,
            best_c=1.0,
            best_l1_ratio=0.5,
            grid_scores=np.zeros((1, 1)),
            c_grid=(1.0,),
            l1_grid=(0.5,),
            **common,
        )
    if kind == "svm":
        return svm.SVMClassificationResult(
            coefficients=table,
            repeat_aucs=(cv_auc,),
            repeat_pooled_aucs=(cv_auc,),
            best_c=1.0,
            grid_scores=np.zeros(1),
            grid_scores_sd=np.zeros(1),
            c_grid=(1.0,),
            top_k=1,
            n_support_vectors=2,
            n_support_negative=1,
            n_support_positive=1,
            **common,
        )
    if kind == "lda":
        return lda.LDAClassificationResult(
            coefficients=table,
            repeat_aucs=(cv_auc,),
            repeat_pooled_aucs=(cv_auc,),
            shrinkage_negative=0.1,
            shrinkage_positive=0.1,
            top_k=1,
            **common,
        )
    if kind == "xgb":
        return xgb.XGBClassificationResult(
            importances=table,
            best_params={"max_depth": 2.0, "learning_rate": 0.3},
            grid_scores=np.zeros((1, 1)),
            max_depth_grid=(2,),
            learning_rate_grid=(0.3,),
            importance_type="gain",
            **common,
        )
    raise ValueError(kind)


class _Holder:
    """Minimal fold carrier so ``fold_aucs`` can be used before a result exists."""

    def __init__(self, folds: list[Any]) -> None:
        self.fold_predictions = folds
        self.cv_auc = 0.0
        self.random_state = 0
        self.n_samples = _N_SAMPLES
        self.outcome = "grp"
        self.positive_label = "B"
        self.negative_label = "A"
        self.generalization_target = "samples"
        self.grouped = False


def _planted_pair(
    seed: int,
    swaps_per_fold: Sequence[int],
    *,
    ref_kind: Kind = "en",
    alt_kind: Kind = "svm",
) -> tuple[Any, Any]:
    """Reference with AUC 1.0 per fold; alternative with ``swaps_per_fold[i]``
    discordant pairs in fold ``i`` (AUC ``1 - k / (n_pos · n_neg)``)."""
    splits = _real_folds(seed)
    if len(swaps_per_fold) != len(splits):
        raise ValueError("swaps_per_fold must have one entry per outer fold")
    y = _labels()
    ref_folds = _fold_objects(ref_kind, splits, y, lambda i, yt: _perfect_scores(yt))
    alt_folds = _fold_objects(
        alt_kind, splits, y, lambda i, yt: _degraded_scores(yt, swaps_per_fold[i])
    )
    return _stub_result(ref_kind, ref_folds, seed=seed), _stub_result(
        alt_kind, alt_folds, seed=seed
    )


def _expected_diffs(seed: int, swaps_per_fold: Sequence[int]) -> np.ndarray:
    y = _labels()
    out = []
    for k, (_, test) in zip(swaps_per_fold, _real_folds(seed), strict=True):
        yt = y[test]
        n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())
        out.append(-k / (n_pos * n_neg))
    return np.asarray(out)


_SWAPS = [1, 2, 0, 1, 3, 0, 2, 1, 1, 0]  # 10 folds = 5 splits x 2 repeats


# --------------------------------------------------------------------------- #
# Planted truth
# --------------------------------------------------------------------------- #
def test_planted_difference_recovered() -> None:
    ref, alt = _planted_pair(0, _SWAPS)
    result = cc.compare_classifiers(ref, alt, reading="module")
    expected = _expected_diffs(0, _SWAPS)
    assert result.n_folds == 10
    np.testing.assert_allclose(result.per_fold["auc_reference"], 1.0)
    np.testing.assert_allclose(result.per_fold["diff"], expected, atol=1e-12)
    np.testing.assert_allclose(result.per_fold["auc_alternative"], 1.0 + expected)
    # each fold holds out 8 of 40 (4 positives + 4 negatives): n_train = 32
    assert set(result.per_fold["n_test"]) == {8}
    assert set(result.per_fold["n_train"]) == {32}
    assert result.rho == pytest.approx(0.25)
    assert result.n_losses == sum(k > 0 for k in _SWAPS) == 7
    assert result.n_ties == sum(k == 0 for k in _SWAPS) == 3
    assert result.n_wins == 0
    assert result.mean_diff == pytest.approx(expected.mean())
    assert result.sd_diff == pytest.approx(expected.std(ddof=1))
    assert result.reference_auc == 1.0
    assert result.alternative_auc == pytest.approx(1.0 + expected.mean())
    t, df, p = cc.corrected_resampled_t(expected, [32] * 10, [8] * 10)
    assert (result.corrected_t, result.corrected_t_df, result.corrected_t_p) == (
        t,
        df,
        p,
    )
    assert result.preferred == "reference"
    assert result.reading == "module"
    assert result.reference_label == "classification"
    assert result.alternative_label == "classification-svm"
    assert list(result.per_fold["repeat"]) == [0] * 5 + [1] * 5
    assert list(result.per_fold["fold"]) == list(range(5)) * 2


def test_all_four_result_types_pair() -> None:
    for alt_kind, label in (
        ("svm", "classification-svm"),
        ("lda", "classification-lda"),
        ("xgb", "classification-xgboost"),
    ):
        ref, alt = _planted_pair(3, _SWAPS, ref_kind="en", alt_kind=alt_kind)
        result = cc.compare_classifiers(ref, alt, reading="sparse")
        assert result.alternative_label == label
        assert result.reference_label == "classification"
        np.testing.assert_allclose(
            result.per_fold["diff"], _expected_diffs(3, _SWAPS), atol=1e-12
        )
    # custom labels override the derived ones
    ref, alt = _planted_pair(3, _SWAPS)
    result = cc.compare_classifiers(
        ref, alt, reading="sparse", reference_label="EN", alternative_label="SVM"
    )
    assert (result.reference_label, result.alternative_label) == ("EN", "SVM")


# --------------------------------------------------------------------------- #
# Fold identity
# --------------------------------------------------------------------------- #
def test_identity_mismatch_raises() -> None:
    ref, alt = _planted_pair(0, _SWAPS)
    splits = _real_folds(0)
    y = _labels()

    # seed
    alt_seed = _stub_result("svm", alt.fold_predictions, seed=1)
    with pytest.raises(cc.FoldIdentityError, match="random_state differs"):
        cc.compare_classifiers(ref, alt_seed, reading="module")

    # n_samples
    alt_n = _stub_result("svm", alt.fold_predictions, seed=0, n_samples=41)
    with pytest.raises(cc.FoldIdentityError, match="n_samples differ"):
        cc.compare_classifiers(ref, alt_n, reading="module")

    # permuted test_indices in one fold (same set, different order -> not identical)
    folds = list(alt.fold_predictions)
    f0 = folds[0]
    folds[0] = svm.FoldPrediction(
        f0.y_true, f0.y_score, f0.repeat, f0.fold, f0.test_indices[::-1].copy(), 1.0
    )
    with pytest.raises(cc.FoldIdentityError, match="fold 0 differs"):
        cc.compare_classifiers(
            ref, _stub_result("svm", folds, seed=0), reading="module"
        )

    # fold count
    with pytest.raises(cc.FoldIdentityError, match="fold counts differ"):
        cc.compare_classifiers(
            ref, _stub_result("svm", folds[:-1], seed=0), reading="module"
        )

    # y_true differs although indices match
    folds = list(alt.fold_predictions)
    f0 = folds[0]
    flipped = f0.y_true.copy()
    flipped[0] = 1 - flipped[0]
    folds[0] = svm.FoldPrediction(
        flipped, f0.y_score, f0.repeat, f0.fold, f0.test_indices, 1.0
    )
    with pytest.raises(cc.FoldIdentityError, match=r"held-out labels .* differ"):
        cc.compare_classifiers(
            ref, _stub_result("svm", folds, seed=0), reading="module"
        )

    # repeat == -1: a pre-v0.2 cache carries no identity
    old = _fold_objects(
        "en", splits, y, lambda i, yt: _perfect_scores(yt), repeat_override=-1
    )
    with pytest.raises(cc.FoldIdentityError, match="no fold identity"):
        cc.fold_identity(_stub_result("en", old, seed=0))
    with pytest.raises(cc.FoldIdentityError, match="no fold identity"):
        cc.compare_classifiers(_stub_result("en", old, seed=0), alt, reading="module")

    # positive_label
    alt_lab = _stub_result("svm", alt.fold_predictions, seed=0, positive_label="Z")
    with pytest.raises(cc.FoldIdentityError, match="positive_label differs"):
        cc.compare_classifiers(ref, alt_lab, reading="module")

    # outcome
    alt_out = _stub_result("svm", alt.fold_predictions, seed=0, outcome="other")
    with pytest.raises(cc.FoldIdentityError, match="outcome differs"):
        cc.compare_classifiers(ref, alt_out, reading="module")


def test_fold_identity_matches_the_templates_record() -> None:
    ref, _ = _planted_pair(5, _SWAPS)
    ident = cc.fold_identity(ref)
    splits = _real_folds(5)
    assert len(ident) == 10
    for i, (repeat, fold, idx) in enumerate(ident):
        assert (repeat, fold) == (i // 5, i % 5)
        assert idx == tuple(int(j) for j in splits[i][1])


# --------------------------------------------------------------------------- #
# Ties, the corrected t, the Wilcoxon
# --------------------------------------------------------------------------- #
def test_ties_counted_with_tolerance() -> None:
    swaps = [1] * 10  # every fold: diff = -1/16 = -0.0625
    ref, alt = _planted_pair(0, swaps)
    strict = cc.compare_classifiers(ref, alt, reading="module")
    assert (strict.n_wins, strict.n_losses, strict.n_ties) == (0, 10, 0)
    assert strict.preferred == "reference"
    loose = cc.compare_classifiers(ref, alt, reading="module", tie_tolerance=0.1)
    assert (loose.n_wins, loose.n_losses, loose.n_ties) == (0, 0, 10)
    assert set(loose.per_fold["outcome"]) == {"tie"}
    assert loose.preferred == "tie"
    assert loose.tie_tolerance == 0.1
    # identical results -> every fold an exact tie at the default tolerance
    same = cc.compare_classifiers(ref, ref, reading="sparse")
    assert same.n_ties == 10 and same.preferred == "tie"
    with pytest.raises(ValueError, match="tie_tolerance"):
        cc.compare_classifiers(ref, alt, reading="module", tie_tolerance=-1.0)


def test_corrected_resampled_t_pinned() -> None:
    diff = [0.1, 0.2, 0.0, 0.1, 0.1]
    t, df, p = cc.corrected_resampled_t(diff, [32] * 5, [8] * 5)
    # var(ddof=1) = 0.005, rho = 0.25, se = sqrt(0.005 * (0.2 + 0.25)) = 0.047434
    assert np.var(diff, ddof=1) == pytest.approx(0.005)
    se = math.sqrt(0.005 * (1 / 5 + 0.25))
    assert se == pytest.approx(0.047434, abs=1e-6)
    assert t == pytest.approx(0.1 / se)
    assert t == pytest.approx(2.1082, abs=1e-4)
    assert df == 4
    assert p == pytest.approx(2 * stats.t.sf(2.1082, 4), abs=1e-4)
    assert p == pytest.approx(0.1027, abs=1e-4)
    # uneven folds: rho is the MEAN of per-fold ratios, not the ratio of totals
    n_train = [30, 32, 34, 32, 32]
    n_test = [10, 8, 6, 8, 8]
    rho_mean = float(np.mean(np.asarray(n_test) / np.asarray(n_train)))
    rho_totals = sum(n_test) / sum(n_train)
    assert rho_mean == pytest.approx(0.2519608, abs=1e-6)
    assert rho_totals == pytest.approx(0.25)
    assert rho_mean != pytest.approx(rho_totals, abs=1e-6)
    t2, _, _ = cc.corrected_resampled_t(diff, n_train, n_test)
    assert t2 == pytest.approx(0.1 / math.sqrt(0.005 * (0.2 + rho_mean)))
    assert t2 != pytest.approx(t, abs=1e-6)
    # guards
    with pytest.raises(ValueError, match=">= 2 folds"):
        cc.corrected_resampled_t([0.1], [32], [8])
    with pytest.raises(ValueError, match="must match"):
        cc.corrected_resampled_t(diff, [32] * 4, [8] * 5)
    with pytest.raises(ValueError, match="positive"):
        cc.corrected_resampled_t(diff, [0] * 5, [8] * 5)


def test_zero_variance_diff() -> None:
    assert cc.corrected_resampled_t([0.05] * 4, [32] * 4, [8] * 4) == (math.inf, 3, 0.0)
    assert cc.corrected_resampled_t([-0.05] * 4, [32] * 4, [8] * 4) == (
        -math.inf,
        3,
        0.0,
    )
    assert cc.corrected_resampled_t([0.0] * 4, [32] * 4, [8] * 4) == (0.0, 3, 1.0)
    ref, _ = _planted_pair(0, _SWAPS)
    same = cc.compare_classifiers(ref, ref, reading="sparse")
    assert (same.corrected_t, same.corrected_t_p) == (0.0, 1.0)
    assert same.sd_diff == 0.0 and same.mean_diff == 0.0
    assert same.wilcoxon_p is None and same.n_nonzero_diffs == 0
    # every fold identically better: t = +inf, p = 0 (defined, never NaN)
    ref2, alt2 = _planted_pair(0, [1] * 10)
    r = cc.compare_classifiers(alt2, ref2, reading="sparse")  # alternative = perfect
    assert r.corrected_t == math.inf and r.corrected_t_p == 0.0
    assert r.preferred == "alternative"


def test_wilcoxon_on_nonzero_diffs_only() -> None:
    four = [1, 0, 0, 0, 0, 1, 1, 0, 1, 0]
    ref, alt = _planted_pair(0, four)
    r4 = cc.compare_classifiers(ref, alt, reading="module")
    assert r4.n_nonzero_diffs == 4 and r4.wilcoxon_p is None
    five = [1, 0, 0, 0, 2, 1, 1, 0, 1, 0]
    ref, alt = _planted_pair(0, five)
    r5 = cc.compare_classifiers(ref, alt, reading="module")
    assert r5.n_nonzero_diffs == 5 and r5.wilcoxon_p is not None
    diff = r5.per_fold["diff"].to_numpy()
    expected = float(stats.wilcoxon(diff[diff != 0]).pvalue)
    assert r5.wilcoxon_p == pytest.approx(expected)
    assert 0.0 < r5.wilcoxon_p <= 1.0


def test_reported_cv_auc_not_trusted() -> None:
    ref, alt = _planted_pair(0, _SWAPS)
    lying = _stub_result("en", ref.fold_predictions, seed=0, cv_auc=0.99)
    with pytest.warns(cc.ReportedAUCWarning, match="reported cv_auc"):
        result = cc.compare_classifiers(lying, alt, reading="module")
    assert result.reference_auc == 1.0  # recomputed from the held-out scores
    assert result.reference_reported_cv_auc == 0.99  # recorded, never used
    assert result.alternative_reported_cv_auc == pytest.approx(result.alternative_auc)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cc.compare_classifiers(ref, alt, reading="module")  # honest -> no warning


# --------------------------------------------------------------------------- #
# The seed rule
# --------------------------------------------------------------------------- #
def test_multi_seed_sign_consistent() -> None:
    pairs = [_planted_pair(s, _SWAPS) for s in (0, 1, 2)]
    ms = cc.compare_across_seeds(pairs, reading="module")
    assert ms.seeds == (0, 1, 2)
    assert ms.per_seed_sign == (-1, -1, -1)
    assert ms.sign_consistent and ms.verdict == "reference"
    assert ms.reading == "module"
    assert (ms.reference_label, ms.alternative_label) == (
        "classification",
        "classification-svm",
    )
    assert len(ms.comparisons) == 3
    assert list(ms.seed_table.columns) == [
        "seed",
        "mean_diff",
        "wins",
        "losses",
        "ties",
        "corrected_t_p",
        "wilcoxon_p",
    ]
    assert list(ms.seed_table["seed"]) == [0, 1, 2]
    np.testing.assert_allclose(ms.seed_table["mean_diff"], ms.per_seed_mean_diff)
    # the alternative ahead at every seed -> "alternative"
    flipped = [(alt, ref) for ref, alt in pairs]
    assert cc.compare_across_seeds(flipped, reading="sparse").verdict == "alternative"


def test_multi_seed_sign_flip_is_tie() -> None:
    ref0, alt0 = _planted_pair(0, _SWAPS)
    ref1, alt1 = _planted_pair(1, _SWAPS)
    # seed 0: alternative worse; seed 1: alternative better (sides swapped)
    ms = cc.compare_across_seeds(
        [(ref0, alt0), (alt1, ref1)],
        reading="module",
        reference_label="ref",
        alternative_label="alt",
    )
    assert ms.per_seed_sign == (-1, 1)
    assert not ms.sign_consistent and ms.verdict == "tie"
    # a zero-sign seed also breaks consistency
    ref2, alt2 = _planted_pair(2, [0] * 10)
    ms2 = cc.compare_across_seeds([(ref0, alt0), (ref2, alt2)], reading="module")
    assert ms2.per_seed_sign == (-1, 0)
    assert ms2.verdict == "tie"
    assert math.isnan(ms2.seed_table["wilcoxon_p"].iloc[1])


def test_multi_seed_rejects_duplicate_seed_and_single_pair() -> None:
    pair0 = _planted_pair(0, _SWAPS)
    with pytest.raises(ValueError, match=">= 2"):
        cc.compare_across_seeds([pair0], reading="module")
    with pytest.raises(ValueError, match="distinct"):
        cc.compare_across_seeds([pair0, pair0], reading="module")


# --------------------------------------------------------------------------- #
# result-io round-trip
# --------------------------------------------------------------------------- #
def _assert_same_comparison(
    a: cc.ClassifierComparisonResult, b: cc.ClassifierComparisonResult
) -> None:
    pd.testing.assert_frame_equal(a.per_fold, b.per_fold)
    for name in (
        "reference_label",
        "alternative_label",
        "reading",
        "n_folds",
        "n_wins",
        "n_losses",
        "n_ties",
        "tie_tolerance",
        "mean_diff",
        "sd_diff",
        "reference_auc",
        "alternative_auc",
        "reference_reported_cv_auc",
        "alternative_reported_cv_auc",
        "rho",
        "corrected_t",
        "corrected_t_df",
        "corrected_t_p",
        "wilcoxon_p",
        "n_nonzero_diffs",
        "random_state",
        "n_samples",
        "outcome",
        "positive_label",
        "negative_label",
        "generalization_target",
        "grouped",
    ):
        assert getattr(a, name) == getattr(b, name), name
    assert a.preferred == b.preferred


def test_result_io_round_trip_single_and_multi(tmp_path: Path) -> None:
    ref, alt = _planted_pair(0, _SWAPS)
    single = cc.compare_classifiers(ref, alt, reading="module")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no schema warning on a current result
        save_result(single, tmp_path / "single")
        back = load_result(tmp_path / "single", cc.ClassifierComparisonResult)
    _assert_same_comparison(single, back)
    assert back.wilcoxon_p is not None

    ms = cc.compare_across_seeds(
        [_planted_pair(s, _SWAPS) for s in (0, 1)], reading="module"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_result(ms, tmp_path / "multi")
        back_ms = load_result(tmp_path / "multi", cc.MultiSeedComparisonResult)
    pd.testing.assert_frame_equal(ms.seed_table, back_ms.seed_table)
    assert back_ms.seeds == ms.seeds and isinstance(back_ms.seeds, tuple)
    assert back_ms.per_seed_sign == ms.per_seed_sign
    assert back_ms.per_seed_mean_diff == ms.per_seed_mean_diff
    assert (back_ms.sign_consistent, back_ms.verdict) == (
        ms.sign_consistent,
        ms.verdict,
    )
    assert isinstance(back_ms.comparisons, tuple) and len(back_ms.comparisons) == 2
    for a, b in zip(ms.comparisons, back_ms.comparisons, strict=True):
        _assert_same_comparison(a, b)
    # an inf statistic must survive the JSON manifest, and None a "none" entry
    same = cc.compare_classifiers(alt, ref, reading="sparse")  # the wilcoxon is None
    inf_case = cc.compare_classifiers(
        alt, _planted_pair(0, [1] * 10)[0], reading="sparse"
    )
    for name, res in (("same", same), ("inf", inf_case)):
        save_result(res, tmp_path / name)
        _assert_same_comparison(
            res, load_result(tmp_path / name, cc.ClassifierComparisonResult)
        )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_figures_render_and_dual_export(tmp_path: Path) -> None:
    ref, alt = _planted_pair(0, _SWAPS)
    single = cc.compare_classifiers(ref, alt, reading="module")
    fig, legend = ccfig.plot_paired_folds(single)
    assert isinstance(fig, Figure) and isinstance(legend, Figure)
    assert len(fig.axes) == 2
    plt.close(fig)
    plt.close(legend)
    arts = ccfig.save_paired_folds(single, tmp_path / "cmp", "paired_folds")
    for path in (arts.svg, arts.png, arts.legend_svg, arts.legend_png):
        assert path is not None and path.exists() and path.stat().st_size > 0
    assert arts.legend_png == tmp_path / "cmp" / "paired_folds.legend.png"

    ms = cc.compare_across_seeds(
        [_planted_pair(s, _SWAPS) for s in (0, 1, 2)], reading="module"
    )
    fig_ms, legend_ms = ccfig.plot_paired_folds_multi_seed(ms)
    assert len(fig_ms.axes) == 3
    assert "verdict: reference" in fig_ms.get_suptitle()
    plt.close(fig_ms)
    plt.close(legend_ms)
    arts_ms = ccfig.save_paired_folds_multi_seed(ms, tmp_path / "cmp", "multi_seed")
    for path in (arts_ms.svg, arts_ms.png, arts_ms.legend_svg, arts_ms.legend_png):
        assert path is not None and path.exists() and path.stat().st_size > 0
    assert plt.get_fignums() == []  # save_figure closed everything

    # a tie verdict is stated as such; a None wilcoxon renders as n/a
    ref1, alt1 = _planted_pair(1, _SWAPS)
    tie = cc.compare_across_seeds(
        [(ref, alt), (alt1, ref1)],
        reading="sparse",
        reference_label="ref",
        alternative_label="alt",
    )
    fig_tie, legend_tie = ccfig.plot_paired_folds_multi_seed(tie)
    assert "verdict: tie" in fig_tie.get_suptitle()
    plt.close(fig_tie)
    plt.close(legend_tie)
    same = cc.compare_classifiers(ref, ref, reading="sparse")
    fig_same, legend_same = ccfig.plot_paired_folds(same)
    texts = [t.get_text() for t in fig_same.texts]
    assert any("n/a" in t for t in texts)
    plt.close(fig_same)
    plt.close(legend_same)


def test_figure_error_path_closes_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    ref, alt = _planted_pair(0, _SWAPS)
    single = cc.compare_classifiers(ref, alt, reading="module")
    ms = cc.compare_across_seeds(
        [_planted_pair(s, _SWAPS) for s in (0, 1)], reading="module"
    )
    plt.close("all")

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("planted failure")

    monkeypatch.setattr(ccfig, "_draw_differences", boom)
    with pytest.raises(RuntimeError, match="planted failure"):
        ccfig.plot_paired_folds(single)
    assert plt.get_fignums() == []
    with pytest.raises(RuntimeError, match="planted failure"):
        ccfig.plot_paired_folds_multi_seed(ms)
    assert plt.get_fignums() == []
    monkeypatch.undo()
    # a failure while building the legend closes the main figure too
    monkeypatch.setattr(ccfig, "_legend_figure", boom)
    with pytest.raises(RuntimeError, match="planted failure"):
        ccfig.plot_paired_folds(single)
    assert plt.get_fignums() == []


def test_custom_title_keeps_summary_strip() -> None:
    ref, alt = _planted_pair(0, _SWAPS)
    single = cc.compare_classifiers(ref, alt, reading="module")
    fig, legend = ccfig.plot_paired_folds(single, title="My title")
    try:
        assert fig.get_suptitle() == "My title"
        strip = [t.get_text() for t in fig.texts if "corrected-t p" in t.get_text()]
        assert len(strip) == 1
        text = strip[0]
        assert "W/L/T" in text and "0/7/3" in text
        assert "Wilcoxon p (indicative)" in text
        assert "reading (pre-stated): module" in text
        assert "seed 0" in text
        # the strip sits below the axes, never over the data
        strip_artist = next(t for t in fig.texts if "corrected-t p" in t.get_text())
        _, y = strip_artist.get_position()
        assert y < min(ax.get_position().y0 for ax in fig.axes)
    finally:
        plt.close(fig)
        plt.close(legend)
    fig_ms, legend_ms = ccfig.plot_paired_folds_multi_seed(
        cc.compare_across_seeds(
            [_planted_pair(s, _SWAPS) for s in (0, 1)], reading="module"
        ),
        title="Custom multi",
    )
    try:
        assert fig_ms.get_suptitle() == "Custom multi"
        assert any("the seed rule" in t.get_text() for t in fig_ms.texts)
    finally:
        plt.close(fig_ms)
        plt.close(legend_ms)
