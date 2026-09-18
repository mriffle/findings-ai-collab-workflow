"""Tests for the shrinkage-LDA classification template + its figures.

Layers (conventions/coding.md + lib/AUTHORING.md):
  * the numerics oracle — the in-module dual-form estimator reproduces scikit-learn's
    ``LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`` at machine precision
    (shrinkage intensity, coefficients, intercepts, scores, probabilities; binary and
    the k-class math kept for v0.2), and a 20,000-feature fit is fast;
  * planted-truth — a separable synthetic signal must be recovered (high AUC, the
    planted features top-ranked with a positive weight), and a pure-noise dataset must
    not beat the label-shuffle null;
  * the LDA-specific contract — deterministic; scores are calibrated log-odds; no
    hyperparameter (no grid, no tuning warnings); per-fold + all-data shrinkage
    recorded, saturation warns; top-k membership frequency semantics (hand-computed);
    fold identity + per-repeat AUCs recorded; the CV class-size guard;
  * the outcome/binarize API (already-binary, Threshold, LevelMap, positive_class);
  * fail-loud guards (NaN, numeric-without-binarize, >2 levels, too few per class,
    class smaller than n_splits, bad top_k) and the constant-feature drop /
    non-log-scale warning;
  * grouping (repeats -> grouped; singletons -> row-level + warning);
  * result invariants and the THREE figures (incl. the conditional null figure);
  * real-5xFAD smoke — genotype classification recovers the AD signal (skips if absent).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from analysis import classification_lda as lda
from common import data_loading as dl
from figures import classification_lda as ldafig
from matplotlib.figure import Figure
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

_FAST: dict[str, Any] = {
    "n_repeats": 2,
    "stability_repeats": 3,
}


def _planted(
    n: int = 48,
    p: int = 120,
    n_signal: int = 6,
    effect: float = 2.0,
    seed: int = 0,
    scale: dl.Scale = "log2",
) -> dl.Dataset:
    """A balanced 2-class Dataset, signal planted (higher in B) in n_signal cols.

    Features share a low-rank co-variation component (as proteins do), so the
    Ledoit-Wolf shrinkage stays well below saturation — i.i.d. columns would shrink
    almost entirely onto the diagonal target and trip the saturation warning.
    """
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    z = rng.normal(size=(n, 5))
    w = rng.normal(size=(5, p))
    x = 0.7 * (z @ w) + rng.normal(size=(n, p))
    x[:, :n_signal] += y[:, None] * effect
    names = np.array([f"F{i}" for i in range(p)])
    meta = pd.DataFrame(
        {
            "grp": np.where(y == 1, "B", "A"),
            "dose": y.astype(float) * 50.0,
            "subject": [f"s{i // 2}" for i in range(n)],
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
def planted_result() -> lda.LDAClassificationResult:
    """A recovered planted-signal result *with* a small null — reused across tests."""
    ds = _planted(seed=0)
    return lda.classify_lda(
        ds,
        "grp",
        run_null=True,
        n_permutations=50,
        null_repeats=2,
        random_state=0,
        **_FAST,
    )


# --------------------------------------------------------------------------- #
# The numerics oracle: the dual-form estimator == scikit-learn, at machine precision
# --------------------------------------------------------------------------- #
def _correlated(n: int, p: int, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 5))
    w = rng.standard_normal((5, p))
    x = z @ w + 0.5 * rng.standard_normal((n, p))
    y = np.arange(n) % k
    x[:, :10] += y[:, None] * 0.8
    return x, y


@pytest.mark.parametrize("n,p", [(30, 20), (40, 500), (60, 1500)])
def test_shrinkage_intensity_matches_sklearn(n: int, p: int) -> None:
    x, _ = _correlated(n, p, 2)
    xc = x - x.mean(0)
    assert lda._ledoit_wolf_shrinkage_gram(xc) == pytest.approx(
        ledoit_wolf_shrinkage(x), rel=1e-10, abs=1e-14
    )


@pytest.mark.parametrize(
    "n,p,k", [(30, 20, 2), (50, 400, 2), (45, 300, 3), (60, 900, 4)]
)
def test_estimator_matches_sklearn(n: int, p: int, k: int) -> None:
    x, y = _correlated(n, p, k)
    x = StandardScaler().fit_transform(x)
    ref = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x, y)
    ours = lda._DualShrinkageLDA().fit(x, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, rtol=1e-8, atol=1e-10)
    xt, _ = _correlated(12, p, k, seed=1)
    np.testing.assert_allclose(
        ours.decision_function(xt), ref.decision_function(xt), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        ours.predict_proba(xt), ref.predict_proba(xt), atol=1e-10
    )
    assert (ours.predict(xt) == ref.predict(xt)).all()
    assert ours.shrinkage_.shape == (k,)
    assert bool(((ours.shrinkage_ >= 0.0) & (ours.shrinkage_ <= 1.0)).all())


def test_p_much_larger_than_n_is_fast() -> None:
    x, y = _correlated(80, 20_000, 2)
    x = StandardScaler().fit_transform(x)
    t0 = time.perf_counter()
    m = lda._DualShrinkageLDA().fit(x, y)
    assert time.perf_counter() - t0 < 5.0  # sklearn would need a 20,000^2 covariance
    assert np.isfinite(m.coef_).all()


def test_estimator_guards() -> None:
    x, _ = _correlated(10, 5, 2)
    with pytest.raises(ValueError, match="two classes"):
        lda._DualShrinkageLDA().fit(x, np.zeros(10, dtype=int))
    with pytest.raises(ValueError, match="at least two samples"):
        lda._DualShrinkageLDA().fit(x, np.array([0] * 9 + [1]))
    # a feature constant within every class (a step that separates them) is handled
    # exactly as sklearn handles it (a zero-std feature keeps scale 1 in the target)
    step = x.copy()
    y = np.array([0, 1] * 5)
    step[:, 0] = y
    ref = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(step, y)
    ours = lda._DualShrinkageLDA().fit(step, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------- #
# Planted truth
# --------------------------------------------------------------------------- #
def test_planted_signal_recovered(planted_result: lda.LDAClassificationResult) -> None:
    res = planted_result
    assert res.cv_auc > 0.9
    planted = {f"F{i}" for i in range(6)}
    # dense model: every analyzed feature is in the table
    assert len(res.coefficients) == res.n_features == 120
    top6 = set(res.coefficients.head(6)["feature"])
    assert len(planted & top6) >= 5  # the planted features dominate the ranking
    # signal is higher in class B (the positive class) -> positive weights
    planted_rows = res.coefficients[res.coefficients["feature"].isin(planted)]
    assert bool((planted_rows["coef"] > 0).all())
    assert res.positive_label == "B"


def test_planted_signal_beats_null(planted_result: lda.LDAClassificationResult) -> None:
    assert planted_result.null_p is not None
    assert planted_result.null_p < 0.05
    assert planted_result.validated_eligible


def test_pure_noise_does_not_beat_null() -> None:
    ds = _planted(n=60, p=100, n_signal=0, seed=3)
    res = lda.classify_lda(
        ds,
        "grp",
        run_null=True,
        n_permutations=100,
        null_repeats=2,
        random_state=1,
        **_FAST,
    )
    assert res.null_p is not None
    assert res.null_p > 0.05  # no real signal -> does not clear the null


def test_stability_metrics_in_range(
    planted_result: lda.LDAClassificationResult,
) -> None:
    coef = planted_result.coefficients
    assert bool(((coef["top_k_frequency"] >= 0) & (coef["top_k_frequency"] <= 1)).all())
    assert bool(
        ((coef["sign_consistency"] >= 0) & (coef["sign_consistency"] <= 1)).all()
    )
    assert bool((coef["coef_q25"] <= coef["coef_median"]).all())
    assert bool((coef["coef_median"] <= coef["coef_q75"]).all())
    # exactly top_k features per resample -> the frequencies sum to top_k
    assert coef["top_k_frequency"].sum() >= planted_result.top_k - 1e-9  # ties add
    assert coef["top_k_frequency"].sum() <= planted_result.n_features
    # planted features are perfectly stable
    planted = coef[coef["feature"].isin({f"F{i}" for i in range(6)})]
    assert bool((planted["top_k_frequency"] == 1.0).all())
    assert bool((planted["sign_consistency"] == 1.0).all())


# --------------------------------------------------------------------------- #
# LDA-specific contract
# --------------------------------------------------------------------------- #
def test_determinism() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=2)
    a = lda.classify_lda(ds, "grp", random_state=0, **_FAST)
    b = lda.classify_lda(ds, "grp", random_state=0, **_FAST)
    assert a.cv_auc == b.cv_auc
    assert a.shrinkage_negative == b.shrinkage_negative
    assert a.repeat_pooled_aucs == b.repeat_pooled_aucs
    pd.testing.assert_frame_equal(a.coefficients, b.coefficients)


def test_scores_are_calibrated_log_odds(
    planted_result: lda.LDAClassificationResult,
) -> None:
    scores = np.concatenate([f.y_score for f in planted_result.fold_predictions])
    assert bool((scores < 0).any()) and bool((scores > 0).any())
    # balanced accuracy is the log-odds thresholded at 0, averaged over outer folds
    per_fold = [
        balanced_accuracy_score(f.y_true, (f.y_score >= 0.0).astype(int))
        for f in planted_result.fold_predictions
    ]
    assert planted_result.cv_balanced_accuracy == pytest.approx(
        float(np.mean(per_fold))
    )
    # the score is a log posterior-odds: its sigmoid is the estimator's posterior
    x, y = _correlated(40, 60, 2)
    x = StandardScaler().fit_transform(x)
    m = lda._DualShrinkageLDA().fit(x, y)
    d = m.decision_function(x)
    np.testing.assert_allclose(1 / (1 + np.exp(-d)), m.predict_proba(x)[:, 1])


def test_fold_identity_and_shrinkage_recorded(
    planted_result: lda.LDAClassificationResult,
) -> None:
    folds = planted_result.fold_predictions
    n_splits, n_repeats = 5, 2
    assert len(folds) == n_splits * n_repeats
    assert [f.repeat for f in folds] == [0] * n_splits + [1] * n_splits
    assert [f.fold for f in folds] == list(range(n_splits)) * n_repeats
    for r in range(n_repeats):
        held_out = np.concatenate([f.test_indices for f in folds if f.repeat == r])
        # within one repeat the test folds partition the analyzed samples
        assert sorted(held_out.tolist()) == list(range(planted_result.n_samples))
    for f in folds:
        assert len(f.test_indices) == len(f.y_true) == len(f.y_score)
        # the per-fold shrinkage pair is recorded (the analogue of the SVM's best_c)
        assert 0.0 <= f.shrinkage_negative <= 1.0
        assert 0.0 <= f.shrinkage_positive <= 1.0
    assert 0.0 <= planted_result.shrinkage_negative <= 1.0
    assert 0.0 <= planted_result.shrinkage_positive <= 1.0


def test_repeat_aucs(planted_result: lda.LDAClassificationResult) -> None:
    res = planted_result
    assert len(res.repeat_aucs) == len(res.repeat_pooled_aucs) == 2
    assert all(0.0 <= a <= 1.0 for a in res.repeat_aucs)
    assert all(0.0 <= a <= 1.0 for a in res.repeat_pooled_aucs)
    # mean-of-fold per repeat averages back to the overall CV AUC
    assert float(np.mean(res.repeat_aucs)) == pytest.approx(res.cv_auc)
    assert min(res.repeat_pooled_aucs) > 0.9  # strong planted signal


def test_shrinkage_saturation_warns_on_iid_features() -> None:
    # i.i.d. columns: the sample covariance carries no structure, Ledoit-Wolf shrinks
    # it almost entirely onto the diagonal target (lambda ~ 0.97 at 60 rows per class)
    # -> the direction is a standardized mean difference and the template says so
    rng = np.random.default_rng(0)
    n, p = 120, 300
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(size=(n, p))
    x[:, :3] += y[:, None] * 2.0
    names = np.array([f"F{i}" for i in range(p)])
    ds = dl.Dataset(
        abundances=x,
        feature_names=names,
        feature_metadata=pd.DataFrame({"feature": names}),
        metadata=pd.DataFrame({"grp": np.where(y == 1, "B", "A")}),
        scale="log2",
    )
    with pytest.warns(lda.ShrinkageSaturationWarning, match="standardized mean"):
        res = lda.classify_lda(ds, "grp", n_splits=5, n_repeats=1, stability_repeats=1)
    assert max(res.shrinkage_negative, res.shrinkage_positive) >= 0.95
    # the correlated planted data does not trip it
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=lda.ShrinkageSaturationWarning)
        clean = lda.classify_lda(_planted(seed=0), "grp", **_FAST)
    assert max(clean.shrinkage_negative, clean.shrinkage_positive) < 0.95


def test_saturation_warning_helper() -> None:
    with pytest.warns(lda.ShrinkageSaturationWarning, match="B \\(lambda = 0.970\\)"):
        lda._warn_if_shrinkage_saturated(0.3, 0.97, "A", "B")
    with pytest.warns(lda.ShrinkageSaturationWarning, match="A .* and B"):
        lda._warn_if_shrinkage_saturated(0.95, 1.0, "A", "B")
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=lda.ShrinkageSaturationWarning)
        lda._warn_if_shrinkage_saturated(0.5, 0.949, "A", "B")


def test_class_size_guard_for_cv() -> None:
    """No inner CV, so the bound is the plain stratified one + 2 rows per train fold."""
    fast: dict[str, Any] = {"n_repeats": 1, "stability_repeats": 1}
    ds = _planted(n=40, p=30, n_signal=3, seed=0)
    ds.metadata["grp"] = ["B"] * 4 + ["A"] * 36
    with pytest.raises(ValueError, match="n_splits"):
        lda.classify_lda(ds, "grp", n_splits=5, **fast)
    # 3 positives, 2 splits: a training fold keeps 3 - ceil(3/2) = 1 < 3 -> no shrinkage
    ds.metadata["grp"] = ["B"] * 3 + ["A"] * 37
    with pytest.raises(ValueError, match="inside every training fold"):
        lda.classify_lda(ds, "grp", n_splits=2, **fast)
    # 5 positives, 5 splits: 5 >= 5 and 5 - 1 = 4 >= 3 -> runs (the SVM's nested
    # guard would refuse this; the LDA has no inner CV)
    ds.metadata["grp"] = ["B"] * 5 + ["A"] * 35
    res = lda.classify_lda(ds, "grp", n_splits=5, **fast)
    assert res.n_positive == 5


def test_top_k_ties_are_all_members() -> None:
    resample = np.array([[3.0, 2.0, 2.0, 1.0], [3.0, 2.0, 2.0, 1.0]])
    member = lda._top_k_membership(resample, top_k=2)
    # rank-2 weight is tied between features 1 and 2: both are members
    np.testing.assert_array_equal(member[0], [True, True, True, False])
    final = np.array([3.0, 2.0, 2.0, 1.0])
    table = lda._coefficient_table(final, resample, np.array(list("ABCD")), 2)
    freq = table.set_index("feature")["top_k_frequency"]
    assert freq["B"] == freq["C"] == 1.0
    assert freq["D"] == 0.0


def test_top_k_frequency_semantics() -> None:
    # 3 resamples x 3 features, hand-computed
    resample = np.array(
        [
            [3.0, -1.0, 0.5],
            [2.0, -2.5, 0.1],
            [1.0, 0.2, 4.0],
        ]
    )
    final = np.array([2.0, -1.0, 3.0])
    names = np.array(["A", "B", "C"])
    table = lda._coefficient_table(final, resample, names, top_k=2).set_index("feature")
    # top-2 per resample: {A,B}, {B,A}, {C,A} -> A 3/3, B 2/3, C 1/3
    assert table.loc["A", "top_k_frequency"] == pytest.approx(1.0)
    assert table.loc["B", "top_k_frequency"] == pytest.approx(2 / 3)
    assert table.loc["C", "top_k_frequency"] == pytest.approx(1 / 3)
    # sign consistency over ALL resamples: A +++ -> 1; B --+ -> 1/3; C +++ -> 1
    assert table.loc["A", "sign_consistency"] == pytest.approx(1.0)
    assert table.loc["B", "sign_consistency"] == pytest.approx(1 / 3)
    assert table.loc["C", "sign_consistency"] == pytest.approx(1.0)
    # medians over all resamples; sorted by |final| descending -> C, A, B
    assert table.loc["A", "coef_median"] == pytest.approx(2.0)
    assert list(table.index) == ["C", "A", "B"]
    assert bool((table["n_resamples"] == 3).all())
    with pytest.raises(ValueError, match="top_k"):
        lda.classify_lda(_planted(n=40, p=30, seed=0), "grp", top_k=0, **_FAST)
    with pytest.raises(ValueError, match="exceeds"):
        lda.classify_lda(_planted(n=40, p=30, seed=0), "grp", top_k=31, **_FAST)


# --------------------------------------------------------------------------- #
# Outcome / binarize API
# --------------------------------------------------------------------------- #
def test_positive_class_override_flips_sign() -> None:
    ds = _planted(seed=0)
    a = lda.classify_lda(ds, "grp", random_state=0, **_FAST)
    b = lda.classify_lda(ds, "grp", positive_class="A", random_state=0, **_FAST)
    assert a.positive_label == "B"
    assert b.positive_label == "A"
    # LDA is exactly symmetric under a label swap: every weight is negated
    fa = a.coefficients.set_index("feature")["coef"]
    fb = b.coefficients.set_index("feature")["coef"].reindex(fa.index)
    np.testing.assert_allclose(fa.to_numpy(), -fb.to_numpy(), rtol=1e-8, atol=1e-10)


def test_threshold_binarize_continuous() -> None:
    ds = _planted(seed=0)
    res = lda.classify_lda(
        ds, "dose", binarize=lda.Threshold(cut=25.0), random_state=0, **_FAST
    )
    assert res.positive_label == "high"
    assert res.negative_label == "low"
    assert res.n_samples == 48
    assert res.cv_auc > 0.9


def test_threshold_drop_band_excludes_middle() -> None:
    rng = np.random.default_rng(5)
    n, p = 60, 40
    dose = np.repeat([0.0, 40.0, 100.0], n // 3)
    x = rng.normal(size=(n, p))
    x[:, 0] += (dose >= 100.0) * 3.0
    ds = dl.Dataset(
        abundances=x,
        feature_names=np.array([f"F{i}" for i in range(p)]),
        feature_metadata=pd.DataFrame({"feature": [f"F{i}" for i in range(p)]}),
        metadata=pd.DataFrame({"dose": dose}),
        scale="log2",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=lda.ShrinkageSaturationWarning)
        res = lda.classify_lda(
            ds,
            "dose",
            binarize=lda.Threshold(cut=100.0, drop_below=1.0, drop_at_or_above=100.0),
            random_state=0,
            **_FAST,
        )
    # the 40.0 middle band (20 samples) is dropped
    assert res.n_dropped_unassigned == 20
    assert res.n_samples == 40


def test_level_map_subset_drops_unlisted() -> None:
    rng = np.random.default_rng(2)
    n, p = 60, 40
    genotype = np.repeat(["WT", "Het", "Hom"], n // 3)
    x = rng.normal(size=(n, p))
    ds = dl.Dataset(
        abundances=x,
        feature_names=np.array([f"F{i}" for i in range(p)]),
        feature_metadata=pd.DataFrame({"feature": [f"F{i}" for i in range(p)]}),
        metadata=pd.DataFrame({"genotype": genotype}),
        scale="log2",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=lda.ShrinkageSaturationWarning)
        res = lda.classify_lda(
            ds,
            "genotype",
            binarize=lda.LevelMap(positive=("Hom",), negative=("WT",)),
            random_state=0,
            **_FAST,
        )
    assert res.n_samples == 40  # Het dropped
    assert res.n_dropped_unassigned == 20


# --------------------------------------------------------------------------- #
# Guards / edges
# --------------------------------------------------------------------------- #
def test_nan_abundance_raises() -> None:
    ds = _planted(seed=0)
    ds.abundances[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        lda.classify_lda(ds, "grp", **_FAST)


def test_numeric_outcome_without_binarize_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match="numeric"):
        lda.classify_lda(ds, "dose", **_FAST)


def test_multiclass_without_binarize_raises() -> None:
    rng = np.random.default_rng(0)
    ds = dl.Dataset(
        abundances=rng.normal(size=(30, 10)),
        feature_names=np.array([f"F{i}" for i in range(10)]),
        feature_metadata=pd.DataFrame({"feature": [f"F{i}" for i in range(10)]}),
        metadata=pd.DataFrame({"g": np.repeat(["a", "b", "c"], 10)}),
        scale="log2",
    )
    with pytest.raises(ValueError, match="binary only"):
        lda.classify_lda(ds, "g", **_FAST)


def test_too_few_per_class_raises() -> None:
    rng = np.random.default_rng(0)
    ds = dl.Dataset(
        abundances=rng.normal(size=(6, 10)),
        feature_names=np.array([f"F{i}" for i in range(10)]),
        feature_metadata=pd.DataFrame({"feature": [f"F{i}" for i in range(10)]}),
        metadata=pd.DataFrame({"g": ["a", "a", "a", "a", "a", "b"]}),
        scale="log2",
    )
    with pytest.raises(ValueError, match="per class"):
        lda.classify_lda(ds, "g", **_FAST)


def test_class_smaller_than_n_splits_raises() -> None:
    ds = _planted(n=24, p=30, n_signal=3, seed=0)
    ds.metadata["grp"] = ["A"] * 4 + ["B"] * 20
    with pytest.raises(ValueError, match="n_splits"):
        lda.classify_lda(ds, "grp", n_splits=5, **_FAST)


def test_bad_null_permutation_raises() -> None:
    ds = _planted(n=40, p=30, seed=0)
    with pytest.raises(ValueError, match="null_permutation must be"):
        lda.classify_lda(ds, "grp", null_permutation="rows", **_FAST)  # type: ignore[arg-type]


def test_constant_features_dropped() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.abundances[:, 10] = 7.0  # a constant feature
    ds.abundances[:, 11] = 0.0  # an all-zero feature
    res = lda.classify_lda(ds, "grp", random_state=0, **_FAST)
    assert res.n_dropped_constant == 2
    assert res.n_features == 28
    assert len(res.coefficients) == 28
    assert "F10" not in set(res.coefficients["feature"])


def test_non_log_scale_warns() -> None:
    ds = _planted(seed=0, scale="linear")
    with pytest.warns(lda.ClassificationScaleWarning):
        lda.classify_lda(ds, "grp", random_state=0, **_FAST)


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def test_grouping_engaged_with_repeats() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    # subject pairs consecutive rows and each subject has ONE label (blocks share y),
    # so the group-level null permutation (one label per unit) is well-defined
    y = np.array([(i // 2) % 2 for i in range(60)])
    rng = np.random.default_rng(1)
    ds.abundances[:, :4] = rng.normal(size=(60, 4)) + y[:, None] * 2.0
    ds.metadata["grp"] = np.where(y == 1, "B", "A")
    ds.metadata["subject"] = [f"s{i // 2}" for i in range(60)]
    res = lda.classify_lda(
        ds,
        "grp",
        groups="subject",
        generalization_target="individuals",
        n_splits=3,
        n_repeats=1,
        stability_repeats=2,
        random_state=0,
        run_null=True,
        n_permutations=5,
        null_repeats=1,
    )
    assert res.grouped is True
    assert res.groups_column == "subject"
    assert res.generalization_target == "individuals"
    assert res.null_p is not None  # the grouped null path ran
    # fold integrity: no subject straddles an outer train/test split
    subject = ds.metadata["subject"].to_numpy()
    for f in res.fold_predictions:
        test_subjects = set(subject[f.test_indices])
        train_mask = np.ones(len(subject), dtype=bool)
        train_mask[f.test_indices] = False
        assert not (test_subjects & set(subject[train_mask]))


def test_single_class_outer_fold_raises() -> None:
    # two units, each carrying one class: grouped 2-fold CV puts one class per test
    # fold, so the fold AUC is undefined -> fail loud, never a silent NaN cv_auc
    ds = _planted(n=40, p=30, n_signal=3, seed=0)
    y = np.array([0] * 20 + [1] * 20)
    rng = np.random.default_rng(3)
    ds.abundances[:, :3] = rng.normal(size=(40, 3)) + y[:, None] * 2.0
    ds.metadata["grp"] = np.where(y == 1, "B", "A")
    ds.metadata["subject"] = np.where(y == 1, "u1", "u0")
    with pytest.raises(ValueError, match="single class"):
        lda.classify_lda(
            ds,
            "grp",
            groups="subject",
            n_splits=2,
            n_repeats=1,
            stability_repeats=1,
        )


def test_grouped_null_rejects_mixed_label_unit() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    # _planted alternates labels within each consecutive pair -> every unit is mixed
    ds.metadata["subject"] = [f"s{i // 2}" for i in range(60)]
    with pytest.raises(ValueError, match=r"both classes.*within_units"):
        lda.classify_lda(
            ds,
            "grp",
            groups="subject",
            run_null=True,
            n_permutations=2,
            n_splits=3,
            n_repeats=1,
            stability_repeats=1,
        )
    # the permutation helper itself refuses too
    y = np.array([0, 1, 1, 1])
    g = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="both classes"):
        lda._permute_labels(y, g, np.random.default_rng(0))
    # and preserves class counts on clean units
    y2 = np.array([0, 0, 1, 1, 1, 1])
    g2 = np.array(["a", "a", "b", "b", "c", "c"])
    perm = lda._permute_labels(y2, g2, np.random.default_rng(0))
    assert perm.sum() == 4


def test_singleton_groups_fall_back_to_rowlevel() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.metadata["subject"] = [f"s{i}" for i in range(40)]  # all unique
    with pytest.warns(lda.SingletonGroupsWarning):
        res = lda.classify_lda(ds, "grp", groups="subject", random_state=0, **_FAST)
    assert res.grouped is False
    assert res.groups_column is None


# --------------------------------------------------------------------------- #
# Figures (three — no tuning figure)
# --------------------------------------------------------------------------- #
def test_all_figures_render(
    planted_result: lda.LDAClassificationResult, tmp_path: Path
) -> None:
    for fig in (
        ldafig.plot_roc(planted_result),
        ldafig.plot_null(planted_result),
        ldafig.plot_coefficients(planted_result),
    ):
        assert isinstance(fig, Figure)
        plt.close(fig)
    art = ldafig.save_roc(planted_result, tmp_path, "roc")
    assert art.svg.exists() and art.png.exists()
    assert art.legend_svg is None  # colorbars/on-axes legends, no separate legend
    art2 = ldafig.save_coefficients(planted_result, tmp_path, "coef")
    assert art2.png.exists()
    assert not hasattr(ldafig, "plot_hyperparameter_curve")  # three figures, by design


def test_roc_annotates_shrinkage(planted_result: lda.LDAClassificationResult) -> None:
    fig = ldafig.plot_roc(planted_result)
    texts = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "Ledoit-Wolf shrinkage" in texts
    assert "calibrated posterior" in texts
    plt.close(fig)


def test_null_figure_requires_null() -> None:
    ds = _planted(seed=0)
    res = lda.classify_lda(ds, "grp", run_null=False, random_state=0, **_FAST)
    with pytest.raises(ValueError, match="null was not run"):
        ldafig.plot_null(res)
    fig = ldafig.plot_roc(res)  # annotates "null not run"
    assert "null not run" in " ".join(t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)


def test_coefficient_figure_top_n_defaults_to_top_k(
    planted_result: lda.LDAClassificationResult,
) -> None:
    fig = ldafig.plot_coefficients(planted_result)
    assert len(fig.axes[0].get_yticks()) == planted_result.top_k
    plt.close(fig)
    fig = ldafig.plot_coefficients(planted_result, top_n=3)
    assert len(fig.axes[0].get_yticks()) == 3
    plt.close(fig)
    with pytest.raises(ValueError, match="top_n"):
        ldafig.plot_coefficients(planted_result, top_n=0)


# --------------------------------------------------------------------------- #
# Prior feature-list restriction (leakage-safe; matched/unmatched counts recorded)
# --------------------------------------------------------------------------- #
def test_feature_list_restricts_and_records() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    res = lda.classify_lda(
        ds,
        "grp",
        feature_list=[f"F{i}" for i in range(10)],
        top_k=5,
        random_state=0,
        **_FAST,
    )
    assert res.n_features_requested == 10
    assert res.n_features_matched == 10
    assert res.n_features <= 10
    assert set(res.coefficients["feature"]) <= {f"F{i}" for i in range(10)}


def test_feature_list_no_match_raises() -> None:
    ds = _planted(n=40, p=20, seed=0)
    with pytest.raises(ValueError, match="matched none"):
        lda.classify_lda(ds, "grp", feature_list=["ZZZ", "QQQ"], **_FAST)


def test_feature_list_poor_match_warns() -> None:
    ds = _planted(n=40, p=20, n_signal=3, seed=0)
    # 2 real + 8 absent -> 2/10 matched, below the 0.5 warn fraction
    flist = [f"F{i}" for i in range(2)] + [f"ABSENT{i}" for i in range(8)]
    with pytest.warns(lda.FeatureListWarning):
        res = lda.classify_lda(
            ds, "grp", feature_list=flist, top_k=1, random_state=0, **_FAST
        )
    assert res.n_features_requested == 10
    assert res.n_features_matched == 2


def test_feature_list_noted_in_figure_title(
    planted_result: lda.LDAClassificationResult,
) -> None:
    from dataclasses import replace

    # whole proteome -> no note
    fig = ldafig.plot_roc(planted_result)
    assert "prior feature list" not in fig.get_suptitle()
    plt.close(fig)
    # full match -> "N features"
    fig = ldafig.plot_roc(
        replace(planted_result, n_features_requested=150, n_features_matched=150)
    )
    assert "prior feature list · 150 features" in fig.get_suptitle()
    plt.close(fig)
    # partial match -> "N of M matched" (reaches a non-ROC figure too)
    fig = ldafig.plot_coefficients(
        replace(planted_result, n_features_requested=150, n_features_matched=142)
    )
    assert "prior feature list · 142 of 150 matched" in fig.get_suptitle()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The null's permutation scheme: unit-level (default) vs within-unit (batches)
# --------------------------------------------------------------------------- #
_NULL_FAST: dict[str, Any] = {
    "top_k": 5,
    "n_splits": 3,
    "n_repeats": 1,
    "stability_repeats": 1,
    "null_repeats": 1,
    "random_state": 0,
}


def _pure_units(ds: dl.Dataset) -> dl.Dataset:
    """Add an ``animal`` column: repeated units that each carry ONE class."""
    y = (ds.metadata["grp"].to_numpy() == "B").astype(int)
    ds.metadata["animal"] = [f"a{y[i]}_{i // 4}" for i in range(len(y))]
    return ds


def test_within_unit_null_planted_signal_and_noise() -> None:
    # _planted alternates labels within each consecutive pair, so its ``subject`` pairs
    # are mixed-label units — exactly a batch-grouped design (each batch holds both
    # classes), the case the within-unit scheme exists for.
    ds = _planted(n=40, p=20, n_signal=4, seed=0)
    res = lda.classify_lda(
        ds,
        "grp",
        groups="subject",
        generalization_target="batches",
        run_null=True,
        null_permutation="within_units",
        n_permutations=19,
        **_NULL_FAST,
    )
    assert res.grouped is True
    assert res.null_permutation == "within_units"
    assert res.null_aucs is not None and res.observed_auc is not None
    assert res.null_p == pytest.approx(1 / 20)  # observed beats all 19 permutations
    assert res.observed_auc > float(np.max(res.null_aucs))
    assert res.validated_eligible
    # pure noise on the same design does not beat its within-unit null
    noise = lda.classify_lda(
        _planted(n=40, p=20, n_signal=0, seed=1),
        "grp",
        groups="subject",
        generalization_target="batches",
        run_null=True,
        null_permutation="within_units",
        n_permutations=19,
        **_NULL_FAST,
    )
    assert noise.null_p is not None and noise.null_p > 0.2


def test_within_unit_permutation_preserves_unit_counts() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    g = np.array(["a"] * 7 + ["b"] * 3 + ["c"] * 4)  # c is a pure unit
    rng = np.random.default_rng(0)
    changed = False
    for _ in range(50):
        perm = lda._permute_labels(y, g, rng, "within_units")
        for u in ("a", "b", "c"):
            assert perm[g == u].sum() == y[g == u].sum()
        np.testing.assert_array_equal(perm[g == "c"], y[g == "c"])  # untouched
        changed |= not np.array_equal(perm, y)
    assert changed
    # the default (unit-level) scheme refuses the mixed units and names the remedy
    with pytest.raises(ValueError, match="within_units"):
        lda._permute_labels(y, g, rng, "units")
    # ungrouped: a plain row shuffle whatever the scheme
    plain = lda._permute_labels(y, None, np.random.default_rng(0), "within_units")
    assert plain.sum() == y.sum() and plain.shape == y.shape


def test_within_unit_arrangements_hand_computed() -> None:
    # two 3/3 batches -> C(6,3)^2 = 400
    y = np.array([0, 0, 0, 1, 1, 1] * 2)
    g = np.array(["a"] * 6 + ["b"] * 6)
    assert lda._within_unit_arrangements(y, g) == 400
    # every unit pure -> exactly one arrangement (the identity)
    assert (
        lda._within_unit_arrangements(np.array([0, 0, 1, 1]), np.array(list("aabb")))
        == 1
    )
    # six batches of 20 with 10 positives each: 184756**6 ~ 4e31 > 2**63 — the count is
    # an exact Python int; an int64 product would overflow silently
    y6 = np.tile(np.array([0] * 10 + [1] * 10), 6)
    g6 = np.repeat([f"b{i}" for i in range(6)], 20)
    assert lda._within_unit_arrangements(y6, g6) == 184756**6


def test_within_unit_null_guards() -> None:
    kw: dict[str, Any] = {
        "run_null": True,
        "null_permutation": "within_units",
        "n_permutations": 5,
        **_NULL_FAST,
    }
    ds = _planted(n=40, p=20, n_signal=4, seed=0)
    # needs grouped CV: no groups column at all
    with pytest.raises(ValueError, match="needs grouped CV"):
        lda.classify_lda(ds, "grp", **kw)
    # singleton groups fall back to row-level (warns) -> still refused, naming the col
    single = _planted(n=40, p=20, n_signal=4, seed=0)
    single.metadata["unit"] = [f"u{i}" for i in range(40)]
    with (
        pytest.warns(lda.SingletonGroupsWarning),
        pytest.raises(ValueError, match="'unit'"),
    ):
        lda.classify_lda(single, "grp", groups="unit", **kw)
    # every unit single-class: the within-unit shuffle is the identity -> refused
    with pytest.raises(ValueError, match="no freedom"):
        lda.classify_lda(
            _pure_units(_planted(n=40, p=20, n_signal=4, seed=0)),
            "grp",
            groups="animal",
            **kw,
        )
    # too few distinct arrangements for n_permutations: warns, still runs. Four batches
    # of 2/2 -> C(4,2)^4 = 1296 arrangements < 1297 requested (2-fold grouped CV keeps
    # 4 rows per class in every training fold, the shrinkage estimator's minimum of 3).
    small = _planted(n=16, p=10, n_signal=3, seed=0)
    small.metadata["batch"] = np.repeat(["a", "b", "c", "d"], 4)
    with (
        pytest.warns(lda.NullPermutationWarning, match="1296 distinct"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", category=lda.ShrinkageSaturationWarning)
        res = lda.classify_lda(
            small,
            "grp",
            groups="batch",
            run_null=True,
            null_permutation="within_units",
            n_permutations=1297,
            **{**_NULL_FAST, "n_splits": 2},
        )
    assert res.null_permutation == "within_units"
    assert res.null_p is not None


def test_null_permutation_recorded_and_titled() -> None:
    ds = _planted(n=40, p=20, n_signal=4, seed=0)
    assert (
        lda.classify_lda(ds, "grp", **_NULL_FAST).null_permutation is None
    )  # null not run
    rows = lda.classify_lda(ds, "grp", run_null=True, n_permutations=3, **_NULL_FAST)
    assert rows.null_permutation == "samples"  # row-level CV: a plain shuffle
    units = lda.classify_lda(
        _pure_units(_planted(n=40, p=20, n_signal=4, seed=0)),
        "grp",
        groups="animal",
        run_null=True,
        n_permutations=3,
        **_NULL_FAST,
    )
    assert units.null_permutation == "units"
    within = lda.classify_lda(
        ds,
        "grp",
        groups="subject",
        run_null=True,
        null_permutation="within_units",
        n_permutations=3,
        **_NULL_FAST,
    )
    assert within.null_permutation == "within_units"
    # the null figure names the scheme so a reviewer sees which null was run
    for res, needle in (
        (rows, "permutations, shrinkage re-estimated"),
        (units, "unit-level"),
        (within, "within-unit"),
    ):
        fig = ldafig.plot_null(res)
        assert needle in fig.get_suptitle()
        plt.close(fig)
    assert "unit-level" not in ldafig.plot_null(rows).get_suptitle()
    plt.close("all")


# --------------------------------------------------------------------------- #
# Real-5xFAD smoke (git-ignored data; skips cleanly without it)
# --------------------------------------------------------------------------- #
_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)


def _load_5xfad_genotype() -> dl.Dataset:
    import dataclasses

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
    meta["Disease"] = np.where(meta["Genotype"].to_numpy() == "5xFAD", "5xFAD", "nonAD")
    return dataclasses.replace(
        logged, abundances=logged.abundances[mask, :], metadata=meta
    )


@pytest.mark.slow
@_skip_no_data
def test_smoke_5xfad_genotype_recovers_ad_signal() -> None:
    """Pins the design-preview numbers (scratch/shrinkage-lda, 2026-09-17)."""
    exp = _load_5xfad_genotype()
    res = lda.classify_lda(exp, "Disease", positive_class="5xFAD", random_state=0)
    assert res.n_samples == 52
    assert res.positive_label == "5xFAD"
    assert res.cv_auc == pytest.approx(0.848, abs=0.05)  # the preview's paired value
    # the covariance carried real structure (the preview: 0.47 / 0.24)
    assert res.shrinkage_negative == pytest.approx(0.47, abs=0.1)
    assert res.shrinkage_positive == pytest.approx(0.24, abs=0.1)
    # the APP transgene / canonical AD proteins dominate the top of the ranking
    top = " ".join(res.coefficients.head(15)["feature"].astype(str)).upper()
    assert ("5XFAD" in top) or ("APOE" in top) or ("A4_MOUSE" in top)


@pytest.mark.slow
@_skip_no_data
def test_smoke_5xfad_cohort_held_out_within_unit_null() -> None:
    """A batch-grouped design on real data: cohort-held-out CV + the within-cohort null.

    5xFAD has two cohorts, each holding both genotypes — the unit-level null refuses
    (a mixed unit), the within-unit null runs and the genotype signal clears it.
    """
    exp = _load_5xfad_genotype()
    common: dict[str, Any] = {
        "positive_class": "5xFAD",
        "groups": "Cohort",
        "generalization_target": "batches",
        "n_splits": 2,
        "n_repeats": 1,
        "stability_repeats": 1,
        "null_repeats": 3,
        "n_permutations": 100,
        "random_state": 0,
        "run_null": True,
    }
    with pytest.raises(ValueError, match=r"both classes.*within_units"):
        lda.classify_lda(exp, "Disease", **common)  # the default scheme refuses
    res = lda.classify_lda(exp, "Disease", null_permutation="within_units", **common)
    assert res.grouped is True and res.groups_column == "Cohort"
    assert res.null_permutation == "within_units"
    assert res.observed_auc is not None and res.null_aucs is not None
    assert res.observed_auc > 0.7
    assert abs(float(np.mean(res.null_aucs)) - 0.5) < 0.05
    assert res.null_p is not None and res.null_p < 0.05
