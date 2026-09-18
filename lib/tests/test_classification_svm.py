"""Tests for the linear-SVM classification template + its figures.

Layers (conventions/coding.md + lib/AUTHORING.md):
  * planted-truth — a separable synthetic signal must be recovered (high AUC, the
    planted features top-ranked with a positive weight), and a pure-noise dataset must
    not beat the label-shuffle null;
  * the SVM-specific contract — deterministic; scores are margins (not probabilities);
    a 1-D C grid with the smallest-C tie-break on a plateau; a single-element grid pins
    the hard-margin limit; top-k membership frequency semantics (hand-computed); fold
    identity + per-repeat AUCs recorded;
  * the outcome/binarize API (already-binary, Threshold, LevelMap, positive_class);
  * fail-loud guards (NaN, numeric-without-binarize, >2 levels, too few per class,
    class smaller than n_splits, bad C grid, bad top_k) and the constant-feature drop /
    non-log-scale warning;
  * grouping (repeats -> grouped; singletons -> row-level + warning);
  * result invariants and the four figures (incl. the conditional null figure);
  * real-5xFAD smoke — genotype classification recovers the AD signal (skips if absent).

No ConvergenceWarning filter: SVC with ``max_iter=-1`` never warns, so any warning
should surface.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from analysis import classification_svm as svm
from common import data_loading as dl
from figures import classification_svm as svmfig
from matplotlib.figure import Figure
from sklearn.metrics import balanced_accuracy_score

_FAST: dict[str, Any] = {
    "c_grid": [0.1, 1.0],
    "n_repeats": 2,
    "stability_repeats": 3,
    "n_jobs": 1,
}


def _planted(
    n: int = 48,
    p: int = 120,
    n_signal: int = 6,
    effect: float = 2.0,
    seed: int = 0,
    scale: dl.Scale = "log2",
) -> dl.Dataset:
    """A balanced 2-class Dataset, signal planted (higher in B) in n_signal cols."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    x = rng.normal(size=(n, p))
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
def planted_result() -> svm.SVMClassificationResult:
    """A recovered planted-signal result *with* a small null — reused across tests."""
    ds = _planted(seed=0)
    return svm.classify_svm(
        ds,
        "grp",
        run_null=True,
        n_permutations=50,
        null_repeats=2,
        random_state=0,
        **_FAST,
    )


# --------------------------------------------------------------------------- #
# Planted truth
# --------------------------------------------------------------------------- #
def test_planted_signal_recovered(planted_result: svm.SVMClassificationResult) -> None:
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


def test_planted_signal_beats_null(planted_result: svm.SVMClassificationResult) -> None:
    assert planted_result.null_p is not None
    assert planted_result.null_p < 0.05
    assert planted_result.validated_eligible


def test_pure_noise_does_not_beat_null() -> None:
    ds = _planted(n=60, p=100, n_signal=0, seed=3)
    res = svm.classify_svm(
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
    planted_result: svm.SVMClassificationResult,
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
# SVM-specific contract
# --------------------------------------------------------------------------- #
def test_determinism() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=2)
    a = svm.classify_svm(ds, "grp", random_state=0, **_FAST)
    b = svm.classify_svm(ds, "grp", random_state=0, **_FAST)
    assert a.cv_auc == b.cv_auc
    assert a.best_c == b.best_c
    assert a.repeat_pooled_aucs == b.repeat_pooled_aucs
    pd.testing.assert_frame_equal(a.coefficients, b.coefficients)


def test_scores_are_margins_not_probabilities(
    planted_result: svm.SVMClassificationResult,
) -> None:
    scores = np.concatenate([f.y_score for f in planted_result.fold_predictions])
    assert bool((scores < 0).any())  # a probability never is; a margin often is
    # balanced accuracy is the margin thresholded at 0, averaged over outer folds
    per_fold = [
        balanced_accuracy_score(f.y_true, (f.y_score >= 0.0).astype(int))
        for f in planted_result.fold_predictions
    ]
    expected = float(np.mean(per_fold))
    assert planted_result.cv_balanced_accuracy == pytest.approx(expected)


def test_fold_identity_recorded(planted_result: svm.SVMClassificationResult) -> None:
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
        assert f.best_c in planted_result.c_grid  # the per-fold tuned C is recorded


def test_repeat_aucs(planted_result: svm.SVMClassificationResult) -> None:
    res = planted_result
    assert len(res.repeat_aucs) == len(res.repeat_pooled_aucs) == 2
    assert all(0.0 <= a <= 1.0 for a in res.repeat_aucs)
    assert all(0.0 <= a <= 1.0 for a in res.repeat_pooled_aucs)
    # mean-of-fold per repeat averages back to the overall nested-CV AUC
    assert float(np.mean(res.repeat_aucs)) == pytest.approx(res.cv_auc)
    assert min(res.repeat_pooled_aucs) > 0.9  # strong planted signal


def test_tuning_noise_diagnostic_recorded(
    planted_result: svm.SVMClassificationResult,
) -> None:
    res = planted_result
    assert res.plateau_start_c in res.c_grid
    # the plateau start is the smallest C within tolerance of the all-data maximum
    assert res.plateau_start_c == svm._select_c(res.grid_scores, res.c_grid, "best")
    assert res.n_folds_sub_plateau is not None
    expected = sum(f.best_c < res.plateau_start_c for f in res.fold_predictions)
    assert res.n_folds_sub_plateau == expected
    assert 0 <= res.n_folds_sub_plateau <= len(res.fold_predictions)
    fig = svmfig.plot_hyperparameter_curve(res)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert any(lab.startswith("per-fold tuned C") for lab in labels)
    plt.close(fig)


def test_tuning_noise_warning_threshold() -> None:
    with pytest.warns(svm.TuningNoiseWarning, match="6 of 10"):
        svm._warn_if_tuning_noisy(6, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=svm.TuningNoiseWarning)
        svm._warn_if_tuning_noisy(2, 10)  # 20 % is under the 25 % threshold
        svm._warn_if_tuning_noisy(0, 0)


def test_class_size_guard_accounts_for_inner_cv() -> None:
    """The inner CV runs on the outer training fold, with fewer minority samples."""
    fast: dict[str, Any] = {
        "c_grid": [1.0],
        "n_repeats": 1,
        "stability_repeats": 1,
        "n_jobs": 1,
    }
    for n_pos in (5, 6):  # pass the naive >= n_splits check, fail the nested one
        ds = _planted(n=40, p=30, n_signal=3, seed=0)
        ds.metadata["grp"] = ["B"] * n_pos + ["A"] * (40 - n_pos)
        with pytest.raises(ValueError, match="inside every outer training fold"):
            svm.classify_svm(ds, "grp", n_splits=5, **fast)
    ds = _planted(n=40, p=30, n_signal=3, seed=0)
    ds.metadata["grp"] = ["B"] * 7 + ["A"] * 33  # 7 - ceil(7/5) = 5 >= 5 -> runs
    res = svm.classify_svm(ds, "grp", n_splits=5, **fast)
    assert res.n_positive == 7


def test_select_c_ignores_failed_fits() -> None:
    grid = (0.1, 1.0, 10.0)
    scores = np.array([np.nan, 0.9, 0.9])  # the C=0.1 fit failed
    assert svm._select_c(scores, grid, "best") == 1.0
    assert svm._select_c(scores, grid, "smoothed") == 1.0  # not the NaN C


def test_tuning_noise_gate_skips_upper_edge() -> None:
    grid = (1e-3, 1e-2, 1e-1)
    # a plateau below the top edge: the sub-plateau count is meaningful
    assert svm._should_warn_tuning_noise(1e-2, grid)
    assert svm._should_warn_tuning_noise(1e-3, grid)
    # the plateau start is the upper edge: the curve was still rising -> no noise
    # warning (CGridEdgeWarning's "extend upward" is the relevant advice)
    assert not svm._should_warn_tuning_noise(1e-1, grid)
    # a single-element grid (a deliberate pin) never warns
    assert not svm._should_warn_tuning_noise(1e6, (1e6,))
    # under select="smoothed" the gate keys on the unsmoothed plateau start, so a
    # raw curve that peaks only at the top edge is recognised even when the smoothed
    # pick lands one step lower
    raw = np.array([0.80, 0.85, 0.95])
    assert svm._select_c(raw, grid, "best") == 1e-1
    assert svm._select_c(raw, grid, "smoothed") == 1e-1
    assert not svm._should_warn_tuning_noise(svm._select_c(raw, grid, "best"), grid)


def test_bad_select_raises() -> None:
    ds = _planted(n=40, p=30, seed=0)
    with pytest.raises(ValueError, match="select"):
        svm.classify_svm(ds, "grp", select="bogus", **_FAST)  # type: ignore[arg-type]


def test_sub_plateau_count_hand_built() -> None:
    grid = (0.01, 0.1, 1.0, 10.0)
    scores = np.array([0.80, 0.90, 0.90, 0.90])
    plateau_start = svm._select_c(scores, grid, "best")
    assert plateau_start == 0.1
    # smoothed curve [0.85, 0.867, 0.9, 0.9] -> a smoothed pick can sit above it
    assert svm._select_c(scores, grid, "smoothed") == 1.0
    folds = [
        svm.FoldPrediction(
            y_true=np.array([0, 1]),
            y_score=np.array([-1.0, 1.0]),
            repeat=0,
            fold=i,
            test_indices=np.array([i]),
            best_c=c,
        )
        for i, c in enumerate((0.01, 0.01, 0.1, 1.0, 10.0))
    ]
    assert svm._count_sub_plateau(folds, plateau_start) == 2  # strictly below 0.1


def test_top_k_ties_are_all_members() -> None:
    resample = np.array([[3.0, 2.0, 2.0, 1.0], [3.0, 2.0, 2.0, 1.0]])
    member = svm._top_k_membership(resample, top_k=2)
    # rank-2 weight is tied between features 1 and 2: both are members
    np.testing.assert_array_equal(member[0], [True, True, True, False])
    final = np.array([3.0, 2.0, 2.0, 1.0])
    table = svm._coefficient_table(final, resample, np.array(list("ABCD")), 2)
    freq = table.set_index("feature")["top_k_frequency"]
    assert freq["B"] == freq["C"] == 1.0
    assert freq["D"] == 0.0


def test_support_vector_counts(planted_result: svm.SVMClassificationResult) -> None:
    res = planted_result
    assert res.n_support_vectors == res.n_support_negative + res.n_support_positive
    assert 1 <= res.n_support_negative <= res.n_negative
    assert 1 <= res.n_support_positive <= res.n_positive


def test_single_c_pins_hard_margin() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    res = svm.classify_svm(
        ds, "grp", c_grid=[1e6], n_repeats=1, stability_repeats=2, n_jobs=1
    )
    assert res.best_c == 1e6
    assert res.c_grid == (1e6,)
    assert res.grid_scores.shape == res.grid_scores_sd.shape == (1,)
    assert res.cv_auc > 0.9
    fig = svmfig.plot_hyperparameter_curve(res)  # a single point renders
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_tie_breaks_to_smallest_c() -> None:
    grid = (0.1, 1.0, 10.0, 100.0)
    plateau = np.array([0.90, 0.95, 0.95, 0.95])
    # "best": the first (smallest) C on the plateau
    assert svm._select_c(plateau, grid, "best") == 1.0
    # within the tie tolerance still counts as tied
    near = np.array([0.95, 0.95 + 1e-9, 0.94, 0.94])
    assert svm._select_c(near, grid, "best") == 0.1
    # "smoothed": 3-point neighbour mean -> [0.925, 0.933.., 0.95, 0.95] -> 10.0
    assert svm._select_c(plateau, grid, "smoothed") == 10.0
    with pytest.raises(ValueError, match="non-finite"):
        svm._select_c(np.array([np.nan, np.nan]), (0.1, 1.0), "best")


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
    table = svm._coefficient_table(final, resample, names, top_k=2).set_index("feature")
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
        svm.classify_svm(_planted(n=40, p=30, seed=0), "grp", top_k=0, **_FAST)
    with pytest.raises(ValueError, match="exceeds"):
        svm.classify_svm(_planted(n=40, p=30, seed=0), "grp", top_k=31, **_FAST)


# --------------------------------------------------------------------------- #
# Outcome / binarize API
# --------------------------------------------------------------------------- #
def test_positive_class_override_flips_sign() -> None:
    ds = _planted(seed=0)
    a = svm.classify_svm(ds, "grp", random_state=0, **_FAST)
    b = svm.classify_svm(ds, "grp", positive_class="A", random_state=0, **_FAST)
    assert a.positive_label == "B"
    assert b.positive_label == "A"
    # the SVM is symmetric under a label swap: every weight is negated (up to the
    # solver tolerance — the two problems are solved independently)
    fa = a.coefficients.set_index("feature")["coef"]
    fb = b.coefficients.set_index("feature")["coef"].reindex(fa.index)
    assert bool((np.sign(fa.to_numpy()) == -np.sign(fb.to_numpy())).all())
    np.testing.assert_allclose(fa.to_numpy(), -fb.to_numpy(), rtol=0.05, atol=1e-3)


def test_threshold_binarize_continuous() -> None:
    ds = _planted(seed=0)
    res = svm.classify_svm(
        ds, "dose", binarize=svm.Threshold(cut=25.0), random_state=0, **_FAST
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
    res = svm.classify_svm(
        ds,
        "dose",
        binarize=svm.Threshold(cut=100.0, drop_below=1.0, drop_at_or_above=100.0),
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
    res = svm.classify_svm(
        ds,
        "genotype",
        binarize=svm.LevelMap(positive=("Hom",), negative=("WT",)),
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
        svm.classify_svm(ds, "grp", **_FAST)


def test_numeric_outcome_without_binarize_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match="numeric"):
        svm.classify_svm(ds, "dose", **_FAST)


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
        svm.classify_svm(ds, "g", **_FAST)


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
        svm.classify_svm(ds, "g", **_FAST)


def test_class_smaller_than_n_splits_raises() -> None:
    ds = _planted(n=24, p=30, n_signal=3, seed=0)
    ds.metadata["grp"] = ["A"] * 4 + ["B"] * 20
    with pytest.raises(ValueError, match="n_splits"):
        svm.classify_svm(ds, "grp", n_splits=5, **_FAST)


def test_bad_c_grid_raises() -> None:
    ds = _planted(n=40, p=30, seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        svm.classify_svm(ds, "grp", c_grid=[])
    with pytest.raises(ValueError, match="> 0"):
        svm.classify_svm(ds, "grp", c_grid=[0.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        svm.classify_svm(ds, "grp", c_grid=[1.0, 0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        svm.classify_svm(ds, "grp", c_grid=[1.0, 1.0])


def test_c_grid_edge_warns() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    # separable planted data: every C above the knee is the same hard-margin solution,
    # so a grid that starts on the plateau selects its smallest C -> lower-edge warning
    with pytest.warns(svm.CGridEdgeWarning, match="SMALLEST"):
        res = svm.classify_svm(
            ds, "grp", c_grid=[1.0, 10.0], n_repeats=1, stability_repeats=1, n_jobs=1
        )
    assert res.best_c == 1.0
    # a single-element grid is a deliberate pin: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=svm.CGridEdgeWarning)
        svm.classify_svm(
            ds, "grp", c_grid=[1.0], n_repeats=1, stability_repeats=1, n_jobs=1
        )


def test_constant_features_dropped() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.abundances[:, 10] = 7.0  # a constant feature
    ds.abundances[:, 11] = 0.0  # an all-zero feature
    res = svm.classify_svm(ds, "grp", random_state=0, **_FAST)
    assert res.n_dropped_constant == 2
    assert res.n_features == 28
    assert len(res.coefficients) == 28
    assert "F10" not in set(res.coefficients["feature"])


def test_non_log_scale_warns() -> None:
    ds = _planted(seed=0, scale="linear")
    with pytest.warns(svm.ClassificationScaleWarning):
        svm.classify_svm(ds, "grp", random_state=0, **_FAST)


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
    res = svm.classify_svm(
        ds,
        "grp",
        groups="subject",
        generalization_target="individuals",
        n_splits=3,
        c_grid=[1.0],
        n_repeats=1,
        stability_repeats=2,
        n_jobs=1,
        random_state=0,
        run_null=True,
        n_permutations=5,
        null_repeats=1,
    )
    assert res.grouped is True
    assert res.groups_column == "subject"
    assert res.generalization_target == "individuals"
    assert res.null_p is not None  # the grouped null path (explicit margins) ran
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
        svm.classify_svm(
            ds,
            "grp",
            groups="subject",
            n_splits=2,
            c_grid=[1.0],
            n_repeats=1,
            stability_repeats=1,
            n_jobs=1,
        )


def test_grouped_null_rejects_mixed_label_unit() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    # _planted alternates labels within each consecutive pair -> every unit is mixed
    ds.metadata["subject"] = [f"s{i // 2}" for i in range(60)]
    with pytest.raises(ValueError, match=r"both classes.*within_units"):
        svm.classify_svm(
            ds,
            "grp",
            groups="subject",
            run_null=True,
            n_permutations=2,
            n_splits=3,
            c_grid=[1.0],
            n_repeats=1,
            stability_repeats=1,
            n_jobs=1,
        )
    # the permutation helper itself refuses too
    y = np.array([0, 1, 1, 1])
    g = np.array(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="both classes"):
        svm._permute_labels(y, g, np.random.default_rng(0))
    # and preserves class counts on clean units
    y2 = np.array([0, 0, 1, 1, 1, 1])
    g2 = np.array(["a", "a", "b", "b", "c", "c"])
    perm = svm._permute_labels(y2, g2, np.random.default_rng(0))
    assert perm.sum() == 4


def test_singleton_groups_fall_back_to_rowlevel() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.metadata["subject"] = [f"s{i}" for i in range(40)]  # all unique
    with pytest.warns(svm.SingletonGroupsWarning):
        res = svm.classify_svm(ds, "grp", groups="subject", random_state=0, **_FAST)
    assert res.grouped is False
    assert res.groups_column is None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_all_figures_render(
    planted_result: svm.SVMClassificationResult, tmp_path: Path
) -> None:
    for fig in (
        svmfig.plot_roc(planted_result),
        svmfig.plot_null(planted_result),
        svmfig.plot_coefficients(planted_result),
        svmfig.plot_hyperparameter_curve(planted_result),
    ):
        assert isinstance(fig, Figure)
        plt.close(fig)
    art = svmfig.save_roc(planted_result, tmp_path, "roc")
    assert art.svg.exists() and art.png.exists()
    assert art.legend_svg is None  # colorbars/on-axes legends, no separate legend
    art2 = svmfig.save_hyperparameter_curve(planted_result, tmp_path, "curve")
    assert art2.png.exists()


def test_null_figure_requires_null() -> None:
    ds = _planted(seed=0)
    res = svm.classify_svm(ds, "grp", run_null=False, random_state=0, **_FAST)
    with pytest.raises(ValueError, match="null was not run"):
        svmfig.plot_null(res)
    fig = svmfig.plot_roc(res)  # annotates "null not run"
    plt.close(fig)


def test_coefficient_figure_top_n_defaults_to_top_k(
    planted_result: svm.SVMClassificationResult,
) -> None:
    fig = svmfig.plot_coefficients(planted_result)
    assert len(fig.axes[0].get_yticks()) == planted_result.top_k
    plt.close(fig)
    fig = svmfig.plot_coefficients(planted_result, top_n=3)
    assert len(fig.axes[0].get_yticks()) == 3
    plt.close(fig)
    with pytest.raises(ValueError, match="top_n"):
        svmfig.plot_coefficients(planted_result, top_n=0)


def test_curve_figure_rejects_mismatched_grid(
    planted_result: svm.SVMClassificationResult,
) -> None:
    from dataclasses import replace

    bad = replace(planted_result, grid_scores=np.array([0.5]))
    with pytest.raises(ValueError, match="do not match"):
        svmfig.plot_hyperparameter_curve(bad)


def _mean_roc_label(fig: Figure) -> str:
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [
        t.get_text() for t in legend.get_texts() if t.get_text().startswith("mean ROC")
    ]
    assert len(labels) == 1
    return labels[0]


def _caption(fig: Figure) -> str:
    captions = [t for t in fig.texts if "balanced accuracy" in t.get_text()]
    assert len(captions) == 1  # a figure-level caption, not an axes text
    return captions[0].get_text()


def _line_xy(line: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(line.get_xdata(), dtype=float),
        np.asarray(line.get_ydata(), dtype=float),
    )


def _is_vline(line: Any) -> bool:
    xs = _line_xy(line)[0]
    return xs.size == 2 and bool(xs[0] == xs[1])


def test_roc_legend_auc_equals_cv_auc(
    planted_result: svm.SVMClassificationResult,
) -> None:
    # the number on the figure is the number in the finding — not a re-derived one
    fig = svmfig.plot_roc(planted_result)
    label = _mean_roc_label(fig)
    assert (
        f"AUC = {planted_result.cv_auc:.3f} ± {planted_result.cv_auc_sd:.3f}" in label
    )
    plt.close(fig)


def test_roc_mean_curve_has_no_origin_notch(
    planted_result: svm.SVMClassificationResult,
) -> None:
    # a near-perfect classifier rises vertically at FPR = 0; the drawn mean curve keeps
    # that rise instead of being forced back through the origin
    assert planted_result.cv_auc > 0.9
    fig = svmfig.plot_roc(planted_result)
    lines = [
        ln for ln in fig.axes[0].lines if str(ln.get_label()).startswith("mean ROC")
    ]
    assert len(lines) == 1
    xs, ys = _line_xy(lines[0])
    assert float(xs[0]) == 0.0
    assert float(ys[0]) > 0.0
    assert float(ys[-1]) == 1.0
    plt.close(fig)


def test_roc_annotation_sits_below_axes(
    planted_result: svm.SVMClassificationResult,
) -> None:
    fig = svmfig.plot_roc(planted_result)
    ax = fig.axes[0]
    captions = [t for t in fig.texts if "balanced accuracy" in t.get_text()]
    assert len(captions) == 1
    assert captions[0].get_position()[1] < ax.get_position().y0
    assert not ax.texts  # nothing opaque sits over the plot region
    plt.close(fig)


def test_roc_annotation_shows_pooled_repeat_aucs(
    planted_result: svm.SVMClassificationResult,
) -> None:
    res = planted_result
    fig = svmfig.plot_roc(res)
    text = _caption(fig)
    plt.close(fig)
    # each repeat's mean-of-fold AUC listed, the pooled-OOF range shown — and no
    # between-repeat SD masquerading as the estimate's uncertainty
    for a in res.repeat_aucs:
        assert f"{a:.3f}" in text
    assert "pooled-OOF AUC by repeat" in text
    assert f"{min(res.repeat_pooled_aucs):.3f}" in text
    assert f"{max(res.repeat_pooled_aucs):.3f}" in text
    assert "per-repeat AUC =" not in text
    assert "scores = SVM margin (uncalibrated)" in text


def test_null_line_at_observed_auc() -> None:
    ds = _planted(n=40, p=20, n_signal=4, seed=0)
    res = svm.classify_svm(ds, "grp", run_null=True, n_permutations=3, **_NULL_FAST)
    assert res.observed_auc is not None
    fig = svmfig.plot_null(res)
    vlines = [ln for ln in fig.axes[0].lines if _is_vline(ln)]
    assert len(vlines) == 1
    assert float(_line_xy(vlines[0])[0][0]) == pytest.approx(res.observed_auc)
    plt.close(fig)


def test_curve_ylabel_names_tuning_metric(
    planted_result: svm.SVMClassificationResult,
) -> None:
    from dataclasses import replace

    fig = svmfig.plot_hyperparameter_curve(planted_result)
    assert fig.axes[0].get_ylabel() == "mean inner-CV AUC"
    plt.close(fig)
    fig = svmfig.plot_hyperparameter_curve(
        replace(planted_result, tuning_metric="balanced_accuracy")
    )
    assert fig.axes[0].get_ylabel() == "mean inner-CV balanced accuracy"
    legend = fig.axes[0].get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert "mean inner-CV balanced accuracy" in labels
    plt.close(fig)


def test_curve_fold_rug_in_own_axes(
    planted_result: svm.SVMClassificationResult,
) -> None:
    res = planted_result
    fig = svmfig.plot_hyperparameter_curve(res)
    assert len(fig.axes) == 2  # the curve + its own short rug axes beneath
    curve, rug = fig.axes
    assert rug.get_position().y1 <= curve.get_position().y0
    counts = [int(t.get_text()) for t in rug.texts]
    assert sum(counts) == len(res.fold_predictions)  # every fold counted once
    assert len(counts) == len({f.best_c for f in res.fold_predictions})
    assert curve.get_xlabel() == "" and rug.get_xlabel().startswith("C ")
    assert rug.get_xscale() == "log"  # shares the curve's log x
    legend = curve.get_legend()
    assert legend is not None
    assert any(t.get_text().startswith("per-fold tuned C") for t in legend.get_texts())
    plt.close(fig)


def test_feature_list_caveat_survives_custom_title(
    planted_result: svm.SVMClassificationResult,
) -> None:
    from dataclasses import replace

    res = replace(planted_result, n_features_requested=150, n_features_matched=142)
    for fig in (
        svmfig.plot_roc(res, title="MY TITLE"),
        svmfig.plot_null(res, title="MY TITLE"),
        svmfig.plot_coefficients(res, title="MY TITLE"),
        svmfig.plot_hyperparameter_curve(res, title="MY TITLE"),
    ):
        sup = fig.get_suptitle()
        assert "MY TITLE" in sup
        assert "prior feature list · 142 of 150 matched" in sup
        plt.close(fig)


def test_figure_error_path_closes(
    planted_result: svm.SVMClassificationResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    plt.close("all")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected")

    monkeypatch.setattr(svmfig, "_apply_title", boom)
    for plot in (
        svmfig.plot_roc,
        svmfig.plot_null,
        svmfig.plot_coefficients,
        svmfig.plot_hyperparameter_curve,
    ):
        with pytest.raises(RuntimeError, match="injected"):
            plot(planted_result)
        assert plt.get_fignums() == []  # no figure leaked on the error path


# --------------------------------------------------------------------------- #
# Prior feature-list restriction (leakage-safe; matched/unmatched counts recorded)
# --------------------------------------------------------------------------- #
def test_feature_list_restricts_and_records() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    res = svm.classify_svm(
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
        svm.classify_svm(ds, "grp", feature_list=["ZZZ", "QQQ"], **_FAST)


def test_feature_list_poor_match_warns() -> None:
    ds = _planted(n=40, p=20, n_signal=3, seed=0)
    # 2 real + 8 absent -> 2/10 matched, below the 0.5 warn fraction
    flist = [f"F{i}" for i in range(2)] + [f"ABSENT{i}" for i in range(8)]
    with pytest.warns(svm.FeatureListWarning):
        res = svm.classify_svm(
            ds, "grp", feature_list=flist, top_k=1, random_state=0, **_FAST
        )
    assert res.n_features_requested == 10
    assert res.n_features_matched == 2


def test_feature_list_noted_in_figure_title(
    planted_result: svm.SVMClassificationResult,
) -> None:
    from dataclasses import replace

    # whole proteome -> no note
    fig = svmfig.plot_roc(planted_result)
    assert "prior feature list" not in fig.get_suptitle()
    plt.close(fig)
    # full match -> "N features"
    fig = svmfig.plot_roc(
        replace(planted_result, n_features_requested=150, n_features_matched=150)
    )
    assert "prior feature list · 150 features" in fig.get_suptitle()
    plt.close(fig)
    # partial match -> "N of M matched"
    fig = svmfig.plot_hyperparameter_curve(
        replace(planted_result, n_features_requested=150, n_features_matched=142)
    )
    assert "prior feature list · 142 of 150 matched" in fig.get_suptitle()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The null's permutation scheme: unit-level (default) vs within-unit (batches)
# --------------------------------------------------------------------------- #
# Cheapest valid nested run + a one-repeat null (the null is the expensive part).
_NULL_FAST: dict[str, Any] = {
    "c_grid": [1.0],
    "top_k": 5,
    "n_splits": 3,
    "n_repeats": 1,
    "stability_repeats": 1,
    "null_repeats": 1,
    "n_jobs": 1,
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
    res = svm.classify_svm(
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
    noise = svm.classify_svm(
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
        perm = svm._permute_labels(y, g, rng, "within_units")
        for u in ("a", "b", "c"):
            assert perm[g == u].sum() == y[g == u].sum()
        np.testing.assert_array_equal(perm[g == "c"], y[g == "c"])  # untouched
        changed |= not np.array_equal(perm, y)
    assert changed
    # the default (unit-level) scheme refuses the mixed units and names the remedy
    with pytest.raises(ValueError, match="within_units"):
        svm._permute_labels(y, g, rng, "units")
    # ungrouped: a plain row shuffle whatever the scheme
    plain = svm._permute_labels(y, None, np.random.default_rng(0), "within_units")
    assert plain.sum() == y.sum() and plain.shape == y.shape


def test_within_unit_arrangements_hand_computed() -> None:
    # two 3/3 batches -> C(6,3)^2 = 400
    y = np.array([0, 0, 0, 1, 1, 1] * 2)
    g = np.array(["a"] * 6 + ["b"] * 6)
    assert svm._within_unit_arrangements(y, g) == 400
    # every unit pure -> exactly one arrangement (the identity)
    assert (
        svm._within_unit_arrangements(np.array([0, 0, 1, 1]), np.array(list("aabb")))
        == 1
    )
    # six batches of 20 with 10 positives each: 184756**6 ~ 4e31 > 2**63 — the count is
    # an exact Python int; an int64 product would overflow silently
    y6 = np.tile(np.array([0] * 10 + [1] * 10), 6)
    g6 = np.repeat([f"b{i}" for i in range(6)], 20)
    assert svm._within_unit_arrangements(y6, g6) == 184756**6


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
        svm.classify_svm(ds, "grp", **kw)
    # singleton groups fall back to row-level (warns) -> still refused, naming the col
    single = _planted(n=40, p=20, n_signal=4, seed=0)
    single.metadata["unit"] = [f"u{i}" for i in range(40)]
    with (
        pytest.warns(svm.SingletonGroupsWarning),
        pytest.raises(ValueError, match="'unit'"),
    ):
        svm.classify_svm(single, "grp", groups="unit", **kw)
    # every unit single-class: the within-unit shuffle is the identity -> refused
    with pytest.raises(ValueError, match="no freedom"):
        svm.classify_svm(
            _pure_units(_planted(n=40, p=20, n_signal=4, seed=0)),
            "grp",
            groups="animal",
            **kw,
        )
    # a bad scheme name
    with pytest.raises(ValueError, match="null_permutation must be"):
        svm.classify_svm(ds, "grp", null_permutation="rows", **_NULL_FAST)  # type: ignore[arg-type]
    # too few distinct arrangements for n_permutations: warns, still runs. Four batches
    # of 1/1 -> C(2,1)^4 = 16 arrangements < 17 requested. (Four units, not two: the
    # inner tuning CV is grouped too, so every outer training fold needs >= n_splits
    # units to partition.)
    small = _planted(n=8, p=10, n_signal=3, seed=0)
    small.metadata["batch"] = ["a", "a", "b", "b", "c", "c", "d", "d"]
    with pytest.warns(svm.NullPermutationWarning, match="16 distinct"):
        res = svm.classify_svm(
            small,
            "grp",
            groups="batch",
            run_null=True,
            null_permutation="within_units",
            n_permutations=17,
            **{**_NULL_FAST, "n_splits": 2},
        )
    assert res.null_permutation == "within_units"
    assert res.null_p is not None


def test_null_permutation_recorded_and_titled() -> None:
    ds = _planted(n=40, p=20, n_signal=4, seed=0)
    assert (
        svm.classify_svm(ds, "grp", **_NULL_FAST).null_permutation is None
    )  # null not run
    rows = svm.classify_svm(ds, "grp", run_null=True, n_permutations=3, **_NULL_FAST)
    assert rows.null_permutation == "samples"  # row-level CV: a plain shuffle
    units = svm.classify_svm(
        _pure_units(_planted(n=40, p=20, n_signal=4, seed=0)),
        "grp",
        groups="animal",
        run_null=True,
        n_permutations=3,
        **_NULL_FAST,
    )
    assert units.null_permutation == "units"
    within = svm.classify_svm(
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
        (rows, "permutations, fixed"),
        (units, "unit-level"),
        (within, "within-unit"),
    ):
        fig = svmfig.plot_null(res)
        assert needle in fig.get_suptitle()
        plt.close(fig)
    assert "unit-level" not in svmfig.plot_null(rows).get_suptitle()
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


@pytest.mark.slow
@_skip_no_data
def test_smoke_5xfad_genotype_recovers_ad_signal() -> None:
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
    exp = dataclasses.replace(
        logged, abundances=logged.abundances[mask, :], metadata=meta
    )

    res = svm.classify_svm(
        exp,
        "Disease",
        positive_class="5xFAD",
        c_grid=[0.01, 0.1, 1.0],
        n_repeats=2,
        stability_repeats=3,
        n_jobs=1,
        random_state=0,
    )
    assert res.n_samples == 52
    assert res.positive_label == "5xFAD"
    assert res.cv_auc > 0.8  # strong, separable disease signal
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
    exp = dataclasses.replace(
        logged, abundances=logged.abundances[mask, :], metadata=meta
    )
    common: dict[str, Any] = {
        "positive_class": "5xFAD",
        "groups": "Cohort",
        "generalization_target": "batches",
        "c_grid": [1e-3],
        "n_splits": 2,
        "n_repeats": 1,
        "stability_repeats": 1,
        "null_repeats": 3,
        "n_permutations": 100,
        "n_jobs": -1,
        "random_state": 0,
        "run_null": True,
    }
    with pytest.raises(ValueError, match=r"both classes.*within_units"):
        svm.classify_svm(exp, "Disease", **common)  # the default scheme refuses
    res = svm.classify_svm(exp, "Disease", null_permutation="within_units", **common)
    assert res.grouped is True and res.groups_column == "Cohort"
    assert res.null_permutation == "within_units"
    assert res.observed_auc is not None and res.null_aucs is not None
    assert res.observed_auc == pytest.approx(0.84, abs=0.05)
    assert abs(float(np.mean(res.null_aucs)) - 0.5) < 0.05
    assert res.null_p is not None and res.null_p < 0.05


def test_smoothed_edge_warning_keys_on_raw_curve(monkeypatch: Any) -> None:
    """Under select="smoothed" the edge warning follows the unsmoothed plateau start.

    Raw curve [0.80, 0.90, 0.95, 0.94] on (0.01, 0.1, 1, 10): the raw pick is the
    interior C = 1 (no edge), while 3-point smoothing scores the top edge higher
    (mean of two) — the old code warned "extend c_grid upward" on a curve that had
    already plateaued.
    """
    ds = _planted(n=24, p=10, n_signal=3, seed=0)
    x = ds.abundances
    y = (ds.metadata["grp"] == "B").to_numpy().astype(int)
    defaults = {
        k: v.default for k, v in inspect.signature(svm.classify_svm).parameters.items()
    }
    defaults.update(c_grid=(0.01, 0.1, 1.0, 10.0), select="smoothed", n_jobs=1)
    cfg = svm._Config(
        **{f.name: defaults[f.name] for f in dataclasses.fields(svm._Config)}
    )
    raw = np.array([0.80, 0.90, 0.95, 0.94])
    assert svm._select_c(raw, cfg.c_grid, "smoothed") == 10.0  # the smoothed pick
    monkeypatch.setattr(
        svm, "_tune_c", lambda x, y, groups, cfg: (10.0, raw, np.zeros(4))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=svm.CGridEdgeWarning)
        fit = svm._all_data_fit(x, y, None, cfg)
    assert (fit.best_c, fit.plateau_start_c) == (10.0, 1.0)


def test_smoothed_needs_three_grid_points() -> None:
    ds = _planted(n=24, p=10, n_signal=3, seed=0)
    with pytest.raises(ValueError, match="at least 3 values"):
        svm.classify_svm(ds, "grp", c_grid=[0.1, 1.0], select="smoothed", n_jobs=1)
