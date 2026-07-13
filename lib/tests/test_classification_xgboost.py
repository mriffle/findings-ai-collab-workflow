"""Tests for the XGBoost (gradient-boosted-tree) classification template + its figures.

Layers (conventions/coding.md + lib/AUTHORING.md):
  * planted-truth — a separable synthetic signal must be recovered (high AUC, the
    planted features dominating the gain-importance ranking), and a pure-noise dataset
    must not beat the label-shuffle null; results are deterministic at a fixed seed;
  * the outcome/binarize API (already-binary, Threshold, LevelMap, positive_class);
  * fail-loud guards (NaN, numeric-without-binarize, >2 levels, too few per class, bad
    grid) and the constant-feature drop;
  * the deliberate divergence — trees are scale-invariant, so **no** scale warning fires
    on a non-log Dataset (unlike the elastic-net classifier);
  * grouping (repeats -> grouped; singletons -> row-level + warning);
  * result invariants and the four figures (incl. the conditional null figure);
  * real-5xFAD smoke — genotype classification recovers the AD signal (skips if absent).

The wrapper is tested, not XGBoost itself: importances are pinned as invariants
(magnitude ranking, non-negativity, stability in range) rather than exact values.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from analysis import classification_xgboost as xgb
from common import data_loading as dl
from figures import classification_xgboost as xgbfig
from matplotlib.figure import Figure

# A small, fast configuration so the suite stays quick (few trees, short grid, 1 job).
_FAST: dict[str, Any] = {
    "max_depth_grid": [2, 3],
    "learning_rate_grid": [0.1, 0.3],
    "n_estimators": 60,
    "n_repeats": 2,
    "stability_repeats": 3,
    "n_jobs": 1,
}


def _planted(
    n: int = 48,
    p: int = 120,
    n_signal: int = 6,
    effect: float = 2.5,
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
def planted_result() -> xgb.XGBClassificationResult:
    """A recovered planted-signal result *with* a small null — reused across tests."""
    ds = _planted(seed=0)
    return xgb.classify_xgboost(
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
def test_planted_signal_recovered(
    planted_result: xgb.XGBClassificationResult,
) -> None:
    res = planted_result
    assert res.cv_auc > 0.9
    planted = {f"F{i}" for i in range(6)}
    selected = set(res.importances["feature"])
    # every planted feature carries non-zero gain (appears in the selected set)
    assert planted <= selected
    # and they dominate the top of the importance ranking
    top6 = set(res.importances.head(6)["feature"])
    assert len(planted & top6) >= 5
    assert res.positive_label == "B"
    # importances are unsigned (non-negative) and there is no sign column
    assert bool((res.importances["importance"] >= 0).all())
    assert "sign_consistency" not in res.importances.columns


def test_planted_signal_beats_null(
    planted_result: xgb.XGBClassificationResult,
) -> None:
    assert planted_result.null_p is not None
    assert planted_result.null_p < 0.05
    assert planted_result.validated_eligible


def test_pure_noise_does_not_beat_null() -> None:
    ds = _planted(n=60, p=100, n_signal=0, seed=3)
    res = xgb.classify_xgboost(
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
    planted_result: xgb.XGBClassificationResult,
) -> None:
    imp = planted_result.importances
    assert bool(
        ((imp["selection_frequency"] >= 0) & (imp["selection_frequency"] <= 1)).all()
    )
    # every listed feature has non-zero gain in the final model (selected-only)
    assert bool((imp["importance"] > 0).all())
    # where a feature was selected in >=1 resample the IQR bounds are ordered and
    # non-negative (a feature never re-selected has selection_frequency 0 and NaN
    # quantiles — the honest "used on full data but unstable" read)
    scored = imp[imp["importance_q25"].notna() & imp["importance_q75"].notna()]
    assert bool((scored["importance_q75"] >= scored["importance_q25"]).all())
    assert bool((scored["importance_q25"] >= 0).all())
    # planted features are stably selected across resamples — clearly above noise,
    # though not always at 1.0 (redundant signal features substitute for each other,
    # the "low selection frequency != unimportant" caveat the finding must carry)
    planted = imp[imp["feature"].isin({f"F{i}" for i in range(6)})]
    assert bool((planted["selection_frequency"] >= 0.5).all())
    assert float(planted["selection_frequency"].median()) >= 0.8


def test_determinism() -> None:
    """A fixed seed + single job reproduces the importances and performance exactly."""
    ds = _planted(n=40, p=60, n_signal=4, seed=0)
    a = xgb.classify_xgboost(ds, "grp", random_state=0, **_FAST)
    b = xgb.classify_xgboost(ds, "grp", random_state=0, **_FAST)
    assert a.cv_auc == b.cv_auc
    assert a.best_params == b.best_params
    pd.testing.assert_frame_equal(a.importances, b.importances)


# --------------------------------------------------------------------------- #
# Outcome / binarize API
# --------------------------------------------------------------------------- #
def test_positive_class_sets_label() -> None:
    ds = _planted(seed=0)
    a = xgb.classify_xgboost(ds, "grp", random_state=0, **_FAST)
    b = xgb.classify_xgboost(ds, "grp", positive_class="A", random_state=0, **_FAST)
    assert a.positive_label == "B"
    assert b.positive_label == "A"


def test_threshold_binarize_continuous() -> None:
    ds = _planted(seed=0)
    res = xgb.classify_xgboost(
        ds, "dose", binarize=xgb.Threshold(cut=25.0), random_state=0, **_FAST
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
    res = xgb.classify_xgboost(
        ds,
        "dose",
        binarize=xgb.Threshold(cut=100.0, drop_below=1.0, drop_at_or_above=100.0),
        random_state=0,
        **_FAST,
    )
    assert res.n_dropped_unassigned == 20  # the 40.0 middle band is dropped
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
    res = xgb.classify_xgboost(
        ds,
        "genotype",
        binarize=xgb.LevelMap(positive=("Hom",), negative=("WT",)),
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
        xgb.classify_xgboost(ds, "grp", **_FAST)


def test_numeric_outcome_without_binarize_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match="numeric"):
        xgb.classify_xgboost(ds, "dose", **_FAST)


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
        xgb.classify_xgboost(ds, "g", **_FAST)


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
        xgb.classify_xgboost(ds, "g", **_FAST)


def test_bad_max_depth_grid_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match=">= 1"):
        xgb.classify_xgboost(ds, "grp", max_depth_grid=[0], learning_rate_grid=[0.1])


def test_bad_learning_rate_grid_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match="> 0"):
        xgb.classify_xgboost(ds, "grp", max_depth_grid=[2], learning_rate_grid=[0.0])


def test_constant_features_dropped() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.abundances[:, 10] = 7.0  # a constant feature
    ds.abundances[:, 11] = 0.0  # an all-zero feature
    res = xgb.classify_xgboost(ds, "grp", random_state=0, **_FAST)
    assert res.n_dropped_constant == 2
    assert res.n_features == 28
    assert "F10" not in set(res.importances["feature"])


def test_no_scale_warning_on_linear() -> None:
    """The deliberate divergence: trees are scale-invariant, so no scale warning."""
    ds = _planted(n=40, p=30, n_signal=4, seed=0, scale="linear")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = xgb.classify_xgboost(ds, "grp", random_state=0, **_FAST)
    assert not any("scale" in str(w.message).lower() for w in caught)
    assert res.cv_auc > 0.8  # and it still recovers the signal on linear data


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def test_grouping_engaged_with_repeats() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    ds.metadata["subject"] = [f"s{i // 2}" for i in range(60)]  # pairs share a label
    res = xgb.classify_xgboost(
        ds,
        "grp",
        groups="subject",
        generalization_target="individuals",
        max_depth_grid=[2],
        learning_rate_grid=[0.3],
        n_estimators=40,
        n_splits=3,
        n_repeats=1,
        stability_repeats=2,
        n_jobs=1,
        random_state=0,
    )
    assert res.grouped is True
    assert res.groups_column == "subject"
    assert res.generalization_target == "individuals"


def test_singleton_groups_fall_back_to_rowlevel() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.metadata["subject"] = [f"s{i}" for i in range(40)]  # all unique
    with pytest.warns(xgb.SingletonGroupsWarning):
        res = xgb.classify_xgboost(ds, "grp", groups="subject", random_state=0, **_FAST)
    assert res.grouped is False
    assert res.groups_column is None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_all_figures_render(
    planted_result: xgb.XGBClassificationResult, tmp_path: Path
) -> None:
    for fig in (
        xgbfig.plot_roc(planted_result),
        xgbfig.plot_null(planted_result),
        xgbfig.plot_importance(planted_result),
        xgbfig.plot_hyperparameter_heatmap(planted_result),
    ):
        assert isinstance(fig, Figure)
        plt.close(fig)
    art = xgbfig.save_roc(planted_result, tmp_path, "roc")
    assert art.svg.exists() and art.png.exists()
    assert art.legend_svg is None  # colorbars/on-axes legends, no separate legend


def test_null_figure_requires_null() -> None:
    ds = _planted(seed=0)
    res = xgb.classify_xgboost(ds, "grp", run_null=False, random_state=0, **_FAST)
    with pytest.raises(ValueError, match="null was not run"):
        xgbfig.plot_null(res)


def test_importance_figure_top_n(
    planted_result: xgb.XGBClassificationResult,
) -> None:
    fig = xgbfig.plot_importance(planted_result, top_n=3)
    assert isinstance(fig, Figure)
    plt.close(fig)
    with pytest.raises(ValueError, match="top_n"):
        xgbfig.plot_importance(planted_result, top_n=0)


# --------------------------------------------------------------------------- #
# Prior feature-list restriction (leakage-safe; matched/unmatched counts recorded)
# --------------------------------------------------------------------------- #
def test_feature_list_restricts_and_records() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    res = xgb.classify_xgboost(
        ds, "grp", feature_list=[f"F{i}" for i in range(10)], random_state=0, **_FAST
    )
    assert res.n_features_requested == 10
    assert res.n_features_matched == 10
    assert res.n_features <= 10
    assert set(res.importances["feature"]) <= {f"F{i}" for i in range(10)}


def test_feature_list_no_match_raises() -> None:
    ds = _planted(n=40, p=20, seed=0)
    with pytest.raises(ValueError, match="matched none"):
        xgb.classify_xgboost(ds, "grp", feature_list=["ZZZ", "QQQ"], **_FAST)


def test_feature_list_poor_match_warns() -> None:
    ds = _planted(n=40, p=20, n_signal=3, seed=0)
    # 2 real + 8 absent -> 2/10 matched, below the 0.5 warn fraction
    flist = [f"F{i}" for i in range(2)] + [f"ABSENT{i}" for i in range(8)]
    with pytest.warns(xgb.FeatureListWarning):
        res = xgb.classify_xgboost(
            ds, "grp", feature_list=flist, random_state=0, **_FAST
        )
    assert res.n_features_requested == 10
    assert res.n_features_matched == 2


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

    res = xgb.classify_xgboost(
        exp,
        "Disease",
        positive_class="5xFAD",
        max_depth_grid=[2, 3],
        learning_rate_grid=[0.05, 0.1],
        n_estimators=150,
        n_repeats=2,
        stability_repeats=3,
        n_jobs=1,
        random_state=0,
    )
    assert res.n_samples == 52
    assert res.positive_label == "5xFAD"
    assert res.cv_auc > 0.8  # strong, separable disease signal
    # the APP transgene / canonical AD proteins dominate the top of the ranking
    top = " ".join(res.importances.head(15)["feature"].astype(str)).upper()
    assert ("5XFAD" in top) or ("APOE" in top) or ("A4_MOUSE" in top)
