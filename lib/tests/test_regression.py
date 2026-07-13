"""Tests for the elastic-net linear regression template + its figures.

Layers (conventions/coding.md + lib/AUTHORING.md):
  * planted-truth — a planted linear signal must be recovered (high R², the planted
    features selected with the right sign), and a pure-noise target must not beat the
    target-shuffle null;
  * the continuous-target API (numeric direct; a categorical outcome raises; a
    non-finite outcome row is dropped and counted);
  * the optional ``feature_list`` restriction (restricts + records counts; no-match
    raises; poor-match warns);
  * fail-loud guards (NaN, bad grids, too few samples) and the constant-feature drop /
    non-log-scale warning;
  * grouping (repeats -> grouped; singletons -> row-level + warning);
  * result invariants and the four figures (incl. the conditional null figure);
  * real-trex smoke — time-since-exposure regression recovers the exposure signal (skips
    if absent).
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
from analysis import regression as rg
from common import data_loading as dl
from figures import regression as rgfig
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_FAST: dict[str, Any] = {
    "alpha_grid": [0.05, 0.1, 0.5],
    "l1_ratios": [0.5, 1.0],
    "n_repeats": 2,
    "stability_repeats": 3,
    "n_jobs": 1,
    "max_iter": 5000,
    "tol": 1e-3,
}


def _planted(
    n: int = 80,
    p: int = 60,
    n_signal: int = 5,
    effect: float = 3.0,
    noise: float = 1.0,
    seed: int = 0,
    scale: dl.Scale = "log2",
) -> dl.Dataset:
    """A Dataset whose numeric target ``y`` is a linear combo of n_signal features."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:n_signal] = effect
    y = x @ beta + rng.normal(scale=noise, size=n)
    names = np.array([f"F{i}" for i in range(p)])
    meta = pd.DataFrame(
        {
            "y": y,
            "grp": np.where(y > float(np.median(y)), "hi", "lo"),
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
def planted_result() -> rg.RegressionResult:
    """A recovered planted-signal result *with* a small null — reused across tests."""
    ds = _planted(seed=0)
    return rg.regress(
        ds,
        "y",
        run_null=True,
        n_permutations=50,
        null_repeats=2,
        random_state=0,
        **_FAST,
    )


# --------------------------------------------------------------------------- #
# Planted truth
# --------------------------------------------------------------------------- #
def test_planted_signal_recovered(planted_result: rg.RegressionResult) -> None:
    res = planted_result
    assert res.cv_r2 > 0.7
    planted = {f"F{i}" for i in range(5)}
    selected = set(res.coefficients["feature"])
    assert planted <= selected  # every planted feature is selected
    top5 = set(res.coefficients.head(5)["feature"])
    assert len(planted & top5) >= 4  # they dominate the top of the ranking
    # positive effect -> positive coefficients
    planted_rows = res.coefficients[res.coefficients["feature"].isin(planted)]
    assert bool((planted_rows["coef"] > 0).all())


def test_planted_signal_beats_null(planted_result: rg.RegressionResult) -> None:
    assert planted_result.null_p is not None
    assert planted_result.null_p < 0.05
    assert planted_result.validated_eligible


def test_pure_noise_does_not_beat_null() -> None:
    ds = _planted(n=60, p=80, n_signal=0, seed=3)
    res = rg.regress(
        ds,
        "y",
        run_null=True,
        n_permutations=100,
        null_repeats=2,
        random_state=1,
        **_FAST,
    )
    assert res.null_p is not None
    assert res.null_p > 0.05  # no real signal -> does not clear the null


def test_stability_metrics_in_range(planted_result: rg.RegressionResult) -> None:
    coef = planted_result.coefficients
    assert bool(
        ((coef["selection_frequency"] >= 0) & (coef["selection_frequency"] <= 1)).all()
    )
    assert bool(
        ((coef["sign_consistency"] >= 0) & (coef["sign_consistency"] <= 1)).all()
    )
    # every listed feature is non-zero in the final model (selected-only)
    assert bool((coef["coef"].abs() > 0).all())
    # planted features are perfectly stable
    planted = coef[coef["feature"].isin({f"F{i}" for i in range(5)})]
    assert bool((planted["selection_frequency"] == 1.0).all())
    assert bool((planted["sign_consistency"] == 1.0).all())


# --------------------------------------------------------------------------- #
# Continuous-target API
# --------------------------------------------------------------------------- #
def test_numeric_outcome_used_directly(planted_result: rg.RegressionResult) -> None:
    assert planted_result.outcome == "y"
    assert planted_result.n_samples == 80
    assert planted_result.n_dropped_missing_target == 0


def test_categorical_outcome_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match="numeric"):
        rg.regress(ds, "grp", **_FAST)


def test_missing_target_rows_dropped() -> None:
    ds = _planted(n=60, p=30, n_signal=4, seed=0)
    ds.metadata.loc[:9, "y"] = np.nan  # 10 rows without an outcome
    res = rg.regress(ds, "y", random_state=0, **_FAST)
    assert res.n_dropped_missing_target == 10
    assert res.n_samples == 50


# --------------------------------------------------------------------------- #
# feature_list restriction
# --------------------------------------------------------------------------- #
def test_feature_list_restricts_and_records() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    res = rg.regress(
        ds,
        "y",
        feature_list=[f"F{i}" for i in range(10)],
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
        rg.regress(ds, "y", feature_list=["ZZZ", "QQQ"], **_FAST)


def test_feature_list_poor_match_warns() -> None:
    ds = _planted(n=40, p=20, n_signal=3, seed=0)
    # 2 real + 8 absent -> 2/10 matched, below the 0.5 warn fraction
    flist = [f"F{i}" for i in range(2)] + [f"ABSENT{i}" for i in range(8)]
    with pytest.warns(rg.FeatureListWarning):
        res = rg.regress(ds, "y", feature_list=flist, random_state=0, **_FAST)
    assert res.n_features_requested == 10
    assert res.n_features_matched == 2


# --------------------------------------------------------------------------- #
# Guards / edges
# --------------------------------------------------------------------------- #
def test_nan_abundance_raises() -> None:
    ds = _planted(seed=0)
    ds.abundances[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        rg.regress(ds, "y", **_FAST)


def test_bad_l1_ratio_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        rg.regress(ds, "y", l1_ratios=[0.5, 1.5], alpha_grid=[0.1])


def test_bad_alpha_grid_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match="positive"):
        rg.regress(ds, "y", alpha_grid=[0.0, 0.1], l1_ratios=[0.5])


def test_too_few_samples_raises() -> None:
    ds = _planted(n=8, p=10, n_signal=2, seed=0)
    with pytest.raises(ValueError, match="finite outcome"):
        rg.regress(ds, "y", n_splits=5, alpha_grid=[0.1], l1_ratios=[0.5], n_jobs=1)


def test_constant_features_dropped() -> None:
    ds = _planted(n=50, p=30, n_signal=4, seed=0)
    ds.abundances[:, 10] = 7.0  # a constant feature
    ds.abundances[:, 11] = 0.0  # an all-zero feature
    res = rg.regress(ds, "y", random_state=0, **_FAST)
    assert res.n_dropped_constant == 2
    assert res.n_features == 28
    assert "F10" not in set(res.coefficients["feature"])


def test_non_log_scale_warns() -> None:
    ds = _planted(seed=0, scale="linear")
    with pytest.warns(rg.RegressionScaleWarning):
        rg.regress(ds, "y", random_state=0, **_FAST)


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def test_grouping_engaged_with_repeats() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    ds.metadata["subject"] = [f"s{i // 2}" for i in range(60)]  # each subject twice
    res = rg.regress(
        ds,
        "y",
        groups="subject",
        generalization_target="individuals",
        n_splits=3,
        alpha_grid=[0.1],
        l1_ratios=[0.5],
        n_repeats=1,
        stability_repeats=2,
        n_jobs=1,
        max_iter=5000,
        tol=1e-3,
        random_state=0,
    )
    assert res.grouped is True
    assert res.groups_column == "subject"
    assert res.generalization_target == "individuals"


def test_singleton_groups_fall_back_to_rowlevel() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.metadata["subject"] = [f"s{i}" for i in range(40)]  # all unique
    with pytest.warns(rg.SingletonGroupsWarning):
        res = rg.regress(ds, "y", groups="subject", random_state=0, **_FAST)
    assert res.grouped is False
    assert res.groups_column is None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_all_figures_render(
    planted_result: rg.RegressionResult, tmp_path: Path
) -> None:
    for fig in (
        rgfig.plot_predicted_vs_observed(planted_result),
        rgfig.plot_null(planted_result),
        rgfig.plot_coefficients(planted_result),
        rgfig.plot_hyperparameter_heatmap(planted_result),
    ):
        assert isinstance(fig, Figure)
        plt.close(fig)
    art = rgfig.save_predicted_vs_observed(planted_result, tmp_path, "scatter")
    assert art.svg.exists() and art.png.exists()
    assert art.legend_svg is None  # colorbars/on-axes legends, no separate legend


def test_null_figure_requires_null() -> None:
    ds = _planted(seed=0)
    res = rg.regress(ds, "y", run_null=False, random_state=0, **_FAST)
    with pytest.raises(ValueError, match="null was not run"):
        rgfig.plot_null(res)


def test_coefficient_figure_top_n(planted_result: rg.RegressionResult) -> None:
    fig = rgfig.plot_coefficients(planted_result, top_n=3)
    assert isinstance(fig, Figure)
    plt.close(fig)
    with pytest.raises(ValueError, match="top_n"):
        rgfig.plot_coefficients(planted_result, top_n=0)


def test_feature_list_noted_in_figure_title(
    planted_result: rg.RegressionResult,
) -> None:
    from dataclasses import replace

    fig = rgfig.plot_predicted_vs_observed(planted_result)  # whole proteome -> no note
    assert "prior feature list" not in fig.get_suptitle()
    plt.close(fig)
    fig = rgfig.plot_predicted_vs_observed(
        replace(planted_result, n_features_requested=150, n_features_matched=150)
    )
    assert "prior feature list · 150 features" in fig.get_suptitle()
    plt.close(fig)
    fig = rgfig.plot_predicted_vs_observed(
        replace(planted_result, n_features_requested=150, n_features_matched=142)
    )
    assert "prior feature list · 142 of 150 matched" in fig.get_suptitle()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Real-trex smoke (git-ignored data; skips cleanly without it)
# --------------------------------------------------------------------------- #
_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "trex"
_PROT = _TESTDATA / "data" / "proteins_unnormalized_wide.tsv"
_META = _TESTDATA / "metadata" / "metadata_trex.tsv"
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/trex not present (git-ignored)",
)


@pytest.mark.slow
@_skip_no_data
def test_smoke_trex_time_recovers_signal() -> None:
    import dataclasses

    from common import batch_correct as bc
    from common import normalize as norm

    ds = dl.load_wide_data(
        _PROT, _META, join_key="replicate_key", scale="log2", metadata_sep="\t"
    )
    meta = ds.metadata
    dose = pd.to_numeric(meta["Dose_cGy"], errors="coerce").to_numpy()
    days = pd.to_numeric(
        meta["daysaftertreatment"].str.replace(" days", "", regex=False),
        errors="coerce",
    ).to_numpy()
    # irradiated T&E Skin Punch samples with a valid time-since-exposure
    mask = (
        (meta["Sample_Type"].to_numpy() == "Skin Punch")
        & meta["Plate"].notna().to_numpy()
        & np.isfinite(days)
        & np.isfinite(dose)
        & (dose > 0)
    )
    sub_meta = meta.loc[mask].reset_index(drop=True).copy()
    sub_meta["days"] = days[mask]
    sub = dataclasses.replace(ds, abundances=ds.abundances[mask, :], metadata=sub_meta)
    sub = norm.median_center(sub)
    sub = bc.combat_correct(sub, batch_column="Plate")

    res = rg.regress(
        sub,
        "days",
        run_null=True,
        n_permutations=50,
        null_repeats=2,
        alpha_grid=[0.1, 0.5, 1.0],
        l1_ratios=[0.5, 1.0],
        n_repeats=2,
        stability_repeats=3,
        n_jobs=1,
        max_iter=5000,
        tol=1e-3,
        random_state=0,
    )
    # a strong, real continuous signal that clears the null (the manuscript reports
    # R2 ~ 0.90 on more samples; honest nested CV on this subset lands lower but strong)
    assert res.n_samples > 100
    assert res.cv_r2 > 0.5
    assert res.null_p is not None
    assert res.null_p < 0.05
    assert len(res.coefficients) > 0
