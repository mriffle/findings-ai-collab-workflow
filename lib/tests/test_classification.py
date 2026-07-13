"""Tests for the elastic-net logistic classification template + its figures.

Layers (conventions/coding.md + lib/AUTHORING.md):
  * planted-truth — a separable synthetic signal must be recovered (high AUC, the
    planted features selected with a positive sign), and a pure-noise dataset must not
    beat the label-shuffle null;
  * the outcome/binarize API (already-binary, Threshold, LevelMap, positive_class);
  * fail-loud guards (NaN, numeric-without-binarize, >2 levels, too few per class, bad
    grid) and the constant-feature drop / non-log-scale warning;
  * grouping (repeats -> grouped; singletons -> row-level + warning);
  * result invariants and the four figures (incl. the conditional null figure);
  * real-5xFAD smoke — genotype classification recovers the AD signal (skips if absent).
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
from analysis import classification as cls
from common import data_loading as dl
from figures import classification as clsfig
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

_FAST: dict[str, Any] = {
    "c_grid": [0.1, 1.0],
    "l1_ratios": [0.5, 1.0],
    "n_repeats": 2,
    "stability_repeats": 3,
    "n_jobs": 1,
    "max_iter": 2000,
    "tol": 1e-3,
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
def planted_result() -> cls.ClassificationResult:
    """A recovered planted-signal result *with* a small null — reused across tests."""
    ds = _planted(seed=0)
    return cls.classify(
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
def test_planted_signal_recovered(planted_result: cls.ClassificationResult) -> None:
    res = planted_result
    assert res.cv_auc > 0.9
    planted = {f"F{i}" for i in range(6)}
    selected = set(res.coefficients["feature"])
    assert planted <= selected  # every planted feature is selected
    top6 = set(res.coefficients.head(6)["feature"])
    assert len(planted & top6) >= 5  # they dominate the top of the ranking
    # signal is higher in class B (the positive class) -> positive coefficients
    planted_rows = res.coefficients[res.coefficients["feature"].isin(planted)]
    assert bool((planted_rows["coef"] > 0).all())
    assert res.positive_label == "B"


def test_planted_signal_beats_null(planted_result: cls.ClassificationResult) -> None:
    assert planted_result.null_p is not None
    assert planted_result.null_p < 0.05
    assert planted_result.validated_eligible


def test_pure_noise_does_not_beat_null() -> None:
    ds = _planted(n=60, p=100, n_signal=0, seed=3)
    res = cls.classify(
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


def test_stability_metrics_in_range(planted_result: cls.ClassificationResult) -> None:
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
    planted = coef[coef["feature"].isin({f"F{i}" for i in range(6)})]
    assert bool((planted["selection_frequency"] == 1.0).all())
    assert bool((planted["sign_consistency"] == 1.0).all())


# --------------------------------------------------------------------------- #
# Outcome / binarize API
# --------------------------------------------------------------------------- #
def test_positive_class_override_flips_sign() -> None:
    ds = _planted(seed=0)
    a = cls.classify(ds, "grp", random_state=0, **_FAST)
    b = cls.classify(ds, "grp", positive_class="A", random_state=0, **_FAST)
    assert a.positive_label == "B"
    assert b.positive_label == "A"
    # same features, opposite coefficient signs
    fa = a.coefficients.set_index("feature")["coef"]
    fb = b.coefficients.set_index("feature")["coef"]
    common = fa.index.intersection(fb.index)
    assert len(common) > 0
    assert bool((np.sign(fa[common]) == -np.sign(fb[common])).all())


def test_threshold_binarize_continuous() -> None:
    ds = _planted(seed=0)
    res = cls.classify(
        ds, "dose", binarize=cls.Threshold(cut=25.0), random_state=0, **_FAST
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
    res = cls.classify(
        ds,
        "dose",
        binarize=cls.Threshold(cut=100.0, drop_below=1.0, drop_at_or_above=100.0),
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
    res = cls.classify(
        ds,
        "genotype",
        binarize=cls.LevelMap(positive=("Hom",), negative=("WT",)),
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
        cls.classify(ds, "grp", **_FAST)


def test_numeric_outcome_without_binarize_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match="numeric"):
        cls.classify(ds, "dose", **_FAST)


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
        cls.classify(ds, "g", **_FAST)


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
        cls.classify(ds, "g", **_FAST)


def test_bad_l1_ratio_raises() -> None:
    ds = _planted(seed=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cls.classify(ds, "grp", l1_ratios=[0.5, 1.5], c_grid=[1.0])


def test_constant_features_dropped() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.abundances[:, 10] = 7.0  # a constant feature
    ds.abundances[:, 11] = 0.0  # an all-zero feature
    res = cls.classify(ds, "grp", random_state=0, **_FAST)
    assert res.n_dropped_constant == 2
    assert res.n_features == 28
    assert "F10" not in set(res.coefficients["feature"])


def test_non_log_scale_warns() -> None:
    ds = _planted(seed=0, scale="linear")
    with pytest.warns(cls.ClassificationScaleWarning):
        cls.classify(ds, "grp", random_state=0, **_FAST)


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def test_grouping_engaged_with_repeats() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    # subject pairs consecutive rows; each subject has one label (blocks share y)
    ds.metadata["subject"] = [f"s{i // 2}" for i in range(60)]
    res = cls.classify(
        ds,
        "grp",
        groups="subject",
        generalization_target="individuals",
        n_splits=3,
        c_grid=[1.0],
        l1_ratios=[0.5],
        n_repeats=1,
        stability_repeats=2,
        n_jobs=1,
        max_iter=2000,
        tol=1e-3,
        random_state=0,
    )
    assert res.grouped is True
    assert res.groups_column == "subject"
    assert res.generalization_target == "individuals"


def test_singleton_groups_fall_back_to_rowlevel() -> None:
    ds = _planted(n=40, p=30, n_signal=4, seed=0)
    ds.metadata["subject"] = [f"s{i}" for i in range(40)]  # all unique
    with pytest.warns(cls.SingletonGroupsWarning):
        res = cls.classify(ds, "grp", groups="subject", random_state=0, **_FAST)
    assert res.grouped is False
    assert res.groups_column is None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_all_figures_render(
    planted_result: cls.ClassificationResult, tmp_path: Path
) -> None:
    for fig in (
        clsfig.plot_roc(planted_result),
        clsfig.plot_null(planted_result),
        clsfig.plot_coefficients(planted_result),
        clsfig.plot_hyperparameter_heatmap(planted_result),
    ):
        assert isinstance(fig, Figure)
        plt.close(fig)
    art = clsfig.save_roc(planted_result, tmp_path, "roc")
    assert art.svg.exists() and art.png.exists()
    assert art.legend_svg is None  # colorbars/on-axes legends, no separate legend


def test_null_figure_requires_null() -> None:
    ds = _planted(seed=0)
    res = cls.classify(ds, "grp", run_null=False, random_state=0, **_FAST)
    with pytest.raises(ValueError, match="null was not run"):
        clsfig.plot_null(res)


def test_coefficient_figure_top_n(planted_result: cls.ClassificationResult) -> None:
    fig = clsfig.plot_coefficients(planted_result, top_n=3)
    assert isinstance(fig, Figure)
    plt.close(fig)
    with pytest.raises(ValueError, match="top_n"):
        clsfig.plot_coefficients(planted_result, top_n=0)


# --------------------------------------------------------------------------- #
# Prior feature-list restriction (leakage-safe; matched/unmatched counts recorded)
# --------------------------------------------------------------------------- #
def test_feature_list_restricts_and_records() -> None:
    ds = _planted(n=60, p=40, n_signal=4, seed=0)
    res = cls.classify(
        ds, "grp", feature_list=[f"F{i}" for i in range(10)], random_state=0, **_FAST
    )
    assert res.n_features_requested == 10
    assert res.n_features_matched == 10
    assert res.n_features <= 10
    assert set(res.coefficients["feature"]) <= {f"F{i}" for i in range(10)}


def test_feature_list_no_match_raises() -> None:
    ds = _planted(n=40, p=20, seed=0)
    with pytest.raises(ValueError, match="matched none"):
        cls.classify(ds, "grp", feature_list=["ZZZ", "QQQ"], **_FAST)


def test_feature_list_poor_match_warns() -> None:
    ds = _planted(n=40, p=20, n_signal=3, seed=0)
    # 2 real + 8 absent -> 2/10 matched, below the 0.5 warn fraction
    flist = [f"F{i}" for i in range(2)] + [f"ABSENT{i}" for i in range(8)]
    with pytest.warns(cls.FeatureListWarning):
        res = cls.classify(ds, "grp", feature_list=flist, random_state=0, **_FAST)
    assert res.n_features_requested == 10
    assert res.n_features_matched == 2


def test_feature_list_noted_in_figure_title(
    planted_result: cls.ClassificationResult,
) -> None:
    from dataclasses import replace

    # whole proteome -> no note
    fig = clsfig.plot_roc(planted_result)
    assert "prior feature list" not in fig.get_suptitle()
    plt.close(fig)
    # full match -> "N features"
    fig = clsfig.plot_roc(
        replace(planted_result, n_features_requested=150, n_features_matched=150)
    )
    assert "prior feature list · 150 features" in fig.get_suptitle()
    plt.close(fig)
    # partial match -> "N of M matched"
    fig = clsfig.plot_roc(
        replace(planted_result, n_features_requested=150, n_features_matched=142)
    )
    assert "prior feature list · 142 of 150 matched" in fig.get_suptitle()
    plt.close(fig)


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

    res = cls.classify(
        exp,
        "Disease",
        positive_class="5xFAD",
        c_grid=[0.1, 1.0],
        l1_ratios=[0.5, 0.9],
        n_repeats=2,
        stability_repeats=3,
        n_jobs=1,
        max_iter=1500,
        tol=1e-3,
        random_state=0,
    )
    assert res.n_samples == 52
    assert res.positive_label == "5xFAD"
    assert res.cv_auc > 0.8  # strong, separable disease signal
    # the APP transgene / canonical AD proteins dominate the top of the ranking
    top = " ".join(res.coefficients.head(15)["feature"].astype(str)).upper()
    assert ("5XFAD" in top) or ("APOE" in top) or ("A4_MOUSE" in top)
