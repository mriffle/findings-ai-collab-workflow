"""Tests for the differential-abundance template (differential_abundance.py).

Layers:
  * unit/BH — planted-truth on ``bh_adjust`` (hand-computed step-up, NaN preservation);
  * unit/design — categorical/continuous typing, reference levels, multi-level k-1
    terms, the low-cardinality-numeric and non-log-scale warnings, and the fail-loud
    guards (NaN abundances, singular design, unknown method, bad alpha, missing column,
    contrast==covariate, covariates with a two-group test);
  * unit/methods — planted effects + CIs for ols / moderated / welch / mannwhitney on
    tiny hand-computable matrices, the moderated prior (>=50-feature requirement,
    coefficient unchanged), and the constant-feature → NaN-p (excluded from BH) rule;
  * smoke — real 5xFAD proteins (git-ignored) reproducing the captured oracle (the
    disease contrast effect/prior/hit-counts); skips cleanly when the data is absent.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from analysis import differential_abundance as da
from common import data_loading as dl


def _dataset(
    abundances: np.ndarray,
    metadata: dict[str, Any],
    *,
    scale: dl.Scale = "log2",
    feature_names: np.ndarray | None = None,
) -> dl.Dataset:
    """Build a Dataset (default log2 scale) from an abundance matrix + metadata."""
    n_samples, n_features = abundances.shape
    names = (
        feature_names
        if feature_names is not None
        else np.array([f"F{j}" for j in range(n_features)], dtype=str)
    )
    meta = pd.DataFrame(metadata, index=[f"s{i}" for i in range(n_samples)])
    return dl.Dataset(
        abundances=np.asarray(abundances, dtype=float),
        feature_names=names,
        feature_metadata=pd.DataFrame({"feature": names}),
        metadata=meta,
        scale=scale,
    )


# --------------------------------------------------------------------------- #
# bh_adjust — planted truth
# --------------------------------------------------------------------------- #
def test_bh_adjust_planted_values() -> None:
    """Hand-computed BH on [0.01, 0.02, 0.03, 0.04] (m=4)."""
    p = np.array([0.01, 0.02, 0.03, 0.04])
    # ranked adj: 0.04, 0.04, 0.04, 0.04 -> monotone -> all 0.04
    q = da.bh_adjust(p)
    np.testing.assert_allclose(q, [0.04, 0.04, 0.04, 0.04])


def test_bh_adjust_preserves_nan_and_excludes_from_family() -> None:
    p = np.array([0.01, np.nan, 0.5])
    q = da.bh_adjust(p)
    assert np.isnan(q[1])
    # m=2 finite: ranked 0.01*2/1=0.02, 0.5*2/2=0.5
    assert q[0] == pytest.approx(0.02)
    assert q[2] == pytest.approx(0.5)


def test_bh_adjust_all_nan_returns_all_nan() -> None:
    q = da.bh_adjust(np.array([np.nan, np.nan]))
    assert np.all(np.isnan(q))


# --------------------------------------------------------------------------- #
# OLS — planted effect, SE, CI
# --------------------------------------------------------------------------- #
def test_ols_planted_two_group_effect_and_ci() -> None:
    """A balanced 2-group, 1-feature fit with hand-computed coefficient + SE + CI.

    group A=[0,2], B=[3,5]; ref=A -> coef = mean(B)-mean(A) = 3.0. Within-group
    residuals are +-1, RSS=4, df=2, sigma2=2; dummy (X'X)^-1 diag is 1.0 -> SE=sqrt2.
    """
    ab = np.array([[0.0], [2.0], [3.0], [5.0]])
    ds = _dataset(ab, {"grp": ["A", "A", "B", "B"]})
    res = da.differential_abundance(ds, "grp", method="ols")
    row = res.contrast_table.iloc[0]
    assert res.contrast_terms == ("grp[B vs A]",)
    assert row["effect"] == pytest.approx(3.0)
    se = np.sqrt(2.0)
    from scipy import stats

    half = float(stats.t.ppf(0.975, df=2)) * se
    assert row["ci_low"] == pytest.approx(3.0 - half)
    assert row["ci_high"] == pytest.approx(3.0 + half)
    assert row["n"] == 4
    assert row["mean_abundance"] == pytest.approx(np.mean(ab))


def test_ols_reference_override_flips_sign() -> None:
    ab = np.array([[0.0], [2.0], [3.0], [5.0]])
    ds = _dataset(ab, {"grp": ["A", "A", "B", "B"]})
    res = da.differential_abundance(ds, "grp", reference={"grp": "B"}, method="ols")
    assert res.contrast_terms == ("grp[A vs B]",)
    assert res.contrast_table.iloc[0]["effect"] == pytest.approx(-3.0)


def test_ols_covariate_terms_in_full_table_only_contrast_flagged() -> None:
    rng = np.random.default_rng(1)
    ab = rng.standard_normal((8, 3)) + 10.0
    ds = _dataset(
        ab,
        {
            "grp": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        },
    )
    res = da.differential_abundance(ds, "grp", covariates=["sex"], method="ols")
    terms = set(res.table["term"].unique())
    assert terms == {"grp[B vs A]", "sex[M vs F]"}
    assert bool(res.table[res.table["term"] == "grp[B vs A]"]["is_contrast"].all())
    assert not bool(res.table[res.table["term"] == "sex[M vs F]"]["is_contrast"].any())
    # contrast_table is the contrast rows only.
    assert set(res.contrast_table["term"].unique()) == {"grp[B vs A]"}


def test_continuous_contrast_is_a_slope() -> None:
    """A feature equal to 2*x has OLS slope 2 on a numeric contrast."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ab = (2.0 * x).reshape(-1, 1)
    ds = _dataset(ab, {"dose": list(x)})
    res = da.differential_abundance(ds, "dose", method="ols")
    assert res.contrast_terms == ("dose",)
    assert res.contrast_table.iloc[0]["effect"] == pytest.approx(2.0)


def test_multilevel_contrast_emits_k_minus_one_terms() -> None:
    rng = np.random.default_rng(2)
    ab = rng.standard_normal((9, 4)) + 10.0
    ds = _dataset(ab, {"grp": ["A", "A", "A", "B", "B", "B", "C", "C", "C"]})
    res = da.differential_abundance(ds, "grp", method="ols")
    assert res.contrast_terms == ("grp[B vs A]", "grp[C vs A]")


# --------------------------------------------------------------------------- #
# Moderated
# --------------------------------------------------------------------------- #
def _many_feature_dataset(
    n_features: int = 60, seed: int = 3
) -> tuple[dl.Dataset, int]:
    """A 2-group dataset with a few planted up features (the planted index is 0)."""
    rng = np.random.default_rng(seed)
    n = 12
    grp = ["A"] * 6 + ["B"] * 6
    ab = rng.standard_normal((n, n_features)) + 12.0
    ab[6:, 0] += 4.0  # feature 0: +4 in group B
    return _dataset(ab, {"grp": grp}), 0


def test_moderated_populates_prior_and_keeps_coefficient() -> None:
    ds, _planted = _many_feature_dataset()
    ols = da.differential_abundance(ds, "grp", method="ols")
    mod = da.differential_abundance(ds, "grp", method="moderated")
    assert mod.prior_variance is not None and mod.prior_variance > 0
    assert mod.prior_df is not None
    # Coefficients are unchanged by moderation (only the variance/p changes).
    ols_eff = ols.contrast_table.set_index("feature").loc["F0", "effect"]
    mod_eff = mod.contrast_table.set_index("feature").loc["F0", "effect"]
    assert mod_eff == pytest.approx(ols_eff)
    # The planted feature is the strongest hit.
    assert mod.contrast_table.iloc[0]["feature"] == "F0"


def test_moderated_below_min_features_raises() -> None:
    rng = np.random.default_rng(4)
    ab = rng.standard_normal((12, 10)) + 12.0
    ds = _dataset(ab, {"grp": ["A"] * 6 + ["B"] * 6})
    with pytest.raises(ValueError, match="needs 50 features"):
        da.differential_abundance(ds, "grp", method="moderated")


# --------------------------------------------------------------------------- #
# Welch / Mann-Whitney
# --------------------------------------------------------------------------- #
def test_welch_planted_mean_difference() -> None:
    """A=[1,3], B=[4,8]; ref=A -> effect = mean(B)-mean(A) = 4.0."""
    ab = np.array([[1.0], [3.0], [4.0], [8.0]])
    ds = _dataset(ab, {"grp": ["A", "A", "B", "B"]})
    res = da.differential_abundance(ds, "grp", method="welch")
    assert res.contrast_table.iloc[0]["effect"] == pytest.approx(4.0)
    assert res.contrast_terms == ("grp[B vs A]",)


def test_mannwhitney_planted_hodges_lehmann_shift() -> None:
    """A=[1,1], B=[2,4]; pairwise B-A diffs = {1,1,3,3} -> HL median = 2.0."""
    ab = np.array([[1.0], [1.0], [2.0], [4.0]])
    ds = _dataset(ab, {"grp": ["A", "A", "B", "B"]})
    res = da.differential_abundance(ds, "grp", method="mannwhitney")
    assert res.contrast_table.iloc[0]["effect"] == pytest.approx(2.0)
    assert "Hodges-Lehmann" in res.effect_label


def test_two_group_with_covariates_raises() -> None:
    ab = np.array([[1.0], [3.0], [4.0], [8.0]])
    ds = _dataset(ab, {"grp": ["A", "A", "B", "B"], "sex": ["M", "F", "M", "F"]})
    for method in ("welch", "mannwhitney"):
        with pytest.raises(ValueError, match="cannot control for"):
            da.differential_abundance(ds, "grp", covariates=["sex"], method=method)


def test_two_group_continuous_contrast_raises() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    ds = _dataset(x.reshape(-1, 1), {"dose": list(x)})
    with pytest.raises(ValueError, match="categorical contrast"):
        da.differential_abundance(ds, "dose", method="welch")


def test_two_group_multilevel_pairwise_terms() -> None:
    rng = np.random.default_rng(5)
    ab = rng.standard_normal((12, 3)) + 10.0
    ds = _dataset(ab, {"grp": ["A"] * 4 + ["B"] * 4 + ["C"] * 4})
    res = da.differential_abundance(ds, "grp", method="welch")
    assert res.contrast_terms == ("grp[B vs A]", "grp[C vs A]")


# --------------------------------------------------------------------------- #
# Guards & edges
# --------------------------------------------------------------------------- #
def test_nan_abundances_raises() -> None:
    ab = np.array([[1.0], [np.nan], [3.0], [5.0]])
    ds = _dataset(ab, {"grp": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="NaN/inf"):
        da.differential_abundance(ds, "grp", method="ols")


def test_non_log_scale_warns() -> None:
    ab = np.array([[1.0], [2.0], [3.0], [5.0]])
    ds = _dataset(ab, {"grp": ["A", "A", "B", "B"]}, scale="linear")
    with pytest.warns(da.DifferentialAbundanceScaleWarning):
        da.differential_abundance(ds, "grp", method="ols")


def test_low_cardinality_numeric_warns() -> None:
    rng = np.random.default_rng(6)
    ab = rng.standard_normal((8, 3)) + 10.0
    ds = _dataset(
        ab,
        {
            "grp": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "batch": [1, 1, 2, 2, 3, 3, 1, 2],
        },
    )
    with pytest.warns(da.LowCardinalityNumericWarning):
        da.differential_abundance(ds, "grp", covariates=["batch"], method="ols")


def test_categorical_override_treats_numeric_as_factor() -> None:
    rng = np.random.default_rng(7)
    ab = rng.standard_normal((8, 3)) + 10.0
    ds = _dataset(
        ab,
        {
            "grp": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "batch": [1, 1, 2, 2, 1, 1, 2, 2],
        },
    )
    res = da.differential_abundance(
        ds, "grp", covariates=["batch"], categorical=["batch"], method="ols"
    )
    assert "batch[2 vs 1]" in set(res.table["term"].unique())


def test_unknown_method_raises() -> None:
    ds = _dataset(np.array([[1.0], [2.0], [3.0], [5.0]]), {"grp": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="Unknown method"):
        da.differential_abundance(ds, "grp", method="ttest")  # type: ignore[arg-type]


def test_bad_alpha_raises() -> None:
    ds = _dataset(np.array([[1.0], [2.0], [3.0], [5.0]]), {"grp": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="alpha"):
        da.differential_abundance(ds, "grp", method="ols", alpha=1.5)


def test_missing_contrast_column_raises() -> None:
    ds = _dataset(np.array([[1.0], [2.0], [3.0], [5.0]]), {"grp": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="not in metadata"):
        da.differential_abundance(ds, "nope", method="ols")


def test_contrast_equal_covariate_raises() -> None:
    ds = _dataset(np.array([[1.0], [2.0], [3.0], [5.0]]), {"grp": ["A", "A", "B", "B"]})
    with pytest.raises(ValueError, match="also listed as a covariate"):
        da.differential_abundance(ds, "grp", covariates=["grp"], method="ols")


def test_singular_design_raises() -> None:
    """A covariate perfectly aliased with the contrast makes the design singular."""
    rng = np.random.default_rng(8)
    ab = rng.standard_normal((6, 3)) + 10.0
    ds = _dataset(
        ab,
        {"grp": ["A", "A", "A", "B", "B", "B"], "dup": ["x", "x", "x", "y", "y", "y"]},
    )
    with pytest.raises(ValueError, match="singular"):
        da.differential_abundance(ds, "grp", covariates=["dup"], method="ols")


def test_constant_feature_is_untestable_nan_p_not_crash() -> None:
    """An all-equal feature (zero residual variance) yields NaN p, kept out of BH."""
    ab = np.array([[0.0, 7.0], [2.0, 7.0], [3.0, 7.0], [5.0, 7.0]])
    ds = _dataset(ab, {"grp": ["A", "A", "B", "B"]})
    res = da.differential_abundance(ds, "grp", method="ols")
    by_feat = res.contrast_table.set_index("feature")
    assert np.isnan(by_feat.loc["F1", "p"])
    assert np.isnan(by_feat.loc["F1", "q"])
    assert np.isnan(by_feat.loc["F1", "ci_low"])
    assert np.isfinite(by_feat.loc["F0", "p"])


def test_too_few_samples_for_params_raises() -> None:
    ab = np.array([[1.0], [3.0]])
    ds = _dataset(ab, {"grp": ["A", "B"], "sex": ["M", "F"]})
    with pytest.raises(ValueError, match="Not enough samples"):
        da.differential_abundance(ds, "grp", covariates=["sex"], method="ols")


# --------------------------------------------------------------------------- #
# Smoke — real 5xFAD proteins (git-ignored)
# --------------------------------------------------------------------------- #
_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)


def _real_experimental() -> dl.Dataset:
    """Load proteins, median+log2, drop pools, add a binary Disease column."""
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


@_skip_no_data
def test_smoke_disease_contrast_reproduces_oracle() -> None:
    """The disease contrast reproduces the captured OLS/moderated oracle numbers."""
    exp = _real_experimental()
    ref = {"Disease": "nonAD", "Gender": "F", "Treatment": "ISO", "Cohort": "2"}
    covs = ["Gender", "Treatment", "Cohort"]

    mod = da.differential_abundance(
        exp, "Disease", covariates=covs, reference=ref, method="moderated"
    )
    assert mod.n_samples == 52
    assert mod.contrast_terms == ("Disease[5xFAD vs nonAD]",)
    assert mod.prior_variance == pytest.approx(0.36724, abs=1e-3)
    assert mod.prior_df == pytest.approx(1.131, abs=1e-2)

    # APP (amyloid precursor) is sharply up in 5xFAD; pin its captured effect.
    by_feat = mod.contrast_table.set_index("feature")
    app = by_feat.loc["sp|P05067|5xFADA4_HUMAN"]
    assert app["effect"] == pytest.approx(3.3416, abs=1e-2)
    assert app["q"] < 1e-8
    assert app["ci_low"] < app["effect"] < app["ci_high"]

    hits = int(np.sum(mod.contrast_table["q"].to_numpy() <= 0.05))
    assert hits == 61

    ols = da.differential_abundance(
        exp, "Disease", covariates=covs, reference=ref, method="ols"
    )
    assert int(np.sum(ols.contrast_table["q"].to_numpy() <= 0.05)) == 63


@_skip_no_data
def test_smoke_welch_and_mannwhitney_hit_counts() -> None:
    exp = _real_experimental()
    ref = {"Disease": "nonAD"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        welch = da.differential_abundance(exp, "Disease", reference=ref, method="welch")
        mw = da.differential_abundance(
            exp, "Disease", reference=ref, method="mannwhitney"
        )
    assert int(np.sum(welch.contrast_table["q"].to_numpy() <= 0.05)) == 47
    assert int(np.sum(mw.contrast_table["q"].to_numpy() <= 0.05)) == 66
