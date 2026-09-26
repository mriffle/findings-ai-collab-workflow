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

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from analysis import differential_abundance as da
from common import data_loading as dl
from scipy import stats


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
# Rank deficiency — the explicit check (np.linalg.inv does not reliably raise)
# --------------------------------------------------------------------------- #
def test_numerically_singular_design_raises() -> None:
    """A rank-12-of-13 design that np.linalg.inv silently 'inverts' now raises."""
    blocks = np.r_[np.repeat(np.arange(8), 2), [8, 9, 10]]
    trt = np.r_[np.tile([0.0, 1.0], 8), [0.0, 1.0, 0.0]]
    sex = (blocks % 2).astype(float)
    design = np.column_stack(
        [np.ones(blocks.size), trt, sex]
        + [(blocks == b).astype(float) for b in range(1, 11)]
    )
    abund = np.random.default_rng(0).normal(size=(blocks.size, 3))
    with pytest.raises(ValueError, match="singular"):
        da._fit_ols(design, abund)


# --------------------------------------------------------------------------- #
# Unit of analysis — decision table + unit fixed effects (planted truth)
# --------------------------------------------------------------------------- #
def _paired(
    n_pairs: int = 8, n_features: int = 60, *, singletons: int = 0, seed: int = 1
) -> dl.Dataset:
    """``n_pairs`` units each measured under ctrl + trt, plus ctrl-only singletons."""
    rng = np.random.default_rng(seed)
    unit_effect = rng.normal(0, 2.0, size=(n_pairs + singletons, n_features))
    shift = rng.normal(0, 1.0, size=n_features)
    rows, units, cond = [], [], []
    for u in range(n_pairs):
        for c in ("ctrl", "trt"):
            noise = rng.normal(0, 0.5, size=n_features)
            rows.append(unit_effect[u] + (shift if c == "trt" else 0.0) + noise)
            units.append(f"m{u}")
            cond.append(c)
    for s in range(singletons):
        rows.append(unit_effect[n_pairs + s] + rng.normal(0, 0.5, size=n_features))
        units.append(f"solo{s}")
        cond.append("ctrl")
    return _dataset(np.vstack(rows), {"mouse": units, "cond": cond})


def test_ols_with_unit_is_the_paired_t_test() -> None:
    ds = _paired(n_pairs=8, n_features=5)
    res = da.differential_abundance(ds, "cond", method="ols", unit="mouse")
    tab = res.contrast_table.set_index("feature").loc[ds.feature_names]
    trt = ds.abundances[1::2]
    ctrl = ds.abundances[0::2]
    ref = stats.ttest_rel(trt, ctrl, axis=0)
    np.testing.assert_allclose(tab["statistic"], ref.statistic, rtol=1e-10)
    np.testing.assert_allclose(tab["p"], ref.pvalue, rtol=1e-10)
    d = trt - ctrl
    half = stats.t.ppf(0.975, df=7) * d.std(axis=0, ddof=1) / np.sqrt(8)
    np.testing.assert_allclose(tab["ci_low"], d.mean(axis=0) - half, rtol=1e-10)
    assert res.df_residual == 7.0
    assert res.unit_fixed_effects is True
    assert res.n_units == 8
    assert res.n_informative_units == 8
    # Unit dummies are estimated, never tested: only the contrast term is reported.
    assert set(res.table["term"]) == {"cond[trt vs ctrl]"}


def test_singleton_units_leave_the_paired_result_unchanged() -> None:
    base = _paired(n_pairs=8, n_features=5)
    extra = np.random.default_rng(99).normal(size=(3, 5))
    meta = pd.concat(
        [
            base.metadata,
            pd.DataFrame(
                {"mouse": ["s0", "s1", "s2"], "cond": ["ctrl", "trt", "ctrl"]},
                index=["x0", "x1", "x2"],
            ),
        ]
    )
    padded_ds = dataclasses.replace(
        base, abundances=np.vstack([base.abundances, extra]), metadata=meta
    )
    paired = da.differential_abundance(base, "cond", method="ols", unit="mouse")
    padded = da.differential_abundance(padded_ds, "cond", method="ols", unit="mouse")
    a = paired.contrast_table.set_index("feature").sort_index()
    b = padded.contrast_table.set_index("feature").sort_index()
    for col in ("effect", "ci_low", "ci_high", "statistic", "p"):
        np.testing.assert_allclose(a[col], b[col], rtol=1e-9)
    assert padded.df_residual == 7.0
    assert padded.n_units == 11
    assert padded.n_informative_units == 8


def test_moderated_with_unit_matches_explicit_paired_design() -> None:
    ds = _paired(n_pairs=10, n_features=80)
    res = da.differential_abundance(ds, "cond", method="moderated", unit="mouse")
    units = np.repeat(np.arange(10), 2)
    design = np.column_stack(
        [np.ones(20), np.tile([0.0, 1.0], 10)]
        + [(units == u).astype(float) for u in range(1, 10)]
    )
    fit = da._fit_ols(design, ds.abundances)
    s0, d0 = da._fit_f_distribution_prior(fit.sigma2, fit.df)
    assert res.prior_variance == pytest.approx(s0)
    assert res.prior_df == pytest.approx(d0)
    sigma2 = (d0 * s0 + fit.df * fit.sigma2) / (d0 + fit.df)
    t = fit.coefficients[1] / np.sqrt(fit.inv_diag[1] * sigma2)
    tab = res.contrast_table.set_index("feature").loc[ds.feature_names]
    np.testing.assert_allclose(tab["statistic"], t, rtol=1e-9)


def test_three_level_within_unit_contrast_with_missing_levels_matches_lstsq() -> None:
    rng = np.random.default_rng(7)
    plan = {  # unit -> levels observed (a 3-level within-unit factor, incomplete)
        "u0": ["a", "b", "c"],
        "u1": ["a", "b", "c"],
        "u2": ["a", "b"],
        "u3": ["a", "c"],
        "u4": ["b", "c"],
        "u5": ["a", "b", "c"],
    }
    units = [u for u, lv in plan.items() for _ in lv]
    levels = [lvl for lv in plan.values() for lvl in lv]
    ab = rng.normal(size=(len(units), 4))
    ds = _dataset(ab, {"u": units, "lvl": levels})
    res = da.differential_abundance(ds, "lvl", method="ols", unit="u")
    uidx = np.array([int(u[1:]) for u in units])
    lv = np.array(levels)
    design = np.column_stack(
        [np.ones(len(units)), lv == "b", lv == "c"] + [uidx == k for k in range(1, 6)]
    ).astype(float)
    coef, *_ = np.linalg.lstsq(design, ab, rcond=None)
    tab = res.table.set_index(["term", "feature"])
    for k, term in ((1, "lvl[b vs a]"), (2, "lvl[c vs a]")):
        got = tab.loc[term].loc[ds.feature_names, "effect"].to_numpy()
        np.testing.assert_allclose(got, coef[k], rtol=1e-9, atol=1e-12)
    assert set(res.table["term"]) == {"lvl[b vs a]", "lvl[c vs a]"}


def test_unit_without_repeats_is_identical_to_no_unit() -> None:
    ab = np.random.default_rng(2).normal(size=(8, 6))
    ds = _dataset(ab, {"grp": ["A"] * 4 + ["B"] * 4, "id": [f"i{k}" for k in range(8)]})
    plain = da.differential_abundance(ds, "grp", method="ols")
    united = da.differential_abundance(ds, "grp", method="ols", unit="id")
    pd.testing.assert_frame_equal(plain.table, united.table)
    assert united.unit == "id"
    assert united.n_units == 8
    assert united.n_informative_units == 0
    assert united.unit_fixed_effects is False


def test_unit_refuses_technical_replicates() -> None:
    ab = np.random.default_rng(3).normal(size=(6, 4))
    meta = {"animal": ["a", "a", "b", "c", "c", "d"], "geno": list("WWWKKK")}
    with pytest.raises(ValueError, match=r"technical replicates.*aggregate_replicates"):
        da.differential_abundance(
            _dataset(ab, meta), "geno", method="ols", unit="animal"
        )


def test_unit_flags_between_unit_contrast_with_repeated_measures() -> None:
    ab = np.random.default_rng(4).normal(size=(8, 4))
    meta = {
        "animal": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "geno": list("WWWWKKKK"),
        "time": ["t0", "t1"] * 4,
    }
    with pytest.raises(ValueError, match="mixed model") as err:
        da.differential_abundance(
            _dataset(ab, meta), "geno", covariates=["time"], method="ols", unit="animal"
        )
    assert "summarize=" in str(err.value)


def test_unit_level_covariate_is_absorbed_and_refused() -> None:
    ds = _paired(n_pairs=6, n_features=4)
    ds.metadata["sex"] = ["M" if int(u[1:]) % 2 else "F" for u in ds.metadata["mouse"]]
    with pytest.raises(ValueError, match="absorbed by the unit fixed effects"):
        da.differential_abundance(
            ds, "cond", covariates=["sex"], method="ols", unit="mouse"
        )


def test_contrast_level_never_within_a_unit_is_refused() -> None:
    ab = np.random.default_rng(5).normal(size=(7, 3))
    meta = {"u": ["a", "a", "b", "b", "c", "c", "z"], "lvl": list("xyxyxyw")}
    # Level "w" appears only in the singleton unit "z": not estimable within units.
    with pytest.raises(ValueError, match="not estimable from within-unit"):
        da.differential_abundance(_dataset(ab, meta), "lvl", method="ols", unit="u")
    # With "x" as reference, the term "w vs x" is constant within every unit: the
    # per-term check names it.
    with pytest.raises(ValueError, match=r"'lvl\[w vs x\]' never varies"):
        da.differential_abundance(
            _dataset(ab, meta), "lvl", reference={"lvl": "x"}, method="ols", unit="u"
        )


def test_unit_needs_two_informative_units() -> None:
    ab = np.random.default_rng(6).normal(size=(4, 3))
    meta = {"u": ["a", "a", "b", "c"], "cond": ["x", "y", "x", "y"]}
    with pytest.raises(ValueError, match="at least 2"):
        da.differential_abundance(_dataset(ab, meta), "cond", method="ols", unit="u")


@pytest.mark.parametrize("method", ["welch", "mannwhitney"])
def test_independent_two_group_tests_refuse_paired_units(method: str) -> None:
    with pytest.raises(ValueError, match="signed_rank"):
        da.differential_abundance(
            _paired(n_pairs=6, n_features=4),
            "cond",
            method=method,  # type: ignore[arg-type]
            unit="mouse",
        )


def test_unit_column_guards() -> None:
    ds = _paired(n_pairs=4, n_features=3)
    with pytest.raises(ValueError, match="not in metadata"):
        da.differential_abundance(ds, "cond", unit="nope")
    with pytest.raises(ValueError, match="cannot also be the contrast"):
        da.differential_abundance(ds, "cond", unit="cond")
    ds.metadata.iloc[0, ds.metadata.columns.get_loc("mouse")] = None
    with pytest.raises(ValueError, match="missing for 1 row"):
        da.differential_abundance(ds, "cond", unit="mouse")


# --------------------------------------------------------------------------- #
# Weights — weighted least squares (statsmodels WLS oracle)
# --------------------------------------------------------------------------- #
def _weighted(n: int = 30, n_features: int = 6) -> dl.Dataset:
    rng = np.random.default_rng(8)
    grp = np.where(np.arange(n) < n // 2, "A", "B")
    sex = np.tile(["F", "M", "M"], n // 3)
    ab = rng.normal(size=(n, n_features)) + (grp == "B")[:, None] * 0.8
    w = rng.uniform(0.5, 2.0, size=n)
    return _dataset(ab, {"grp": grp, "sex": sex, "w": w})


def test_wls_matches_statsmodels() -> None:
    import statsmodels.api as sm

    ds = _weighted()
    res = da.differential_abundance(
        ds, "grp", covariates=["sex"], method="ols", weights="w"
    )
    md = ds.metadata
    x = np.column_stack([np.ones(len(md)), md["grp"] == "B", md["sex"] == "M"]).astype(
        float
    )
    tab = res.table.set_index(["term", "feature"])
    for j, feat in enumerate(ds.feature_names):
        ref = sm.WLS(ds.abundances[:, j], x, weights=md["w"].to_numpy()).fit()
        row = tab.loc[("grp[B vs A]", feat)]
        assert row["effect"] == pytest.approx(ref.params[1], rel=1e-9)
        assert row["p"] == pytest.approx(ref.pvalues[1], rel=1e-8)
        lo, hi = ref.conf_int()[1]
        assert row["ci_low"] == pytest.approx(lo, rel=1e-9)
        assert row["ci_high"] == pytest.approx(hi, rel=1e-9)
    assert res.weights == "w"


def test_unit_weights_equal_ols_and_scale_is_irrelevant() -> None:
    ds = _weighted()
    ds.metadata["one"] = 1.0
    ds.metadata["w3"] = ds.metadata["w"] * 3.0
    plain = da.differential_abundance(ds, "grp", method="ols")
    ones = da.differential_abundance(ds, "grp", method="ols", weights="one")
    pd.testing.assert_frame_equal(plain.table, ones.table)
    big = _weighted(n_features=60)
    big.metadata["w3"] = big.metadata["w"] * 3.0
    a = da.differential_abundance(big, "grp", method="moderated", weights="w")
    b = da.differential_abundance(big, "grp", method="moderated", weights="w3")
    np.testing.assert_allclose(a.table["p"], b.table["p"], rtol=1e-9)


def test_weights_guards() -> None:
    ds = _weighted()
    ds.metadata["bad"] = ds.metadata["w"].to_numpy() * np.r_[-1.0, np.ones(29)]
    with pytest.raises(ValueError, match="finite and positive"):
        da.differential_abundance(ds, "grp", method="ols", weights="bad")
    with pytest.raises(ValueError, match="not in metadata"):
        da.differential_abundance(ds, "grp", method="ols", weights="nope")
    with pytest.raises(ValueError, match="linear model"):
        da.differential_abundance(ds, "grp", method="welch", weights="w")


def test_unweighted_aggregated_dataset_warns() -> None:
    ds = _weighted()
    ds.metadata["n_replicates"] = np.tile([1, 2], 15)
    with pytest.warns(da.UnweightedReplicatesWarning, match="precision_weight"):
        da.differential_abundance(ds, "grp", method="ols")
    ds.metadata["n_replicates"] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", da.UnweightedReplicatesWarning)
        da.differential_abundance(ds, "grp", method="ols")  # equal counts: silent


# --------------------------------------------------------------------------- #
# signed_rank — Wilcoxon on within-unit pairs
# --------------------------------------------------------------------------- #
def test_signed_rank_matches_scipy_on_the_paired_differences() -> None:
    ds = _paired(n_pairs=12, n_features=20, seed=9)
    ds.abundances[1, 3] = ds.abundances[0, 3]  # one zero difference in feature F3
    # Shuffle rows: pairing must follow the unit, not the row order.
    perm = np.random.default_rng(1).permutation(24)
    shuffled = dataclasses.replace(
        ds, abundances=ds.abundances[perm], metadata=ds.metadata.iloc[perm]
    )
    res = da.differential_abundance(
        shuffled, "cond", method="signed_rank", unit="mouse"
    )
    tab = res.contrast_table.set_index("feature").loc[ds.feature_names]
    d = ds.abundances[1::2] - ds.abundances[0::2]
    for j in range(d.shape[1]):
        nz = d[:, j][d[:, j] != 0]
        exact = nz.size <= 50 and np.unique(np.abs(nz)).size == nz.size
        ref = stats.wilcoxon(
            nz,
            zero_method="wilcox",
            correction=False,
            method="exact" if exact else "approx",
        )
        assert tab["p"].iloc[j] == pytest.approx(ref.pvalue, rel=1e-10)
    assert int(tab["n"].iloc[0]) == 24
    assert res.method == "signed_rank"
    assert "pseudo-median of paired differences" in res.effect_label
    assert res.unit == "mouse"


def test_hodges_lehmann_pseudo_median_planted() -> None:
    st = da._signed_rank_stats(np.array([[1.0], [2.0], [4.0]]), 0.05)
    # Walsh averages 1, 1.5, 2, 2.5, 3, 4 -> median 2.25.
    assert st.effect[0] == pytest.approx(2.25)
    assert np.isnan(st.ci_low[0])  # n'=3: no exact 95% interval exists


def test_all_zero_differences_are_untestable() -> None:
    diffs = np.column_stack([np.zeros(8), np.arange(1.0, 9.0)])
    st = da._signed_rank_stats(diffs, 0.05)
    assert np.isnan(st.p[0])
    assert np.isnan(st.q[0])
    assert np.isfinite(st.p[1])
    assert st.q[1] == pytest.approx(st.p[1])  # a BH family of one


def test_exact_null_cdf_matches_brute_force_enumeration() -> None:
    n = 8
    totals = [sum(r + 1 for r in range(n) if mask >> r & 1) for mask in range(2**n)]
    counts = np.bincount(totals, minlength=n * (n + 1) // 2 + 1)
    np.testing.assert_allclose(
        da._signed_rank_null_cdf(n), np.cumsum(counts) / 2.0**n, rtol=1e-12
    )


def test_exact_ci_index_edges() -> None:
    assert da._walsh_ci_index(5, 0.05, exact=True) is None  # P(T+=0)=1/32 > .025
    assert da._walsh_ci_index(6, 0.05, exact=True) == 0  # 1/64 <= .025 < 2/64


def test_signed_rank_ci_coverage() -> None:
    rng = np.random.default_rng(12)
    diffs = rng.normal(0.5, 1.0, size=(10, 800))
    st = da._signed_rank_stats(diffs, 0.05)
    covered = (st.ci_low <= 0.5) & (st.ci_high >= 0.5)
    assert covered.mean() >= 0.93


def test_signed_rank_unpaired_units_warn_and_drop() -> None:
    ds = _paired(n_pairs=7, n_features=4, singletons=2)
    with pytest.warns(da.UnpairedUnitWarning, match="2 unit"):
        res = da.differential_abundance(ds, "cond", method="signed_rank", unit="mouse")
    assert int(res.contrast_table["n"].iloc[0]) == 14


def test_signed_rank_low_power_warns() -> None:
    with pytest.warns(da.LowPowerRankTestWarning, match="5 pairs"):
        da.differential_abundance(
            _paired(n_pairs=5, n_features=3), "cond", method="signed_rank", unit="mouse"
        )


def test_signed_rank_guards() -> None:
    ds = _paired(n_pairs=6, n_features=3)
    with pytest.raises(ValueError, match="paired test: pass unit="):
        da.differential_abundance(ds, "cond", method="signed_rank")
    ds.metadata["batch"] = ["b1", "b2"] * 6
    with pytest.raises(ValueError, match="cannot adjust for covariates"):
        da.differential_abundance(
            ds, "cond", covariates=["batch"], method="signed_rank", unit="mouse"
        )
    ds.metadata["dose"] = [0.0, 1.0] * 6
    with pytest.raises(ValueError, match="needs a categorical contrast"):
        da.differential_abundance(ds, "dose", method="signed_rank", unit="mouse")


# --------------------------------------------------------------------------- #
# Review fixes — unit grouping, messages, warning scope, extra designs
# --------------------------------------------------------------------------- #
def test_unit_ids_differing_only_in_type_are_distinct_units() -> None:
    ab = np.random.default_rng(13).normal(size=(4, 3))
    meta = {"u": [1, "1", 2, "2"], "grp": ["A", "A", "B", "B"]}
    res = da.differential_abundance(_dataset(ab, meta), "grp", method="ols", unit="u")
    assert res.n_units == 4  # not merged into two units (which would raise)


def test_implicit_repeats_message_names_both_readings() -> None:
    ab = np.random.default_rng(14).normal(size=(8, 3))
    meta = {"animal": list("aabbccdd"), "geno": list("WWWWKKKK")}
    with pytest.raises(ValueError, match="technical replicates") as err:
        da.differential_abundance(
            _dataset(ab, meta), "geno", method="ols", unit="animal"
        )
    assert "repeated measures" in str(err.value)
    assert "mixed model" in str(err.value)


def test_unweighted_warning_silent_without_unequal_counts_or_for_rank_tests() -> None:
    ds = _weighted()
    ds.metadata["n_replicates"] = 1
    ds.metadata["precision_weight"] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", da.UnweightedReplicatesWarning)
        da.differential_abundance(ds, "grp", method="ols")  # all one run: silent
    paired = _paired(n_pairs=8, n_features=4)
    paired.metadata["n_replicates"] = np.tile([1, 2], 8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", da.UnweightedReplicatesWarning)
        da.differential_abundance(paired, "cond", method="signed_rank", unit="mouse")


def test_missing_covariate_error_points_to_summarized_fraction_columns() -> None:
    ds = _weighted()
    ds.metadata["batch[frac B2]"] = np.tile([0.0, 0.5, 1.0], 10)
    with pytest.raises(ValueError, match=r"summarized into \['batch\[frac B2\]'\]"):
        da.differential_abundance(ds, "grp", covariates=["batch"], method="ols")


def test_signed_rank_three_level_contrast_pairs_each_level_with_the_reference() -> None:
    rng = np.random.default_rng(15)
    n_units, n_features = 12, 5
    base = rng.normal(0, 2, size=(n_units, n_features))
    shifts = {"a": 0.0, "b": 1.0, "c": -1.0}
    rows, units, lvls = [], [], []
    for u in range(n_units):
        for lvl, sh in shifts.items():
            rows.append(base[u] + sh + rng.normal(0, 0.3, size=n_features))
            units.append(f"u{u}")
            lvls.append(lvl)
    ab = np.vstack(rows)
    ds = _dataset(ab, {"u": units, "lvl": lvls})
    res = da.differential_abundance(ds, "lvl", method="signed_rank", unit="u")
    assert res.contrast_terms == ("lvl[b vs a]", "lvl[c vs a]")
    tab = res.table.set_index(["term", "feature"])
    lv = np.array(lvls)
    for term, level in (("lvl[b vs a]", "b"), ("lvl[c vs a]", "c")):
        d = ab[lv == level] - ab[lv == "a"]
        for j, feat in enumerate(ds.feature_names):
            ref = stats.wilcoxon(
                d[:, j], zero_method="wilcox", correction=False, method="exact"
            )
            assert tab.loc[(term, feat), "p"] == pytest.approx(ref.pvalue, rel=1e-10)
            assert int(tab.loc[(term, feat), "n"]) == 2 * n_units


def test_continuous_within_unit_contrast_is_a_within_unit_slope() -> None:
    rng = np.random.default_rng(16)
    units = np.repeat(np.arange(6), 3)
    dose = np.tile([0.0, 1.5, 4.0], 6)
    ab = rng.normal(size=(18, 4)) + rng.normal(0, 3, size=(6, 4))[units]
    ab = ab + 0.7 * dose[:, None]
    ds = _dataset(ab, {"u": [f"u{k}" for k in units], "dose": dose})
    res = da.differential_abundance(ds, "dose", method="ols", unit="u")
    design = np.column_stack(
        [np.ones(18), dose] + [(units == k).astype(float) for k in range(1, 6)]
    )
    coef, *_ = np.linalg.lstsq(design, ab, rcond=None)
    tab = res.contrast_table.set_index("feature").loc[ds.feature_names]
    np.testing.assert_allclose(tab["effect"], coef[1], rtol=1e-9)
    assert res.unit_fixed_effects is True
    assert res.df_residual == 18 - 7


def test_unit_and_weights_result_round_trips(tmp_path: Path) -> None:
    from analysis import result_io as rio

    ds = _paired(n_pairs=8, n_features=4)
    ds.metadata["w"] = np.linspace(0.5, 1.5, 16)
    res = da.differential_abundance(ds, "cond", method="ols", unit="mouse", weights="w")
    rio.save_result(res, tmp_path / "r")
    back = rio.load_result(tmp_path / "r", da.DifferentialAbundanceResult)
    assert (back.unit, back.n_units, back.weights) == ("mouse", 8, "w")
    assert back.unit_fixed_effects is True
    assert back.df_residual == res.df_residual
    pd.testing.assert_frame_equal(back.table, res.table)


# --------------------------------------------------------------------------- #
# Low-cardinality warning — integer codes only
# --------------------------------------------------------------------------- #
def test_low_cardinality_warning_skips_fraction_columns() -> None:
    ab = np.random.default_rng(10).normal(size=(12, 3))
    meta = {"grp": ["A", "B"] * 6, "batch_frac": [0.0, 0.5, 1.0] * 4}
    with warnings.catch_warnings():
        warnings.simplefilter("error", da.LowCardinalityNumericWarning)
        da.differential_abundance(
            _dataset(ab, meta), "grp", covariates=["batch_frac"], method="ols"
        )


# --------------------------------------------------------------------------- #
# Result cache — a v0.1-shaped result reloads with the new fields defaulted
# --------------------------------------------------------------------------- #
def test_v01_cached_result_reloads_with_defaults(tmp_path: Path) -> None:
    import json

    from analysis import result_io as rio

    ab = np.random.default_rng(11).normal(size=(8, 3))
    res = da.differential_abundance(
        _dataset(ab, {"grp": ["A", "B"] * 4}), "grp", method="ols"
    )
    rio.save_result(res, tmp_path / "old")
    manifest_path = tmp_path / "old" / "_result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_fields = (
        "unit",
        "n_units",
        "n_informative_units",
        "unit_fixed_effects",
        "weights",
        "df_residual",
    )
    for name in new_fields:
        del manifest["fields"][name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.warns(rio.ResultSchemaWarning):
        loaded = rio.load_result(tmp_path / "old", da.DifferentialAbundanceResult)
    assert loaded.unit is None
    assert loaded.unit_fixed_effects is False
    assert loaded.df_residual is None
    pd.testing.assert_frame_equal(loaded.table, res.table)


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


def _real_experimental_runs() -> dl.Dataset:
    """All 74 experimental runs (no keep-one collapse), median + log2, Disease added."""
    from common import normalize as norm

    ds = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    logged = norm.log2_transform(norm.normalize(ds, "median"))
    mask = logged.metadata["Genotype"].to_numpy() != "na"
    meta = logged.metadata.loc[mask].copy()
    meta["Disease"] = np.where(meta["Genotype"].to_numpy() == "5xFAD", "5xFAD", "nonAD")
    return dataclasses.replace(
        logged, abundances=logged.abundances[mask, :], metadata=meta
    )


@_skip_no_data
def test_smoke_unit_refuses_pseudoreplicated_runs() -> None:
    """unit='Sample ID' on the 74 runs (22 animals twice) is refused, not tested."""
    with pytest.raises(ValueError, match="technical replicates"):
        da.differential_abundance(
            _real_experimental_runs(),
            "Disease",
            covariates=["Gender", "Treatment", "Cohort"],
            unit="Sample ID",
        )


@_skip_no_data
def test_smoke_aggregated_weighted_pipeline() -> None:
    """Runs → aggregate (52 animals, precision weights) → weighted moderated DE."""
    from common import replicates as rp

    covs = ["Gender", "Treatment", "Cohort"]
    ref = {"Disease": "nonAD", "Gender": "F", "Treatment": "ISO", "Cohort": "2"}
    agg = rp.aggregate_replicates(
        _real_experimental_runs(),
        "Sample ID",
        run_level=("Replicate", "RunOrder", "Technical Replicate"),
        weight_design=("Disease", *covs),
    )
    res = da.differential_abundance(
        agg.dataset,
        "Disease",
        covariates=covs,
        reference=ref,
        unit="Sample ID",
        weights="precision_weight",
    )
    assert res.n_samples == 52
    assert res.n_units == 52
    assert res.weights == "precision_weight"
    assert res.unit_fixed_effects is False
    tab = res.contrast_table
    assert int((tab["q"] < 0.05).sum()) == 62
    app = tab.loc[tab["feature"].str.contains("5xFADA4_HUMAN"), "effect"]
    assert float(app.iloc[0]) == pytest.approx(3.429, abs=1e-3)
