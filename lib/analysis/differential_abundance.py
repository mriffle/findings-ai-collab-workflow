"""Univariate differential abundance — one feature at a time, then BH across features.

TEMPLATE (lib/) — a *seed* for a project's differential-abundance script, not a finished
analysis. Copy it into the project's ``scripts/`` and adapt the call site per study
(which metadata column is the contrast, which are nuisance covariates). Held to the
correctness charter (conventions/correctness.md) and the statistics convention
(conventions/statistics.md): **assume nothing, verify everything, fail loud; no bare p —
every result is effect + CI + corrected p.**

The four canonical univariate tests are **one family with a swappable** ``method=`` —
the *downstream is identical* (a per-feature effect + CI + p + BH-q table; the volcano
and p-value histogram read off it), so they are not four templates:

  * ``ols``        — per-feature linear model (general form; adjusts for covariates).
  * ``moderated``  — limma-style empirical-Bayes variance moderation of the OLS fit
                     (the **default** — borrowing variance across features is more
                     powerful and honest at omics scale; conventions/statistics.md).
  * ``welch``      — Welch's unequal-variance two-group t-test (no covariates).
  * ``mannwhitney``— Mann-Whitney U rank test, the nonparametric fallback when
                     normality fails (no covariates; effect = Hodges-Lehmann shift).

The study-agnostic interface is a **contrast + covariates** specification over the
:class:`~common.data_loading.Dataset` metadata (§A.0b of the build plan): name the
column under test and the nuisance columns to adjust for; the treatment-contrast design
is assembled internally. There are **no per-study design-matrix builders** — those bake
in one study's schema and belong in the project copy (interaction / transformed models
are a documented project-local adaptation until a ``formula=`` escape hatch lands).

Scale: the effect is a **log2 fold change**, so the input should be on a log scale
(``log2``/``glog2``); on a non-log scale the test still runs but the effect is a raw
difference and the normality assumption is violated, so a
:class:`DifferentialAbundanceScaleWarning` is raised and the effect label stays honest
about the scale. **Missing values are an upstream Stage-2 decision** (the matrix should
already be resolved — dropped / imputed / zeroed; conventions/statistics.md): this
template **never silently imputes or zeros**, it **raises** on any ``NaN`` in the
abundances as a defense-in-depth backstop. A feature that is constant within the
analyzed samples is legitimately untestable — its p is left ``NaN`` and excluded from
the BH family (not an error).

Sample set: the caller passes the **experimental subset** (controls excluded upstream;
conventions/statistics.md) and records the analyzed set in the finding's
``provenance.params``. This template does not re-filter — it tests whatever it is given.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from common.data_loading import Dataset
from scipy import optimize, special, stats

Method = Literal["ols", "moderated", "welch", "mannwhitney"]

# Scales on which the effect is a genuine log fold change and the t-test normality
# assumption is reasonable. Outside this set the test warns (see _check_scale).
_LOG2_LIKE: frozenset[str] = frozenset({"log2", "glog2"})

# Below this many features the empirical-Bayes prior fit (2 parameters from the spread
# of observed variances) is ill-conditioned. Real omics scopes have thousands; this
# catches a programming error (moderating a handful of features).
_MIN_FEATURES_FOR_PRIOR_FIT = 50

# A numeric metadata column with at most this many distinct values, used as a continuous
# term, is suspicious — the classic ``batch = 1, 2, 3`` miscoded-factor trap. Warn so
# the scientist can pass it via ``categorical=`` if it is really a factor.
_LOW_CARDINALITY_NUMERIC = 5

__script_meta__: dict[str, object] = {
    "template": {"name": "differential-abundance", "version": "0.1"},
    "kind": "analysis",
    "provides": [
        "Method",
        "DifferentialAbundanceScaleWarning",
        "LowCardinalityNumericWarning",
        "DifferentialAbundanceResult",
        "differential_abundance",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Univariate differential abundance over a Dataset: one swappable family "
        "(ols / moderated / welch / mannwhitney) sharing a per-feature effect + CI + p "
        "+ BH-q table. Study-agnostic contrast + covariates API over the sample "
        "metadata (treatment-contrast design assembled internally; no per-study "
        "builders); moderated (limma-style empirical Bayes) is the default. Warns on a "
        "non-log scale (effect is a log2 fold change), raises on NaN abundances "
        "(missing handling is upstream), leaves constant features untestable "
        "(NaN p, excluded from BH). Requires scipy."
    ),
}


class DifferentialAbundanceScaleWarning(UserWarning):
    """The abundances are not on a log2-like scale, so the effect is not a log2 FC."""


class LowCardinalityNumericWarning(UserWarning):
    """A numeric metadata column with very few distinct values is used as continuous.

    The likely cause is a factor encoded as integers (``batch = 1, 2, 3``). Pass it via
    ``categorical=`` to treat it as a factor, or confirm it is a continuous slope.
    """


@dataclass(frozen=True)
class DifferentialAbundanceResult:
    """Per-feature differential-abundance results for one contrast.

    Attributes
    ----------
    table:
        The full long-format results — one row per ``(feature, term)`` for **every**
        non-intercept design term (the contrast term(s) **and** the covariate terms), so
        "report all tests run" is honored. Columns: ``feature``, ``term``,
        ``is_contrast`` (bool), ``effect``, ``ci_low``, ``ci_high``, ``statistic``,
        ``p``, ``q`` (BH within each term), ``mean_abundance``, ``n``. Sorted with the
        contrast term(s) first, then ``q`` ascending. For ``welch``/``mannwhitney`` only
        the contrast term(s) appear (those tests do not adjust for covariates).
    contrast_table:
        Convenience view of :attr:`table` restricted to the contrast term(s) — the
        deliverable that the volcano and the finding read. (A property; see below.)
    contrast, covariates, method, reference:
        The resolved request (``reference`` is the per-column reference level actually
        used; the effect sign is *non-reference vs reference*).
    contrast_terms:
        The contrast term name(s): one for a binary/continuous contrast, ``k-1`` for a
        ``k``-level categorical contrast (each non-reference level vs the reference).
    effect_label:
        Human-readable effect description for the volcano x-axis (e.g.
        ``"log2 fold change"``), honest about the scale and the estimator.
    n_samples:
        Number of samples in the analyzed dataset.
    prior_variance, prior_df:
        The empirical-Bayes prior ``(s0^2, d0)`` for ``method="moderated"``; ``None``
        otherwise. ``prior_df`` is ``inf`` in the fully-shrunk limit.
    """

    table: pd.DataFrame
    contrast: str
    covariates: tuple[str, ...]
    method: Method
    reference: dict[str, str]
    contrast_terms: tuple[str, ...]
    effect_label: str
    n_samples: int
    prior_variance: float | None
    prior_df: float | None

    @property
    def contrast_table(self) -> pd.DataFrame:
        """The contrast-term rows of :attr:`table`, sorted by ``q`` ascending."""
        sub = self.table[self.table["is_contrast"]].copy()
        return sub.sort_values("q", kind="stable", na_position="last").reset_index(
            drop=True
        )


# --------------------------------------------------------------------------- #
# Multiple testing
# --------------------------------------------------------------------------- #
def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p-values (``NaN`` preserved & excluded).

    Self-contained (no statsmodels dependency): the standard step-up procedure over the
    finite p-values, with monotone enforcement and clipping to ``[0, 1]``.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = adj
    out[valid] = restored
    return out


# --------------------------------------------------------------------------- #
# Internal: term spec + design construction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Term:
    """One non-intercept design column: its name, values, and contrast flag."""

    name: str
    data: np.ndarray
    is_contrast: bool


def _is_numeric(series: pd.Series) -> bool:
    """True if the column is a numeric dtype (and thus a continuous-slope candidate)."""
    return bool(pd.api.types.is_numeric_dtype(series))


def _column_terms(
    metadata: pd.DataFrame,
    column: str,
    *,
    is_contrast: bool,
    categorical: frozenset[str],
    reference: dict[str, str],
) -> tuple[list[_Term], str | None]:
    """Build the design term(s) for one metadata column.

    Returns ``(terms, used_reference)`` — ``used_reference`` is the reference level for
    categorical column (``None`` for a continuous one), for provenance.
    """
    series = metadata[column]
    treat_categorical = column in categorical or not _is_numeric(series)

    if not treat_categorical:
        values = series.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Continuous column {column!r} has non-finite values; clean or drop "
                f"those samples upstream."
            )
        n_distinct = int(np.unique(values).size)
        if n_distinct < 2:
            raise ValueError(
                f"Column {column!r} is constant ({n_distinct} distinct value); it "
                f"cannot be a contrast or covariate."
            )
        if n_distinct <= _LOW_CARDINALITY_NUMERIC:
            warnings.warn(
                f"Numeric column {column!r} has only {n_distinct} distinct values and "
                f"is being used as a continuous slope. If it is really a factor (e.g. "
                f"batch coded 1/2/3), pass categorical=[{column!r}].",
                LowCardinalityNumericWarning,
                stacklevel=3,
            )
        return [_Term(name=column, data=values, is_contrast=is_contrast)], None

    labels = series.astype(str).to_numpy()
    levels = sorted(set(labels.tolist()))
    if len(levels) < 2:
        raise ValueError(
            f"Categorical column {column!r} has {len(levels)} level(s); a contrast or "
            f"covariate needs at least 2."
        )
    ref = reference.get(column, levels[0])
    if ref not in levels:
        raise ValueError(
            f"reference level {ref!r} for column {column!r} is not present; "
            f"levels are {levels}."
        )
    terms = [
        _Term(
            name=f"{column}[{level} vs {ref}]",
            data=(labels == level).astype(float),
            is_contrast=is_contrast,
        )
        for level in levels
        if level != ref
    ]
    return terms, ref


def _effect_label(scale: str, method: Method) -> str:
    """Human-readable effect description, honest about the scale and the estimator."""
    if scale in _LOG2_LIKE:
        base = "log2 fold change"
    elif scale == "ln":
        base = "ln fold change"
    elif scale == "log10":
        base = "log10 fold change"
    elif scale == "zscore":
        base = "z-score difference"
    else:
        base = f"difference ({scale} scale)"
    if method == "mannwhitney":
        return f"{base} (Hodges-Lehmann shift)"
    return base


def _check_scale(scale: str) -> None:
    if scale not in _LOG2_LIKE:
        warnings.warn(
            f"Differential abundance on scale {scale!r}: the effect is reported as a "
            f"fold change but is only a true log2 fold change on a log2-like scale "
            f"(log2/glog2), and the t-test normality assumption is for log-abundances. "
            f"Run on log2-transformed data unless you have a specific reason not to.",
            DifferentialAbundanceScaleWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------- #
# Internal: OLS / moderated core
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _OLSFit:
    coefficients: np.ndarray  # (n_params, n_features)
    sigma2: np.ndarray  # (n_features,) residual variance
    inv_diag: np.ndarray  # (n_params,) diag of (X'X)^-1
    df: int


def _fit_ols(design: np.ndarray, abundances: np.ndarray) -> _OLSFit:
    """Vectorized closed-form per-feature OLS sharing one ``(X'X)^-1``.

    Coefficients/SEs/p-values match per-feature ``statsmodels.OLS``; the single
    factorization makes it an omics-scale matrix op. ``abundances`` is ``(n_samples,
    n_features)``; ``design`` is ``(n_samples, n_params)`` with a leading intercept.
    """
    n_samples, n_params = design.shape
    df = n_samples - n_params
    if df <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) to fit {n_params} parameters (df={df}). "
            f"Drop covariates or collect more samples."
        )
    xtx = design.T @ design
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Design matrix is singular — perfect collinearity among the contrast and "
            "covariates within the analyzed samples (e.g. a covariate constant in one "
            "arm, or two aliased factors). Drop the redundant term."
        ) from exc
    coefficients = xtx_inv @ design.T @ abundances
    residuals = abundances - design @ coefficients
    rss = np.sum(residuals * residuals, axis=0)
    sigma2 = rss / df
    inv_diag = np.diag(xtx_inv).copy()
    return _OLSFit(coefficients=coefficients, sigma2=sigma2, inv_diag=inv_diag, df=df)


def _fit_f_distribution_prior(sigma2: np.ndarray, df: int) -> tuple[float, float]:
    """Smyth's ``fitFDist`` (limma): scaled inverse-chi2 prior by method-of-moments.

    Fits ``sigma2_i ~ s0^2 * chi2(d0) / d0`` from the spread of ``log(sigma2)`` with
    digamma/trigamma identities. Returns ``(s0^2, d0)``; ``d0`` is ``inf`` (``s0^2``
    from the ``d0 -> inf`` limit) when the observed variances are no more spread than
    sampling alone predicts.
    """
    sigma2 = np.asarray(sigma2, dtype=float)
    valid = np.isfinite(sigma2) & (sigma2 > 0)
    n_valid = int(valid.sum())
    if n_valid < _MIN_FEATURES_FOR_PRIOR_FIT:
        raise ValueError(
            f"Variance moderation needs {_MIN_FEATURES_FOR_PRIOR_FIT} features "
            f"with finite positive residual variance; got {n_valid}. Use method='ols' "
            f"for a small feature set."
        )
    if df <= 0:
        raise ValueError(f"df must be positive; got {df}.")

    z = np.log(sigma2[valid])
    z_mean = float(z.mean())
    z_var = float(z.var(ddof=1))

    digamma_d_half = float(special.digamma(df / 2))
    target = z_var - float(special.polygamma(1, df / 2))

    def _s0_sq_at_d0_inf() -> float:
        log_s0_sq = z_mean - digamma_d_half + float(np.log(df / 2))
        return float(np.exp(log_s0_sq))

    if target <= 0:
        return _s0_sq_at_d0_inf(), float("inf")

    def f(x: float) -> float:
        return float(special.polygamma(1, x / 2)) - target

    lo, hi = 1e-6, 1e6
    if f(hi) > 0:
        return _s0_sq_at_d0_inf(), float("inf")

    d0 = float(optimize.brentq(f, lo, hi))
    log_s0_sq = (
        z_mean
        - digamma_d_half
        + float(np.log(df / 2))
        + float(special.digamma(d0 / 2))
        - float(np.log(d0 / 2))
    )
    return float(np.exp(log_s0_sq)), d0


@dataclass(frozen=True)
class _TermStats:
    """Per-feature statistics for one design term."""

    effect: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    statistic: np.ndarray
    p: np.ndarray
    q: np.ndarray


def _term_stats_from_t(
    effect: np.ndarray, se: np.ndarray, df: float, alpha: float
) -> _TermStats:
    """Two-sided t-test stats + a (1-alpha) CI from effect/SE/df, ``NaN`` where se<=0.

    A constant feature gives ``se == 0`` (untestable): t/p/CI are ``NaN`` and it drops
    out of the BH family, rather than producing a spuriously infinite t.
    """
    good = se > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(good, effect / se, np.nan)
        if np.isinf(df):
            p = 2.0 * np.asarray(stats.norm.sf(np.abs(t)), dtype=float)
            crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        else:
            p = 2.0 * np.asarray(stats.t.sf(np.abs(t), df=df), dtype=float)
            crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    half = np.where(good, crit * se, np.nan)
    ci_low = np.where(good, effect - half, np.nan)
    ci_high = np.where(good, effect + half, np.nan)
    return _TermStats(
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        statistic=t,
        p=p,
        q=bh_adjust(p),
    )


# --------------------------------------------------------------------------- #
# Internal: two-group tests (welch / mann-whitney)
# --------------------------------------------------------------------------- #
def _welch_stats(group1: np.ndarray, group0: np.ndarray, alpha: float) -> _TermStats:
    """Welch (unequal-variance) two-group t per feature: mean diff (g1-g0) + CI."""
    n1, n0 = group1.shape[0], group0.shape[0]
    m1, m0 = group1.mean(axis=0), group0.mean(axis=0)
    v1 = group1.var(axis=0, ddof=1)
    v0 = group0.var(axis=0, ddof=1)
    effect = m1 - m0
    se = np.sqrt(v1 / n1 + v0 / n0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Welch-Satterthwaite df, per feature.
        num = (v1 / n1 + v0 / n0) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1)
        df_feat = np.where(den > 0, num / den, np.nan)
        good = (se > 0) & np.isfinite(df_feat)
        t = np.where(good, effect / se, np.nan)
        p = np.where(good, 2.0 * stats.t.sf(np.abs(t), df=df_feat), np.nan)
        crit = np.where(good, stats.t.ppf(1.0 - alpha / 2.0, df=df_feat), np.nan)
    half = crit * se
    ci_low = np.where(good, effect - half, np.nan)
    ci_high = np.where(good, effect + half, np.nan)
    return _TermStats(
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        statistic=np.asarray(t, dtype=float),
        p=np.asarray(p, dtype=float),
        q=bh_adjust(np.asarray(p, dtype=float)),
    )


def _mannwhitney_stats(
    group1: np.ndarray, group0: np.ndarray, alpha: float
) -> _TermStats:
    """Mann-Whitney U per feature + Hodges-Lehmann shift (g1-g0) with a rank CI.

    The effect is the median of all pairwise differences ``x_i - y_j`` (Hodges-Lehmann),
    on the data's own (log) scale so it lines up with the other methods' effect. The
    CI is the distribution-free Mann-Whitney rank interval: order statistics of the
    pairwise differences at the rank implied by the normal approximation to the null.
    """
    n1, n0 = group1.shape[0], group0.shape[0]
    n_features = group1.shape[1]
    u_stat, p = stats.mannwhitneyu(group1, group0, axis=0, alternative="two-sided")
    u_stat = np.asarray(u_stat, dtype=float)
    p = np.asarray(p, dtype=float)

    # Pairwise differences x_i - y_j: shape (n1*n0, n_features), sorted per feature.
    diffs = (group1[:, None, :] - group0[None, :, :]).reshape(n1 * n0, n_features)
    diffs.sort(axis=0)
    n_pairs = n1 * n0
    effect = np.median(diffs, axis=0)

    # Rank offset for the (1-alpha) CI from the normal approximation to the U null.
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    mean_u = n_pairs / 2.0
    sd_u = float(np.sqrt(n_pairs * (n1 + n0 + 1) / 12.0))
    k = int(np.floor(mean_u - z * sd_u))  # 0-based lower order-statistic index
    if 0 <= k < n_pairs and (n_pairs - 1 - k) >= 0:
        ci_low = diffs[k, :]
        ci_high = diffs[n_pairs - 1 - k, :]
    else:  # too few samples to bound — honest NaN rather than a fake interval
        ci_low = np.full(n_features, np.nan)
        ci_high = np.full(n_features, np.nan)
    return _TermStats(
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        statistic=u_stat,
        p=p,
        q=bh_adjust(p),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def differential_abundance(
    dataset: Dataset,
    contrast: str,
    *,
    covariates: tuple[str, ...] | list[str] = (),
    reference: dict[str, str] | None = None,
    method: Method = "moderated",
    categorical: tuple[str, ...] | list[str] = (),
    alpha: float = 0.05,
) -> DifferentialAbundanceResult:
    """Test every feature for differential abundance across ``contrast``.

    Parameters
    ----------
    dataset:
        The **experimental subset** on a **log2-like** scale (controls excluded and
        missing values resolved upstream — conventions/statistics.md). Abundances are
        finite (a ``NaN`` raises).
    contrast:
        The metadata column under test. A non-numeric column (or one named in
        ``categorical``) is a factor — its effect is *non-reference vs reference*, with
        ``k-1`` terms for ``k`` levels. A numeric column is a continuous slope.
    covariates:
        Nuisance metadata columns to adjust for (batch lives here for significance
        testing — conventions/statistics.md). Only ``ols``/``moderated`` adjust; passing
        covariates with ``welch``/``mannwhitney`` is an error.
    reference:
        Optional per-column reference level ``{column: level}``. Default is the
        sorted-first level. The reference controls the effect sign and is recorded.
    method:
        ``"moderated"`` (default), ``"ols"``, ``"welch"``, or ``"mannwhitney"``.
    categorical:
        Metadata columns to force-treat as factors even though they are numeric (the
        ``batch = 1/2/3`` case). Numeric columns not listed here are continuous slopes
        (with a low-cardinality warning).
    alpha:
        Two-sided significance level for the confidence intervals (default ``0.05`` →
        95% CI). Does not affect the BH-q values.

    Returns
    -------
    DifferentialAbundanceResult
    """
    if method not in ("ols", "moderated", "welch", "mannwhitney"):
        raise ValueError(
            f"Unknown method {method!r}; expected ols/moderated/welch/mannwhitney."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")

    covariates = tuple(covariates)
    categorical_set = frozenset(categorical)
    reference = dict(reference) if reference is not None else {}

    metadata = dataset.metadata
    abundances = np.asarray(dataset.abundances, dtype=float)
    feature_names = np.asarray(dataset.feature_names)
    if abundances.ndim != 2:
        raise ValueError(f"abundances must be 2D; got shape {abundances.shape}.")
    n_samples, n_features = abundances.shape
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must equal n_features "
            f"({n_features})."
        )
    if len(metadata) != n_samples:
        raise ValueError(
            f"metadata has {len(metadata)} rows but abundances has {n_samples} samples."
        )
    if not np.all(np.isfinite(abundances)):
        raise ValueError(
            "abundances contain NaN/inf. Missing-value handling is an upstream Stage-2 "
            "decision (drop/impute/zero — conventions/statistics.md); this test does "
            "not silently impute. Resolve missingness before differential abundance."
        )

    if contrast not in metadata.columns:
        raise ValueError(f"contrast column {contrast!r} not in metadata.")
    if contrast in covariates:
        raise ValueError(f"contrast {contrast!r} also listed as a covariate.")
    missing_cov = [c for c in covariates if c not in metadata.columns]
    if missing_cov:
        raise ValueError(f"covariate column(s) not in metadata: {missing_cov}.")

    _check_scale(dataset.scale)
    effect_label = _effect_label(dataset.scale, method)
    mean_abundance_all = abundances.mean(axis=0)

    if method in ("welch", "mannwhitney"):
        result = _run_two_group(
            method=method,
            contrast=contrast,
            covariates=covariates,
            categorical_set=categorical_set,
            reference=reference,
            metadata=metadata,
            abundances=abundances,
            feature_names=feature_names,
            effect_label=effect_label,
            alpha=alpha,
        )
        return result

    return _run_linear_model(
        method=method,
        contrast=contrast,
        covariates=covariates,
        categorical_set=categorical_set,
        reference=reference,
        metadata=metadata,
        abundances=abundances,
        feature_names=feature_names,
        mean_abundance_all=mean_abundance_all,
        effect_label=effect_label,
        alpha=alpha,
    )


def _resolve_terms(
    *,
    contrast: str,
    covariates: tuple[str, ...],
    categorical_set: frozenset[str],
    reference: dict[str, str],
    metadata: pd.DataFrame,
) -> tuple[list[_Term], dict[str, str]]:
    """Build all non-intercept terms (contrast first) and the references used."""
    used_reference: dict[str, str] = {}
    contrast_terms, ref = _column_terms(
        metadata,
        contrast,
        is_contrast=True,
        categorical=categorical_set,
        reference=reference,
    )
    if ref is not None:
        used_reference[contrast] = ref
    terms = list(contrast_terms)
    for cov in covariates:
        cov_terms, cov_ref = _column_terms(
            metadata,
            cov,
            is_contrast=False,
            categorical=categorical_set,
            reference=reference,
        )
        if cov_ref is not None:
            used_reference[cov] = cov_ref
        terms.extend(cov_terms)
    return terms, used_reference


def _assemble_table(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    # Contrast terms first, then ascending q (NaN last) within the ordering.
    table = table.sort_values(
        ["is_contrast", "q"],
        ascending=[False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    return table


def _term_rows(
    *,
    term_name: str,
    is_contrast: bool,
    stats_: _TermStats,
    feature_names: np.ndarray,
    mean_abundance: np.ndarray,
    n: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": feature_names,
            "term": term_name,
            "is_contrast": is_contrast,
            "effect": stats_.effect,
            "ci_low": stats_.ci_low,
            "ci_high": stats_.ci_high,
            "statistic": stats_.statistic,
            "p": stats_.p,
            "q": stats_.q,
            "mean_abundance": mean_abundance,
            "n": n,
        }
    )


def _run_linear_model(
    *,
    method: Method,
    contrast: str,
    covariates: tuple[str, ...],
    categorical_set: frozenset[str],
    reference: dict[str, str],
    metadata: pd.DataFrame,
    abundances: np.ndarray,
    feature_names: np.ndarray,
    mean_abundance_all: np.ndarray,
    effect_label: str,
    alpha: float,
) -> DifferentialAbundanceResult:
    terms, used_reference = _resolve_terms(
        contrast=contrast,
        covariates=covariates,
        categorical_set=categorical_set,
        reference=reference,
        metadata=metadata,
    )
    n_samples = abundances.shape[0]
    intercept = np.ones((n_samples, 1), dtype=float)
    design = np.column_stack([intercept, *(t.data for t in terms)])

    fit = _fit_ols(design, abundances)

    prior_variance: float | None = None
    prior_df: float | None = None
    if method == "moderated":
        s0_sq, d0 = _fit_f_distribution_prior(fit.sigma2, fit.df)
        prior_variance, prior_df = s0_sq, d0
        if np.isinf(d0):
            sigma2_eff = np.full_like(fit.sigma2, s0_sq)
            df_eff: float = float("inf")
        else:
            sigma2_eff = (d0 * s0_sq + fit.df * fit.sigma2) / (d0 + fit.df)
            df_eff = float(d0 + fit.df)
    else:
        sigma2_eff = fit.sigma2
        df_eff = float(fit.df)

    frames: list[pd.DataFrame] = []
    contrast_terms: list[str] = []
    for idx, term in enumerate(terms, start=1):
        if term.is_contrast:
            contrast_terms.append(term.name)
        effect = fit.coefficients[idx, :]
        se = np.sqrt(fit.inv_diag[idx] * sigma2_eff)
        stats_ = _term_stats_from_t(effect, se, df_eff, alpha)
        frames.append(
            _term_rows(
                term_name=term.name,
                is_contrast=term.is_contrast,
                stats_=stats_,
                feature_names=feature_names,
                mean_abundance=mean_abundance_all,
                n=n_samples,
            )
        )

    table = _assemble_table(pd.concat(frames, ignore_index=True).to_dict("records"))
    return DifferentialAbundanceResult(
        table=table,
        contrast=contrast,
        covariates=covariates,
        method=method,
        reference=used_reference,
        contrast_terms=tuple(contrast_terms),
        effect_label=effect_label,
        n_samples=n_samples,
        prior_variance=prior_variance,
        prior_df=prior_df,
    )


def _run_two_group(
    *,
    method: Method,
    contrast: str,
    covariates: tuple[str, ...],
    categorical_set: frozenset[str],
    reference: dict[str, str],
    metadata: pd.DataFrame,
    abundances: np.ndarray,
    feature_names: np.ndarray,
    effect_label: str,
    alpha: float,
) -> DifferentialAbundanceResult:
    if covariates:
        raise ValueError(
            f"method={method!r} is an unadjusted two-group test and cannot control for "
            f"covariates {list(covariates)}. Use method='ols'/'moderated' to adjust, "
            f"or drop the covariates."
        )
    series = metadata[contrast]
    if contrast not in categorical_set and _is_numeric(series):
        raise ValueError(
            f"method={method!r} needs a categorical contrast (two groups per term); "
            f"{contrast!r} is numeric. Use method='ols'/'moderated' for a continuous "
            f"contrast, or pass categorical=[{contrast!r}] if it is a coded factor."
        )
    labels = series.astype(str).to_numpy()
    levels = sorted(set(labels.tolist()))
    if len(levels) < 2:
        raise ValueError(
            f"contrast {contrast!r} has {len(levels)} level(s); need >= 2."
        )
    ref = reference.get(contrast, levels[0])
    if ref not in levels:
        raise ValueError(
            f"reference level {ref!r} for {contrast!r} not present; levels {levels}."
        )

    ref_mask = labels == ref
    group0 = abundances[ref_mask, :]
    frames: list[pd.DataFrame] = []
    contrast_terms: list[str] = []
    for level in levels:
        if level == ref:
            continue
        level_mask = labels == level
        group1 = abundances[level_mask, :]
        n_pair = int(level_mask.sum() + ref_mask.sum())
        if int(level_mask.sum()) < 2 or int(ref_mask.sum()) < 2:
            raise ValueError(
                f"two-group test for {contrast}={level!r} vs {ref!r} needs >=2 samples "
                f"per group; got {int(level_mask.sum())} and {int(ref_mask.sum())}."
            )
        if method == "welch":
            stats_ = _welch_stats(group1, group0, alpha)
        else:
            stats_ = _mannwhitney_stats(group1, group0, alpha)
        # Mean abundance over the two groups actually compared in this term.
        pair_mean = abundances[level_mask | ref_mask, :].mean(axis=0)
        term_name = f"{contrast}[{level} vs {ref}]"
        contrast_terms.append(term_name)
        frames.append(
            _term_rows(
                term_name=term_name,
                is_contrast=True,
                stats_=stats_,
                feature_names=feature_names,
                mean_abundance=pair_mean,
                n=n_pair,
            )
        )

    table = _assemble_table(pd.concat(frames, ignore_index=True).to_dict("records"))
    return DifferentialAbundanceResult(
        table=table,
        contrast=contrast,
        covariates=(),
        method=method,
        reference={contrast: ref},
        contrast_terms=tuple(contrast_terms),
        effect_label=effect_label,
        n_samples=abundances.shape[0],
        prior_variance=None,
        prior_df=None,
    )
