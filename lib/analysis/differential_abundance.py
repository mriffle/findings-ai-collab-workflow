"""Univariate differential abundance — one feature at a time, then BH across features.

TEMPLATE (lib/) — a *seed* for a project's differential-abundance script, not a finished
analysis. Copy it into the project's ``scripts/`` and adapt the call site per study
(which metadata column is the contrast, which are nuisance covariates). Held to the
correctness charter (conventions/correctness.md) and the statistics convention
(conventions/statistics.md): **assume nothing, verify everything, fail loud; no bare p —
every result is effect + CI + corrected p.**

The canonical univariate tests are **one family with a swappable** ``method=`` — the
*downstream is identical* (a per-feature effect + CI + p + BH-q table; the volcano and
p-value histogram read off it), so they are not separate templates:

  * ``ols``        — per-feature linear model (general form; adjusts for covariates).
  * ``moderated``  — limma-style empirical-Bayes variance moderation of the OLS fit
                     (the **default** — borrowing variance across features is more
                     powerful and honest at omics scale; conventions/statistics.md).
  * ``welch``      — Welch's unequal-variance two-group t-test (no covariates).
  * ``mannwhitney``— Mann-Whitney U rank test, the nonparametric fallback when
                     normality fails (no covariates; effect = Hodges-Lehmann shift).
  * ``signed_rank``— Wilcoxon signed-rank test on within-unit pairs (needs ``unit=``;
                     the nonparametric paired test; effect = Hodges-Lehmann
                     pseudo-median of the paired differences).

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

UNIT OF ANALYSIS (v0.2). Every tested row must be an independent unit. Pass ``unit=``
(the metadata column naming the biological unit — animal, patient, subject) and the
template **refuses pseudoreplication** by a fixed decision table:

1. No unit repeats → the plain model (the normal state after ``aggregate_replicates``).
2. A unit holds more than one row at the same contrast level → **raise**: technical
   replicates (aggregate first), or — if those rows differ on a covariate — a
   between-unit contrast with repeated measures, which needs a mixed model (not
   supported; the error says so).
3. Otherwise the contrast varies *within* units (paired / repeated-measures): ``ols`` /
   ``moderated`` add unit fixed effects (estimated, **not tested**, not in the table —
   the table carries every tested term), so each unit is its own control. On a binary
   contrast ``method="ols"`` with ``unit=`` **is** the paired t-test; ``moderated`` is
   limma's paired design (``~ unit + contrast``). ``signed_rank`` is the nonparametric
   paired test. ``welch`` / ``mannwhitney`` refuse repeated units.

``weights=`` names a metadata column of per-row precision weights (normally the
``precision_weight`` column ``aggregate_replicates`` emits when units were averaged
over unequal run counts): ``ols`` / ``moderated`` then fit weighted least squares (rows
of the design and the abundances scaled by ``sqrt(w)`` — invariant to the weights'
overall scale), and the moderation applies unchanged. ``mean_abundance`` stays
unweighted.

Sample set: the caller passes the **experimental subset** (controls excluded upstream;
conventions/statistics.md) and records the analyzed set in the finding's
``provenance.params``. This template does not re-filter — it tests whatever it is given.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from common.data_loading import Dataset
from scipy import optimize, special, stats

Method = Literal["ols", "moderated", "welch", "mannwhitney", "signed_rank"]
_METHODS: tuple[str, ...] = ("ols", "moderated", "welch", "mannwhitney", "signed_rank")

# Metadata columns written by the ``aggregate-replicates`` template. Spelled out here
# (not imported) so this seed stands alone; used only to warn when an aggregated
# Dataset with unequal run counts is tested without its precision weights.
_N_REPLICATES = "n_replicates"
_PRECISION_WEIGHT = "precision_weight"

# The signed-rank CI uses the exact null distribution of T+ up to this many nonzero
# pairs (and no tied |differences|); the normal approximation above it.
_EXACT_SIGNED_RANK_MAX = 50

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
    "template": {"name": "differential-abundance", "version": "0.2"},
    "kind": "analysis",
    "provides": [
        "Method",
        "DifferentialAbundanceScaleWarning",
        "LowCardinalityNumericWarning",
        "UnweightedReplicatesWarning",
        "UnpairedUnitWarning",
        "LowPowerRankTestWarning",
        "DifferentialAbundanceResult",
        "differential_abundance",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Univariate differential abundance over a Dataset: one swappable family "
        "(ols / moderated / welch / mannwhitney / signed_rank) sharing a per-feature "
        "effect + CI + p + BH-q table. Study-agnostic contrast + covariates API over "
        "the sample metadata (treatment-contrast design assembled internally; no "
        "per-study builders); moderated (limma-style empirical Bayes) is the default. "
        "unit= refuses pseudoreplication (technical replicates / repeated measures) "
        "and fits unit fixed effects for within-unit (paired) contrasts; weights= fits "
        "weighted least squares (precision weights from aggregate-replicates). Warns "
        "on a "
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
    ``categorical=`` to treat it as a factor, or confirm it is a continuous slope. Only
    all-integer columns trigger it (a fraction column such as ``Batch[frac B]`` does
    not).
    """


class UnweightedReplicatesWarning(UserWarning):
    """An aggregated Dataset with unequal run counts is fit without its weights.

    Units averaged over more runs are more precise; when run counts differ (and they
    often track the contrast), pass ``weights="precision_weight"``. Raised only for
    ``ols``/``moderated``: the rank and two-group tests cannot take weights, so unequal
    unit precision is unaccounted for there — prefer ``moderated`` with weights.
    """


class UnpairedUnitWarning(UserWarning):
    """Some units lack one of the two compared levels and drop out of a paired term."""


class LowPowerRankTestWarning(UserWarning):
    """Too few pairs for the signed-rank test to reach ``alpha`` at all.

    With ``m`` pairs the smallest attainable two-sided p is ``2 / 2**m``; at ``m <= 5``
    it exceeds 0.05, so no feature can be significant whatever the data.
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
        contrast term(s) first, then ``q`` ascending. For ``welch``/``mannwhitney``/
        ``signed_rank`` only the contrast term(s) appear (those tests do not adjust for
        covariates).
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
    unit, n_units, n_informative_units, unit_fixed_effects:
        The ``unit=`` column, its number of distinct units, how many units the
        contrast varies within, and whether unit fixed effects entered the model
        (``None``/``False`` when ``unit`` was not given — defaults keep older cached
        results loadable).
    weights:
        The ``weights=`` column used for weighted least squares, or ``None``.
    df_residual:
        The residual degrees of freedom of the linear model (before moderation);
        ``None`` for the rank / two-group tests.
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
    unit: str | None = None
    n_units: int | None = None
    n_informative_units: int | None = None
    unit_fixed_effects: bool = False
    weights: str | None = None
    df_residual: float | None = None

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
        all_integer = bool(np.all(values == np.round(values)))
        if n_distinct <= _LOW_CARDINALITY_NUMERIC and all_integer:
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
    if method == "signed_rank":
        return f"{base} (Hodges-Lehmann pseudo-median of paired differences)"
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
    singular = (
        "Design matrix is singular — perfect collinearity among the contrast and "
        "covariates within the analyzed samples (e.g. a covariate constant in one "
        "arm, or two aliased factors). Drop the redundant term."
    )
    # np.linalg.inv does not reliably raise on a rank-deficient design — it can return
    # a numerically garbage inverse — so check the rank explicitly first.
    if int(np.linalg.matrix_rank(design)) < n_params:
        raise ValueError(singular)
    xtx = design.T @ design
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError(singular) from exc
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


def _signed_rank_null_cdf(n: int) -> np.ndarray:
    """Exact null CDF of the Wilcoxon ``T+`` for ``n`` untied nonzero differences.

    ``cdf[t] = P(T+ <= t)`` for ``t = 0 .. n(n+1)/2``, from the subset-sum counts of
    ``{1..n}`` (each rank enters ``T+`` with probability 1/2).
    """
    max_sum = n * (n + 1) // 2
    counts = np.zeros(max_sum + 1, dtype=float)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        counts[rank:] = counts[rank:] + counts[:-rank].copy()
    return np.asarray(np.cumsum(counts) / 2.0**n, dtype=float)


def _walsh_ci_index(n: int, alpha: float, *, exact: bool) -> int | None:
    """0-based order-statistic index ``c`` of the signed-rank CI (Walsh averages).

    The ``(1 - alpha)`` interval is ``[W[c], W[M - 1 - c]]`` (``M = n(n+1)/2`` sorted
    Walsh averages). Exact: the largest ``c`` with ``P(T+ <= c) <= alpha/2`` (coverage
    ``>= 1 - alpha``); ``None`` when even ``c = 0`` is too likely (``n <= 5`` at
    ``alpha = 0.05``) — no honest interval exists. Otherwise the normal approximation.
    """
    n_walsh = n * (n + 1) // 2
    if exact:
        cdf = _signed_rank_null_cdf(n)
        ok = np.flatnonzero(cdf <= alpha / 2.0)
        return int(ok[-1]) if ok.size else None
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    sd = float(np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0))
    c = int(np.floor(n_walsh / 2.0 - z * sd))
    return c if 0 <= c < n_walsh else None


def _signed_rank_stats(diffs: np.ndarray, alpha: float) -> _TermStats:
    """Wilcoxon signed-rank per feature on paired differences + a Hodges-Lehmann CI.

    ``diffs`` is ``(n_pairs, n_features)`` of ``level - reference`` within each unit.
    Zero differences are dropped per feature (R ``wilcox.test`` / scipy
    ``zero_method="wilcox"``), so the test, the estimate, and the CI all use the same
    nonzero count ``n'``. A feature with ``n' = 0`` is untestable: ``NaN`` p, excluded
    from BH (scipy would report p = 1). The effect is the Hodges-Lehmann pseudo-median
    (the median of the Walsh averages ``(d_i + d_j)/2``, ``i <= j``); the p-value and
    the CI are exact for ``n' <= 50`` with no tied ``|d|``, normal-approximation above.
    """
    n_features = diffs.shape[1]
    effect = np.full(n_features, np.nan)
    ci_low = np.full(n_features, np.nan)
    ci_high = np.full(n_features, np.nan)
    statistic = np.full(n_features, np.nan)
    p = np.full(n_features, np.nan)
    index_cache: dict[tuple[int, bool], int | None] = {}
    for j in range(n_features):
        d = diffs[:, j]
        nz = d[d != 0.0]
        n = int(nz.size)
        if n == 0:
            continue
        abs_d = np.abs(nz)
        exact = n <= _EXACT_SIGNED_RANK_MAX and np.unique(abs_d).size == n
        ranks = stats.rankdata(abs_d)
        statistic[j] = float(ranks[nz > 0].sum())
        res = stats.wilcoxon(
            nz,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact" if exact else "approx",
        )
        p[j] = float(res.pvalue)
        rows, cols = np.triu_indices(n)
        walsh = np.sort((nz[rows] + nz[cols]) / 2.0)
        effect[j] = float(np.median(walsh))
        key = (n, bool(exact))
        if key not in index_cache:
            index_cache[key] = _walsh_ci_index(n, alpha, exact=exact)
        c = index_cache[key]
        if c is not None:
            ci_low[j] = walsh[c]
            ci_high[j] = walsh[walsh.size - 1 - c]
    return _TermStats(
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        statistic=statistic,
        p=p,
        q=bh_adjust(p),
    )


# --------------------------------------------------------------------------- #
# Internal: unit of analysis + weights
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _UnitPlan:
    """How ``unit=`` resolved: codes per row, counts, and the fixed-effects decision."""

    column: str
    codes: np.ndarray  # (n_samples,) unit index per row, first-appearance order
    n_units: int
    n_informative: int
    within: bool  # True -> the contrast varies within units (paired / repeated)


def _resolve_unit(
    metadata: pd.DataFrame,
    unit: str,
    contrast: str,
    covariates: tuple[str, ...],
    categorical_set: frozenset[str],
) -> _UnitPlan:
    """Apply the unit-of-analysis decision table (module docstring); raise on case 2."""
    if unit not in metadata.columns:
        raise ValueError(f"unit column {unit!r} not in metadata.")
    if unit == contrast or unit in covariates:
        raise ValueError(
            f"unit column {unit!r} cannot also be the contrast or a covariate."
        )
    series = metadata[unit]
    if bool(series.isna().any()):
        raise ValueError(
            f"unit column {unit!r} is missing for {int(series.isna().sum())} row(s); "
            f"every row must name its unit."
        )
    # Group on the raw values (1 and "1" are different units), not their text.
    codes_raw, uniques = pd.factorize(series, sort=False)
    codes = np.asarray(codes_raw, dtype=np.intp)
    n_units = len(uniques)
    counts = np.bincount(codes, minlength=n_units)
    if not bool((counts > 1).any()):
        return _UnitPlan(unit, codes, n_units, n_informative=0, within=False)

    contrast_series = metadata[contrast]
    is_categorical = contrast in categorical_set or not _is_numeric(contrast_series)
    level = (
        contrast_series.astype(str).to_numpy()
        if is_categorical
        else contrast_series.to_numpy(dtype=float).astype(str)
    )
    cells = pd.DataFrame({"unit": codes, "level": level})
    cell_size = cells.groupby(["unit", "level"], sort=False)["unit"].transform("size")
    crowded = cell_size.to_numpy() > 1
    if bool(crowded.any()):
        crowded_units = sorted({str(uniques[c]) for c in codes[crowded]})[:5]
        varying_cov: list[str] = []
        if covariates:
            crowded_meta = metadata.loc[crowded, list(covariates)].reset_index(
                drop=True
            )
            keys = [
                cells["unit"].to_numpy()[crowded],
                cells["level"].to_numpy()[crowded],
            ]
            per_cell = crowded_meta.groupby(keys).nunique(dropna=False)
            varying_cov = [c for c in covariates if bool((per_cell[c] > 1).any())]
        if varying_cov:
            raise ValueError(
                f"unit {unit!r}: some units have several rows at the same {contrast!r} "
                f"level that differ on {varying_cov} (e.g. units {crowded_units}). "
                f"Either (a) they are technical replicates spanning {varying_cov} — "
                f"average them with aggregate_replicates(dataset, by={unit!r}, "
                f"summarize={tuple(varying_cov)!r}, ...) and adjust for the summarized "
                f"columns — or (b) they are repeated measures over {varying_cov} "
                f"with a between-unit contrast, which needs a mixed model (a random "
                f"unit effect, e.g. limma duplicateCorrelation) that this template "
                f"does not fit: analyze one level of {varying_cov} at a time, or "
                f"average over it if the scientific question allows."
            )
        raise ValueError(
            f"unit {unit!r}: some units have several rows at the same {contrast!r} "
            f"level (e.g. units {crowded_units}); testing them as separate samples is "
            f"pseudoreplication. If they are technical replicates, average them first "
            f"with aggregate_replicates(dataset, by={unit!r}, ...) and test the "
            f"aggregated Dataset (with weights='precision_weight' when run counts "
            f"differ). If instead they are repeated measures — different timepoints or "
            f"conditions of the same unit, recorded in a column not passed as a "
            f"covariate or not in the metadata at all — this is a between-unit "
            f"contrast with repeated measures, which needs a mixed model this template "
            f"does not fit; do not average it away without the scientist's say-so."
        )
    n_levels = cells.groupby("unit", sort=False)["level"].nunique().to_numpy()
    n_informative = int((n_levels > 1).sum())
    if n_informative < 2:
        raise ValueError(
            f"unit {unit!r}: the contrast {contrast!r} varies within only "
            f"{n_informative} unit(s); a within-unit comparison needs at least 2."
        )
    return _UnitPlan(unit, codes, n_units, n_informative=n_informative, within=True)


def _resolve_weights(metadata: pd.DataFrame, weights: str) -> np.ndarray:
    if weights not in metadata.columns:
        raise ValueError(f"weights column {weights!r} not in metadata.")
    series = metadata[weights]
    if not _is_numeric(series):
        raise ValueError(f"weights column {weights!r} must be numeric.")
    values = series.to_numpy(dtype=float)
    if not bool(np.all(np.isfinite(values) & (values > 0))):
        raise ValueError(f"weights column {weights!r} must be finite and positive.")
    return np.asarray(values, dtype=float)


def _warn_if_unweighted_replicates(metadata: pd.DataFrame) -> None:
    """Warn when an aggregated Dataset with *unequal* run counts is fit unweighted.

    Called only for ``ols``/``moderated`` — the only methods that can take weights
    (the rank and two-group tests cannot, so a warning there could not be resolved).
    """
    if _N_REPLICATES not in metadata.columns:
        return
    if len(set(metadata[_N_REPLICATES].tolist())) > 1:
        warnings.warn(
            f"The Dataset is aggregated over unequal run counts ({_N_REPLICATES!r} "
            f"varies) but no weights= was given: units averaged over more runs are "
            f"more precise. Pass weights={_PRECISION_WEIGHT!r}, or confirm equal "
            f"weighting is intended.",
            UnweightedReplicatesWarning,
            stacklevel=3,
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
    unit: str | None = None,
    weights: str | None = None,
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
        covariates with ``welch``/``mannwhitney``/``signed_rank`` is an error.
    reference:
        Optional per-column reference level ``{column: level}``. Default is the
        sorted-first level. The reference controls the effect sign and is recorded.
    method:
        ``"moderated"`` (default), ``"ols"``, ``"welch"``, ``"mannwhitney"``, or
        ``"signed_rank"`` (paired; requires ``unit=``).
    categorical:
        Metadata columns to force-treat as factors even though they are numeric (the
        ``batch = 1/2/3`` case). Numeric columns not listed here are continuous slopes
        (with a low-cardinality warning).
    alpha:
        Two-sided significance level for the confidence intervals (default ``0.05`` →
        95% CI). Does not affect the BH-q values.
    unit:
        The metadata column naming the independent biological unit. Applies the
        unit-of-analysis decision table (module docstring): refuses technical
        replicates and between-unit repeated measures, and fits unit fixed effects
        when the contrast varies within units. Always pass it when the study has a
        unit column — it is what makes pseudoreplication a hard error.
    weights:
        A numeric metadata column of finite positive per-row precision weights
        (``precision_weight`` from ``aggregate_replicates``) → weighted least squares
        for ``ols``/``moderated``. Rejected by the rank and two-group tests.

    Returns
    -------
    DifferentialAbundanceResult
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; expected {'/'.join(_METHODS)}.")
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
        hints = [
            f"{c!r} was summarized into {frac}"
            for c in missing_cov
            if (
                frac := [
                    str(m) for m in metadata.columns if str(m).startswith(f"{c}[frac ")
                ]
            )
        ]
        hint = (
            f" ({'; '.join(hints)} — pass those columns, e.g. "
            f"aggregate_replicates(...).summarize_columns[...])"
            if hints
            else ""
        )
        raise ValueError(f"covariate column(s) not in metadata: {missing_cov}.{hint}")

    _check_scale(dataset.scale)
    effect_label = _effect_label(dataset.scale, method)
    mean_abundance_all = abundances.mean(axis=0)

    plan = (
        _resolve_unit(metadata, unit, contrast, covariates, categorical_set)
        if unit is not None
        else None
    )
    if method == "signed_rank" and (plan is None or not plan.within):
        raise ValueError(
            "method='signed_rank' is a paired test: pass unit= naming the unit whose "
            "rows are paired, with the contrast varying within units."
        )
    if method in ("welch", "mannwhitney") and plan is not None and plan.within:
        raise ValueError(
            f"method={method!r} treats rows as independent, but units repeat across "
            f"{contrast!r} levels (a paired / within-unit design). Use "
            f"method='moderated' or 'ols' with unit= (unit fixed effects; 'ols' is the "
            f"paired t-test), or method='signed_rank'."
        )
    weight_values: np.ndarray | None = None
    if weights is not None:
        if method not in ("ols", "moderated"):
            raise ValueError(
                f"weights= applies to the linear model (ols/moderated), not "
                f"method={method!r}."
            )
        weight_values = _resolve_weights(metadata, weights)
    elif method in ("ols", "moderated"):
        _warn_if_unweighted_replicates(metadata)

    if method == "signed_rank":
        assert plan is not None  # narrowed above
        return _run_signed_rank(
            contrast=contrast,
            covariates=covariates,
            categorical_set=categorical_set,
            reference=reference,
            metadata=metadata,
            abundances=abundances,
            feature_names=feature_names,
            effect_label=effect_label,
            alpha=alpha,
            plan=plan,
        )

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
        return _with_unit(result, plan)

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
        plan=plan,
        weights=weights,
        weight_values=weight_values,
    )


def _with_unit(
    result: DifferentialAbundanceResult, plan: _UnitPlan | None
) -> DifferentialAbundanceResult:
    """Stamp the unit bookkeeping onto a result (no-op without ``unit=``)."""
    if plan is None:
        return result
    return replace(
        result,
        unit=plan.column,
        n_units=plan.n_units,
        n_informative_units=plan.n_informative,
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
    plan: _UnitPlan | None,
    weights: str | None,
    weight_values: np.ndarray | None,
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
    columns = [intercept, *(t.data for t in terms)]
    fixed_effects = plan is not None and plan.within
    if fixed_effects:
        assert plan is not None
        _check_not_absorbed(terms, plan)
        # k-1 unit dummies (first-appearance unit is the reference): estimated, never
        # tested, never reported — each unit becomes its own control.
        columns.extend((plan.codes == u).astype(float) for u in range(1, plan.n_units))
    design = np.column_stack(columns)
    if (
        plan is not None
        and fixed_effects
        and int(np.linalg.matrix_rank(design)) < design.shape[1]
    ):
        raise ValueError(
            f"With unit fixed effects ({plan.column!r}) the design is "
            f"rank-deficient: some contrast level or covariate is not estimable from "
            f"within-unit differences (e.g. a level observed only in units that hold "
            f"no other level, or a covariate that is a unit-level property). Drop that "
            f"level / covariate, or test it between units on aggregated data."
        )

    if weight_values is not None:
        root_w = np.sqrt(weight_values)
        fit = _fit_ols(design * root_w[:, None], abundances * root_w[:, None])
    else:
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
        unit=plan.column if plan is not None else None,
        n_units=plan.n_units if plan is not None else None,
        n_informative_units=plan.n_informative if plan is not None else None,
        unit_fixed_effects=fixed_effects,
        weights=weights,
        df_residual=float(fit.df),
    )


def _check_not_absorbed(terms: list[_Term], plan: _UnitPlan) -> None:
    """Raise on a design column that is constant within every unit.

    Such a column is a function of the unit, so the unit fixed effects absorb it
    exactly (a singular design): a unit-level covariate (sex of the animal) is already
    adjusted for, and a contrast level that never varies within a unit is not
    estimable from within-unit differences.
    """
    for term in terms:
        frame = pd.DataFrame({"unit": plan.codes, "value": term.data})
        if int(frame.groupby("unit")["value"].nunique().max()) <= 1:
            if term.is_contrast:
                raise ValueError(
                    f"contrast term {term.name!r} never varies within a unit "
                    f"{plan.column!r}, so it is not estimable from within-unit "
                    f"differences. Drop that level or test it between units on "
                    f"aggregated data."
                )
            raise ValueError(
                f"covariate term {term.name!r} is constant within every unit "
                f"{plan.column!r}: it is a unit-level property, already absorbed by "
                f"the unit fixed effects. Drop it from covariates."
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


def _run_signed_rank(
    *,
    contrast: str,
    covariates: tuple[str, ...],
    categorical_set: frozenset[str],
    reference: dict[str, str],
    metadata: pd.DataFrame,
    abundances: np.ndarray,
    feature_names: np.ndarray,
    effect_label: str,
    alpha: float,
    plan: _UnitPlan,
) -> DifferentialAbundanceResult:
    """Wilcoxon signed-rank per term (level vs reference) over within-unit pairs.

    The decision table has already guaranteed at most one row per (unit, level), so a
    unit holding both compared levels contributes exactly one paired difference. Units
    holding only one of the two drop out of that term (``UnpairedUnitWarning``).
    """
    if covariates:
        raise ValueError(
            f"method='signed_rank' cannot adjust for covariates {list(covariates)}; "
            f"use method='moderated' with unit= to adjust, or drop the covariates."
        )
    series = metadata[contrast]
    if contrast not in categorical_set and _is_numeric(series):
        raise ValueError(
            f"method='signed_rank' needs a categorical contrast; {contrast!r} is "
            f"numeric. Use method='ols'/'moderated' with unit= for a within-unit "
            f"slope, or pass categorical=[{contrast!r}] if it is a coded factor."
        )
    labels = series.astype(str).to_numpy()
    levels = sorted(set(labels.tolist()))
    ref = reference.get(contrast, levels[0])
    if ref not in levels:
        raise ValueError(
            f"reference level {ref!r} for {contrast!r} not present; levels {levels}."
        )
    row_of = {
        (int(u), str(lbl)): i
        for i, (u, lbl) in enumerate(zip(plan.codes, labels, strict=True))
    }
    frames: list[pd.DataFrame] = []
    contrast_terms: list[str] = []
    for level in levels:
        if level == ref:
            continue
        pairs = [
            (row_of[(u, level)], row_of[(u, ref)])
            for u in range(plan.n_units)
            if (u, level) in row_of and (u, ref) in row_of
        ]
        term_name = f"{contrast}[{level} vs {ref}]"
        unpaired = sum(
            1
            for u in range(plan.n_units)
            if ((u, level) in row_of) != ((u, ref) in row_of)
        )
        if unpaired:
            warnings.warn(
                f"{term_name}: {unpaired} unit(s) hold only one of the two levels and "
                f"are excluded from this paired term ({len(pairs)} pairs remain).",
                UnpairedUnitWarning,
                stacklevel=3,
            )
        if not pairs:
            raise ValueError(
                f"{term_name}: no unit holds both levels; nothing to pair."
            )
        if 2.0 / 2.0 ** len(pairs) > alpha:
            warnings.warn(
                f"{term_name}: with {len(pairs)} pairs the smallest attainable "
                f"two-sided signed-rank p is {2.0 / 2.0 ** len(pairs):.3g} > "
                f"alpha={alpha}; no feature can reach significance.",
                LowPowerRankTestWarning,
                stacklevel=3,
            )
        lvl_rows = np.array([a for a, _ in pairs], dtype=np.intp)
        ref_rows = np.array([b for _, b in pairs], dtype=np.intp)
        stats_ = _signed_rank_stats(abundances[lvl_rows] - abundances[ref_rows], alpha)
        used = np.concatenate([lvl_rows, ref_rows])
        contrast_terms.append(term_name)
        frames.append(
            _term_rows(
                term_name=term_name,
                is_contrast=True,
                stats_=stats_,
                feature_names=feature_names,
                mean_abundance=abundances[used].mean(axis=0),
                n=int(used.size),
            )
        )

    table = _assemble_table(pd.concat(frames, ignore_index=True).to_dict("records"))
    return DifferentialAbundanceResult(
        table=table,
        contrast=contrast,
        covariates=(),
        method="signed_rank",
        reference={contrast: ref},
        contrast_terms=tuple(contrast_terms),
        effect_label=effect_label,
        n_samples=abundances.shape[0],
        prior_variance=None,
        prior_df=None,
        unit=plan.column,
        n_units=plan.n_units,
        n_informative_units=plan.n_informative,
    )
