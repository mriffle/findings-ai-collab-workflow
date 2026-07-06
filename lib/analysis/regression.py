"""Multivariable regression for feature finding — leakage-safe elastic-net linear.

TEMPLATE (lib/) — a *seed* for a project's regression script, not a finished analysis.
Copy it into the project's ``scripts/`` and adapt the call site per study (which
metadata column is the continuous outcome, the independent unit, whether to restrict
to a prior feature list). Held to the correctness charter (conventions/correctness.md)
and the statistics convention (conventions/statistics.md): assume nothing, verify
everything, fail loud.

**What this answers.** *Can the proteome predict a continuous outcome, and how well?*
— the continuous-outcome twin of the elastic-net logistic classifier
(``analysis.classification``). Its coefficients are reported as a caveated
*minimal-optimal* interpretation of the predictor, **not** an all-relevant feature
selection (that is Boruta's job; the two complement each other — FEATURE_FINDING.md
§B). Elastic net tuned for prediction yields the smallest sufficient predictive basis;
with correlated features L1 keeps one of a cluster and zeros its neighbours, so a
**low selection frequency does not mean a feature is unimportant** — it may be
redundant with a selected neighbour. State that in the finding.

**The three coupled deliverables** (all in :class:`RegressionResult`), the *same shape
as the classifier* (the user's steer — stay consistent with how the classifier does CV
and reports features/coefficients):

  * **Performance vs a target-shuffle null** — leakage-safe **nested CV** (tune
    ``(alpha, l1_ratio)`` in inner folds, estimate on outer folds), in-fold
    ``StandardScaler``. The null is **opt-in** (``run_null=True``): the gate that
    licenses trusting the coefficients; it maps to the exploratory/validated
    distinction — run it to be eligible for ``validated``, skip it and the finding is
    capped at ``exploratory`` (coefficients flagged "not tested against a null").
  * **All-data coefficients** — tune on all data, refit on all data; the reported
    **standardized** signed coefficients (magnitude = importance, sign = direction).
  * **Cross-fold stability** — a dedicated fixed-hyperparameter resampling loop giving
    each feature a **selection frequency**, **sign consistency**, and coefficient
    spread.

**Generalization target & grouping.** Choose the CV scheme to answer a stated
question: name the target (``"samples"`` / ``"individuals"`` / ``"batches"``) and hold
out folds at that unit. Grouping is used **only when the ``groups`` column actually
has repeats** — if every unit appears once, a new sample *is* a new individual and
row-level folds already estimate individual-level performance, so nothing is grouped.
When grouped, the null permutes the target at the group level.

**Prior feature list (optional).** ``feature_list=`` restricts to a curated /
hypothesis-driven set of features (e.g. radiation-responsive proteins) before fitting
— cutting dimensionality (proteomics is ``p >> n``) can sharpen a weak signal and aids
interpretability. **Leakage caveat:** the list must be defined **independent of this
outcome** (a list derived from this same data's target is circular; a published/domain
list is fine). Matched/unmatched counts are recorded in the result for provenance.

Scale / missing / sample set: the input should be the **experimental subset** on a
**log2-like** scale with **missing values already resolved upstream** (Stage-2). This
template **never silently imputes or zeros** — it **raises** on any ``NaN`` in the
abundances — warns on a non-log scale, and drops constant / all-zero features (they
carry no signal). The outcome must be **numeric** (a categorical outcome is a
classification task — use ``analysis.classification``); samples with a missing
(non-finite) outcome are dropped. Covariate confounding is **not** handled here: it is
a Stage-1 caveat surfaced collaboratively in Stage 4 (conventions/statistics.md),
gated by consequence.

Requires scikit-learn.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from common.data_loading import Dataset
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    RepeatedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GeneralizationTarget = Literal["samples", "individuals", "batches"]
Selection = Literal["best", "smoothed"]

# Scales on which the input is a genuine log abundance. StandardScaler tolerates the
# scale, but linear-scale abundances are right-skewed and leave outlier z-scores the
# model is sensitive to; outside this set the regressor warns (see _check_scale).
_LOG2_LIKE: frozenset[str] = frozenset({"log2", "glog2", "log10", "ln", "zscore"})

# Default hyperparameter grid: alpha (regularization strength) log-spaced from weak to
# strong; l1_ratio spanning the ridge-like grouping regime (0.1) to fully sparse (1.0).
# Exposed as arguments; these are the defaults (mirroring the source oracle's grid).
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
DEFAULT_L1_RATIOS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

# |coef| at or below this counts as "selected out" (numerically zero).
_ZERO_COEF = 1e-10

# Below this fraction of the requested feature list matching the data, warn.
_FEATURE_LIST_WARN_FRACTION = 0.5

__script_meta__: dict[str, object] = {
    "template": {"name": "regression", "version": "0.1"},
    "kind": "analysis",
    "provides": [
        "GeneralizationTarget",
        "Selection",
        "RegressionScaleWarning",
        "SingletonGroupsWarning",
        "FeatureListWarning",
        "FoldPrediction",
        "RegressionResult",
        "regress",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Leakage-safe elastic-net linear REGRESSION over a Dataset: nested CV "
        "performance (R2/RMSE/MAE, in-fold StandardScaler) vs an opt-in "
        "target-shuffle null (the exploratory/validated gate), all-data "
        "standardized coefficients, and a fixed-hyperparameter stability loop "
        "(selection frequency + sign consistency) — reported together, the same "
        "shape as the classifier. The continuous-outcome twin of "
        "analysis.classification; coefficients are a caveated minimal-optimal "
        "interpretation (a low selection frequency != unimportant), not an "
        "all-relevant selection (that is Boruta). Numeric outcome used directly (a "
        "categorical outcome is classification); group-aware CV only when the groups "
        "column has repeats (else row-level = individual-level; group-level null "
        "permutation); optional prior-knowledge feature_list restriction "
        "(leakage-safe, provenance-recorded); four result figures. Warns on non-log "
        "scale, raises on NaN (missing handling upstream), drops constant features. "
        "Requires scikit-learn."
    ),
}


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #
class RegressionScaleWarning(UserWarning):
    """The abundances are not on a log-like scale (standardization leaves skew)."""


class SingletonGroupsWarning(UserWarning):
    """The ``groups`` column has no repeated units, so grouping is a no-op.

    Every unit appears once, so a held-out sample is already a held-out unit; row-level
    CV is used (grouping singletons only degrades fold structure).
    """


class FeatureListWarning(UserWarning):
    """The ``feature_list`` matched few (or a small fraction) of the data's features."""


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FoldPrediction:
    """Held-out predictions from one outer CV fold (feeds the scatter plot)."""

    y_true: np.ndarray
    y_pred: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class RegressionResult:
    """Everything the four figures and the finding read.

    Attributes
    ----------
    coefficients:
        Per-feature table, one row per feature that is **non-zero in the final
        model** (the selected set), sorted by ``abs_coef`` descending. Columns:
        ``feature``, ``coef`` (all-data standardized), ``abs_coef``,
        ``selection_frequency`` (fraction of stability resamples non-zero),
        ``sign_consistency`` (fraction of selecting resamples agreeing on sign),
        ``coef_median``, ``coef_q25``, ``coef_q75`` (over the non-zero resamples),
        ``n_resamples``. The trust annotation on estimates.
    fold_predictions:
        Held-out ``(y_true, y_pred, test_indices)`` per outer nested-CV fold — the
        predicted-vs-observed scatter input.
    cv_r2, cv_r2_sd, cv_rmse, cv_mae:
        Nested-CV performance (mean over outer folds; ``cv_r2_sd`` = the fold SD).
    best_alpha, best_l1_ratio:
        The all-data-tuned hyperparameters (also used for the stability loop and null).
    grid_scores, alpha_grid, l1_grid:
        The all-data tuning surface — ``grid_scores`` is
        ``(len(alpha_grid), len(l1_grid))`` mean inner-CV score; the hyperparameter
        heatmap reads these.
    null_r2s, observed_r2, null_p:
        The target-shuffle null (fixed-hyperparameter procedure): the permutation
        R² distribution, the observed R² by the *same* procedure, and the empirical
        p ``(#{null >= observed} + 1) / (n_perm + 1)``. All ``None`` when
        ``run_null`` was ``False`` — the finding is then capped at ``exploratory``.
    validated_eligible:
        ``True`` iff the null was run (the coefficient report is licensed).
    outcome:
        The metadata column regressed.
    generalization_target, grouped, groups_column:
        The CV design: the target claimed, whether folds were grouped (only when the
        ``groups`` column had repeats), and that column's name.
    n_samples, n_features, n_dropped_constant, n_dropped_missing_target:
        Analyzed counts (for ``provenance.params``); ``n_dropped_missing_target`` is
        samples excluded for a non-finite outcome.
    n_features_requested, n_features_matched:
        When a ``feature_list`` was supplied: its unique size and how many matched
        the data (both ``None`` otherwise) — for provenance.
    random_state:
        The recorded seed.
    """

    coefficients: pd.DataFrame
    fold_predictions: list[FoldPrediction]
    cv_r2: float
    cv_r2_sd: float
    cv_rmse: float
    cv_mae: float
    best_alpha: float
    best_l1_ratio: float
    grid_scores: np.ndarray
    alpha_grid: tuple[float, ...]
    l1_grid: tuple[float, ...]
    outcome: str
    generalization_target: GeneralizationTarget
    grouped: bool
    groups_column: str | None
    n_samples: int
    n_features: int
    n_dropped_constant: int
    n_dropped_missing_target: int
    random_state: int
    null_r2s: np.ndarray | None = None
    observed_r2: float | None = None
    null_p: float | None = None
    n_features_requested: int | None = None
    n_features_matched: int | None = None
    feature_names: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    @property
    def validated_eligible(self) -> bool:
        """``True`` iff the target-shuffle null was run (licenses ``validated``)."""
        return self.null_p is not None


# --------------------------------------------------------------------------- #
# Internal: outcome resolution (numeric column -> y + keep mask)
# --------------------------------------------------------------------------- #
def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series))


def _resolve_target(series: pd.Series, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a numeric outcome column to (y_full, keep) — non-finite -> dropped."""
    if not _is_numeric(series):
        raise ValueError(
            f"regression needs a numeric outcome; {outcome!r} is categorical. Use "
            f"analysis.classification for a class outcome (or parse the column to "
            f"numbers upstream, e.g. '14 days' -> 14)."
        )
    y_full = series.to_numpy(dtype=float)
    keep = np.isfinite(y_full)
    return y_full, keep


# --------------------------------------------------------------------------- #
# Internal: feature-list restriction (optional prior knowledge)
# --------------------------------------------------------------------------- #
def _resolve_feature_list(
    feature_names: np.ndarray, feature_list: Sequence[str] | None
) -> tuple[np.ndarray, int | None, int | None]:
    """Return (keep-mask over features, n_requested, n_matched).

    An all-True mask is returned when no list is given (with ``None`` counts). Raises
    when a list matches nothing; warns on a poor match.
    """
    if feature_list is None:
        return np.ones(len(feature_names), dtype=bool), None, None
    requested = {str(f) for f in feature_list}
    n_requested = len(requested)
    if n_requested == 0:
        raise ValueError("feature_list is empty; pass None to use all features.")
    mask = np.isin(feature_names.astype(str), list(requested))
    n_matched = int(mask.sum())
    if n_matched == 0:
        raise ValueError(
            f"feature_list matched none of the {len(feature_names)} data features "
            f"(is the id scheme the same, e.g. UniProt accessions?)."
        )
    if n_matched < _FEATURE_LIST_WARN_FRACTION * n_requested:
        warnings.warn(
            f"feature_list matched only {n_matched}/{n_requested} data features — "
            f"check the id scheme matches. Proceeding with the matched subset.",
            FeatureListWarning,
            stacklevel=3,
        )
    return mask, n_requested, n_matched


# --------------------------------------------------------------------------- #
# Internal: estimator + CV construction
# --------------------------------------------------------------------------- #
def _build_pipeline(
    alpha: float, l1_ratio: float, max_iter: int, tol: float, random_state: int
) -> Pipeline:
    """StandardScaler + elastic-net linear regression (in-fold scaling, no leakage).

    ``selection="random"`` (coordinate descent over a random feature order) speeds
    convergence on wide matrices, seeded for reproducibility.
    """
    en = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        selection="random",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("en", en)])


def _resolve_grouping(
    groups: str | None,
    metadata: pd.DataFrame,
    keep: np.ndarray,
) -> np.ndarray | None:
    """Return per-sample group ids (kept samples) if grouping applies, else ``None``.

    Grouping applies only when the ``groups`` column has at least one repeated unit; all
    singletons -> ``None`` (row-level CV) with a :class:`SingletonGroupsWarning`.
    """
    if groups is None:
        return None
    if groups not in metadata.columns:
        raise ValueError(f"groups column {groups!r} not in metadata.")
    g = np.asarray(metadata[groups].astype(str).to_numpy())[keep]
    _, counts = np.unique(g, return_counts=True)
    if int(counts.max(initial=0)) <= 1:
        warnings.warn(
            f"groups column {groups!r} has no repeated units (each appears once), so a "
            f"held-out sample is already a held-out unit; using row-level CV.",
            SingletonGroupsWarning,
            stacklevel=3,
        )
        return None
    return np.asarray(g)


class _RepeatedGroupKFold:
    """Repeat GroupKFold with a reseeded shuffle each repeat.

    sklearn ships ``GroupKFold`` but (before 1.4) not a shuffled/repeated variant; we
    shuffle group order per repeat with a per-repeat seed so the stability loop and null
    get many grouped resamples that keep each unit intact within a fold.
    """

    def __init__(self, n_splits: int, n_repeats: int, random_state: int) -> None:
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(
        self, x: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for repeat in range(self.n_repeats):
            rng = np.random.default_rng(self.random_state + repeat)
            # Relabel groups by a shuffled order so folds differ across repeats.
            units = np.unique(groups)
            perm = rng.permutation(len(units))
            remap = {u: int(perm[i]) for i, u in enumerate(units)}
            shuffled = np.array([remap[g] for g in groups])
            cv = GroupKFold(n_splits=self.n_splits)
            out.extend(cv.split(x, y, shuffled))
        return out


def _make_cv(
    n_splits: int,
    n_repeats: int,
    grouped: bool,
    random_state: int,
) -> RepeatedKFold | _RepeatedGroupKFold:
    """Repeated K-fold — group-aware when grouped (keeps units intact)."""
    if grouped:
        return _RepeatedGroupKFold(n_splits, n_repeats, random_state)
    return RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )


def _split(
    cv: RepeatedKFold | _RepeatedGroupKFold,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Uniform split() over the grouped/ungrouped CV objects."""
    if isinstance(cv, _RepeatedGroupKFold):
        assert groups is not None
        return cv.split(x, y, groups)
    return list(cv.split(x, y))


# --------------------------------------------------------------------------- #
# Internal: hyperparameter selection (best cell, or neighborhood-smoothed)
# --------------------------------------------------------------------------- #
def _neighborhood_smooth(scores: np.ndarray) -> np.ndarray:
    """Average each grid cell with its axis-aligned neighbours (von Neumann).

    Plateau-seeking: avoids latching onto an isolated high-scoring cell that is likely
    noise. Edge cells average only the neighbours that exist.
    """
    padded = np.pad(scores, 1, mode="constant", constant_values=np.nan)
    stack = np.stack(
        [
            padded[1:-1, 1:-1],  # centre
            padded[:-2, 1:-1],  # up
            padded[2:, 1:-1],  # down
            padded[1:-1, :-2],  # left
            padded[1:-1, 2:],  # right
        ]
    )
    return np.asarray(np.nanmean(stack, axis=0), dtype=float)


def _select_cell(
    grid_scores: np.ndarray,
    alpha_grid: tuple[float, ...],
    l1_grid: tuple[float, ...],
    select: Selection,
) -> tuple[float, float]:
    surface = _neighborhood_smooth(grid_scores) if select == "smoothed" else grid_scores
    row, col = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    return alpha_grid[row], l1_grid[col]


# --------------------------------------------------------------------------- #
# Internal: the three CV uses
# --------------------------------------------------------------------------- #
def _nested_performance(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    cfg: _Config,
) -> tuple[list[FoldPrediction], float, float, float, float]:
    """Tune-in-fold repeated K-fold -> per-fold predictions + R²/RMSE/MAE."""
    outer = _make_cv(cfg.n_splits, cfg.n_repeats, groups is not None, cfg.random_state)
    param_grid = {"en__alpha": list(cfg.alpha_grid), "en__l1_ratio": list(cfg.l1_grid)}
    inner = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    folds: list[FoldPrediction] = []
    r2s: list[float] = []
    rmses: list[float] = []
    maes: list[float] = []
    for train, test in _split(outer, x, y, groups):
        search = GridSearchCV(
            _build_pipeline(
                cfg.alpha_grid[0],
                cfg.l1_grid[0],
                cfg.max_iter,
                cfg.tol,
                cfg.random_state,
            ),
            param_grid,
            cv=inner,
            scoring=cfg.tuning_metric,
            n_jobs=cfg.n_jobs,
        )
        search.fit(x[train], y[train])
        pred = np.asarray(search.predict(x[test]), dtype=float)
        folds.append(
            FoldPrediction(y_true=y[test].copy(), y_pred=pred, test_indices=test.copy())
        )
        r2s.append(float(r2_score(y[test], pred)))
        rmses.append(float(root_mean_squared_error(y[test], pred)))
        maes.append(float(mean_absolute_error(y[test], pred)))
    return (
        folds,
        float(np.mean(r2s)),
        float(np.std(r2s)),
        float(np.mean(rmses)),
        float(np.mean(maes)),
    )


def _all_data_fit(
    x: np.ndarray, y: np.ndarray, cfg: _Config
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Tune (alpha, l1_ratio) on all data + refit -> best_alpha, best_l1, coef, grid."""
    param_grid = {"en__alpha": list(cfg.alpha_grid), "en__l1_ratio": list(cfg.l1_grid)}
    inner = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    search = GridSearchCV(
        _build_pipeline(
            cfg.alpha_grid[0], cfg.l1_grid[0], cfg.max_iter, cfg.tol, cfg.random_state
        ),
        param_grid,
        cv=inner,
        scoring=cfg.tuning_metric,
        n_jobs=cfg.n_jobs,
    )
    search.fit(x, y)
    grid_scores = np.asarray(
        search.cv_results_["mean_test_score"], dtype=float
    ).reshape(len(cfg.alpha_grid), len(cfg.l1_grid))
    best_alpha, best_l1 = _select_cell(
        grid_scores, cfg.alpha_grid, cfg.l1_grid, cfg.select
    )
    final = _build_pipeline(
        best_alpha, best_l1, cfg.max_iter, cfg.tol, cfg.random_state
    )
    final.fit(x, y)
    en = final.named_steps["en"]
    coef = np.asarray(en.coef_, dtype=float).ravel()
    return best_alpha, best_l1, coef, grid_scores


def _stability(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_alpha: float,
    best_l1: float,
    cfg: _Config,
) -> np.ndarray:
    """Per-resample standardized coefficients at fixed hyperparameters."""
    cv = _make_cv(
        cfg.n_splits, cfg.stability_repeats, groups is not None, cfg.random_state
    )
    coefs: list[np.ndarray] = []
    for train, _ in _split(cv, x, y, groups):
        model = _build_pipeline(
            best_alpha, best_l1, cfg.max_iter, cfg.tol, cfg.random_state
        )
        model.fit(x[train], y[train])
        coefs.append(np.asarray(model.named_steps["en"].coef_, dtype=float).ravel())
    return np.asarray(coefs, dtype=float)


def _fixed_cv_r2(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_alpha: float,
    best_l1: float,
    cfg: _Config,
) -> float:
    """Mean R² from repeated K-fold at FIXED hyperparameters (observed + each null)."""
    cv = _make_cv(cfg.n_splits, cfg.null_repeats, groups is not None, cfg.random_state)
    model = _build_pipeline(
        best_alpha, best_l1, cfg.max_iter, cfg.tol, cfg.random_state
    )
    if groups is None:
        scores = cross_val_score(model, x, y, cv=cv, scoring="r2", n_jobs=cfg.n_jobs)
        return float(np.mean(scores))
    r2s: list[float] = []
    for train, test in _split(cv, x, y, groups):
        model.fit(x[train], y[train])
        pred = np.asarray(model.predict(x[test]), dtype=float)
        r2s.append(float(r2_score(y[test], pred)))
    return float(np.mean(r2s))


def _permute_target(
    y: np.ndarray, groups: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle the target — at group level when grouped (keeps a unit's value intact).

    For a grouped design the outcome is (usually) a per-unit property, so permuting the
    per-unit value across units — not the per-row value — is the honest null.
    """
    if groups is None:
        return np.asarray(rng.permutation(y), dtype=float)
    units, inverse = np.unique(groups, return_inverse=True)
    unit_value = np.array([float(y[groups == u].mean()) for u in units], dtype=float)
    shuffled = rng.permutation(unit_value)
    return np.asarray(shuffled[inverse], dtype=float)


def _null_distribution(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_alpha: float,
    best_l1: float,
    cfg: _Config,
) -> tuple[np.ndarray, float, float]:
    """Target-shuffle null (fixed hyperparameters) -> null_r2s, observed, p."""
    observed = _fixed_cv_r2(x, y, groups, best_alpha, best_l1, cfg)
    rng = np.random.default_rng(cfg.random_state)
    nulls = np.array(
        [
            _fixed_cv_r2(
                x, _permute_target(y, groups, rng), groups, best_alpha, best_l1, cfg
            )
            for _ in range(cfg.n_permutations)
        ],
        dtype=float,
    )
    p = float((np.sum(nulls >= observed) + 1) / (cfg.n_permutations + 1))
    return nulls, observed, p


# --------------------------------------------------------------------------- #
# Internal: coefficient table (identical shape to the classifier's)
# --------------------------------------------------------------------------- #
def _coefficient_table(
    final_coef: np.ndarray, resample_coef: np.ndarray, feature_names: np.ndarray
) -> pd.DataFrame:
    """Assemble the per-feature table for features non-zero in the final model."""
    selected = np.abs(final_coef) > _ZERO_COEF
    nonzero = np.abs(resample_coef) > _ZERO_COEF
    n_resample = resample_coef.shape[0]
    sel_freq = nonzero.mean(axis=0)
    with np.errstate(invalid="ignore"):
        sign_sum = np.abs(np.sum(np.sign(resample_coef), axis=0))
        n_nz = nonzero.sum(axis=0)
        sign_cons = np.where(n_nz > 0, sign_sum / np.maximum(n_nz, 1), 0.0)
    masked = np.where(nonzero, resample_coef, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=0)
        q25 = np.nanpercentile(masked, 25, axis=0)
        q75 = np.nanpercentile(masked, 75, axis=0)
    idx = np.where(selected)[0]
    table = pd.DataFrame(
        {
            "feature": feature_names[idx],
            "coef": final_coef[idx],
            "abs_coef": np.abs(final_coef[idx]),
            "selection_frequency": sel_freq[idx],
            "sign_consistency": sign_cons[idx],
            "coef_median": median[idx],
            "coef_q25": q25[idx],
            "coef_q75": q75[idx],
            "n_resamples": n_resample,
        }
    )
    return table.sort_values("abs_coef", ascending=False, kind="stable").reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- #
# Internal: resolved configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Config:
    alpha_grid: tuple[float, ...]
    l1_grid: tuple[float, ...]
    select: Selection
    tuning_metric: str
    n_splits: int
    n_repeats: int
    stability_repeats: int
    null_repeats: int
    n_permutations: int
    max_iter: int
    tol: float
    n_jobs: int
    random_state: int


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def regress(
    dataset: Dataset,
    outcome: str,
    *,
    groups: str | None = None,
    generalization_target: GeneralizationTarget = "samples",
    feature_list: Sequence[str] | None = None,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    l1_ratios: Sequence[float] = DEFAULT_L1_RATIOS,
    select: Selection = "best",
    tuning_metric: str = "r2",
    n_splits: int = 5,
    n_repeats: int = 5,
    stability_repeats: int = 10,
    run_null: bool = False,
    n_permutations: int = 1000,
    null_repeats: int = 3,
    max_iter: int = 10000,
    tol: float = 1e-4,
    n_jobs: int = -1,
    random_state: int = 0,
) -> RegressionResult:
    """Fit a leakage-safe elastic-net linear regressor; report the three parts.

    Parameters
    ----------
    dataset:
        The **experimental subset** on a **log2-like** scale, missing values resolved
        upstream (a ``NaN`` raises). Constant/all-zero features are dropped.
    outcome:
        The **numeric** metadata column to predict. A categorical column raises (that
        is a classification task — use ``analysis.classification``). Samples with a
        non-finite outcome are dropped and counted.
    groups:
        Metadata column naming the independent unit (subject/animal/batch). Used for
        group-aware CV **only if it has repeats**; all-singletons -> row-level CV.
    generalization_target:
        The performance question — ``"samples"``, ``"individuals"``, or ``"batches"``.
        Recorded; report performance as "on unseen <target>".
    feature_list:
        Optional curated / hypothesis-driven feature ids to restrict to before fitting
        (prior knowledge; cuts dimensionality, can sharpen a weak signal). **Must be
        defined independent of ``outcome``** (a list derived from this data's target is
        circular). Matched/unmatched counts are recorded.
    alpha_grid, l1_ratios:
        Elastic-net grid — ``alpha`` regularization strength, ``l1_ratio``
        0.1=ridge-like grouping .. 1=fully sparse.
    select:
        ``"best"`` (highest inner-CV score) or ``"smoothed"`` (neighborhood-smoothed,
        plateau-seeking) cell selection.
    tuning_metric:
        Inner-CV scoring for tuning (default ``"r2"``).
    n_splits, n_repeats:
        Outer nested-CV folds and repeats (the honest performance estimate).
    stability_repeats:
        Repeats of the fixed-hyperparameter stability loop (``n_splits`` folds each).
    run_null:
        If ``True``, run the **target-shuffle null** (the gate that licenses trusting
        the coefficients and enables a ``validated`` finding). Opt-in because it is the
        compute cost. If ``False``, ``null_*`` are ``None`` and the finding is capped at
        ``exploratory``.
    n_permutations, null_repeats:
        Number of target permutations, and the (lighter) fixed-hyperparameter CV repeats
        per permutation.
    max_iter, tol:
        ElasticNet coordinate-descent convergence controls.
    n_jobs:
        Parallelism for the inner grid search / fixed-CV scoring (``-1`` = all cores).
    random_state:
        Recorded seed for every stochastic step.

    Returns
    -------
    RegressionResult
    """
    alpha_grid_t = tuple(float(a) for a in alpha_grid)
    l1_grid_t = tuple(float(v) for v in l1_ratios)
    if not alpha_grid_t or not l1_grid_t:
        raise ValueError("alpha_grid and l1_ratios must be non-empty.")
    if any(v <= 0.0 for v in alpha_grid_t):
        raise ValueError(f"alpha_grid values must be positive; got {alpha_grid_t}.")
    if any(not 0.0 < v <= 1.0 for v in l1_grid_t):
        raise ValueError(f"l1_ratios must be in (0, 1]; got {l1_grid_t}.")

    metadata = dataset.metadata
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(f"abundances must be 2D; got shape {abundances.shape}.")
    n_samples, _ = abundances.shape
    if len(metadata) != n_samples:
        raise ValueError(
            f"metadata has {len(metadata)} rows but abundances has {n_samples} samples."
        )
    if outcome not in metadata.columns:
        raise ValueError(f"outcome column {outcome!r} not in metadata.")
    if not np.all(np.isfinite(abundances)):
        raise ValueError(
            "abundances contain NaN/inf. Missing-value handling is an upstream Stage-2 "
            "decision (conventions/statistics.md); this regressor does not silently "
            "impute. Resolve missingness before regression."
        )
    _check_scale(dataset.scale)

    y_full, keep = _resolve_target(metadata[outcome], outcome)
    y = y_full[keep]
    n_dropped_missing_target = int(n_samples - int(keep.sum()))
    x_kept = abundances[keep, :]

    feature_names = np.asarray(dataset.feature_names)
    feat_mask, n_requested, n_matched = _resolve_feature_list(
        feature_names, feature_list
    )
    x_listed = x_kept[:, feat_mask]
    listed_names = feature_names[feat_mask]

    non_constant = np.std(x_listed, axis=0) > 0.0
    n_dropped_constant = int((~non_constant).sum())
    x = x_listed[:, non_constant]
    kept_features = listed_names[non_constant]
    if x.shape[1] == 0:
        raise ValueError("No non-constant features remain after dropping constants.")

    n_kept = int(keep.sum())
    if n_kept < 2 * n_splits:
        raise ValueError(
            f"Need >= 2*n_splits ({2 * n_splits}) samples with a finite outcome; got "
            f"{n_kept}. Lower n_splits or check the outcome column."
        )

    groups_kept = _resolve_grouping(groups, metadata, keep)
    grouped = groups_kept is not None

    cfg = _Config(
        alpha_grid=alpha_grid_t,
        l1_grid=l1_grid_t,
        select=select,
        tuning_metric=tuning_metric,
        n_splits=n_splits,
        n_repeats=n_repeats,
        stability_repeats=stability_repeats,
        null_repeats=null_repeats,
        n_permutations=n_permutations,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    folds, cv_r2, cv_r2_sd, cv_rmse, cv_mae = _nested_performance(
        x, y, groups_kept, cfg
    )
    best_alpha, best_l1, final_coef, grid_scores = _all_data_fit(x, y, cfg)
    resample_coef = _stability(x, y, groups_kept, best_alpha, best_l1, cfg)
    coeff_table = _coefficient_table(final_coef, resample_coef, kept_features)

    null_r2s: np.ndarray | None = None
    observed_r2: float | None = None
    null_p: float | None = None
    if run_null:
        null_r2s, observed_r2, null_p = _null_distribution(
            x, y, groups_kept, best_alpha, best_l1, cfg
        )

    return RegressionResult(
        coefficients=coeff_table,
        fold_predictions=folds,
        cv_r2=cv_r2,
        cv_r2_sd=cv_r2_sd,
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
        best_alpha=best_alpha,
        best_l1_ratio=best_l1,
        grid_scores=grid_scores,
        alpha_grid=alpha_grid_t,
        l1_grid=l1_grid_t,
        outcome=outcome,
        generalization_target=generalization_target,
        grouped=grouped,
        groups_column=groups if grouped else None,
        n_samples=n_kept,
        n_features=int(x.shape[1]),
        n_dropped_constant=n_dropped_constant,
        n_dropped_missing_target=n_dropped_missing_target,
        random_state=random_state,
        null_r2s=null_r2s,
        observed_r2=observed_r2,
        null_p=null_p,
        n_features_requested=n_requested,
        n_features_matched=n_matched,
        feature_names=kept_features,
    )


def _check_scale(scale: str) -> None:
    if scale not in _LOG2_LIKE:
        warnings.warn(
            f"Regression on scale {scale!r}: in-fold standardization tolerates the "
            f"scale, but linear-scale abundances are right-skewed and leave outlier "
            f"z-scores the model is sensitive to. Run on log data unless you have a "
            f"specific reason not to.",
            RegressionScaleWarning,
            stacklevel=3,
        )
