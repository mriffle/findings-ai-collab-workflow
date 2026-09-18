"""Multivariable classification for feature finding — leakage-safe elastic-net logistic.

TEMPLATE (lib/) — a *seed* for a project's classification script, not a finished
analysis. Copy it into the project's ``scripts/`` and adapt the call site per study
(which metadata column is the outcome, how to reduce it to two classes, the independent
unit). Held to the correctness charter (conventions/correctness.md) and the statistics
convention (conventions/statistics.md): assume nothing, verify everything, fail loud.

**What this answers.** *Can the proteome predict the class, and how well?* — an
elastic-net logistic classifier whose coefficients are reported as a caveated
interpretation of the classifier, **not** an all-relevant feature selection (that is
Boruta's job; the two complement each other — FEATURE_FINDING.md §B). Elastic net tuned
for prediction yields the *minimal-optimal* set (the smallest sufficient predictive
basis); with correlated features L1 keeps one of a cluster and zeros its neighbours,
so a **low selection frequency does not mean a feature is unimportant** — it may be
redundant with a selected neighbour. State that in the finding.

**The three coupled deliverables** (all in :class:`ClassificationResult`):

  * **Performance vs a label-shuffle null** — leakage-safe **nested CV** (tune
    ``(C, l1_ratio)`` in inner folds, estimate on outer folds), in-fold
    ``StandardScaler``, ``class_weight="balanced"``. The null is **opt-in**
    (``run_null=True``): the gate that licenses trusting the coefficients; it maps to
    the exploratory/validated distinction — run it to be eligible for ``validated``,
    skip it and the finding is capped at ``exploratory`` (coefficients flagged "not
    tested against a null").
  * **All-data coefficients** — tune on all data, refit on all data; the reported
    **standardized** signed coefficients (magnitude = importance, sign = direction).
  * **Cross-fold stability** — a dedicated fixed-hyperparameter resampling loop giving
    each feature a **selection frequency**, **sign consistency**, and coefficient
    distribution.

**Generalization target & grouping.** Choose the CV scheme to answer a stated question:
name the target (``"samples"`` / ``"individuals"`` / ``"batches"``) and hold out
folds at that unit. Grouping is used **only when the ``groups`` column actually has
repeats** — if every unit appears once, a new sample *is* a new individual and
row-level folds already estimate individual-level performance, so nothing is grouped
(grouping singletons only hurts class balance). When grouped, the null permutes labels
at the unit level by default (``null_permutation="units"``, one label per unit) or
*within* each unit for a batch-grouped design whose units hold both classes
(``null_permutation="within_units"``, preserving per-unit class counts).

Scale / missing / sample set: the input should be the **experimental subset** on a
**log2-like** scale with **missing values already resolved upstream** (Stage-2). This
template **never silently imputes or zeros** — it **raises** on any ``NaN`` — warns on a
non-log scale, and drops constant / all-zero features (they carry no signal). Covariate
confounding is **not** handled here: it is a Stage-1 caveat surfaced collaboratively in
Stage 4 (conventions/statistics.md), gated by consequence.

Requires scikit-learn.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from common.data_loading import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GeneralizationTarget = Literal["samples", "individuals", "batches"]
Selection = Literal["best", "smoothed"]
NullPermutation = Literal["units", "within_units"]

# Scales on which the input is a genuine log abundance. StandardScaler tolerates the
# scale, but linear-scale abundances are right-skewed and leave outlier z-scores the
# model is sensitive to; outside this set the classifier warns (see _check_scale).
_LOG2_LIKE: frozenset[str] = frozenset({"log2", "glog2", "log10", "ln", "zscore"})

# Default hyperparameter grid: C log-spaced from strong to weak regularization; l1_ratio
# spanning the ridge-like grouping regime (0.25) to fully sparse (1.0). Exposed as
# arguments; these are the defaults (mirroring the source oracle's search space).
DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_L1_RATIOS: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

# |coef| at or below this counts as "selected out" (numerically zero).
_ZERO_COEF = 1e-10

__script_meta__: dict[str, object] = {
    "template": {"name": "classification", "version": "0.4"},
    "kind": "analysis",
    "provides": [
        "GeneralizationTarget",
        "Selection",
        "NullPermutation",
        "Threshold",
        "LevelMap",
        "BinarizeSpec",
        "ClassificationScaleWarning",
        "SingletonGroupsWarning",
        "NullPermutationWarning",
        "FeatureListWarning",
        "FoldPrediction",
        "ClassificationResult",
        "classify",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Leakage-safe elastic-net logistic CLASSIFICATION over a Dataset: nested CV "
        "performance vs an opt-in label-shuffle null (the exploratory/validated "
        "gate), all-data standardized coefficients, and a fixed-hyperparameter "
        "stability loop (selection frequency + sign consistency) — reported together. "
        "Study-agnostic outcome + binarize API (binary direct; a non-binary outcome "
        "needs an explicit rule); group-aware CV (outer and inner tuning folds alike) "
        "only when the groups column has repeats; the null permutes at the unit level "
        "or, for a batch-grouped design, within units (null_permutation); four result "
        "figures; fold identity (repeat/fold/test_indices) recorded on every fold for "
        "paired comparison across classifier templates. Warns on non-log scale, "
        "raises on NaN (missing handling upstream), drops constant features. Binary "
        "outcomes only (v0.1). v0.4: tuning table indexed by the recorded parameter "
        "values, one selection rule in every loop, a missing outcome/group value "
        "handled, tuning_metric + inner_cv_grouped recorded. Requires scikit-learn."
    ),
}


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #
class ClassificationScaleWarning(UserWarning):
    """The abundances are not on a log-like scale (standardization leaves skew)."""


class SingletonGroupsWarning(UserWarning):
    """The ``groups`` column has no repeated units, so grouping is a no-op.

    Every unit appears once, so a held-out sample is already a held-out unit; row-level
    CV is used (grouping singletons only degrades class balance).
    """


class NullPermutationWarning(UserWarning):
    """The within-unit null can reach fewer distinct label arrangements than requested.

    ``prod_u C(n_u, k_u)`` is below ``n_permutations``: permutation draws must repeat
    and the empirical p resolves only to about ``1 / arrangements``. The p stays valid
    (a Monte-Carlo p with duplicate draws is still a p); it is just coarse.
    """


class FeatureListWarning(UserWarning):
    """The ``feature_list`` matched few (or a small fraction) of the data's features."""


_FEATURE_LIST_WARN_FRACTION = 0.5


def _resolve_feature_list(
    feature_names: np.ndarray, feature_list: Sequence[str] | None
) -> tuple[np.ndarray, int | None, int | None]:
    """Return (keep-mask over features, n_requested, n_matched).

    An all-True mask is returned when no list is given (with ``None`` counts). Raises
    when a list matches nothing; warns on a poor match. Restricting to a prior /
    curated list is applied to the whole matrix **before** the CV — leakage-safe only
    because such a list is defined **independent of the outcome** (see ``classify``).
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
# Outcome binarization spec — the caller's rule reducing an outcome to two classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Threshold:
    """Binarize a **continuous** outcome at a cut: ``value >= cut`` -> positive class.

    Optionally drop an ambiguous middle band: with ``drop_below``/``drop_at_or_above``,
    samples with ``drop_below <= value < drop_at_or_above`` are excluded (the "compare
    the extremes, drop the middle" design). Unassigned samples are dropped and counted
    in provenance.
    """

    cut: float
    positive_label: str = "high"
    negative_label: str = "low"
    drop_below: float | None = None
    drop_at_or_above: float | None = None


@dataclass(frozen=True)
class LevelMap:
    """Binarize a **categorical** outcome by assigning levels to the two classes.

    Levels listed in neither ``positive`` nor ``negative`` are **dropped** (so this also
    expresses "compare these two of k levels, drop the rest"). Levels must not overlap.
    A missing value is unassigned: dropped and counted in ``n_dropped_unassigned``.
    """

    positive: tuple[str, ...]
    negative: tuple[str, ...]


BinarizeSpec = Threshold | LevelMap


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FoldPrediction:
    """Held-out predictions from one outer CV fold (feeds the ROC curve).

    ``repeat`` / ``fold`` locate the fold in the repeated outer CV and ``test_indices``
    are the held-out sample positions **in the analyzed sample set** (after the
    binarize drop mask), so a same-seed run of another classifier template can be
    compared fold-by-fold (a *paired* comparison). Defaulted for back-compatibility
    with results cached before v0.2 (they load with ``repeat == fold == -1`` and an
    empty ``test_indices``).
    """

    y_true: np.ndarray
    y_prob: np.ndarray
    repeat: int = -1
    fold: int = -1
    test_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


@dataclass(frozen=True)
class ClassificationResult:
    """Everything the four figures and the finding read.

    Attributes
    ----------
    coefficients:
        Per-feature table, one row per feature that is **non-zero in the final model**
        (the selected set), sorted by ``abs_coef`` descending. Columns: ``feature``,
        ``coef`` (all-data standardized), ``abs_coef``, ``selection_frequency``
        (fraction of stability resamples non-zero), ``sign_consistency`` (fraction of
        selecting resamples agreeing on sign), ``coef_median``, ``coef_q25``,
        ``coef_q75`` (over the non-zero resamples). The trust annotation on estimates.
    fold_predictions:
        Held-out ``(y_true, y_prob, repeat, fold, test_indices)`` per outer nested-CV
        fold — the ROC input and the paired-comparison record (v0.2).
    cv_auc, cv_auc_sd, cv_balanced_accuracy, cv_average_precision:
        Nested-CV performance (mean over outer folds; ``_sd`` is the fold SD of AUC).
    best_c, best_l1_ratio:
        The all-data-tuned hyperparameters (also used for the stability loop and null).
    grid_scores, c_grid, l1_grid:
        The all-data tuning surface — ``grid_scores`` is ``(len(c_grid), len(l1_grid))``
        mean inner-CV AUC; the hyperparameter heatmap reads these.
    null_aucs, observed_auc, null_p:
        The label-shuffle null (fixed-hyperparameter procedure): the permutation AUC
        distribution, the observed AUC computed by the *same* procedure, and the
        empirical p ``(#{null >= observed} + 1) / (n_perm + 1)``. All ``None`` when
        ``run_null`` was ``False`` — the finding is then capped at ``exploratory``.
    validated_eligible:
        ``True`` iff the null was run (the coefficient report is licensed).
    outcome, positive_label, negative_label:
        The resolved binary problem (``positive_label`` is class 1; coefficient sign is
        *toward positive*).
    generalization_target, grouped, groups_column:
        The CV design: the target claimed, whether folds were grouped (only when the
        ``groups`` column had repeats), and that column's name.
    n_samples, n_positive, n_negative, n_features, n_dropped_constant,
    n_dropped_unassigned:
        Analyzed counts (for ``provenance.params``); ``n_dropped_unassigned`` is samples
        excluded by the binarize rule.
    random_state:
        The recorded seed.
    n_features_requested, n_features_matched:
        When a ``feature_list`` was supplied: its unique size and how many matched the
        data's features (``None`` when no list was given). Recorded for provenance.
    null_permutation:
        The null's permutation scheme as applied: ``"samples"`` (row-level CV, plain
        shuffle), ``"units"``, or ``"within_units"``; ``None`` when the null was not run
        or on a result cached before v0.3.
    tuning_metric, inner_cv_grouped:
        The inner-CV scorer the tuning surface was scored with, and whether the inner
        tuning folds were grouped (they are whenever the outer folds are; v0.4 — a
        result cached earlier reloads with ``False``, which is what it was).
    """

    coefficients: pd.DataFrame
    fold_predictions: list[FoldPrediction]
    cv_auc: float
    cv_auc_sd: float
    cv_balanced_accuracy: float
    cv_average_precision: float
    best_c: float
    best_l1_ratio: float
    grid_scores: np.ndarray
    c_grid: tuple[float, ...]
    l1_grid: tuple[float, ...]
    outcome: str
    positive_label: str
    negative_label: str
    generalization_target: GeneralizationTarget
    grouped: bool
    groups_column: str | None
    n_samples: int
    n_positive: int
    n_negative: int
    n_features: int
    n_dropped_constant: int
    n_dropped_unassigned: int
    random_state: int
    null_aucs: np.ndarray | None = None
    observed_auc: float | None = None
    null_p: float | None = None
    n_features_requested: int | None = None
    n_features_matched: int | None = None
    null_permutation: str | None = None
    feature_names: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    tuning_metric: str = "roc_auc"
    inner_cv_grouped: bool = False

    @property
    def validated_eligible(self) -> bool:
        """``True`` iff the label-shuffle null was run (licenses ``validated``)."""
        return self.null_p is not None


# --------------------------------------------------------------------------- #
# Internal: label resolution (outcome -> binary y + drop mask)
# --------------------------------------------------------------------------- #
def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series))


@dataclass(frozen=True)
class _Labels:
    y: np.ndarray  # (n_kept,) int 0/1
    keep: np.ndarray  # (n_samples,) bool — samples assigned a class
    positive_label: str
    negative_label: str


def _labels_and_missing(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """String labels plus a missing mask, robust to pandas' NA handling.

    ``astype(str)`` leaves a missing value as a float ``nan`` (pandas >= 3 string
    dtype) or as the text ``"nan"`` / ``"<NA>"`` (older object columns), so a string
    comparison cannot detect it. The mask comes from ``isna()`` first; a missing row
    gets the empty label and is never compared as a level or a unit.
    """
    missing = np.asarray(series.isna().to_numpy(), dtype=bool)
    labels = np.where(missing, "", series.astype(str).to_numpy()).astype(str)
    return labels, missing


def _resolve_labels(
    series: pd.Series,
    binarize: BinarizeSpec | None,
    positive_class: str | None,
    outcome: str,
) -> _Labels:
    """Reduce an outcome to a binary label + a keep mask (unassigned -> dropped)."""
    if binarize is None:
        return _resolve_already_binary(series, positive_class, outcome)
    if isinstance(binarize, Threshold):
        return _resolve_threshold(series, binarize, outcome)
    return _resolve_level_map(series, binarize, outcome)


def _resolve_already_binary(
    series: pd.Series, positive_class: str | None, outcome: str
) -> _Labels:
    if _is_numeric(series):
        raise ValueError(
            f"outcome {outcome!r} is numeric; a classifier needs classes. Pass "
            f"binarize=Threshold(cut=...) to split it, or a LevelMap if it is a coded "
            f"factor."
        )
    labels, missing = _labels_and_missing(series)
    keep = ~missing
    levels = sorted(set(labels[keep].tolist()))
    if len(levels) != 2:
        raise ValueError(
            f"outcome {outcome!r} has {len(levels)} classes {levels}; v0.1 is binary "
            f"only. Pass binarize=LevelMap(positive=..., negative=...) to choose two "
            f"(others are dropped)."
        )
    if positive_class is not None and positive_class not in levels:
        raise ValueError(
            f"positive_class {positive_class!r} not among outcome levels {levels}."
        )
    pos = positive_class if positive_class is not None else levels[1]
    neg = next(level for level in levels if level != pos)
    y = np.where(labels == pos, 1, 0)[keep].astype(int)
    return _Labels(y=y, keep=keep, positive_label=pos, negative_label=neg)


def _resolve_threshold(series: pd.Series, spec: Threshold, outcome: str) -> _Labels:
    if not _is_numeric(series):
        raise ValueError(
            f"binarize=Threshold needs a numeric outcome; {outcome!r} is categorical. "
            f"Use a LevelMap."
        )
    values = series.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if (spec.drop_below is None) != (spec.drop_at_or_above is None):
        raise ValueError(
            "Threshold drop band needs both drop_below and drop_at_or_above, or "
            "neither."
        )
    in_band = np.zeros_like(finite)
    if spec.drop_below is not None and spec.drop_at_or_above is not None:
        if not spec.drop_below <= spec.cut <= spec.drop_at_or_above:
            raise ValueError(
                f"Threshold cut {spec.cut} must lie within the drop band "
                f"[{spec.drop_below}, {spec.drop_at_or_above}]."
            )
        in_band = (values >= spec.drop_below) & (values < spec.drop_at_or_above)
    keep = finite & ~in_band
    y = (values[keep] >= spec.cut).astype(int)
    if int(y.sum()) == 0 or int((y == 0).sum()) == 0:
        raise ValueError(
            f"Threshold(cut={spec.cut}) on {outcome!r} yields only one class; choose a "
            f"cut inside the value range."
        )
    return _Labels(
        y=y,
        keep=keep,
        positive_label=spec.positive_label,
        negative_label=spec.negative_label,
    )


def _resolve_level_map(series: pd.Series, spec: LevelMap, outcome: str) -> _Labels:
    labels, missing = _labels_and_missing(series)
    pos_set, neg_set = set(spec.positive), set(spec.negative)
    overlap = pos_set & neg_set
    if overlap:
        raise ValueError(f"LevelMap positive/negative overlap on {sorted(overlap)}.")
    present = set(labels[~missing].tolist())
    unknown = (pos_set | neg_set) - present
    if unknown:
        raise ValueError(
            f"LevelMap references levels {sorted(unknown)} absent from {outcome!r} "
            f"(present: {sorted(present)})."
        )
    is_pos = np.isin(labels, list(pos_set)) & ~missing
    is_neg = np.isin(labels, list(neg_set)) & ~missing
    keep = is_pos | is_neg
    y = is_pos[keep].astype(int)
    return _Labels(
        y=y,
        keep=keep,
        positive_label="|".join(sorted(pos_set)),
        negative_label="|".join(sorted(neg_set)),
    )


# --------------------------------------------------------------------------- #
# Internal: estimator + CV construction
# --------------------------------------------------------------------------- #
def _build_pipeline(
    c: float | None, l1_ratio: float, max_iter: int, tol: float, random_state: int
) -> Pipeline:
    """StandardScaler + elastic-net logistic regression (in-fold scaling, no leakage).

    sklearn (>=1.8) selects elastic net by ``l1_ratio`` alone (0=L2, 1=L1, in-between=
    elastic net) — ``penalty=`` is deprecated. ``class_weight="balanced"`` handles class
    imbalance. ``C=None`` leaves the estimator default until GridSearchCV sets it.
    """
    lr = LogisticRegression(
        solver="saga",
        l1_ratio=l1_ratio,
        C=1.0 if c is None else c,
        class_weight="balanced",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("lr", lr)])


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
    labels, missing = _labels_and_missing(metadata[groups])
    g = labels[keep]
    missing_kept = np.flatnonzero(missing[keep])
    if missing_kept.size:
        raise ValueError(
            f"groups column {groups!r} has a missing value for {missing_kept.size} "
            f"analyzed sample(s) (positions {missing_kept[:10].tolist()}): a sample "
            f"without a unit cannot be assigned to a fold. Fill or drop it upstream."
        )
    _, counts = np.unique(g, return_counts=True)
    if int(counts.max(initial=0)) <= 1:
        warnings.warn(
            f"groups column {groups!r} has no repeated units (each appears once), "
            f"so a held-out sample is already a held-out unit; using row-level CV. "
            f"The claimed generalization_target still holds and is recorded with "
            f"grouped=False.",
            SingletonGroupsWarning,
            stacklevel=3,
        )
        return None
    return np.asarray(g)


def _make_cv(
    n_splits: int,
    n_repeats: int,
    grouped: bool,
    random_state: int,
) -> RepeatedStratifiedKFold | _RepeatedStratifiedGroupKFold:
    """Repeated stratified K-fold — group-aware when grouped (keeps units intact)."""
    if grouped:
        return _RepeatedStratifiedGroupKFold(n_splits, n_repeats, random_state)
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )


class _RepeatedStratifiedGroupKFold:
    """Repeat StratifiedGroupKFold with a reseeded shuffle each repeat.

    sklearn ships ``StratifiedGroupKFold`` but not a *repeated* variant; we repeat it
    with a per-repeat seed so the stability loop and null get many grouped resamples.
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
            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state + repeat,
            )
            out.extend(cv.split(x, y, groups))
        return out


def _split(
    cv: RepeatedStratifiedKFold | _RepeatedStratifiedGroupKFold,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Uniform split() over the grouped/ungrouped CV objects."""
    if isinstance(cv, _RepeatedStratifiedGroupKFold):
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
    c_grid: tuple[float, ...],
    l1_grid: tuple[float, ...],
    select: Selection,
) -> tuple[float, float]:
    """The best (or neighbourhood-smoothed best) cell of the inner-CV surface.

    A cell whose own inner fit failed (NaN score) is never a candidate, even if
    smoothing gave it a finite neighbour mean; an all-NaN surface raises. Ties break
    to the first cell in row-major order (the most regularized ``C`` at the lowest
    ``l1_ratio``), which is also GridSearchCV's own first-max rule, so
    ``select="best"`` reproduces its choice exactly.
    """
    surface = _neighborhood_smooth(grid_scores) if select == "smoothed" else grid_scores
    surface = np.where(np.isfinite(grid_scores), surface, np.nan)
    if not np.any(np.isfinite(surface)):
        raise ValueError("inner-CV scores are all non-finite; cannot select a cell.")
    row, col = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    return c_grid[row], l1_grid[col]


def _grid_table(
    cv_results: Mapping[str, Any],
    row_key: str,
    row_grid: tuple[float, ...],
    col_key: str,
    col_grid: tuple[float, ...],
) -> np.ndarray:
    """Arrange GridSearchCV's flat ``mean_test_score`` as a ``(rows, cols)`` table.

    Indexed by the parameter values GridSearchCV records for each result — never by
    position. ``ParameterGrid`` enumerates parameters in *sorted key* order, so a
    positional reshape is right only when the key names happen to sort the way the
    table is laid out; looking the values up makes the layout independent of naming.
    Raises if a cell is written twice or left unfilled.
    """
    row_index = {float(v): i for i, v in enumerate(row_grid)}
    col_index = {float(v): j for j, v in enumerate(col_grid)}
    scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    rows = np.asarray(cv_results[row_key], dtype=float)
    cols = np.asarray(cv_results[col_key], dtype=float)
    table = np.full((len(row_grid), len(col_grid)), np.nan)
    filled = np.zeros(table.shape, dtype=bool)
    for score, r, c in zip(scores, rows, cols, strict=True):
        if float(r) not in row_index or float(c) not in col_index:
            raise RuntimeError(
                f"GridSearchCV evaluated a parameter pair ({r}, {c}) not in the grid."
            )
        i, j = row_index[float(r)], col_index[float(c)]
        if filled[i, j]:
            raise RuntimeError(f"GridSearchCV reported the cell ({r}, {c}) twice.")
        table[i, j] = score
        filled[i, j] = True
    if not filled.all():
        raise RuntimeError("GridSearchCV left grid cells unevaluated; cannot align.")
    return table


def _make_inner_cv(
    grouped: bool, n_splits: int, random_state: int
) -> StratifiedKFold | StratifiedGroupKFold:
    """The inner tuning splitter: group-aware whenever the outer CV is.

    A row-level inner split under a grouped outer CV puts a unit's replicates on both
    sides of a tuning fold, so the tuning surface is inflated (AUC near 1 on pure
    noise) and the selected hyperparameters reward memorizing the unit rather than
    generalizing to a new one. The seed fixes the inner folds, so GridSearchCV reuses
    exactly the folds :func:`_check_inner_folds` inspected.
    """
    if grouped:
        return StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _check_inner_folds(
    inner: StratifiedKFold | StratifiedGroupKFold,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
) -> None:
    """Refuse an inner tuning fold whose test half holds a single class.

    Its score is undefined, and GridSearchCV would otherwise carry a NaN into the
    tuning surface or fail deep inside. Under grouped CV this happens when a unit
    carries most of one class; the fix is fewer splits or more units, said plainly.
    """
    for j, (_, test) in enumerate(inner.split(x, y, groups)):
        if len(np.unique(y[test])) < 2:
            raise ValueError(
                f"inner tuning fold {j} holds a single class (n={len(test)}), so the "
                f"tuning score is undefined. With grouped CV this happens when a unit "
                f"carries most of one class — lower n_splits or use more units."
            )


def _tune(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, cfg: _Config
) -> tuple[float, float, np.ndarray]:
    """Inner-CV grid search -> (best_c, best_l1, the (C x l1_ratio) score table).

    ``refit=False``: the caller refits at the cell :func:`_select_cell` chose, so the
    ``select`` rule governs every fit — the outer folds and the all-data model alike —
    not GridSearchCV's own argmax. The inner splitter is grouped whenever ``groups``
    is given (:func:`_make_inner_cv`).
    """
    inner = _make_inner_cv(groups is not None, cfg.n_splits, cfg.random_state)
    _check_inner_folds(inner, x, y, groups)
    search = GridSearchCV(
        _build_pipeline(None, cfg.l1_grid[0], cfg.max_iter, cfg.tol, cfg.random_state),
        {"lr__C": list(cfg.c_grid), "lr__l1_ratio": list(cfg.l1_grid)},
        cv=inner,
        scoring=cfg.tuning_metric,
        n_jobs=cfg.n_jobs,
        refit=False,
    )
    search.fit(x, y, groups=groups)
    grid_scores = _grid_table(
        search.cv_results_, "param_lr__C", cfg.c_grid, "param_lr__l1_ratio", cfg.l1_grid
    )
    best_c, best_l1 = _select_cell(grid_scores, cfg.c_grid, cfg.l1_grid, cfg.select)
    return best_c, best_l1, grid_scores


# --------------------------------------------------------------------------- #
# Internal: the three CV uses
# --------------------------------------------------------------------------- #
def _nested_performance(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    cfg: _Config,
) -> tuple[list[FoldPrediction], float, float, float, float]:
    """Tune-in-fold repeated stratified CV -> per-fold ROC input + AUC/balacc/AP.

    Each outer fold: tune on the training fold (inner CV, grouped like the outer),
    refit at the selected cell on the training fold, score the untouched test fold.
    """
    outer = _make_cv(cfg.n_splits, cfg.n_repeats, groups is not None, cfg.random_state)
    folds: list[FoldPrediction] = []
    aucs: list[float] = []
    accs: list[float] = []
    aps: list[float] = []
    for i, (train, test) in enumerate(_split(outer, x, y, groups)):
        if len(np.unique(y[test])) < 2:
            raise ValueError(
                f"outer test fold {i} holds a single class (n={len(test)}), so its "
                f"AUC is undefined. With grouped CV this happens when a unit carries "
                f"most of one class — use more units per fold (lower n_splits) or "
                f"row-level CV."
            )
        groups_train = None if groups is None else groups[train]
        best_c, best_l1, _ = _tune(x[train], y[train], groups_train, cfg)
        model = _build_pipeline(
            best_c, best_l1, cfg.max_iter, cfg.tol, cfg.random_state
        )
        model.fit(x[train], y[train])
        prob = np.asarray(model.predict_proba(x[test]), dtype=float)[:, 1]
        folds.append(
            FoldPrediction(
                y_true=y[test].copy(),
                y_prob=prob,
                repeat=i // cfg.n_splits,
                fold=i % cfg.n_splits,
                test_indices=np.asarray(test, dtype=int).copy(),
            )
        )
        aucs.append(float(roc_auc_score(y[test], prob)))
        accs.append(float(balanced_accuracy_score(y[test], (prob >= 0.5).astype(int))))
        aps.append(float(average_precision_score(y[test], prob)))
    return (
        folds,
        float(np.mean(aucs)),
        float(np.std(aucs)),
        float(np.mean(accs)),
        float(np.mean(aps)),
    )


def _all_data_fit(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, cfg: _Config
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Tune (C, l1_ratio) on all data + refit -> best_c, best_l1, coef, grid."""
    best_c, best_l1, grid_scores = _tune(x, y, groups, cfg)
    final = _build_pipeline(best_c, best_l1, cfg.max_iter, cfg.tol, cfg.random_state)
    final.fit(x, y)
    lr = final.named_steps["lr"]
    coef = np.asarray(lr.coef_, dtype=float).ravel()
    return best_c, best_l1, coef, grid_scores


def _stability(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_c: float,
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
            best_c, best_l1, cfg.max_iter, cfg.tol, cfg.random_state
        )
        model.fit(x[train], y[train])
        coefs.append(np.asarray(model.named_steps["lr"].coef_, dtype=float).ravel())
    return np.asarray(coefs, dtype=float)


def _fixed_cv_auc(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_c: float,
    best_l1: float,
    cfg: _Config,
) -> float:
    """Mean AUC from repeated K-fold at FIXED hyperparameters (observed + each null)."""
    cv = _make_cv(cfg.n_splits, cfg.null_repeats, groups is not None, cfg.random_state)
    model = _build_pipeline(best_c, best_l1, cfg.max_iter, cfg.tol, cfg.random_state)
    if groups is None:
        scores = cross_val_score(
            model,
            x,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=cfg.n_jobs,
            error_score="raise",
        )
        return float(np.mean(scores))
    aucs: list[float] = []
    for train, test in _split(cv, x, y, groups):
        if len(np.unique(y[test])) < 2:
            raise ValueError(
                f"null-CV test fold holds a single class (n={len(test)}), so its AUC "
                f"is undefined. A unit-level permutation under grouped CV can move a "
                f"whole class into one fold — lower n_splits or use more units."
            )
        model.fit(x[train], y[train])
        prob = np.asarray(model.predict_proba(x[test]), dtype=float)[:, 1]
        aucs.append(float(roc_auc_score(y[test], prob)))
    return float(np.mean(aucs))


def _permute_labels(
    y: np.ndarray,
    groups: np.ndarray | None,
    rng: np.random.Generator,
    scheme: NullPermutation = "units",
) -> np.ndarray:
    """Shuffle labels for one null draw.

    Ungrouped: a plain row shuffle. Grouped, ``scheme="units"``: shuffle one label per
    unit (a unit's samples keep a common label — the null for a label that is a *unit*
    property, e.g. a subject/animal). Grouped, ``scheme="within_units"``: shuffle labels
    *inside* each unit, preserving every unit's class counts — the restricted
    (within-block) permutation for a batch-grouped design whose units hold both
    classes; see :func:`_permute_within_units`.
    """
    if groups is None:
        return np.asarray(rng.permutation(y), dtype=int)
    if scheme == "within_units":
        return _permute_within_units(y, groups, rng)
    units, inverse = np.unique(groups, return_inverse=True)
    unit_label = _unit_labels(y, groups, units)
    shuffled = rng.permutation(unit_label)
    return np.asarray(shuffled[inverse], dtype=int)


def _unit_labels(y: np.ndarray, groups: np.ndarray, units: np.ndarray) -> np.ndarray:
    """One label per unit — raises if any unit carries both classes.

    A unit-level null permutes *unit* labels, which is only defined when each unit has
    one label; a mixed unit means ``groups`` is not the unit of the outcome (rounding
    a mixed unit would silently change the class balance of the permuted labels). A
    **batch-grouped design** (``generalization_target="batches"``, each batch holding
    both classes) is the within-unit case: pass ``null_permutation="within_units"``.
    """
    out = np.empty(len(units), dtype=int)
    for i, u in enumerate(units):
        labels = np.unique(y[groups == u])
        if len(labels) != 1:
            raise ValueError(
                f"groups unit {u!r} carries both classes; a unit-level label-shuffle "
                f"null needs one label per unit. Use a groups column that is the unit "
                f"of the outcome, run without groups, or — for a batch-grouped design "
                f"whose units hold both classes — pass null_permutation='within_units'."
            )
        out[i] = int(labels[0])
    return out


def _permute_within_units(
    y: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle labels inside each unit, preserving every unit's class counts.

    The restricted permutation for exchangeable blocks (Anderson & ter Braak 2003;
    Winkler et al. 2015): under H0 the label is independent of the features *given the
    unit*, so labels are exchangeable within a unit but not across units. Every draw
    keeps each unit's class composition — hence the unit↔label association, the global
    class balance, and (because ``StratifiedGroupKFold`` assigns units to folds from
    their class counts) the grouped CV folds themselves — so only the labels move.
    Assumes samples within a unit are exchangeable: not for a nested design (subjects
    repeated within a batch), which needs a multi-level block permutation.
    """
    out = np.asarray(y, dtype=int).copy()
    for u in np.unique(groups):
        idx = np.flatnonzero(groups == u)
        out[idx] = y[rng.permutation(idx)]
    return out


def _within_unit_arrangements(y: np.ndarray, groups: np.ndarray) -> int:
    """Distinct label arrangements a within-unit permutation can reach.

    ``prod_u C(n_u, k_u)`` over units (``k_u`` positives of ``n_u``), as an exact Python
    ``int`` — never a numpy product, which overflows int64 silently on a few large
    units. ``1`` means every unit carries one class: the permutation is the identity
    and a null built on it would reproduce the observed AUC (p = 1) — the caller
    refuses. Below ``n_permutations`` the draws must repeat and the empirical p resolves
    only to about ``1 / arrangements`` — the caller warns.
    """
    total = 1
    for u in np.unique(groups):
        in_unit = groups == u
        total *= math.comb(int(in_unit.sum()), int(y[in_unit].sum()))
    return total


def _null_distribution(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_c: float,
    best_l1: float,
    cfg: _Config,
) -> tuple[np.ndarray, float, float]:
    """Label-shuffle null (fixed hyperparameters) -> null_aucs, observed, p."""
    observed = _fixed_cv_auc(x, y, groups, best_c, best_l1, cfg)
    if not np.isfinite(observed):
        raise ValueError(
            "the observed null-CV score is not finite; a permutation p cannot be "
            "formed from it (a NaN would read as p = 1 / (n_permutations + 1))."
        )
    rng = np.random.default_rng(cfg.random_state)
    nulls = np.array(
        [
            _fixed_cv_auc(
                x,
                _permute_labels(y, groups, rng, cfg.null_permutation),
                groups,
                best_c,
                best_l1,
                cfg,
            )
            for _ in range(cfg.n_permutations)
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(nulls)):
        raise ValueError(
            "a null draw produced a non-finite score; a permutation p cannot be formed "
            "(a NaN draw would silently count as 'not above the observed')."
        )
    p = float((np.sum(nulls >= observed) + 1) / (cfg.n_permutations + 1))
    return nulls, observed, p


# --------------------------------------------------------------------------- #
# Internal: coefficient table
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
    c_grid: tuple[float, ...]
    l1_grid: tuple[float, ...]
    select: Selection
    tuning_metric: str
    n_splits: int
    n_repeats: int
    stability_repeats: int
    null_repeats: int
    n_permutations: int
    null_permutation: NullPermutation
    max_iter: int
    tol: float
    n_jobs: int
    random_state: int


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def classify(
    dataset: Dataset,
    outcome: str,
    *,
    binarize: BinarizeSpec | None = None,
    positive_class: str | None = None,
    groups: str | None = None,
    generalization_target: GeneralizationTarget = "samples",
    feature_list: Sequence[str] | None = None,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    l1_ratios: Sequence[float] = DEFAULT_L1_RATIOS,
    select: Selection = "best",
    tuning_metric: str = "roc_auc",
    n_splits: int = 5,
    n_repeats: int = 5,
    stability_repeats: int = 10,
    run_null: bool = False,
    n_permutations: int = 1000,
    null_repeats: int = 3,
    null_permutation: NullPermutation = "units",
    max_iter: int = 5000,
    tol: float = 1e-4,
    n_jobs: int = -1,
    random_state: int = 0,
) -> ClassificationResult:
    """Fit a leakage-safe elastic-net logistic classifier; report the three parts.

    Parameters
    ----------
    dataset:
        The **experimental subset** on a **log2-like** scale, missing values resolved
        upstream (a ``NaN`` raises). Constant/all-zero features are dropped.
    outcome:
        The metadata column that defines the class. An already-binary categorical
        column is used directly; a continuous or >2-level column needs ``binarize``.
    binarize:
        The rule reducing a non-binary outcome to two classes — :class:`Threshold` (cut
        a continuous column) or :class:`LevelMap` (assign categorical levels; unlisted
        ones are dropped). Required unless ``outcome`` already has exactly two classes.
    positive_class:
        For an already-binary categorical outcome, which level is class 1 (the
        coefficient sign is *toward* it). Default is the sorted-second level.
    groups:
        Metadata column naming the independent unit (subject/animal/batch). Used for
        group-aware CV **only if it has repeats**; all-singletons -> row-level CV.
        With ``run_null=True`` the null permutes labels at the unit level by default
        (every unit must then carry one class — a subject/animal); a batch-grouped
        design, whose units hold both classes, runs its null with
        ``null_permutation="within_units"``.
    generalization_target:
        The performance question — ``"samples"``, ``"individuals"``, or ``"batches"``.
        Recorded; report performance as "on unseen <target>". With no repeats, unseen
        sample and unseen individual coincide.
    feature_list:
        Optional curated / hypothesis-driven feature ids to restrict to before fitting
        (prior knowledge; cuts dimensionality, can sharpen a weak signal). **Must be
        defined independent of ``outcome``** (a list derived from this data's class is
        circular). Applied to the whole matrix once (leakage-safe, since the list is
        outcome-independent); matched/unmatched counts are recorded.
    c_grid, l1_ratios:
        Elastic-net grid (``l1_ratio`` 0=ridge grouping .. 1=fully sparse).
    select:
        ``"best"`` (highest inner-CV score) or ``"smoothed"`` (neighborhood-smoothed,
        plateau-seeking) cell selection.
    tuning_metric:
        Inner-CV scoring for tuning (default ``"roc_auc"``).
    n_splits, n_repeats:
        Outer nested-CV folds and repeats (the honest performance estimate).
    stability_repeats:
        Repeats of the fixed-hyperparameter stability loop (``n_splits`` folds each).
    run_null:
        If ``True``, run the **label-shuffle null** (the gate that licenses trusting the
        coefficients and enables a ``validated`` finding). Opt-in because it is the main
        compute cost. If ``False``, ``null_*`` are ``None`` and the finding is capped at
        ``exploratory``.
    n_permutations, null_repeats:
        Number of label permutations, and the (lighter) fixed-hyperparameter CV repeats
        per permutation.
    null_permutation:
        How the label-shuffle null permutes under grouped CV. ``"units"`` (default)
        shuffles one label per unit — the null for a label that is a unit property
        (``generalization_target="individuals"``; a unit holding both classes raises).
        ``"within_units"`` shuffles labels inside each unit, preserving every unit's
        class counts — the restricted permutation for a batch-grouped design
        (``generalization_target="batches"``, batches holding both classes): it tests
        whether the features predict the label *beyond batch*, keeps the grouped folds
        identical to the observed run, and needs grouped CV with some mixed unit
        (all-single-class units raise; fewer distinct arrangements than
        ``n_permutations`` warn — :class:`NullPermutationWarning`). Under row-level
        CV ``"units"`` is the plain row shuffle and ``"within_units"`` raises. The
        scheme applied is recorded on the result and belongs in the cache fingerprint.
    max_iter, tol:
        saga convergence controls.
    n_jobs:
        Parallelism for the inner grid search / fixed-CV scoring (``-1`` = all cores).
    random_state:
        Recorded seed for every stochastic step.

    Returns
    -------
    ClassificationResult
    """
    c_grid_t = tuple(float(c) for c in c_grid)
    l1_grid_t = tuple(float(v) for v in l1_ratios)
    if not c_grid_t or not l1_grid_t:
        raise ValueError("c_grid and l1_ratios must be non-empty.")
    if any(not 0.0 <= v <= 1.0 for v in l1_grid_t):
        raise ValueError(f"l1_ratios must be in [0, 1]; got {l1_grid_t}.")
    if select not in ("best", "smoothed"):
        raise ValueError(f"select must be 'best' or 'smoothed'; got {select!r}.")
    if select == "smoothed" and len(c_grid_t) * len(l1_grid_t) < 3:
        raise ValueError(
            "select='smoothed' needs a grid of at least 3 cells: on 2 cells the "
            "smoothed surface is flat and the first cell is always chosen."
        )

    if null_permutation not in ("units", "within_units"):
        raise ValueError(
            f"null_permutation must be 'units' or 'within_units'; "
            f"got {null_permutation!r}."
        )
    if generalization_target not in ("samples", "individuals", "batches"):
        raise ValueError(
            f"generalization_target must be 'samples', 'individuals' or 'batches'; "
            f"got {generalization_target!r}."
        )
    if generalization_target != "samples" and groups is None:
        raise ValueError(
            f"generalization_target={generalization_target!r} claims performance on "
            f"unseen {generalization_target}, which needs the unit column: pass "
            f"groups=<column>. Leave groups unset only with "
            f"generalization_target='samples'."
        )
    if n_splits < 2 or n_repeats < 1 or stability_repeats < 1:
        raise ValueError(
            f"n_splits must be >= 2 and n_repeats / stability_repeats >= 1; got "
            f"n_splits={n_splits}, n_repeats={n_repeats}, "
            f"stability_repeats={stability_repeats}."
        )
    if n_permutations < 1 or null_repeats < 1:
        raise ValueError(
            f"n_permutations and null_repeats must be >= 1 (a null with no draws "
            f"would still read as licensed); got n_permutations={n_permutations}, "
            f"null_repeats={null_repeats}."
        )

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
            "decision (conventions/statistics.md); this classifier does not silently "
            "impute. Resolve missingness before classification."
        )
    _check_scale(dataset.scale)

    labels = _resolve_labels(metadata[outcome], binarize, positive_class, outcome)
    keep = labels.keep
    y = labels.y
    n_dropped_unassigned = int(n_samples - int(keep.sum()))
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

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Need >=2 samples per class; got positive={n_pos}, negative={n_neg}."
        )
    _check_class_sizes_for_nested_cv(n_pos, n_neg, n_splits)

    groups_kept = _resolve_grouping(groups, metadata, keep)
    grouped = groups_kept is not None
    if run_null and null_permutation == "within_units":
        # Fail before the expensive CV: a within-unit null needs grouped CV + freedom.
        if groups_kept is None:
            raise ValueError(
                f"null_permutation='within_units' needs grouped CV, but this run is "
                f"row-level (groups={groups!r}: not given, or no repeated units — see "
                f"SingletonGroupsWarning). Use null_permutation='units' (the default)."
            )
        n_arrangements = _within_unit_arrangements(y, groups_kept)
        if n_arrangements == 1:
            raise ValueError(
                "null_permutation='within_units' has no freedom: every groups unit "
                "carries one class, so a within-unit shuffle is the identity and the "
                "null would reproduce the observed AUC (p = 1). Use "
                "null_permutation='units' — the label is a unit property here."
            )
        if n_arrangements < n_permutations:
            warnings.warn(
                f"Within-unit permutation can reach only {n_arrangements} distinct "
                f"label arrangements, fewer than n_permutations={n_permutations}: "
                f"draws will repeat and the empirical p resolves only to about "
                f"1/{n_arrangements}. Larger or more mixed units give a finer null.",
                NullPermutationWarning,
                stacklevel=2,
            )
    elif run_null and groups_kept is not None:
        # Fail before the expensive CV: a unit-level null needs one label per unit.
        _unit_labels(y, groups_kept, np.unique(groups_kept))
    null_scheme: str | None = None
    if run_null:
        null_scheme = "samples" if groups_kept is None else null_permutation

    cfg = _Config(
        c_grid=c_grid_t,
        l1_grid=l1_grid_t,
        select=select,
        tuning_metric=tuning_metric,
        n_splits=n_splits,
        n_repeats=n_repeats,
        stability_repeats=stability_repeats,
        null_repeats=null_repeats,
        n_permutations=n_permutations,
        null_permutation=null_permutation,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    folds, cv_auc, cv_auc_sd, cv_acc, cv_ap = _nested_performance(
        x, y, groups_kept, cfg
    )
    best_c, best_l1, final_coef, grid_scores = _all_data_fit(x, y, groups_kept, cfg)
    resample_coef = _stability(x, y, groups_kept, best_c, best_l1, cfg)
    coeff_table = _coefficient_table(final_coef, resample_coef, kept_features)

    null_aucs: np.ndarray | None = None
    observed_auc: float | None = None
    null_p: float | None = None
    if run_null:
        null_aucs, observed_auc, null_p = _null_distribution(
            x, y, groups_kept, best_c, best_l1, cfg
        )

    return ClassificationResult(
        coefficients=coeff_table,
        fold_predictions=folds,
        cv_auc=cv_auc,
        cv_auc_sd=cv_auc_sd,
        cv_balanced_accuracy=cv_acc,
        cv_average_precision=cv_ap,
        best_c=best_c,
        best_l1_ratio=best_l1,
        grid_scores=grid_scores,
        c_grid=c_grid_t,
        l1_grid=l1_grid_t,
        outcome=outcome,
        positive_label=labels.positive_label,
        negative_label=labels.negative_label,
        generalization_target=generalization_target,
        grouped=grouped,
        groups_column=groups if grouped else None,
        n_samples=int(keep.sum()),
        n_positive=n_pos,
        n_negative=n_neg,
        n_features=int(x.shape[1]),
        n_dropped_constant=n_dropped_constant,
        n_dropped_unassigned=n_dropped_unassigned,
        random_state=random_state,
        null_aucs=null_aucs,
        observed_auc=observed_auc,
        null_p=null_p,
        n_features_requested=n_requested,
        n_features_matched=n_matched,
        null_permutation=null_scheme,
        feature_names=kept_features,
        tuning_metric=tuning_metric,
        inner_cv_grouped=grouped,
    )


def _check_scale(scale: str) -> None:
    if scale not in _LOG2_LIKE:
        warnings.warn(
            f"Classification on scale {scale!r}: in-fold standardization tolerates the "
            f"scale, but linear-scale abundances are right-skewed and leave outlier "
            f"z-scores the model is sensitive to. Run on log data unless you have a "
            f"specific reason not to.",
            ClassificationScaleWarning,
            stacklevel=3,
        )


def _check_class_sizes_for_nested_cv(n_pos: int, n_neg: int, n_splits: int) -> None:
    """Each class must keep >= n_splits samples inside every outer *training* fold.

    The inner tuning CV runs on the outer training fold, which has lost up to
    ``ceil(min_class / n_splits)`` minority samples to the outer test fold; guarding
    only ``min_class >= n_splits`` would let the inner CV fail deep inside. The bound
    is exact for row-level stratified folds. Under grouped CV (outer and inner folds
    alike) it is necessary, not sufficient — a whole minority unit can leave with a
    test fold — and the inner-fold check refuses such a split loudly, never silently.
    """
    min_class = min(n_pos, n_neg)
    inner_min = min_class - math.ceil(min_class / n_splits)
    if inner_min < n_splits:
        raise ValueError(
            f"Nested CV needs each class to keep >= n_splits ({n_splits}) samples "
            f"inside every outer training fold (min class >= n_splits + "
            f"ceil(min class / n_splits)); got positive={n_pos}, negative={n_neg}, "
            f"leaving only {inner_min} of the minority class for the inner CV. Lower "
            f"n_splits to proceed."
        )
