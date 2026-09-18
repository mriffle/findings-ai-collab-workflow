"""Linear-SVM classification for feature finding — leakage-safe soft-margin SVM.

TEMPLATE (lib/) — a *seed* for a project's classification script, not a finished
analysis. Copy it into the project's ``scripts/`` and adapt the call site per study
(which metadata column is the outcome, how to reduce it to two classes, the independent
unit). Held to the correctness charter (conventions/correctness.md) and the statistics
convention (conventions/statistics.md): assume nothing, verify everything, fail loud.

**What this answers.** *Can the proteome predict the class, and how well?* — a
**linear support-vector machine** (``SVC(kernel="linear")``): the maximum-margin
separating hyperplane, the *second linear model* beside the elastic-net classifier
(``analysis.classification``). Its signed standardized weights are reported as a
caveated interpretation of the classifier, **not** an all-relevant feature selection
(that is Boruta's job — FEATURE_FINDING.md §B). A max-margin weight vector is
**dense**: with correlated features the weight is *shared* across a cluster (ridge-like)
rather than concentrated on one member (as L1 does), so the ranking tends to be denser
around a co-regulated module. State that in the finding. Small-*n*/high-*p* proteomics
is typically linearly separable, which is the regime the max-margin geometry suits.

**The three coupled deliverables** (all in :class:`SVMClassificationResult`), the *same
shape as the elastic-net classifier* so the readouts line up:

  * **Performance vs a label-shuffle null** — leakage-safe **nested CV** (tune ``C`` in
    inner folds, estimate on outer folds), in-fold ``StandardScaler``,
    ``class_weight="balanced"``. The null is **opt-in** (``run_null=True``): the gate
    that licenses trusting the weights; it maps to the exploratory/validated
    distinction — run it to be eligible for ``validated``, skip it and the finding is
    capped at ``exploratory`` (weights flagged "not tested against a null").
  * **All-data weights** — tune on all data, refit on all data; the reported
    **standardized** signed weights (magnitude = importance, sign = direction) plus the
    support-vector counts.
  * **Cross-fold stability** — a dedicated fixed-``C`` resampling loop giving each
    feature a **top-k membership frequency**, **sign consistency**, and weight
    distribution.

**Divergences from the elastic-net classifier** (documented, deliberate — each follows
from the estimator, not from a different philosophy):

  1. **Dense, not sparse.** Every feature carries a non-zero weight, so *selection
     frequency* (fraction of resamples non-zero) is meaningless. The stability read is
     the **top-k membership frequency** — the fraction of stability resamples in which
     the feature ranks in the top ``top_k`` by |weight| — and the coefficient table
     holds **every** feature (sorted by |weight|), not a selected subset.
  2. **Scores are signed margins, not probabilities.** Every AUC is computed from
     ``decision_function`` (the signed distance to the hyperplane); no Platt scaling /
     ``probability=True`` anywhere (calibration is neither needed for a rank metric nor
     free — it would add an inner CV per fit). Balanced accuracy thresholds the margin
     at ``0``. The fold record carries ``y_score``, not ``y_prob``.
  3. **A 1-D ``C`` grid → a tuning *curve*, not a heatmap.** On separable data the
     inner-CV curve **plateaus** for every ``C`` above the hard-margin threshold (the
     solutions coincide), so ties are the normal case: the **smallest ``C`` on the
     plateau** (most regularized among the maxima) is chosen, deterministically, and the
     same rule governs both the outer loop and the all-data fit. The **hard-margin SVM
     is the large-``C`` limit** of this grid; pass ``c_grid=(1e6,)`` to pin it. Where
     the plateau begins scales roughly with ``1 / n_features`` (the kernel scale of
     standardized data is ~p), so the default grid reaches down to ``1e-5``, and a
     :class:`CGridEdgeWarning` fires when the selected ``C`` is a grid edge (the grid
     did not bracket the optimum — the C-curve figure shows it). A wide grid has a
     cost on tiny data: inner folds cannot resolve its low-``C`` end and sometimes
     tune below the plateau, so the result records the **plateau start** and the
     number of outer folds that tuned below it (a :class:`TuningNoiseWarning` past
     25 %), and the C-curve figure draws the per-fold picks — narrow the grid when
     that happens.
  4. **Fold identity is recorded** (``repeat``, ``fold``, ``test_indices`` on every
     fold record) and **per-repeat AUCs** are reported — the mean of a repeat's fold
     AUCs (primary) and the AUC of that repeat's *pooled* out-of-fold scores
     (supplementary; see :class:`SVMClassificationResult`) — so a comparison against
     another classifier run with the same seed is a **paired** per-fold comparison.
  5. **Support-vector counts** (total and per class) are reported: the number of
     samples that define the boundary is part of what makes the SVM interpretable.

``class_weight="balanced"`` scales the per-class box constraint to ``C · n / (2 n_c)``,
recomputed **inside each training fold**; the grid is the *base* ``C``. ``SVC`` with a
linear kernel is deterministic (libsvm's SMO; the ``random_state`` of ``SVC`` only
seeds Platt scaling, which is not used), so no seed is passed to the estimator.

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
non-log scale (a linear model on standardized features, like the elastic net), and
drops constant / all-zero features (they carry no signal). Covariate confounding is
**not** handled here: it is a Stage-1 caveat surfaced collaboratively in Stage 4
(conventions/statistics.md), gated by consequence.

Requires scikit-learn.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd
from common.data_loading import Dataset
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
from sklearn.svm import SVC

GeneralizationTarget = Literal["samples", "individuals", "batches"]
Selection = Literal["best", "smoothed"]
NullPermutation = Literal["units", "within_units"]

# Scales on which the input is a genuine log abundance. StandardScaler tolerates the
# scale, but linear-scale abundances are right-skewed and leave outlier z-scores the
# model is sensitive to; outside this set the classifier warns (see _check_scale).
_LOG2_LIKE: frozenset[str] = frozenset({"log2", "glog2", "log10", "ln", "zscore"})

# Default C grid: broad, log-spaced (half decades) from strong regularization to the
# hard-margin regime. Where the plateau begins scales roughly with 1/n_features (the
# kernel scale of standardized data is ~p): on a ~9,000-protein matrix the soft-margin
# regime sits around 1e-5..1e-3 and everything above is the hard margin; a 100-feature
# panel shifts the knee ~100x higher. So the grid reaches far down. Must be strictly
# increasing (the smallest-C tie-break depends on it). Exposed as an argument; this is
# the default — the C-curve figure shows whether it bracketed the optimum, and
# CGridEdgeWarning fires when it did not.
DEFAULT_C_GRID: tuple[float, ...] = (
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    100.0,
)

# Inner-CV scores within this distance of the maximum count as tied (a plateau). Far
# below the AUC granularity 1/(n_pos*n_neg) of any inner fold, so it only merges
# genuinely identical solutions (e.g. every C above the hard-margin threshold).
_TIE_TOL = 1e-6

# Fraction of outer folds that may tune C *below* the all-data plateau start before the
# template warns that the grid extends further than the inner folds can resolve.
_SUB_PLATEAU_WARN_FRACTION = 0.25

__script_meta__: dict[str, object] = {
    "template": {"name": "classification-svm", "version": "0.4"},
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
        "CGridEdgeWarning",
        "TuningNoiseWarning",
        "FoldPrediction",
        "SVMClassificationResult",
        "classify_svm",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Leakage-safe linear-SVM CLASSIFICATION (soft-margin SVC(kernel='linear'), C "
        "tuned; hard-margin is the large-C limit) over a Dataset — the second linear "
        "model beside analysis.classification, the same readouts. Nested CV "
        "performance (tune C in inner folds on ROC AUC from decision_function scores "
        "— no probability calibration) vs an opt-in label-shuffle null (the "
        "exploratory/validated gate); all-data standardized signed weights + "
        "support-vector counts; and a fixed-C stability loop (top-k membership "
        "frequency + sign consistency — the dense-model replacement for selection "
        "frequency) — reported together. Fold identity (repeat/fold/test_indices) and "
        "per-repeat AUCs recorded so a same-seed comparison with another classifier "
        "is paired per fold. Smallest-C tie-break on the inner-CV plateau. "
        "Study-agnostic outcome + binarize API (binary direct; a non-binary outcome "
        "needs an explicit rule); group-aware CV (outer and inner tuning folds alike) "
        "only when the groups column has repeats; the null permutes at the unit level "
        "or, for a batch-grouped design, within units (null_permutation). Warns on "
        "non-log scale, raises on NaN (missing handling upstream), drops constant "
        "features. Binary outcomes only (v0.1). v0.4: inner tuning folds grouped, the "
        "grid-edge warning keyed on the raw curve, smoothing needs three points, "
        "tuning_metric + inner_cv_grouped recorded. Requires scikit-learn."
    ),
}


# --------------------------------------------------------------------------- #
# Warnings
# (Duplicated from analysis.classification so this template is a self-contained seed.)
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


class CGridEdgeWarning(UserWarning):
    """The all-data-selected ``C`` sits at an edge of ``c_grid``.

    At the **lower** edge the inner-CV curve is still on its plateau at the smallest
    ``C`` tried — the grid never reached the soft-margin regime, so "the most
    regularized point on the plateau" is only the most regularized point *tried*
    (extend ``c_grid`` downward; the knee scales roughly with ``1 / n_features``). At
    the **upper** edge the curve was still rising — extend it upward. Either way the
    grid did not bracket the optimum, and the C-curve figure will show it. Keyed on
    the unsmoothed inner-CV curve (``plateau_start_c``), whatever ``select`` is.
    """


class TuningNoiseWarning(UserWarning):
    """Many outer folds tuned ``C`` below the all-data plateau start.

    On small data the inner folds cannot resolve the low-``C`` end of the grid (their
    AUC is too coarse), so the tuner sometimes picks an over-regularized ``C`` that the
    all-data curve places below the plateau. The nested estimate stays honest (it is
    what a new sample would experience) but it is being lowered by tuning noise, not by
    the data: narrow ``c_grid`` to bracket the knee (see the C-curve figure) and re-run.
    """


_FEATURE_LIST_WARN_FRACTION = 0.5


def _resolve_feature_list(
    feature_names: np.ndarray, feature_list: Sequence[str] | None
) -> tuple[np.ndarray, int | None, int | None]:
    """Return (keep-mask over features, n_requested, n_matched).

    An all-True mask is returned when no list is given (with ``None`` counts). Raises
    when a list matches nothing; warns on a poor match. Restricting to a prior /
    curated list is applied to the whole matrix **before** the CV — leakage-safe only
    because such a list is defined **independent of the outcome** (see
    :func:`classify_svm`).
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
# (Duplicated from analysis.classification so this template is a self-contained seed.)
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
    """Held-out margins from one outer CV fold (feeds the ROC curve).

    ``y_score`` is the SVM **decision function** (signed distance to the hyperplane,
    positive toward the positive class) — a score, not a probability. ``repeat`` /
    ``fold`` locate the fold in the repeated outer CV and ``test_indices`` are the
    held-out sample positions **in the analyzed sample set** (after the binarize drop
    mask), so a same-seed run of another classifier can be compared fold-by-fold.
    ``best_c`` is the ``C`` the inner CV chose on this fold's training data — read
    across folds it shows how stable the tuning is (a fold that picked a soft margin
    while the all-data fit sits on the hard-margin plateau is tuning noise).
    """

    y_true: np.ndarray
    y_score: np.ndarray
    repeat: int
    fold: int
    test_indices: np.ndarray
    best_c: float


@dataclass(frozen=True)
class SVMClassificationResult:
    """Everything the four figures and the finding read.

    Attributes
    ----------
    coefficients:
        Per-feature table, one row per analyzed feature (the model is **dense**), sorted
        by ``abs_coef`` descending. Columns: ``feature``, ``coef`` (all-data
        standardized weight), ``abs_coef``, ``top_k_frequency`` (fraction of stability
        resamples in which the feature ranks in the top ``top_k`` by |weight|),
        ``sign_consistency`` (``|sum(sign)| / n_resamples`` over all resamples; an exact
        zero counts as disagreement), ``coef_median``, ``coef_q25``, ``coef_q75`` (over
        all resamples), ``n_resamples``. The trust annotation on estimates.
    fold_predictions:
        Held-out ``(y_true, y_score, repeat, fold, test_indices, best_c)`` per outer
        nested-CV fold — the ROC input, the paired-comparison record, and the per-fold
        tuned ``C`` (tuning-stability read).
    cv_auc, cv_auc_sd, cv_balanced_accuracy, cv_average_precision:
        Nested-CV performance (mean over outer folds; ``_sd`` is the fold SD of AUC).
        Balanced accuracy thresholds the margin at 0.
    repeat_aucs:
        Per outer repeat, the **mean of that repeat's fold AUCs** (primary; their mean
        is ``cv_auc``). Length ``n_repeats``.
    repeat_pooled_aucs:
        Per outer repeat, the AUC of the **pooled** out-of-fold scores of that repeat's
        ``n_splits`` fold models. **Supplementary, with a caveat:** each fold model has
        its own weights, intercept and scaler, so the pooled scores share no common
        scale and the pooled ranking is not that of any single classifier. A large gap
        between pooled and mean-of-fold AUC signals fold-to-fold score-scale
        instability, not better or worse performance.
    best_c:
        The all-data-tuned ``C`` (also used for the stability loop and null).
    grid_scores, grid_scores_sd, c_grid:
        The all-data tuning curve — mean inner-CV AUC per ``C`` and its SD over inner
        folds; the hyperparameter-curve figure reads these.
    top_k:
        The ``k`` defining ``top_k_frequency``.
    plateau_start_c, n_folds_sub_plateau:
        The tuning-stability diagnostic: the smallest ``C`` whose all-data inner-CV
        score is within tolerance of the maximum (where the plateau starts on the
        unsmoothed curve), and how many outer folds tuned a ``C`` strictly below it. A
        large count means the grid extends below what the inner folds can resolve
        (:class:`TuningNoiseWarning` past 25 %); the C-curve figure draws the per-fold
        picks as ticks. ``plateau_start_c`` is always taken from the **unsmoothed**
        curve, so under ``select="best"`` it equals ``best_c``, while under
        ``select="smoothed"`` the per-fold picks follow the smoothed rule and the
        count mixes the two (a smoothed pick can sit below the unsmoothed start).
        ``None`` only on a result cached before v0.2.
    n_support_vectors, n_support_negative, n_support_positive:
        Support vectors of the all-data fit (total, and per class). With balanced class
        weights and a small ``C`` nearly every sample is a bounded support vector — a
        sign of heavy regularization, not a fault; on the hard-margin plateau only the
        boundary samples remain.
    null_aucs, observed_auc, null_p:
        The label-shuffle null (fixed-``C`` procedure): the permutation AUC
        distribution, the observed AUC computed by the *same* procedure, and the
        empirical p ``(#{null >= observed} + 1) / (n_perm + 1)``. ``C`` is **fixed at
        the all-data-tuned value** for the observed run and every permutation (the
        elastic-net convention): the test does not include tuning variance, and
        ``observed_auc`` (the fixed-``C`` procedure) is what ``null_p`` belongs to —
        it is not ``cv_auc``. All ``None`` when ``run_null`` was ``False`` — the
        finding is then capped at ``exploratory``.
    validated_eligible:
        ``True`` iff the null was run (the weight report is licensed).
    outcome, positive_label, negative_label:
        The resolved binary problem (``positive_label`` is class 1; weight sign is
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
    """

    coefficients: pd.DataFrame
    fold_predictions: list[FoldPrediction]
    cv_auc: float
    cv_auc_sd: float
    cv_balanced_accuracy: float
    cv_average_precision: float
    repeat_aucs: tuple[float, ...]
    repeat_pooled_aucs: tuple[float, ...]
    best_c: float
    grid_scores: np.ndarray
    grid_scores_sd: np.ndarray
    c_grid: tuple[float, ...]
    top_k: int
    n_support_vectors: int
    n_support_negative: int
    n_support_positive: int
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
    plateau_start_c: float | None = None
    n_folds_sub_plateau: int | None = None
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
# (Duplicated from analysis.classification so this template is a self-contained seed.)
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
def _build_pipeline(c: float, tol: float, max_iter: int) -> Pipeline:
    """StandardScaler + linear-kernel SVC (in-fold scaling, no leakage).

    ``class_weight="balanced"`` scales each class's box constraint to
    ``C * n / (2 * n_c)``, recomputed from the training fold the pipeline is fitted on
    (fold-local). ``probability`` stays ``False`` — every score in this template is the
    ``decision_function`` margin, so no Platt scaling and no ``random_state`` (libsvm's
    SMO is deterministic; the seed would only feed the unused probability CV).
    """
    svm = SVC(
        kernel="linear",
        C=c,
        class_weight="balanced",
        tol=tol,
        max_iter=max_iter,
    )
    return Pipeline([("scaler", StandardScaler()), ("svm", svm)])


def _scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """The SVM margin for each row of ``x`` (positive toward the positive class)."""
    return np.asarray(model.decision_function(x), dtype=float).ravel()


# (Grouping + CV construction duplicated from analysis.classification so this template
# is a self-contained seed. Kept byte-identical: the identical-splits guarantee across
# the classifier templates rests on it — see lib/tests/test_classification_splits.py.)
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
# Internal: hyperparameter selection on the 1-D C curve (best, or smoothed) — one
# rule for the outer loop and the all-data fit alike
# --------------------------------------------------------------------------- #
def _smooth_1d(scores: np.ndarray) -> np.ndarray:
    """Average each grid point with its two neighbours (edges average two values).

    Plateau-seeking: avoids latching onto an isolated high-scoring C that is likely
    noise. The 1-D counterpart of the elastic-net template's von-Neumann smoothing.
    """
    padded = np.pad(scores, 1, mode="constant", constant_values=np.nan)
    stack = np.stack([padded[:-2], padded[1:-1], padded[2:]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.asarray(np.nanmean(stack, axis=0), dtype=float)


def _select_c(
    scores: np.ndarray, c_grid: tuple[float, ...], select: Selection
) -> float:
    """The smallest C whose (optionally smoothed) inner-CV score is within tolerance
    of the maximum — the most regularized point on the plateau.

    On separable data every C above the hard-margin threshold yields the same
    solution, so the curve plateaus and the maximum is tied; the first index (the grid
    is strictly increasing) is the deterministic, most-regularized choice.
    """
    surface = _smooth_1d(scores) if select == "smoothed" else scores
    # A C whose own inner fit failed (NaN score) is never a candidate, even if
    # smoothing gave it a finite neighbour-mean.
    surface = np.where(np.isfinite(scores), surface, np.nan)
    if not np.any(np.isfinite(surface)):
        raise ValueError("inner-CV scores are all non-finite; cannot select C.")
    best = float(np.nanmax(surface))
    idx = int(np.flatnonzero(surface >= best - _TIE_TOL)[0])
    return c_grid[idx]


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


def _tune_c(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, cfg: _Config
) -> tuple[float, np.ndarray, np.ndarray]:
    """Inner-CV grid search over C -> (selected C, mean scores, SD over inner folds).

    ``refit=False``: the pipeline is refitted by the caller at the C chosen by
    :func:`_select_c`, so the ``select`` rule (and the smallest-C tie-break) governs
    every fit, not GridSearchCV's own argmax. The inner splitter is grouped whenever
    ``groups`` is given (:func:`_make_inner_cv`).
    """
    inner = _make_inner_cv(groups is not None, cfg.n_splits, cfg.random_state)
    _check_inner_folds(inner, x, y, groups)
    search = GridSearchCV(
        _build_pipeline(cfg.c_grid[0], cfg.tol, cfg.max_iter),
        {"svm__C": list(cfg.c_grid)},
        cv=inner,
        scoring=cfg.tuning_metric,
        n_jobs=cfg.n_jobs,
        refit=False,
    )
    search.fit(x, y, groups=groups)
    tried = np.asarray(search.cv_results_["param_svm__C"], dtype=float)
    if tried.shape != (len(cfg.c_grid),) or not np.allclose(tried, cfg.c_grid):
        raise RuntimeError(
            "GridSearchCV did not evaluate the C grid in order; cannot align scores."
        )
    means = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    sds = np.asarray(search.cv_results_["std_test_score"], dtype=float)
    return _select_c(means, cfg.c_grid, cfg.select), means, sds


# --------------------------------------------------------------------------- #
# Internal: the three CV uses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Performance:
    folds: list[FoldPrediction]
    auc: float
    auc_sd: float
    balanced_accuracy: float
    average_precision: float
    repeat_aucs: tuple[float, ...]
    repeat_pooled_aucs: tuple[float, ...]


def _nested_performance(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    cfg: _Config,
) -> _Performance:
    """Tune-in-fold repeated stratified CV -> per-fold ROC input + AUC/balacc/AP.

    Each outer fold: tune C on the training fold (inner CV), refit at that C on the
    training fold, score the untouched test fold by its margin. Fold identity is
    recorded (``repeat = i // n_splits``; both CV kinds emit repeats contiguously).
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
        best_c, _, _ = _tune_c(x[train], y[train], groups_train, cfg)
        model = _build_pipeline(best_c, cfg.tol, cfg.max_iter)
        model.fit(x[train], y[train])
        score = _scores(model, x[test])
        folds.append(
            FoldPrediction(
                y_true=y[test].copy(),
                y_score=score,
                repeat=i // cfg.n_splits,
                fold=i % cfg.n_splits,
                test_indices=np.asarray(test, dtype=int).copy(),
                best_c=best_c,
            )
        )
        aucs.append(float(roc_auc_score(y[test], score)))
        accs.append(float(balanced_accuracy_score(y[test], (score >= 0.0).astype(int))))
        aps.append(float(average_precision_score(y[test], score)))
    repeat_aucs, repeat_pooled = _per_repeat_aucs(folds, aucs)
    return _Performance(
        folds=folds,
        auc=float(np.mean(aucs)),
        auc_sd=float(np.std(aucs)),
        balanced_accuracy=float(np.mean(accs)),
        average_precision=float(np.mean(aps)),
        repeat_aucs=repeat_aucs,
        repeat_pooled_aucs=repeat_pooled,
    )


def _per_repeat_aucs(
    folds: list[FoldPrediction], fold_aucs: list[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean-of-fold AUC, pooled-OOF AUC) per outer repeat, in repeat order."""
    repeats = sorted({f.repeat for f in folds})
    means: list[float] = []
    pooled: list[float] = []
    for r in repeats:
        idx = [i for i, f in enumerate(folds) if f.repeat == r]
        means.append(float(np.mean([fold_aucs[i] for i in idx])))
        y_true = np.concatenate([folds[i].y_true for i in idx])
        y_score = np.concatenate([folds[i].y_score for i in idx])
        pooled.append(float(roc_auc_score(y_true, y_score)))
    return tuple(means), tuple(pooled)


@dataclass(frozen=True)
class _AllDataFit:
    best_c: float
    plateau_start_c: float
    coef: np.ndarray
    grid_scores: np.ndarray
    grid_scores_sd: np.ndarray
    n_support_negative: int
    n_support_positive: int


def _all_data_fit(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, cfg: _Config
) -> _AllDataFit:
    """Tune C on all data + refit -> best C, standardized weights, curve, SV counts.

    The grid-edge warning is keyed on the *unsmoothed* plateau start, not the
    (possibly smoothed) pick: 3-point smoothing averages an edge with two values and
    an interior point with three, which systematically favours an edge pick, so a
    smoothed pick at the top of the grid is not evidence the curve was still rising.
    """
    best_c, means, sds = _tune_c(x, y, groups, cfg)
    plateau_start = _select_c(means, cfg.c_grid, "best")  # unsmoothed, smallest-C
    _warn_if_grid_edge(plateau_start, cfg.c_grid)
    final = _build_pipeline(best_c, cfg.tol, cfg.max_iter)
    final.fit(x, y)
    svm = final.named_steps["svm"]
    # coef_ of a linear-kernel SVC is dual_coef_ @ support_vectors_ on the
    # *standardized* inputs (the scaler precedes it) — the standardized weight. Copied:
    # the property returns a read-only view.
    coef = np.asarray(svm.coef_, dtype=float).ravel().copy()
    n_support = np.asarray(svm.n_support_, dtype=int)  # in classes_ order: [0, 1]
    return _AllDataFit(
        best_c=best_c,
        plateau_start_c=plateau_start,
        coef=coef,
        grid_scores=means,
        grid_scores_sd=sds,
        n_support_negative=int(n_support[0]),
        n_support_positive=int(n_support[1]),
    )


def _warn_if_grid_edge(best_c: float, c_grid: tuple[float, ...]) -> None:
    """Warn when the selected C is a grid edge (the grid did not bracket it)."""
    if len(c_grid) < 2:
        return  # a single-element grid is a deliberate pin (e.g. hard margin)
    if best_c == c_grid[0]:
        warnings.warn(
            f"Selected C = {best_c:g} is the SMALLEST value in c_grid: the inner-CV "
            f"curve is still on its plateau there, so the grid never reached the "
            f"soft-margin regime and the choice is only 'the most regularized C "
            f"tried'. Extend c_grid downward (the knee scales ~1/n_features).",
            CGridEdgeWarning,
            stacklevel=4,
        )
    elif best_c == c_grid[-1]:
        warnings.warn(
            f"Selected C = {best_c:g} is the LARGEST value in c_grid: the inner-CV "
            f"curve was still rising at the top of the grid. Extend c_grid upward.",
            CGridEdgeWarning,
            stacklevel=4,
        )


def _count_sub_plateau(folds: list[FoldPrediction], plateau_start_c: float) -> int:
    """Outer folds whose tuned C is strictly below the all-data plateau start."""
    return int(sum(f.best_c < plateau_start_c for f in folds))


def _should_warn_tuning_noise(
    plateau_start_c: float, c_grid: tuple[float, ...]
) -> bool:
    """The sub-plateau count is meaningful only when a plateau exists below the top.

    When the (unsmoothed) plateau start *is* the upper grid edge the curve was still
    rising, every fold is trivially "below" it, and the relevant advice is
    :class:`CGridEdgeWarning`'s "extend upward" — not "narrow the grid". Keyed on the
    plateau start, not ``best_c``, so it is right under ``select="smoothed"`` too.
    """
    return len(c_grid) > 1 and plateau_start_c != c_grid[-1]


def _warn_if_tuning_noisy(n_sub_plateau: int, n_folds: int) -> None:
    """Warn when more than ``_SUB_PLATEAU_WARN_FRACTION`` of outer folds under-tuned."""
    if n_folds == 0 or n_sub_plateau <= _SUB_PLATEAU_WARN_FRACTION * n_folds:
        return
    warnings.warn(
        f"{n_sub_plateau} of {n_folds} outer folds tuned C below the all-data plateau "
        f"start: the inner folds cannot resolve the low-C end of c_grid, so tuning "
        f"noise (not the data) is lowering the nested estimate. Narrow c_grid to "
        f"bracket the knee shown in the C-curve figure and re-run.",
        TuningNoiseWarning,
        stacklevel=3,
    )


def _stability(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_c: float,
    cfg: _Config,
) -> np.ndarray:
    """Per-resample standardized weights at fixed C.

    Refits on the training side of ``stability_repeats`` x ``n_splits`` stratified
    resamples (the test side is discarded — this is a subsample-refit device, not an
    evaluation). Note the shared ``random_state``: the first ``n_repeats`` resamples
    coincide with the outer-CV training folds (inherited from the elastic-net template).
    """
    cv = _make_cv(
        cfg.n_splits, cfg.stability_repeats, groups is not None, cfg.random_state
    )
    coefs: list[np.ndarray] = []
    for train, _ in _split(cv, x, y, groups):
        model = _build_pipeline(best_c, cfg.tol, cfg.max_iter)
        model.fit(x[train], y[train])
        coefs.append(np.asarray(model.named_steps["svm"].coef_, dtype=float).ravel())
    return np.asarray(coefs, dtype=float)


def _fixed_cv_auc(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_c: float,
    cfg: _Config,
) -> float:
    """Mean AUC from repeated K-fold at FIXED C (observed + each null).

    The ``"roc_auc"`` scorer resolves to ``decision_function`` for an SVC pipeline
    (response-method order ``decision_function`` then ``predict_proba``), oriented
    toward ``classes_[1]`` — the same margin the grouped path computes explicitly.
    """
    cv = _make_cv(cfg.n_splits, cfg.null_repeats, groups is not None, cfg.random_state)
    model = _build_pipeline(best_c, cfg.tol, cfg.max_iter)
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
        aucs.append(float(roc_auc_score(y[test], _scores(model, x[test]))))
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
    cfg: _Config,
) -> tuple[np.ndarray, float, float]:
    """Label-shuffle null (fixed C) -> null_aucs, observed, p."""
    observed = _fixed_cv_auc(x, y, groups, best_c, cfg)
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
# Internal: coefficient table (dense — every feature; top-k membership, not
# selection frequency)
# --------------------------------------------------------------------------- #
def _top_k_membership(resample_coef: np.ndarray, top_k: int) -> np.ndarray:
    """(n_resamples, n_features) bool — is the feature in the resample's top-k |w|?

    Membership is ``|w| >= the k-th largest |w|`` of that resample, so features **tied**
    with the k-th weight are all members (a duplicated / perfectly collinear protein
    row gets the same frequency as its twin, never an arbitrary 1.0 vs 0.0 split); a
    resample with ties at rank k therefore has more than ``top_k`` members, and the
    frequencies sum to at least ``top_k``.
    """
    mag = np.abs(resample_coef)
    kth = np.sort(mag, axis=1)[:, ::-1][:, top_k - 1]
    member: np.ndarray = mag >= kth[:, None]
    return member


def _coefficient_table(
    final_coef: np.ndarray,
    resample_coef: np.ndarray,
    feature_names: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    """Assemble the per-feature table for every analyzed feature (dense model)."""
    n_resample = resample_coef.shape[0]
    top_k_freq = _top_k_membership(resample_coef, top_k).mean(axis=0)
    sign_cons = np.abs(np.sign(resample_coef).sum(axis=0)) / n_resample
    median = np.median(resample_coef, axis=0)
    q25 = np.percentile(resample_coef, 25, axis=0)
    q75 = np.percentile(resample_coef, 75, axis=0)
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": final_coef,
            "abs_coef": np.abs(final_coef),
            "top_k_frequency": top_k_freq,
            "sign_consistency": sign_cons,
            "coef_median": median,
            "coef_q25": q25,
            "coef_q75": q75,
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
    top_k: int
    select: Selection
    tuning_metric: str
    n_splits: int
    n_repeats: int
    stability_repeats: int
    null_repeats: int
    n_permutations: int
    null_permutation: NullPermutation
    tol: float
    max_iter: int
    n_jobs: int
    random_state: int


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def classify_svm(
    dataset: Dataset,
    outcome: str,
    *,
    binarize: BinarizeSpec | None = None,
    positive_class: str | None = None,
    groups: str | None = None,
    generalization_target: GeneralizationTarget = "samples",
    feature_list: Sequence[str] | None = None,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    top_k: int = 20,
    select: Selection = "best",
    tuning_metric: str = "roc_auc",
    n_splits: int = 5,
    n_repeats: int = 5,
    stability_repeats: int = 10,
    run_null: bool = False,
    n_permutations: int = 1000,
    null_repeats: int = 3,
    null_permutation: NullPermutation = "units",
    tol: float = 1e-3,
    max_iter: int = -1,
    n_jobs: int = -1,
    random_state: int = 0,
) -> SVMClassificationResult:
    """Fit a leakage-safe linear SVM classifier; report the three parts.

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
        For an already-binary categorical outcome, which level is class 1 (the weight
        sign is *toward* it). Default is the sorted-second level.
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
    c_grid:
        The soft-margin ``C`` grid: strictly increasing, all ``> 0``. Broad and
        log-spaced by default (``1e-5`` .. ``100``); the hard-margin SVM is its
        large-``C`` limit (a single-element grid such as ``(1e6,)`` pins it; the inner
        search still runs over that one value, so the nested class-size guard still
        applies). Where the inner-CV curve reaches its plateau scales roughly with
        ``1 / n_features``, so a much smaller or larger feature set may need the grid
        shifted; a :class:`CGridEdgeWarning` says when the selected ``C`` sits at a
        grid edge (the grid did not bracket the optimum).
    top_k:
        The ``k`` of the stability read: a feature's ``top_k_frequency`` is the fraction
        of stability resamples in which it ranks in the top ``k`` by |weight|. Must be
        ``1 <= top_k <= n_features`` (after constant dropping).
    select:
        ``"best"`` (highest inner-CV score) or ``"smoothed"`` (3-point neighbour-
        smoothed, plateau-seeking) selection on the C curve. Ties resolve to the
        smallest C either way.
    tuning_metric:
        Inner-CV scoring for tuning (default ``"roc_auc"``, computed from the margin).
    n_splits, n_repeats:
        Outer nested-CV folds and repeats (the honest performance estimate).
    stability_repeats:
        Repeats of the fixed-``C`` stability loop (``n_splits`` folds each).
    run_null:
        If ``True``, run the **label-shuffle null** (the gate that licenses trusting the
        weights and enables a ``validated`` finding). Opt-in because it is the main
        compute cost. If ``False``, ``null_*`` are ``None`` and the finding is capped at
        ``exploratory``.
    n_permutations, null_repeats:
        Number of label permutations, and the (lighter) fixed-``C`` CV repeats per
        permutation.
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
    tol, max_iter:
        libsvm stopping tolerance and iteration cap (``-1`` = no cap, the SVC default;
        with no cap there is no convergence warning to silence).
    n_jobs:
        Parallelism for the inner grid search / fixed-CV scoring (``-1`` = all cores).
        The grouped null path fits serially (its per-fold loop ignores ``n_jobs``).
    random_state:
        Recorded seed for every stochastic step (the CV shuffles and the null's
        permutations; the SVM solver itself is deterministic).

    Returns
    -------
    SVMClassificationResult
    """
    c_grid_t = tuple(float(c) for c in c_grid)
    if not c_grid_t:
        raise ValueError("c_grid must be non-empty.")
    if any(not np.isfinite(c) or c <= 0.0 for c in c_grid_t):
        raise ValueError(f"c_grid values must be finite and > 0; got {c_grid_t}.")
    if any(b <= a for a, b in pairwise(c_grid_t)):
        raise ValueError(
            f"c_grid must be strictly increasing (the tie-break rule 'smallest C among "
            f"the inner-CV maxima' depends on it); got {c_grid_t}."
        )
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}.")
    if select not in ("best", "smoothed"):
        raise ValueError(f"select must be 'best' or 'smoothed'; got {select!r}.")
    if select == "smoothed" and len(c_grid_t) < 3:
        raise ValueError(
            "select='smoothed' needs a c_grid of at least 3 values: on 2 points the "
            "smoothed curve is flat and the first C is always chosen."
        )
    if max_iter != -1 and max_iter < 1:
        raise ValueError(f"max_iter must be -1 (no cap) or >= 1; got {max_iter}.")

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
    if top_k > x.shape[1]:
        raise ValueError(
            f"top_k ({top_k}) exceeds the {x.shape[1]} analyzed features; a top-k "
            f"membership frequency of 1.0 everywhere would be meaningless. Lower top_k."
        )

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
        top_k=top_k,
        select=select,
        tuning_metric=tuning_metric,
        n_splits=n_splits,
        n_repeats=n_repeats,
        stability_repeats=stability_repeats,
        null_repeats=null_repeats,
        n_permutations=n_permutations,
        null_permutation=null_permutation,
        tol=tol,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    perf = _nested_performance(x, y, groups_kept, cfg)
    fit = _all_data_fit(x, y, groups_kept, cfg)
    n_sub_plateau = _count_sub_plateau(perf.folds, fit.plateau_start_c)
    if _should_warn_tuning_noise(fit.plateau_start_c, c_grid_t):
        _warn_if_tuning_noisy(n_sub_plateau, len(perf.folds))
    resample_coef = _stability(x, y, groups_kept, fit.best_c, cfg)
    coeff_table = _coefficient_table(fit.coef, resample_coef, kept_features, top_k)

    null_aucs: np.ndarray | None = None
    observed_auc: float | None = None
    null_p: float | None = None
    if run_null:
        null_aucs, observed_auc, null_p = _null_distribution(
            x, y, groups_kept, fit.best_c, cfg
        )

    return SVMClassificationResult(
        coefficients=coeff_table,
        fold_predictions=perf.folds,
        cv_auc=perf.auc,
        cv_auc_sd=perf.auc_sd,
        cv_balanced_accuracy=perf.balanced_accuracy,
        cv_average_precision=perf.average_precision,
        repeat_aucs=perf.repeat_aucs,
        repeat_pooled_aucs=perf.repeat_pooled_aucs,
        best_c=fit.best_c,
        grid_scores=fit.grid_scores,
        grid_scores_sd=fit.grid_scores_sd,
        c_grid=c_grid_t,
        top_k=top_k,
        n_support_vectors=fit.n_support_negative + fit.n_support_positive,
        n_support_negative=fit.n_support_negative,
        n_support_positive=fit.n_support_positive,
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
        plateau_start_c=fit.plateau_start_c,
        n_folds_sub_plateau=n_sub_plateau,
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
