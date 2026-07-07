"""XGBoost classification for feature finding — leakage-safe gradient-boosted trees.

TEMPLATE (lib/) — a *seed* for a project's XGBoost-classification script, not a finished
analysis. Copy it into the project's ``scripts/`` and adapt the call site per study
(which metadata column is the outcome, how to reduce it to two classes, the independent
unit). Held to the correctness charter (conventions/correctness.md) and the statistics
convention (conventions/statistics.md): assume nothing, verify everything, fail loud.

**What this answers.** *Can the proteome predict the class, and how well?* — the same
question as the elastic-net logistic classifier (``analysis.classification``), but a
**non-linear gradient-boosted-tree** model (``xgboost.XGBClassifier``) that captures
feature interactions and non-monotone effects a linear model cannot. Its feature
**importances** are reported as a caveated interpretation of the classifier, **not** an
all-relevant selection (that is Boruta's job — FEATURE_FINDING.md §B). Unlike the
elastic-net coefficients, a tree importance is **unsigned** (a magnitude, no direction):
it says a feature was *useful for splitting*, not whether it pushes toward the positive
class. With correlated proteomics features the trees spread importance across a cluster,
so a **low selection frequency does not mean a feature is unimportant** — it may be
redundant with a co-selected neighbour. State that in the finding.

**The three coupled deliverables** (all in :class:`XGBClassificationResult`), the *same
shape as the elastic-net classifier* so the readouts line up:

  * **Performance vs a label-shuffle null** — leakage-safe **nested CV** (tune
    ``(max_depth, learning_rate)`` in inner folds, estimate on outer folds). The null is
    **opt-in** (``run_null=True``): the gate that licenses trusting the importances; it
    maps to the exploratory/validated distinction — run it to be eligible for
    ``validated``, skip it and the finding is capped at ``exploratory`` (importances
    flagged "not tested against a null").
  * **All-data importances** — tune on all data, refit on all data; the reported
    **gain-based** importances (magnitude = usefulness for prediction; unsigned).
  * **Cross-fold stability** — a dedicated fixed-hyperparameter resampling loop giving
    each feature a **selection frequency** (fraction of resamples non-zero) and
    an importance distribution (median + IQR).

**Divergences from the elastic-net classifier** (documented, deliberate — trees are a
different animal):

  * **No standardization and no scale warning.** A tree splits on per-feature cuts,
    so it is invariant to any monotone rescaling of a feature; there is nothing to
    standardize (no in-fold ``StandardScaler``) and log-vs-linear scale does not affect
    the model — this template does **not** warn on a non-log scale (the elastic-net one
    does).
  * **Class imbalance via ``scale_pos_weight``** (``n_neg/n_pos``), recomputed **per
    fold** (mirrors how ``class_weight="balanced"`` is fold-local inside a sklearn
    pipeline), not ``class_weight``.
  * **Unsigned gain importance**, so there is **no sign consistency** and the importance
    plot has no zero line — a magnitude view, closer in spirit to the Boruta importance
    plot than to the signed coefficient plot.
  * **A 2-D tuning grid** (``max_depth`` x ``learning_rate``; the other tree knobs are
    held-constant scalar arguments) so nested CV stays tractable and the hyperparameter
    surface renders as the same 2-D heatmap the elastic-net classifier uses. Widen the
    grid in the project copy if a study warrants it.
  * **Missing values still raise.** XGBoost can natively route ``NaN`` (learned default
    split directions), but silently doing so is an un-audited imputation-like step;
    consistent with the rest of the analysis family, missing handling is an upstream
    Stage-2 decision and a ``NaN`` here **raises**.

**Generalization target & grouping.** Choose the CV scheme to answer a stated question:
name the target (``"samples"``/``"individuals"``/``"batches"``) and hold out folds at
that unit. Grouping is used **only when the ``groups`` column has repeats** — if
every unit appears once, a new sample *is* a new individual and row-level folds already
estimate individual-level performance, so nothing is grouped. When grouped, the null
permutes labels at the group level.

Scale / missing / sample set: the input should be the **experimental subset** with
**missing values already resolved upstream** (Stage-2). Constant / all-zero features are
dropped (they carry no signal). Covariate confounding is **not** handled here: it is a
Stage-1 caveat surfaced collaboratively in Stage 4 (conventions/statistics.md), gated by
consequence.

Requires scikit-learn and xgboost.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
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
)
from xgboost import XGBClassifier

GeneralizationTarget = Literal["samples", "individuals", "batches"]
Selection = Literal["best", "smoothed"]

# Default 2-D tuning grid: tree depth (model capacity) x learning rate (step size). The
# remaining XGBoost knobs are held-constant scalar arguments so the search stays 2-D
# (tractable nested CV + a 2-D hyperparameter heatmap). Exposed as arguments; widen in a
# project copy if warranted.
DEFAULT_MAX_DEPTH_GRID: tuple[int, ...] = (2, 3, 4, 5)
DEFAULT_LEARNING_RATE_GRID: tuple[float, ...] = (0.03, 0.1, 0.3)

# Gain (importance) at or below this counts as "selected out" — the feature was never
# used in a split, the tree analogue of a coefficient shrunk to zero.
_ZERO_IMPORTANCE = 1e-12

__script_meta__: dict[str, object] = {
    "template": {"name": "classification-xgboost", "version": "0.1"},
    "kind": "analysis",
    "provides": [
        "GeneralizationTarget",
        "Selection",
        "Threshold",
        "LevelMap",
        "BinarizeSpec",
        "SingletonGroupsWarning",
        "FoldPrediction",
        "XGBClassificationResult",
        "classify_xgboost",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Leakage-safe gradient-boosted-tree CLASSIFICATION (xgboost) over a Dataset — "
        "the non-linear sibling of analysis.classification, the same readouts. Nested "
        "CV performance (tune max_depth x learning_rate in inner folds, fold-local "
        "scale_pos_weight for imbalance) vs an opt-in label-shuffle null (the "
        "exploratory/validated gate); all-data gain importances; and a "
        "fixed-hyperparameter stability loop (selection frequency + importance IQR) — "
        "reported together. Importances are UNSIGNED (magnitude, no direction; no sign "
        "consistency) and a caveated interpretation (a low selection frequency != "
        "unimportant), not an all-relevant selection (that is Boruta). Trees are "
        "scale-invariant, so no standardization and no scale warning (unlike the "
        "elastic-net classifier). Study-agnostic outcome + binarize API (binary "
        "direct; a continuous/multi-level outcome needs a Threshold/LevelMap rule, "
        "unassigned samples dropped); group-aware CV only when the groups column has "
        "repeats (else row-level = individual-level; group-level null permutation); "
        "generalization_target recorded. Raises on NaN (missing handling upstream; "
        "native NaN routing deliberately unused), drops constant features. Binary "
        "outcomes only (v0.1). Uses common.data_loading. Requires scikit-learn + "
        "xgboost. Study-agnostic; fail-loud."
    ),
}


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #
class SingletonGroupsWarning(UserWarning):
    """The ``groups`` column has no repeated units, so grouping is a no-op.

    Every unit appears once, so a held-out sample is already a held-out unit; row-level
    CV is used (grouping singletons only degrades class balance).
    """


# --------------------------------------------------------------------------- #
# Outcome binarization spec — the caller's rule reducing an outcome to two classes.
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
    """

    positive: tuple[str, ...]
    negative: tuple[str, ...]


BinarizeSpec = Threshold | LevelMap


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FoldPrediction:
    """Held-out predictions from one outer CV fold (feeds the ROC curve)."""

    y_true: np.ndarray
    y_prob: np.ndarray


@dataclass(frozen=True)
class XGBClassificationResult:
    """Everything the four figures and the finding read.

    Attributes
    ----------
    importances:
        Per-feature table, one row per feature with **non-zero gain in the final model**
        (the selected set), sorted by ``importance`` descending. Columns: ``feature``,
        ``importance`` (all-data gain, >= 0, **unsigned**), ``selection_frequency``
        (fraction of stability resamples with non-zero gain), ``importance_median``,
        ``importance_q25``, ``importance_q75`` (over the non-zero resamples),
        ``n_resamples``. The trust annotation on estimates. There is **no** sign column
        (tree importance carries no direction).
    fold_predictions:
        Held-out ``(y_true, y_prob)`` per outer nested-CV fold — the ROC input.
    cv_auc, cv_auc_sd, cv_balanced_accuracy, cv_average_precision:
        Nested-CV performance (mean over outer folds; ``_sd`` is the fold SD of AUC).
    best_params:
        The all-data-tuned hyperparameters ``{"max_depth", "learning_rate"}`` (also
        used for the stability loop and null).
    grid_scores, max_depth_grid, learning_rate_grid:
        The all-data tuning surface — ``grid_scores`` is
        ``(len(max_depth_grid), len(learning_rate_grid))`` mean inner-CV AUC; the
        hyperparameter heatmap reads these.
    importance_type:
        The XGBoost importance measure reported (default ``"gain"``).
    null_aucs, observed_auc, null_p:
        The label-shuffle null (fixed-hyperparameter procedure): the permutation AUC
        distribution, the observed AUC computed by the *same* procedure, and the
        empirical p ``(#{null >= observed} + 1) / (n_perm + 1)``. All ``None`` when
        ``run_null`` was ``False`` — the finding is then capped at ``exploratory``.
    validated_eligible:
        ``True`` iff the null was run (the importance report is licensed).
    outcome, positive_label, negative_label:
        The resolved binary problem (``positive_label`` is class 1).
    generalization_target, grouped, groups_column:
        The CV design: the target claimed, whether folds were grouped (only when the
        ``groups`` column had repeats), and that column's name.
    n_samples, n_positive, n_negative, n_features, n_dropped_constant,
    n_dropped_unassigned:
        Analyzed counts (for ``provenance.params``); ``n_dropped_unassigned`` is samples
        excluded by the binarize rule.
    random_state:
        The recorded seed.
    """

    importances: pd.DataFrame
    fold_predictions: list[FoldPrediction]
    cv_auc: float
    cv_auc_sd: float
    cv_balanced_accuracy: float
    cv_average_precision: float
    best_params: dict[str, float]
    grid_scores: np.ndarray
    max_depth_grid: tuple[int, ...]
    learning_rate_grid: tuple[float, ...]
    importance_type: str
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
    feature_names: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    @property
    def validated_eligible(self) -> bool:
        """``True`` iff the label-shuffle null was run (licenses ``validated``)."""
        return self.null_p is not None


# --------------------------------------------------------------------------- #
# Internal: label resolution (outcome -> binary y + drop mask)
# (Duplicated from analysis.classification — self-contained seed.)
# --------------------------------------------------------------------------- #
def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series))


@dataclass(frozen=True)
class _Labels:
    y: np.ndarray  # (n_kept,) int 0/1
    keep: np.ndarray  # (n_samples,) bool — samples assigned a class
    positive_label: str
    negative_label: str


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
    labels = series.astype(str).to_numpy()
    keep = labels != "nan"
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
    labels = series.astype(str).to_numpy()
    pos_set, neg_set = set(spec.positive), set(spec.negative)
    overlap = pos_set & neg_set
    if overlap:
        raise ValueError(f"LevelMap positive/negative overlap on {sorted(overlap)}.")
    present = set(labels.tolist())
    unknown = (pos_set | neg_set) - present
    if unknown:
        raise ValueError(
            f"LevelMap references levels {sorted(unknown)} absent from {outcome!r} "
            f"(present: {sorted(present)})."
        )
    is_pos = np.isin(labels, list(pos_set))
    is_neg = np.isin(labels, list(neg_set))
    keep = is_pos | is_neg
    y = is_pos[keep].astype(int)
    return _Labels(
        y=y,
        keep=keep,
        positive_label="|".join(sorted(pos_set)),
        negative_label="|".join(sorted(neg_set)),
    )


# --------------------------------------------------------------------------- #
# Internal: estimator factory + fold-local class weighting
# --------------------------------------------------------------------------- #
def _scale_pos_weight(y: np.ndarray) -> float:
    """Return ``n_neg / n_pos`` for XGBoost imbalance (1.0 if a class is empty)."""
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0.0 or n_neg == 0.0:
        return 1.0
    return n_neg / n_pos


def _make_xgb(
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
    cfg: _Config,
    n_jobs: int,
) -> XGBClassifier:
    """Construct an ``XGBClassifier`` with the swept + held-constant settings.

    No pipeline/scaler: a tree splits on per-feature thresholds, so it is invariant to
    monotone rescaling and there is nothing to standardize. ``scale_pos_weight`` is
    passed by the caller from the *training* fold's class ratio (fold-local weighting).
    """
    return XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_weight=cfg.min_child_weight,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        gamma=cfg.gamma,
        objective="binary:logistic",
        tree_method="hist",
        importance_type=cfg.importance_type,
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        random_state=cfg.random_state,
        verbosity=0,
    )


def _importances(model: XGBClassifier) -> np.ndarray:
    """Gain-based importances (>= 0, sum to 1; unused features are exactly 0)."""
    return np.asarray(model.feature_importances_, dtype=float).ravel()


# --------------------------------------------------------------------------- #
# Internal: CV construction (group-aware, stratified — copied shape from classifier)
# --------------------------------------------------------------------------- #
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
            f"groups column {groups!r} has no repeated units (each appears once), "
            f"so a held-out sample is already a held-out unit; using row-level CV.",
            SingletonGroupsWarning,
            stacklevel=3,
        )
        return None
    return np.asarray(g)


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
    max_depth_grid: tuple[int, ...],
    learning_rate_grid: tuple[float, ...],
    select: Selection,
) -> tuple[int, float]:
    surface = _neighborhood_smooth(grid_scores) if select == "smoothed" else grid_scores
    row, col = np.unravel_index(int(np.nanargmax(surface)), surface.shape)
    return int(max_depth_grid[row]), float(learning_rate_grid[col])


# --------------------------------------------------------------------------- #
# Internal: the three CV uses
# --------------------------------------------------------------------------- #
def _param_grid(cfg: _Config) -> dict[str, list[int] | list[float]]:
    # max_depth must stay int (XGBoost rejects a float); learning_rate is float.
    return {
        "max_depth": [int(v) for v in cfg.max_depth_grid],
        "learning_rate": [float(v) for v in cfg.learning_rate_grid],
    }


def _nested_performance(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    cfg: _Config,
) -> tuple[list[FoldPrediction], float, float, float, float]:
    """Tune-in-fold repeated stratified CV -> per-fold ROC input + AUC/balacc/AP."""
    outer = _make_cv(cfg.n_splits, cfg.n_repeats, groups is not None, cfg.random_state)
    param_grid = _param_grid(cfg)
    inner = StratifiedKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state
    )
    folds: list[FoldPrediction] = []
    aucs: list[float] = []
    accs: list[float] = []
    aps: list[float] = []
    for train, test in _split(outer, x, y, groups):
        spw = _scale_pos_weight(y[train])
        base = _make_xgb(
            cfg.max_depth_grid[0], cfg.learning_rate_grid[0], spw, cfg, n_jobs=1
        )
        search = GridSearchCV(
            base, param_grid, cv=inner, scoring=cfg.tuning_metric, n_jobs=cfg.n_jobs
        )
        search.fit(x[train], y[train])
        prob = np.asarray(search.predict_proba(x[test]), dtype=float)[:, 1]
        folds.append(FoldPrediction(y_true=y[test].copy(), y_prob=prob))
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
    x: np.ndarray, y: np.ndarray, cfg: _Config
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Tune (max_depth, learning_rate) on all data + refit -> best + importance/grid."""
    param_grid = _param_grid(cfg)
    inner = StratifiedKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state
    )
    spw = _scale_pos_weight(y)
    base = _make_xgb(
        cfg.max_depth_grid[0], cfg.learning_rate_grid[0], spw, cfg, n_jobs=1
    )
    search = GridSearchCV(
        base, param_grid, cv=inner, scoring=cfg.tuning_metric, n_jobs=cfg.n_jobs
    )
    search.fit(x, y)
    grid_scores = np.asarray(
        search.cv_results_["mean_test_score"], dtype=float
    ).reshape(len(cfg.max_depth_grid), len(cfg.learning_rate_grid))
    best_depth, best_lr = _select_cell(
        grid_scores, cfg.max_depth_grid, cfg.learning_rate_grid, cfg.select
    )
    final = _make_xgb(best_depth, best_lr, spw, cfg, n_jobs=cfg.n_jobs)
    final.fit(x, y)
    return best_depth, best_lr, _importances(final), grid_scores


def _stability(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_depth: int,
    best_lr: float,
    cfg: _Config,
) -> np.ndarray:
    """Per-resample gain importances at fixed hyperparameters (fold-local weighting)."""
    cv = _make_cv(
        cfg.n_splits, cfg.stability_repeats, groups is not None, cfg.random_state
    )
    imps: list[np.ndarray] = []
    for train, _ in _split(cv, x, y, groups):
        spw = _scale_pos_weight(y[train])
        model = _make_xgb(best_depth, best_lr, spw, cfg, n_jobs=cfg.n_jobs)
        model.fit(x[train], y[train])
        imps.append(_importances(model))
    return np.asarray(imps, dtype=float)


def _fixed_cv_auc(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_depth: int,
    best_lr: float,
    cfg: _Config,
) -> float:
    """Mean AUC from repeated K-fold at FIXED hyperparameters (observed + each null)."""
    cv = _make_cv(cfg.n_splits, cfg.null_repeats, groups is not None, cfg.random_state)
    aucs: list[float] = []
    for train, test in _split(cv, x, y, groups):
        spw = _scale_pos_weight(y[train])
        model = _make_xgb(best_depth, best_lr, spw, cfg, n_jobs=cfg.n_jobs)
        model.fit(x[train], y[train])
        prob = np.asarray(model.predict_proba(x[test]), dtype=float)[:, 1]
        aucs.append(float(roc_auc_score(y[test], prob)))
    return float(np.mean(aucs))


def _permute_labels(
    y: np.ndarray, groups: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle labels — at group level when grouped (keeps a unit's label intact)."""
    if groups is None:
        return np.asarray(rng.permutation(y), dtype=int)
    units, inverse = np.unique(groups, return_inverse=True)
    unit_label = np.array(
        [round(float(y[groups == u].mean())) for u in units], dtype=int
    )
    shuffled = rng.permutation(unit_label)
    return np.asarray(shuffled[inverse], dtype=int)


def _null_distribution(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    best_depth: int,
    best_lr: float,
    cfg: _Config,
) -> tuple[np.ndarray, float, float]:
    """Label-shuffle null (fixed hyperparameters) -> null_aucs, observed, p."""
    observed = _fixed_cv_auc(x, y, groups, best_depth, best_lr, cfg)
    rng = np.random.default_rng(cfg.random_state)
    nulls = np.array(
        [
            _fixed_cv_auc(
                x, _permute_labels(y, groups, rng), groups, best_depth, best_lr, cfg
            )
            for _ in range(cfg.n_permutations)
        ],
        dtype=float,
    )
    p = float((np.sum(nulls >= observed) + 1) / (cfg.n_permutations + 1))
    return nulls, observed, p


# --------------------------------------------------------------------------- #
# Internal: importance table (unsigned — the tree analogue of the coefficient table)
# --------------------------------------------------------------------------- #
def _importance_table(
    final_imp: np.ndarray, resample_imp: np.ndarray, feature_names: np.ndarray
) -> pd.DataFrame:
    """Assemble the per-feature table for features with non-zero final-model gain.

    Unsigned throughout — a tree importance is a magnitude, so there is no sign
    consistency (unlike the elastic-net coefficient table).
    """
    selected = final_imp > _ZERO_IMPORTANCE
    nonzero = resample_imp > _ZERO_IMPORTANCE
    n_resample = resample_imp.shape[0]
    sel_freq = nonzero.mean(axis=0)
    masked = np.where(nonzero, resample_imp, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=0)
        q25 = np.nanpercentile(masked, 25, axis=0)
        q75 = np.nanpercentile(masked, 75, axis=0)
    idx = np.where(selected)[0]
    table = pd.DataFrame(
        {
            "feature": feature_names[idx],
            "importance": final_imp[idx],
            "selection_frequency": sel_freq[idx],
            "importance_median": median[idx],
            "importance_q25": q25[idx],
            "importance_q75": q75[idx],
            "n_resamples": n_resample,
        }
    )
    return table.sort_values("importance", ascending=False, kind="stable").reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- #
# Internal: resolved configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Config:
    max_depth_grid: tuple[int, ...]
    learning_rate_grid: tuple[float, ...]
    n_estimators: int
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    reg_alpha: float
    reg_lambda: float
    gamma: float
    importance_type: str
    select: Selection
    tuning_metric: str
    n_splits: int
    n_repeats: int
    stability_repeats: int
    null_repeats: int
    n_permutations: int
    n_jobs: int
    random_state: int


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def classify_xgboost(
    dataset: Dataset,
    outcome: str,
    *,
    binarize: BinarizeSpec | None = None,
    positive_class: str | None = None,
    groups: str | None = None,
    generalization_target: GeneralizationTarget = "samples",
    max_depth_grid: Sequence[int] = DEFAULT_MAX_DEPTH_GRID,
    learning_rate_grid: Sequence[float] = DEFAULT_LEARNING_RATE_GRID,
    n_estimators: int = 300,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    min_child_weight: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    importance_type: str = "gain",
    select: Selection = "best",
    tuning_metric: str = "roc_auc",
    n_splits: int = 5,
    n_repeats: int = 5,
    stability_repeats: int = 10,
    run_null: bool = False,
    n_permutations: int = 1000,
    null_repeats: int = 3,
    n_jobs: int = -1,
    random_state: int = 0,
) -> XGBClassificationResult:
    """Fit a leakage-safe gradient-boosted-tree classifier; report the three parts.

    Parameters
    ----------
    dataset:
        The **experimental subset**, missing values resolved upstream (``NaN`` raises).
        Constant/all-zero features are dropped. Unlike the elastic-net classifier there
        is no scale requirement — trees are invariant to monotone rescaling.
    outcome:
        The metadata column that defines the class. An already-binary categorical column
        is used directly; a continuous or >2-level column needs ``binarize``.
    binarize:
        The rule reducing a non-binary outcome to two classes: :class:`Threshold` (cut a
        continuous column) or :class:`LevelMap` (assign categorical levels; unlisted
        are dropped). Required unless ``outcome`` already has exactly two classes.
    positive_class:
        For a binary categorical outcome, which level is class 1. Default is the
        sorted-second level.
    groups:
        Metadata column naming the independent unit (subject/animal/batch). Used for
        group-aware CV **only if it has repeats**; all-singletons -> row-level CV.
    generalization_target:
        The performance question — ``"samples"``, ``"individuals"``, or ``"batches"``.
        Recorded; report performance as "on unseen <target>". With no repeats, unseen
        sample and unseen individual coincide.
    max_depth_grid, learning_rate_grid:
        The 2-D tuning grid (tree depth x step size). The remaining XGBoost knobs
        (``n_estimators``, ``subsample``, ``colsample_bytree``, ``min_child_weight``,
        ``reg_alpha``, ``reg_lambda``, ``gamma``) are held-constant scalar arguments so
        the search stays 2-D (tractable nested CV + a 2-D heatmap). Widen the grid if a
        study warrants it.
    importance_type:
        The XGBoost feature-importance measure (``"gain"`` default; also ``"weight"``,
        ``"cover"``, ``"total_gain"``, ``"total_cover"``).
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
        importances and enables a ``validated`` finding). Opt-in because it is the main
        compute cost — and trees are heavier per fit than the elastic-net model, so
        consider fewer ``n_permutations`` / ``n_estimators`` for a first pass. If
        ``False``, ``null_*`` are ``None`` and the finding is capped at ``exploratory``.
    n_permutations, null_repeats:
        Number of label permutations, and the (lighter) fixed-hyperparameter CV repeats
        per permutation.
    n_jobs:
        Parallelism for the inner grid search and per-fit tree building (``-1`` = all
        cores).
    random_state:
        Recorded seed for every stochastic step.

    Returns
    -------
    XGBClassificationResult
    """
    depth_grid_t = tuple(int(v) for v in max_depth_grid)
    lr_grid_t = tuple(float(v) for v in learning_rate_grid)
    if not depth_grid_t or not lr_grid_t:
        raise ValueError("max_depth_grid and learning_rate_grid must be non-empty.")
    if any(v < 1 for v in depth_grid_t):
        raise ValueError(f"max_depth_grid values must be >= 1; got {depth_grid_t}.")
    if any(v <= 0.0 for v in lr_grid_t):
        raise ValueError(f"learning_rate_grid values must be > 0; got {lr_grid_t}.")

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
            "impute (native NaN routing is deliberately unused for auditability). "
            "Resolve missingness before classification."
        )
    # No scale check: trees are invariant to per-feature monotone rescaling, so log-vs-
    # linear scale does not affect the model (unlike the elastic-net classifier).

    labels = _resolve_labels(metadata[outcome], binarize, positive_class, outcome)
    keep = labels.keep
    y = labels.y
    n_dropped_unassigned = int(n_samples - int(keep.sum()))
    x_kept = abundances[keep, :]

    feature_names = np.asarray(dataset.feature_names)
    non_constant = np.std(x_kept, axis=0) > 0.0
    n_dropped_constant = int((~non_constant).sum())
    x = x_kept[:, non_constant]
    kept_features = feature_names[non_constant]
    if x.shape[1] == 0:
        raise ValueError("No non-constant features remain after dropping constants.")

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"Need >=2 samples per class; got positive={n_pos}, negative={n_neg}."
        )
    if min(n_pos, n_neg) < n_splits:
        raise ValueError(
            f"Each class needs >= n_splits ({n_splits}) samples for stratified folds; "
            f"got positive={n_pos}, negative={n_neg}. Lower n_splits to proceed."
        )

    groups_kept = _resolve_grouping(groups, metadata, keep)
    grouped = groups_kept is not None

    cfg = _Config(
        max_depth_grid=depth_grid_t,
        learning_rate_grid=lr_grid_t,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        importance_type=importance_type,
        select=select,
        tuning_metric=tuning_metric,
        n_splits=n_splits,
        n_repeats=n_repeats,
        stability_repeats=stability_repeats,
        null_repeats=null_repeats,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    folds, cv_auc, cv_auc_sd, cv_acc, cv_ap = _nested_performance(
        x, y, groups_kept, cfg
    )
    best_depth, best_lr, final_imp, grid_scores = _all_data_fit(x, y, cfg)
    resample_imp = _stability(x, y, groups_kept, best_depth, best_lr, cfg)
    imp_table = _importance_table(final_imp, resample_imp, kept_features)

    null_aucs: np.ndarray | None = None
    observed_auc: float | None = None
    null_p: float | None = None
    if run_null:
        null_aucs, observed_auc, null_p = _null_distribution(
            x, y, groups_kept, best_depth, best_lr, cfg
        )

    return XGBClassificationResult(
        importances=imp_table,
        fold_predictions=folds,
        cv_auc=cv_auc,
        cv_auc_sd=cv_auc_sd,
        cv_balanced_accuracy=cv_acc,
        cv_average_precision=cv_ap,
        best_params={"max_depth": float(best_depth), "learning_rate": best_lr},
        grid_scores=grid_scores,
        max_depth_grid=depth_grid_t,
        learning_rate_grid=lr_grid_t,
        importance_type=importance_type,
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
        feature_names=kept_features,
    )
