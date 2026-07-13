"""All-relevant feature selection with Boruta.

TEMPLATE (lib/) — a *seed* for a project's Boruta script, not a finished analysis. Copy
it into the project's ``scripts/`` and adapt the call site per study (which metadata
column is the target, whether it is a regression or classification target). Held to the
correctness charter (conventions/correctness.md) and the statistics convention
(conventions/statistics.md): **assume nothing, verify everything, fail loud.**

**What this answers.** *Which features are all jointly relevant?* — Boruta's
**all-relevant** selection, the complement to the elastic-net classifier's
**minimal-optimal** set (FEATURE_FINDING.md §B). Real features compete against **shadow
(permuted-copy) features** over a Random Forest; across iterations each feature is
called **Confirmed / Tentative / Rejected** by whether its importance beats the best
shadow (binomial test with internal Bonferroni). Where an L1-penalized classifier keeps
one of a correlated cluster and zeros its neighbours, Boruta confirms **every** feature
carrying signal — so a whole co-regulated pathway is recovered, not just one of it.

**Leakage posture (why this template is simpler than the classifier).** Boruta's shadow
features **are a built-in null**, so it needs no external label-shuffle null. And its
output is a *selection*, not a held-out performance estimate — so it legitimately runs
**once on the whole experimental matrix**, with no nested CV, no fold machinery, and no
generalization-target claim (conventions/statistics.md §B). The in-fold-preprocessing
rule binds any *downstream performance claim* on the selected set, not this selection
run (the canonical wiring even feeds Boruta the ComBat-corrected matrix — a legitimate
use of ComBat output, since this is selection/ML, not a per-feature significance test).

**Task.** Inferred from the ``target`` column dtype — numeric → **regression**
(``RandomForestRegressor``); categorical → **classification**
(``RandomForestClassifier(class_weight="balanced")``, multiclass-native). A numeric
column with very few distinct values warns (the ``dose_rate = 0/1`` miscoded-factor
trap); pass ``task=`` to override. Samples whose target is missing are dropped, counted.

Scale / missing / sample set: pass the **experimental subset** on a **log2-like** scale
with **missing values resolved upstream** (Stage-2). This template **never silently
imputes or zeros** — it **raises** on any ``NaN`` — warns on a non-log scale, and drops
constant / all-zero features (they carry no signal). The all-relevant evidence is a
**decision + importance + shadow-null**, not a significance statistic — there is no
per-feature q (conventions/findings.md §2.3).

**Tuning the selection (when the defaults land badly).** The default run
(``max_iter=100``, ``alpha=0.05``, ``perc=100``) is a starting point, not a fixed
contract — a few knobs trade sensitivity against specificity, and adjusting them is a
legitimate, *recorded* choice (put every non-default into ``provenance.params`` and the
cache fingerprint, since it changes the selection):

- **Too few features confirmed** → give the selection more room: raise ``max_iter``
  (try ``200``) so borderline **Tentatives** get more iterations to resolve into
  **Confirmed**, and/or **lower** ``perc`` toward ``95`` or ``90`` — the shadow
  threshold is ``percentile(imp_sha, perc)``, so a lower percentile is a more lenient
  noise floor that more real features clear.
- **Too many features for a tractable downstream analysis** → tighten the accept/reject
  test: lower ``alpha`` to ``0.01`` (from the ``0.05`` default), a stricter bar that
  confirms fewer, higher-confidence features.

**Caching.** Boruta is minutes-slow on a real scope, so a project driver will usually
cache the result. This template deliberately ships **no** save/load: caching is a
project-local concern (the file-format convention reserves pickle for non-LLM consumers,
and a pickle of an arbitrary result is an ``S301`` smell). Cache in the project script.

Requires the ``boruta`` package (``BorutaPy``) and scikit-learn.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from boruta import BorutaPy
from common.data_loading import Dataset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

Task = Literal["regression", "classification"]

# Scales on which the input is a genuine log abundance. A Random Forest is per-feature
# scale-tolerant (it splits on ranks), so Boruta *runs* on any scale, but the canonical
# pipeline is log; outside this set the selection warns (see _check_scale).
_LOG2_LIKE: frozenset[str] = frozenset({"log2", "glog2", "log10", "ln", "zscore"})

# A numeric target with at most this many distinct values, run as regression, is suspect
# — the classic ``dose_rate = 0, 1`` miscoded-factor trap. Warn so the scientist can
# pass task="classification" if it is really a class label.
_LOW_CARDINALITY_NUMERIC = 5

# A Random Forest needs a handful of samples to bootstrap meaningfully; below this the
# selection is degenerate. (Small-n is otherwise the scientist's call — the stats-review
# flags an underpowered run as exploratory; this floor only catches a mistake.)
_MIN_SAMPLES = 5

# Boruta defaults (mirroring the source oracle). All exposed as arguments.
_DEFAULT_MAX_ITER = 100
_DEFAULT_ALPHA = 0.05
_DEFAULT_PERC = 100.0

__script_meta__: dict[str, object] = {
    "template": {"name": "boruta", "version": "0.1"},
    "kind": "analysis",
    "provides": [
        "Task",
        "BorutaScaleWarning",
        "LowCardinalityTargetWarning",
        "BorutaShadowHistoryError",
        "BorutaResult",
        "boruta_select",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "All-relevant feature selection with Boruta over a Dataset — the complement to "
        "the elastic-net classifier's minimal-optimal set. Real features compete "
        "against shadow (permuted) features over a Random Forest; each is Confirmed / "
        "Tentative / Rejected (the shadows are a built-in null, so no external null "
        "and no CV — one run on the whole experimental matrix). Task inferred from the "
        "target dtype (numeric -> regression; categorical -> classification, "
        "class_weight balanced, multiclass-native), with a low-cardinality-numeric "
        "warning and an override; missing-target samples dropped. Warns on a non-log "
        "scale, raises on NaN abundances (missing handling is upstream), drops "
        "constant features. Evidence is a decision + importance + shadow-null (no "
        "per-feature q). Ships no cache "
        "(project-local). Requires boruta (BorutaPy) and scikit-learn. Study-agnostic; "
        "fail-loud."
    ),
}


# --------------------------------------------------------------------------- #
# Warnings / errors
# --------------------------------------------------------------------------- #
class BorutaScaleWarning(UserWarning):
    """The abundances are not on a log-like scale (Boruta runs; log is canonical)."""


class LowCardinalityTargetWarning(UserWarning):
    """A numeric target with very few distinct values is used as a regression target.

    The likely cause is a class label encoded as integers (``dose_rate = 0, 1``). Pass
    ``task="classification"`` to treat it as classes, or confirm it is continuous.
    """


class BorutaShadowHistoryError(RuntimeError):
    """The per-iteration shadow-threshold capture did not line up with BorutaPy output.

    Raised (fail-loud) when the private-method interception this template relies on
    stops matching ``importance_history_`` — e.g. after a BorutaPy internals change. It
    isolates the one fragile dependency so a silent mismatch never reaches a finding.
    """


# --------------------------------------------------------------------------- #
# BorutaPy subclass — capture the per-iteration shadow threshold
# --------------------------------------------------------------------------- #
class _BorutaWithShadowHistory(BorutaPy):  # type: ignore[misc]
    """BorutaPy that also records the per-iteration shadow threshold it discards.

    Upstream BorutaPy computes ``percentile(imp_sha, perc)`` each iteration to make its
    accept/reject decisions but keeps it only in a local variable. We intercept the
    private ``_add_shadows_get_imps`` to capture the *same* value into
    ``sha_max_history_`` (one entry per iteration, aligned to
    ``importance_history_[1:]``) — the noise floor the plot draws as its line. Depends
    on a private API of a lightly-maintained package; the alignment is verified
    fail-loud in :func:`boruta_select`.
    """

    def _add_shadows_get_imps(
        self, x: np.ndarray, y: np.ndarray, dec_reg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        imp_real, imp_sha = super()._add_shadows_get_imps(x, y, dec_reg)
        self._sha_hist.append(float(np.percentile(imp_sha, self.perc)))
        return imp_real, imp_sha

    def fit(self, x: np.ndarray, y: np.ndarray) -> _BorutaWithShadowHistory:
        self._sha_hist: list[float] = []
        super().fit(x, y)
        self.sha_max_history_ = np.asarray(self._sha_hist, dtype=float)
        del self._sha_hist
        return self


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BorutaResult:
    """Everything the importance figure and the finding read.

    Attributes
    ----------
    feature_names:
        ``(n_features,)`` the analyzed features (after dropping constant ones), in input
        column order.
    decision:
        ``(n_features,)`` object array of ``"Confirmed"`` / ``"Tentative"`` /
        ``"Rejected"`` (aligned to ``feature_names``).
    ranking:
        ``(n_features,)`` int Boruta ranking (1 = best; all Confirmed share rank 1).
    importance:
        ``(n_features,)`` per-feature **median** importance across the iterations it
        participated in (``nanmedian`` over ``importance_history``; 0 if never scored).
    importance_history:
        ``(n_iter, n_features)`` per-iteration real-feature importances; ``NaN`` once a
        feature has been decided (and thus excluded) at that iteration.
    shadow_max_history:
        ``(n_iter,)`` the per-iteration shadow threshold
        (``percentile(imp_sha, perc)``) — the floor a real feature must beat for a hit.
    shadow_threshold:
        ``median(shadow_max_history)`` — the single reference line the plot draws (also
        BorutaPy's own tentative-resolution threshold).
    target, task:
        The resolved target column and whether it ran as regression / classification.
    classes:
        The class labels for a classification run (``None`` for regression), for
        provenance.
    n_confirmed, n_tentative, n_rejected:
        Decision counts (``n_confirmed + n_tentative + n_rejected == n_features``).
    n_iter:
        Number of Boruta iterations actually run.
    n_samples, n_features:
        Analyzed counts after dropping missing-target samples and constant features.
    n_dropped_constant, n_dropped_missing_target:
        For ``provenance.params`` — features dropped as constant, and samples dropped
        because their target was missing.
    max_iter, alpha, perc, random_state:
        The recorded run parameters.
    """

    feature_names: np.ndarray
    decision: np.ndarray
    ranking: np.ndarray
    importance: np.ndarray
    importance_history: np.ndarray
    shadow_max_history: np.ndarray
    shadow_threshold: float
    target: str
    task: Task
    classes: tuple[str, ...] | None
    n_confirmed: int
    n_tentative: int
    n_rejected: int
    n_iter: int
    n_samples: int
    n_features: int
    n_dropped_constant: int
    n_dropped_missing_target: int
    max_iter: int
    alpha: float
    perc: float
    random_state: int

    @property
    def table(self) -> pd.DataFrame:
        """Per-feature results table sorted ranking↑ then importance↓ (the finding art).

        Columns: ``feature``, ``decision``, ``ranking``, ``importance``. Confirmed
        features (rank 1, highest importance) sort to the top; this is the *selection*
        evidence (no per-feature q — conventions/findings.md §2.3).
        """
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "decision": self.decision,
                "ranking": self.ranking,
                "importance": self.importance,
            }
        )
        return df.sort_values(
            ["ranking", "importance"], ascending=[True, False], kind="stable"
        ).reset_index(drop=True)

    @property
    def confirmed_features(self) -> np.ndarray:
        """The Confirmed feature names, in importance-descending order."""
        table = self.table
        confirmed = table.loc[table["decision"] == "Confirmed", "feature"].to_numpy()
        return np.asarray(confirmed)


# --------------------------------------------------------------------------- #
# Internal: task + target resolution
# --------------------------------------------------------------------------- #
def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series))


def _resolve_task(series: pd.Series, task: Task | None, target: str) -> Task:
    """Resolve regression vs classification from an override or the target dtype."""
    if task is not None:
        if task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification'; got {task!r}."
            )
        return task
    if _is_numeric(series):
        finite = series.to_numpy(dtype=float)
        n_distinct = int(np.unique(finite[np.isfinite(finite)]).size)
        # Warn only for a genuine low-cardinality factor (2..threshold); a constant
        # (<= 1 distinct) is not a miscoded factor and is caught downstream as constant.
        if 1 < n_distinct <= _LOW_CARDINALITY_NUMERIC:
            warnings.warn(
                f"Numeric target {target!r} has only {n_distinct} distinct values and "
                f"is being run as regression. If it is really a class label (e.g. a "
                f"group coded 0/1), pass task='classification'.",
                LowCardinalityTargetWarning,
                stacklevel=3,
            )
        return "regression"
    return "classification"


@dataclass(frozen=True)
class _Target:
    y: np.ndarray  # (n_kept,)
    keep: np.ndarray  # (n_samples,) bool
    classes: tuple[str, ...] | None


def _resolve_target(series: pd.Series, task: Task, target: str) -> _Target:
    """Reduce the target to a fit-ready ``y`` + a keep mask (missing → dropped)."""
    if task == "regression":
        values = series.to_numpy(dtype=float)
        keep = np.isfinite(values)
        y = values[keep]
        if int(np.unique(y).size) < 2:
            raise ValueError(
                f"Regression target {target!r} is constant over the non-missing "
                f"samples; it carries no signal to select against."
            )
        return _Target(y=y, keep=keep, classes=None)

    # Detect missing via isna() before stringifying — pandas 3.0's astype(str) keeps NaN
    # as missing, not the literal "nan", so a string comparison would not catch it.
    keep = ~series.isna().to_numpy()
    y = series[keep].astype(str).to_numpy()
    values, counts = np.unique(y, return_counts=True)
    classes = tuple(str(v) for v in values)
    if len(classes) < 2:
        raise ValueError(
            f"Classification target {target!r} has {len(classes)} class(es) over the "
            f"non-missing samples; need at least 2."
        )
    if int(counts.min()) < 2:
        thin = {c: int(n) for c, n in zip(classes, counts, strict=True) if n < 2}
        raise ValueError(
            f"Classification target {target!r} has class(es) with < 2 samples {thin}; "
            f"each class needs at least 2."
        )
    return _Target(y=y, keep=keep, classes=classes)


def _check_scale(scale: str) -> None:
    if scale not in _LOG2_LIKE:
        warnings.warn(
            f"Boruta on scale {scale!r}: a Random Forest is per-feature "
            f"scale-tolerant, but the canonical pipeline is log. Run on log data "
            f"unless you have a specific reason not to.",
            BorutaScaleWarning,
            stacklevel=3,
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def boruta_select(
    dataset: Dataset,
    target: str,
    *,
    task: Task | None = None,
    n_estimators: int | Literal["auto"] = "auto",
    max_iter: int = _DEFAULT_MAX_ITER,
    alpha: float = _DEFAULT_ALPHA,
    perc: float = _DEFAULT_PERC,
    two_step: bool = True,
    rf_n_jobs: int = -1,
    random_state: int = 0,
) -> BorutaResult:
    """Run Boruta all-relevant selection for one target over a :class:`Dataset`.

    Parameters
    ----------
    dataset:
        The **experimental subset** on a **log2-like** scale, missing values resolved
        upstream (a ``NaN`` raises). Constant/all-zero features are dropped.
    target:
        The metadata column to select features against. A numeric column runs as
        regression, a categorical column as (multiclass-capable) classification;
        override with ``task=``.
    task:
        ``"regression"`` or ``"classification"``. ``None`` (default) infers from the
        target dtype (with a warning on a low-cardinality numeric column).
    n_estimators:
        Boruta's internal Random Forest size — an int or ``"auto"`` (BorutaPy grows it
        with the surviving feature count; the default).
    max_iter:
        Maximum Boruta iterations (default 100). Raise to ~200 when too few features
        Confirm / many stay Tentative — more iterations resolve borderline calls.
    alpha:
        Significance level for the per-feature accept/reject binomial test (def. 0.05).
        Lower to 0.01 to confirm fewer, higher-confidence features when the Confirmed
        set is too large for a tractable downstream analysis.
    perc:
        Shadow-importance percentile that sets the threshold (default 100 = the max
        shadow importance; lower is more lenient). Lower toward 95/90 to confirm more
        features when too few clear the default max-shadow floor.
    two_step:
        BorutaPy's two-step (Bonferroni then BH-style) correction (default ``True``).
    rf_n_jobs:
        Parallel jobs for the internal Random Forest (``-1`` = all cores).
    random_state:
        Recorded seed for the forest and the shadow permutations.

    Returns
    -------
    BorutaResult
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if not 0.0 < perc <= 100.0:
        raise ValueError(f"perc must be in (0, 100]; got {perc}.")

    metadata = dataset.metadata
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(f"abundances must be 2D; got shape {abundances.shape}.")
    n_samples, _ = abundances.shape
    if len(metadata) != n_samples:
        raise ValueError(
            f"metadata has {len(metadata)} rows but abundances has {n_samples} samples."
        )
    if target not in metadata.columns:
        raise ValueError(f"target column {target!r} not in metadata.")
    if not np.all(np.isfinite(abundances)):
        raise ValueError(
            "abundances contain NaN/inf. Missing-value handling is an upstream Stage-2 "
            "decision (conventions/statistics.md); this template does not silently "
            "impute. Resolve missingness before running Boruta."
        )
    _check_scale(dataset.scale)

    resolved_task = _resolve_task(metadata[target], task, target)
    tgt = _resolve_target(metadata[target], resolved_task, target)
    keep = tgt.keep
    n_dropped_missing = int(n_samples - int(keep.sum()))

    x_kept = abundances[keep, :]
    if x_kept.shape[0] < _MIN_SAMPLES:
        raise ValueError(
            f"Only {x_kept.shape[0]} samples remain after dropping missing-target "
            f"samples; Boruta needs at least {_MIN_SAMPLES}."
        )

    feature_names = np.asarray(dataset.feature_names)
    non_constant = np.std(x_kept, axis=0) > 0.0
    n_dropped_constant = int((~non_constant).sum())
    x = x_kept[:, non_constant]
    kept_features = feature_names[non_constant]
    if x.shape[1] == 0:
        raise ValueError("No non-constant features remain after dropping constants.")

    estimator = _build_estimator(resolved_task, rf_n_jobs, random_state)
    boruta = _BorutaWithShadowHistory(
        estimator=estimator,
        n_estimators=n_estimators,
        max_iter=max_iter,
        alpha=alpha,
        perc=perc,
        two_step=two_step,
        random_state=random_state,
        verbose=0,
    )
    with warnings.catch_warnings():
        # BorutaPy's internal RF fits are noisy (deprecation / convergence chatter from
        # a lightly-maintained package); our own scale/target warnings fired above.
        warnings.simplefilter("ignore")
        boruta.fit(x, tgt.y)

    return _assemble_result(
        boruta=boruta,
        kept_features=kept_features,
        target=target,
        task=resolved_task,
        classes=tgt.classes,
        n_samples=int(x.shape[0]),
        n_dropped_constant=n_dropped_constant,
        n_dropped_missing=n_dropped_missing,
        max_iter=max_iter,
        alpha=alpha,
        perc=perc,
        random_state=random_state,
    )


def _build_estimator(task: Task, rf_n_jobs: int, random_state: int) -> object:
    """The Random Forest Boruta wraps — balanced classes for a classification target."""
    if task == "regression":
        return RandomForestRegressor(random_state=random_state, n_jobs=rf_n_jobs)
    return RandomForestClassifier(
        random_state=random_state, class_weight="balanced", n_jobs=rf_n_jobs
    )


def _assemble_result(
    *,
    boruta: _BorutaWithShadowHistory,
    kept_features: np.ndarray,
    target: str,
    task: Task,
    classes: tuple[str, ...] | None,
    n_samples: int,
    n_dropped_constant: int,
    n_dropped_missing: int,
    max_iter: int,
    alpha: float,
    perc: float,
    random_state: int,
) -> BorutaResult:
    """Read the fitted BorutaPy state into a validated :class:`BorutaResult`."""
    n_features = int(kept_features.size)
    decision = np.full(n_features, "Rejected", dtype=object)
    decision[np.asarray(boruta.support_, dtype=bool)] = "Confirmed"
    decision[np.asarray(boruta.support_weak_, dtype=bool)] = "Tentative"
    ranking = np.asarray(boruta.ranking_, dtype=int)

    # importance_history_ is (n_iter + 1, n_features): an initial zero row + one per
    # iteration. Strip the zero row so it aligns with the per-iteration shadow history.
    importance_history = np.asarray(boruta.importance_history_, dtype=float)[1:]
    shadow_max_history = np.asarray(boruta.sha_max_history_, dtype=float)

    # Fail-loud isolation of the private-API dependency: the captured shadow threshold
    # must have exactly one entry per stripped importance-history row (guaranteed by
    # BorutaPy 0.4.3's _fit loop; a mismatch means the internals moved under us).
    if shadow_max_history.shape[0] != importance_history.shape[0]:
        raise BorutaShadowHistoryError(
            f"Shadow-threshold capture ({shadow_max_history.shape[0]} entries) does "
            f"not align with importance history ({importance_history.shape[0]} "
            f"iterations); the BorutaPy internals this template intercepts changed."
        )
    if shadow_max_history.size == 0:
        raise BorutaShadowHistoryError(
            "Boruta ran zero iterations, so there is no shadow history; cannot select."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        importance = np.nanmedian(importance_history, axis=0)
    importance = np.where(np.isnan(importance), 0.0, importance)

    n_confirmed = int(np.sum(decision == "Confirmed"))
    n_tentative = int(np.sum(decision == "Tentative"))
    n_rejected = n_features - n_confirmed - n_tentative

    return BorutaResult(
        feature_names=kept_features,
        decision=decision,
        ranking=ranking,
        importance=importance,
        importance_history=importance_history,
        shadow_max_history=shadow_max_history,
        shadow_threshold=float(np.median(shadow_max_history)),
        target=target,
        task=task,
        classes=classes,
        n_confirmed=n_confirmed,
        n_tentative=n_tentative,
        n_rejected=n_rejected,
        n_iter=int(importance_history.shape[0]),
        n_samples=n_samples,
        n_features=n_features,
        n_dropped_constant=n_dropped_constant,
        n_dropped_missing_target=n_dropped_missing,
        max_iter=max_iter,
        alpha=alpha,
        perc=perc,
        random_state=random_state,
    )
