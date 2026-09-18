"""Paired per-fold comparison of two classifier results evaluated on identical folds.

TEMPLATE (lib/) — a *seed* for a project's classifier-comparison script, not a finished
analysis. Copy it into the project's ``scripts/`` and adapt the call site per study
(which two cached results, which pre-stated reading). Held to the correctness charter
(conventions/correctness.md) and the statistics convention (conventions/statistics.md):
assume nothing, verify everything, fail loud.

**What this answers.** *On the same held-out samples, does the alternative classifier
rank them better than the reference?* — the paired evaluation the linear-classifier
selection rule requires (conventions/statistics.md, *Choosing among the linear
classifiers*). The four classifier templates (``classification`` elastic net,
``classification-svm``, ``classification-lda``, ``classification-xgboost``) record
**fold identity** (``repeat`` / ``fold`` / ``test_indices``) on every outer fold and
build their outer CV from byte-identical code, so a same-seed run of two of them is
evaluated on the *same* partitions. This module **verifies** that identity fold for
fold (it never trusts "same seed"), recomputes each fold's ROC AUC from the held-out
scores (it never trusts a reported mean), and reports the paired differences.

**Fold dependence — why a plain paired test over-claims.** The J = ``n_splits`` x
``n_repeats`` per-fold differences are *not* independent: folds within a repeat share
most of their training data, and repeats re-use every sample. A paired t-test or a
Wilcoxon signed-rank over those J values treats them as J independent draws and is
**anti-conservative**. So:

  * the **primary evidence** is descriptive — the per-fold differences themselves and
    the win / loss / tie counts from the alternative's point of view;
  * the **decisive p** is the **corrected resampled t-test** (Nadeau & Bengio 2003):
    ``t = mean(d) / sqrt(var(d) · (1/J + n_test/n_train))``, df = J - 1, whose
    variance term inflates the naive ``var/J`` by the train/test overlap ratio — the
    standard correction for repeated CV comparisons;
  * the **Wilcoxon signed-rank** (on the non-zero differences only, ``None`` below five
    of them) is reported **as indicative only** — it is what a scientist reaches for,
    so it is here, labelled, not silently omitted and not presented as decisive.

**The seed rule.** A single seed's verdict can be a draw of the fold assignment (seed
noise of a few hundredths of AUC at n ≈ 40 is routine). :func:`compare_across_seeds`
runs the paired comparison for several seeds and calls the verdict **only when the
sign of the mean difference is consistent across every seed** — otherwise it is a tie,
whatever any one seed's p said.

**The pre-stated reading.** Two linear classifiers answer different feature questions
(a sparse minimal set vs dense module weights vs a covariance-adjusted direction), and
a reading chosen *after* both feature lists are visible picks the better story. The
comparison therefore takes the ``reading`` the scientist stated **before** the
alternative ran and records it on the result; the stats-reviewer fails a tie settled
post hoc. The tree model (``"non-linear"``) is admitted too: the statistic is about
held-out rankings on identical folds, not about the model family.

Inputs are typed **structurally** (:class:`ComparableResult`, a ``Protocol``) so this
module imports none of the four classifier modules and any future template that
records fold identity pairs the same way. A pre-v0.2 cached result (``repeat == -1``)
carries no fold identity and is refused — re-run it.

Requires scipy (``stats.t``, ``stats.wilcoxon``) and scikit-learn (``roc_auc_score``).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

Reading = Literal["sparse", "module", "covariance-adjusted", "non-linear"]

__script_meta__: dict[str, object] = {
    "template": {"name": "classifier-comparison", "version": "0.1"},
    "kind": "analysis",
    "provides": [
        "Reading",
        "ComparableResult",
        "FoldIdentityError",
        "ReportedAUCWarning",
        "ClassifierComparisonResult",
        "MultiSeedComparisonResult",
        "fold_identity",
        "fold_aucs",
        "corrected_resampled_t",
        "compare_classifiers",
        "compare_across_seeds",
    ],
    "uses": [],
    "seeded_from": None,
    "description": (
        "Paired per-fold comparison of two classifier results (elastic net / SVM / "
        "LDA / XGBoost, any pair) evaluated on identical outer folds: verifies fold "
        "identity fold-for-fold (n_samples, seed, repeat/fold/test_indices, y_true; "
        "refuses a pre-v0.2 cache), recomputes each fold's ROC AUC from the held-out "
        "scores (warns when a reported cv_auc disagrees; never uses it), and reports "
        "the per-fold differences + win/loss/tie counts as the primary evidence, the "
        "corrected resampled t-test (Nadeau & Bengio 2003; the decisive p, fold "
        "dependence corrected) and an indicative-only Wilcoxon on the non-zero "
        "differences. compare_across_seeds calls a verdict only when the sign of the "
        "mean difference is consistent across every seed. Records the pre-stated "
        "reading. Structural (Protocol) inputs; imports no classifier module; "
        "result-io round-trippable. Requires scipy + scikit-learn."
    ),
}

# Reported vs recomputed mean AUC may differ by float accumulation only.
_REPORTED_AUC_TOLERANCE = 1e-6
# Below this many non-zero differences the Wilcoxon signed-rank has no resolution.
_WILCOXON_MIN_NONZERO = 5

_CLASS_LABELS: dict[str, str] = {
    "ClassificationResult": "classification",
    "SVMClassificationResult": "classification-svm",
    "LDAClassificationResult": "classification-lda",
    "XGBClassificationResult": "classification-xgboost",
}


# --------------------------------------------------------------------------- #
# Errors / warnings
# --------------------------------------------------------------------------- #
class FoldIdentityError(ValueError):
    """The two results were not evaluated on the same outer folds (or carry none)."""


class ReportedAUCWarning(UserWarning):
    """A result's reported ``cv_auc`` disagrees with the AUCs recomputed per fold.

    The comparison always uses the recomputed values; the reported one is recorded
    for the record (``*_reported_cv_auc``) and never enters the statistics.
    """


# --------------------------------------------------------------------------- #
# The structural input contract
# --------------------------------------------------------------------------- #
class ComparableResult(Protocol):
    """What a classifier result must expose to be compared (read-only view).

    Satisfied by every classifier template's result dataclass. Members are declared
    as read-only properties so a frozen dataclass field of a narrower type (e.g. a
    ``Literal`` generalization target, a ``list`` of a module's own ``FoldPrediction``)
    satisfies the protocol; nothing is ever written back.
    """

    @property
    def fold_predictions(self) -> Sequence[Any]: ...
    @property
    def cv_auc(self) -> float: ...
    @property
    def random_state(self) -> int: ...
    @property
    def n_samples(self) -> int: ...
    @property
    def outcome(self) -> str: ...
    @property
    def positive_label(self) -> str: ...
    @property
    def negative_label(self) -> str: ...
    @property
    def generalization_target(self) -> str: ...
    @property
    def grouped(self) -> bool: ...


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassifierComparisonResult:
    """One paired comparison (a single seed). Every field is ``result-io`` serializable.

    Attributes
    ----------
    per_fold:
        One row per outer fold: ``repeat``, ``fold``, ``n_train``, ``n_test``,
        ``auc_reference``, ``auc_alternative``, ``diff`` (= alternative - reference),
        ``outcome`` (``win`` / ``loss`` / ``tie`` from the alternative's point of view,
        a tie when ``|diff| <= tie_tolerance``). The primary evidence.
    reference_label, alternative_label, reading:
        Which template each side is (derived from the result class unless given) and
        the reading the scientist stated **before** the alternative ran.
    n_folds, n_wins, n_losses, n_ties, tie_tolerance:
        The win/loss/tie counts and the tolerance that defined a tie.
    mean_diff, sd_diff:
        Mean and sample SD (ddof = 1) of the per-fold differences.
    reference_auc, alternative_auc:
        The **recomputed** mean per-fold AUCs (the numbers to report).
    reference_reported_cv_auc, alternative_reported_cv_auc:
        The ``cv_auc`` each result carried — recorded, never used.
    rho, corrected_t, corrected_t_df, corrected_t_p:
        The corrected resampled t-test (Nadeau & Bengio 2003): the mean test/train
        ratio, the statistic, its df (J - 1) and the two-sided p — the decisive p.
    wilcoxon_p, n_nonzero_diffs:
        Wilcoxon signed-rank over the non-zero differences — **indicative only**
        (fold dependence makes it anti-conservative); ``None`` below five non-zero.
    random_state, n_samples, outcome, positive_label, negative_label,
    generalization_target, grouped:
        The shared design both results were verified to have.
    """

    per_fold: pd.DataFrame
    reference_label: str
    alternative_label: str
    reading: str
    n_folds: int
    n_wins: int
    n_losses: int
    n_ties: int
    tie_tolerance: float
    mean_diff: float
    sd_diff: float
    reference_auc: float
    alternative_auc: float
    reference_reported_cv_auc: float
    alternative_reported_cv_auc: float
    rho: float
    corrected_t: float
    corrected_t_df: int
    corrected_t_p: float
    wilcoxon_p: float | None
    n_nonzero_diffs: int
    random_state: int
    n_samples: int
    outcome: str
    positive_label: str
    negative_label: str
    generalization_target: str
    grouped: bool

    @property
    def preferred(self) -> str:
        """``"alternative"`` / ``"reference"`` / ``"tie"`` on the sign of the mean.

        A tie when ``|mean_diff| <= tie_tolerance``. Descriptive — the sign, not a
        significance call; the seed rule (:func:`compare_across_seeds`) decides.
        """
        if abs(self.mean_diff) <= self.tie_tolerance:
            return "tie"
        return "alternative" if self.mean_diff > 0 else "reference"


@dataclass(frozen=True)
class MultiSeedComparisonResult:
    """The seed rule applied to several paired comparisons of the same two templates.

    Attributes
    ----------
    comparisons, seeds:
        The per-seed :class:`ClassifierComparisonResult` (in the order given) and
        their seeds (each distinct).
    per_seed_mean_diff, per_seed_sign:
        The mean difference per seed and its sign (``+1`` alternative ahead, ``-1``
        reference ahead, ``0`` within the tie tolerance).
    sign_consistent, verdict:
        ``True`` iff every sign is equal and non-zero; the verdict is then that side
        (``"alternative"`` / ``"reference"``), else ``"tie"``.
    reading, reference_label, alternative_label:
        As on each comparison (verified identical across seeds).
    seed_table:
        One row per seed: ``seed``, ``mean_diff``, ``wins``, ``losses``, ``ties``,
        ``corrected_t_p``, ``wilcoxon_p`` (NaN where indicative Wilcoxon was ``None``).
    """

    comparisons: tuple[ClassifierComparisonResult, ...]
    seeds: tuple[int, ...]
    per_seed_mean_diff: tuple[float, ...]
    per_seed_sign: tuple[int, ...]
    sign_consistent: bool
    verdict: str
    reading: str
    reference_label: str
    alternative_label: str
    seed_table: pd.DataFrame


# --------------------------------------------------------------------------- #
# Fold identity and per-fold AUCs
# --------------------------------------------------------------------------- #
def fold_identity(
    result: ComparableResult,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """The ``(repeat, fold, test_indices)`` of every outer fold, in order.

    Raises :class:`FoldIdentityError` when any fold carries ``repeat == -1`` — a
    result cached before the classifier templates recorded fold identity (v0.2); it
    cannot be paired, so re-run it.
    """
    out: list[tuple[int, int, tuple[int, ...]]] = []
    for i, fold in enumerate(result.fold_predictions):
        repeat = int(fold.repeat)
        if repeat == -1:
            raise FoldIdentityError(
                f"fold {i} of {_label_for(result)} carries no fold identity "
                f"(repeat == -1): the result predates fold-identity recording "
                f"(classifier templates < v0.2). Re-run the analysis; a cached result "
                f"without identity cannot be paired."
            )
        out.append((repeat, int(fold.fold), tuple(int(j) for j in fold.test_indices)))
    return tuple(out)


def _fold_scores(fold: Any, index: int) -> np.ndarray:
    score = getattr(fold, "y_prob", None)
    if score is None:
        score = getattr(fold, "y_score", None)
    if score is None:
        raise ValueError(
            f"fold {index} carries neither y_prob nor y_score; cannot compute its AUC."
        )
    return np.asarray(score, dtype=float)


def fold_aucs(result: ComparableResult) -> np.ndarray:
    """ROC AUC recomputed per outer fold from the held-out ``y_true`` and scores.

    Uses ``y_prob`` (probability classifiers) or ``y_score`` (margin / log-odds
    classifiers); AUC is rank-based so the two are directly comparable.
    """
    aucs = [
        float(roc_auc_score(np.asarray(f.y_true, dtype=int), _fold_scores(f, i)))
        for i, f in enumerate(result.fold_predictions)
    ]
    return np.asarray(aucs, dtype=float)


# --------------------------------------------------------------------------- #
# The corrected resampled t-test (Nadeau & Bengio 2003)
# --------------------------------------------------------------------------- #
def corrected_resampled_t(
    diff: np.ndarray | Sequence[float],
    n_train: np.ndarray | Sequence[int],
    n_test: np.ndarray | Sequence[int],
) -> tuple[float, int, float]:
    """``(t, df, p)`` of the corrected resampled t-test over per-fold differences.

    ``rho = mean_j(n_test_j / n_train_j)`` (a per-fold ratio, averaged — with uneven
    folds this is not the ratio of the totals), and
    ``t = mean(d) / sqrt(var(d, ddof=1) · (1/J + rho))`` with ``df = J - 1``; the
    two-sided p is ``2 · t.sf(|t|, df)``. The ``rho`` term is the Nadeau & Bengio
    (2003) inflation of the naive ``var/J`` for the train/test overlap between folds.

    Zero variance (every difference identical) is defined, not NaN: ``t = ±inf`` and
    ``p = 0.0`` when the common difference is non-zero, and ``t = 0.0``, ``p = 1.0``
    when it is zero (the two classifiers ranked every fold identically). Needs at
    least two folds.
    """
    d = np.asarray(diff, dtype=float)
    tr = np.asarray(n_train, dtype=float)
    te = np.asarray(n_test, dtype=float)
    j = int(d.size)
    if j < 2:
        raise ValueError(f"the corrected resampled t needs >= 2 folds; got {j}.")
    if tr.shape != d.shape or te.shape != d.shape:
        raise ValueError(
            f"n_train {tr.shape} / n_test {te.shape} must match diff {d.shape}."
        )
    if not np.all(np.isfinite(d)):
        raise ValueError("per-fold differences must be finite.")
    if np.any(tr <= 0) or np.any(te <= 0):
        raise ValueError("n_train and n_test must be positive for every fold.")
    rho = float(np.mean(te / tr))
    mean = float(np.mean(d))
    var = float(np.var(d, ddof=1))
    df = j - 1
    if var == 0.0:
        if mean == 0.0:
            return 0.0, df, 1.0
        return math.copysign(math.inf, mean), df, 0.0
    t = mean / math.sqrt(var * (1.0 / j + rho))
    p = float(2.0 * stats.t.sf(abs(t), df))
    return t, df, p


# --------------------------------------------------------------------------- #
# The paired comparison
# --------------------------------------------------------------------------- #
def _label_for(result: ComparableResult) -> str:
    name = type(result).__name__
    return _CLASS_LABELS.get(name, name)


def _check_same_design(
    reference: ComparableResult,
    alternative: ComparableResult,
    ref_label: str,
    alt_label: str,
) -> None:
    """Every check the pairing rests on, each with its own message, in order."""
    if reference.n_samples != alternative.n_samples:
        raise FoldIdentityError(
            f"n_samples differ: {ref_label} analyzed {reference.n_samples}, "
            f"{alt_label} analyzed {alternative.n_samples} — the two were not run on "
            f"the same analyzed sample set (same outcome / binarize / feature_list?)."
        )
    if reference.random_state != alternative.random_state:
        raise FoldIdentityError(
            f"random_state differs: {ref_label} used {reference.random_state}, "
            f"{alt_label} used {alternative.random_state} — the outer folds are not "
            f"the same partition; re-run one with the other's seed."
        )
    ref_ident = fold_identity(reference)
    alt_ident = fold_identity(alternative)
    if len(ref_ident) != len(alt_ident):
        raise FoldIdentityError(
            f"fold counts differ: {ref_label} has {len(ref_ident)} outer folds, "
            f"{alt_label} has {len(alt_ident)} (different n_splits / n_repeats?)."
        )
    for i, (a, b) in enumerate(zip(ref_ident, alt_ident, strict=True)):
        if a != b:
            raise FoldIdentityError(
                f"fold {i} differs between {ref_label} (repeat {a[0]}, fold {a[1]}, "
                f"{len(a[2])} held out) and {alt_label} (repeat {b[0]}, fold {b[1]}, "
                f"{len(b[2])} held out): the held-out sample sets are not identical, "
                f"so the comparison would not be paired."
            )
        y_ref = np.asarray(reference.fold_predictions[i].y_true)
        y_alt = np.asarray(alternative.fold_predictions[i].y_true)
        if y_ref.shape != y_alt.shape or not np.array_equal(y_ref, y_alt):
            raise FoldIdentityError(
                f"fold {i}: the held-out labels (y_true) differ between {ref_label} "
                f"and {alt_label} although the held-out indices match — the label "
                f"encodings are not the same (positive_class / binarize rule?)."
            )
    if reference.outcome != alternative.outcome:
        raise FoldIdentityError(
            f"outcome differs: {ref_label} predicts {reference.outcome!r}, "
            f"{alt_label} predicts {alternative.outcome!r}."
        )
    if reference.positive_label != alternative.positive_label:
        raise FoldIdentityError(
            f"positive_label differs: {ref_label} has {reference.positive_label!r}, "
            f"{alt_label} has {alternative.positive_label!r} — the AUCs would be "
            f"oriented toward different classes."
        )


def _recomputed_aucs(result: ComparableResult, label: str) -> np.ndarray:
    aucs = fold_aucs(result)
    reported = float(result.cv_auc)
    recomputed = float(aucs.mean())
    if abs(recomputed - reported) > _REPORTED_AUC_TOLERANCE:
        warnings.warn(
            f"{label}: the reported cv_auc ({reported:.6f}) differs from the mean of "
            f"the per-fold AUCs recomputed from its held-out scores "
            f"({recomputed:.6f}); the recomputed values are used, the reported one is "
            f"only recorded.",
            ReportedAUCWarning,
            stacklevel=3,
        )
    return aucs


def _wilcoxon_indicative(diff: np.ndarray) -> tuple[float | None, int]:
    """Wilcoxon signed-rank on the non-zero differences; ``None`` below five."""
    nonzero = diff[diff != 0.0]
    n = int(nonzero.size)
    if n < _WILCOXON_MIN_NONZERO:
        return None, n
    return float(stats.wilcoxon(nonzero).pvalue), n


def compare_classifiers(
    reference: ComparableResult,
    alternative: ComparableResult,
    *,
    reading: Reading,
    reference_label: str | None = None,
    alternative_label: str | None = None,
    tie_tolerance: float = 1e-9,
) -> ClassifierComparisonResult:
    """Paired per-fold comparison of ``alternative`` against ``reference``.

    Parameters
    ----------
    reference, alternative:
        Two classifier results run with the **same seed on the same analyzed
        samples** (the elastic net is the reference in the selection rule). Fold
        identity is verified fold for fold; any mismatch raises
        :class:`FoldIdentityError` naming what differs.
    reading:
        The feature reading the scientist stated **before** the alternative ran —
        ``"sparse"`` (elastic net's minimal set), ``"module"`` (SVM's dense weights),
        ``"covariance-adjusted"`` (LDA's direction) or ``"non-linear"`` (the tree
        model). Recorded, not acted on: it is the tie-break the reviewer checks was
        pre-stated.
    reference_label, alternative_label:
        Names for the two sides; default to a short name derived from each result
        class (``classification`` / ``classification-svm`` / ``classification-lda`` /
        ``classification-xgboost``).
    tie_tolerance:
        ``|diff| <= tie_tolerance`` counts a fold as a tie (default ``1e-9``: exact
        equality up to float noise).

    Returns
    -------
    ClassifierComparisonResult
        Per-fold differences (``diff = alternative - reference``) + win/loss/tie
        counts, the corrected resampled t (decisive), the indicative Wilcoxon, and
        the recomputed mean AUCs of both sides.
    """
    if tie_tolerance < 0:
        raise ValueError(f"tie_tolerance must be >= 0; got {tie_tolerance}.")
    ref_label = (
        reference_label if reference_label is not None else _label_for(reference)
    )
    alt_label = (
        alternative_label if alternative_label is not None else _label_for(alternative)
    )
    _check_same_design(reference, alternative, ref_label, alt_label)
    n_folds = len(reference.fold_predictions)
    if n_folds < 2:
        raise ValueError(f"a paired comparison needs >= 2 outer folds; got {n_folds}.")

    auc_ref = _recomputed_aucs(reference, ref_label)
    auc_alt = _recomputed_aucs(alternative, alt_label)
    diff = auc_alt - auc_ref
    n_test = np.asarray(
        [len(f.test_indices) for f in reference.fold_predictions], dtype=int
    )
    n_train = int(reference.n_samples) - n_test
    if np.any(n_train <= 0):
        raise FoldIdentityError(
            "a fold holds out every sample (n_train <= 0); test_indices do not fit "
            "n_samples."
        )
    outcome = np.where(
        np.abs(diff) <= tie_tolerance, "tie", np.where(diff > 0, "win", "loss")
    )
    per_fold = pd.DataFrame(
        {
            "repeat": [int(f.repeat) for f in reference.fold_predictions],
            "fold": [int(f.fold) for f in reference.fold_predictions],
            "n_train": n_train.astype(int),
            "n_test": n_test.astype(int),
            "auc_reference": auc_ref,
            "auc_alternative": auc_alt,
            "diff": diff,
            "outcome": [str(o) for o in outcome],
        }
    )
    t, df, p = corrected_resampled_t(diff, n_train, n_test)
    wilcoxon_p, n_nonzero = _wilcoxon_indicative(diff)
    return ClassifierComparisonResult(
        per_fold=per_fold,
        reference_label=ref_label,
        alternative_label=alt_label,
        reading=str(reading),
        n_folds=n_folds,
        n_wins=int((outcome == "win").sum()),
        n_losses=int((outcome == "loss").sum()),
        n_ties=int((outcome == "tie").sum()),
        tie_tolerance=float(tie_tolerance),
        mean_diff=float(diff.mean()),
        sd_diff=float(np.std(diff, ddof=1)),
        reference_auc=float(auc_ref.mean()),
        alternative_auc=float(auc_alt.mean()),
        reference_reported_cv_auc=float(reference.cv_auc),
        alternative_reported_cv_auc=float(alternative.cv_auc),
        rho=float(np.mean(n_test / n_train)),
        corrected_t=t,
        corrected_t_df=df,
        corrected_t_p=p,
        wilcoxon_p=wilcoxon_p,
        n_nonzero_diffs=n_nonzero,
        random_state=int(reference.random_state),
        n_samples=int(reference.n_samples),
        outcome=str(reference.outcome),
        positive_label=str(reference.positive_label),
        negative_label=str(reference.negative_label),
        generalization_target=str(reference.generalization_target),
        grouped=bool(reference.grouped),
    )


# --------------------------------------------------------------------------- #
# The seed rule
# --------------------------------------------------------------------------- #
def compare_across_seeds(
    pairs: Sequence[tuple[ComparableResult, ComparableResult]],
    *,
    reading: Reading,
    reference_label: str | None = None,
    alternative_label: str | None = None,
    tie_tolerance: float = 1e-9,
) -> MultiSeedComparisonResult:
    """Apply the seed rule: a verdict only when the sign agrees across every seed.

    ``pairs`` are ``(reference, alternative)`` result pairs, one per seed (at least
    two, each seed distinct); each goes through :func:`compare_classifiers`. The
    verdict is ``"alternative"`` / ``"reference"`` when every seed's mean difference
    has the same non-zero sign, else ``"tie"`` — a sign flip across seeds means the
    single-seed difference was fold-assignment noise.
    """
    if len(pairs) < 2:
        raise ValueError(
            f"the seed rule needs >= 2 (reference, alternative) pairs at distinct "
            f"seeds; got {len(pairs)}. A single seed cannot show sign consistency."
        )
    comparisons = tuple(
        compare_classifiers(
            ref,
            alt,
            reading=reading,
            reference_label=reference_label,
            alternative_label=alternative_label,
            tie_tolerance=tie_tolerance,
        )
        for ref, alt in pairs
    )
    seeds = tuple(c.random_state for c in comparisons)
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"seeds must be distinct across pairs; got {list(seeds)} — the same seed "
            f"twice re-uses one fold assignment and shows nothing about seed noise."
        )
    labels = {(c.reference_label, c.alternative_label) for c in comparisons}
    if len(labels) != 1:
        raise ValueError(
            f"every pair must compare the same two templates; got labels {labels}."
        )
    means = tuple(c.mean_diff for c in comparisons)
    signs = tuple(0 if abs(m) <= tie_tolerance else (1 if m > 0 else -1) for m in means)
    consistent = len(set(signs)) == 1 and signs[0] != 0
    verdict = ("alternative" if signs[0] > 0 else "reference") if consistent else "tie"
    seed_table = pd.DataFrame(
        {
            "seed": list(seeds),
            "mean_diff": list(means),
            "wins": [c.n_wins for c in comparisons],
            "losses": [c.n_losses for c in comparisons],
            "ties": [c.n_ties for c in comparisons],
            "corrected_t_p": [c.corrected_t_p for c in comparisons],
            "wilcoxon_p": [
                math.nan if c.wilcoxon_p is None else c.wilcoxon_p for c in comparisons
            ],
        }
    )
    ref_label, alt_label = next(iter(labels))
    return MultiSeedComparisonResult(
        comparisons=comparisons,
        seeds=seeds,
        per_seed_mean_diff=means,
        per_seed_sign=signs,
        sign_consistent=consistent,
        verdict=verdict,
        reading=str(reading),
        reference_label=ref_label,
        alternative_label=alt_label,
        seed_table=seed_table,
    )
