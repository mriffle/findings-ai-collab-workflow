"""Verified ComBat batch correction for a :class:`~common.data_loading.Dataset`.

TEMPLATE (lib/) — a *seed* for a project's ``common`` batch-correction module, not a
finished script. Copy it into the project's ``scripts/`` and adapt the call sites per
study. Held to the correctness charter (conventions/correctness.md): **assume nothing,
verify everything, fail loud.**

Study-agnostic: the batch axis and the covariates of interest are named by metadata
column; no biology is hardcoded. It operates on the standard ``Dataset`` contract and
returns a new ``Dataset`` on the same scale.

BATCH LABEL ONLY — a deliberate anti-cheating decision (do not "fix" this):
:func:`combat_correct` hands ComBat the batch vector and **nothing else** (pycombat's
``X`` "effects to preserve" stays ``None``). When batch is confounded with the biology
of interest, telling ComBat to *protect* that biology lets it attribute the confounded
variance to biology rather than batch — which launders a possible batch artifact into
the very signal you then test for. Batch-only correction is the conservative choice: it
cannot manufacture signal. Its honest cost is the mirror risk — it can *remove* real
biology that aligns with batch — so two safeguards are mandatory:

  1. Run :func:`assess_batch_confounding` first and surface the result to the scientist.
     It warns (graded) when batch is perfectly **or** strongly confounded with a
     covariate of interest (correction would then delete/attenuate that effect).
     Scientist sign-off before correcting is a workflow gate (the Stage-3/4 command).
  2. Keep the uncorrected ``Dataset`` and run the key analysis BOTH ways as a robustness
     check. "Signal survives" is supporting evidence; "signal disappears" means it is
     confounded with batch. Two honest caveats: partial confounding *attenuates* real
     biology too, and feeding ComBat output to a naive significance test inflates
     significance (variance deflation, Nygaard 2016) — for *testing*, prefer modeling
     batch as a covariate over pre-correcting (conventions/statistics.md). Use ComBat
     output for visualization/clustering and this robustness check.

SCALE: ComBat's additive + multiplicative empirical-Bayes model assumes a roughly
Gaussian per-feature distribution, so it must run on a **log** scale (``log2`` /
``log10`` / ``ln`` / ``glog2`` / ``zscore``), never on linear intensities. The scale tag
enforces this.

Dataset-specific decisions (which samples are QC and should be excluded from the batch
fit, sample relabelings) live in the project copy, never here. ``sample_mask`` is the
generic hook: rows it excludes pass through uncorrected.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pycombat import Combat

from common.data_loading import LOG_SCALES, Dataset

__script_meta__: dict[str, object] = {
    "template": {"name": "batch-correct-combat", "version": "0.2"},
    "kind": "module",
    "provides": [
        "BatchConfoundingWarning",
        "BatchPassthroughWarning",
        "ConfoundingReport",
        "assess_batch_confounding",
        "combat_correct",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Verified ComBat batch correction on a Dataset: batch-label-only (no "
        "covariate preserved, by design), log-scale guarded, near-constant-feature "
        "passthrough (warned), fail-loud, returns an independent Dataset. Ships a "
        "confounding assessment (bias-corrected Cramér's V) that warns on confounding "
        "with a covariate of interest. Requires pycombat."
    ),
}

# Default association above which assess_batch_confounding warns even when confounding
# is not perfect (a covariate this aliased with batch loses much signal to correction).
_DEFAULT_WARN_THRESHOLD = 0.5


class BatchConfoundingWarning(UserWarning):
    """Warning that the batch axis is (near-)confounded with a covariate of interest."""


class BatchPassthroughWarning(UserWarning):
    """Warning that some (near-)constant features were passed through uncorrected."""


@dataclass(frozen=True)
class ConfoundingReport:
    """Degree of confounding between the batch axis and one covariate of interest.

    Attributes
    ----------
    batch_column, covariate_column:
        The two metadata columns compared.
    crosstab:
        Contingency table (batch levels x covariate levels) of sample counts.
    cramers_v:
        Bias-corrected (Bergsma) Cramér's V association in ``[0, 1]`` (0 = independent,
        1 = perfect association). ``0.0`` when either axis has fewer than two levels.
    perfectly_confounded:
        ``True`` when the covariate is constant within every batch — i.e. the
        covariate's variation is entirely *between* batches, so batch correction
        would remove that covariate's effect wholesale.
    message:
        Human-readable summary for surfacing to the scientist.
    """

    batch_column: str
    covariate_column: str
    crosstab: pd.DataFrame
    cramers_v: float
    perfectly_confounded: bool
    message: str


def _require_log_scale(dataset: Dataset) -> None:
    """Refuse correction unless ``dataset`` is on a log-ish (Gaussian-ish) scale."""
    if dataset.scale not in LOG_SCALES:
        raise ValueError(
            f"combat_correct requires a log-ish scale {sorted(LOG_SCALES)} but the "
            f"Dataset is on scale {dataset.scale!r}. ComBat's empirical-Bayes model "
            f"assumes a roughly Gaussian per-feature distribution; run it on a log "
            f"scale (e.g. after normalize + log2_transform), not on linear intensities."
        )


def _require_no_na(series: pd.Series, column: str, context: str) -> None:
    """Refuse if a metadata column has missing values (fail loud, not silent-drop).

    ``pd.crosstab`` silently drops NaN rows and ``str(NaN)`` would otherwise become a
    phantom ``"nan"`` batch — either way a missing label corrupts the verdict quietly,
    which is the worst failure mode for a safety check.
    """
    if series.isna().any():
        n_bad = int(series.isna().sum())
        raise ValueError(
            f"{context}: metadata column {column!r} has {n_bad} missing value(s). "
            f"Resolve or explicitly relabel them (a missing batch/covariate label "
            f"would be silently dropped from the analysis)."
        )


def _require_finite(abundances: np.ndarray) -> None:
    """Refuse correction if any abundance is NaN or Inf (ComBat needs complete data)."""
    if not np.isfinite(abundances).all():
        n_bad = int((~np.isfinite(abundances)).sum())
        raise ValueError(
            f"combat_correct requires complete data but abundances contain {n_bad} "
            f"non-finite value(s) (NaN/Inf). Resolve missingness before correcting."
        )


def _batch_vector(dataset: Dataset, batch_column: str) -> np.ndarray:
    """Extract the batch label vector (as strings) from the sample metadata."""
    if batch_column not in dataset.metadata.columns:
        raise ValueError(
            f"batch_column {batch_column!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )
    _require_no_na(dataset.metadata[batch_column], batch_column, "combat_correct")
    batch: np.ndarray = dataset.metadata[batch_column].to_numpy().astype(str)
    if batch.shape[0] != dataset.abundances.shape[0]:
        raise ValueError(
            f"batch vector length {batch.shape[0]} does not match n_samples "
            f"{dataset.abundances.shape[0]}; metadata is not row-aligned to abundances."
        )
    return batch


def _combat_core(
    abundances: np.ndarray, batch: np.ndarray, min_variance: float
) -> np.ndarray:
    """ComBat-correct ``abundances`` (n_samples, n_features) by ``batch``, label only.

    Globally (near-)constant features — pooled per-feature variance ``<= min_variance``
    — cannot be standardized by ComBat's empirical-Bayes step, and because that step
    pools information *across features*, a single such feature can drive the WHOLE
    corrected matrix to NaN. They are detected up front and passed through UNCORRECTED
    (warned), so the output keeps the input shape. (A feature constant within one batch
    but variable overall is fine and IS corrected — pycombat standardizes by a pooled,
    not per-batch, variance, so the earlier per-batch-variance filter wrongly excluded
    those.)
    """
    abundances = np.asarray(abundances, dtype=float)
    batch = np.asarray(batch).astype(str)
    if abundances.ndim != 2:
        raise ValueError(
            f"abundances must be 2D (n_samples, n_features); got {abundances.shape}."
        )
    if batch.shape[0] != abundances.shape[0]:
        raise ValueError(
            f"batch length {batch.shape[0]} does not match n_samples "
            f"{abundances.shape[0]}."
        )

    unique = np.unique(batch)
    if unique.shape[0] < 2:
        raise ValueError(f"ComBat needs >= 2 distinct batches; got {unique.tolist()}.")
    for level in unique:
        n = int((batch == level).sum())
        if n < 2:
            raise ValueError(
                f"Batch {level!r} has only {n} sample(s); ComBat needs >= 2 per batch. "
                f"(If a sample_mask was applied, the count is over the masked subset.)"
            )

    correctable = abundances.var(axis=0) > min_variance
    n_passthrough = int((~correctable).sum())
    if n_passthrough:
        warnings.warn(
            f"{n_passthrough} feature(s) had pooled variance <= {min_variance} "
            f"(globally (near-)constant) and were passed through UNCORRECTED — they "
            f"cannot be batch-standardized. They sit on a different footing from the "
            f"corrected features.",
            BatchPassthroughWarning,
            stacklevel=3,
        )

    out = abundances.copy()
    if int(correctable.sum()) > 0:
        # batch label only: X (effects to preserve) is left None by design.
        corrected = np.asarray(
            Combat().fit_transform(abundances[:, correctable], batch), dtype=float
        )
        if not np.isfinite(corrected).all():
            raise ValueError(
                "ComBat returned non-finite values even after excluding constant "
                "features; near-constant features likely remain. Raise `min_variance` "
                "to exclude them, or inspect the inputs."
            )
        out[:, correctable] = corrected
    return out


def combat_correct(
    dataset: Dataset,
    batch_column: str,
    *,
    sample_mask: np.ndarray | None = None,
    min_variance: float = 0.0,
) -> Dataset:
    """ComBat-correct a log-scale :class:`Dataset` by ``batch_column`` (label only).

    Parameters
    ----------
    dataset:
        Input dataset on a log-ish scale (``log2`` / ``log10`` / ``ln`` / ``glog2`` /
        ``zscore``; raised otherwise). Abundances must be finite.
    batch_column:
        Sample-metadata column naming the batch of each sample (no missing values).
    sample_mask:
        Optional boolean mask, length ``n_samples``. Only ``True`` rows are passed to
        ComBat (e.g. drop QC samples whose batch label is meaningless); the rest are
        returned UNCORRECTED. The ≥2-distinct / ≥2-per-batch checks run on the masked
        subset, so a mask that drops a batch below 2 fails loud. **Masked-out rows are
        raw, not corrected — never compare them against corrected rows.**
    min_variance:
        Pooled per-feature variance at or below which a feature is treated as (near-)
        constant and passed through uncorrected (default ``0.0`` = exactly constant).
        Raise it if ComBat still returns non-finite values (near-constant features).

    Returns
    -------
    Dataset
        A new, independent dataset with batch-corrected abundances on the same scale;
        feature names, feature metadata, and sample metadata are copied through.

    Notes
    -----
    Run :func:`assess_batch_confounding` and obtain scientist sign-off first, and keep
    the uncorrected dataset for the both-ways comparison — see the module docstring.
    """
    _require_log_scale(dataset)
    _require_finite(dataset.abundances)
    batch = _batch_vector(dataset, batch_column)
    abundances = np.asarray(dataset.abundances, dtype=float)

    if sample_mask is None:
        corrected = _combat_core(abundances, batch, min_variance)
    else:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != (abundances.shape[0],):
            raise ValueError(
                f"sample_mask shape {mask.shape} does not match n_samples "
                f"({abundances.shape[0]},)."
            )
        corrected = abundances.copy()
        corrected[mask] = _combat_core(abundances[mask], batch[mask], min_variance)

    return replace(
        dataset,
        abundances=corrected,
        metadata=dataset.metadata.copy(),
        feature_metadata=dataset.feature_metadata.copy(),
        feature_names=dataset.feature_names.copy(),
    )


def _cramers_v(crosstab: pd.DataFrame) -> float:
    """Bias-corrected Cramér's V for a contingency table (Bergsma 2013).

    The textbook V is badly upward-biased at small n (e.g. mean ~0.41 under true
    independence at n=12, 3x3) — exactly the regime of many omics studies — which would
    inflate the confounding number the scientist acts on. Bergsma's correction subtracts
    the expected-under-independence inflation. Returns ``0.0`` for a degenerate table
    (one row/col, or n <= 1).
    """
    observed = crosstab.to_numpy(dtype=float)
    n = float(observed.sum())
    n_rows, n_cols = observed.shape
    if n <= 1 or min(n_rows, n_cols) < 2:
        return 0.0
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (n_rows - 1) * (n_cols - 1) / (n - 1))
    rows_corr = n_rows - (n_rows - 1) ** 2 / (n - 1)
    cols_corr = n_cols - (n_cols - 1) ** 2 / (n - 1)
    denom = min(rows_corr - 1, cols_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def assess_batch_confounding(
    dataset: Dataset,
    batch_column: str,
    covariate_columns: Sequence[str],
    *,
    warn_threshold: float = _DEFAULT_WARN_THRESHOLD,
) -> list[ConfoundingReport]:
    """Quantify confounding between the batch axis and each covariate of interest.

    For each covariate, builds the batch x covariate contingency table, computes a
    bias-corrected Cramér's V, and flags **perfect** confounding (covariate constant
    within every batch, so batch correction would delete its effect entirely). Emits a
    :class:`BatchConfoundingWarning` when confounding is perfect **or** the association
    is strong (``cramers_v >= warn_threshold``, default 0.5) — partial confounding still
    attenuates real signal, so the warning is graded, not perfect-only. The warning is
    the "loud" half; scientist sign-off before correcting is a workflow gate, not
    enforced here. Raises (fail loud) if the batch or any covariate column has missing
    values — they would otherwise be silently dropped from the crosstab.

    Returns one :class:`ConfoundingReport` per covariate, in input order.
    """
    if batch_column not in dataset.metadata.columns:
        raise ValueError(
            f"batch_column {batch_column!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )
    _require_no_na(
        dataset.metadata[batch_column], batch_column, "assess_batch_confounding"
    )
    batch = dataset.metadata[batch_column].astype(str)

    reports: list[ConfoundingReport] = []
    for covariate_column in covariate_columns:
        if covariate_column not in dataset.metadata.columns:
            raise ValueError(
                f"covariate column {covariate_column!r} is not a metadata column "
                f"{list(dataset.metadata.columns)}."
            )
        _require_no_na(
            dataset.metadata[covariate_column],
            covariate_column,
            "assess_batch_confounding",
        )
        covariate = dataset.metadata[covariate_column].astype(str)
        crosstab = pd.crosstab(batch, covariate)
        cramers_v = _cramers_v(crosstab)
        # Perfectly confounded: every batch row has exactly one non-zero cell, i.e. the
        # covariate takes a single level within each batch (no within-batch variation).
        nonzero_per_batch = (crosstab.to_numpy() > 0).sum(axis=1)
        perfectly_confounded = bool(np.all(nonzero_per_batch == 1))

        if perfectly_confounded:
            message = (
                f"{batch_column!r} is PERFECTLY confounded with {covariate_column!r}: "
                f"each batch contains a single {covariate_column!r} level (Cramér's V="
                f"{cramers_v:.2f}). Batch-only ComBat correction would remove this "
                f"covariate's effect wholesale. Report corrected AND uncorrected "
                f"results and treat survival-of-correction as the test; carry the "
                f"confound into the finding's caveats."
            )
            warnings.warn(message, BatchConfoundingWarning, stacklevel=2)
        elif cramers_v >= warn_threshold:
            message = (
                f"{batch_column!r} is STRONGLY confounded with {covariate_column!r} "
                f"(Cramér's V={cramers_v:.2f} >= {warn_threshold}). Batch correction "
                f"will attenuate this covariate's signal; report corrected AND "
                f"uncorrected results and carry the confound into the caveats."
            )
            warnings.warn(message, BatchConfoundingWarning, stacklevel=2)
        else:
            message = (
                f"{batch_column!r} vs {covariate_column!r}: Cramér's V={cramers_v:.2f} "
                f"(below the {warn_threshold} warn threshold; within-batch variation "
                f"in {covariate_column!r} largely survives batch correction)."
            )

        reports.append(
            ConfoundingReport(
                batch_column=batch_column,
                covariate_column=covariate_column,
                crosstab=crosstab,
                cramers_v=cramers_v,
                perfectly_confounded=perfectly_confounded,
                message=message,
            )
        )
    return reports
