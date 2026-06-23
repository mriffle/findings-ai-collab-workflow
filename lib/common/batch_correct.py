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
     It warns loudly when batch is perfectly confounded with a covariate of interest
     (correction would then delete that covariate's effect). Getting scientist sign-off
     before correcting is a workflow gate (the Stage-3/4 command), not enforced here.
  2. Keep the uncorrected ``Dataset`` and run the key analysis BOTH ways. "Signal
     survives batch-only correction" is supporting evidence; "signal disappears" means
     it is confounded with batch and cannot be attributed to biology. Carry this into
     the finding's caveats (conventions/statistics.md).

SCALE: ComBat's additive + multiplicative empirical-Bayes model assumes a roughly
Gaussian per-feature distribution, so it must run on a **log-ish** scale (``log2`` /
``glog2`` / ``zscore``), never on linear intensities. The scale tag enforces this.

Dataset-specific decisions (which samples are QC and should be excluded from the batch
fit, sample relabelings) live in the project copy, never here. ``sample_mask`` is the
generic hook: rows it excludes pass through uncorrected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pycombat import Combat

from common.data_loading import Dataset

__script_meta__: dict[str, object] = {
    "template": {"name": "batch-correct-combat", "version": "0.1"},
    "kind": "module",
    "provides": [
        "BatchConfoundingWarning",
        "ConfoundingReport",
        "assess_batch_confounding",
        "combat_correct",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Verified ComBat batch correction on a Dataset: batch-label-only (no "
        "covariate preserved, by design), log-scale guarded, zero-variance-in-batch "
        "passthrough, fail-loud. Ships a confounding assessment that warns loudly when "
        "the batch axis is confounded with a covariate of interest. Requires pycombat."
    ),
}

# Scales ComBat may run on (roughly Gaussian per feature). "linear" is refused.
_LOG_SCALES = frozenset({"log2", "glog2", "zscore"})


class BatchConfoundingWarning(UserWarning):
    """Warning that the batch axis is (near-)confounded with a covariate of interest."""


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
        Cramér's V association in ``[0, 1]`` (0 = independent, 1 = perfect
        association). ``0.0`` when either axis has fewer than two levels.
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
    if dataset.scale not in _LOG_SCALES:
        raise ValueError(
            f"combat_correct requires a log-ish scale {sorted(_LOG_SCALES)} but the "
            f"Dataset is on scale {dataset.scale!r}. ComBat's empirical-Bayes model "
            f"assumes a roughly Gaussian per-feature distribution; run it on log2 / "
            f"glog2 / zscore data (e.g. after normalize + log2_transform), not on "
            f"linear intensities."
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
    batch = dataset.metadata[batch_column].to_numpy().astype(str)
    if batch.shape[0] != dataset.abundances.shape[0]:
        raise ValueError(
            f"batch vector length {batch.shape[0]} does not match n_samples "
            f"{dataset.abundances.shape[0]}; metadata is not row-aligned to abundances."
        )
    return batch


def _combat_core(abundances: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """ComBat-correct ``abundances`` (n_samples, n_features) by ``batch``, label only.

    Features with zero variance in *any* single batch cannot be standardized (ComBat
    divides by the per-batch SD) and would come back NaN — they are detected up front
    and passed through unchanged so the output keeps the input shape.
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
                f"Batch {level!r} has only {n} sample(s); ComBat needs >= 2 per batch."
            )

    correctable = np.ones(abundances.shape[1], dtype=bool)
    for level in unique:
        level_var = abundances[batch == level].var(axis=0)
        correctable &= level_var > 0

    out = abundances.copy()
    if int(correctable.sum()) > 0:
        # batch label only: X (effects to preserve) is left None by design.
        corrected = Combat().fit_transform(abundances[:, correctable], batch)
        out[:, correctable] = np.asarray(corrected, dtype=float)
    return out


def combat_correct(
    dataset: Dataset,
    batch_column: str,
    *,
    sample_mask: np.ndarray | None = None,
) -> Dataset:
    """ComBat-correct a log-scale :class:`Dataset` by ``batch_column`` (label only).

    Parameters
    ----------
    dataset:
        Input dataset on a log-ish scale (``log2`` / ``glog2`` / ``zscore``; raised
        otherwise). Abundances must be finite.
    batch_column:
        Sample-metadata column naming the batch of each sample.
    sample_mask:
        Optional boolean mask, length ``n_samples``. Only ``True`` rows are passed to
        ComBat (e.g. drop QC samples whose batch label is meaningless); the rest are
        returned unchanged. The batch vector is taken from ``batch_column`` and subset
        to the masked rows.

    Returns
    -------
    Dataset
        A new dataset with batch-corrected abundances on the same scale; feature names,
        feature metadata, and sample metadata are passed through unchanged.

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
        corrected = _combat_core(abundances, batch)
    else:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != (abundances.shape[0],):
            raise ValueError(
                f"sample_mask shape {mask.shape} does not match n_samples "
                f"({abundances.shape[0]},)."
            )
        corrected = abundances.copy()
        corrected[mask] = _combat_core(abundances[mask], batch[mask])

    return replace(dataset, abundances=corrected)


def _cramers_v(crosstab: pd.DataFrame) -> float:
    """Cramér's V for a contingency table (0 = independent, 1 = perfect association)."""
    observed = crosstab.to_numpy(dtype=float)
    n = observed.sum()
    n_rows, n_cols = observed.shape
    if n == 0 or min(n_rows, n_cols) < 2:
        return 0.0
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    return float(np.sqrt(chi2 / (n * (min(n_rows, n_cols) - 1))))


def assess_batch_confounding(
    dataset: Dataset,
    batch_column: str,
    covariate_columns: Sequence[str],
) -> list[ConfoundingReport]:
    """Quantify confounding between the batch axis and each covariate of interest.

    For each covariate, builds the batch x covariate contingency table, computes a
    Cramér's V, and flags perfect confounding (covariate constant within every batch, so
    batch correction would delete its effect). Emits a :class:`BatchConfoundingWarning`
    for each perfectly-confounded covariate — the "loud warning" the scientist must see
    before deciding to correct (sign-off is a workflow gate, not enforced here).

    Returns one :class:`ConfoundingReport` per covariate, in input order.
    """
    import warnings

    if batch_column not in dataset.metadata.columns:
        raise ValueError(
            f"batch_column {batch_column!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )
    batch = dataset.metadata[batch_column].astype(str)

    reports: list[ConfoundingReport] = []
    for covariate_column in covariate_columns:
        if covariate_column not in dataset.metadata.columns:
            raise ValueError(
                f"covariate column {covariate_column!r} is not a metadata column "
                f"{list(dataset.metadata.columns)}."
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
        else:
            message = (
                f"{batch_column!r} vs {covariate_column!r}: Cramér's V={cramers_v:.2f} "
                f"(not perfectly confounded; some within-batch variation in "
                f"{covariate_column!r} survives batch correction)."
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
