"""Verified normalization for a :class:`~common.data_loading.Dataset`.

TEMPLATE (lib/) — a *seed* for a project's ``common`` normalization module, not a
finished script. Copy it into the project's ``scripts/`` and adapt the call sites
(which method, in what order) per study. Held to the correctness charter
(conventions/correctness.md): **assume nothing, verify everything, fail loud.**

Study-agnostic by design. It operates on the standard ``Dataset`` contract (abundances
``(n_samples, n_features)`` + scale tag) and returns a new ``Dataset`` on the right
scale, so the downstream templates (PCA, differential abundance, figures) compose
unchanged. It hardcodes no biology and makes no sample selection — exclusions and
relabelings are study-specific decisions that live in the project copy and run *before*
or *after* normalization, **never** inside it.

Methods (all consume **linear**-scale abundances; pronoms normalizers, rows = samples):
  * ``"median"`` — divide each sample by its median, rescaled to the mean of medians
    (pronoms ``MedianNormalizer``). **Linear in -> linear out.** Follow with
    :func:`log2_transform` if you want a log scale.
  * ``"mad"`` — ``log2(x+1)`` per sample, then ``(log_x - median) / (1.4826 * MAD)``
    (pronoms ``MADNormalizer(log_transform=True, scale_to_sigma=True)``). Output is a
    robust per-sample z-score, centred at 0 with negatives expected -> ``"zscore"``.
  * ``"vsn"`` — arsinh variance-stabilizing normalization (pronoms ``VSNNormalizer``,
    Huber et al. 2002). Output is a glog2 scale where residual variance is roughly
    constant across the abundance range -> scale ``"glog2"``.

THE DOUBLE-LOG TRAP (why the scale tag is load-bearing): ``"mad"`` and ``"vsn"`` already
log-transform internally, and :func:`log2_transform` logs explicitly. Logging twice
silently corrupts every value. This module **refuses** any normalization or log step
whose input ``Dataset`` is not on the ``"linear"`` scale — the guard is the tag, not a
convention. Choose exactly one of {``mad``, ``vsn``, ``median``+``log2``}; never stack
log-producing steps.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from pronoms.normalizers import MADNormalizer, MedianNormalizer, VSNNormalizer

from common.data_loading import Dataset, Scale

__script_meta__: dict[str, object] = {
    "template": {"name": "normalize", "version": "0.1"},
    "kind": "module",
    "provides": ["NormalizationMethod", "normalize", "log2_transform"],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Verified Dataset normalization (median / mad / vsn) + log2 transform: "
        "scale-tag guarded against double-logging, fail-loud, shape-preserving. "
        "Study-agnostic; consumes and returns the standard Dataset contract."
    ),
}

NormalizationMethod = Literal["median", "mad", "vsn"]

# Each method's output scale (see the module docstring for what each scale means).
_OUTPUT_SCALE: dict[NormalizationMethod, Scale] = {
    "median": "linear",
    "mad": "zscore",
    "vsn": "glog2",
}


def _require_linear(dataset: Dataset, operation: str) -> None:
    """Refuse ``operation`` unless ``dataset`` is on the ``"linear"`` scale.

    The guard against the double-log trap: ``mad`` / ``vsn`` / ``log2`` assume linear
    input and produce a log-ish output, so re-applying one to an already-transformed
    matrix silently corrupts it. Median normalization is likewise a linear-scale step
    (dividing by a per-sample median is only meaningful on raw intensities).
    """
    if dataset.scale != "linear":
        raise ValueError(
            f"{operation} requires linear-scale abundances but the Dataset is on scale "
            f"{dataset.scale!r}. The 'mad'/'vsn' methods and log2_transform already "
            f"log-transform internally; applying one to non-linear data double-logs "
            f"and corrupts every value. Normalize/transform exactly once from linear."
        )


def _require_finite(abundances: np.ndarray, operation: str) -> None:
    """Refuse ``operation`` if any abundance is NaN or Inf (fail loud at the boundary).

    Normalization needs complete data; the pronoms normalizers reject non-finite input
    too, but this gives a workflow-actionable message pointing at the loader's missing
    policy rather than a generic library error.
    """
    if not np.isfinite(abundances).all():
        n_bad = int((~np.isfinite(abundances)).sum())
        raise ValueError(
            f"{operation} requires complete data but abundances contain {n_bad} "
            f"non-finite value(s) (NaN/Inf). Resolve missingness explicitly before "
            f"normalizing (e.g. the loader's missing-value policy / imputation)."
        )


def normalize(dataset: Dataset, method: NormalizationMethod) -> Dataset:
    """Normalize a linear-scale :class:`Dataset`; return a new one on the output scale.

    Parameters
    ----------
    dataset:
        Input dataset on the ``"linear"`` scale (raised otherwise). Abundances must be
        finite.
    method:
        ``"median"`` (→ linear), ``"mad"`` (→ zscore), or ``"vsn"`` (→ glog2). See the
        module docstring for the exact transform each applies.

    Returns
    -------
    Dataset
        A new dataset with normalized abundances and the appropriate ``scale``; feature
        names, feature metadata, and sample metadata are passed through unchanged.
    """
    if method not in _OUTPUT_SCALE:
        raise ValueError(
            f"Unknown normalization method {method!r}; use one of "
            f"{sorted(_OUTPUT_SCALE)}."
        )
    _require_linear(dataset, f"normalize(method={method!r})")
    _require_finite(dataset.abundances, f"normalize(method={method!r})")

    normalized = _normalize_abundances(dataset.abundances, method)
    if normalized.shape != dataset.abundances.shape:
        raise ValueError(
            f"normalize(method={method!r}) changed the matrix shape "
            f"{dataset.abundances.shape} -> {normalized.shape}; expected it preserved."
        )
    return replace(dataset, abundances=normalized, scale=_OUTPUT_SCALE[method])


def log2_transform(dataset: Dataset) -> Dataset:
    """Apply ``log2(x + 1)`` to a linear-scale :class:`Dataset`; return a log2 one.

    The ``+1`` pseudocount keeps not-detected zeros at exactly ``0`` and avoids
    ``log2(0) = -inf``. Pair this with ``method="median"`` (which stays linear) to reach
    a log scale; do **not** apply it to ``"mad"`` / ``"vsn"`` output (already logged) —
    the scale guard enforces this.
    """
    _require_linear(dataset, "log2_transform")
    _require_finite(dataset.abundances, "log2_transform")
    if np.any(dataset.abundances < 0):
        raise ValueError(
            "log2_transform requires non-negative abundances (log2(x+1) is undefined "
            "below -1); the Dataset has negative values."
        )
    return replace(dataset, abundances=np.log2(dataset.abundances + 1.0), scale="log2")


def _normalize_abundances(
    abundances: np.ndarray, method: NormalizationMethod
) -> np.ndarray:
    """Dispatch to the pronoms normalizer for ``method`` (linear, rows = samples).

    ``scale_to_sigma=True`` is passed explicitly to ``MADNormalizer`` to pin the
    sigma-equivalent divisor (and silence its transitional DeprecationWarning). VSN gets
    a C-contiguous float64 copy, as its native engine expects.
    """
    # pronoms is untyped, so .normalize() is Any; np.asarray pins it back to an ndarray.
    if method == "median":
        return np.asarray(MedianNormalizer().normalize(abundances))
    if method == "mad":
        mad = MADNormalizer(log_transform=True, scale_to_sigma=True)
        return np.asarray(mad.normalize(abundances))
    if method == "vsn":
        contiguous = np.ascontiguousarray(abundances, dtype=np.float64)
        return np.asarray(VSNNormalizer().normalize(contiguous))
    raise ValueError(
        f"Unknown normalization method {method!r}."
    )  # unreachable; mypy guard
