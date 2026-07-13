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

Going the other way: :func:`to_linear` inverts a plain-log scale (``log2`` / ``log10`` /
``ln``) back to ``"linear"`` — the strictly-positive ``base**x``. Its purpose is reading
a linear-scale metric off log-derived data, notably a CV (``std/mean``) of a
ComBat-batch-corrected matrix (ComBat runs on log, CV needs linear). It refuses the
scales with no clean linear inverse (``glog2`` / ``zscore`` / ``ratio``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from pronoms.normalizers import MADNormalizer, MedianNormalizer, VSNNormalizer

from common.data_loading import LOG_SCALES, Dataset, Scale

__script_meta__: dict[str, object] = {
    "template": {"name": "normalize", "version": "0.4"},
    "kind": "module",
    "provides": [
        "NormalizationMethod",
        "normalize",
        "log2_transform",
        "to_linear",
        "median_center",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Verified Dataset normalization (median / mad / vsn), log2 transform, plain-log"
        " -> linear inversion (to_linear), and log-domain median centering: scale-tag "
        "guarded against double-logging, fail-loud, shape-preserving, returns an "
        "independent (non-aliased) Dataset. Study-agnostic; consumes and returns the "
        "standard Dataset contract."
    ),
}

NormalizationMethod = Literal["median", "mad", "vsn"]

# Each method's output scale (see the module docstring for what each scale means).
_OUTPUT_SCALE: dict[NormalizationMethod, Scale] = {
    "median": "linear",
    "mad": "zscore",
    "vsn": "glog2",
}


def _independent(dataset: Dataset, abundances: np.ndarray, scale: Scale) -> Dataset:
    """Build a new Dataset sharing NO mutable state with ``dataset``.

    ``dataclasses.replace`` copies the unchanged fields *by reference*, which aliases
    ``metadata`` / ``feature_metadata`` / ``feature_names`` between input and output, so
    a downstream in-place mutation of one would corrupt the other. That breaks the
    keep-the-original-and-compare contract the batch-correction workflow relies on, so
    the carried fields are copied defensively.
    """
    return replace(
        dataset,
        abundances=abundances,
        scale=scale,
        metadata=dataset.metadata.copy(),
        feature_metadata=dataset.feature_metadata.copy(),
        feature_names=dataset.feature_names.copy(),
    )


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
    return _independent(dataset, normalized, _OUTPUT_SCALE[method])


def log2_transform(dataset: Dataset, *, pseudocount: float = 1.0) -> Dataset:
    """Apply ``log2(x + pseudocount)`` to a linear :class:`Dataset`; return a log2 one.

    ``pseudocount`` defaults to ``1.0``, which keeps not-detected zeros at exactly ``0``
    and avoids ``log2(0) = -inf`` — appropriate for raw intensity where ``0`` means
    not-detected. A different value shifts zeros to ``log2(pseudocount)``; choose it to
    suit the dynamic range (a ``+1`` count is negligible for large intensities but
    distorts values near 1, e.g. ratio data). Must be positive.

    Pair this with ``method="median"`` (which stays linear) to reach a log scale; do
    **not** apply it to ``"mad"`` / ``"vsn"`` output (already logged) — the scale guard
    enforces this.
    """
    if pseudocount <= 0:
        raise ValueError(f"pseudocount must be positive; got {pseudocount}.")
    _require_linear(dataset, "log2_transform")
    _require_finite(dataset.abundances, "log2_transform")
    if np.any(dataset.abundances < -pseudocount):
        raise ValueError(
            f"log2_transform requires abundances >= -pseudocount ({-pseudocount}); the "
            f"Dataset has smaller values (log2 would be undefined)."
        )
    return _independent(dataset, np.log2(dataset.abundances + pseudocount), "log2")


# Plain-log scales this inverts back to linear (base ** x). glog2 (VSN/arsinh) and
# zscore are variance-stabilized / centered, not a plain base**x log, so they have no
# clean linear inverse; ratio is not a log; linear is already there.
_INVERTIBLE_LOG_SCALES: frozenset[Scale] = frozenset({"log2", "log10", "ln"})


def to_linear(dataset: Dataset) -> Dataset:
    """Invert a plain-log :class:`Dataset` (``log2`` / ``log10`` / ``ln``) to linear.

    The mathematical inverse of the log **scale**: ``2**x`` / ``10**x`` / ``exp(x)``,
    so the output is **strictly positive** and tagged ``"linear"``. The canonical use is
    reading a *linear-scale* metric off log-derived data — e.g. a **coefficient of
    variation** (``std / mean``, which :func:`figures.cv.compute_cv` refuses on a log
    scale) of a **ComBat-batch-corrected** matrix: ComBat runs on log data, so de-log
    its output before a CV comparison of raw / normalized / batch-corrected states.

    This inverts the *scale*, not a specific forward transform: data produced by
    ``log2_transform(x, pseudocount=p)`` de-logs to ``x + p`` (offset by the
    pseudocount), which is immaterial for a relative / CV comparison — and keeps every
    value positive, so a subsequent CV is well defined. If you need exact intensity
    recovery, subtract the pseudocount yourself (handle any resulting non-positives).

    Refuses (fail loud) ``"linear"`` (already there), ``"glog2"`` / ``"zscore"`` (no
    clean linear inverse), and ``"ratio"`` (not a log scale).
    """
    if dataset.scale not in _INVERTIBLE_LOG_SCALES:
        raise ValueError(
            f"to_linear inverts a plain-log scale {sorted(_INVERTIBLE_LOG_SCALES)} "
            f"back to linear but the Dataset is on scale {dataset.scale!r}. 'linear' "
            f"is already linear; 'glog2'/'zscore' are variance-stabilized/centered "
            f"with no clean linear inverse; 'ratio' is not a log scale."
        )
    _require_finite(dataset.abundances, "to_linear")
    if dataset.scale == "log2":
        linear = np.exp2(dataset.abundances)
    elif dataset.scale == "log10":
        linear = np.power(10.0, dataset.abundances)
    else:  # "ln"
        linear = np.exp(dataset.abundances)
    _require_finite(linear, "to_linear (result overflowed)")
    return _independent(dataset, linear, "linear")


def median_center(dataset: Dataset) -> Dataset:
    """Subtract each sample's median — the log-domain analogue of median normalization.

    On a **log** scale, the per-sample size-factor correction is *subtractive* (median
    centering), not the *divisive* correction :func:`normalize` with ``"median"`` does
    on linear data. Use this when a study's data already arrives log-scale (``log2`` /
    ``log10`` / ``ln`` / ``glog2``) and you want per-sample median alignment without
    returning to linear. The output keeps the input scale.

    Refuses ``"linear"`` (use ``normalize(method="median")`` there) and ``"ratio"``.
    """
    if dataset.scale not in LOG_SCALES:
        raise ValueError(
            f"median_center requires a log scale {sorted(LOG_SCALES)} (it is the "
            f"log-domain median normalization) but the Dataset is on scale "
            f"{dataset.scale!r}. For linear data use normalize(method='median')."
        )
    _require_finite(dataset.abundances, "median_center")
    centered = dataset.abundances - np.median(dataset.abundances, axis=1, keepdims=True)
    return _independent(dataset, centered, dataset.scale)


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
