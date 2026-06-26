"""Coefficient-of-variation (CV) distribution figures for a :class:`Dataset`.

TEMPLATE (lib/) — a *seed* for a project's CV-plotting module, not a finished script.
Copy it into the project's ``scripts/`` and adapt the call sites per study. Held to the
correctness charter (conventions/correctness.md): **assume nothing, verify everything,
fail loud.**

What it draws (one figure): overlaid per-feature CV distributions — one curve per
labelled :class:`Dataset` — as translucent histograms plus Gaussian-KDE lines, with each
distribution's median CV marked by a dashed vertical line and annotated. The canonical
use is comparing **normalization states of one dataset** (unnormalized vs normalized vs
batch-corrected) to show whether processing tightened the technical spread, but the
labels are the caller's — any set of comparable linear-scale Datasets works.

CV is the coefficient of variation, ``CV = std / mean`` computed **per feature across
the samples** on a **linear** intensity scale (proteomics: ``0`` = not detected, a real
low observation, kept). It is a unit-free measure of relative dispersion: lower median
CV means tighter, more reproducible quantification.

Scale (a HARD refuse, unlike the PCA template's warn): CV is only meaningful on a
positive linear scale. On a log or centered scale (``log2``/``glog2``/``zscore``, and
the ``"mad"``/``"vsn"`` normalizers' output) the per-feature mean sits near 0, so
``std/mean`` explodes or flips sign — mathematically ill-defined, not merely
suboptimal. So :func:`compute_cv` **raises** :class:`CVScaleError` on a non-linear
``Dataset.scale``. Compute CV on linear data: ``normalize(method="median")`` stays
linear, and a batch-corrected log matrix must be inverse-transformed back to linear
first (the project copy does that; this template only requires the linear input).

Convention wiring (conventions/visualization.md): the overlaid states' colors come from
the project color registry (:mod:`figures.colors`) so a state keeps its color across
every CV figure, capped at eight (the registry raises beyond that). The main figure
carries **no** baked-in legend; the legend is rendered as its **own figure** (a swatch
key) and saved beside it as ``<base>.legend.{svg,png}`` via
:func:`figures.figure_io.save_figure`.

**Control samples are rendered separately from experimental samples** (conventions/
visualization.md): subset the :class:`Dataset` *before* plotting and call
:func:`save_cv` once per subset (experimental, each QC pool) — exactly as the PCA
template expects exclusions to be applied upstream. This template makes NO study
decisions: which states to overlay, which samples belong to which subset, and the
registry namespace are all the caller's choice. Sample exclusions and relabelings live
in the project copy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common.data_loading import Dataset
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "cv-plot", "version": "0.1"},
    "kind": "module",
    "provides": [
        "CVScaleError",
        "CVResult",
        "CVPlot",
        "compute_cv",
        "plot_cv_distribution",
        "save_cv",
    ],
    "uses": ["common.data_loading", "figures.colors", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "CV distribution figures from one or more Datasets: per-feature CV (std/mean) "
        "computed across samples on a linear scale, overlaid as histogram + "
        "Gaussian-KDE curves with annotated median lines. Canonical use is comparing "
        "normalization states (unnormalized/normalized/batch-corrected). Hard-refuses "
        "a non-linear scale (std/mean ill-defined where mean ~ 0); state colors from "
        "the project registry with the >8-category guard; dual-export plus a separate "
        "legend image. Study-agnostic; fail-loud."
    ),
}

# Default color-registry namespace. CV figures overlay *methodological* states
# (normalization stages), not a biological variable, so a single shared default keeps
# the same color for, say, "Normalized" across the protein and precursor CV figures. The
# caller may override it; it is a workflow convention, not a study-specific fact.
DEFAULT_CV_CATEGORY = "NormalizationState"

# The x-axis (and KDE grid) upper bound when ``max_cv`` is not given: the 99.5th
# percentile of the pooled finite CVs, so a long right tail (a handful of huge-CV
# features) does not crush the bulk of the distribution into the left edge.
_UPPER_PERCENTILE = 99.5


class CVScaleError(ValueError):
    """Raised when CV is requested on a non-linear abundance scale.

    ``CV = std / mean`` is only meaningful on a positive **linear** scale. On a log or
    centered scale (``log2``/``glog2``/``zscore``, and the ``"mad"``/``"vsn"``
    normalizer output) the per-feature mean sits near 0, so ``std/mean`` explodes or
    flips sign. The
    fix is never to compute it anyway — it is to compute CV on linear data
    (``normalize(method="median")`` stays linear; inverse-transform batch-corrected log
    data back to linear first).
    """


@dataclass(frozen=True)
class CVResult:
    """The per-feature CVs underlying a CV figure.

    Attributes
    ----------
    cvs:
        ``{label: (n_features,) CV array}`` in the input order. ``NaN`` marks a feature
        whose mean is <= 0 (an all-zero / undefined feature), preserved so the count of
        droppable features is recoverable.
    medians:
        ``{label: median CV}`` (NaNs omitted) — the value the figure's dashed line
        marks. ``NaN`` only if a label had no finite CV at all.
    """

    cvs: dict[str, np.ndarray]
    medians: dict[str, float]


@dataclass
class CVPlot:
    """A rendered CV figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (overlaid CV distributions, no baked legend).
    legend_figure:
        A standalone swatch legend, saved beside the main figure as
        ``<base>.legend.{svg,png}`` by :func:`save_cv`.
    result:
        The underlying :class:`CVResult`.
    color_map:
        ``{label: hex}`` actually drawn (one per overlaid state).
    """

    figure: Figure
    legend_figure: Figure
    result: CVResult
    color_map: dict[str, str]


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #


def compute_cv(dataset: Dataset) -> np.ndarray:
    """Per-feature coefficient of variation (``std / mean``) across samples.

    Computed on the **linear** scale (``ddof=1`` sample std). A feature with mean <= 0
    (an all-zero column, or one left undefined by missingness) yields ``NaN`` rather
    than a spurious or exploding ratio — this is a deliberate hardening over a bare
    ``mean == 0`` check, since a linear intensity should never be negative and a <= 0
    mean means the ratio is not interpretable.

    Raises (fail loud) :class:`CVScaleError` if ``dataset.scale`` is not ``"linear"``
    (see the module docstring), ``ValueError`` if abundances are not 2D, or if there are
    fewer than two samples (CV needs a spread to measure).
    """
    if dataset.scale != "linear":
        raise CVScaleError(
            f"CV (std/mean) requires linear-scale abundances but the Dataset is on "
            f"scale {dataset.scale!r}. On log/centered scales the per-feature mean "
            f"sits near 0, so std/mean is ill-defined (explodes or flips sign). "
            f"Compute CV on linear data: normalize(method='median') stays linear, or "
            f"inverse-transform batch-corrected log data back to linear first."
        )
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(
            f"abundances must be 2D (n_samples, n_features); got {abundances.shape}."
        )
    n_samples = abundances.shape[0]
    if n_samples < 2:
        raise ValueError(
            f"CV needs >= 2 samples to have a spread to measure; got {n_samples}."
        )

    means = np.nanmean(abundances, axis=0)
    stds = np.nanstd(abundances, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cvs = stds / means
    cvs[means <= 0] = np.nan
    return np.asarray(cvs, dtype=float)


def _compute_result(datasets: Mapping[str, Dataset]) -> CVResult:
    """Compute each state's CV array + median (the scale guard fires per Dataset)."""
    cvs = {label: compute_cv(ds) for label, ds in datasets.items()}
    medians = {label: _nanmedian(cv) for label, cv in cvs.items()}
    return CVResult(cvs=cvs, medians=medians)


def _nanmedian(values: np.ndarray) -> float:
    """Median of the finite values (``NaN`` if none) without a runtime warning."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def plot_cv_distribution(
    datasets: Mapping[str, Dataset],
    *,
    category: str = DEFAULT_CV_CATEGORY,
    feature_type: str = "feature",
    max_cv: float | None = None,
    n_bins: int = 60,
    title: str | None = None,
    legend_title: str | None = None,
    show_medians: bool = True,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> CVPlot:
    """Overlay the CV distributions of one or more labelled linear-scale Datasets.

    Parameters
    ----------
    datasets:
        Ordered ``{label: Dataset}`` (insertion order = draw/legend order). Every
        Dataset must be on the ``"linear"`` scale (raised otherwise) and share identical
        ``feature_names`` (the states being compared are the same features in the same
        order). One Dataset is allowed (a single CV distribution).
    category:
        Color-registry namespace for the labels. Default ``"NormalizationState"``.
    feature_type:
        Noun for the features in the axis label (e.g. ``"protein"``, ``"precursor"``).
    max_cv:
        Optional upper bound (positive) on the CV x-axis and KDE grid. When omitted the
        axis extends to the 99.5th percentile of the pooled CVs.
    n_bins:
        Number of (shared) histogram bins, so the overlaid states are directly
        comparable. Must be >= 1.
    title:
        Optional figure suptitle.
    legend_title:
        Title for the companion legend figure; defaults to ``category``.
    show_medians:
        Draw + annotate each state's median CV (default ``True``).
    registry_path:
        Color registry JSON. Default ``state/color_registry.json``.
    persist_colors:
        Write newly assigned colors back to the registry (default ``True``).

    Returns
    -------
    CVPlot
        The main figure, the companion legend figure, the :class:`CVResult`, and the
        ``{label: hex}`` color map drawn.

    Raises
    ------
    CVScaleError
        If any Dataset is not on the ``"linear"`` scale.
    ValueError
        On an empty ``datasets``, an invalid ``n_bins``/``max_cv``, or mismatched
        ``feature_names``.
    CategoricalPaletteExceededError
        If the labels exceed the palette's categorical capacity (the >8 guard).
    """
    if len(datasets) == 0:
        raise ValueError("datasets is empty; pass >= 1 labelled Dataset to plot.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")
    if max_cv is not None and not max_cv > 0:
        raise ValueError(f"max_cv must be positive when given; got {max_cv}.")

    # Validate + compute BEFORE creating the figure, so the scale/alignment guards (the
    # likely failures) raise with no figure to leak. The only post-figure raise is the
    # registry's >8-category guard, handled by the try/except below.
    _validate_feature_alignment(datasets)
    result = _compute_result(datasets)
    labels = list(datasets.keys())
    title_for_legend = legend_title if legend_title is not None else category

    with publication_style():
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            color_map = assign_colors(
                category,
                labels,
                registry_path=registry_path,
                persist=persist_colors,
            )
            bin_upper = _upper_bound(result, labels, max_cv)
            _draw_distributions(
                ax,
                result,
                labels,
                color_map,
                bin_upper=bin_upper,
                n_bins=n_bins,
                feature_type=feature_type,
            )
            if show_medians:
                _annotate_medians(ax, result, labels, color_map)
            if title is not None:
                fig.suptitle(title, fontsize=14, weight="bold")
            legend_figure = _legend_figure(labels, color_map, title_for_legend)
        except BaseException:
            plt.close(fig)
            raise

    return CVPlot(
        figure=fig, legend_figure=legend_figure, result=result, color_map=color_map
    )


def save_cv(
    datasets: Mapping[str, Dataset],
    output_dir: str | Path,
    base_name: str,
    *,
    category: str = DEFAULT_CV_CATEGORY,
    feature_type: str = "feature",
    max_cv: float | None = None,
    n_bins: int = 60,
    title: str | None = None,
    legend_title: str | None = None,
    show_medians: bool = True,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render a CV figure (:func:`plot_cv_distribution`) and save it dual-export.

    Writes the main figure as ``<base>.{svg,png}`` and its companion legend as
    ``<base>.legend.{svg,png}`` via :func:`figures.figure_io.save_figure`, and returns
    their paths. ``save_figure`` is called with the default ``close=True``, so both
    figures are closed even if saving fails (no leak on a bad ``base_name``/``dpi``).
    """
    plot = plot_cv_distribution(
        datasets,
        category=category,
        feature_type=feature_type,
        max_cv=max_cv,
        n_bins=n_bins,
        title=title,
        legend_title=legend_title,
        show_medians=show_medians,
        registry_path=registry_path,
        persist_colors=persist_colors,
    )
    return save_figure(
        plot.figure,
        output_dir,
        base_name,
        legend_fig=plot.legend_figure,
        dpi=dpi,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate_feature_alignment(datasets: Mapping[str, Dataset]) -> None:
    """Refuse overlaying CVs computed over different feature sets (fail loud).

    The compared states must be the same features in the same order — otherwise the
    overlaid distributions are not measuring the same thing. ``n_samples`` may differ
    (CV is a per-feature reduction over samples, so it stays well-defined).
    """
    reference_names: np.ndarray | None = None
    reference_label = ""
    for label, ds in datasets.items():
        names = np.asarray(ds.feature_names)
        if reference_names is None:
            reference_names = names
            reference_label = label
            continue
        if names.shape != reference_names.shape or not np.array_equal(
            names, reference_names
        ):
            raise ValueError(
                f"Dataset {label!r} has different feature_names than "
                f"{reference_label!r}; CV distributions are only comparable across the "
                f"same features in the same order (the overlaid Datasets should be "
                f"normalization states of one dataset). Align them before plotting."
            )


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _upper_bound(result: CVResult, labels: list[str], max_cv: float | None) -> float:
    """The x-axis / bin / KDE upper bound: ``max_cv`` if given, else the 99.5th pct."""
    if max_cv is not None:
        return float(max_cv)
    pooled = _pooled_finite(result, labels)
    if pooled.size == 0:
        return 1.0  # no finite CVs at all — degenerate fallback so bins/axes stay valid
    upper = float(np.percentile(pooled, _UPPER_PERCENTILE))
    return upper if upper > 0 else 1.0


def _pooled_finite(result: CVResult, labels: list[str]) -> np.ndarray:
    """All finite CVs across every state, concatenated (for shared bins/limits)."""
    parts = [result.cvs[label][np.isfinite(result.cvs[label])] for label in labels]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _draw_distributions(
    ax: Axes,
    result: CVResult,
    labels: list[str],
    color_map: dict[str, str],
    *,
    bin_upper: float,
    n_bins: int,
    feature_type: str,
) -> None:
    """Histogram + KDE for each state on shared bins; set the axis labels and limits."""
    # Explicit list[float] (not the ndarray) so matplotlib's typed `bins` accepts it.
    bins = [float(edge) for edge in np.linspace(0.0, bin_upper, n_bins + 1)]
    for label in labels:
        finite = result.cvs[label][np.isfinite(result.cvs[label])]
        color = color_map[label]
        ax.hist(
            finite,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
            label=label,
            edgecolor="none",
        )
        _overlay_kde(ax, finite, color=color, x_upper=bin_upper)

    ax.set_xlabel(f"{feature_type} CV (std / mean)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_xlim(0.0, bin_upper)


def _overlay_kde(ax: Axes, finite: np.ndarray, *, color: str, x_upper: float) -> None:
    """Overlay a Gaussian-KDE curve for the finite CVs (skip if degenerate)."""
    if finite.size < 2 or float(finite.std()) <= 0.0:
        return  # < 2 points or no spread -> gaussian_kde is undefined/singular
    lower = float(finite.min())
    if not lower < x_upper:
        return
    kde = stats.gaussian_kde(finite)
    x_grid = np.linspace(lower, x_upper, 300)
    density = np.asarray(kde(x_grid), dtype=float)
    ax.plot(x_grid, density, color=color, linewidth=2.25)


def _annotate_medians(
    ax: Axes, result: CVResult, labels: list[str], color_map: dict[str, str]
) -> None:
    """Dashed median line + a text label for each state (finite medians only)."""
    for label in labels:
        med = result.medians[label]
        if np.isfinite(med):
            ax.axvline(med, color=color_map[label], linestyle="--", linewidth=1.2)

    y_top = ax.get_ylim()[1]
    # Even vertical spacing so the median annotations do not overlap each other.
    for i, label in enumerate(labels):
        med = result.medians[label]
        if not np.isfinite(med):
            continue
        ax.text(
            med,
            y_top * (0.95 - 0.07 * i),
            f" median={med:.3f}",
            color=color_map[label],
            fontsize=13,
            va="top",
            ha="left",
        )


# --------------------------------------------------------------------------- #
# Legend figure (rendered separately so it never overlaps the plot)
# --------------------------------------------------------------------------- #


def _legend_figure(
    labels: list[str], color_map: dict[str, str], legend_title: str
) -> Figure:
    """Standalone swatch legend: one colored line per state, in ``labels`` order."""
    height = max(1.4, 0.35 * len(labels) + 0.8)
    fig, ax = plt.subplots(figsize=(3.0, height))
    ax.axis("off")
    handles = [
        Line2D([0], [0], color=color_map[label], linewidth=2.5) for label in labels
    ]
    ax.legend(
        handles,
        labels,
        title=legend_title,
        loc="center",
        frameon=True,
        fontsize=12,
        title_fontsize=13,
    )
    return fig
