"""Identification-depth (detected-features-per-run) QC bar charts for a `Dataset`.

TEMPLATE (lib/) — a *seed* for a project's identification-depth module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

What it draws (one figure): a **stack of per-sample bar charts**, one panel per labelled
:class:`Dataset`, sharing the same samples. Each bar is the number of *detected*
features in that sample (proteins / precursors / peptides quantified in the run). This
is the most universal DIA QC view — the headline of every search-engine QC report —
because it catches dropouts an abundance summary smooths over: a failing injection, an
LC problem, or acquisition-order drift shows up as a short bar even when the surviving
intensities look normal. The canonical use is two panels — **protein-level** over
**precursor-level** depth — read together, but the labels/levels are the caller's and a
single panel is allowed.

Detection: a feature is *detected* in a sample when its value is finite and strictly
greater than ``min_intensity`` (default ``0.0``; in DIA a ``0`` is "not detected"). The
count is taken **per sample over the features** (``axis=1``).

Sample order: bars are drawn in the :class:`Dataset`'s **row order**. Order the samples
by **acquisition order** upstream (the loader's ``order_by``) so the x-axis is run order
and a drifting or failing block of runs is visible left-to-right — exactly as the
abundance-boxplot template expects upstream ordering.

Bar color (the one categorical channel): pass ``color_by`` a metadata column (e.g. a
sample-class column distinguishing experimental from pooled-QC / reference samples) and
the bars are colored through the project color registry (:mod:`figures.colors`) so a
value keeps its color across figures, capped at eight (the registry raises beyond that).
With no ``color_by`` the bars are a single neutral color. **Annotation stripes are
deliberately omitted** (unlike the abundance-boxplot template): identification depth's
primary categorical encoding *is* the bar color, so a second dimension is better served
by faceting or a separate figure — this keeps the template tight. If you need a second
dimension, color by it and facet on the first.

Scale (a HARD refuse, like the CV template — unlike PCA/correlation/abundance-boxplot's
warn): a detection count is a property of the **raw linear matrix**. On a centered or
standardized scale (``log2``/``glog2``/``zscore``/...) a "0" is an ordinary value and
``> 0`` no longer means "detected", so the count is meaningless. So
:func:`compute_detection_counts` **raises** :class:`IdDepthScaleError` on a non-linear
``Dataset.scale``. Compute depth on the unnormalized data, before any transform — it is
a first-look QC, and ``normalize(method="median")`` stays linear if you must normalize.

Reference line: each panel marks a dashed reference median. Pass ``reference_mask`` a
boolean over the samples (e.g. the experimental subset) and the line is that subset's
median detected count, so a low run stands out against where the real samples sit; with
no mask the line is the all-sample median.

Convention wiring (conventions/visualization.md): the main figure carries **no**
baked-in legend; when ``color_by`` is set the legend is rendered as its **own figure**
(a swatch key) and saved beside the plot as ``<base>.legend.{svg,png}`` via
:func:`figures.figure_io.save_figure`.

**Control samples are rendered with the experimental samples here**, distinguished by
the ``color_by`` color and the reference line — a deliberate divergence from the
"render controls separately" default (conventions/visualization.md), shared with the
sample-correlation template: the cross-class comparison (do the pools sit high and
tight? is a real sample a dropout?) *is* the deliverable. This template makes NO study
decisions: which levels to stack, the sample order, which column colors the bars, and
which samples define the reference median are all the caller's choice. Sample exclusions
and relabelings live in the project copy.
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

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "id-depth", "version": "0.1"},
    "kind": "module",
    "provides": [
        "IdDepthScaleError",
        "IdDepthResult",
        "IdDepthPlot",
        "compute_detection_counts",
        "plot_id_depth",
        "save_id_depth",
    ],
    "uses": ["common.data_loading", "figures.colors", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Identification-depth QC bar charts from one or more Datasets: detected "
        "features per sample (finite and > min_intensity), one stacked panel per "
        "feature level (e.g. protein over precursor), bars drawn in acquisition order "
        "to surface dropouts/drift. Bars optionally colored by a categorical metadata "
        "column via the project registry with the >8-category guard; a per-panel "
        "reference-median line over an optional sample subset; hard-refuses a "
        "non-linear scale (detection is a raw-linear property); dual-export plus a "
        "separate legend image. Uses common.data_loading, figures.colors, "
        "figures.figure_io. Study-agnostic; fail-loud."
    ),
}

# Bar color when no ``color_by`` is given: a neutral slate, outside the categorical
# palette so it consumes no registry slot and reads as "uncategorized".
UNCATEGORIZED_BAR_COLOR = "#6e7f8d"


class IdDepthScaleError(ValueError):
    """Raised when a detection count is requested on a non-linear abundance scale.

    A detection count is a property of the **raw linear matrix**: a feature is detected
    when its intensity is finite and ``> min_intensity`` (a ``0`` is "not detected"). On
    a log or centered scale (``log2``/``glog2``/``zscore``, and the ``"mad"``/``"vsn"``
    normalizer output) a ``0`` is an ordinary value and ``> 0`` no longer means
    "detected", so the count is meaningless. Compute depth on the unnormalized data,
    before any transform (it is a first-look QC); ``normalize(method='median')`` stays
    linear if normalization is needed first.
    """


@dataclass(frozen=True)
class IdDepthResult:
    """The per-sample detection counts underlying an identification-depth figure.

    Attributes
    ----------
    sample_ids:
        ``(n_samples,)`` the sample identifiers (the shared order across panels).
    counts:
        ``{label: (n_samples,) detected-feature count}`` in panel order — the QC signal.
    reference_median:
        ``{label: median detected count}`` the panel's dashed line marks, taken over the
        ``reference_mask`` subset when given, else all samples. ``NaN`` only if the
        subset was empty.
    reference_is_subset:
        ``True`` when the reference median was taken over a caller-supplied subset (the
        line is labelled "ref. median"), ``False`` when over all samples ("median").
    """

    sample_ids: np.ndarray
    counts: dict[str, np.ndarray]
    reference_median: dict[str, float]
    reference_is_subset: bool


@dataclass
class IdDepthPlot:
    """A rendered identification-depth figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (stacked per-sample bar panels, no baked legend).
    legend_figure:
        A standalone swatch legend when ``color_by`` was set (saved beside the main
        figure as ``<base>.legend.{svg,png}`` by :func:`save_id_depth`); ``None`` when
        the bars were uncolored, in which case no legend file is written.
    result:
        The underlying :class:`IdDepthResult`.
    color_map:
        ``{value: hex}`` drawn for ``color_by`` (empty when the bars were uncolored).
    """

    figure: Figure
    legend_figure: Figure | None
    result: IdDepthResult
    color_map: dict[str, str]


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #


def compute_detection_counts(
    dataset: Dataset, *, min_intensity: float = 0.0
) -> np.ndarray:
    """Per-sample count of detected features (finite and ``> min_intensity``).

    Detection is a property of the **raw linear matrix**, so this raises
    :class:`IdDepthScaleError` unless ``dataset.scale == "linear"`` (see the module
    docstring). A ``0`` (or any value ``<= min_intensity``) and a ``NaN`` both count as
    not detected. The reduction is over the features (``axis=1``), one count per sample.

    Raises :class:`IdDepthScaleError` on a non-linear scale, or ``ValueError`` if
    abundances are not 2D.
    """
    if dataset.scale != "linear":
        raise IdDepthScaleError(
            f"Identification depth (a detected-feature count) requires linear-scale "
            f"abundances but the Dataset is on scale {dataset.scale!r}. On log/centered"
            f" scales a 0 is an ordinary value, so '> {min_intensity}' no longer means "
            f"'detected'. Count depth on the unnormalized data, before any transform "
            f"(normalize(method='median') stays linear if needed first)."
        )
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(
            f"abundances must be 2D (n_samples, n_features); got {abundances.shape}."
        )
    detected = np.isfinite(abundances) & (abundances > min_intensity)
    return np.asarray(detected.sum(axis=1), dtype=int)


def _compute_result(
    datasets: Mapping[str, Dataset],
    sample_ids: np.ndarray,
    *,
    min_intensity: float,
    reference_mask: np.ndarray | None,
) -> IdDepthResult:
    """Per-sample counts + the per-level reference median (scale guard fires here)."""
    counts = {
        label: compute_detection_counts(ds, min_intensity=min_intensity)
        for label, ds in datasets.items()
    }
    reference: dict[str, float] = {}
    for label, count in counts.items():
        subset = count[reference_mask] if reference_mask is not None else count
        reference[label] = float(np.median(subset)) if subset.size else float("nan")
    return IdDepthResult(
        sample_ids=sample_ids,
        counts=counts,
        reference_median=reference,
        reference_is_subset=reference_mask is not None,
    )


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def plot_id_depth(
    datasets: Mapping[str, Dataset],
    *,
    color_by: str | None = None,
    min_intensity: float = 0.0,
    reference_mask: np.ndarray | None = None,
    show_reference_line: bool = True,
    ylabel: str = "Detected features",
    title: str | None = None,
    legend_title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> IdDepthPlot:
    """Stack a per-sample identification-depth bar chart for each labelled level.

    Parameters
    ----------
    datasets:
        Ordered ``{label: Dataset}`` (insertion order = top-to-bottom panel order, e.g.
        ``{"Protein": ds_prot, "Precursor": ds_prec}``). Every Dataset must be on the
        ``"linear"`` scale (raised otherwise) and have the **same samples in the same
        order** (identical ``metadata.index``); ``n_features`` may differ per level.
        One level is allowed. Order the samples by acquisition order upstream so the
        bars read as run order.
    color_by:
        Metadata column (read from the first level) whose value colors each bar, through
        the project color registry (capped at eight). ``None`` (default) draws uniform
        :data:`UNCATEGORIZED_BAR_COLOR` bars and no legend. Missing values are refused.
    min_intensity:
        Detection threshold; a feature counts as detected when finite and
        ``> min_intensity`` (default ``0.0``).
    reference_mask:
        Optional boolean array over the samples (in the Datasets' row order) selecting
        the subset whose median sets each panel's reference line (e.g. the experimental
        samples). ``None`` uses all samples.
    show_reference_line:
        Draw + annotate the per-panel reference median (default ``True``).
    ylabel:
        Y-axis label for every panel (default ``"Detected features"``).
    title:
        Optional figure suptitle.
    legend_title:
        Title for the companion legend figure; defaults to ``color_by``.
    registry_path:
        Color registry JSON. Default ``state/color_registry.json``.
    persist_colors:
        Write newly assigned categorical colors back to the registry (default ``True``).

    Returns
    -------
    IdDepthPlot
        The main figure, the companion legend figure (``None`` when ``color_by`` is
        ``None``), the :class:`IdDepthResult`, and the ``{value: hex}`` color map.

    Raises
    ------
    IdDepthScaleError
        If any Dataset is not on the ``"linear"`` scale.
    ValueError
        On an empty ``datasets``, a non-2D / empty / non-row-aligned Dataset, levels
        whose samples differ in identity or order, a ``reference_mask`` of the wrong
        length, or a ``color_by`` column that is missing or has missing values.
    CategoricalPaletteExceededError
        If ``color_by`` exceeds the palette's categorical capacity (the >8 guard).
    """
    if len(datasets) == 0:
        raise ValueError("datasets is empty; pass >= 1 labelled Dataset to plot.")

    # Validate + compute BEFORE creating the figure, so the alignment / scale guards
    # (the likely failures) raise with no figure to leak. The only post-figure raise is
    # the registry's >8-category guard, handled by the try/except below.
    sample_ids = _validate_sample_alignment(datasets)
    n_samples = sample_ids.size
    reference_mask = _validate_reference_mask(reference_mask, n_samples)
    reference = next(iter(datasets.values()))
    bar_values = _resolve_color_by(reference, color_by)
    result = _compute_result(
        datasets, sample_ids, min_intensity=min_intensity, reference_mask=reference_mask
    )

    labels = list(datasets.keys())
    title_for_legend = legend_title if legend_title is not None else (color_by or "")

    color_map: dict[str, str] = {}
    legend_figure: Figure | None = None
    with publication_style():
        fig, raw_axes = plt.subplots(
            len(labels),
            1,
            figsize=_figsize(len(labels), n_samples),
            sharex=True,
            constrained_layout=True,
        )
        axes = list(np.atleast_1d(raw_axes))
        try:
            if bar_values is not None:
                color_map = assign_colors(
                    str(color_by),
                    list(bar_values),
                    registry_path=registry_path,
                    persist=persist_colors,
                )
                bar_colors = [color_map[v] for v in bar_values]
            else:
                bar_colors = [UNCATEGORIZED_BAR_COLOR] * n_samples

            for ax, label in zip(axes, labels, strict=True):
                _draw_bars(
                    ax,
                    result.counts[label],
                    bar_colors,
                    reference_median=result.reference_median[label],
                    reference_is_subset=result.reference_is_subset,
                    show_reference_line=show_reference_line,
                    panel_label=label,
                    ylabel=ylabel,
                )

            axes[-1].set_xticks(np.arange(n_samples))
            axes[-1].set_xticklabels(sample_ids, rotation=90, fontsize=5)
            axes[-1].set_xlabel(
                f"sample (acquisition order, n={n_samples})", fontsize=12
            )
            if title is not None:
                fig.suptitle(title, fontsize=15, weight="bold")

            if bar_values is not None:
                legend_figure = _legend_figure(bar_values, color_map, title_for_legend)
        except BaseException:
            plt.close(fig)
            raise

    return IdDepthPlot(
        figure=fig,
        legend_figure=legend_figure,
        result=result,
        color_map=color_map,
    )


def save_id_depth(
    datasets: Mapping[str, Dataset],
    output_dir: str | Path,
    base_name: str,
    *,
    color_by: str | None = None,
    min_intensity: float = 0.0,
    reference_mask: np.ndarray | None = None,
    show_reference_line: bool = True,
    ylabel: str = "Detected features",
    title: str | None = None,
    legend_title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render an identification-depth figure (:func:`plot_id_depth`) and save it.

    Writes the main figure as ``<base>.{svg,png}`` and, when ``color_by`` is set, its
    companion legend as ``<base>.legend.{svg,png}`` via
    :func:`figures.figure_io.save_figure`, and returns their paths. Both figures are
    closed even if saving fails (no leak on a bad ``base_name``/``dpi``).
    """
    plot = plot_id_depth(
        datasets,
        color_by=color_by,
        min_intensity=min_intensity,
        reference_mask=reference_mask,
        show_reference_line=show_reference_line,
        ylabel=ylabel,
        title=title,
        legend_title=legend_title,
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
# Validation (fail loud)
# --------------------------------------------------------------------------- #


def _validate_sample_alignment(datasets: Mapping[str, Dataset]) -> np.ndarray:
    """Refuse stacking panels whose samples differ; return the shared sample ids.

    Every level must be the same samples in the same order (bars line up across
    panels). ``n_features`` may differ between levels. Also refuses a non-2D / empty
    Dataset or a row-misaligned metadata. A sample with zero detected features is *kept*
    (a zero-depth run is exactly the QC signal this plot exists to surface).
    """
    reference_ids: np.ndarray | None = None
    reference_label = ""
    for label, ds in datasets.items():
        abundances = np.asarray(ds.abundances, dtype=float)
        if abundances.ndim != 2:
            raise ValueError(
                f"Dataset {label!r} abundances must be 2D (n_samples, n_features); "
                f"got {abundances.shape}."
            )
        n_samples, n_features = abundances.shape
        if n_samples < 1 or n_features < 1:
            raise ValueError(
                f"Dataset {label!r} is empty ({n_samples} samples x {n_features} "
                f"features); need >= 1 of each to draw a depth bar chart."
            )
        ids = ds.metadata.index.to_numpy().astype(str)
        if ids.shape[0] != n_samples:
            raise ValueError(
                f"Dataset {label!r} has {ids.shape[0]} metadata rows but {n_samples} "
                f"abundance samples; it is not row-aligned."
            )
        if reference_ids is None:
            reference_ids = ids
            reference_label = label
            continue
        if ids.shape != reference_ids.shape or not np.array_equal(ids, reference_ids):
            raise ValueError(
                f"Dataset {label!r} has different samples (or a different order) than "
                f"{reference_label!r}; the stacked levels must be the same samples in "
                f"the same order so the panels align bar-for-bar."
            )
    assert (
        reference_ids is not None
    )  # guaranteed: datasets is non-empty (checked above)
    return reference_ids


def _validate_reference_mask(
    reference_mask: np.ndarray | None, n_samples: int
) -> np.ndarray | None:
    """Coerce + length-check the reference mask (a boolean over the samples)."""
    if reference_mask is None:
        return None
    mask = np.asarray(reference_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != n_samples:
        raise ValueError(
            f"reference_mask must be a 1D boolean array of length n_samples="
            f"{n_samples}; got shape {mask.shape}."
        )
    return mask


def _resolve_color_by(dataset: Dataset, color_by: str | None) -> np.ndarray | None:
    """Validate + extract ``color_by`` as per-sample strings (or ``None``)."""
    if color_by is None:
        return None
    if color_by not in dataset.metadata.columns:
        raise ValueError(
            f"color_by {color_by!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )
    series = dataset.metadata[color_by]
    if bool(series.isna().any()):
        n_bad = int(series.isna().sum())
        raise ValueError(
            f"color_by {color_by!r} has {n_bad} missing value(s); a missing label "
            f"cannot be colored. Resolve or relabel them first."
        )
    return np.asarray(series.to_numpy(), dtype=str)


# --------------------------------------------------------------------------- #
# Layout & drawing
# --------------------------------------------------------------------------- #


def _figsize(n_levels: int, n_samples: int) -> tuple[float, float]:
    """Width scales with sample count; height with the panel stack."""
    width = max(12.0, n_samples * 0.18)
    height = 3.0 * n_levels + 1.2
    return (width, height)


def _draw_bars(
    ax: Axes,
    counts: np.ndarray,
    bar_colors: list[str],
    *,
    reference_median: float,
    reference_is_subset: bool,
    show_reference_line: bool,
    panel_label: str,
    ylabel: str,
) -> None:
    """Draw one bar per sample (colored), with the dashed reference-median line."""
    x = np.arange(counts.size)
    ax.bar(x, counts, color=bar_colors, edgecolor="white", linewidth=0.3, width=0.85)

    if show_reference_line and np.isfinite(reference_median):
        ax.axhline(
            reference_median, color="#333333", linestyle="--", linewidth=1.0, zorder=3
        )
        tag = "ref. median" if reference_is_subset else "median"
        ax.text(
            counts.size - 0.5,
            reference_median,
            f"  {tag} {reference_median:,.0f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    ax.set_xlim(-0.5, counts.size - 0.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(panel_label, loc="left", fontsize=12, weight="bold")
    ax.margins(x=0.01)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# --------------------------------------------------------------------------- #
# Legend figure (rendered separately so it never overlaps the plot)
# --------------------------------------------------------------------------- #


def _legend_figure(
    bar_values: np.ndarray, color_map: dict[str, str], legend_title: str
) -> Figure:
    """Standalone swatch legend: one swatch per ``color_by`` value, first-seen order."""
    seen: list[str] = []
    for value in bar_values.astype(str):
        if value not in seen:
            seen.append(value)
    height = max(1.4, 0.35 * len(seen) + 0.8)
    fig, ax = plt.subplots(figsize=(3.2, height))
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=12,
            markerfacecolor=color_map[value],
            markeredgecolor="none",
        )
        for value in seen
    ]
    ax.legend(
        handles,
        seen,
        title=legend_title or None,
        loc="center",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        ncol=1 if len(seen) <= 6 else 2,
    )
    return fig
