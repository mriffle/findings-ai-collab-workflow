"""Per-sample abundance box plots across processing states for a :class:`Dataset`.

TEMPLATE (lib/) — a *seed* for a project's abundance-boxplot module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

What it draws (one figure): a **stack of per-sample box plots**, one panel per labelled
:class:`Dataset`, sharing the same sample columns. Each box summarizes all of a sample's
feature abundances, so the panel shows the per-sample abundance distribution across the
cohort. The canonical use is comparing **processing states of one dataset** —
``raw`` vs ``normalized`` vs ``batch-corrected`` — read top-to-bottom: in raw data the
per-sample medians wander (loading / depth differences); a good normalization flattens
them to a common level; batch correction removes any residual batch-related shift. The
labels are the caller's, and a single state (one panel) is allowed.

Sample order & box color: boxes are drawn in the :class:`Dataset`'s **row order** and
colored by **column position** with a sequential colormap (default ``"cool"``), so
position encodes order. Order the Dataset by **acquisition order** upstream (the
loader's ``order_by``) and the color gradient makes run-order drift visible — a
left-to-right abundance trend in the raw panel that the normalized panel should erase.
Position /
sequential coloring is the right encoding for an ordinal axis (conventions/
visualization.md), so it deliberately does **not** consume the categorical color budget.

Scale (a WARN, like the PCA / correlation templates, not a hard refuse — unlike the CV
template). Abundances span orders of magnitude, so a box plot is read on a **log or
centered** scale; on a raw **linear** scale a handful of very abundant features stretch
every box and crush the bulk. :func:`plot_abundance_boxplots` therefore **warns**
(:class:`AbundanceBoxplotScaleWarning`) for any state whose ``Dataset.scale`` is not
log-ish (``log2``/``glog2``/``zscore``/...). The fix is to ``log2_transform`` (or VSN)
first; a box plot is mathematically fine on any scale, so it is not refused. Each
panel's y-axis label is derived from that state's recorded ``scale`` tag, so a
median-normalized
``log2`` panel and a VSN ``glog2`` panel are labeled honestly.

Convention wiring (conventions/visualization.md): optional metadata **annotation
stripes** below the boxes (a colored row per dimension, e.g. batch / sample type) are
colored through the project color registry (:mod:`figures.colors`) as their own
namespaces (categorical, capped at eight — the registry raises beyond that) or a
perceptually-uniform colormap (continuous). The samples align across panels, so one
shared stripe block sits beneath the stack. The legend — the acquisition-order colorbar
plus a key per annotation — is rendered as its **own figure** and saved beside the plot
as ``<base>.legend.{svg,png}`` via :func:`figures.figure_io.save_figure`, so nothing
overlaps the data.

**Control samples are rendered separately from experimental samples** (conventions/
visualization.md): subset the :class:`Dataset`(s) *before* plotting and call
:func:`save_abundance_boxplots` once per subset — exactly as the CV / PCA templates
expect exclusions to be applied upstream. This template makes NO study decisions: which
states to stack, which samples belong to which subset, the sample order, and the
annotation/registry namespaces are all the caller's choice. Sample exclusions and
relabelings live in the project copy.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.data_loading import LOG_SCALES, Dataset
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "abundance-boxplot", "version": "0.1"},
    "kind": "module",
    "provides": [
        "AbundanceBoxplotScaleWarning",
        "AbundanceBoxplotResult",
        "AbundanceBoxplotPlot",
        "compute_sample_medians",
        "plot_abundance_boxplots",
        "save_abundance_boxplots",
    ],
    "uses": ["common.data_loading", "figures.colors", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Per-sample abundance box plots from one or more Datasets: one stacked panel "
        "per processing state (raw/normalized/batch-corrected), each a box per sample "
        "over its feature abundances, boxes colored by acquisition-order position to "
        "surface drift. Optional registry-colored categorical + viridis continuous "
        "annotation stripes; warns on a non-log scale (box plots are read on a log "
        "scale); per-panel y-labels derived from the scale tag; dual-export plus a "
        "separate legend image. Uses common.data_loading, figures.colors, "
        "figures.figure_io. Study-agnostic; fail-loud."
    ),
}

# Default sequential colormap for the per-sample boxes (position == acquisition order).
DEFAULT_BOX_COLORMAP = "cool"

# Default perceptually-uniform colormap for continuous annotation stripes.
_CONTINUOUS_CMAP = "viridis"

# Per-scale y-axis label, derived from the Dataset's recorded scale tag so each panel is
# labeled honestly (a median-normalized log2 panel vs a VSN glog2 panel differ).
_SCALE_YLABEL: dict[str, str] = {
    "linear": "{ft} abundance (linear)",
    "log2": "log2 {ft} abundance",
    "log10": "log10 {ft} abundance",
    "ln": "ln {ft} abundance",
    "glog2": "{ft} abundance (glog2 / VSN)",
    "zscore": "{ft} abundance (robust z)",
    "ratio": "{ft} ratio",
}


class AbundanceBoxplotScaleWarning(UserWarning):
    """Warning that an abundance box plot is being drawn on a non-log scale.

    Abundances span orders of magnitude, so a box plot is read on a log / centered
    scale. On a raw linear scale a few very abundant features stretch every box and
    crush the bulk of the distribution. ``log2_transform`` (or VSN) the data first. The
    plot is still produced — a box plot is defined on any scale — so this is a warning,
    not a refusal (contrast the CV template's hard :class:`CVScaleError`).
    """


@dataclass(frozen=True)
class _Annotation:
    """One resolved annotation dimension to draw as a stripe below the boxes."""

    name: str
    continuous: bool
    # Categorical: string values per sample. Continuous: float values per sample.
    per_sample: np.ndarray


@dataclass(frozen=True)
class AbundanceBoxplotResult:
    """The per-sample summaries underlying an abundance box-plot figure.

    Attributes
    ----------
    sample_ids:
        ``(n_samples,)`` the sample identifiers (the shared column order across panels).
    medians:
        ``{label: (n_samples,) per-sample median abundance}`` in panel order. The QC
        signal: raw medians wander, normalization flattens them. ``NaN`` for a sample
        with no finite features (refused upstream, so a backstop only).
    scales:
        ``{label: scale}`` the scale tag each panel was drawn on (drives the y-label).
    """

    sample_ids: np.ndarray
    medians: dict[str, np.ndarray]
    scales: dict[str, str]


@dataclass
class AbundanceBoxplotPlot:
    """A rendered abundance box-plot figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (stacked per-sample box panels + any annotation
        stripes; no baked legend).
    legend_figure:
        A standalone legend (the acquisition-order colorbar + a key per annotation),
        saved beside the main figure as ``<base>.legend.{svg,png}`` by
        :func:`save_abundance_boxplots`.
    result:
        The underlying :class:`AbundanceBoxplotResult`.
    color_maps:
        ``{annotation_name: {value: hex}}`` for each categorical annotation drawn.
    """

    figure: Figure
    legend_figure: Figure
    result: AbundanceBoxplotResult
    color_maps: dict[str, dict[str, str]]


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #


def compute_sample_medians(dataset: Dataset) -> np.ndarray:
    """Per-sample median abundance over the finite features (one value per sample).

    Computed on the :class:`Dataset`'s own scale (the caller is responsible for the
    scale — this is a descriptive reduction, not a scale-sensitive statistic like CV).
    A sample with no finite features yields ``NaN`` rather than raising, so the array
    stays aligned to the samples; the plot path refuses such samples upfront.

    Raises ``ValueError`` if abundances are not 2D.
    """
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(
            f"abundances must be 2D (n_samples, n_features); got {abundances.shape}."
        )
    return np.array([_finite_median(row) for row in abundances], dtype=float)


def _finite_median(row: np.ndarray) -> float:
    """Median of a sample's finite feature values (``NaN`` if it has none)."""
    finite = row[np.isfinite(row)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def plot_abundance_boxplots(
    datasets: Mapping[str, Dataset],
    *,
    categorical_annotations: Sequence[str] = (),
    continuous_annotations: Sequence[str] = (),
    feature_type: str = "feature",
    box_colormap: str = DEFAULT_BOX_COLORMAP,
    continuous_colormap: str = _CONTINUOUS_CMAP,
    show_outliers: bool = True,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> AbundanceBoxplotPlot:
    """Stack a per-sample box plot for each labelled state of one dataset.

    Parameters
    ----------
    datasets:
        Ordered ``{label: Dataset}`` (insertion order = top-to-bottom panel order).
        Every Dataset must have the **same samples in the same order** (identical
        ``metadata.index``); ``n_features`` may differ between states. One state is
        allowed. Order the samples by acquisition order upstream so the box color
        gradient encodes run order.
    categorical_annotations:
        Metadata columns (read from the first state) drawn as registry-colored stripes
        below the boxes, each its own color namespace (capped at eight). Missing values
        are refused.
    continuous_annotations:
        Metadata columns drawn as ``continuous_colormap`` stripes. Values must be
        numeric and present. A column may not appear in both annotation lists.
    feature_type:
        Noun for the features in the y-axis label (e.g. ``"protein"``, ``"precursor"``).
    box_colormap:
        Sequential matplotlib colormap for the per-sample boxes (default ``"cool"``);
        position along it encodes the sample's order.
    continuous_colormap:
        Matplotlib colormap for continuous annotation stripes (default ``"viridis"``).
    show_outliers:
        Draw flier points beyond the whiskers (default ``True``).
    title:
        Optional figure suptitle.
    registry_path:
        Color registry JSON. Default ``state/color_registry.json``.
    persist_colors:
        Write newly assigned categorical colors back to the registry (default ``True``).

    Returns
    -------
    AbundanceBoxplotPlot
        The main figure, the companion legend figure, the
        :class:`AbundanceBoxplotResult`, and the ``{name: {value: hex}}`` categorical
        maps.

    Raises
    ------
    ValueError
        On an empty ``datasets``, a non-2D / empty / non-row-aligned Dataset, states
        whose samples differ in identity or order, a sample with no finite features, or
        an annotation column that is missing / in both lists / has missing values.
    CategoricalPaletteExceededError
        If any categorical annotation exceeds the palette's capacity (the >8 guard).
    """
    if len(datasets) == 0:
        raise ValueError("datasets is empty; pass >= 1 labelled Dataset to plot.")

    # Validate + compute BEFORE creating the figure, so the alignment / finiteness
    # guards (the likely failures) raise with no figure to leak. The only post-figure
    # raise is the registry's >8-category guard, handled by the try/except below.
    sample_ids = _validate_sample_alignment(datasets)
    reference = next(iter(datasets.values()))
    annotations = _resolve_annotations(
        reference, categorical_annotations, continuous_annotations
    )
    result = _compute_result(datasets, sample_ids)
    _warn_non_log(datasets, feature_type=feature_type)

    labels = list(datasets.keys())
    n_samples = sample_ids.size

    color_maps: dict[str, dict[str, str]] = {}
    with publication_style():
        fig = plt.figure(figsize=_figsize(len(labels), len(annotations), n_samples))
        try:
            axes = _build_layout(
                fig, n_states=len(labels), n_annotations=len(annotations)
            )
            if title is not None:
                fig.suptitle(title, fontsize=15, weight="bold")

            for ax, label in zip(axes.panels, labels, strict=True):
                _draw_boxes(
                    ax,
                    datasets[label],
                    state_label=label,
                    feature_type=feature_type,
                    box_colormap=box_colormap,
                    show_outliers=show_outliers,
                )

            color_maps = _draw_annotation_stripes(
                axes.stripes,
                annotations,
                n_samples=n_samples,
                continuous_colormap=continuous_colormap,
                registry_path=registry_path,
                persist_colors=persist_colors,
            )

            bottom = axes.stripes[-1] if axes.stripes else axes.panels[-1]
            bottom.set_xlabel(f"sample (acquisition order, n={n_samples})", fontsize=12)

            legend_figure = _legend_figure(
                annotations,
                color_maps,
                box_colormap=box_colormap,
                continuous_colormap=continuous_colormap,
                n_samples=n_samples,
            )
        except BaseException:
            plt.close(fig)
            raise

    return AbundanceBoxplotPlot(
        figure=fig,
        legend_figure=legend_figure,
        result=result,
        color_maps=color_maps,
    )


def save_abundance_boxplots(
    datasets: Mapping[str, Dataset],
    output_dir: str | Path,
    base_name: str,
    *,
    categorical_annotations: Sequence[str] = (),
    continuous_annotations: Sequence[str] = (),
    feature_type: str = "feature",
    box_colormap: str = DEFAULT_BOX_COLORMAP,
    continuous_colormap: str = _CONTINUOUS_CMAP,
    show_outliers: bool = True,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render an abundance box-plot figure and save it (dual-export + legend image).

    Writes the main figure as ``<base>.{svg,png}`` and its companion legend as
    ``<base>.legend.{svg,png}`` via :func:`figures.figure_io.save_figure`, and returns
    their paths. Both figures are closed even if saving fails (no leak on a bad
    ``base_name``/``dpi``).
    """
    plot = plot_abundance_boxplots(
        datasets,
        categorical_annotations=categorical_annotations,
        continuous_annotations=continuous_annotations,
        feature_type=feature_type,
        box_colormap=box_colormap,
        continuous_colormap=continuous_colormap,
        show_outliers=show_outliers,
        title=title,
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

    Every state must be the same samples in the same order (columns line up across
    panels). ``n_features`` may differ between states. Also refuses a non-2D / empty
    Dataset, a row-misaligned metadata, or a sample with no finite features (it cannot
    be boxplotted and signals a sample that should have been dropped upstream).
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
                f"features); need >= 1 of each to draw a box plot."
            )
        ids = ds.metadata.index.to_numpy().astype(str)
        if ids.shape[0] != n_samples:
            raise ValueError(
                f"Dataset {label!r} has {ids.shape[0]} metadata rows but {n_samples} "
                f"abundance samples; it is not row-aligned."
            )
        finite_counts = np.isfinite(abundances).sum(axis=1)
        if bool((finite_counts == 0).any()):
            empty = ids[finite_counts == 0].tolist()
            raise ValueError(
                f"Dataset {label!r}: {len(empty)} sample(s) have no finite feature "
                f"values, so they cannot be boxplotted: {empty[:5]}. Drop them "
                f"upstream."
            )
        if reference_ids is None:
            reference_ids = ids
            reference_label = label
            continue
        if ids.shape != reference_ids.shape or not np.array_equal(ids, reference_ids):
            raise ValueError(
                f"Dataset {label!r} has different samples (or a different order) than "
                f"{reference_label!r}; the stacked states must be the same samples in "
                f"the same order so the panels align column-for-column."
            )
    assert (
        reference_ids is not None
    )  # guaranteed: datasets is non-empty (checked above)
    return reference_ids


def _resolve_annotations(
    dataset: Dataset,
    categorical: Sequence[str],
    continuous: Sequence[str],
) -> list[_Annotation]:
    """Validate + extract each annotation column from the reference state's metadata."""
    overlap = set(categorical) & set(continuous)
    if overlap:
        raise ValueError(
            f"annotation column(s) {sorted(overlap)} appear in both "
            f"categorical_annotations and continuous_annotations; choose one."
        )
    resolved: list[_Annotation] = []
    for name in categorical:
        if name not in dataset.metadata.columns:
            raise ValueError(
                f"categorical annotation {name!r} is not a metadata column "
                f"{list(dataset.metadata.columns)}."
            )
        series = dataset.metadata[name]
        if bool(series.isna().any()):
            n_bad = int(series.isna().sum())
            raise ValueError(
                f"categorical annotation {name!r} has {n_bad} missing value(s); a "
                f"missing label cannot be colored. Resolve or relabel them first."
            )
        resolved.append(
            _Annotation(
                name=name, continuous=False, per_sample=series.to_numpy().astype(str)
            )
        )
    for name in continuous:
        if name not in dataset.metadata.columns:
            raise ValueError(
                f"continuous annotation {name!r} is not a metadata column "
                f"{list(dataset.metadata.columns)}."
            )
        numeric = pd.to_numeric(dataset.metadata[name], errors="coerce")
        if bool(numeric.isna().any()):
            n_bad = int(numeric.isna().sum())
            raise ValueError(
                f"continuous annotation {name!r} has {n_bad} value(s) that are missing "
                f"or non-numeric; cannot map them to a colormap."
            )
        resolved.append(
            _Annotation(
                name=name, continuous=True, per_sample=numeric.to_numpy(dtype=float)
            )
        )
    return resolved


def _compute_result(
    datasets: Mapping[str, Dataset], sample_ids: np.ndarray
) -> AbundanceBoxplotResult:
    """Per-sample medians + scale tag for each state (the QC signal + provenance)."""
    medians = {label: compute_sample_medians(ds) for label, ds in datasets.items()}
    scales = {label: str(ds.scale) for label, ds in datasets.items()}
    return AbundanceBoxplotResult(sample_ids=sample_ids, medians=medians, scales=scales)


def _warn_non_log(datasets: Mapping[str, Dataset], *, feature_type: str) -> None:
    """Warn for any state on a non-log scale (box plots are read on a log scale)."""
    non_log = [label for label, ds in datasets.items() if ds.scale not in LOG_SCALES]
    if non_log:
        warnings.warn(
            f"Abundance box plot state(s) {non_log} are on a non-log scale "
            f"(not one of "
            f"{sorted(LOG_SCALES)}). {feature_type.capitalize()} abundances span "
            f"orders "
            f"of magnitude, so on a linear scale a few abundant features stretch every "
            f"box and crush the bulk; log2_transform (or VSN) first. The plot is still "
            f"drawn (a box plot is defined on any scale).",
            AbundanceBoxplotScaleWarning,
            stacklevel=2,
        )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Axes:
    """The axes of the abundance-boxplot layout."""

    panels: list[Axes]
    stripes: list[Axes]


def _figsize(n_states: int, n_annotations: int, n_samples: int) -> tuple[float, float]:
    """Width scales with sample count; height with the panel + stripe stack."""
    width = max(12.0, n_samples * 0.18)
    height = 3.2 * n_states + 0.45 * n_annotations + 1.0
    return (width, height)


def _build_layout(fig: Figure, *, n_states: int, n_annotations: int) -> _Axes:
    """Stack: [box panel per state...] then [annotation stripe per dimension...].

    A single-column gridspec with a tall row per state and a thin row per annotation.
    All rows share the first panel's x-axis, so every panel and stripe aligns
    column-for-column on the same samples.
    """
    n_rows = n_states + n_annotations
    panel_h = 16.0
    stripe_h = 1.2
    height_ratios = [panel_h] * n_states + [stripe_h] * n_annotations

    gs = GridSpec(
        n_rows,
        1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.18,
        left=0.10,
        right=0.97,
        top=0.93,
        bottom=0.10,
    )

    first = fig.add_subplot(gs[0, 0])
    panels: list[Axes] = [first]
    for i in range(1, n_states):
        panels.append(fig.add_subplot(gs[i, 0], sharex=first))

    stripes: list[Axes] = [
        fig.add_subplot(gs[n_states + j, 0], sharex=first) for j in range(n_annotations)
    ]
    return _Axes(panels=panels, stripes=stripes)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _draw_boxes(
    ax: Axes,
    dataset: Dataset,
    *,
    state_label: str,
    feature_type: str,
    box_colormap: str,
    show_outliers: bool,
) -> None:
    """Draw one box per sample, colored by position, with a black median line."""
    abundances = np.asarray(dataset.abundances, dtype=float)
    n_samples = abundances.shape[0]
    per_sample = [row[np.isfinite(row)] for row in abundances]

    box = ax.boxplot(
        per_sample,
        positions=np.arange(n_samples),
        widths=0.6,
        showfliers=show_outliers,
        patch_artist=True,
        flierprops={
            "marker": ".",
            "markersize": 4,
            "markeredgecolor": "none",
            "alpha": 0.3,
        },
    )

    # Color each box's artists by position (acquisition order) on a sequential colormap.
    cmap = plt.get_cmap(box_colormap)
    denom = max(n_samples - 1, 1)
    fliers = box["fliers"] if show_outliers else []
    for i in range(n_samples):
        color = cmap(i / denom)
        box["boxes"][i].set_facecolor(color)
        box["boxes"][i].set_edgecolor(color)
        for whisker in box["whiskers"][2 * i : 2 * i + 2]:
            whisker.set_color(color)
        for cap in box["caps"][2 * i : 2 * i + 2]:
            cap.set_color(color)
        if i < len(fliers):
            fliers[i].set_markerfacecolor(color)

    # Median line stays black for readability against the colored box.
    for median_line in box["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(1.2)

    ax.set_xticks([])
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylabel(_scale_ylabel(str(dataset.scale), feature_type), fontsize=11)
    ax.set_title(state_label, loc="left", fontsize=12, weight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _scale_ylabel(scale: str, feature_type: str) -> str:
    """Y-axis label derived from the state's recorded scale tag."""
    template = _SCALE_YLABEL.get(scale, "{ft} abundance")
    return template.format(ft=feature_type)


def _draw_annotation_stripes(
    axes: list[Axes],
    annotations: list[_Annotation],
    *,
    n_samples: int,
    continuous_colormap: str,
    registry_path: str | Path,
    persist_colors: bool,
) -> dict[str, dict[str, str]]:
    """Draw one colored stripe per annotation; return the categorical color maps."""
    color_maps: dict[str, dict[str, str]] = {}
    for ax, annotation in zip(axes, annotations, strict=True):
        values = annotation.per_sample
        if annotation.continuous:
            rgba = _continuous_rgba(values.astype(float), continuous_colormap)
        else:
            color_map = assign_colors(
                annotation.name,
                [str(v) for v in values],
                registry_path=registry_path,
                persist=persist_colors,
            )
            color_maps[annotation.name] = color_map
            rgba = np.array([to_rgba(color_map[str(v)]) for v in values])

        ax.imshow(
            rgba.reshape(1, -1, 4),
            aspect="auto",
            interpolation="none",
            extent=(-0.5, n_samples - 0.5, -0.5, 0.5),
        )
        ax.set_xlim(-0.5, n_samples - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(
            annotation.name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=11,
            labelpad=10,
        )
    return color_maps


def _continuous_rgba(values: np.ndarray, colormap: str) -> np.ndarray:
    """Map a continuous annotation to RGBA via ``colormap`` (constant -> mid-color)."""
    vmin = float(values.min())
    vmax = float(values.max())
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
    cmap = plt.get_cmap(colormap)
    return np.asarray(cmap(norm(values)), dtype=float)


# --------------------------------------------------------------------------- #
# Legend figure (rendered separately so it never overlaps the plot)
# --------------------------------------------------------------------------- #


def _legend_figure(
    annotations: list[_Annotation],
    color_maps: dict[str, dict[str, str]],
    *,
    box_colormap: str,
    continuous_colormap: str,
    n_samples: int,
) -> Figure:
    """A standalone legend: the acquisition-order colorbar + a key per annotation.

    Row 0 is always the box colormap as a horizontal colorbar (1..n, the sample order
    the boxes encode). Each subsequent row is one annotation: a titled swatch block
    (categorical) or a horizontal colorbar (continuous).
    """
    n_rows = 1 + len(annotations)
    fig, raw_axes = plt.subplots(n_rows, 1, figsize=(4.0, max(1.6, 1.2 * n_rows)))
    axes_list = list(np.atleast_1d(raw_axes))

    order_norm = Normalize(vmin=1.0, vmax=float(max(n_samples, 2)))
    order_mappable = ScalarMappable(cmap=plt.get_cmap(box_colormap), norm=order_norm)
    order_mappable.set_array(np.asarray([], dtype=float))
    order_cbar = fig.colorbar(
        order_mappable, cax=axes_list[0], orientation="horizontal"
    )
    order_cbar.set_label("sample order (acquisition →)", fontsize=11)

    for ax, annotation in zip(axes_list[1:], annotations, strict=True):
        if annotation.continuous:
            values = annotation.per_sample.astype(float)
            vmin = float(values.min())
            vmax = float(values.max())
            norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
            mappable = ScalarMappable(cmap=plt.get_cmap(continuous_colormap), norm=norm)
            mappable.set_array(np.asarray([], dtype=float))
            cbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
            cbar.set_label(annotation.name, fontsize=12)
        else:
            ax.axis("off")
            color_map = color_maps[annotation.name]
            seen: list[str] = []
            for value in annotation.per_sample.astype(str):
                if value not in seen:
                    seen.append(value)
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    markersize=11,
                    markerfacecolor=color_map[value],
                    markeredgecolor="none",
                )
                for value in seen
            ]
            ax.legend(
                handles,
                seen,
                title=annotation.name,
                loc="center",
                frameon=True,
                fontsize=11,
                title_fontsize=12,
                ncol=1 if len(seen) <= 4 else 2,
            )
    fig.tight_layout()
    return fig
