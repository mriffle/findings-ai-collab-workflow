"""Sample-vs-sample correlation heatmaps for a :class:`Dataset` (a sanity check).

TEMPLATE (lib/) — a *seed* for a project's correlation-plotting module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

What it draws (one figure): a hierarchically-clustered **sample x sample** correlation
heatmap — each cell is the correlation between two samples' whole feature profiles —
with one or more **annotation stripes** above it (a colored row per metadata dimension,
e.g. genotype / sex / sample type) so you can read off whether the clustering lines up
with the design. The purpose is a *sanity check*, not a hypothesis test: experimental
samples should generally correlate well (a shared proteomic background), control /
reference samples of the **same kind** should correlate with each other even more
tightly, and a sample that clusters far from its peers is a flag worth chasing. Some
individual features legitimately diverge (a drug responder), but the bulk background
should not look vastly different sample to sample.

Method & scale (a WARN, like the PCA template, not a hard refuse). ``"pearson"`` (the
default) measures linear agreement; on a **linear** intensity scale it is dominated by a
handful of very abundant features, so a single outlier on those can drag a pair's r down
to a misleadingly low value even when the backgrounds are alike.
:func:`plot_sample_correlation` therefore **warns** (:class:`CorrelationScaleWarning`)
when Pearson runs on a non-log scale — normalize + ``log2_transform`` (or VSN) first.
``"spearman"`` ranks first, so it is scale-invariant and never warns; it is the
robust choice when abundant features dominate. Correlation is mathematically defined on
any scale, so neither is refused — the choice is the scientist's.

Convention wiring (conventions/visualization.md): each **categorical** annotation
dimension is colored through the project color registry (:mod:`figures.colors`) as its
own namespace, so a value keeps its color across every figure, capped at eight (the
registry raises beyond that); **continuous** dimensions use a perceptually-uniform
colormap. The heatmap's own correlation colorbar is the primary data's value scale and
stays on the main figure; the annotation **keys** (a swatch block per categorical
dimension + a colorbar per continuous one) are rendered as a **separate legend figure**
and saved beside the plot as ``<base>.legend.{svg,png}`` via
:func:`figures.figure_io.save_figure`, so nothing overlaps the heatmap.

**Control samples appear *with* the experimental samples here (a deliberate divergence
from the "render controls separately" rule).** For most QC plots a control's tight
cluster would distort the experimental spread, so they are split out; but this plot's
entire value is the cross-class comparison — *do the controls cluster with each other
and apart from the biology?* — so the canonical figure shows every sample together and
makes the classes **visibly distinct via a Sample Type annotation stripe** (preferring
the control *subtype*, since pooled-digest / pooled-lysate / brain references are
different matrices and should each self-cluster). That satisfies the convention's intent
("visibly distinct, never silently merged"). This template makes NO study decisions:
which columns annotate, the correlation method, and the registry namespaces are all the
caller's choice. Sample exclusions and relabelings live in the project copy, applied to
the ``Dataset`` first.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

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
from scipy.cluster.hierarchy import dendrogram, linkage

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "sample-correlation", "version": "0.1"},
    "kind": "module",
    "provides": [
        "CorrelationScaleWarning",
        "CorrelationMethod",
        "CorrelationResult",
        "CorrelationPlot",
        "compute_correlation",
        "plot_sample_correlation",
        "save_sample_correlation",
    ],
    "uses": ["common.data_loading", "figures.colors", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Sample-vs-sample correlation heatmap from a Dataset: hierarchically-clustered "
        "(scipy average linkage) Pearson/Spearman correlation of whole sample "
        "profiles, with registry-colored categorical + viridis continuous stripes for "
        "a quick design sanity check (do experimental samples cohere, do controls "
        "self-cluster). Warns on Pearson over a non-log scale (abundant-feature "
        "dominance); the >8-category guard applies per annotation dimension; the "
        "correlation colorbar stays on the figure while annotation keys are a separate "
        "legend image; dual-export. Uses common.data_loading, figures.colors, "
        "figures.figure_io. Study-agnostic; fail-loud."
    ),
}

# The correlation methods supported. Kendall is intentionally omitted: it is
# O(n^2 log n) per pair and adds nothing over Spearman for a sample-similarity check.
CorrelationMethod = str  # one of {"pearson", "spearman"}; validated in compute.
_METHODS: frozenset[str] = frozenset({"pearson", "spearman"})

# Heatmap colormap (diverging, reversed so high r is warm) + the fixed upper bound. vmin
# is the off-diagonal minimum (the diagonal is always 1, so anchoring vmin to it would
# waste the whole color range) — set per figure in :func:`_draw_heatmap`.
_HEATMAP_CMAP = "RdYlBu_r"
_VMAX = 1.0

# Linkage defaults — Euclidean distance between samples' correlation-profile rows with
# average linkage (the seaborn-clustermap default the source used). Exposed as args.
_LINKAGE_METRIC = "euclidean"
_LINKAGE_METHOD = "average"

# Pretty symbol for the colorbar label, by method.
_METHOD_SYMBOL: dict[str, str] = {"pearson": "Pearson r", "spearman": "Spearman rho"}

# Default perceptually-uniform colormap for continuous annotation stripes.
_CONTINUOUS_CMAP = "viridis"

# Color drawn for a NaN heatmap cell (a sample with no defined correlation). Constant
# samples are refused upstream, so this is only a visual backstop.
_BAD_COLOR = "#bdbdbd"


class CorrelationScaleWarning(UserWarning):
    """Warning that Pearson correlation is running on a non-log (e.g. linear) scale.

    On a linear intensity scale Pearson is dominated by the few most abundant features,
    so a pair of biologically-similar samples can show a misleadingly low r. Normalize +
    ``log2_transform`` (or VSN) first, or use ``method="spearman"`` (rank-based, so
    scale-invariant).
    """


@dataclass(frozen=True)
class _Annotation:
    """One resolved annotation dimension to draw as a stripe above the heatmap."""

    name: str
    continuous: bool
    # Categorical: string values per sample. Continuous: float values per sample.
    per_sample: np.ndarray


@dataclass(frozen=True)
class CorrelationResult:
    """The sample-vs-sample correlation underlying the figure.

    Attributes
    ----------
    matrix:
        ``(n_samples, n_samples)`` correlation matrix in the **input** sample order
        (NOT reordered by clustering). Symmetric, unit diagonal.
    sample_ids:
        ``(n_samples,)`` the unique sample identifiers (the matrix row/column order).
    method:
        ``"pearson"`` or ``"spearman"``.
    order:
        ``(n_samples,)`` int index giving the clustered display order (``leaves`` of the
        linkage). Identity (``0..n-1``) when ``cluster=False``.
    """

    matrix: np.ndarray
    sample_ids: np.ndarray
    method: str
    order: np.ndarray


@dataclass
class CorrelationPlot:
    """A rendered correlation figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (clustered heatmap + annotation stripes + the
        correlation colorbar; no baked annotation legend).
    legend_figure:
        A standalone legend (swatch block per categorical dimension + colorbar per
        continuous one), saved beside the main figure as ``<base>.legend.{svg,png}`` by
        :func:`save_sample_correlation`. ``None`` when there are no annotations.
    result:
        The underlying :class:`CorrelationResult`.
    color_maps:
        ``{annotation_name: {value: hex}}`` for each categorical annotation drawn.
    """

    figure: Figure
    legend_figure: Figure | None
    result: CorrelationResult
    color_maps: dict[str, dict[str, str]]


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #


def compute_correlation(
    dataset: Dataset,
    *,
    method: CorrelationMethod = "pearson",
    sample_label_column: str | None = None,
    linkage_metric: str = _LINKAGE_METRIC,
    linkage_method: str = _LINKAGE_METHOD,
    cluster: bool = True,
) -> CorrelationResult:
    """Correlate every pair of samples over their full feature profiles.

    The correlation is computed between samples (columns of ``abundances.T``) over all
    features, exactly as a sample-similarity QC would: high off-diagonal values mean the
    samples share a proteomic background.

    Parameters
    ----------
    dataset:
        The data. ``abundances`` must be finite (resolve missingness first). The sample
        identifiers come from ``dataset.metadata.index`` and must be unique.
    method:
        ``"pearson"`` (default) or ``"spearman"``.
    sample_label_column:
        Unused for the computation (kept for symmetry with the plot API); validated
        there. Present here only so a caller can fail fast on a bad column name.
    linkage_metric, linkage_method:
        Passed to :func:`scipy.cluster.hierarchy.linkage` over the rows of the
        correlation matrix (default Euclidean distance + average linkage — the
        clustering the source seaborn clustermap used). Ignored when ``cluster=False``.
    cluster:
        Cluster the samples (default ``True``); the returned ``order`` is the linkage
        leaf order. ``False`` keeps the input order (``order`` = identity).

    Returns
    -------
    CorrelationResult
        The (input-order) matrix, the sample ids, the method, and the display ``order``.

    Raises
    ------
    ValueError
        On an unknown ``method``, non-2D / non-finite abundances, fewer than two
        samples, non-unique sample ids, a missing ``sample_label_column``, or a sample
        of zero variance across features (its correlations and the clustering are
        undefined — drop or fix it).
    """
    if method not in _METHODS:
        raise ValueError(
            f"method {method!r} is not supported; use one of {sorted(_METHODS)}."
        )
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(
            f"abundances must be 2D (n_samples, n_features); got {abundances.shape}."
        )
    n_samples = abundances.shape[0]
    if n_samples < 2:
        raise ValueError(f"correlation needs >= 2 samples to compare; got {n_samples}.")
    if not np.isfinite(abundances).all():
        n_bad = int((~np.isfinite(abundances)).sum())
        raise ValueError(
            f"compute_correlation requires finite abundances but found {n_bad} "
            f"non-finite value(s) (NaN/Inf). Resolve missingness first (the loader's "
            f"missing-value policy / imputation / normalization)."
        )

    sample_ids = dataset.metadata.index.to_numpy().astype(str)
    if len(sample_ids) != n_samples:
        raise ValueError(
            f"metadata has {len(sample_ids)} rows but abundances has {n_samples} "
            f"samples; the Dataset is not row-aligned."
        )
    duplicate_ids = sample_ids[pd.Index(sample_ids).duplicated(keep=False)]
    if duplicate_ids.size:
        raise ValueError(
            f"sample identifiers (metadata index) must be unique to label a "
            f"correlation matrix; duplicates: {sorted(set(duplicate_ids.tolist()))}."
        )
    if sample_label_column is not None and sample_label_column not in (
        dataset.metadata.columns
    ):
        raise ValueError(
            f"sample_label_column {sample_label_column!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )

    # A sample with no spread across features has an undefined correlation with anyone
    # (std 0 -> 0/0), which seeds NaNs into the matrix and breaks linkage. Fail loud.
    constant = np.nanstd(abundances, axis=1) == 0.0
    if bool(constant.any()):
        bad = sample_ids[constant].tolist()
        raise ValueError(
            f"{len(bad)} sample(s) are constant across all features, so their "
            f"correlation is undefined: {bad[:5]}. Drop or fix them before correlating."
        )

    frame = pd.DataFrame(abundances.T, columns=pd.Index(sample_ids))
    # method is runtime-validated against _METHODS above; narrow it to pandas' literal.
    corr_method = cast(Literal["pearson", "kendall", "spearman"], method)
    matrix = frame.corr(method=corr_method).to_numpy(dtype=float)

    if cluster:
        order = _linkage_order(matrix, linkage_metric, linkage_method)
    else:
        order = np.arange(n_samples)
    return CorrelationResult(
        matrix=matrix, sample_ids=sample_ids, method=method, order=order
    )


def _linkage_order(matrix: np.ndarray, metric: str, method: str) -> np.ndarray:
    """Leaf order of an average-linkage clustering over the correlation-profile rows."""
    linkage_z = linkage(matrix, metric=metric, method=method)
    leaves = dendrogram(linkage_z, no_plot=True)["leaves"]
    return np.asarray(leaves, dtype=int)


def _linkage_matrix(matrix: np.ndarray, metric: str, method: str) -> np.ndarray:
    """The raw linkage ``Z`` (for drawing the dendrogram aligned to the heatmap)."""
    return np.asarray(linkage(matrix, metric=metric, method=method), dtype=float)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def plot_sample_correlation(
    dataset: Dataset,
    *,
    categorical_annotations: Sequence[str] = (),
    continuous_annotations: Sequence[str] = (),
    method: CorrelationMethod = "pearson",
    cluster: bool = True,
    sample_label_column: str | None = None,
    feature_type: str = "feature",
    continuous_colormap: str = _CONTINUOUS_CMAP,
    linkage_metric: str = _LINKAGE_METRIC,
    linkage_method: str = _LINKAGE_METHOD,
    show_tick_labels: bool = True,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> CorrelationPlot:
    """Render a clustered sample-correlation heatmap with annotation stripes.

    Parameters
    ----------
    dataset:
        The data to correlate. Abundances must be finite; Pearson on a non-log scale
        warns (see the module docstring).
    categorical_annotations:
        Metadata columns drawn as registry-colored stripes (each its own color
        namespace, capped at eight values). Missing values are refused.
    continuous_annotations:
        Metadata columns drawn as ``continuous_colormap`` stripes. Values must be
        numeric and present. A column may not appear in both annotation lists.
    method:
        ``"pearson"`` (default) or ``"spearman"``.
    cluster:
        Hierarchically cluster + reorder the samples (default ``True``) and draw a
        dendrogram; ``False`` keeps the input order with no dendrogram.
    sample_label_column:
        Metadata column to use for axis tick labels; default is the sample id (index).
    feature_type:
        Noun for the features in the title (e.g. ``"protein"``, ``"precursor"``).
    continuous_colormap:
        Matplotlib colormap for continuous annotation stripes (default ``"viridis"``).
    linkage_metric, linkage_method:
        Clustering controls (see :func:`compute_correlation`).
    show_tick_labels:
        Draw the per-sample tick labels (default ``True``); turn off for very large n.
    title:
        Optional figure suptitle.
    registry_path:
        Color registry JSON. Default ``state/color_registry.json``.
    persist_colors:
        Write newly assigned categorical colors back to the registry (default ``True``).

    Returns
    -------
    CorrelationPlot
        The main figure, the companion legend figure (``None`` if no annotations), the
        :class:`CorrelationResult`, and the ``{name: {value: hex}}`` categorical maps.

    Raises
    ------
    ValueError
        On a bad ``method``, an annotation column that is missing / in both lists / has
        missing values, a non-finite or constant sample (see
        :func:`compute_correlation`).
    CategoricalPaletteExceededError
        If any categorical annotation exceeds the palette's capacity (the >8 guard).
    """
    annotations = _resolve_annotations(
        dataset, categorical_annotations, continuous_annotations
    )
    # Compute (and run the constant-sample / finiteness / id-uniqueness guards) BEFORE
    # creating the figure, so the likely failures raise with no figure to leak.
    result = compute_correlation(
        dataset,
        method=method,
        sample_label_column=sample_label_column,
        linkage_metric=linkage_metric,
        linkage_method=linkage_method,
        cluster=cluster,
    )
    if method == "pearson" and dataset.scale not in LOG_SCALES:
        warnings.warn(
            f"Pearson correlation is running on scale {dataset.scale!r}, which is not "
            f"log-ish {sorted(LOG_SCALES)}. On a linear scale a few very abundant "
            f"{feature_type}s dominate, so r can read misleadingly low for "
            f"biologically-similar samples; normalize + log2_transform (or VSN) first, "
            f"or use method='spearman' (rank-based, scale-invariant).",
            CorrelationScaleWarning,
            stacklevel=2,
        )

    order = result.order
    matrix_ordered = result.matrix[np.ix_(order, order)]
    display_labels = _display_labels(dataset, sample_label_column)[order]

    linkage_z = (
        _linkage_matrix(result.matrix, linkage_metric, linkage_method)
        if cluster
        else None
    )

    color_maps: dict[str, dict[str, str]] = {}
    with publication_style():
        fig = plt.figure(figsize=_figsize(len(annotations)))
        try:
            axes = _build_layout(fig, n_annotations=len(annotations), cluster=cluster)
            if title is not None:
                fig.suptitle(title, fontsize=16, weight="bold", y=0.99)

            if linkage_z is not None and axes.dendrogram is not None:
                _draw_dendrogram(axes.dendrogram, linkage_z, n_samples=len(order))

            color_maps = _draw_annotation_stripes(
                axes.stripes,
                annotations,
                order=order,
                continuous_colormap=continuous_colormap,
                registry_path=registry_path,
                persist_colors=persist_colors,
            )

            _draw_heatmap(
                axes.heatmap,
                axes.colorbar,
                matrix_ordered,
                display_labels=display_labels,
                method=result.method,
                feature_type=feature_type,
                show_tick_labels=show_tick_labels,
            )

            legend_figure = (
                _legend_figure(annotations, color_maps, continuous_colormap)
                if annotations
                else None
            )
        except BaseException:
            plt.close(fig)
            raise

    return CorrelationPlot(
        figure=fig,
        legend_figure=legend_figure,
        result=result,
        color_maps=color_maps,
    )


def save_sample_correlation(
    dataset: Dataset,
    output_dir: str | Path,
    base_name: str,
    *,
    categorical_annotations: Sequence[str] = (),
    continuous_annotations: Sequence[str] = (),
    method: CorrelationMethod = "pearson",
    cluster: bool = True,
    sample_label_column: str | None = None,
    feature_type: str = "feature",
    continuous_colormap: str = _CONTINUOUS_CMAP,
    linkage_metric: str = _LINKAGE_METRIC,
    linkage_method: str = _LINKAGE_METHOD,
    show_tick_labels: bool = True,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render a correlation figure (:func:`plot_sample_correlation`) and save it.

    Writes ``<base>.{svg,png}`` and, when there are annotations, the companion legend
    ``<base>.legend.{svg,png}`` via :func:`figures.figure_io.save_figure`. Both figures
    are closed even if saving fails (no leak on a bad ``base_name``/``dpi``).
    """
    plot = plot_sample_correlation(
        dataset,
        categorical_annotations=categorical_annotations,
        continuous_annotations=continuous_annotations,
        method=method,
        cluster=cluster,
        sample_label_column=sample_label_column,
        feature_type=feature_type,
        continuous_colormap=continuous_colormap,
        linkage_metric=linkage_metric,
        linkage_method=linkage_method,
        show_tick_labels=show_tick_labels,
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
# Annotation resolution (fail loud)
# --------------------------------------------------------------------------- #


def _resolve_annotations(
    dataset: Dataset,
    categorical: Sequence[str],
    continuous: Sequence[str],
) -> list[_Annotation]:
    """Validate + extract each annotation column; categorical first, then continuous."""
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
        if series.isna().any():
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
        if numeric.isna().any():
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


def _display_labels(dataset: Dataset, sample_label_column: str | None) -> np.ndarray:
    """Per-sample axis tick labels: ``sample_label_column`` if given, else the index."""
    if sample_label_column is None:
        return dataset.metadata.index.to_numpy().astype(str)
    return dataset.metadata[sample_label_column].to_numpy().astype(str)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Axes:
    """The axes of the correlation layout."""

    heatmap: Axes
    colorbar: Axes
    stripes: list[Axes]
    dendrogram: Axes | None


def _figsize(n_annotations: int) -> tuple[float, float]:
    """Figure size: a square-ish heatmap plus a little height per annotation stripe."""
    return (16.0, 14.0 + 0.35 * n_annotations)


def _build_layout(fig: Figure, *, n_annotations: int, cluster: bool) -> _Axes:
    """Stack: [dendrogram?] / [annotation stripes...] / [heatmap], + a side colorbar.

    A 2-column gridspec (content + a thin colorbar column) with one row for the optional
    dendrogram, one per annotation stripe, and a tall heatmap row. Every content row
    shares the heatmap's x-extent, so stripes and dendrogram align column-for-column
    with the heatmap (the alignment is enforced explicitly by each drawer's xlim).
    """
    has_dendro = cluster
    n_rows = (1 if has_dendro else 0) + n_annotations + 1

    dendro_h = 2.2
    stripe_h = 0.45
    heatmap_h = 16.0
    height_ratios: list[float] = []
    if has_dendro:
        height_ratios.append(dendro_h)
    height_ratios.extend([stripe_h] * n_annotations)
    height_ratios.append(heatmap_h)

    gs = GridSpec(
        n_rows,
        2,
        figure=fig,
        width_ratios=[40, 1],
        height_ratios=height_ratios,
        hspace=0.06,
        wspace=0.02,
        left=0.18,
        right=0.90,
        top=0.95,
        bottom=0.12,
    )

    row = 0
    dendro_ax: Axes | None = None
    if has_dendro:
        dendro_ax = fig.add_subplot(gs[row, 0])
        dendro_ax.axis("off")
        row += 1

    stripe_axes: list[Axes] = []
    for _ in range(n_annotations):
        ax = fig.add_subplot(gs[row, 0])
        stripe_axes.append(ax)
        row += 1

    heatmap_ax = fig.add_subplot(gs[row, 0])
    colorbar_ax = fig.add_subplot(gs[row, 1])
    return _Axes(
        heatmap=heatmap_ax,
        colorbar=colorbar_ax,
        stripes=stripe_axes,
        dendrogram=dendro_ax,
    )


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _draw_dendrogram(ax: Axes, linkage_z: np.ndarray, *, n_samples: int) -> None:
    """Draw the linkage as a top dendrogram, scaled to the heatmap's column coordinates.

    scipy places leaf ``i`` at x = ``10*i + 5``; the heatmap (imshow, default extent)
    places column ``i`` at integer ``i``. So each link's x coordinates are mapped by
    ``(x - 5) / 10`` to land exactly over the heatmap columns.
    """
    dendro = dendrogram(linkage_z, no_plot=True)
    for xs, ys in zip(dendro["icoord"], dendro["dcoord"], strict=True):
        x_mapped = [(x - 5.0) / 10.0 for x in xs]
        ax.plot(x_mapped, ys, color="#444444", linewidth=1.0)
    ax.set_xlim(-0.5, n_samples - 0.5)
    ax.set_ylim(bottom=0.0)
    ax.axis("off")


def _draw_annotation_stripes(
    axes: list[Axes],
    annotations: list[_Annotation],
    *,
    order: np.ndarray,
    continuous_colormap: str,
    registry_path: str | Path,
    persist_colors: bool,
) -> dict[str, dict[str, str]]:
    """Draw one colored stripe per annotation; return the categorical color maps."""
    color_maps: dict[str, dict[str, str]] = {}
    for ax, annotation in zip(axes, annotations, strict=True):
        ordered = annotation.per_sample[order]
        if annotation.continuous:
            rgba = _continuous_rgba(ordered, continuous_colormap)
        else:
            color_map = assign_colors(
                annotation.name,
                [str(v) for v in annotation.per_sample],
                registry_path=registry_path,
                persist=persist_colors,
            )
            color_maps[annotation.name] = color_map
            rgba = np.array([to_rgba(color_map[str(v)]) for v in ordered])

        ax.imshow(
            rgba.reshape(1, -1, 4),
            aspect="auto",
            interpolation="none",
            extent=(-0.5, len(ordered) - 0.5, -0.5, 0.5),
        )
        ax.set_xlim(-0.5, len(ordered) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Annotation name to the left of its stripe.
        ax.set_ylabel(
            annotation.name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=12,
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


def _draw_heatmap(
    ax: Axes,
    colorbar_ax: Axes,
    matrix: np.ndarray,
    *,
    display_labels: np.ndarray,
    method: str,
    feature_type: str,
    show_tick_labels: bool,
) -> None:
    """Draw the correlation heatmap (off-diagonal-min vmin) + its colorbar."""
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    finite_off = off_diagonal[np.isfinite(off_diagonal)]
    vmin = float(finite_off.min()) if finite_off.size else 0.0

    cmap = plt.get_cmap(_HEATMAP_CMAP).with_extremes(bad=_BAD_COLOR)
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=_VMAX,
        aspect="auto",
        interpolation="none",
    )

    n = len(matrix)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # origin upper (row 0 at top), matching imshow
    if show_tick_labels:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display_labels, rotation=90, fontsize=5)
        ax.set_yticklabels(display_labels, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel(f"sample ({feature_type} profile)", fontsize=13)

    cbar = ax.figure.colorbar(image, cax=colorbar_ax)
    cbar.set_label(_METHOD_SYMBOL.get(method, method), fontsize=13)


# --------------------------------------------------------------------------- #
# Legend figure (rendered separately so it never overlaps the heatmap)
# --------------------------------------------------------------------------- #


def _legend_figure(
    annotations: list[_Annotation],
    color_maps: dict[str, dict[str, str]],
    continuous_colormap: str,
) -> Figure:
    """A standalone legend: a swatch block per categorical + a colorbar per continuous.

    Each annotation gets its own row (a sub-axes). Categorical rows draw a titled swatch
    legend (value -> color); continuous rows draw a horizontal colorbar over the
    variable's range.
    """
    n = len(annotations)
    fig, raw_axes = plt.subplots(n, 1, figsize=(4.0, max(1.6, 1.3 * n)))
    axes_list = list(np.atleast_1d(raw_axes))

    for ax, annotation in zip(axes_list, annotations, strict=True):
        if annotation.continuous:
            values = annotation.per_sample.astype(float)
            vmin = float(values.min())
            vmax = float(values.max())
            norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)
            mappable = ScalarMappable(cmap=plt.get_cmap(continuous_colormap), norm=norm)
            mappable.set_array(np.asarray([], dtype=float))
            cbar = ax.figure.colorbar(mappable, cax=ax, orientation="horizontal")
            cbar.set_label(annotation.name, fontsize=12)
        else:
            ax.axis("off")
            color_map = color_maps[annotation.name]
            # Unique values in first-appearance order (matches the stripe / registry).
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
