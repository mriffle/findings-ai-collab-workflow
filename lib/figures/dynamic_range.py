"""Dynamic-range / rank-abundance QC figures for a :class:`Dataset`.

TEMPLATE (lib/) — a *seed* for a project's dynamic-range module, not a finished script.
Copy it into the project's ``scripts/`` and adapt the call sites per study. Held to the
correctness charter (conventions/correctness.md): **assume nothing, verify everything,
fail loud.**

What it draws (one figure): features ranked by abundance (most → least) on the x-axis vs
their log2 abundance on the y-axis — the classic rank-abundance curve. It shows the
**dynamic range** quantified (how many orders of magnitude) and whether a handful of
hyper-abundant features dominate (e.g. albumin / other contaminants). Two modes:

* **Whole-cohort** (default): a single curve of the per-feature **median** of detected
  values, with a shaded **IQR band** (per-sample 25-75 pct) behind it. The QC sanity
  check. Optional ``highlight_features`` mark named proteins of interest at their
  (rank, abundance) with leader-line labels — empty at QC, populated downstream.
* **Per-class** (``class_by`` set): one **independently-ranked** median curve per sample
  class (experimental vs pooled-QC / reference), registry-colored — the "do all classes
  cover the same dynamic range?" comparison. Highlights are not available in this mode
  (each class has its own ranking, so a feature has no single rank).

The annotation lifecycle (conventions/visualization.md): ``highlight_features`` is the
one mechanism for marking proteins of interest, and *when* you populate it is the
workflow's choice. At QC it is typically empty (or marks **contaminants**, known a
priori). Downstream, once proteins of interest are ascertainable, the same plot is
re-rendered with **domain / hypothesis targets** and/or the **differential-abundance
top hits** — turning the QC sanity check into a results / communication figure. The
highlight set may be split into registry-colored **groups** (``highlight_groups``, e.g.
"contaminant" vs "of interest"), each shown in the separate legend.

Scale (a HARD refuse, like the CV / id-depth / missingness templates): the dynamic range
is read on the **raw linear matrix** — "detected" (``> min_intensity``) is a linear
concept and the y-axis is the log2 of the linear abundance. So
:func:`compute_dynamic_range` **raises** :class:`DynamicRangeScaleError` on a non-linear
``Dataset.scale``. Run it on the unnormalized data (a first-look QC); a never-detected
feature has no abundance and is left off the curve.

Convention wiring: per-class / per-group colors come from the project color registry
(:mod:`figures.colors`, capped at eight); the legend (class swatches, or highlight-group
swatches) is rendered as a **separate image** (``<base>.legend.{svg,png}``) via
:func:`figures.figure_io.save_figure`. In ``class_by`` mode **controls are shown with
the experimental samples** (overlaid, class-colored curves) — the labeled exception
shared with the sample-correlation / id-depth / missingness templates: the cross-class
comparison is the deliverable. This template makes NO study decisions: the detection
threshold, which column (if any) splits the curves, and which features are highlighted
are the caller's. Sample exclusions and relabelings live in the project copy.
"""

from __future__ import annotations

import warnings
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
    "template": {"name": "dynamic-range", "version": "0.1"},
    "kind": "module",
    "provides": [
        "DynamicRangeScaleError",
        "DynamicRangeResult",
        "DynamicRangePlot",
        "compute_dynamic_range",
        "plot_dynamic_range",
        "save_dynamic_range",
    ],
    "uses": ["common.data_loading", "figures.colors", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Dynamic-range / rank-abundance QC figures from a Dataset: features ranked by "
        "abundance vs log2 abundance, showing the quantified dynamic range and any "
        "hyper-abundant dominance. Whole-cohort median + IQR band (default) with "
        "optional leader-labelled highlight_features (proteins of interest / "
        "contaminants, optionally registry-colored groups), or a per-class "
        "independently-ranked overlay (class_by). Hard-refuses a non-linear scale "
        "(a raw-linear property); log2 y. "
        "Dual-export plus a separate legend image. Uses common.data_loading, "
        "figures.colors, figures.figure_io. Study-agnostic; fail-loud."
    ),
}

# Default color for highlighted features when no ``highlight_groups`` are given.
DEFAULT_HIGHLIGHT_COLOR = "#0072B2"
_BAND_COLOR = "#bdbdbd"
_CURVE_COLOR = "#333333"


class DynamicRangeScaleError(ValueError):
    """Raised when the dynamic range is requested on a non-linear abundance scale.

    The rank-abundance view is read on the **raw linear matrix**: "detected"
    (``> min_intensity``) is a linear concept and the y-axis is the log2 of the linear
    abundance. On a log or centered scale (``log2``/``glog2``/``zscore``, and the
    ``"mad"``/``"vsn"`` normalizer output) those no longer hold. Run it on the
    unnormalized data, before any transform (it is a first-look QC).
    """


@dataclass(frozen=True)
class DynamicRangeResult:
    """The ranked per-feature abundance summary underlying a dynamic-range figure.

    Only features detected in at least one sample appear (a never-detected feature has
    no abundance to rank); arrays are ordered most → least abundant.

    Attributes
    ----------
    feature_names_ranked:
        ``(k,)`` feature ids, most → least abundant.
    log2_median:
        ``(k,)`` log2 of the per-feature median of detected values (the curve).
    log2_q25, log2_q75:
        ``(k,)`` log2 of the per-feature 25th / 75th percentile of detected values (the
        IQR band).
    dynamic_range_orders:
        ``log10(max median / min median)`` over the ranked features — the orders of
        magnitude spanned.
    n_features_total:
        Features in the input Dataset.
    n_features_detected:
        Features detected in >= 1 sample (the length ``k`` of the ranked arrays).
    """

    feature_names_ranked: np.ndarray
    log2_median: np.ndarray
    log2_q25: np.ndarray
    log2_q75: np.ndarray
    dynamic_range_orders: float
    n_features_total: int
    n_features_detected: int

    @property
    def ranks(self) -> np.ndarray:
        """``(k,)`` ranks ``1..k`` aligned to the ranked arrays."""
        return np.arange(1, self.n_features_detected + 1)


@dataclass
class DynamicRangePlot:
    """A rendered dynamic-range figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (rank-abundance curve, no baked legend).
    legend_figure:
        A standalone swatch legend when one is needed (the ``class_by`` classes, or the
        ``highlight_groups``), saved beside the main figure as
        ``<base>.legend.{svg,png}`` by :func:`save_dynamic_range`; ``None`` for a plain
        or single-color-highlight plot.
    result:
        The underlying whole-cohort :class:`DynamicRangeResult`.
    color_map:
        ``{class-or-group: hex}`` drawn from the registry (empty when none was used).
    """

    figure: Figure
    legend_figure: Figure | None
    result: DynamicRangeResult
    color_map: dict[str, str]


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #


def compute_dynamic_range(
    dataset: Dataset, *, min_intensity: float = 0.0
) -> DynamicRangeResult:
    """Rank features by median detected abundance; return the ranked log2 summary.

    Detection (``> min_intensity``) and the dynamic range are raw-linear properties, so
    this raises :class:`DynamicRangeScaleError` unless ``dataset.scale == "linear"``.
    Features detected in no sample are excluded (no abundance to rank).

    Raises :class:`DynamicRangeScaleError` on a non-linear scale, or ``ValueError`` on a
    non-2D / empty Dataset or a feature-name mismatch.
    """
    if dataset.scale != "linear":
        raise DynamicRangeScaleError(
            f"Dynamic range requires linear-scale abundances but the Dataset is on "
            f"scale {dataset.scale!r}. 'detected' (> {min_intensity}) and the log2 "
            f"y-axis are raw-linear properties. Run it on the unnormalized data."
        )
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(
            f"abundances must be 2D (n_samples, n_features); got {abundances.shape}."
        )
    names = np.asarray(dataset.feature_names)
    n_total = abundances.shape[1]
    if names.shape[0] != n_total:
        raise ValueError(
            f"feature_names has {names.shape[0]} entries but abundances has {n_total} "
            f"features; they are not aligned."
        )
    if abundances.shape[0] < 1 or n_total < 1:
        raise ValueError(
            f"Dataset is empty ({abundances.shape[0]} samples x {n_total} features)."
        )

    median, q25, q75, detected_count = _detected_quartiles(abundances, min_intensity)
    keep = detected_count >= 1
    if not bool(keep.any()):
        raise ValueError(
            "no feature is detected in any sample (all <= min_intensity); none to rank."
        )

    med_keep = median[keep]
    order = np.argsort(med_keep)[::-1]
    med_ranked = med_keep[order]
    orders = (
        float(np.log10(med_ranked[0] / med_ranked[-1]))
        if med_ranked[-1] > 0
        else float("nan")
    )

    return DynamicRangeResult(
        feature_names_ranked=names[keep][order],
        log2_median=np.log2(med_ranked),
        log2_q25=np.log2(q25[keep][order]),
        log2_q75=np.log2(q75[keep][order]),
        dynamic_range_orders=orders,
        n_features_total=int(n_total),
        n_features_detected=int(keep.sum()),
    )


def _detected_quartiles(
    abundances: np.ndarray, min_intensity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature median / q25 / q75 of detected (> min_intensity) values + count."""
    mask = np.isfinite(abundances) & (abundances > min_intensity)
    positive = np.where(mask, abundances, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # all-NaN columns -> NaN, dropped
        median = np.nanmedian(positive, axis=0)
        q25 = np.nanpercentile(positive, 25, axis=0)
        q75 = np.nanpercentile(positive, 75, axis=0)
    return median, q25, q75, mask.sum(axis=0)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def plot_dynamic_range(
    dataset: Dataset,
    *,
    class_by: str | None = None,
    highlight_features: Mapping[str, str] | None = None,
    highlight_groups: Mapping[str, str] | None = None,
    highlight_color: str = DEFAULT_HIGHLIGHT_COLOR,
    min_intensity: float = 0.0,
    log_rank: bool = False,
    title: str | None = None,
    legend_title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> DynamicRangePlot:
    """Draw the rank-abundance curve for one Dataset (whole-cohort or per-class).

    Parameters
    ----------
    dataset:
        A single linear-scale :class:`Dataset` (raised otherwise).
    class_by:
        Metadata column to split the curve by (one independently-ranked median curve per
        class, registry-colored). ``None`` (default) draws the whole-cohort median + IQR
        band. **Mutually exclusive with highlights** (each class has its own ranking).
    highlight_features:
        ``{feature_id: display label}`` to mark at their (rank, log2 median). Only valid
        in the whole-cohort mode. Unknown or never-detected ids are refused.
    highlight_groups:
        Optional ``{feature_id: group}`` coloring the highlights by group through the
        registry (the >8 guard applies) with a group legend; when omitted all highlights
        use ``highlight_color``. Every highlighted id must have a group if given.
    highlight_color:
        Color for highlights when ``highlight_groups`` is not given.
    min_intensity:
        Detection threshold (default ``0.0``).
    log_rank:
        Use a log-scaled rank x-axis (spreads the high-abundance head). Default linear.
    title:
        Optional figure suptitle.
    legend_title:
        Title for the companion legend figure; defaults to ``class_by`` / "highlights".
    registry_path:
        Color registry JSON. Default ``state/color_registry.json``.
    persist_colors:
        Write newly assigned colors back to the registry (default ``True``).

    Returns
    -------
    DynamicRangePlot
        The figure, the companion legend figure (or ``None``), the whole-cohort
        :class:`DynamicRangeResult`, and the ``{class-or-group: hex}`` color map.

    Raises
    ------
    DynamicRangeScaleError
        If the Dataset is not on the ``"linear"`` scale.
    ValueError
        On an empty Dataset, ``class_by`` and highlights both given, a missing
        ``class_by`` column, a highlighted id that is unknown / never-detected, or an
        incomplete ``highlight_groups``.
    CategoricalPaletteExceededError
        If ``class_by`` / ``highlight_groups`` exceed the palette capacity (>8 guard).
    """
    if class_by is not None and highlight_features:
        raise ValueError(
            "class_by and highlight_features are mutually exclusive: per-class curves "
            "are ranked independently, so a highlight has no single rank. Pass one."
        )
    result = compute_dynamic_range(dataset, min_intensity=min_intensity)

    color_map: dict[str, str] = {}
    legend_figure: Figure | None = None
    with publication_style():
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        try:
            if class_by is not None:
                color_map = _draw_per_class(
                    ax,
                    dataset,
                    class_by,
                    min_intensity=min_intensity,
                    registry_path=registry_path,
                    persist_colors=persist_colors,
                )
                legend_figure = _legend_figure(
                    color_map, legend_title if legend_title is not None else class_by
                )
            else:
                _draw_curve(ax, result)
                if highlight_features:
                    color_map = _draw_highlights(
                        ax,
                        dataset,
                        result,
                        highlight_features,
                        highlight_groups,
                        highlight_color,
                        registry_path=registry_path,
                        persist_colors=persist_colors,
                    )
                    if highlight_groups:
                        legend_figure = _legend_figure(
                            color_map,
                            legend_title if legend_title is not None else "highlights",
                        )

            if log_rank:
                ax.set_xscale("log")
            ax.set_xlabel("abundance rank (1 = most abundant)")
            ax.set_ylabel("log2 abundance (median of detected)")
            ax.grid(True, linestyle=":", alpha=0.4)
            if title is not None:
                fig.suptitle(title, fontsize=15, weight="bold")
        except BaseException:
            plt.close(fig)
            raise

    return DynamicRangePlot(
        figure=fig, legend_figure=legend_figure, result=result, color_map=color_map
    )


def save_dynamic_range(
    dataset: Dataset,
    output_dir: str | Path,
    base_name: str,
    *,
    class_by: str | None = None,
    highlight_features: Mapping[str, str] | None = None,
    highlight_groups: Mapping[str, str] | None = None,
    highlight_color: str = DEFAULT_HIGHLIGHT_COLOR,
    min_intensity: float = 0.0,
    log_rank: bool = False,
    title: str | None = None,
    legend_title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render a dynamic-range figure (:func:`plot_dynamic_range`) and save it.

    Writes ``<base>.{svg,png}`` and, when a legend is built, ``<base>.legend.{svg,png}``
    via :func:`figures.figure_io.save_figure`; both figures are closed even if saving
    fails.
    """
    plot = plot_dynamic_range(
        dataset,
        class_by=class_by,
        highlight_features=highlight_features,
        highlight_groups=highlight_groups,
        highlight_color=highlight_color,
        min_intensity=min_intensity,
        log_rank=log_rank,
        title=title,
        legend_title=legend_title,
        registry_path=registry_path,
        persist_colors=persist_colors,
    )
    return save_figure(
        plot.figure, output_dir, base_name, legend_fig=plot.legend_figure, dpi=dpi
    )


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def _draw_curve(ax: Axes, result: DynamicRangeResult) -> None:
    """The whole-cohort median curve + IQR band."""
    ranks = result.ranks
    ax.fill_between(
        ranks,
        result.log2_q25,
        result.log2_q75,
        color=_BAND_COLOR,
        alpha=0.55,
        label="per-sample IQR",
    )
    ax.plot(
        ranks, result.log2_median, color=_CURVE_COLOR, linewidth=1.7, label="median"
    )
    ax.set_title(
        f"Dynamic range  ({result.dynamic_range_orders:.1f} orders, "
        f"n={result.n_features_detected})",
        loc="left",
        fontsize=13,
        weight="bold",
    )
    ax.legend(loc="upper right", frameon=True, fontsize=10)


def _draw_per_class(
    ax: Axes,
    dataset: Dataset,
    class_by: str,
    *,
    min_intensity: float,
    registry_path: str | Path,
    persist_colors: bool,
) -> dict[str, str]:
    """One independently-ranked median curve per class; returns the color map."""
    if class_by not in dataset.metadata.columns:
        raise ValueError(
            f"class_by {class_by!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )
    series = dataset.metadata[class_by]
    if bool(series.isna().any()):
        raise ValueError(
            f"class_by {class_by!r} has {int(series.isna().sum())} missing value(s); "
            f"resolve or relabel them first."
        )
    classes = series.to_numpy().astype(str)
    abundances = np.asarray(dataset.abundances, dtype=float)
    ordered = _ordered_unique(classes)
    color_map = assign_colors(
        class_by, ordered, registry_path=registry_path, persist=persist_colors
    )
    for cls in ordered:
        members = classes == cls
        median, _, _, count = _detected_quartiles(abundances[members], min_intensity)
        med = median[count >= 1]
        med.sort()
        ax.plot(
            np.arange(1, med.size + 1),
            np.log2(med[::-1]),
            color=color_map[cls],
            linewidth=2.0,
            label=f"{cls} (n={int(members.sum())})",
        )
    ax.set_title(
        "Dynamic range by sample class", loc="left", fontsize=13, weight="bold"
    )
    return color_map


@dataclass(frozen=True)
class _Highlight:
    """One resolved highlight: its rank position, log2 abundance, label, and color."""

    x: int
    y: float
    text: str
    color: str


def _draw_highlights(
    ax: Axes,
    dataset: Dataset,
    result: DynamicRangeResult,
    highlight_features: Mapping[str, str],
    highlight_groups: Mapping[str, str] | None,
    highlight_color: str,
    *,
    registry_path: str | Path,
    persist_colors: bool,
) -> dict[str, str]:
    """Mark + label the highlighted features; return the group color map if grouped."""
    rank_of = {name: i + 1 for i, name in enumerate(result.feature_names_ranked)}
    known = set(np.asarray(dataset.feature_names).astype(str))
    color_map: dict[str, str] = {}
    if highlight_groups is not None:
        missing = [f for f in highlight_features if f not in highlight_groups]
        if missing:
            raise ValueError(
                f"highlight_groups is missing group(s) for {missing[:5]}; every "
                f"highlighted feature needs a group when highlight_groups is given."
            )
        color_map = assign_colors(
            "highlight_group",
            _ordered_unique(np.asarray(list(highlight_groups.values()))),
            registry_path=registry_path,
            persist=persist_colors,
        )

    items: list[_Highlight] = []
    for fid, label in highlight_features.items():
        if fid not in known:
            raise ValueError(f"highlighted feature {fid!r} is not in the Dataset.")
        if fid not in rank_of:
            raise ValueError(
                f"highlighted feature {fid!r} is never detected, so it has no "
                f"abundance to mark; drop it from highlight_features."
            )
        idx = rank_of[fid] - 1
        color = (
            color_map[highlight_groups[fid]] if highlight_groups else highlight_color
        )
        ax.scatter(
            [rank_of[fid]],
            [result.log2_median[idx]],
            s=55,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        items.append(
            _Highlight(
                x=rank_of[fid],
                y=float(result.log2_median[idx]),
                text=label,
                color=color,
            )
        )

    finite = result.log2_median[np.isfinite(result.log2_median)]
    _place_labels(
        ax, items, result.n_features_detected, float(finite.max()), float(finite.min())
    )
    return color_map


def _place_labels(
    ax: Axes, items: list[_Highlight], n: int, y_top: float, y_bot: float
) -> None:
    """Leader-line labels, de-collided and clamped inside the axes.

    Head points (left half) are labelled in a column to the right with one leader per
    point (unambiguous ownership); tail points (right half) are labelled below their
    points in open space. Both stacks are de-collided and the y-range is padded so no
    label leaves the axes (no title collision).
    """
    ax.set_ylim(y_bot - 1.5, y_top + 1.0)
    gap = (y_top - y_bot) * 0.052

    head = sorted((h for h in items if h.x < n * 0.5), key=lambda h: h.y, reverse=True)
    col_x = n * 0.14
    cur = y_top
    for h in head:
        ly = min(h.y, cur)
        cur = ly - gap
        _annotate(ax, h, col_x, ly, "left")

    tail = sorted((h for h in items if h.x >= n * 0.5), key=lambda h: h.y, reverse=True)
    cur = min((h.y for h in tail), default=y_bot) - gap * 1.6
    for h in tail:
        ly = max(cur, y_bot + gap)
        cur = ly - gap
        _annotate(ax, h, float(h.x - n * 0.01), ly, "right")


def _annotate(
    ax: Axes, item: _Highlight, label_x: float, label_y: float, ha: str
) -> None:
    """One leader-line annotation from its label position back to the point."""
    ax.annotate(
        item.text,
        xy=(item.x, item.y),
        xytext=(label_x, label_y),
        fontsize=9,
        color=item.color,
        va="center",
        ha=ha,
        arrowprops={"arrowstyle": "-", "color": item.color, "lw": 0.7},
    )


def _ordered_unique(values: np.ndarray) -> list[str]:
    """Distinct values in first-seen order (stable, study-agnostic)."""
    seen: list[str] = []
    for value in values.astype(str):
        if value not in seen:
            seen.append(value)
    return seen


# --------------------------------------------------------------------------- #
# Legend figure (rendered separately so it never overlaps the plot)
# --------------------------------------------------------------------------- #


def _legend_figure(color_map: dict[str, str], legend_title: str) -> Figure:
    """Standalone swatch legend: one line per class / highlight group."""
    labels = list(color_map.keys())
    height = max(1.4, 0.35 * len(labels) + 0.8)
    fig, ax = plt.subplots(figsize=(3.4, height))
    ax.axis("off")
    handles = [Line2D([0], [0], color=color_map[k], linewidth=2.5) for k in labels]
    ax.legend(
        handles,
        labels,
        title=legend_title or None,
        loc="center",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    return fig
