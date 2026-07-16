"""Publication figures for a GO/KEGG over-representation result.

TEMPLATE (lib/) — a *seed* for a project's enrichment-figure module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

Three complementary views of an :class:`~analysis.enrichment.EnrichmentResult`, on the
shared figure foundation (:mod:`figures.figure_io` dual-export + a separate legend;
source colors from the project registry, :mod:`figures.colors`, a ``"EnrichmentSource"``
category well within the eight-color budget):

  * :func:`enrichment_dotplot` — one panel per source; y = term, x = gene ratio, dot
    **size** = intersection size (# query genes in the term), **color** = ``-log10`` of
    the corrected p (viridis). The richest single view. The source label sits *inside*
    each panel (never a title in the inter-panel gap that could collide with the tick
    labels of the panel above), and the whole key (color scale + size legend) is the
    separate legend image — so nothing overlaps a data point.
  * :func:`enrichment_barplot` — the top terms across sources as horizontal bars, x =
    ``-log10`` corrected p, bar color = source, gene counts inline, a dashed line at the
    significance threshold. Cleanest for a slide or paper.
  * :func:`enrichment_manhattan` — the g:Profiler Manhattan: every tested term laid
    out along x grouped by source, y = ``-log10`` corrected p, significance line, and
    the most-significant terms labelled collision-free (:mod:`textalloc`). Needs full
    tested-term set, so it **raises** unless the result was produced with
    ``enrich(all_results=True)`` (the default).

Each function returns an :class:`EnrichmentFigure` (the main figure + its companion
legend figure); the ``save_*`` wrappers dual-export both via
:func:`figures.figure_io.save_figure`. Figures never leak on an error path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textalloc as ta
from analysis.enrichment import EnrichmentResult
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "enrichment-figures", "version": "0.1"},
    "kind": "module",
    "provides": [
        "EnrichmentFigure",
        "enrichment_dotplot",
        "enrichment_barplot",
        "enrichment_manhattan",
        "save_enrichment_dotplot",
        "save_enrichment_barplot",
        "save_enrichment_manhattan",
    ],
    "uses": [
        "analysis.enrichment",
        "figures.colors",
        "figures.figure_io",
    ],
    "seeded_from": None,
    "description": (
        "Three publication figures for an EnrichmentResult on the figure foundation: a "
        "faceted dot plot (per-source panels; x = gene ratio, dot size = intersection, "
        "color = -log10 corrected p, in-panel source labels, key in a separate legend "
        "image), a grouped bar chart (top terms, x = -log10 corrected p, bar "
        "color = source, gene counts inline, threshold line), and a g:Profiler-style "
        "Manhattan (all terms grouped by source, -log10 p on y, top terms labelled "
        "collision-free via textalloc; requires all_results). Source colors from the "
        "project registry; dual-export plus a separate legend image. Study-agnostic; "
        "fail-loud."
    ),
}

# Registry namespace for ontology-source colors (GO:BP / GO:MF / GO:CC / KEGG / ...),
# a methodological category well within the eight-color budget.
DEFAULT_SOURCE_CATEGORY = "EnrichmentSource"
_CMAP_NAME = "viridis"


@dataclass
class EnrichmentFigure:
    """A rendered enrichment figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (no baked-in key).
    legend_figure:
        The standalone legend (color scale / size key / source swatches), saved beside
        main figure as ``<base>.legend.{svg,png}`` by the ``save_*`` wrappers.
    """

    figure: Figure
    legend_figure: Figure


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _neg_log10p(pvalues: np.ndarray) -> np.ndarray:
    """``-log10(p)`` with zeros floored to the tiniest positive double; ``NaN`` kept."""
    p = np.asarray(pvalues, dtype=float)
    floor = float(np.finfo(float).tiny)
    safe = np.where(np.isnan(p), np.nan, np.maximum(p, floor))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(-np.log10(safe), dtype=float)


def _dot_size(counts: np.ndarray) -> np.ndarray:
    """Marker area for an intersection size (shared by the plot and its legend key)."""
    return 30.0 + 22.0 * np.asarray(counts, dtype=float) ** 0.6


def _present_sources(table: pd.DataFrame, order: tuple[str, ...]) -> list[str]:
    """Sources that appear in ``table``, in the requested order (extras last)."""
    seen = set(np.asarray(table["source"]).tolist())
    present = [s for s in order if s in seen]
    extra = sorted(s for s in seen if s not in set(order))
    return present + extra


def _source_colors(
    sources: list[str],
    source_category: str,
    registry_path: str | Path,
    persist: bool,
) -> dict[str, str]:
    return assign_colors(
        source_category, sources, registry_path=registry_path, persist=persist
    )


def _top_rows(table: pd.DataFrame, source: str, top_n: int) -> pd.DataFrame:
    """The ``top_n`` top rows of one source, most-significant last (for y)."""
    rows = table[table["source"] == source].sort_values("p_value", kind="stable")
    return rows.head(top_n).iloc[::-1]


def _empty_message_figure(message: str) -> tuple[Figure, Figure]:
    """A placeholder main + legend figure when there is nothing to plot."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, color="gray")
    legend, lax = plt.subplots(figsize=(2, 1))
    lax.axis("off")
    return fig, legend


def _source_legend_figure(color_map: dict[str, str], sources: list[str]) -> Figure:
    """Standalone legend: one swatch per ontology source, in the given order."""
    fig, ax = plt.subplots(figsize=(2.6, 0.4 * len(sources) + 0.8))
    ax.axis("off")
    handles = [
        Patch(facecolor=color_map[s], edgecolor="black", linewidth=0.4) for s in sources
    ]
    ax.legend(
        handles,
        sources,
        title="source",
        loc="center",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    return fig


# --------------------------------------------------------------------------- #
# Dot plot
# --------------------------------------------------------------------------- #
def enrichment_dotplot(
    result: EnrichmentResult,
    *,
    top_n: int = 10,
    source_category: str = DEFAULT_SOURCE_CATEGORY,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> EnrichmentFigure:
    """Faceted dot plot of the significant terms (top ``top_n`` per source).

    x = gene ratio (intersection / query size), dot size = intersection, dot color =
    ``-log10`` corrected p (viridis). Source labels are in-panel; the color scale + size
    key are the separate legend image.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}.")
    sig = result.significant_table
    present = _present_sources(sig, result.sources)
    with publication_style():
        if not present:
            fig, legend = _empty_message_figure("No significant terms to plot")
            return EnrichmentFigure(figure=fig, legend_figure=legend)
        nlp_all = _neg_log10p(np.asarray(sig["p_value"]))
        norm = Normalize(vmin=float(np.nanmin(nlp_all)), vmax=float(np.nanmax(nlp_all)))
        cmap = plt.get_cmap(_CMAP_NAME)
        panels = {s: _top_rows(sig, s, top_n) for s in present}
        heights = [max(len(panels[s]), 1) for s in present]
        fig, axes = plt.subplots(
            len(present),
            1,
            figsize=(8.6, 1.0 + 0.46 * sum(heights)),
            gridspec_kw={"height_ratios": heights},
            squeeze=False,
        )
        try:
            color_map = _source_colors(
                present, source_category, registry_path, persist_colors
            )
            fig.subplots_adjust(
                hspace=0.42, top=0.9, left=0.42, right=0.975, bottom=0.09
            )
            for ax, source in zip(axes[:, 0], present, strict=True):
                _draw_dotplot_panel(ax, panels[source], source, color_map, norm, cmap)
            axes[-1, 0].set_xlabel(
                "gene ratio  (query hits in term / query size)", fontsize=10
            )
            if title is not None:
                fig.suptitle(title, fontsize=13)
            legend = _dotplot_legend_figure(norm, cmap)
        except BaseException:
            plt.close(fig)
            raise
    return EnrichmentFigure(figure=fig, legend_figure=legend)


def _draw_dotplot_panel(
    ax: Axes,
    rows: pd.DataFrame,
    source: str,
    color_map: dict[str, str],
    norm: Normalize,
    cmap: Colormap,
) -> None:
    """One source's panel: stems + dots + in-panel source label."""
    ys = np.arange(len(rows))
    gene_ratio = np.asarray(rows["gene_ratio"], dtype=float)
    sizes = np.asarray(rows["intersection_size"], dtype=float)
    nlp = _neg_log10p(np.asarray(rows["p_value"]))
    for y, x in zip(ys, gene_ratio, strict=True):
        ax.hlines(y, 0, x, color="#d9d9d9", lw=1.0, zorder=1)
    ax.scatter(
        gene_ratio,
        ys,
        s=_dot_size(sizes),
        c=[cmap(norm(v)) for v in nlp],
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([str(name)[:54] for name in rows["term_name"]], fontsize=8.5)
    ax.margins(y=0.30)
    ax.grid(axis="x", ls=":", alpha=0.35)
    ax.set_xlim(left=0)
    ax.annotate(
        source,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(3, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=10.5,
        color=color_map.get(source, "black"),
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
    )


def _dotplot_legend_figure(norm: Normalize, cmap: Colormap) -> Figure:
    """Separate legend image: the ``-log10 p`` color scale and the dot-size key."""
    leg = plt.figure(figsize=(2.7, 4.0))
    ax_cbar = leg.add_axes((0.30, 0.58, 0.12, 0.34))
    cbar = leg.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar)
    cbar.set_label(r"$-\log_{10}$(corrected $p$)", fontsize=9)
    ax_size = leg.add_axes((0.0, 0.02, 1.0, 0.44))
    ax_size.set_axis_off()
    ax_size.text(
        0.5,
        0.95,
        "intersection\n(genes in term)",
        fontsize=9,
        ha="center",
        va="top",
        transform=ax_size.transAxes,
    )
    for i, n in enumerate((10, 40, 80)):
        y = 0.58 - i * 0.24
        ax_size.scatter(
            [0.34],
            [y],
            s=_dot_size(np.array([n]))[0],
            c="lightgray",
            edgecolors="black",
            linewidths=0.5,
            transform=ax_size.transAxes,
        )
        ax_size.text(
            0.56,
            y,
            str(n),
            va="center",
            ha="left",
            fontsize=9,
            transform=ax_size.transAxes,
        )
    return leg


def save_enrichment_dotplot(
    result: EnrichmentResult,
    output_dir: str | Path,
    base_name: str,
    *,
    top_n: int = 10,
    source_category: str = DEFAULT_SOURCE_CATEGORY,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render (:func:`enrichment_dotplot`) and save dual-export + separate legend."""
    figure = enrichment_dotplot(
        result,
        top_n=top_n,
        source_category=source_category,
        title=title,
        registry_path=registry_path,
        persist_colors=persist_colors,
    )
    return save_figure(
        figure.figure,
        output_dir,
        base_name,
        legend_fig=figure.legend_figure,
        dpi=dpi,
    )


# --------------------------------------------------------------------------- #
# Bar chart
# --------------------------------------------------------------------------- #
def enrichment_barplot(
    result: EnrichmentResult,
    *,
    top_n: int = 20,
    source_category: str = DEFAULT_SOURCE_CATEGORY,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> EnrichmentFigure:
    """Top ``top_n`` significant terms across sources as horizontal bars.

    x = ``-log10`` corrected p, bar color = source, gene counts inline, dashed guide at
    the significance threshold. The source swatches are the separate legend image.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}.")
    sig = result.significant_table
    present = _present_sources(sig, result.sources)
    with publication_style():
        if not present:
            fig, legend = _empty_message_figure("No significant terms to plot")
            return EnrichmentFigure(figure=fig, legend_figure=legend)
        rows = sig.sort_values("p_value", kind="stable").head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(rows) + 1.4))
        try:
            color_map = _source_colors(
                present, source_category, registry_path, persist_colors
            )
            _draw_barplot(ax, rows, color_map, result.user_threshold)
            if title is not None:
                fig.suptitle(title, fontsize=13)
            legend = _source_legend_figure(color_map, present)
        except BaseException:
            plt.close(fig)
            raise
    return EnrichmentFigure(figure=fig, legend_figure=legend)


def _draw_barplot(
    ax: Axes, rows: pd.DataFrame, color_map: dict[str, str], threshold: float
) -> None:
    """Horizontal bars (x = -log10 p) colored by source, term + count inline."""
    ys = np.arange(len(rows))
    nlp = _neg_log10p(np.asarray(rows["p_value"]))
    colors = [color_map.get(str(s), "gray") for s in rows["source"]]
    ax.barh(ys, nlp, color=colors, edgecolor="black", linewidth=0.4, zorder=2)
    for y, name, count in zip(
        ys, rows["term_name"], rows["intersection_size"], strict=True
    ):
        ax.text(
            0.1,
            float(y),
            f"  {str(name)[:50]}  (n={int(count)})",
            va="center",
            ha="left",
            fontsize=8,
            color="black",
            zorder=3,
        )
    ax.axvline(
        -np.log10(threshold), color="gray", linestyle="--", linewidth=1, zorder=1
    )
    ax.set_yticks([])
    ax.set_xlabel(r"$-\log_{10}$(corrected $p$)", fontsize=10)


def save_enrichment_barplot(
    result: EnrichmentResult,
    output_dir: str | Path,
    base_name: str,
    *,
    top_n: int = 20,
    source_category: str = DEFAULT_SOURCE_CATEGORY,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render (:func:`enrichment_barplot`) and save dual-export + separate legend."""
    figure = enrichment_barplot(
        result,
        top_n=top_n,
        source_category=source_category,
        title=title,
        registry_path=registry_path,
        persist_colors=persist_colors,
    )
    return save_figure(
        figure.figure,
        output_dir,
        base_name,
        legend_fig=figure.legend_figure,
        dpi=dpi,
    )


# --------------------------------------------------------------------------- #
# Manhattan
# --------------------------------------------------------------------------- #
def enrichment_manhattan(
    result: EnrichmentResult,
    *,
    label_top: int = 12,
    source_gap: int = 40,
    source_category: str = DEFAULT_SOURCE_CATEGORY,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> EnrichmentFigure:
    """g:Profiler-style Manhattan of **all** tested terms, grouped by source.

    y = ``-log10`` corrected p; the significance threshold is a dashed line; the
    ``label_top`` most-significant terms are labelled collision-free (:mod:`textalloc`).
    Requires the full tested-term set (``enrich(all_results=True)``) — else there is
    no point cloud below the threshold and it **raises**.
    """
    if label_top < 0:
        raise ValueError(f"label_top must be >= 0; got {label_top}.")
    if not result.all_results_included:
        raise ValueError(
            "enrichment_manhattan needs the full tested-term set; produce it with "
            "enrich(all_results=True) (the default)."
        )
    table = result.table
    present = _present_sources(table, result.sources)
    with publication_style():
        if not present:
            fig, legend = _empty_message_figure("No tested terms to plot")
            return EnrichmentFigure(figure=fig, legend_figure=legend)
        fig, ax = plt.subplots(figsize=(11, 5))
        try:
            color_map = _source_colors(
                present, source_category, registry_path, persist_colors
            )
            _draw_manhattan(
                ax,
                table,
                present,
                color_map,
                result.user_threshold,
                label_top,
                source_gap,
            )
            if title is not None:
                fig.suptitle(title, fontsize=13)
            legend = _source_legend_figure(color_map, present)
        except BaseException:
            plt.close(fig)
            raise
    return EnrichmentFigure(figure=fig, legend_figure=legend)


def _draw_manhattan(
    ax: Axes,
    table: pd.DataFrame,
    present: list[str],
    color_map: dict[str, str],
    threshold: float,
    label_top: int,
    source_gap: int,
) -> None:
    """Scatter every term grouped by source; label most-significant collision-free."""
    x_cursor = 0
    ticks: list[float] = []
    label_x: list[float] = []
    label_y: list[float] = []
    label_text: list[str] = []
    thr = -np.log10(threshold)
    for source in present:
        rows = table[table["source"] == source].sort_values("term_id", kind="stable")
        xs = np.arange(len(rows)) + x_cursor
        nlp = _neg_log10p(np.asarray(rows["p_value"]))
        ax.scatter(
            xs,
            nlp,
            s=14,
            c=color_map.get(source, "gray"),
            alpha=0.75,
            edgecolors="none",
        )
        names = list(rows["term_name"])
        for xi, yi, name in zip(xs, nlp, names, strict=True):
            if yi > thr:
                label_x.append(float(xi))
                label_y.append(float(yi))
                label_text.append(str(name))
        ticks.append(x_cursor + len(rows) / 2.0)
        x_cursor += len(rows) + source_gap
    ax.axhline(thr, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(present, fontweight="bold")
    for tick_label, source in zip(ax.get_xticklabels(), present, strict=True):
        tick_label.set_color(color_map.get(source, "black"))
    ax.set_ylabel(r"$-\log_{10}$(corrected $p$)")
    ax.set_xlabel("functional term  (grouped by source)")
    ax.margins(x=0.01)
    _label_manhattan(ax, label_x, label_y, label_text, label_top)


def _label_manhattan(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    texts: list[str],
    label_top: int,
) -> None:
    """Label the ``label_top`` most-significant terms collision-free via textalloc."""
    if label_top == 0 or not xs:
        return
    order = np.argsort(ys)[::-1][:label_top]
    sel_x = [xs[i] for i in order]
    sel_y = [ys[i] for i in order]
    sel_t = [texts[i][:34] for i in order]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ta.allocate(
        ax,
        sel_x,
        sel_y,
        sel_t,
        x_scatter=xs,
        y_scatter=ys,
        textsize=7,
        draw_lines=True,
        linecolor="0.5",
        linewidth=0.5,
        textcolor="black",
        nbr_candidates=400,
        avoid_label_lines_overlap=True,
        xlims=(x0, x1),
        ylims=(y0, y1),
    )


def save_enrichment_manhattan(
    result: EnrichmentResult,
    output_dir: str | Path,
    base_name: str,
    *,
    label_top: int = 12,
    source_category: str = DEFAULT_SOURCE_CATEGORY,
    title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render (:func:`enrichment_manhattan`) and save dual-export + separate legend."""
    figure = enrichment_manhattan(
        result,
        label_top=label_top,
        source_category=source_category,
        title=title,
        registry_path=registry_path,
        persist_colors=persist_colors,
    )
    return save_figure(
        figure.figure,
        output_dir,
        base_name,
        legend_fig=figure.legend_figure,
        dpi=dpi,
    )
