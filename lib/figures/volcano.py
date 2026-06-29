"""Volcano figure for a differential-abundance contrast.

TEMPLATE (lib/) — a *seed* for a project's volcano module, not a finished script. Copy
it into the project's ``scripts/`` and adapt the call sites per study. Held to the
correctness charter (conventions/correctness.md): **assume nothing, verify everything,
fail loud.**

What it draws (one figure): the per-feature **effect size** (log2 fold change) on x vs
the **statistical significance** ``-log10(BH q)`` on y, with three-way significance
coloring — not significant, significant & up, significant & down — and the **hit counts
carried in the (separate) legend**. It is the headline read of a univariate test: how
big *and* how certain each feature's change is, at a glance.

It pairs with :mod:`analysis.differential_abundance`: the decoupled core
:func:`plot_volcano` takes plain ``effect`` / ``q`` arrays (so any source works and is
trivially testable), and :func:`volcano_from_result` is the family bridge that pulls the
chosen contrast term's columns straight off a
:class:`~analysis.differential_abundance.DifferentialAbundanceResult`.

Convention wiring (conventions/visualization.md): the up/down colors come from the
project color registry (:mod:`figures.colors`) — a two-value ``"Significance"`` category
(plus a gray *background* for NS, which consumes no palette slot), so "up" and "down"
keep one color across every volcano. The main figure carries **no** baked-in legend; the
legend (with the hit counts) is rendered as its **own** figure and saved beside the plot
as ``<base>.legend.{svg,png}`` via :func:`figures.figure_io.save_figure`.

Top-hit labels (``annotate_top``): the most-significant hits are labelled
collision-free by :mod:`textalloc` (a dependency this template adds) — each label is
repelled off the others and the data, leader-lined back to its point, and kept inside
the axes, so a dense cluster of co-significant features stays readable rather than
overprinting.

Significance: a feature is a hit when ``q <= fdr`` (and, when ``effect_threshold`` is
given, ``|effect| >= effect_threshold`` — the classic fold-change gate, drawn as
vertical guides). A ``q`` that underflowed to 0 (a t-statistic so extreme the survival
function hit 0) is floored to the smallest positive double so ``-log10`` stays finite
and the point plots at the top, never dropping out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import textalloc as ta
from analysis.differential_abundance import DifferentialAbundanceResult
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "volcano", "version": "0.2"},
    "kind": "module",
    "provides": [
        "VolcanoCounts",
        "VolcanoPlot",
        "plot_volcano",
        "volcano_from_result",
        "save_volcano",
    ],
    "uses": [
        "analysis.differential_abundance",
        "figures.colors",
        "figures.figure_io",
    ],
    "seeded_from": None,
    "description": (
        "Volcano figure for a differential-abundance contrast: effect (log2 fold "
        "change) on x vs -log10(BH q) on y, three-way significance coloring "
        "(NS / up / down) with hit counts in a separate legend. Decoupled array core "
        "plus a bridge that reads a DifferentialAbundanceResult contrast term. Up/down "
        "colors from the project registry (NS gray background, no slot); q underflow "
        "floored to finfo.tiny; optional fold-change gate. Optional annotate_top "
        "labels the most-significant hits collision-free via textalloc (labels "
        "repelled off each other and the point cloud, leader lines, clamped inside the "
        "axes). Dual-export plus a separate legend image. Study-agnostic; fail-loud."
    ),
}

# Color-registry namespace for the up/down significance colors — a methodological
# category (like the CV plot's NormalizationState), so "up"/"down" keep one color.
DEFAULT_SIGNIFICANCE_CATEGORY = "Significance"
_UP_KEY = "up"
_DOWN_KEY = "down"
_NS_KEY = "NS"

# textalloc tuning for annotate_top label placement. Settled against the dense-cluster
# / crowded / long-accession stress cases on real data: small font; gray leader lines;
# a candidate count high enough to resolve a near-coincident ortholog pair (e.g. the two
# APP labels at the q-ceiling) without crossing leaders. They are intentionally
# conservative — adapt them in the project copy if a study labels far more hits.
_LABEL_FONTSIZE = 8
_LEADER_COLOR = "0.45"
_LEADER_WIDTH = 0.6
_LABEL_CANDIDATES = 900
_LABEL_MARGIN = 0.012
_LABEL_MIN_DISTANCE = 0.018
_LABEL_MAX_DISTANCE = 0.28
# Nudge the top label bound down a hair so a box can't ride the top spine / suptitle.
_LABEL_TOP_PAD_FRAC = 0.02


@dataclass(frozen=True)
class VolcanoCounts:
    """The three significance bucket counts drawn (and shown in the legend)."""

    up: int
    down: int
    ns: int


@dataclass
class VolcanoPlot:
    """A rendered volcano figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (the volcano scatter, no baked legend).
    legend_figure:
        A standalone legend (up / down / NS swatches with hit counts), saved beside the
        main figure as ``<base>.legend.{svg,png}`` by :func:`save_volcano`.
    counts:
        The :class:`VolcanoCounts` actually drawn.
    color_map:
        ``{"up": hex, "down": hex, "NS": hex}`` used.
    """

    figure: Figure
    legend_figure: Figure
    counts: VolcanoCounts
    color_map: dict[str, str]


def _safe_neg_log10(qvalues: np.ndarray) -> np.ndarray:
    """``-log10(q)`` with zeros floored to the tiniest positive double; ``NaN`` kept."""
    q = np.asarray(qvalues, dtype=float)
    floor = float(np.finfo(float).tiny)
    safe = np.where(np.isnan(q), np.nan, np.maximum(q, floor))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(-np.log10(safe), dtype=float)


# --------------------------------------------------------------------------- #
# Core (decoupled — plain arrays)
# --------------------------------------------------------------------------- #
def plot_volcano(
    effect: np.ndarray,
    qvalues: np.ndarray,
    *,
    fdr: float = 0.05,
    effect_threshold: float | None = None,
    effect_label: str = "log2 fold change",
    direction_labels: tuple[str, str] = ("up", "down"),
    labels: np.ndarray | None = None,
    annotate_top: int = 0,
    category: str = DEFAULT_SIGNIFICANCE_CATEGORY,
    title: str | None = None,
    legend_title: str = "Significance",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> VolcanoPlot:
    """Render a volcano from per-feature effects and BH q-values.

    Parameters
    ----------
    effect:
        ``(n_features,)`` effect sizes (x-axis) — log2 fold change on log data. Sign
        convention is the caller's (for the family: non-reference vs reference).
    qvalues:
        ``(n_features,)`` BH-adjusted p-values (same length as ``effect``). ``NaN`` (an
        untestable feature) is dropped from the plot and the counts.
    fdr:
        q threshold for a hit and the horizontal guide line (default ``0.05``).
    effect_threshold:
        Optional ``|effect|`` gate: with it, a hit needs ``q <= fdr`` **and**
        ``|effect| >= effect_threshold`` (vertical guides drawn). ``None`` = q only.
    effect_label:
        x-axis label (e.g. ``result.effect_label``).
    direction_labels:
        ``(positive, negative)`` legend nouns for the sign of the effect (e.g.
        ``("higher in 5xFAD", "higher in non-AD")``). Default ``("up", "down")``.
    labels:
        Optional ``(n_features,)`` names, required only when ``annotate_top > 0``.
    annotate_top:
        Label this many top hits (smallest q) with their name (default ``0`` = none).
        Labels are placed collision-free by :mod:`textalloc` — repelled off one another
        and the point cloud, leader-lined back to their point, and clamped inside the
        axes — so even a tight cluster of co-significant hits stays legible.
    category:
        Color-registry namespace for the up/down colors. Default ``"Significance"``.
    title:
        Optional figure suptitle.
    legend_title:
        Title for the companion legend figure.
    registry_path, persist_colors:
        Color registry JSON and whether to write new color assignments back.

    Returns
    -------
    VolcanoPlot
    """
    eff = np.asarray(effect, dtype=float)
    q = np.asarray(qvalues, dtype=float)
    if eff.shape != q.shape:
        raise ValueError(
            f"effect and qvalues must have the same shape; got {eff.shape} and "
            f"{q.shape}."
        )
    if eff.ndim != 1:
        raise ValueError(f"effect/qvalues must be 1D; got {eff.ndim}D.")
    if not 0.0 < fdr < 1.0:
        raise ValueError(f"fdr must be in (0, 1); got {fdr}.")
    if effect_threshold is not None and effect_threshold < 0:
        raise ValueError(
            f"effect_threshold must be non-negative when given; got {effect_threshold}."
        )
    if annotate_top < 0:
        raise ValueError(f"annotate_top must be >= 0; got {annotate_top}.")
    if annotate_top > 0 and labels is None:
        raise ValueError("annotate_top > 0 requires labels (feature names).")
    name_arr = np.asarray(labels) if labels is not None else None
    if name_arr is not None and name_arr.shape != eff.shape:
        raise ValueError(
            f"labels must match effect length; got {name_arr.shape} vs {eff.shape}."
        )

    testable = np.isfinite(q) & np.isfinite(eff)
    gate = np.abs(eff) >= effect_threshold if effect_threshold is not None else True
    sig = testable & (q <= fdr) & gate
    up = sig & (eff > 0)
    down = sig & (eff < 0)
    ns = testable & ~sig
    counts = VolcanoCounts(up=int(up.sum()), down=int(down.sum()), ns=int(ns.sum()))

    neg_log_q = _safe_neg_log10(q)

    with publication_style():
        fig, ax = plt.subplots(figsize=(7, 6))
        try:
            color_map = assign_colors(
                category,
                [_UP_KEY, _DOWN_KEY],
                registry_path=registry_path,
                background_values=[_NS_KEY],
                persist=persist_colors,
            )
            _draw_volcano(
                ax,
                eff=eff,
                neg_log_q=neg_log_q,
                up=up,
                down=down,
                ns=ns,
                color_map=color_map,
                fdr=fdr,
                effect_threshold=effect_threshold,
                effect_label=effect_label,
            )
            if annotate_top > 0 and name_arr is not None:
                _annotate_top(ax, eff, q, neg_log_q, sig, name_arr, annotate_top)
            if title is not None:
                fig.suptitle(title, fontsize=14, weight="bold")
            legend_figure = _legend_figure(
                color_map, counts, direction_labels, legend_title
            )
        except BaseException:
            plt.close(fig)
            raise

    return VolcanoPlot(
        figure=fig, legend_figure=legend_figure, counts=counts, color_map=color_map
    )


# --------------------------------------------------------------------------- #
# Family bridge — read a DifferentialAbundanceResult
# --------------------------------------------------------------------------- #
def volcano_from_result(
    result: DifferentialAbundanceResult,
    *,
    term: str | None = None,
    fdr: float = 0.05,
    effect_threshold: float | None = None,
    annotate_top: int = 0,
    direction_labels: tuple[str, str] | None = None,
    title: str | None = None,
    legend_title: str = "Significance",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> VolcanoPlot:
    """Volcano for one contrast term of a :class:`DifferentialAbundanceResult`.

    ``term`` selects which contrast coefficient to plot (required when the contrast has
    more than one — a ``k>2`` factor); it defaults to the sole contrast term. Effect,
    q, and feature columns are read straight off the result table.
    """
    term = _resolve_term(result, term)
    rows = result.table[result.table["term"] == term]
    eff = rows["effect"].to_numpy(dtype=float)
    q = rows["q"].to_numpy(dtype=float)
    names = rows["feature"].to_numpy()
    labels_arg = names if annotate_top > 0 else None
    dlabels = direction_labels if direction_labels is not None else ("up", "down")
    return plot_volcano(
        eff,
        q,
        fdr=fdr,
        effect_threshold=effect_threshold,
        effect_label=result.effect_label,
        direction_labels=dlabels,
        labels=labels_arg,
        annotate_top=annotate_top,
        title=title if title is not None else term,
        legend_title=legend_title,
        registry_path=registry_path,
        persist_colors=persist_colors,
    )


def _resolve_term(result: DifferentialAbundanceResult, term: str | None) -> str:
    """Pick the contrast term to plot; fail loud on an ambiguous or unknown choice."""
    terms = result.contrast_terms
    if term is None:
        if len(terms) != 1:
            raise ValueError(
                f"This result has {len(terms)} contrast terms {list(terms)}; pass "
                f"term= to choose which to plot."
            )
        return terms[0]
    if term not in terms:
        raise ValueError(
            f"term {term!r} is not a contrast term; choose one of {list(terms)}."
        )
    return term


def save_volcano(
    effect: np.ndarray,
    qvalues: np.ndarray,
    output_dir: str | Path,
    base_name: str,
    *,
    fdr: float = 0.05,
    effect_threshold: float | None = None,
    effect_label: str = "log2 fold change",
    direction_labels: tuple[str, str] = ("up", "down"),
    labels: np.ndarray | None = None,
    annotate_top: int = 0,
    category: str = DEFAULT_SIGNIFICANCE_CATEGORY,
    title: str | None = None,
    legend_title: str = "Significance",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render a volcano (:func:`plot_volcano`) and save it dual-export + legend."""
    plot = plot_volcano(
        effect,
        qvalues,
        fdr=fdr,
        effect_threshold=effect_threshold,
        effect_label=effect_label,
        direction_labels=direction_labels,
        labels=labels,
        annotate_top=annotate_top,
        category=category,
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
def _draw_volcano(
    ax: Axes,
    *,
    eff: np.ndarray,
    neg_log_q: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    ns: np.ndarray,
    color_map: dict[str, str],
    fdr: float,
    effect_threshold: float | None,
    effect_label: str,
) -> None:
    """Scatter the three buckets and draw the threshold guides + axis labels."""
    ax.scatter(
        eff[ns],
        neg_log_q[ns],
        c=color_map[_NS_KEY],
        s=12,
        alpha=0.5,
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        eff[down],
        neg_log_q[down],
        c=color_map[_DOWN_KEY],
        s=22,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )
    ax.scatter(
        eff[up],
        neg_log_q[up],
        c=color_map[_UP_KEY],
        s=22,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1, zorder=0)
    ax.axhline(-np.log10(fdr), color="gray", linestyle="--", linewidth=1, zorder=0)
    if effect_threshold is not None and effect_threshold > 0:
        for x in (-effect_threshold, effect_threshold):
            ax.axvline(x, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel(effect_label)
    ax.set_ylabel(r"$-\log_{10}$(BH $q$)")
    ax.grid(True, alpha=0.25)


def _annotate_top(
    ax: Axes,
    eff: np.ndarray,
    q: np.ndarray,
    neg_log_q: np.ndarray,
    sig: np.ndarray,
    names: np.ndarray,
    annotate_top: int,
) -> None:
    """Label the ``annotate_top`` most-significant hits, collision-free.

    The hits with the smallest q (among the significant features) are labelled and laid
    out by :mod:`textalloc`, which repels each label off the others *and* off the full
    plotted point cloud, draws a thin leader line from every label back to its point,
    and clamps each label inside the axes. So a dense cluster of co-significant features
    — e.g. an ortholog pair sitting on top of each other at the q underflow ceiling —
    stays legible instead of overprinting into a mash, and the dot each label belongs to
    is unambiguous. Does nothing when no feature is significant.
    """
    hit_idx = np.flatnonzero(sig)
    if hit_idx.size == 0:
        return
    order = hit_idx[np.argsort(q[hit_idx], kind="stable")][:annotate_top]
    texts = [str(names[i]) for i in order]
    # The full plotted cloud (every testable feature) is the static set labels avoid;
    # neg_log_q is NaN exactly where q is NaN, which the scatter already skips.
    finite = np.isfinite(eff) & np.isfinite(neg_log_q)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ta.allocate(
        ax,
        eff[order],
        neg_log_q[order],
        texts,
        x_scatter=eff[finite],
        y_scatter=neg_log_q[finite],
        textsize=_LABEL_FONTSIZE,
        draw_lines=True,
        linecolor=_LEADER_COLOR,
        linewidth=_LEADER_WIDTH,
        textcolor="black",
        margin=_LABEL_MARGIN,
        min_distance=_LABEL_MIN_DISTANCE,
        max_distance=_LABEL_MAX_DISTANCE,
        nbr_candidates=_LABEL_CANDIDATES,
        avoid_label_lines_overlap=True,
        avoid_crossing_label_lines=True,
        xlims=(x0, x1),
        ylims=(y0, y1 - _LABEL_TOP_PAD_FRAC * (y1 - y0)),
    )


def _legend_figure(
    color_map: dict[str, str],
    counts: VolcanoCounts,
    direction_labels: tuple[str, str],
    legend_title: str,
) -> Figure:
    """Standalone legend: up / down / NS swatches with the hit counts."""
    up_label, down_label = direction_labels
    entries = [
        (_UP_KEY, f"{up_label} (n={counts.up})"),
        (_DOWN_KEY, f"{down_label} (n={counts.down})"),
        (_NS_KEY, f"NS (n={counts.ns})"),
    ]
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map[key],
            markeredgecolor="white",
            markersize=9,
        )
        for key, _ in entries
    ]
    ax.legend(
        handles,
        [text for _, text in entries],
        title=legend_title,
        loc="center",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    return fig
