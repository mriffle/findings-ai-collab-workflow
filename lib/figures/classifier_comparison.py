"""Result figures for a paired per-fold classifier comparison.

TEMPLATE (lib/) — a *seed* for a project's comparison-figures module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study.
Held to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

Two figures read the results of ``analysis.classifier_comparison``:

  * :func:`plot_paired_folds` — for one :class:`ClassifierComparisonResult`: a
    **slope chart** (left; each outer fold's reference AUC joined to its alternative
    AUC, the segment colored win / loss / tie from the alternative's point of view) and
    the **per-fold difference dot plot** (right; ``alternative - reference`` per fold
    over a zero line, with the mean difference and its corrected-t interval
    ``mean ± t_{df, 0.975} · sqrt(var · (1/J + rho))`` — Nadeau & Bengio 2003). A
    **summary strip below the axes** carries the numbers the reader needs while
    looking (W/L/T, mean Δ, the decisive corrected-t p, the *indicative* Wilcoxon p or
    "n/a", the pre-stated reading, the seed) — text under the plot, never an opaque
    box over the data. The win/loss/tie key is the **separate legend figure**.
  * :func:`plot_paired_folds_multi_seed` — for a :class:`MultiSeedComparisonResult`:
    one difference-dot panel per seed on a shared y axis, each with its mean marked,
    the suptitle stating the seed-rule verdict (a verdict only when the sign of the
    mean difference agrees across every seed).

Both are dual-exported through :func:`figures.figure_io.save_figure` with the legend
as ``<base>.legend.{svg,png}``; no figure leaks on an error path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis.classifier_comparison import (
    ClassifierComparisonResult,
    MultiSeedComparisonResult,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "classifier-comparison-figures", "version": "0.1"},
    "kind": "figure",
    "provides": [
        "plot_paired_folds",
        "plot_paired_folds_multi_seed",
        "save_paired_folds",
        "save_paired_folds_multi_seed",
    ],
    "uses": ["analysis.classifier_comparison", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Two figures for a paired classifier comparison: the single-seed "
        "slope chart (reference -> alternative AUC per outer fold, segments colored "
        "win / loss / tie) beside the per-fold difference dot plot (zero line, mean "
        "difference with its corrected-t interval) with a summary strip below the "
        "axes (W/L/T, mean delta, corrected-t p, indicative Wilcoxon p or n/a, "
        "reading, seed); and the multi-seed figure (one difference panel per seed, "
        "shared y, per-seed mean, the seed-rule verdict in the suptitle). The "
        "win/loss/tie key is a separate legend image. Dual-export via figure-io; "
        "no figure leaks on an error path. Study-agnostic; fail-loud."
    ),
}

# Fixed Okabe-Ito entries for the three fold outcomes. They encode no metadata
# category, so they are house-style constants rather than color-registry slots.
_WIN_COLOR = "#D55E00"  # vermillion — the alternative ranked this fold better
_LOSS_COLOR = "#0072B2"  # blue — the reference ranked it better
_TIE_COLOR = "#999999"  # gray — identical AUC (within the tie tolerance)
_MEAN_COLOR = "black"
_OUTCOME_COLORS: dict[str, str] = {
    "win": _WIN_COLOR,
    "loss": _LOSS_COLOR,
    "tie": _TIE_COLOR,
}


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #
def _outcome_colors(result: ClassifierComparisonResult) -> list[str]:
    outcomes = result.per_fold["outcome"].astype(str).to_numpy()
    unknown = sorted(set(outcomes) - set(_OUTCOME_COLORS))
    if unknown:
        raise ValueError(f"per_fold.outcome holds unknown values {unknown}.")
    return [_OUTCOME_COLORS[o] for o in outcomes]


def _corrected_half_width(result: ClassifierComparisonResult) -> float:
    """Half-width of the corrected-t interval on the mean difference.

    ``t_{df, 0.975} · sqrt(var · (1/J + rho))`` — the same variance the decisive
    statistic uses, so the interval and the p agree. Zero when the differences have
    no variance.
    """
    j = result.n_folds
    var = result.sd_diff**2
    if var == 0.0:
        return 0.0
    q = float(stats.t.ppf(0.975, result.corrected_t_df))
    return q * float(np.sqrt(var * (1.0 / j + result.rho)))


def _format_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def _draw_slope(ax: Axes, result: ClassifierComparisonResult) -> None:
    """Reference AUC -> alternative AUC, one segment per outer fold."""
    frame = result.per_fold
    ref = frame["auc_reference"].to_numpy(dtype=float)
    alt = frame["auc_alternative"].to_numpy(dtype=float)
    colors = _outcome_colors(result)
    for r, a, c in zip(ref, alt, colors, strict=True):
        ax.plot([0, 1], [r, a], color=c, lw=1.2, alpha=0.75, zorder=2)
    ax.scatter(np.zeros_like(ref), ref, color=colors, s=22, zorder=3)
    ax.scatter(np.ones_like(alt), alt, color=colors, s=22, zorder=3)
    ax.plot(
        [0, 1],
        [result.reference_auc, result.alternative_auc],
        color=_MEAN_COLOR,
        lw=2.4,
        marker="D",
        markersize=6,
        zorder=4,
    )
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"{result.reference_label}\nmean {result.reference_auc:.3f}",
            f"{result.alternative_label}\nmean {result.alternative_auc:.3f}",
        ]
    )
    ax.set_ylabel("held-out ROC AUC (per outer fold)")
    ax.set_title("paired per-fold AUC", fontsize=11)


def _draw_differences(
    ax: Axes,
    result: ClassifierComparisonResult,
    *,
    show_interval: bool = True,
    xlabel: str = "outer fold (repeats separated by dotted lines)",
) -> None:
    """Per-fold ``alternative - reference`` dots, zero line, mean (+ interval)."""
    frame = result.per_fold
    diff = frame["diff"].to_numpy(dtype=float)
    repeats = frame["repeat"].to_numpy(dtype=int)
    x = np.arange(1, len(diff) + 1)
    colors = _outcome_colors(result)
    ax.axhline(0.0, color="black", lw=0.8, zorder=1)
    # A dotted separator wherever the repeat index changes, so the fold-dependence
    # structure (folds within a repeat share training data) is visible.
    for i in range(1, len(repeats)):
        if repeats[i] != repeats[i - 1]:
            ax.axvline(i + 0.5, color="lightgray", ls=":", lw=0.8, zorder=1)
    if show_interval:
        half = _corrected_half_width(result)
        ax.axhspan(
            result.mean_diff - half,
            result.mean_diff + half,
            color=_MEAN_COLOR,
            alpha=0.10,
            lw=0,
            zorder=1,
        )
    ax.axhline(result.mean_diff, color=_MEAN_COLOR, ls="--", lw=1.4, zorder=2)
    ax.scatter(x, diff, color=colors, s=30, edgecolor="black", linewidths=0.4, zorder=3)
    ax.set_xlim(0.4, len(diff) + 0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Δ AUC  ({result.alternative_label} - {result.reference_label})")


def _summary_lines(result: ClassifierComparisonResult) -> list[str]:
    wilcoxon = (
        f"Wilcoxon p (indicative) {_format_p(result.wilcoxon_p)}"
        if result.wilcoxon_p is not None
        else f"Wilcoxon p (indicative) n/a (<5 non-zero; {result.n_nonzero_diffs})"
    )
    return [
        (
            f"W/L/T ({result.alternative_label} vs {result.reference_label}) = "
            f"{result.n_wins}/{result.n_losses}/{result.n_ties} of {result.n_folds} "
            f"folds   ·   mean Δ AUC = {result.mean_diff:+.3f}   ·   "
            f"corrected-t p {_format_p(result.corrected_t_p)} "
            f"(t = {result.corrected_t:.2f}, df = {result.corrected_t_df})"
        ),
        (
            f"{wilcoxon}   ·   reading (pre-stated): {result.reading}   ·   "
            f"seed {result.random_state}   ·   n = {result.n_samples} samples, "
            f"unseen {result.generalization_target}"
        ),
    ]


def _summary_strip(fig: Figure, lines: list[str], y: float = 0.03) -> None:
    """The numbers the reader needs, as text **below** the axes (never over data)."""
    fig.text(0.5, y, "\n".join(lines), ha="center", va="bottom", fontsize=8.5)


def _legend_figure() -> Figure:
    """Standalone swatch key for the three fold outcomes."""
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    try:
        ax.axis("off")
        handles = [
            Line2D(
                [0],
                [0],
                color=_WIN_COLOR,
                marker="o",
                lw=1.5,
                label="win — alternative AUC higher",
            ),
            Line2D(
                [0],
                [0],
                color=_LOSS_COLOR,
                marker="o",
                lw=1.5,
                label="loss — reference AUC higher",
            ),
            Line2D(
                [0],
                [0],
                color=_TIE_COLOR,
                marker="o",
                lw=1.5,
                label="tie — equal within tolerance",
            ),
            Line2D(
                [0],
                [0],
                color=_MEAN_COLOR,
                ls="--",
                lw=1.4,
                label="mean Δ (band: corrected-t 95% interval)",
            ),
        ]
        ax.legend(
            handles=handles,
            title="per-fold outcome",
            loc="center",
            frameon=True,
            fontsize=10,
            title_fontsize=11,
        )
    except BaseException:
        plt.close(fig)
        raise
    return fig


# --------------------------------------------------------------------------- #
# Single-seed figure
# --------------------------------------------------------------------------- #
def plot_paired_folds(
    result: ClassifierComparisonResult, *, title: str | None = None
) -> tuple[Figure, Figure]:
    """Slope chart + per-fold difference dot plot, with a summary strip below.

    Returns ``(figure, legend_figure)``; the legend is the win/loss/tie key rendered
    as its own image. A custom ``title`` replaces the suptitle only — the summary
    strip is always drawn.
    """
    if len(result.per_fold) == 0:
        raise ValueError("the comparison has no folds to draw.")
    with publication_style():
        fig, (ax_slope, ax_diff) = plt.subplots(
            1, 2, figsize=(11.0, 5.4), gridspec_kw={"width_ratios": [1.0, 1.5]}
        )
        try:
            _draw_slope(ax_slope, result)
            _draw_differences(ax_diff, result)
            ax_diff.set_title(
                "Δ AUC per fold (dashed: mean; band: corrected-t interval)",
                fontsize=11,
            )
            default = (
                f"{result.alternative_label} vs {result.reference_label} — "
                f"{result.positive_label} vs {result.negative_label}, paired on "
                f"{result.n_folds} shared outer folds"
            )
            fig.suptitle(title if title is not None else default, fontsize=13)
            fig.subplots_adjust(
                left=0.08, right=0.98, top=0.86, bottom=0.26, wspace=0.3
            )
            _summary_strip(fig, _summary_lines(result), y=0.04)
        except BaseException:
            plt.close(fig)
            raise
        try:
            legend = _legend_figure()
        except BaseException:
            plt.close(fig)
            raise
    return fig, legend


# --------------------------------------------------------------------------- #
# Multi-seed figure
# --------------------------------------------------------------------------- #
def plot_paired_folds_multi_seed(
    result: MultiSeedComparisonResult, *, title: str | None = None
) -> tuple[Figure, Figure]:
    """One per-fold difference panel per seed (shared y), verdict in the suptitle.

    Each panel marks its seed's mean difference (dashed line); the suptitle states
    the seed-rule verdict — a side only when the sign is consistent across every
    seed, otherwise a tie. Returns ``(figure, legend_figure)``.
    """
    n = len(result.comparisons)
    if n == 0:
        raise ValueError("the multi-seed comparison holds no comparisons to draw.")
    with publication_style():
        fig, axes = plt.subplots(
            1,
            n,
            sharey=True,
            figsize=(max(3.4 * n, 6.5) + 0.8, 5.0),
            squeeze=False,
        )
        try:
            for ax, comp, seed, sign in zip(
                axes[0],
                result.comparisons,
                result.seeds,
                result.per_seed_sign,
                strict=True,
            ):
                _draw_differences(ax, comp, show_interval=False, xlabel="outer fold")
                ax.set_title(
                    f"seed {seed}: mean Δ = {comp.mean_diff:+.3f}  "
                    f"(W/L/T {comp.n_wins}/{comp.n_losses}/{comp.n_ties})",
                    fontsize=10,
                )
                # Mark the sign the seed rule reads, right of the folds.
                ax.scatter(
                    [len(comp.per_fold) + 0.4],
                    [comp.mean_diff],
                    marker="D",
                    s=48,
                    color=_MEAN_COLOR,
                    zorder=4,
                    clip_on=False,
                )
                if sign == 0:
                    ax.text(
                        0.98,
                        0.97,
                        "sign: 0 (tie)",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                    )
            for ax in axes[0][1:]:
                ax.set_ylabel("")
            if result.sign_consistent:
                verdict = (
                    f"verdict: {result.verdict} — sign consistent across {n} seeds"
                )
            else:
                verdict = f"verdict: tie — the sign is not consistent across {n} seeds"
            default = (
                f"{result.alternative_label} vs {result.reference_label} across "
                f"seeds {list(result.seeds)}\n{verdict}"
            )
            fig.suptitle(title if title is not None else default, fontsize=12)
            wilc = [
                f"seed {c.random_state}: corrected-t p {_format_p(c.corrected_t_p)}, "
                f"Wilcoxon p (indicative) "
                + (_format_p(c.wilcoxon_p) if c.wilcoxon_p is not None else "n/a")
                for c in result.comparisons
            ]
            # Two seeds per line so the strip never outgrows the figure width.
            lines = ["   ·   ".join(wilc[i : i + 2]) for i in range(0, len(wilc), 2)]
            lines.append(
                f"reading (pre-stated): {result.reading}   ·   the seed rule: a "
                f"verdict only when every seed's mean Δ has the same non-zero sign"
            )
            fig.subplots_adjust(
                left=0.08,
                right=0.97,
                top=0.80,
                bottom=0.14 + 0.05 * len(lines),
                wspace=0.12,
            )
            _summary_strip(fig, lines, y=0.03)
        except BaseException:
            plt.close(fig)
            raise
        try:
            legend = _legend_figure()
        except BaseException:
            plt.close(fig)
            raise
    return fig, legend


# --------------------------------------------------------------------------- #
# Save wrappers
# --------------------------------------------------------------------------- #
def save_paired_folds(
    result: ClassifierComparisonResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_paired_folds` and dual-export it with its legend image."""
    fig, legend = plot_paired_folds(result, title=title)
    return save_figure(fig, output_dir, base_name, legend_fig=legend, dpi=dpi)


def save_paired_folds_multi_seed(
    result: MultiSeedComparisonResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_paired_folds_multi_seed` and dual-export it + legend."""
    fig, legend = plot_paired_folds_multi_seed(result, title=title)
    return save_figure(fig, output_dir, base_name, legend_fig=legend, dpi=dpi)
