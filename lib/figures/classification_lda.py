"""Result figures for a shrinkage-LDA classification.

TEMPLATE (lib/) — a *seed* for a project's classification-figures module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

**Three** figures read a :class:`~analysis.classification_lda.LDAClassificationResult`
— the LDA counterparts of the elastic-net / SVM classifier figures. The first template
with no hyperparameter figure: the shrinkage is analytic, so there is no tuning curve
or heatmap to draw, and three is the complete set (a reviewer must not ask for a
fourth). The shrinkage diagnostic lives in the ROC annotation instead.

  * :func:`plot_roc` — the mean ROC across outer CV folds with a ±1 SD band and a
    chance diagonal, drawn from the **log posterior-odds** (calibrated scores; AUC is
    rank-based); the legend's AUC is the result's CV AUC (the number in the finding).
    Balanced accuracy (log-odds thresholded at the equal-prior cut), average
    precision, per-class N, the per-repeat mean-of-fold and **pooled-OOF** AUCs, the
    all-data **per-class Ledoit-Wolf shrinkage** and the null verdict sit in a caption
    strip **below** the axes, so nothing opaque covers the plot region. The legend
    sits on-axes (lower-right, where a good classifier leaves space) — a documented
    exception to the separate-legend convention (conventions/visualization.md).
  * :func:`plot_null` — the label-shuffle null AUC histogram with the observed AUC
    marked and the empirical p. **Conditional:** only meaningful when the null was run
    (``run_null=True``); it raises otherwise.
  * :func:`plot_coefficients` — the top-N features by |all-data standardized weight|,
    each a diamond at its final weight over its resample IQR, colored by **top-k
    membership frequency** (viridis) — the dense-model stability read that replaces
    selection frequency (the SVM convention). A vertical line at 0 separates the
    classes. ``top_n`` defaults to the result's ``top_k`` so the rows shown are exactly
    the set the colour is defined on.

Colorbars sit beside the axes (they don't overlap the data), so these figures pass no
separate legend figure to :func:`figures.figure_io.save_figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis.classification_lda import LDAClassificationResult
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "classification-lda-figures", "version": "0.3"},
    "kind": "module",
    "provides": [
        "plot_roc",
        "plot_null",
        "plot_coefficients",
        "save_roc",
        "save_null",
        "save_coefficients",
    ],
    "uses": ["analysis.classification_lda", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Three result figures for an LDAClassificationResult (no tuning figure — the "
        "shrinkage is analytic): ROC +-SD across outer folds from the calibrated "
        "log posterior-odds (legend on-axes carrying the result's CV AUC; the "
        "summary — per-repeat mean-of-fold + pooled-OOF AUCs, the per-class "
        "Ledoit-Wolf shrinkage, the null verdict — in a caption strip below the "
        "axes, never over the plot), the label-shuffle null AUC histogram "
        "(conditional on the null being run; shrinkage re-estimated per draw), and "
        "the top-N signed-weight plot (final weight + resample IQR, colored by top-k "
        "membership frequency on viridis — the dense-model stability read; top_n "
        "defaults to the result's top_k). Dual-export via figure-io; colorbars beside "
        "the axes (no separate legend figure). Study-agnostic; fail-loud."
    ),
}

# Fixed Okabe-Ito palette entries for the (non-categorical) ROC styling. These encode no
# metadata category, so they are house-style constants rather than color-registry slots.
_ROC_COLOR = "#0072B2"  # Okabe-Ito blue — the mean ROC + band
_OBSERVED_COLOR = "#D55E00"  # Okabe-Ito vermillion — observed-AUC marker
_CHANCE_COLOR = "#999999"
_NULL_FILL = "#999999"

_ROC_GRID = np.linspace(0.0, 1.0, 100)


# --------------------------------------------------------------------------- #
# ROC
# --------------------------------------------------------------------------- #
def plot_roc(result: LDAClassificationResult, *, title: str | None = None) -> Figure:
    """Mean ROC (± 1 SD) across outer CV folds, with a chance diagonal.

    Each outer fold contributes one ROC curve from its held-out **log-odds**; the
    curves are interpolated onto a common FPR grid (each fold's vertical rises
    preserved — see :func:`_interp_tpr`) and averaged. The legend (chance / mean ROC /
    ±1 SD) is on-axes and its AUC is the result's ``cv_auc`` ± ``cv_auc_sd``; balanced
    accuracy, average precision, per-class N, the per-repeat mean-of-fold and
    pooled-OOF AUCs, the all-data shrinkage intensities and the null verdict form a
    caption strip below the axes.
    """
    if not result.fold_predictions:
        raise ValueError("result has no fold predictions to draw a ROC from.")
    tprs = [_interp_tpr(fold.y_true, fold.y_score) for fold in result.fold_predictions]
    tpr_stack = np.asarray(tprs, dtype=float)
    mean_tpr = tpr_stack.mean(axis=0)
    sd_tpr = tpr_stack.std(axis=0)
    summary = _roc_summary(result)

    with publication_style():
        fig, ax, caption_y = _roc_figure(summary)
        try:
            ax.plot([0, 1], [0, 1], "--", color=_CHANCE_COLOR, label="chance", zorder=1)
            ax.plot(
                _ROC_GRID,
                mean_tpr,
                color=_ROC_COLOR,
                lw=2,
                # The number on the figure is the number in the finding: the result's
                # nested-CV AUC (mean ± SD of the per-fold AUCs), not an AUC re-derived
                # from the drawn mean curve, which is a different quantity.
                label=(
                    f"mean ROC (AUC = {result.cv_auc:.3f} ± {result.cv_auc_sd:.3f}, "
                    "mean of per-fold AUCs)"
                ),
                zorder=3,
            )
            ax.fill_between(
                _ROC_GRID,
                np.clip(mean_tpr - sd_tpr, 0.0, 1.0),
                np.clip(mean_tpr + sd_tpr, 0.0, 1.0),
                color=_ROC_COLOR,
                alpha=0.2,
                label="± 1 SD",
                zorder=2,
            )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right", fontsize=9)
            fig.text(
                0.5,
                caption_y,
                summary,
                ha="center",
                va="top",
                multialignment="left",
                fontsize=_ROC_CAPTION_FONTSIZE,
                linespacing=1.3,
            )
            _apply_title(fig, title, _roc_default_title(result), result)
        except BaseException:
            plt.close(fig)
            raise
    return fig


def _roc_summary(result: LDAClassificationResult) -> str:
    """The ROC's caption-strip text: the numbers a reader needs beside the curve.

    The per-repeat lines list each repeat's mean-of-fold AUC and the range of the
    **pooled out-of-fold** AUC per repeat (one ROC over a repeat's concatenated
    held-out log-odds — the user's own protocol). A between-repeat SD is deliberately
    not shown: it is partition noise, far tighter than the fold SD, and reads as the
    estimate's uncertainty when it is not.
    """
    text = (
        f"balanced accuracy = {result.cv_balanced_accuracy:.3f}  (equal-prior cut)   "
        f"|   average precision = {result.cv_average_precision:.3f}\n"
        f"{result.positive_label}: N={result.n_positive}  |  "
        f"{result.negative_label}: N={result.n_negative}"
    )
    text += _repeat_lines(result.repeat_aucs, result.repeat_pooled_aucs)
    text += "\nscores = LDA log-odds (calibrated posterior)"
    text += (
        f"\nLedoit-Wolf shrinkage λ = {result.shrinkage_negative:.2f} / "
        f"{result.shrinkage_positive:.2f}  ({result.negative_label} / "
        f"{result.positive_label})"
    )
    if result.null_p is None:
        text += "\nnull not run — exploratory"
    else:
        obs = result.observed_auc
        obs_txt = f" (null-CV observed AUC {obs:.3f})" if obs is not None else ""
        text += f"\nvs shuffle null: p = {result.null_p:.4f}{obs_txt}"
    return text


def _repeat_lines(
    repeat_aucs: tuple[float, ...], repeat_pooled_aucs: tuple[float, ...]
) -> str:
    """Two short caption lines: mean-of-fold AUC per repeat, pooled-OOF AUC range."""
    text = ""
    if len(repeat_aucs) > 1:
        text += "\nmean-of-fold AUC by repeat: " + ", ".join(
            f"{a:.3f}" for a in repeat_aucs
        )
    if repeat_pooled_aucs:
        lo, hi = min(repeat_pooled_aucs), max(repeat_pooled_aucs)
        span = f"{lo:.3f} to {hi:.3f}" if len(repeat_pooled_aucs) > 1 else f"{lo:.3f}"
        text += f"\npooled-OOF AUC by repeat: {span}"
    return text


def _interp_tpr(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """One fold's TPR on the common FPR grid, keeping every vertical rise.

    ``roc_curve`` repeats an FPR value wherever the curve rises vertically (several
    thresholds at one false-positive count — including the rise out of the origin at
    FPR = 0). ``np.interp`` at a repeated x returns one of the tied y values
    arbitrarily, so each FPR is first collapsed to its **maximum** TPR: the
    interpolated curve then passes through the top of every vertical segment, and a
    fold that separates perfectly is drawn as perfect (TPR = 1 at FPR = 0) instead of
    being notched back to the origin. Nothing is forced — the curve starts and ends
    where the data put it.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    uniq, inverse = np.unique(fpr, return_inverse=True)
    top = np.full(uniq.shape, -np.inf)
    np.maximum.at(top, inverse, tpr)
    return np.asarray(np.interp(_ROC_GRID, uniq, top), dtype=float)


_ROC_CAPTION_FONTSIZE = 8
_ROC_CAPTION_LINE_IN = 0.16  # vertical room per caption line (inches)
_ROC_TOP_IN = 0.75  # room above the axes for the (possibly two-line) suptitle
_ROC_PLOT_IN = 4.8  # the plot region — the same square as the pre-caption figure
_ROC_XLABEL_IN = 0.7  # clearance under the axes for the tick labels + x label


def _roc_figure(summary: str) -> tuple[Figure, Axes, float]:
    """A square ROC axes over a caption strip sized to hold ``summary``.

    The summary is a figure-level caption **below** the axes — never an opaque box
    over the plot region, where it would hide exactly what a reader most needs to see
    for a near-chance classifier: the mean curve, its ±SD band and the chance
    diagonal. Returns the figure, the axes and the caption's top y (figure fraction).
    """
    n_lines = summary.count("\n") + 1
    strip = _ROC_XLABEL_IN + _ROC_CAPTION_LINE_IN * n_lines + 0.2
    height = _ROC_TOP_IN + _ROC_PLOT_IN + strip
    fig, ax = plt.subplots(figsize=(6.2, height))
    fig.subplots_adjust(bottom=strip / height, top=1.0 - _ROC_TOP_IN / height)
    return fig, ax, (strip - _ROC_XLABEL_IN) / height


def _roc_default_title(result: LDAClassificationResult) -> str:
    return (
        f"{result.positive_label} vs {result.negative_label} — shrinkage LDA "
        f"(repeated CV)"
    )


# --------------------------------------------------------------------------- #
# Null histogram
# --------------------------------------------------------------------------- #


def _null_scheme_note(scheme: str | None) -> str:
    """Title clause naming the null permutation scheme (empty for a plain shuffle)."""
    if scheme == "within_units":
        return ", within-unit permutation"
    if scheme == "units":
        return ", unit-level permutation"
    return ""  # "samples" (a row shuffle needs no qualifier) or None


def plot_null(result: LDAClassificationResult, *, title: str | None = None) -> Figure:
    """Label-shuffle null AUC distribution with the observed AUC and empirical p.

    Raises when the null was not run (``run_null=False``): there is nothing to draw.
    """
    if result.null_aucs is None or result.observed_auc is None or result.null_p is None:
        raise ValueError(
            "null was not run for this result; call classify_lda(..., run_null=True) "
            "to produce a null distribution before plotting it."
        )
    nulls = np.asarray(result.null_aucs, dtype=float)
    with publication_style():
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        try:
            ax.hist(nulls, bins=25, color=_NULL_FILL, edgecolor="white")
            ax.axvline(
                result.observed_auc,
                color=_OBSERVED_COLOR,
                lw=2,
                label=(
                    f"observed AUC = {result.observed_auc:.3f}\n"
                    f"empirical p = {result.null_p:.4f}"
                ),
            )
            ax.set_xlabel("ROC AUC under permuted labels")
            ax.set_ylabel("count")
            ax.legend(fontsize=9)
            n_perm = int(nulls.size)
            _apply_title(
                fig,
                title,
                f"Label-shuffle null ({n_perm} permutations, shrinkage re-estimated "
                f"per draw{_null_scheme_note(result.null_permutation)})",
                result,
            )
        except BaseException:
            plt.close(fig)
            raise
    return fig


# --------------------------------------------------------------------------- #
# Weight (coefficient) plot
# --------------------------------------------------------------------------- #
def plot_coefficients(
    result: LDAClassificationResult,
    *,
    top_n: int | None = None,
    title: str | None = None,
) -> Figure:
    """Top-N features by |weight|, colored by top-k membership frequency.

    Each feature is a diamond at its **all-data standardized weight** over a bar
    spanning its resample IQR (``coef_q25``..``coef_q75``), colored by **top-k
    membership frequency** (viridis). A vertical line at 0 separates the classes (left =
    negative, right = positive). The model is dense, so this is the head of a ranking
    over *every* feature, not a selected subset. ``top_n`` defaults to the result's
    ``top_k`` so the rows shown are the set the colour is defined on.
    """
    if top_n is None:
        top_n = result.top_k
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}.")
    table = result.coefficients
    if len(table) == 0:
        raise ValueError("the coefficient table is empty; nothing to plot.")
    # Largest |coef| at the top: take the head, then reverse so y increases upward.
    sub = table.head(top_n).iloc[::-1].reset_index(drop=True)
    n = len(sub)
    coef = sub["coef"].to_numpy(dtype=float)
    q25 = sub["coef_q25"].to_numpy(dtype=float)
    q75 = sub["coef_q75"].to_numpy(dtype=float)
    freq = sub["top_k_frequency"].to_numpy(dtype=float)
    names = sub["feature"].astype(str).to_numpy()
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")

    with publication_style():
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.32 * n + 1.5)))
        try:
            for i in range(n):
                color = cmap(norm(freq[i]))
                ax.plot([q25[i], q75[i]], [i, i], color=color, lw=2.0, alpha=0.7)
                ax.scatter(
                    [coef[i]],
                    [i],
                    marker="D",
                    s=48,
                    color=color,
                    edgecolor="black",
                    linewidths=0.5,
                    zorder=5,
                )
            ax.axvline(0.0, color="black", lw=0.8, zorder=1)
            ax.set_yticks(range(n))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_xlabel(
                f"standardized LDA weight  "
                f"(- {result.negative_label}   |   {result.positive_label} +)"
            )
            shown = min(top_n, len(table))
            _apply_title(
                fig,
                title,
                f"Top {shown} of {len(table)} features by |weight| "
                f"(diamond = final weight, bar = resample IQR)",
                result,
            )
            mappable = ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(
                mappable,
                ax=ax,
                label=f"top-{result.top_k} membership frequency",
                fraction=0.046,
                pad=0.04,
            )
        except BaseException:
            plt.close(fig)
            raise
    return fig


# --------------------------------------------------------------------------- #
# Save wrappers
# --------------------------------------------------------------------------- #
def save_roc(
    result: LDAClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_roc` and dual-export it."""
    return save_figure(plot_roc(result, title=title), output_dir, base_name, dpi=dpi)


def save_null(
    result: LDAClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_null` and dual-export it (raises if the null was not run)."""
    return save_figure(plot_null(result, title=title), output_dir, base_name, dpi=dpi)


def save_coefficients(
    result: LDAClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    top_n: int | None = None,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_coefficients` and dual-export it."""
    fig = plot_coefficients(result, top_n=top_n, title=title)
    return save_figure(fig, output_dir, base_name, dpi=dpi)


def _feature_list_suffix(result: LDAClassificationResult) -> str:
    """A mandatory caveat line when the model was built on a prior feature list.

    Empty for a whole-proteome run. When a ``feature_list`` was supplied, states the
    panel size (and, on a partial match, how many of the requested ids were found), so
    a restricted-panel figure is never mistaken for a whole-proteome result.
    """
    requested = result.n_features_requested
    matched = result.n_features_matched
    if requested is None or matched is None:
        return ""
    if matched < requested:
        return f"\nprior feature list · {matched} of {requested} matched"
    return f"\nprior feature list · {matched} features"


def _apply_title(
    fig: Figure, title: str | None, default: str, result: LDAClassificationResult
) -> None:
    base = title if title is not None else default
    fig.suptitle(base + _feature_list_suffix(result), fontsize=13, weight="bold")
