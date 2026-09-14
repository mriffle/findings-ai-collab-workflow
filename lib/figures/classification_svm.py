"""Result figures for a linear-SVM classification.

TEMPLATE (lib/) — a *seed* for a project's classification-figures module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

Four figures read a :class:`~analysis.classification_svm.SVMClassificationResult` —
the SVM counterparts of the elastic-net classifier's figures
(``figures.classification``); the first two are the same, the other two follow from
the estimator (dense signed weights; a 1-D ``C`` grid):

  * :func:`plot_roc` — the mean ROC across outer nested-CV folds with a ±1 SD band and a
    chance diagonal, drawn from the **decision-function margins** (uncalibrated scores;
    AUC is rank-based, so no probability is needed). Balanced accuracy (margin
    thresholded at 0), average precision, per-class N, and the per-repeat AUC are
    annotated. The legend sits on-axes (lower-right, where a good classifier leaves
    space) — a documented exception to the separate-legend convention
    (conventions/visualization.md).
  * :func:`plot_null` — the label-shuffle null AUC histogram with the observed AUC
    marked and the empirical p. **Conditional:** only meaningful when the null was run
    (``run_null=True``); it raises otherwise.
  * :func:`plot_coefficients` — the top-N features by |all-data standardized weight|,
    each a diamond at its final weight over its resample IQR, colored by **top-k
    membership frequency** (viridis) — the dense-model stability read that replaces
    selection frequency. A vertical line at 0 separates the classes. ``top_n`` defaults
    to the result's ``top_k`` so the rows shown are exactly the set the colour is
    defined on.
  * :func:`plot_hyperparameter_curve` — the all-data tuning **curve** (mean inner-CV
    AUC ± SD over inner folds vs ``C`` on a log axis) with the selected ``C`` marked.
    Replaces the 2-D heatmap: the SVM tunes one hyperparameter. A plateau on the right
    is the hard-margin regime; the marker sits at its most-regularized end.

Colorbars sit beside the axes (they don't overlap the data), so these figures pass no
separate legend figure to :func:`figures.figure_io.save_figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis.classification_svm import SVMClassificationResult
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from sklearn.metrics import auc, roc_curve

from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "classification-svm-figures", "version": "0.1"},
    "kind": "module",
    "provides": [
        "plot_roc",
        "plot_null",
        "plot_coefficients",
        "plot_hyperparameter_curve",
        "save_roc",
        "save_null",
        "save_coefficients",
        "save_hyperparameter_curve",
    ],
    "uses": ["analysis.classification_svm", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Four result figures for an SVMClassificationResult: ROC +-SD across outer "
        "folds from the decision-function margins (legend on-axes; per-repeat AUC "
        "annotated), the label-shuffle null AUC histogram (conditional on the null "
        "being run), the top-N signed-weight plot (final weight + resample IQR, "
        "colored by top-k membership frequency on viridis — the dense-model stability "
        "read; top_n defaults to the result's top_k), and the 1-D C tuning curve "
        "(mean inner-CV AUC +-SD vs log C, selected C marked) that replaces the 2-D "
        "heatmap. Dual-export via figure-io; colorbars beside the axes (no separate "
        "legend figure). Study-agnostic; fail-loud."
    ),
}

# Fixed Okabe-Ito palette entries for the (non-categorical) ROC styling. These encode no
# metadata category, so they are house-style constants rather than color-registry slots.
_ROC_COLOR = "#0072B2"  # Okabe-Ito blue — the mean ROC + band
_OBSERVED_COLOR = "#D55E00"  # Okabe-Ito vermillion — observed-AUC / selected-C marker
_CHANCE_COLOR = "#999999"
_NULL_FILL = "#999999"

_ROC_GRID = np.linspace(0.0, 1.0, 100)


# --------------------------------------------------------------------------- #
# ROC
# --------------------------------------------------------------------------- #
def plot_roc(result: SVMClassificationResult, *, title: str | None = None) -> Figure:
    """Mean ROC (± 1 SD) across outer nested-CV folds, with a chance diagonal.

    Each outer fold contributes one ROC curve from its held-out **margins**; the curves
    are interpolated onto a common FPR grid and averaged. The legend (chance / mean ROC
    / ±1 SD) is on-axes; balanced accuracy, average precision, per-class N, and the
    per-repeat AUC are annotated.
    """
    if not result.fold_predictions:
        raise ValueError("result has no fold predictions to draw a ROC from.")
    tprs: list[np.ndarray] = []
    aucs: list[float] = []
    for fold in result.fold_predictions:
        fpr, tpr, _ = roc_curve(fold.y_true, fold.y_score)
        interp = np.interp(_ROC_GRID, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
        aucs.append(float(auc(fpr, tpr)))
    tpr_stack = np.asarray(tprs, dtype=float)
    mean_tpr = tpr_stack.mean(axis=0)
    mean_tpr[-1] = 1.0
    sd_tpr = tpr_stack.std(axis=0)
    mean_auc = float(np.mean(aucs))
    sd_auc = float(np.std(aucs))

    with publication_style():
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        try:
            ax.plot([0, 1], [0, 1], "--", color=_CHANCE_COLOR, label="chance", zorder=1)
            ax.plot(
                _ROC_GRID,
                mean_tpr,
                color=_ROC_COLOR,
                lw=2,
                label=f"mean ROC (AUC = {mean_auc:.3f} ± {sd_auc:.3f})",
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
            _roc_annotation(ax, result)
            _apply_title(fig, title, _roc_default_title(result), result)
        except BaseException:
            plt.close(fig)
            raise
    return fig


def _roc_annotation(ax: Axes, result: SVMClassificationResult) -> None:
    text = (
        f"balanced accuracy = {result.cv_balanced_accuracy:.3f}  (margin at 0)\n"
        f"average precision = {result.cv_average_precision:.3f}\n"
        f"{result.positive_label}: N={result.n_positive}  |  "
        f"{result.negative_label}: N={result.n_negative}"
    )
    if len(result.repeat_aucs) > 1:
        rep = np.asarray(result.repeat_aucs, dtype=float)
        text += (
            f"\nper-repeat AUC = {rep.mean():.3f} ± {rep.std():.3f} "
            f"({len(rep)} repeats)"
        )
    text += "\nscores = SVM margin (uncalibrated)"
    if result.null_p is None:
        text += "\nnull not run — exploratory"
    else:
        text += f"\nvs shuffle null: p = {result.null_p:.4f}"
    ax.text(
        0.97,
        0.30,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "lightgray"},
    )


def _roc_default_title(result: SVMClassificationResult) -> str:
    return (
        f"{result.positive_label} vs {result.negative_label} — linear SVM (nested CV)"
    )


# --------------------------------------------------------------------------- #
# Null histogram
# --------------------------------------------------------------------------- #
def plot_null(result: SVMClassificationResult, *, title: str | None = None) -> Figure:
    """Label-shuffle null AUC distribution with the observed AUC and empirical p.

    Raises when the null was not run (``run_null=False``): there is nothing to draw.
    """
    if result.null_aucs is None or result.observed_auc is None or result.null_p is None:
        raise ValueError(
            "null was not run for this result; call classify_svm(..., run_null=True) "
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
                f"Label-shuffle null ({n_perm} permutations, fixed C)",
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
    result: SVMClassificationResult,
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
                f"standardized SVM weight  "
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
# Hyperparameter curve (the 1-D counterpart of the heatmap)
# --------------------------------------------------------------------------- #
def plot_hyperparameter_curve(
    result: SVMClassificationResult, *, title: str | None = None
) -> Figure:
    """The all-data tuning curve (mean inner-CV AUC ± SD vs C) with the selected C."""
    means = np.asarray(result.grid_scores, dtype=float)
    sds = np.asarray(result.grid_scores_sd, dtype=float)
    c_grid = np.asarray(result.c_grid, dtype=float)
    if means.shape != c_grid.shape or sds.shape != c_grid.shape:
        raise ValueError(
            f"grid_scores {means.shape} / grid_scores_sd {sds.shape} do not match the "
            f"C grid {c_grid.shape}."
        )
    if result.best_c not in result.c_grid:
        raise ValueError(f"best_c {result.best_c!r} is not in the C grid.")
    with publication_style():
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        try:
            ax.set_xscale("log")
            if len(c_grid) > 1:
                ax.fill_between(
                    c_grid,
                    means - sds,
                    means + sds,
                    color=_ROC_COLOR,
                    alpha=0.2,
                    label="± 1 SD over inner folds",
                    zorder=1,
                )
                ax.plot(c_grid, means, color=_ROC_COLOR, lw=2, zorder=2)
            ax.scatter(
                c_grid,
                means,
                color=_ROC_COLOR,
                s=28,
                zorder=3,
                label="mean inner-CV AUC",
            )
            ax.axvline(result.best_c, color=_OBSERVED_COLOR, ls="--", lw=1.2, zorder=2)
            best_idx = int(np.flatnonzero(c_grid == result.best_c)[0])
            ax.scatter(
                [result.best_c],
                [means[best_idx]],
                marker="o",
                s=110,
                facecolor="none",
                edgecolor=_OBSERVED_COLOR,
                linewidths=2,
                zorder=4,
                label=f"selected C = {result.best_c:g}",
            )
            finite = np.isfinite(means)
            lo = float(np.min((means - sds)[finite])) if finite.any() else 0.5
            hi = float(np.max((means + sds)[finite])) if finite.any() else 1.0
            ax.set_ylim(max(0.0, min(0.5, lo - 0.02)), min(1.02, max(1.0, hi + 0.02)))
            ax.set_xlabel("C (soft-margin penalty; hard margin = large C)")
            ax.set_ylabel("mean inner-CV AUC")
            ax.legend(loc="lower right", fontsize=9)
            _apply_title(
                fig, title, "Hyperparameter search (all-data, C curve)", result
            )
        except BaseException:
            plt.close(fig)
            raise
    return fig


# --------------------------------------------------------------------------- #
# Save wrappers
# --------------------------------------------------------------------------- #
def save_roc(
    result: SVMClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_roc` and dual-export it."""
    return save_figure(plot_roc(result, title=title), output_dir, base_name, dpi=dpi)


def save_null(
    result: SVMClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_null` and dual-export it (raises if the null was not run)."""
    return save_figure(plot_null(result, title=title), output_dir, base_name, dpi=dpi)


def save_coefficients(
    result: SVMClassificationResult,
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


def save_hyperparameter_curve(
    result: SVMClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_hyperparameter_curve` and dual-export it."""
    fig = plot_hyperparameter_curve(result, title=title)
    return save_figure(fig, output_dir, base_name, dpi=dpi)


def _feature_list_suffix(result: SVMClassificationResult) -> str:
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
    fig: Figure, title: str | None, default: str, result: SVMClassificationResult
) -> None:
    base = title if title is not None else default
    fig.suptitle(base + _feature_list_suffix(result), fontsize=13, weight="bold")
