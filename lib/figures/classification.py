"""Result figures for an elastic-net logistic classification.

TEMPLATE (lib/) — a *seed* for a project's classification-figures module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

Four figures read a :class:`~analysis.classification.ClassificationResult`:

  * :func:`plot_roc` — the mean ROC across outer nested-CV folds with a ±1 SD band and a
    chance diagonal; balanced accuracy, average precision, and per-class N annotated.
    The legend sits on-axes (lower-right, where a good classifier leaves space) — a
    documented exception to the separate-legend convention, like the correlation
    r-colorbar (conventions/visualization.md).
  * :func:`plot_null` — the label-shuffle null AUC histogram with the observed AUC
    marked and the empirical p. **Conditional:** only meaningful when the null was run
    (``run_null=True``); it raises otherwise.
  * :func:`plot_coefficients` — the top-N selected features by |all-data standardized
    coefficient|, each a diamond at its final coefficient over its resample IQR, colored
    by **selection frequency** (viridis). A vertical line at 0 separates the classes;
    the colorbar is the stability read. Selected-only (no rejected features), so it
    shares a *skeleton* with the Boruta importance plot, not an identical look.
  * :func:`plot_hyperparameter_heatmap` — the all-data tuning surface (mean inner-CV AUC
    over the C x l1_ratio grid) with the selected cell boxed. A diagnostic.

Colorbars sit beside the axes (they don't overlap the data), so these figures pass no
separate legend figure to :func:`figures.figure_io.save_figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis.classification import ClassificationResult
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import auc, roc_curve

from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "classification-figures", "version": "0.1"},
    "kind": "module",
    "provides": [
        "plot_roc",
        "plot_null",
        "plot_coefficients",
        "plot_hyperparameter_heatmap",
        "save_roc",
        "save_null",
        "save_coefficients",
        "save_hyperparameter_heatmap",
    ],
    "uses": ["analysis.classification", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Four result figures for a ClassificationResult: ROC +-SD across outer folds "
        "(legend on-axes), the label-shuffle null AUC histogram (conditional on the "
        "null being run), the top-N coefficient plot (final coef + resample IQR, "
        "colored by selection frequency on viridis, selected-only), and the C x "
        "l1_ratio hyperparameter heatmap with the selected cell boxed. Dual-export via "
        "figure-io; colorbars beside the axes (no separate legend figure). "
        "Study-agnostic; fail-loud."
    ),
}

# Fixed Okabe-Ito palette entries for the (non-categorical) ROC styling. These encode no
# metadata category, so they are house-style constants rather than color-registry slots.
_ROC_COLOR = "#0072B2"  # Okabe-Ito blue — the mean ROC + band
_OBSERVED_COLOR = "#D55E00"  # Okabe-Ito vermillion — the observed-AUC marker
_CHANCE_COLOR = "#999999"
_NULL_FILL = "#999999"

_ROC_GRID = np.linspace(0.0, 1.0, 100)


# --------------------------------------------------------------------------- #
# ROC
# --------------------------------------------------------------------------- #
def plot_roc(result: ClassificationResult, *, title: str | None = None) -> Figure:
    """Mean ROC (± 1 SD) across outer nested-CV folds, with a chance diagonal.

    Each outer fold contributes one ROC curve; the curves are interpolated onto a common
    FPR grid and averaged. The legend (chance / mean ROC / ±1 SD) is on-axes; balanced
    accuracy, average precision, and per-class N are annotated.
    """
    if not result.fold_predictions:
        raise ValueError("result has no fold predictions to draw a ROC from.")
    tprs: list[np.ndarray] = []
    aucs: list[float] = []
    for fold in result.fold_predictions:
        fpr, tpr, _ = roc_curve(fold.y_true, fold.y_prob)
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


def _roc_annotation(ax: Axes, result: ClassificationResult) -> None:
    text = (
        f"balanced accuracy = {result.cv_balanced_accuracy:.3f}\n"
        f"average precision = {result.cv_average_precision:.3f}\n"
        f"{result.positive_label}: N={result.n_positive}  |  "
        f"{result.negative_label}: N={result.n_negative}"
    )
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


def _roc_default_title(result: ClassificationResult) -> str:
    return (
        f"{result.positive_label} vs {result.negative_label} — "
        f"elastic-net logistic (nested CV)"
    )


# --------------------------------------------------------------------------- #
# Null histogram
# --------------------------------------------------------------------------- #
def plot_null(result: ClassificationResult, *, title: str | None = None) -> Figure:
    """Label-shuffle null AUC distribution with the observed AUC and empirical p.

    Raises when the null was not run (``run_null=False``): there is nothing to draw.
    """
    if result.null_aucs is None or result.observed_auc is None or result.null_p is None:
        raise ValueError(
            "null was not run for this result; call classify(..., run_null=True) to "
            "produce a null distribution before plotting it."
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
                f"Label-shuffle null ({n_perm} permutations, fixed hyperparameters)",
                result,
            )
        except BaseException:
            plt.close(fig)
            raise
    return fig


# --------------------------------------------------------------------------- #
# Coefficient / importance plot
# --------------------------------------------------------------------------- #
def plot_coefficients(
    result: ClassificationResult, *, top_n: int = 25, title: str | None = None
) -> Figure:
    """Top-N selected features by |coefficient|, colored by selection frequency.

    Each feature is a diamond at its **all-data standardized coefficient** over a bar
    spanning its resample IQR (``coef_q25``..``coef_q75``), colored by **selection
    frequency** (viridis). A vertical line at 0 separates the classes (left = negative,
    right = positive). Selected-only — features zeroed in the final model do not appear.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}.")
    table = result.coefficients
    if len(table) == 0:
        raise ValueError(
            "no features were selected (all coefficients zero); nothing to plot."
        )
    # Largest |coef| at the top: take the head, then reverse so y increases upward.
    sub = table.head(top_n).iloc[::-1].reset_index(drop=True)
    n = len(sub)
    coef = sub["coef"].to_numpy(dtype=float)
    q25 = sub["coef_q25"].to_numpy(dtype=float)
    q75 = sub["coef_q75"].to_numpy(dtype=float)
    freq = sub["selection_frequency"].to_numpy(dtype=float)
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
                f"standardized coefficient  "
                f"(- {result.negative_label}   |   {result.positive_label} +)"
            )
            shown = min(top_n, len(table))
            _apply_title(
                fig,
                title,
                f"Top {shown} selected proteins of {len(table)} "
                f"(diamond = final coef, bar = resample IQR)",
                result,
            )
            mappable = ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(
                mappable, ax=ax, label="selection frequency", fraction=0.046, pad=0.04
            )
        except BaseException:
            plt.close(fig)
            raise
    return fig


# --------------------------------------------------------------------------- #
# Hyperparameter heatmap
# --------------------------------------------------------------------------- #
def plot_hyperparameter_heatmap(
    result: ClassificationResult, *, title: str | None = None
) -> Figure:
    """The all-data tuning surface (mean inner-CV AUC) with the selected cell boxed."""
    grid = np.asarray(result.grid_scores, dtype=float)
    if grid.shape != (len(result.c_grid), len(result.l1_grid)):
        raise ValueError(
            f"grid_scores shape {grid.shape} does not match the "
            f"(C={len(result.c_grid)}, l1={len(result.l1_grid)}) grid."
        )
    with publication_style():
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        try:
            image = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
            ax.set_xticks(range(len(result.l1_grid)))
            ax.set_xticklabels([f"{v:g}" for v in result.l1_grid])
            ax.set_yticks(range(len(result.c_grid)))
            ax.set_yticklabels([f"{v:g}" for v in result.c_grid])
            ax.set_xlabel("l1_ratio")
            ax.set_ylabel("C")
            vmid = 0.5 * (np.nanmin(grid) + np.nanmax(grid))
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    value = grid[r, c]
                    if not np.isfinite(value):
                        continue
                    ax.text(
                        c,
                        r,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white" if value < vmid else "black",
                        fontsize=8,
                    )
            best_row = list(result.c_grid).index(result.best_c)
            best_col = list(result.l1_grid).index(result.best_l1_ratio)
            ax.add_patch(
                Rectangle(
                    (best_col - 0.5, best_row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=_OBSERVED_COLOR,
                    lw=3,
                )
            )
            fig.colorbar(image, ax=ax, label="mean CV AUC", fraction=0.046, pad=0.04)
            _apply_title(fig, title, "Hyperparameter search (all-data)", result)
        except BaseException:
            plt.close(fig)
            raise
    return fig


# --------------------------------------------------------------------------- #
# Save wrappers
# --------------------------------------------------------------------------- #
def save_roc(
    result: ClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_roc` and dual-export it."""
    return save_figure(plot_roc(result, title=title), output_dir, base_name, dpi=dpi)


def save_null(
    result: ClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_null` and dual-export it (raises if the null was not run)."""
    return save_figure(plot_null(result, title=title), output_dir, base_name, dpi=dpi)


def save_coefficients(
    result: ClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    top_n: int = 25,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_coefficients` and dual-export it."""
    fig = plot_coefficients(result, top_n=top_n, title=title)
    return save_figure(fig, output_dir, base_name, dpi=dpi)


def save_hyperparameter_heatmap(
    result: ClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_hyperparameter_heatmap` and dual-export it."""
    fig = plot_hyperparameter_heatmap(result, title=title)
    return save_figure(fig, output_dir, base_name, dpi=dpi)


def _feature_list_suffix(result: ClassificationResult) -> str:
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
    fig: Figure, title: str | None, default: str, result: ClassificationResult
) -> None:
    base = title if title is not None else default
    fig.suptitle(base + _feature_list_suffix(result), fontsize=13, weight="bold")
