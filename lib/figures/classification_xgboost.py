"""Result figures for a gradient-boosted-tree (XGBoost) classification.

TEMPLATE (lib/) — a *seed* for a project's XGBoost-classification-figures module, not a
finished script. Copy it into the project's ``scripts/`` and adapt the call sites per
study. Held to the correctness charter (conventions/correctness.md): **assume nothing,
verify everything, fail loud.**

Four figures read a :class:`~analysis.classification_xgboost.XGBClassificationResult`,
the tree counterparts of the elastic-net classifier figures:

  * :func:`plot_roc` — the mean ROC across outer nested-CV folds with a ±1 SD band and a
    chance diagonal; balanced accuracy, average precision, and per-class N annotated.
    Its legend sits on-axes (lower-right) — the documented legend exception, like the
    elastic-net ROC (conventions/visualization.md).
  * :func:`plot_null` — the label-shuffle null AUC histogram with the observed AUC and
    the empirical p. **Conditional:** only meaningful when the null was run
    (``run_null=True``); it raises otherwise.
  * :func:`plot_importance` — the top-N features by **gain importance**, each a diamond
    at its all-data importance over its resample IQR, colored by **selection frequency**
    (viridis). **Unsigned** — importances are non-negative, so there is **no zero line**
    and the axis starts at 0 (a magnitude view; this is the divergence from the
    elastic-net signed coefficient plot). It shares a *skeleton* with the Boruta
    importance plot, not an identical look.
  * :func:`plot_hyperparameter_heatmap` — the all-data tuning surface (mean inner-CV AUC
    over the ``max_depth`` x ``learning_rate`` grid) with the selected cell boxed. A
    diagnostic.

Colorbars sit beside the axes (they don't overlap the data), so these figures pass no
separate legend figure to :func:`figures.figure_io.save_figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis.classification_xgboost import XGBClassificationResult
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import auc, roc_curve

from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "classification-xgboost-figures", "version": "0.1"},
    "kind": "module",
    "provides": [
        "plot_roc",
        "plot_null",
        "plot_importance",
        "plot_hyperparameter_heatmap",
        "save_roc",
        "save_null",
        "save_importance",
        "save_hyperparameter_heatmap",
    ],
    "uses": ["analysis.classification_xgboost", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Four result figures for an XGBClassificationResult: ROC +-SD across outer "
        "folds (legend on-axes), the label-shuffle null AUC histogram (conditional on "
        "the null being run), the top-N gain-importance plot (diamond = all-data gain, "
        "resample IQR, colored by selection frequency on viridis; UNSIGNED so no zero "
        "line and the axis starts at 0 — the divergence from the signed coefficient "
        "plot), and the max_depth x learning_rate hyperparameter heatmap with the "
        "selected cell boxed. Dual-export via figure-io; colorbars beside the axes (no "
        "separate legend figure). Study-agnostic; fail-loud."
    ),
}

# Fixed Okabe-Ito palette entries for the (non-categorical) styling. These encode no
# metadata category, so they are house-style constants rather than color-registry slots.
_ROC_COLOR = "#0072B2"  # Okabe-Ito blue — the mean ROC + band
_OBSERVED_COLOR = (
    "#D55E00"  # Okabe-Ito vermillion — the observed-AUC marker + best cell
)
_CHANCE_COLOR = "#999999"
_NULL_FILL = "#999999"

_ROC_GRID = np.linspace(0.0, 1.0, 100)


# --------------------------------------------------------------------------- #
# ROC
# --------------------------------------------------------------------------- #
def plot_roc(result: XGBClassificationResult, *, title: str | None = None) -> Figure:
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


def _roc_annotation(ax: Axes, result: XGBClassificationResult) -> None:
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


def _roc_default_title(result: XGBClassificationResult) -> str:
    return f"{result.positive_label} vs {result.negative_label} — XGBoost (nested CV)"


# --------------------------------------------------------------------------- #
# Null histogram
# --------------------------------------------------------------------------- #
def plot_null(result: XGBClassificationResult, *, title: str | None = None) -> Figure:
    """Label-shuffle null AUC distribution with the observed AUC and empirical p.

    Raises when the null was not run (``run_null=False``): there is nothing to draw.
    """
    if result.null_aucs is None or result.observed_auc is None or result.null_p is None:
        raise ValueError(
            "null was not run for this result; call classify_xgboost(..., "
            "run_null=True) to produce a null distribution before plotting it."
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
# Importance plot (unsigned — the tree analogue of the coefficient plot)
# --------------------------------------------------------------------------- #
def plot_importance(
    result: XGBClassificationResult, *, top_n: int = 25, title: str | None = None
) -> Figure:
    """Top-N features by gain importance, colored by selection frequency.

    Each feature is a diamond at its **all-data gain importance** over a bar spanning
    its resample IQR (``importance_q25``..``importance_q75``), colored by **selection
    frequency** (viridis). Importances are **unsigned** (non-negative), so there is no
    zero line and the axis starts at 0 — a magnitude view. Selected-only — features with
    zero gain in the final model do not appear.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}.")
    table = result.importances
    if len(table) == 0:
        raise ValueError(
            "no features had non-zero gain (nothing was selected); nothing to plot."
        )
    # Largest importance at the top: take the head, then reverse so y increases upward.
    sub = table.head(top_n).iloc[::-1].reset_index(drop=True)
    n = len(sub)
    imp = sub["importance"].to_numpy(dtype=float)
    q25 = sub["importance_q25"].to_numpy(dtype=float)
    q75 = sub["importance_q75"].to_numpy(dtype=float)
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
                    [imp[i]],
                    [i],
                    marker="D",
                    s=48,
                    color=color,
                    edgecolor="black",
                    linewidths=0.5,
                    zorder=5,
                )
            ax.set_xlim(left=0.0)
            ax.set_yticks(range(n))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_xlabel(f"{result.importance_type} importance (unsigned)")
            shown = min(top_n, len(table))
            _apply_title(
                fig,
                title,
                f"Top {shown} features of {len(table)} "
                f"(diamond = final gain, bar = resample IQR)",
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
    result: XGBClassificationResult, *, title: str | None = None
) -> Figure:
    """The all-data tuning surface (mean inner-CV AUC) with the selected cell boxed."""
    grid = np.asarray(result.grid_scores, dtype=float)
    n_depth = len(result.max_depth_grid)
    n_lr = len(result.learning_rate_grid)
    if grid.shape != (n_depth, n_lr):
        raise ValueError(
            f"grid_scores shape {grid.shape} does not match the "
            f"(max_depth={n_depth}, learning_rate={n_lr}) grid."
        )
    with publication_style():
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        try:
            image = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
            ax.set_xticks(range(n_lr))
            ax.set_xticklabels([f"{v:g}" for v in result.learning_rate_grid])
            ax.set_yticks(range(n_depth))
            ax.set_yticklabels([f"{v:g}" for v in result.max_depth_grid])
            ax.set_xlabel("learning_rate")
            ax.set_ylabel("max_depth")
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
            best_row = list(result.max_depth_grid).index(
                int(result.best_params["max_depth"])
            )
            best_col = list(result.learning_rate_grid).index(
                result.best_params["learning_rate"]
            )
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
    result: XGBClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_roc` and dual-export it."""
    return save_figure(plot_roc(result, title=title), output_dir, base_name, dpi=dpi)


def save_null(
    result: XGBClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_null` and dual-export it (raises if the null was not run)."""
    return save_figure(plot_null(result, title=title), output_dir, base_name, dpi=dpi)


def save_importance(
    result: XGBClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    top_n: int = 25,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_importance` and dual-export it."""
    fig = plot_importance(result, top_n=top_n, title=title)
    return save_figure(fig, output_dir, base_name, dpi=dpi)


def save_hyperparameter_heatmap(
    result: XGBClassificationResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_hyperparameter_heatmap` and dual-export it."""
    fig = plot_hyperparameter_heatmap(result, title=title)
    return save_figure(fig, output_dir, base_name, dpi=dpi)


def _feature_list_suffix(result: XGBClassificationResult) -> str:
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
    fig: Figure, title: str | None, default: str, result: XGBClassificationResult
) -> None:
    base = title if title is not None else default
    fig.suptitle(base + _feature_list_suffix(result), fontsize=13, weight="bold")
