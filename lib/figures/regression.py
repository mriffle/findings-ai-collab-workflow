"""Result figures for an elastic-net linear regression.

TEMPLATE (lib/) — a *seed* for a project's regression-figures module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

Four figures read a :class:`~analysis.regression.RegressionResult` (the
continuous-outcome counterparts of the classifier figures):

  * :func:`plot_predicted_vs_observed` — per-sample true-vs-predicted scatter
    (predictions averaged across the outer nested-CV repeats), a dashed ``y = x``, a
    fitted line, and R² / RMSE / MAE ± SD + N annotated. The legend sits on-axes —
    a documented exception to the separate-legend convention, like the ROC legend
    (conventions/visualization.md).
  * :func:`plot_null` — the target-shuffle null R² histogram with the observed R² marked
    and the empirical p. **Conditional:** only meaningful when the null was run
    (``run_null=True``); it raises otherwise.
  * :func:`plot_coefficients` — the top-N selected features by |all-data standardized
    coefficient|, each a diamond at its final coefficient over its resample IQR, colored
    by **selection frequency** (viridis). A vertical line at 0 separates the directions;
    the colorbar is the stability read. Selected-only (no rejected features), so it
    shares a *skeleton* with the classifier coefficient plot and the Boruta plot.
  * :func:`plot_hyperparameter_heatmap` — the all-data tuning surface (mean inner-CV R²
    over the alpha x l1_ratio grid) with the selected cell boxed. A diagnostic.

Colorbars sit beside the axes (they don't overlap the data), so these figures pass no
separate legend figure to :func:`figures.figure_io.save_figure`.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis.regression import RegressionResult
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "regression-figures", "version": "0.1"},
    "kind": "module",
    "provides": [
        "plot_predicted_vs_observed",
        "plot_null",
        "plot_coefficients",
        "plot_hyperparameter_heatmap",
        "save_predicted_vs_observed",
        "save_null",
        "save_coefficients",
        "save_hyperparameter_heatmap",
    ],
    "uses": ["analysis.regression", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Four result figures for a RegressionResult: the per-sample true-vs-predicted "
        "scatter (predictions averaged across outer folds; y=x + fit line; R2/RMSE/MAE "
        "annotated; legend on-axes — a documented exception, like the ROC legend), the "
        "target-shuffle null R2 histogram (conditional — raises if the null was not "
        "run), the coefficient plot (top-N by |final standardized coef|, diamond = "
        "final coef over the resample IQR, colored by selection frequency on viridis; "
        "selected-only, so it shares a skeleton with the classifier/Boruta plots), and "
        "the alpha x "
        "l1_ratio hyperparameter heatmap with the selected cell boxed. Colorbars sit "
        "beside the axes (no separate legend image). Dual-export via figure-io. "
        "Study-agnostic; fail-loud."
    ),
}

# Fixed Okabe-Ito palette entries for the (non-categorical) styling. These encode no
# metadata category, so they are house-style constants rather than color-registry slots.
_SCATTER_COLOR = "#0072B2"  # Okabe-Ito blue — the scatter points
_YX_COLOR = "#999999"  # the y = x reference
_FIT_COLOR = "#D55E00"  # Okabe-Ito vermillion — the fitted line + observed marker
_NULL_FILL = "#999999"


# --------------------------------------------------------------------------- #
# Internal: aggregate held-out predictions per sample across outer repeats
# --------------------------------------------------------------------------- #
def _aggregate_predictions(result: RegressionResult) -> tuple[np.ndarray, np.ndarray]:
    """Average each sample's held-out prediction across the outer nested-CV repeats."""
    preds: dict[int, list[float]] = defaultdict(list)
    truth: dict[int, float] = {}
    for fold in result.fold_predictions:
        for idx, y_t, y_p in zip(
            fold.test_indices, fold.y_true, fold.y_pred, strict=True
        ):
            preds[int(idx)].append(float(y_p))
            truth[int(idx)] = float(y_t)
    order = sorted(truth.keys())
    y_true = np.array([truth[i] for i in order], dtype=float)
    y_pred = np.array([float(np.mean(preds[i])) for i in order], dtype=float)
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# Predicted-vs-observed scatter
# --------------------------------------------------------------------------- #
def plot_predicted_vs_observed(
    result: RegressionResult, *, title: str | None = None
) -> Figure:
    """Per-sample true-vs-predicted scatter (averaged over outer folds) with y=x + fit.

    Each sample's held-out prediction is averaged across the outer nested-CV repeats
    so it appears once; a dashed ``y = x`` and a fitted line are overlaid, and
    R² / RMSE / MAE ± SD + N are annotated. The legend is on-axes (upper-left) — the
    documented exception.
    """
    if not result.fold_predictions:
        raise ValueError("result has no fold predictions to draw a scatter from.")
    y_true, y_pred = _aggregate_predictions(result)
    all_vals = np.concatenate([y_true, y_pred])
    margin = float((all_vals.max() - all_vals.min()) * 0.05) or 1.0
    lo = float(all_vals.min()) - margin
    hi = float(all_vals.max()) + margin

    with publication_style():
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        try:
            ax.plot([lo, hi], [lo, hi], "--", color=_YX_COLOR, lw=1.5, label="y = x")
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            xs = np.linspace(lo, hi, 100)
            ax.plot(
                xs,
                slope * xs + intercept,
                "-",
                color=_FIT_COLOR,
                lw=2,
                label=f"fit (slope = {slope:.2f})",
            )
            ax.scatter(
                y_true,
                y_pred,
                s=42,
                alpha=0.6,
                color=_SCATTER_COLOR,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"true {result.outcome}")
            ax.set_ylabel(f"predicted {result.outcome}")
            ax.legend(loc="upper left", fontsize=9)
            _scatter_annotation(ax, result, len(y_true))
            _apply_title(fig, title, _scatter_default_title(result), result)
        except BaseException:
            plt.close(fig)
            raise
    return fig


def _scatter_annotation(ax: Axes, result: RegressionResult, n: int) -> None:
    text = (
        f"R² = {result.cv_r2:.3f} ± {result.cv_r2_sd:.3f}\n"
        f"RMSE = {result.cv_rmse:.2f}\n"
        f"MAE = {result.cv_mae:.2f}\n"
        f"N = {n}"
    )
    if result.null_p is None:
        text += "\nnull not run — exploratory"
    else:
        text += f"\nvs shuffle null: p = {result.null_p:.4f}"
    ax.text(
        0.97,
        0.03,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "lightgray"},
    )


def _scatter_default_title(result: RegressionResult) -> str:
    return f"{result.outcome} — elastic-net regression (nested CV)"


# --------------------------------------------------------------------------- #
# Null histogram
# --------------------------------------------------------------------------- #
def plot_null(result: RegressionResult, *, title: str | None = None) -> Figure:
    """Target-shuffle null R² distribution with the observed R² and empirical p.

    Raises when the null was not run (``run_null=False``): there is nothing to draw.
    """
    if result.null_r2s is None or result.observed_r2 is None or result.null_p is None:
        raise ValueError(
            "null was not run for this result; call regress(..., run_null=True) to "
            "produce a null distribution before plotting it."
        )
    nulls = np.asarray(result.null_r2s, dtype=float)
    with publication_style():
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        try:
            ax.hist(nulls, bins=25, color=_NULL_FILL, edgecolor="white")
            ax.axvline(
                result.observed_r2,
                color=_FIT_COLOR,
                lw=2,
                label=(
                    f"observed R² = {result.observed_r2:.3f}\n"
                    f"empirical p = {result.null_p:.4f}"
                ),
            )
            ax.set_xlabel("R² under permuted target")
            ax.set_ylabel("count")
            ax.legend(fontsize=9)
            n_perm = int(nulls.size)
            _apply_title(
                fig,
                title,
                f"Target-shuffle null ({n_perm} permutations, fixed hyperparameters)",
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
    result: RegressionResult, *, top_n: int = 25, title: str | None = None
) -> Figure:
    """Top-N selected features by |coefficient|, colored by selection frequency.

    Each feature is a diamond at its **all-data standardized coefficient** over a bar
    spanning its resample IQR (``coef_q25``..``coef_q75``), colored by **selection
    frequency** (viridis). A vertical line at 0 separates the directions (left = lower
    outcome, right = higher outcome). Selected-only — features zeroed in the final model
    do not appear.
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
                f"(- lower {result.outcome}   |   higher {result.outcome} +)"
            )
            shown = min(top_n, len(table))
            _apply_title(
                fig,
                title,
                f"Top {shown} selected features of {len(table)} "
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
    result: RegressionResult, *, title: str | None = None
) -> Figure:
    """The all-data tuning surface (mean inner-CV R²) with the selected cell boxed."""
    grid = np.asarray(result.grid_scores, dtype=float)
    if grid.shape != (len(result.alpha_grid), len(result.l1_grid)):
        raise ValueError(
            f"grid_scores shape {grid.shape} does not match the "
            f"(alpha={len(result.alpha_grid)}, l1={len(result.l1_grid)}) grid."
        )
    with publication_style():
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        try:
            image = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
            ax.set_xticks(range(len(result.l1_grid)))
            ax.set_xticklabels([f"{v:g}" for v in result.l1_grid])
            ax.set_yticks(range(len(result.alpha_grid)))
            ax.set_yticklabels([f"{v:g}" for v in result.alpha_grid])
            ax.set_xlabel("l1_ratio")
            ax.set_ylabel("alpha")
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
            best_row = list(result.alpha_grid).index(result.best_alpha)
            best_col = list(result.l1_grid).index(result.best_l1_ratio)
            ax.add_patch(
                Rectangle(
                    (best_col - 0.5, best_row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=_FIT_COLOR,
                    lw=3,
                )
            )
            fig.colorbar(image, ax=ax, label="mean CV R²", fraction=0.046, pad=0.04)
            _apply_title(fig, title, "Hyperparameter search (all-data)", result)
        except BaseException:
            plt.close(fig)
            raise
    return fig


# --------------------------------------------------------------------------- #
# Save wrappers
# --------------------------------------------------------------------------- #
def save_predicted_vs_observed(
    result: RegressionResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_predicted_vs_observed` and dual-export it."""
    fig = plot_predicted_vs_observed(result, title=title)
    return save_figure(fig, output_dir, base_name, dpi=dpi)


def save_null(
    result: RegressionResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_null` and dual-export it (raises if the null was not run)."""
    return save_figure(plot_null(result, title=title), output_dir, base_name, dpi=dpi)


def save_coefficients(
    result: RegressionResult,
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
    result: RegressionResult,
    output_dir: str | Path,
    base_name: str,
    *,
    title: str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_hyperparameter_heatmap` and dual-export it."""
    fig = plot_hyperparameter_heatmap(result, title=title)
    return save_figure(fig, output_dir, base_name, dpi=dpi)


def _feature_list_suffix(result: RegressionResult) -> str:
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
    fig: Figure, title: str | None, default: str, result: RegressionResult
) -> None:
    base = title if title is not None else default
    fig.suptitle(base + _feature_list_suffix(result), fontsize=13, weight="bold")
