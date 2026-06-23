"""PCA scatter figures for a :class:`~common.data_loading.Dataset`.

TEMPLATE (lib/) — a *seed* for a project's PCA-plotting module, not a finished script.
Copy it into the project's ``scripts/`` and adapt the call sites per study. Held to the
correctness charter (conventions/correctness.md): **assume nothing, verify everything,
fail loud.**

What it draws (one figure): two scatter panels — **PC1 vs PC2** and **PC3 vs PC4** —
each ringed with marginal distributions, points colored by one sample-metadata column:

  * **Categorical** (``continuous=False``): a color per group, pulled from the project
    color registry (:mod:`figures.colors`) so a value keeps its color across every
    figure, capped at eight categories (the registry raises beyond that). Marginals are
    per-group Gaussian-KDE density curves; the annotation is a per-PC Kruskal-Wallis
    p-value (Mann-Whitney U for two groups) testing whether groups separate along a PC.
  * **Continuous** (``continuous=True``): points shaded by a perceptually-uniform
    colormap. Marginals are per-PC scatter + OLS regression line with a 95% CI; the
    annotation is the slope and p-value of the variable regressed on that PC.

Convention wiring (conventions/visualization.md): categorical colors come from the
registry (consistent + the >8-category guard); the main figure carries **no** baked-in
legend (a baked legend overlaps the data), and the legend is rendered as its **own
figure** — a swatch key (categorical) or a colorbar (continuous) — saved beside it as
``<base>.legend.{svg,png}`` via :func:`figures.figure_io.save_figure`. **Reference**
samples (QC pools, bridges) can be greyed via ``background_values`` so they sit behind
the biology without consuming a palette slot.

Scale: PCA after per-feature standardization is most meaningful on a roughly symmetric
(log-ish) scale — raw linear intensities are right-skewed and a few large features
dominate. This template **warns** (it does not refuse) when the ``Dataset`` scale is
not log-ish, because standardization partly mitigates skew and PCA on linear data is a
choice, not a correctness bug — unlike ComBat, which hard-refuses. Normalize +
``log2_transform`` (or VSN) first.

This template makes NO study decisions: which column to color by, which samples are
reference, and the registry namespace are all the caller's choice. Sample exclusions and
relabelings live in the project copy, applied to the ``Dataset`` before plotting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.data_loading import LOG_SCALES, Dataset
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "pca-plot", "version": "0.3"},
    "kind": "module",
    "provides": [
        "PCAScaleWarning",
        "PCAResult",
        "PCAPlot",
        "compute_pca",
        "plot_pca",
        "save_pca",
    ],
    "uses": ["common.data_loading", "figures.colors", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "PCA scatter figures from a Dataset: PC1/PC2 + PC3/PC4 panels with marginal "
        "KDE (categorical, Kruskal-Wallis/Mann-Whitney p) or regression (continuous, "
        "slope+p). Per-feature standardized PCA; categorical colors from the project "
        "registry with the >8-category guard; greyable reference samples; warns on "
        "non-log scale; dual-export plus a separate legend image (swatches/colorbar). "
        "Study-agnostic; fail-loud."
    ),
}

# Number of principal components the two-panel layout renders (PC1..PC4).
_N_COMPONENTS = 4


class PCAScaleWarning(UserWarning):
    """Warning that PCA is running on a non-log-ish (e.g. linear) abundance scale."""


@dataclass(frozen=True)
class PCAResult:
    """Outcome of :func:`compute_pca`.

    Attributes
    ----------
    scores:
        ``(n_samples, n_components)`` PCA scores (the projected coordinates).
    explained_variance_ratio:
        ``(n_components,)`` fraction of total variance per component, in ``[0, 1]``.
    standardized:
        Whether features were z-scored before PCA.
    """

    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    standardized: bool


@dataclass
class PCAPlot:
    """A rendered PCA figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (two scatter panels with marginals, no baked legend).
    legend_figure:
        A standalone legend figure — categorical swatches or a continuous colorbar —
        saved beside the main figure as ``<base>.legend.{svg,png}`` by :func:`save_pca`.
    result:
        The underlying :class:`PCAResult`.
    color_map:
        Categorical only: ``{value: hex}`` actually drawn (``None`` for continuous).
    """

    figure: Figure
    legend_figure: Figure
    result: PCAResult
    color_map: dict[str, str] | None


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #


def compute_pca(
    dataset: Dataset,
    *,
    n_components: int = _N_COMPONENTS,
    standardize: bool = True,
    random_state: int = 0,
) -> PCAResult:
    """Run PCA on a :class:`Dataset`'s abundances (samples x features).

    Features are z-scored first when ``standardize`` (the default) so each feature
    contributes comparably — PCA on the correlation rather than covariance matrix, the
    usual choice for omics where feature scales differ wildly.

    ``random_state`` is accepted so it can be **recorded** in a finding's
    ``provenance.seed`` (conventions/coding.md). The full SVD used here is already
    exactly deterministic, so the seed has no effect today; it is plumbed through so the
    value is explicit and so a later switch to a randomized solver stays seeded.

    Raises (fail loud) if abundances are not finite (resolve missingness first) or if
    ``n_components`` exceeds ``min(n_samples, n_features)``.
    """
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(
            f"abundances must be 2D (n_samples, n_features); got {abundances.shape}."
        )
    n_samples, n_features = abundances.shape
    if not np.isfinite(abundances).all():
        n_bad = int((~np.isfinite(abundances)).sum())
        raise ValueError(
            f"compute_pca requires finite abundances but found {n_bad} non-finite "
            f"value(s) (NaN/Inf). Resolve missingness before PCA (the loader's "
            f"missing-value policy / imputation / normalization)."
        )
    if n_samples < 2:
        raise ValueError(f"PCA needs >= 2 samples; got {n_samples}.")
    max_components = min(n_samples, n_features)
    if not 1 <= n_components <= max_components:
        raise ValueError(
            f"n_components={n_components} must be between 1 and "
            f"min(n_samples, n_features)={max_components}."
        )

    matrix = StandardScaler().fit_transform(abundances) if standardize else abundances
    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
    scores = np.asarray(pca.fit_transform(matrix), dtype=float)
    ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    return PCAResult(
        scores=scores, explained_variance_ratio=ratio, standardized=standardize
    )


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def plot_pca(
    dataset: Dataset,
    color_by: str,
    *,
    category: str | None = None,
    continuous: bool = False,
    colormap: str = "viridis",
    background_values: object = (),
    feature_type: str = "feature",
    title: str | None = None,
    legend_title: str | None = None,
    show_marginal_stats: bool = True,
    standardize: bool = True,
    random_state: int = 0,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> PCAPlot:
    """Render a PCA figure colored by the metadata column ``color_by``.

    Parameters
    ----------
    dataset:
        The data to project. Abundances must be finite; warns if the scale is not
        log-ish (see the module docstring).
    color_by:
        Sample-metadata column whose values color the points. Must exist and (for
        ``continuous=False``) have no missing values.
    category:
        Color-registry namespace for categorical coloring; defaults to ``color_by``.
        Ignored when ``continuous``.
    continuous:
        Treat ``color_by`` as a continuous variable (colormap + regression marginals)
        rather than categorical (registry colors + KDE marginals).
    colormap:
        Matplotlib colormap for the continuous case. Default ``"viridis"``
        (perceptually uniform, colorblind-safe).
    background_values:
        Categorical only: values to grey out as reference/background (drawn underneath,
        :data:`~figures.colors.BACKGROUND_COLOR`, not counted toward the color cap).
    feature_type:
        Noun for the features in titles/legend (e.g. ``"protein"``, ``"precursor"``).
    title:
        Optional figure suptitle.
    legend_title:
        Title for the companion legend figure; defaults to ``color_by``.
    show_marginal_stats:
        Annotate the marginal panels with the test statistic (default ``True``).
    standardize:
        Z-score features before PCA (default ``True``); see :func:`compute_pca`.
    random_state:
        Seed forwarded to :func:`compute_pca` (recordable in ``provenance.seed``; inert
        for the deterministic full SVD).
    registry_path:
        Color registry JSON (categorical only). Default ``state/color_registry.json``.
    persist_colors:
        Write newly assigned categorical colors back to the registry (default ``True``).

    Returns
    -------
    PCAPlot
        The main figure, the companion legend figure, the :class:`PCAResult`, and
        (categorical) the color map drawn.
    """
    if color_by not in dataset.metadata.columns:
        raise ValueError(
            f"color_by {color_by!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )
    if dataset.scale not in LOG_SCALES:
        warnings.warn(
            f"PCA is running on scale {dataset.scale!r}, which is not log-ish "
            f"{sorted(LOG_SCALES)}. Raw/linear intensities are right-skewed and a few "
            f"large features can dominate the components; normalize + log2_transform "
            f"(or VSN) first for a more faithful projection.",
            PCAScaleWarning,
            stacklevel=2,
        )

    result = compute_pca(
        dataset,
        n_components=_N_COMPONENTS,
        standardize=standardize,
        random_state=random_state,
    )
    scores = result.scores
    variance_pct = result.explained_variance_ratio * 100.0

    title_for_legend = legend_title if legend_title is not None else color_by

    color_map: dict[str, str] | None
    # Build and render inside the publication style so the figures carry the shared
    # defaults regardless of how the caller saves them (spines/fonts are resolved at
    # artist-creation time, so the style must be active during rendering, not at save).
    # The figure is created BEFORE the color/metadata validators run, so any raise from
    # them is caught to close the orphaned figure — never leak it back to the caller.
    with publication_style():
        fig, panel = _build_layout(title=title, feature_type=feature_type)
        try:
            if continuous:
                values = _continuous_values(dataset, color_by)
                _plot_continuous(scores, values, panel, colormap, show_marginal_stats)
                _finalize_axes(panel, variance_pct, continuous=True)
                color_map = None
                legend_figure = _legend_figure_continuous(
                    values, colormap, title_for_legend
                )
            else:
                labels = _categorical_values(dataset, color_by)
                cat = category if category is not None else color_by
                unique = [str(v) for v in np.unique(labels)]
                color_map = assign_colors(
                    cat,
                    unique,
                    registry_path=registry_path,
                    background_values=background_values,
                    persist=persist_colors,
                )
                background = {str(v) for v in _as_list(background_values)}
                _plot_categorical(
                    scores, labels, panel, color_map, background, show_marginal_stats
                )
                _finalize_axes(panel, variance_pct, continuous=False)
                legend_figure = _legend_figure_categorical(
                    unique, color_map, title_for_legend
                )
        except BaseException:
            plt.close(fig)
            raise

    return PCAPlot(
        figure=fig,
        legend_figure=legend_figure,
        result=result,
        color_map=color_map,
    )


def save_pca(
    dataset: Dataset,
    color_by: str,
    output_dir: str | Path,
    base_name: str,
    *,
    category: str | None = None,
    continuous: bool = False,
    colormap: str = "viridis",
    background_values: object = (),
    feature_type: str = "feature",
    title: str | None = None,
    legend_title: str | None = None,
    show_marginal_stats: bool = True,
    standardize: bool = True,
    random_state: int = 0,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render a PCA figure (:func:`plot_pca`, publication-styled) and save it.

    Writes the main figure as ``<base>.{svg,png}`` and its companion legend as
    ``<base>.legend.{svg,png}`` via :func:`figures.figure_io.save_figure`, and returns
    their paths. ``save_figure`` is called with the default ``close=True``, so the
    figures are closed even if saving fails (no leak on a bad ``base_name``/``dpi``).
    """
    plot = plot_pca(
        dataset,
        color_by,
        category=category,
        continuous=continuous,
        colormap=colormap,
        background_values=background_values,
        feature_type=feature_type,
        title=title,
        legend_title=legend_title,
        show_marginal_stats=show_marginal_stats,
        standardize=standardize,
        random_state=random_state,
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
# Metadata extraction (fail loud)
# --------------------------------------------------------------------------- #


def _categorical_values(dataset: Dataset, color_by: str) -> np.ndarray:
    """Return the categorical color column as a string array; refuse missing values."""
    series = dataset.metadata[color_by]
    if series.isna().any():
        n_bad = int(series.isna().sum())
        raise ValueError(
            f"color_by column {color_by!r} has {n_bad} missing value(s); a missing "
            f"label cannot be colored. Resolve or relabel them before plotting."
        )
    return series.to_numpy().astype(str)


def _continuous_values(dataset: Dataset, color_by: str) -> np.ndarray:
    """Return the continuous color column as a finite float array (fail loud)."""
    series = dataset.metadata[color_by]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        n_bad = int(numeric.isna().sum())
        raise ValueError(
            f"continuous color_by column {color_by!r} has {n_bad} value(s) that are "
            f"missing or non-numeric; cannot map them to a colormap."
        )
    return numeric.to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Panel:
    """The eight axes of the layout (two scatter + four marginal + two title strips)."""

    ax1: Axes
    ax2: Axes
    ax1_top: Axes
    ax1_right: Axes
    ax2_top: Axes
    ax2_right: Axes
    ax1_title: Axes
    ax2_title: Axes


def _build_layout(*, title: str | None, feature_type: str) -> tuple[Figure, _Panel]:
    """Build the figure and its eight axes; return ``(fig, panel)``.

    Port of the source two-panel design: a 4x4 gridspec gives each scatter a top and a
    right marginal axis (shared scales) plus a bold title strip. Axis labels with the
    variance percentages are filled in later by :func:`_finalize_axes`.
    """
    fig = plt.figure(figsize=(18, 12))
    top_margin = 0.92
    if title is not None:
        fig.suptitle(title, fontsize=16, weight="bold", y=0.98)
        top_margin = 0.89

    gs = fig.add_gridspec(
        4,
        4,
        hspace=0.08,
        wspace=0.08,
        width_ratios=[3, 0.5, 3, 0.5],
        height_ratios=[0.3, 0.5, 3, 0.1],
        left=0.08,
        right=0.95,
        top=top_margin,
        bottom=0.14,
    )

    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 2])
    panel = _Panel(
        ax1=ax1,
        ax2=ax2,
        ax1_top=fig.add_subplot(gs[1, 0], sharex=ax1),
        ax1_right=fig.add_subplot(gs[2, 1], sharey=ax1),
        ax2_top=fig.add_subplot(gs[1, 2], sharex=ax2),
        ax2_right=fig.add_subplot(gs[2, 3], sharey=ax2),
        ax1_title=fig.add_subplot(gs[0, 0]),
        ax2_title=fig.add_subplot(gs[0, 2]),
    )
    for ax in (ax1, ax2):
        ax.set_facecolor("white")

    panel.ax1_title.text(
        0.5,
        0.5,
        f"PCA of {feature_type} quantities (PC1 vs PC2)",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    panel.ax1_title.axis("off")
    panel.ax2_title.text(
        0.5,
        0.5,
        f"PCA of {feature_type} quantities (PC3 vs PC4)",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    panel.ax2_title.axis("off")
    return fig, panel


def _finalize_axes(
    panel: _Panel, variance_pct: np.ndarray, *, continuous: bool
) -> None:
    """Set PC axis labels and hide marginal ticks/spines (shared across both modes)."""
    # These PC labels deliberately override the shared style's axes.labelsize (12): on
    # the large 18x12 two-panel canvas 22pt keeps them legible at print scale. A
    # per-figure override, not chartjunk.
    panel.ax1.set_xlabel(f"PC1 ({variance_pct[0]:.1f}% variance)", fontsize=22)
    panel.ax1.set_ylabel(f"PC2 ({variance_pct[1]:.1f}% variance)", fontsize=22)
    panel.ax2.set_xlabel(f"PC3 ({variance_pct[2]:.1f}% variance)", fontsize=22)
    panel.ax2.set_ylabel(f"PC4 ({variance_pct[3]:.1f}% variance)", fontsize=22)

    # Hide ticks/labels on the four marginals via tick_params (not set_xticks([]), which
    # would clobber the Locator shared with the main scatter axes).
    for ax in (panel.ax1_top, panel.ax2_top, panel.ax1_right, panel.ax2_right):
        ax.tick_params(
            left=False,
            right=False,
            top=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
            labelright=False,
            labeltop=False,
        )
        if not continuous:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    if not continuous:
        panel.ax1_right.set_xlim(left=0)
        panel.ax2_right.set_xlim(left=0)


# --------------------------------------------------------------------------- #
# Marginal helpers
# --------------------------------------------------------------------------- #

# A PC column whose spread is below this fraction of the largest score magnitude is
# treated as degenerate — its variance is numerical noise (e.g. a ~0%-variance trailing
# PC on rank-deficient data). Its marginal KDE would be singular and its marginal
# regression slope explodes (cov/~0), so the marginal is skipped rather than drawn.
_DEGENERATE_REL_TOL = 1e-8


def _degenerate_std_floor(scores: np.ndarray) -> float:
    """Std floor below which a PC column counts as degenerate (no real spread)."""
    scale = float(np.abs(scores).max())
    return _DEGENERATE_REL_TOL * scale if scale > 0 else 0.0


# --------------------------------------------------------------------------- #
# Categorical
# --------------------------------------------------------------------------- #


def _plot_categorical(
    scores: np.ndarray,
    labels: np.ndarray,
    panel: _Panel,
    color_map: dict[str, str],
    background: set[str],
    show_marginal_stats: bool,
) -> None:
    """Scatter both panels by group + per-group KDE marginals + per-PC group-test p."""
    unique_labels = [str(v) for v in np.unique(labels)]
    str_labels = labels.astype(str)

    p_values = _group_test_pvalues(scores, str_labels, unique_labels)

    # Background labels underneath, foreground on top.
    draw_order = [lb for lb in unique_labels if lb in background] + [
        lb for lb in unique_labels if lb not in background
    ]
    for label in draw_order:
        mask = str_labels == label
        panel.ax1.scatter(
            scores[mask, 0],
            scores[mask, 1],
            color=color_map[label],
            edgecolor="k",
            alpha=0.8,
            s=72,
        )
        panel.ax2.scatter(
            scores[mask, 2],
            scores[mask, 3],
            color=color_map[label],
            edgecolor="k",
            alpha=0.8,
            s=72,
        )

    marginals = (
        (panel.ax1_top, 0, False),
        (panel.ax1_right, 1, True),
        (panel.ax2_top, 2, False),
        (panel.ax2_right, 3, True),
    )
    std_floor = _degenerate_std_floor(scores)
    for label in unique_labels:
        mask = str_labels == label
        if int(mask.sum()) < 2:
            # gaussian_kde needs >= 2 samples; the singleton's point still scatters.
            continue
        for ax, pc, vertical in marginals:
            pc_vals = scores[mask, pc]
            if float(pc_vals.std()) <= std_floor:
                continue  # identical / numerically-constant values -> KDE is singular
            grid = np.linspace(scores[:, pc].min(), scores[:, pc].max(), 200)
            density = np.asarray(stats.gaussian_kde(pc_vals)(grid), dtype=float)
            if vertical:
                ax.plot(density, grid, color=color_map[label], linewidth=3)
            else:
                ax.plot(grid, density, color=color_map[label], linewidth=3)

    if show_marginal_stats:
        for ax, p_val in zip(
            (panel.ax1_top, panel.ax1_right, panel.ax2_top, panel.ax2_right),
            p_values,
            strict=True,
        ):
            ax.text(
                0.02,
                0.95,
                f"p = {p_val:.4f}",
                transform=ax.transAxes,
                va="top",
                fontsize=16,
            )


def _group_test_pvalues(
    scores: np.ndarray, str_labels: np.ndarray, unique_labels: list[str]
) -> list[float]:
    """Per-PC p-value that the groups differ along that PC (Mann-Whitney/Kruskal)."""
    p_values: list[float] = []
    for pc in range(_N_COMPONENTS):
        groups = [scores[str_labels == label, pc] for label in unique_labels]
        try:
            if len(unique_labels) == 2:
                p_value = float(stats.mannwhitneyu(groups[0], groups[1]).pvalue)
            elif len(unique_labels) >= 2:
                p_value = float(stats.kruskal(*groups).pvalue)
            else:
                p_value = float("nan")  # single group: no separation to test
        except ValueError:
            # Degenerate (e.g. all values identical) — skip rather than crash the plot.
            p_value = float("nan")
        p_values.append(p_value)
    return p_values


# --------------------------------------------------------------------------- #
# Continuous
# --------------------------------------------------------------------------- #


def _plot_continuous(
    scores: np.ndarray,
    values: np.ndarray,
    panel: _Panel,
    colormap: str,
    show_marginal_stats: bool,
) -> None:
    """Shade both panels by a continuous variable + regression marginals."""
    panel.ax1.scatter(
        scores[:, 0],
        scores[:, 1],
        c=values,
        cmap=colormap,
        edgecolor="k",
        alpha=0.8,
        s=72,
    )
    panel.ax2.scatter(
        scores[:, 2],
        scores[:, 3],
        c=values,
        cmap=colormap,
        edgecolor="k",
        alpha=0.8,
        s=72,
    )

    cmap_obj = plt.get_cmap(colormap)
    marginals = (
        (panel.ax1_top, 0, False),
        (panel.ax1_right, 1, True),
        (panel.ax2_top, 2, False),
        (panel.ax2_right, 3, True),
    )
    std_floor = _degenerate_std_floor(scores)
    for ax, pc, vertical in marginals:
        if float(scores[:, pc].std()) <= std_floor:
            # Degenerate (~0-variance) PC: regressing on numerical noise gives an
            # exploding, meaningless slope — skip the marginal rather than draw it.
            continue
        slope, p_value = _plot_regression(
            scores[:, pc], values, ax, cmap_obj, vertical=vertical
        )
        if show_marginal_stats:
            ax.text(
                0.02,
                0.95,
                f"slope = {slope:.3g}, p = {p_value:.4f}",
                transform=ax.transAxes,
                va="top",
                fontsize=16,
            )


def _plot_regression(
    pc_vals: np.ndarray,
    color_vals: np.ndarray,
    ax: Axes,
    cmap: Colormap,
    *,
    vertical: bool,
) -> tuple[float, float]:
    """Scatter (PC value vs colored variable) + OLS line with 95% CI on a marginal axis.

    Returns ``(slope, p_value)`` of ``color_vals ~ pc_vals``.

    The band is the 95% **mean-response** confidence interval, which uses the *residual*
    standard error ``s = sqrt(SSE/(n-2))`` and the Student-t quantile ``t_{0.975, n-2}``
    — not ``linregress``'s ``stderr`` (the *slope* SE, ``s / sqrt(Sxx)``) with ``1.96``,
    which understates the band by a factor of ``~1/sqrt(Sxx)``.
    """
    lr = stats.linregress(pc_vals, color_vals)
    slope = float(lr.slope)
    intercept = float(lr.intercept)
    p_value = float(lr.pvalue)

    x_pred = np.linspace(float(pc_vals.min()), float(pc_vals.max()), 100)
    y_pred = intercept + slope * x_pred
    n = pc_vals.shape[0]
    dof = n - 2
    denom = float(np.sum((pc_vals - pc_vals.mean()) ** 2))
    if dof > 0 and denom > 0:
        residuals = color_vals - (intercept + slope * pc_vals)
        resid_se = float(np.sqrt(float(np.sum(residuals**2)) / dof))
        tcrit = float(stats.t.ppf(0.975, dof))
        mean_se = resid_se * np.sqrt(1.0 / n + (x_pred - pc_vals.mean()) ** 2 / denom)
        band = tcrit * mean_se
    else:
        band = np.zeros_like(x_pred)
    point_colors = cmap(_normalize_for_cmap(color_vals))

    if not vertical:
        ax.scatter(
            pc_vals, color_vals, s=20, alpha=0.7, c=point_colors, edgecolor="none"
        )
        ax.plot(x_pred, y_pred, color="gray", linewidth=3)
        ax.fill_between(x_pred, y_pred - band, y_pred + band, alpha=0.2, color="gray")
    else:
        ax.scatter(
            color_vals, pc_vals, s=20, alpha=0.7, c=point_colors, edgecolor="none"
        )
        ax.plot(y_pred, x_pred, color="gray", linewidth=3)
        ax.fill_betweenx(x_pred, y_pred - band, y_pred + band, alpha=0.2, color="gray")
    return slope, p_value


def _normalize_for_cmap(values: np.ndarray) -> np.ndarray:
    """Min-max scale ``values`` to ``[0, 1]`` for colormap lookup (constant -> 0.5)."""
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - vmin) / (vmax - vmin)


# --------------------------------------------------------------------------- #
# Legend figures (rendered separately so the legend never overlaps the plot)
# --------------------------------------------------------------------------- #


def _legend_figure_categorical(
    labels: list[str], color_map: dict[str, str], legend_title: str
) -> Figure:
    """Standalone swatch legend: one colored marker per label, in ``labels`` order.

    Background/reference labels appear with their gray swatch from ``color_map``, so the
    legend image matches the scatter exactly.
    """
    height = max(2.0, 0.35 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(4.0, height))
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=10,
            markerfacecolor=color_map[label],
            markeredgecolor="k",
        )
        for label in labels
    ]
    ax.legend(
        handles,
        labels,
        title=legend_title,
        loc="center",
        frameon=True,
        fontsize=12,
        title_fontsize=13,
    )
    return fig


def _legend_figure_continuous(
    values: np.ndarray, colormap: str, legend_title: str
) -> Figure:
    """Standalone horizontal colorbar spanning the continuous variable's range."""
    fig, ax = plt.subplots(figsize=(6.0, 1.2))
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.5, top=0.85)
    norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    mappable = ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
    mappable.set_array(np.asarray([], dtype=float))
    cbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")
    cbar.set_label(legend_title, fontsize=13)
    return fig


def _as_list(values: object) -> list[object]:
    """Coerce ``values`` to a list, treating a bare string as a single value."""
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple, set, frozenset)):
        return list(values)
    return [values]
