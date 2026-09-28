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
    per-group Gaussian-KDE density curves (each group's curve has unit area, so shapes
    compare regardless of group size), evaluated across the scatter's full view so no
    tail is cut off in mid-air. The annotation is a per-PC Kruskal-Wallis p-value
    (Mann-Whitney U, two-sided, for two groups) testing whether groups separate along a
    PC. **Greyed reference samples (``background_values``) are excluded from the test**
    — they sit behind the biology, so they must not drive its p-value; with fewer than
    two foreground groups the p is ``n/a``.
  * **Continuous** (``continuous=True``): points shaded by a perceptually-uniform
    colormap. Marginals are per-PC scatter + OLS regression line with a 95% band; the
    annotation is the **Pearson r** and its p-value. ``r`` is reported rather than the
    slope because the slope is in variable-units per PC-score-unit, unreadable on
    marginals without tick labels (the sign of either follows the PC's arbitrary
    orientation).

**Independence and ``unit=``.** Without ``unit=`` every sample is treated as an
independent observation: the p-values are the asymptotic Mann-Whitney / Kruskal-Wallis /
Pearson ones and the band is the t-based mean-response CI — and the figure says so
("samples treated as independent"). When samples are **nested in units** (technical
replicates of one animal; several tissues, timepoints, or cultures from one donor;
littermates), pass the unit column as ``unit=``. Every sample still enters the statistic
— nothing is averaged — but the p-value comes from a **permutation test that respects
the nesting**, chosen from the data:

  * the tested variable is **constant within every unit** (genotype, treatment, age) →
    its values are **permuted between whole units** (each unit keeps one value, carried
    by all its samples), so the null distribution carries whatever correlation the
    samples of a unit share — near-duplicates or only loosely related, either way;
  * it **varies within a unit** (timepoint, tissue region, batch, run order) → the
    statistic is computed on **within-unit deviations** (each sample minus its unit's
    mean — the aligned-rank idea for blocked designs) and the values are **permuted
    within each unit**, testing whether the PC moves with the variable *inside* a unit.
    Units where the variable does not vary carry no within-unit information and drop
    out (the method line counts only the units that remain). The drawn fitted line still
    shows the overall association; the p and ``r`` are the within-unit ones.

The statistic is Kruskal-Wallis H (for two groups, the square of the standardized
two-sided Mann-Whitney U — standardized because a between-unit shuffle changes the group
sizes when units differ in size) or ``|r|``. When the number of distinct arrangements
is at most ``n_permutations`` they are **enumerated exactly** (p = the fraction of
arrangements whose statistic is at least the observed one); otherwise
``n_permutations`` seeded draws give ``p = (1 + hits) / (1 + n_permutations)``. Twenty
or fewer arrangements cannot reach p < 0.05 (3 vs 3 units: 20, smallest p = 0.05) —
:class:`PermutationResolutionWarning`. With nesting, the continuous band is a **cluster
bootstrap** (whole units resampled; 2.5/97.5 percentiles of the fitted line), which is
narrow with very few units. The assumption left is that units are exchangeable: a deeper
hierarchy (samples in animals in litters) needs the top level as ``unit=``.

The marginal statistics are written in each panel's title strip (a statistics line and a
method line naming the test and how the p was obtained), not inside the narrow marginal
axes where they overlapped the curves and overflowed the figure edge. They are also
returned as :class:`MarginalTests` on the :class:`PCAPlot` for provenance. The PCA
scores are unsupervised (the labels play no part in them), so testing the labels against
a PC is not circular; the four p-values are descriptive and uncorrected.

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

import itertools
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    "template": {"name": "pca-plot", "version": "0.5"},
    "kind": "module",
    "provides": [
        "PCAScaleWarning",
        "PermutationResolutionWarning",
        "PCAResult",
        "MarginalTests",
        "PCAPlot",
        "compute_pca",
        "plot_pca",
        "save_pca",
    ],
    "uses": ["common.data_loading", "figures.colors", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "PCA scatter figures from a Dataset: PC1/PC2 + PC3/PC4 panels with marginal "
        "KDE (categorical, Kruskal-Wallis/Mann-Whitney p over the non-greyed groups) "
        "or regression (continuous, Pearson r+p); with unit= the p comes from a "
        "permutation test between or within units (nested samples, no averaging). "
        "Per-feature standardized PCA; "
        "categorical colors from the project registry with the >8-category guard; "
        "greyable reference samples; warns on non-log scale; dual-export plus a "
        "separate legend image (swatches/colorbar). "
        "Study-agnostic; fail-loud."
    ),
}

# Number of principal components the two-panel layout renders (PC1..PC4).
_N_COMPONENTS = 4


class PCAScaleWarning(UserWarning):
    """Warning that PCA is running on a non-log-ish (e.g. linear) abundance scale."""


class PermutationResolutionWarning(UserWarning):
    """Warning that a unit permutation test has too few distinct arrangements to go
    below p = 0.05 (e.g. 3 vs 3 units: 20 arrangements, smallest possible p = 0.05)."""


PermutationScheme = Literal["independent", "between_units", "within_units"]


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


@dataclass(frozen=True)
class MarginalTests:
    """The per-PC marginal statistics drawn on a figure, for provenance.

    Attributes
    ----------
    test:
        ``"Mann-Whitney U"``, ``"Kruskal-Wallis"`` or ``"Pearson"``.
    scheme:
        How the p-value was obtained: ``"independent"`` (asymptotic, samples treated as
        independent), ``"between_units"`` (values permuted between whole units) or
        ``"within_units"`` (values permuted within each unit).
    unit:
        The unit column, or ``None``.
    n_samples:
        Samples entering the test (greyed reference samples excluded; for
        ``"within_units"``, only samples in units where the variable varies).
    n_units:
        Distinct units among them (``None`` without ``unit=``).
    n_arrangements:
        Distinct arrangements under the permutation scheme (exact integer; ``None`` for
        ``"independent"``).
    exact:
        Every arrangement was enumerated (else ``n_draws`` seeded random draws).
    n_draws:
        Arrangements evaluated (``None`` for ``"independent"``).
    p_values:
        One per PC1..PC4; NaN = untestable (rendered ``n/a``).
    r_values:
        Continuous only: Pearson r per PC — of the within-unit deviations for
        ``"within_units"`` (NaN for a skipped degenerate PC).
    """

    test: str
    scheme: PermutationScheme
    unit: str | None
    n_samples: int
    n_units: int | None
    n_arrangements: int | None
    exact: bool
    n_draws: int | None
    p_values: tuple[float, ...]
    r_values: tuple[float, ...] | None


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
    marginal_tests:
        The per-PC marginal statistics (computed whether or not they are drawn).
    """

    figure: Figure
    legend_figure: Figure
    result: PCAResult
    color_map: dict[str, str] | None
    marginal_tests: MarginalTests | None = None


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
    unit: str | None = None,
    n_permutations: int = 9999,
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
        Write the per-PC marginal statistics into each panel's title strip (default
        ``True``): categorical, the group-separation p over the non-greyed groups;
        continuous, Pearson ``r`` and its p.
    unit:
        Sample-metadata column naming the independent unit each sample belongs to (an
        animal, donor, litter). When given, the marginal p-values come from a
        permutation test that respects the nesting (between or within units — see the
        module docstring) and the continuous band is a cluster bootstrap; every tested
        sample must have a unit. ``None`` (default) treats samples as independent, and
        the figure says so.
    n_permutations:
        Random draws for the unit permutation test when the distinct arrangements
        outnumber it (else they are enumerated exactly); also the number of cluster
        bootstrap resamples for the continuous band. Default 9999.
    standardize:
        Z-score features before PCA (default ``True``); see :func:`compute_pca`.
    random_state:
        Seed for the permutation draws and the cluster bootstrap (and forwarded to
        :func:`compute_pca`, where the deterministic full SVD ignores it) — record it
        in ``provenance.seed``.
    registry_path:
        Color registry JSON (categorical only). Default ``state/color_registry.json``.
    persist_colors:
        Write newly assigned categorical colors back to the registry (default ``True``).

    Returns
    -------
    PCAPlot
        The main figure, the companion legend figure, the :class:`PCAResult`,
        (categorical) the color map drawn, and the :class:`MarginalTests`.
    """
    if color_by not in dataset.metadata.columns:
        raise ValueError(
            f"color_by {color_by!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )
    if unit is not None and unit not in dataset.metadata.columns:
        raise ValueError(
            f"unit {unit!r} is not a metadata column {list(dataset.metadata.columns)}."
        )
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int):
        raise TypeError(f"n_permutations must be an int; got {n_permutations!r}.")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}.")
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
    nesting = _Nesting.from_dataset(dataset, unit, n_permutations, random_state)

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
                tests = _plot_continuous(
                    scores, values, panel, colormap, show_marginal_stats, nesting
                )
                _finalize_axes(panel, variance_pct, continuous=True)
                color_map = None
                legend_figure = _legend_figure_continuous(
                    values, colormap, title_for_legend
                )
            else:
                labels = _categorical_values(dataset, color_by)
                cat = category if category is not None else color_by
                unique = [str(v) for v in np.unique(labels)]
                # Coerce once, so the greying (registry), the draw order, and the
                # exclusion from the group test all see the same values — a generator
                # would otherwise be consumed by the registry call.
                background_list = _as_list(background_values)
                color_map = assign_colors(
                    cat,
                    unique,
                    registry_path=registry_path,
                    background_values=background_list,
                    persist=persist_colors,
                )
                background = {str(v) for v in background_list}
                tests = _plot_categorical(
                    scores,
                    labels,
                    panel,
                    color_map,
                    background,
                    show_marginal_stats,
                    nesting,
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
        marginal_tests=tests,
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
    unit: str | None = None,
    n_permutations: int = 9999,
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
    The marginal statistics are drawn on the figure; call :func:`plot_pca` +
    ``save_figure`` instead when the :class:`MarginalTests` are needed for provenance.
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
        unit=unit,
        n_permutations=n_permutations,
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
    arr: np.ndarray = series.to_numpy().astype(str)
    return arr


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
    arr: np.ndarray = numeric.to_numpy(dtype=float)
    return arr


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
        height_ratios=[0.45, 0.5, 3, 0.1],
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
        # Density baseline at 0 on both marginal orientations.
        panel.ax1_right.set_xlim(left=0)
        panel.ax2_right.set_xlim(left=0)
        panel.ax1_top.set_ylim(bottom=0)
        panel.ax2_top.set_ylim(bottom=0)


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


def _format_p(p_value: float) -> str:
    """Render a p-value for an annotation: never ``0.0000``, never a bare ``nan``.

    ``p >= 0.001`` prints to three decimals; smaller values print in scientific
    notation (``3.0e-11``) so a strong separation is not flattened to ``0.0000`` (which
    reads as p = 0); a non-finite p (untestable) prints ``n/a``.
    """
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-3:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def _p_text(p_value: float, tests: MarginalTests) -> str:
    """``p = …``, or ``p ≤ …`` at a Monte Carlo floor (no draw beat the observed
    statistic, so ``1 / (n_draws + 1)`` is a bound, not an estimate)."""
    if (
        not tests.exact
        and tests.n_draws is not None
        and np.isfinite(p_value)
        and p_value <= 1.0 / (tests.n_draws + 1) * (1 + 1e-9)
    ):
        return f"p ≤ {_format_p(p_value)}"
    return f"p = {_format_p(p_value)}"


def _write_panel_stats(title_ax: Axes, stats_text: str, method_text: str) -> None:
    """Put the statistics line and the method line under the panel title.

    The narrow right-hand marginals cannot hold a statistic without overlapping the
    curves and overflowing the figure edge, so both panels' per-PC statistics live here.
    """
    title_ax.texts[0].set_y(0.84)
    title_ax.text(0.5, 0.46, stats_text, ha="center", va="center", fontsize=13)
    title_ax.text(
        0.5, 0.12, method_text, ha="center", va="center", fontsize=10, color="0.3"
    )


def _method_text(tests: MarginalTests) -> str:
    """One line naming the test and how its p was obtained."""
    if tests.scheme == "independent":
        return f"{tests.test}, {tests.n_samples} samples treated as independent"
    how = (
        f"exact over {tests.n_arrangements:,} arrangements"
        if tests.exact
        else f"{tests.n_draws:,} draws"
    )
    if tests.scheme == "between_units":
        return (
            f"{tests.test}, permuted between {tests.n_units} units "
            f"({tests.n_samples} samples), {how}"
        )
    return (
        f"{tests.test} on within-unit deviations, permuted within {tests.n_units} "
        f"units ({tests.n_samples} samples), {how}"
    )


def _view_limits(panel: _Panel) -> list[tuple[float, float]]:
    """The scatter view range of PC1..PC4 (after the scatters are drawn)."""
    views = (
        panel.ax1.get_xlim(),
        panel.ax1.get_ylim(),
        panel.ax2.get_xlim(),
        panel.ax2.get_ylim(),
    )
    return [(float(lo), float(hi)) for lo, hi in views]


def _freeze_view(panel: _Panel, limits: list[tuple[float, float]]) -> None:
    """Pin the scatter views so a marginal curve cannot re-autoscale the shared axes."""
    panel.ax1.set_xlim(limits[0])
    panel.ax1.set_ylim(limits[1])
    panel.ax2.set_xlim(limits[2])
    panel.ax2.set_ylim(limits[3])


# --------------------------------------------------------------------------- #
# Nesting — the unit permutation test and the cluster bootstrap
# --------------------------------------------------------------------------- #

# At or below this many distinct arrangements a permutation p cannot go below 0.05.
_MAX_UNRESOLVED_ARRANGEMENTS = 20


@dataclass(frozen=True)
class _Nesting:
    """The unit structure (or its absence) plus the seeded generator for the draws."""

    unit: str | None
    codes: np.ndarray | None  # int code per sample; -1 = missing unit
    n_permutations: int
    rng: np.random.Generator

    @staticmethod
    def from_dataset(
        dataset: Dataset, unit: str | None, n_permutations: int, random_state: int
    ) -> _Nesting:
        codes: np.ndarray | None = None
        if unit is not None:
            # Group on the values themselves, not their text: 1 and "1" stay distinct.
            codes = np.asarray(
                pd.factorize(dataset.metadata[unit], use_na_sentinel=True)[0],
                dtype=np.int64,
            )
        return _Nesting(
            unit=unit,
            codes=codes,
            n_permutations=n_permutations,
            rng=np.random.default_rng(random_state),
        )

    def units_for(self, mask: np.ndarray) -> np.ndarray | None:
        """Unit codes of the tested samples; raise if any tested sample has none."""
        if self.codes is None:
            return None
        codes = np.asarray(self.codes[mask], dtype=np.int64)
        n_missing = int((codes < 0).sum())
        if n_missing:
            raise ValueError(
                f"unit column {self.unit!r} has {n_missing} missing value(s) among the "
                f"tested samples; every tested sample needs a unit. Give a sample that "
                f"stands alone its own unit (e.g. its sample id)."
            )
        return codes


def _n_distinct(values: np.ndarray) -> int:
    """Distinct orderings of ``values`` (a multiset): ``n! / prod(count!)``."""
    _, counts = np.unique(values, return_counts=True)
    total = math.factorial(int(values.size))
    for count in counts:
        total //= math.factorial(int(count))
    return total


def _distinct_permutations(values: np.ndarray) -> np.ndarray:
    """Every distinct ordering of ``values``, one per row (callers bound the count)."""
    uniq, counts = np.unique(values, return_counts=True)
    remaining = [int(c) for c in counts]
    rows: list[list[int]] = []
    prefix: list[int] = []

    def extend() -> None:
        if len(prefix) == values.size:
            rows.append(list(prefix))
            return
        for k, left in enumerate(remaining):
            if left:
                remaining[k] -= 1
                prefix.append(k)
                extend()
                prefix.pop()
                remaining[k] += 1

    extend()
    out: np.ndarray = uniq[np.asarray(rows, dtype=np.int64)]
    return out


def _varying_units(values: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Mask of samples whose unit has more than one distinct ``values`` entry."""
    frame = pd.DataFrame({"unit": units, "value": values})
    spans = frame.groupby("unit")["value"].transform(lambda v: v.min() != v.max())
    return np.asarray(spans, dtype=bool)


def _within_deviations(values: np.ndarray, units: np.ndarray) -> np.ndarray:
    """``values`` minus the mean of its unit (removes every between-unit difference)."""
    means = pd.Series(values).groupby(units).transform("mean").to_numpy(dtype=float)
    deviations: np.ndarray = np.asarray(values, dtype=float) - means
    return deviations


def _permutation_scheme(values: np.ndarray, units: np.ndarray) -> PermutationScheme:
    """Between units if ``values`` is constant within every unit, else within units."""
    frame = pd.DataFrame({"unit": units, "value": values})
    spans = frame.groupby("unit")["value"].agg(["min", "max"])
    constant = bool((spans["min"] == spans["max"]).all())
    return "between_units" if constant else "within_units"


def _arrangements(
    values: np.ndarray,
    units: np.ndarray,
    scheme: PermutationScheme,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, bool]:
    """Null arrangements of ``values`` under ``scheme``, one per row.

    Returns ``(arrangements, n_distinct, exact)``: every distinct arrangement when there
    are at most ``n_permutations`` of them (the observed one among them), else
    ``n_permutations`` seeded random draws.
    """
    _, first, inverse = np.unique(units, return_index=True, return_inverse=True)
    if scheme == "between_units":
        unit_values = values[first]
        n_distinct = _n_distinct(unit_values)
        if n_distinct <= n_permutations:
            per_unit = _distinct_permutations(unit_values)
            return per_unit[:, inverse], n_distinct, True
        draws = rng.permuted(np.tile(unit_values, (n_permutations, 1)), axis=1)
        return draws[:, inverse], n_distinct, False

    members = [np.flatnonzero(inverse == k) for k in range(first.size)]
    n_distinct = math.prod(_n_distinct(values[m]) for m in members)
    if n_distinct <= n_permutations:
        blocks = [(m, _distinct_permutations(values[m])) for m in members]
        out = np.tile(values, (n_distinct, 1))
        choices = itertools.product(*(range(b.shape[0]) for _, b in blocks))
        for row, choice in enumerate(choices):
            for (m, block), pick in zip(blocks, choice, strict=True):
                out[row, m] = block[pick]
        return out, n_distinct, True
    out = np.tile(values, (n_permutations, 1))
    for m in members:
        if m.size > 1:
            out[:, m] = rng.permuted(out[:, m], axis=1)
    return out, n_distinct, False


def _kruskal_h(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Tie-corrected Kruskal-Wallis H for each row of group ``codes`` (vectorized).

    Equals :func:`scipy.stats.kruskal`'s statistic; for two groups it is the square of
    the standardized two-sided Mann-Whitney U. NaN when every value is tied.
    """
    n = values.size
    ranks = np.asarray(stats.rankdata(values), dtype=float)
    _, ties = np.unique(values, return_counts=True)
    tie_factor = 1.0 - float((ties.astype(float) ** 3 - ties).sum()) / (n**3 - n)
    if tie_factor <= 0:
        return np.full(codes.shape[0], np.nan)
    total = np.zeros(codes.shape[0])
    for g in range(n_groups):
        in_group = codes == g
        size = in_group.sum(axis=1)
        rank_sum = in_group.astype(float) @ ranks
        total += np.divide(
            rank_sum**2, size, out=np.zeros_like(rank_sum), where=size > 0
        )
    h: np.ndarray = (12.0 / (n * (n + 1)) * total - 3.0 * (n + 1)) / tie_factor
    return h


def _abs_pearson(x: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """``|r|`` of ``x`` with each row of ``ys`` (NaN where either is constant)."""
    xc = x - x.mean()
    yc = ys - ys.mean(axis=1, keepdims=True)
    denom = float(np.linalg.norm(xc)) * np.linalg.norm(yc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r: np.ndarray = np.abs((yc @ xc) / denom)
    return r


def _permutation_p(observed: float, null: np.ndarray, *, exact: bool) -> float:
    """Upper-tail p of ``observed`` against the arrangement statistics ``null``."""
    if not np.isfinite(observed):
        return float("nan")
    hits = int(np.sum(null >= observed - 1e-9 * max(1.0, abs(observed))))
    if exact:
        return hits / null.size  # the observed arrangement is among the rows
    return (hits + 1) / (null.size + 1)


def _warn_resolution(n_distinct: int) -> None:
    if n_distinct <= _MAX_UNRESOLVED_ARRANGEMENTS:
        warnings.warn(
            f"The unit permutation test has only {n_distinct} distinct arrangement(s), "
            f"so its smallest attainable p is {1.0 / n_distinct:.3g} — it cannot go "
            f"below 0.05. Too few units vary for the p-values to separate anything.",
            PermutationResolutionWarning,
            stacklevel=4,
        )


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
    nesting: _Nesting,
) -> MarginalTests:
    """Scatter both panels by group + per-group KDE marginals + per-PC group-test p."""
    unique_labels = [str(v) for v in np.unique(labels)]
    str_labels = labels.astype(str)

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

    # Evaluate each KDE across the scatter's whole view (not just the data range), so a
    # tail runs to the frame instead of stopping in mid-air at the extreme sample; then
    # pin the view so the curves cannot widen it.
    limits = _view_limits(panel)
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
            grid = np.linspace(limits[pc][0], limits[pc][1], 200)
            density = np.asarray(stats.gaussian_kde(pc_vals)(grid), dtype=float)
            if vertical:
                ax.plot(density, grid, color=color_map[label], linewidth=3)
            else:
                ax.plot(grid, density, color=color_map[label], linewidth=3)
    _freeze_view(panel, limits)

    tested = [lb for lb in unique_labels if lb not in background]
    tests = _group_tests(scores, str_labels, tested, nesting)
    if show_marginal_stats:
        method = _method_text(tests)
        for title_ax, pcs in ((panel.ax1_title, (0, 1)), (panel.ax2_title, (2, 3))):
            parts = [f"PC{pc + 1} {_p_text(tests.p_values[pc], tests)}" for pc in pcs]
            _write_panel_stats(title_ax, "   ·   ".join(parts), method)
    return tests


def _group_tests(
    scores: np.ndarray,
    str_labels: np.ndarray,
    tested_labels: list[str],
    nesting: _Nesting,
) -> MarginalTests:
    """Per-PC test that the ``tested_labels`` groups differ along that PC.

    Two groups: two-sided Mann-Whitney U; three or more: Kruskal-Wallis. Samples whose
    label is not in ``tested_labels`` (the greyed reference samples) take no part.
    Without a unit the p is the asymptotic one (samples independent); with a unit it is
    the unit permutation p (module docstring). Fewer than two groups, or a degenerate
    PC, gives NaN (rendered ``n/a``).
    """
    test = "Mann-Whitney U" if len(tested_labels) == 2 else "Kruskal-Wallis"
    mask = np.isin(str_labels, tested_labels)
    labels = str_labels[mask]
    units = nesting.units_for(mask)
    n_units = None if units is None else int(np.unique(units).size)
    if len(tested_labels) < 2:
        return MarginalTests(
            test=test,
            scheme="independent" if units is None else "between_units",
            unit=nesting.unit,
            n_samples=int(mask.sum()),
            n_units=n_units,
            n_arrangements=None,
            exact=False,
            n_draws=None,
            p_values=(float("nan"),) * _N_COMPONENTS,
            r_values=None,
        )

    if units is None:
        p_values: list[float] = []
        for pc in range(_N_COMPONENTS):
            groups = [scores[mask, pc][labels == label] for label in tested_labels]
            try:
                if len(groups) == 2:
                    p_value = float(stats.mannwhitneyu(groups[0], groups[1]).pvalue)
                else:
                    p_value = float(stats.kruskal(*groups).pvalue)
            except ValueError:
                # Degenerate (all values identical): older SciPy raises, newer NaN.
                p_value = float("nan")
            p_values.append(p_value)
        return MarginalTests(
            test=test,
            scheme="independent",
            unit=None,
            n_samples=int(mask.sum()),
            n_units=None,
            n_arrangements=None,
            exact=False,
            n_draws=None,
            p_values=tuple(p_values),
            r_values=None,
        )

    codes = np.asarray([tested_labels.index(lb) for lb in labels], dtype=np.int64)
    scheme = _permutation_scheme(codes, units)
    tested_scores = scores[mask]
    if scheme == "within_units":
        # Only units where the label varies carry within-unit information.
        keep = _varying_units(codes, units)
        codes, units, tested_scores = codes[keep], units[keep], tested_scores[keep]
    arrangements, n_distinct, exact = _arrangements(
        codes, units, scheme, nesting.n_permutations, nesting.rng
    )
    _warn_resolution(n_distinct)
    p_values = []
    for pc in range(_N_COMPONENTS):
        vals = tested_scores[:, pc]
        if scheme == "within_units":
            vals = _within_deviations(vals, units)
        observed = float(_kruskal_h(vals, codes[None, :], len(tested_labels))[0])
        null = _kruskal_h(vals, arrangements, len(tested_labels))
        p_values.append(_permutation_p(observed, null, exact=exact))
    return MarginalTests(
        test=test,
        scheme=scheme,
        unit=nesting.unit,
        n_samples=int(codes.size),
        n_units=int(np.unique(units).size),
        n_arrangements=n_distinct,
        exact=exact,
        n_draws=int(arrangements.shape[0]),
        p_values=tuple(p_values),
        r_values=None,
    )


# --------------------------------------------------------------------------- #
# Continuous
# --------------------------------------------------------------------------- #


def _plot_continuous(
    scores: np.ndarray,
    values: np.ndarray,
    panel: _Panel,
    colormap: str,
    show_marginal_stats: bool,
    nesting: _Nesting,
) -> MarginalTests:
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

    tests = _correlation_tests(scores, values, nesting)
    units = nesting.units_for(np.ones(values.size, dtype=bool))
    nested = units is not None and np.unique(units).size < units.size
    cmap_obj = plt.get_cmap(colormap)
    marginals = (
        (panel.ax1_top, 0, False),
        (panel.ax1_right, 1, True),
        (panel.ax2_top, 2, False),
        (panel.ax2_right, 3, True),
    )
    std_floor = _degenerate_std_floor(scores)
    stat_parts: list[str] = []
    for ax, pc, vertical in marginals:
        if float(scores[:, pc].std()) <= std_floor:
            # Degenerate (~0-variance) PC: regressing on numerical noise gives an
            # exploding, meaningless fit — skip the marginal rather than draw it.
            stat_parts.append(f"PC{pc + 1} n/a")
            continue
        _plot_regression(
            scores[:, pc],
            values,
            ax,
            cmap_obj,
            vertical=vertical,
            units=units if nested else None,
            rng=nesting.rng,
            n_boot=nesting.n_permutations,
        )
        assert tests.r_values is not None
        r_value = tests.r_values[pc]
        r_text = f"{r_value:.2f}" if np.isfinite(r_value) else "n/a"
        stat_parts.append(
            f"PC{pc + 1} r = {r_text}, {_p_text(tests.p_values[pc], tests)}"
        )

    if show_marginal_stats:
        method = _method_text(tests)
        for title_ax, idx in ((panel.ax1_title, 0), (panel.ax2_title, 2)):
            _write_panel_stats(
                title_ax, "   ·   ".join(stat_parts[idx : idx + 2]), method
            )
    return tests


def _correlation_tests(
    scores: np.ndarray, values: np.ndarray, nesting: _Nesting
) -> MarginalTests:
    """Per-PC Pearson r of ``values`` with the PC, and its p (asymptotic or unit
    permutation). A degenerate (~0-variance) PC gets NaN r and p."""
    std_floor = _degenerate_std_floor(scores)
    live = [float(scores[:, pc].std()) > std_floor for pc in range(_N_COMPONENTS)]
    units = nesting.units_for(np.ones(values.size, dtype=bool))
    r_values: list[float] = []
    p_values: list[float] = []
    if units is None:
        for pc in range(_N_COMPONENTS):
            if not live[pc]:
                r_values.append(float("nan"))
                p_values.append(float("nan"))
                continue
            lr = stats.linregress(scores[:, pc], values)
            r_values.append(float(lr.rvalue))
            p_values.append(float(lr.pvalue))
        return MarginalTests(
            test="Pearson",
            scheme="independent",
            unit=None,
            n_samples=int(values.size),
            n_units=None,
            n_arrangements=None,
            exact=False,
            n_draws=None,
            p_values=tuple(p_values),
            r_values=tuple(r_values),
        )

    scheme = _permutation_scheme(values, units)
    y, x_all = values, scores
    if scheme == "within_units":
        # Within-unit deviations; units where the variable is constant drop out.
        keep = _varying_units(values, units)
        units, x_all = units[keep], scores[keep]
        y = _within_deviations(values[keep], units)
    arrangements, n_distinct, exact = _arrangements(
        y, units, scheme, nesting.n_permutations, nesting.rng
    )
    _warn_resolution(n_distinct)
    for pc in range(_N_COMPONENTS):
        if not live[pc]:
            r_values.append(float("nan"))
            p_values.append(float("nan"))
            continue
        x = x_all[:, pc]
        if scheme == "within_units":
            x = _within_deviations(x, units)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = float(np.corrcoef(x, y)[0, 1])
        r_values.append(r)
        null = _abs_pearson(x, arrangements)
        p_values.append(_permutation_p(abs(r), null, exact=exact))
    return MarginalTests(
        test="Pearson",
        scheme=scheme,
        unit=nesting.unit,
        n_samples=int(y.size),
        n_units=int(np.unique(units).size),
        n_arrangements=n_distinct,
        exact=exact,
        n_draws=int(arrangements.shape[0]),
        p_values=tuple(p_values),
        r_values=tuple(r_values),
    )


def _cluster_bootstrap_band(
    x: np.ndarray,
    y: np.ndarray,
    units: np.ndarray,
    x_pred: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """95% band of the OLS line from resampling whole units (2.5/97.5 percentiles).

    Resampling units, not samples, keeps each unit's samples together, so the band
    carries their shared variation. ``None`` when too few resamples are fittable.
    """
    _, inverse = np.unique(units, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(int(inverse.max()) + 1)]
    fitted: list[np.ndarray] = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        xs, ys = x[idx], y[idx]
        sxx = float(np.sum((xs - xs.mean()) ** 2))
        if sxx <= 0:
            continue
        slope = float(np.sum((xs - xs.mean()) * (ys - ys.mean()))) / sxx
        fitted.append(ys.mean() + slope * (x_pred - xs.mean()))
    if len(fitted) < 2:
        return None
    lo, hi = np.percentile(np.asarray(fitted), [2.5, 97.5], axis=0)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _plot_regression(
    pc_vals: np.ndarray,
    color_vals: np.ndarray,
    ax: Axes,
    cmap: Colormap,
    *,
    vertical: bool,
    units: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    n_boot: int = 9999,
) -> None:
    """Scatter (PC value vs colored variable) + OLS line with a 95% band on a marginal.

    Without ``units`` the band is the 95% **mean-response** confidence interval, which
    uses the *residual* standard error ``s = sqrt(SSE/(n-2))`` and the Student-t
    quantile ``t_{0.975, n-2}`` — not ``linregress``'s ``stderr`` (the *slope* SE,
    ``s / sqrt(Sxx)``) with ``1.96``, which understates the band by ``~1/sqrt(Sxx)``.
    With ``units`` (samples nested in units) the t band would assume independence, so
    the band is a cluster bootstrap over whole units instead.
    """
    lr = stats.linregress(pc_vals, color_vals)
    slope = float(lr.slope)
    intercept = float(lr.intercept)

    x_pred = np.linspace(float(pc_vals.min()), float(pc_vals.max()), 100)
    y_pred = intercept + slope * x_pred
    lower, upper = y_pred, y_pred
    boot = None
    if units is not None:
        boot = _cluster_bootstrap_band(
            pc_vals,
            color_vals,
            units,
            x_pred,
            rng if rng is not None else np.random.default_rng(0),
            n_boot,
        )
    if boot is not None:
        lower, upper = boot
    else:
        n = pc_vals.shape[0]
        dof = n - 2
        denom = float(np.sum((pc_vals - pc_vals.mean()) ** 2))
        if units is None and dof > 0 and denom > 0:
            residuals = color_vals - (intercept + slope * pc_vals)
            resid_se = float(np.sqrt(float(np.sum(residuals**2)) / dof))
            tcrit = float(stats.t.ppf(0.975, dof))
            mean_se = resid_se * np.sqrt(
                1.0 / n + (x_pred - pc_vals.mean()) ** 2 / denom
            )
            lower, upper = y_pred - tcrit * mean_se, y_pred + tcrit * mean_se
    point_colors = cmap(_normalize_for_cmap(color_vals))

    if not vertical:
        ax.scatter(
            pc_vals, color_vals, s=20, alpha=0.7, c=point_colors, edgecolor="none"
        )
        ax.plot(x_pred, y_pred, color="gray", linewidth=3)
        ax.fill_between(x_pred, lower, upper, alpha=0.2, color="gray")
    else:
        ax.scatter(
            color_vals, pc_vals, s=20, alpha=0.7, c=point_colors, edgecolor="none"
        )
        ax.plot(y_pred, x_pred, color="gray", linewidth=3)
        ax.fill_betweenx(x_pred, lower, upper, alpha=0.2, color="gray")


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
    """Coerce ``values`` to a list, treating a bare string as a single value.

    Any other iterable (list, tuple, set, numpy array, pandas Series/Index, generator)
    is unpacked — the same rule as :mod:`figures.colors`, so a value the registry greys
    is also a value the plot treats as background.
    """
    if isinstance(values, str):
        return [values]
    if isinstance(values, Iterable):
        return list(values)
    return [values]
