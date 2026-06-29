"""Data-completeness / missingness diagnostic figures for a :class:`Dataset`.

TEMPLATE (lib/) — a *seed* for a project's missingness module, not a finished script.
Copy it into the project's ``scripts/`` and adapt the call sites per study. Held to the
correctness charter (conventions/correctness.md): **assume nothing, verify everything,
fail loud.**

What it draws (one figure, two panels):

* **Completeness curve** (left) — features retained vs the required *detection
  fraction*: for a threshold *t*, how many features are detected in at least a fraction
  *t* of the samples. Read it as the **feature-filter decision**: "require presence in
  >= t of the samples -> keep N features." Optionally **overlaid per sample class**
  (``color_by``): experimental vs pooled-QC / reference samples — pools should sit
  higher (more complete), a real QC read. *Caveat:* control classes usually have few
  samples, so their curves are coarse (a "detected in all 3 pools" bar is weak); the
  experimental curve is the decision-relevant one. Class sample counts are shown in the
  legend so the coarseness is visible.
* **MNAR diagnostic** (right) — per-feature **detection rate vs mean log2 abundance** of
  its detected values, as a hexbin (feature density) with a binned-median trend and the
  Pearson correlation annotated. A downward-left pile (low-abundance features detected
  less) is **MNAR / left-censoring** — missingness is *not* at random, which argues for
  a left-censored imputation (MinProb / QRILC) over mean/median/KNN. A flat trend is
  closer to MCAR. This is the view that should drive the imputation decision and the one
  the identification-depth / abundance plots cannot show.

This is the per-**feature** complement to the identification-depth bar chart
(`lib/figures/id-depth`), which owns the per-**sample** view (detected features per
run); ``sample_detection_rate`` is provided in the result for completeness but is
deliberately not drawn here (it is ``1 - id-depth's count / n_features``).

Detection: a feature is *detected* in a sample when its value is finite and strictly
greater than ``min_intensity`` (default ``0.0``; in DIA a ``0`` is "not detected") — the
**same predicate as `lib/figures/id-depth`** (kept as an independent one-liner so this
template stays a self-contained seed). The MNAR abundance axis is the mean log2 of the
*positive* detected values per feature.

Scale (a HARD refuse, like the CV / id-depth templates): completeness and missingness
are properties of the **raw linear matrix** — a ``0`` means "not detected" only on the
unnormalized intensity scale; on a centered/standardized scale a ``0`` is an ordinary
value and ``> 0`` no longer means "detected". So :func:`compute_completeness` **raises**
:class:`MissingnessScaleError` on a non-linear ``Dataset.scale``. Run it on the raw
data, before normalization — a first-look QC that informs the Stage-2 imputation choice.

Convention wiring (conventions/visualization.md): the per-class curve colors come from
the project color registry (:mod:`figures.colors`, capped at eight); the hexbin density
colorbar **stays on the figure** (it is the data's own count scale, like the
sample-correlation r-colorbar), while the sample-class key is rendered as a **separate
legend image** (``<base>.legend.{svg,png}``) via :func:`figures.figure_io.save_figure`.

**Control samples are shown with the experimental samples here** (as overlaid,
class-colored completeness curves) — the same labeled exception as the
sample-correlation and id-depth templates, because the cross-class comparison (are pools
more complete?) is the deliverable. This template makes NO study decisions: the
detection threshold, which
column (if any) splits the curves, and the registry namespace are the caller's choice.
Sample exclusions and relabelings live in the project copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common.data_loading import Dataset
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "missingness", "version": "0.1"},
    "kind": "module",
    "provides": [
        "MissingnessScaleError",
        "CompletenessResult",
        "MissingnessPlot",
        "compute_completeness",
        "plot_missingness",
        "save_missingness",
    ],
    "uses": ["common.data_loading", "figures.colors", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Data-completeness / missingness diagnostics from a Dataset: a two-panel "
        "figure of (1) a feature completeness curve (features retained vs required "
        "detection fraction, optionally overlaid per sample class via the registry + "
        ">8 guard) and (2) an MNAR diagnostic (per-feature detection rate vs mean log2 "
        "abundance, "
        "hexbin + binned-median trend + correlation). Detected = finite and "
        "> min_intensity; hard-refuses a non-linear scale (a raw-linear property). "
        "Dual-export plus a separate legend image. Uses common.data_loading, "
        "figures.colors, figures.figure_io. Study-agnostic; fail-loud."
    ),
}

# Curve color when no ``color_by`` is given (a single whole-cohort completeness curve):
# a neutral steel, outside the categorical palette so it consumes no registry slot.
DEFAULT_CURVE_COLOR = "#0072B2"

# Hexbin colormap for the MNAR panel (perceptually uniform; density on a log scale).
_HEXBIN_CMAP = "viridis"

# Trend-line color for the MNAR binned median (Okabe-Ito vermillion, off the hexbin).
_TREND_COLOR = "#D55E00"

# Number of abundance bins for the MNAR binned-median trend.
_TREND_BINS = 20


class MissingnessScaleError(ValueError):
    """Raised when completeness/missingness is requested on a non-linear scale.

    Completeness is a property of the **raw linear matrix**: a feature is detected when
    its intensity is finite and ``> min_intensity`` (a ``0`` is "not detected"). On a
    log or centered scale (``log2``/``glog2``/``zscore``, and the ``"mad"``/``"vsn"``
    normalizer output) a ``0`` is an ordinary value and ``> 0`` no longer means
    "detected", so the rates are meaningless. Run it on the unnormalized data, before
    any transform (it is a first-look QC that informs the imputation choice).
    """


@dataclass(frozen=True)
class CompletenessResult:
    """The per-feature / per-sample detection summaries underlying a missingness figure.

    Attributes
    ----------
    sample_ids:
        ``(n_samples,)`` the sample identifiers (row order of the Dataset).
    feature_detection_rate:
        ``(n_features,)`` fraction of all samples in which each feature is detected (the
        completeness-curve and MNAR y-axis source).
    sample_detection_rate:
        ``(n_samples,)`` fraction of features each sample detects (``1 -`` per-sample
        missingness). Provided for completeness; not drawn (the id-depth template owns
        the per-sample view).
    feature_mean_log_abundance:
        ``(n_features,)`` mean log2 abundance of each feature's *positive detected*
        values — the MNAR x-axis. ``NaN`` for a feature never detected (no positive
        value).
    mnar_correlation:
        Pearson correlation between ``feature_mean_log_abundance`` and
        ``feature_detection_rate`` over the features where both are finite. Positive
        means low-abundance features are detected less (MNAR / left-censoring). ``NaN``
        if there are fewer than two such features or no spread.
    """

    sample_ids: np.ndarray
    feature_detection_rate: np.ndarray
    sample_detection_rate: np.ndarray
    feature_mean_log_abundance: np.ndarray
    mnar_correlation: float


@dataclass
class MissingnessPlot:
    """A rendered missingness figure plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (completeness curve + MNAR hexbin; the density
        colorbar is on the figure, no baked sample-class legend).
    legend_figure:
        A standalone swatch legend of the sample classes when ``color_by`` was set
        (saved beside the main figure as ``<base>.legend.{svg,png}`` by
        :func:`save_missingness`); ``None`` for a single whole-cohort completeness line.
    result:
        The underlying :class:`CompletenessResult`.
    color_map:
        ``{class: hex}`` drawn for ``color_by`` (empty when uncolored).
    """

    figure: Figure
    legend_figure: Figure | None
    result: CompletenessResult
    color_map: dict[str, str]


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #


def compute_completeness(
    dataset: Dataset, *, min_intensity: float = 0.0
) -> CompletenessResult:
    """Per-feature / per-sample detection rates + the MNAR correlation.

    Detection is a property of the **raw linear matrix**, so this raises
    :class:`MissingnessScaleError` unless ``dataset.scale == "linear"`` (see the module
    docstring). A ``0`` (or any value ``<= min_intensity``) and a ``NaN`` both count as
    not detected.

    Raises :class:`MissingnessScaleError` on a non-linear scale, or ``ValueError`` on a
    non-2D / empty / non-row-aligned Dataset.
    """
    mask, abundances = _detection_mask(dataset, min_intensity)
    sample_ids = _sample_ids(dataset, mask.shape[0])
    return _build_result(mask, abundances, sample_ids)


def _detection_mask(
    dataset: Dataset, min_intensity: float
) -> tuple[np.ndarray, np.ndarray]:
    """The boolean (n_samples, n_features) detected mask + the float abundances.

    Centralizes the scale refuse + shape guards. Detected == finite and
    ``> min_intensity`` (the same predicate as the id-depth template).
    """
    if dataset.scale != "linear":
        raise MissingnessScaleError(
            f"Completeness/missingness requires linear-scale abundances but the "
            f"Dataset is on scale {dataset.scale!r}. On log/centered scales a 0 is an "
            f"ordinary value, so '> {min_intensity}' no longer means 'detected'. Run "
            f"it on the unnormalized data, before any transform."
        )
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2:
        raise ValueError(
            f"abundances must be 2D (n_samples, n_features); got {abundances.shape}."
        )
    n_samples, n_features = abundances.shape
    if n_samples < 1 or n_features < 1:
        raise ValueError(
            f"Dataset is empty ({n_samples} samples x {n_features} features); need "
            f">= 1 of each."
        )
    mask = np.isfinite(abundances) & (abundances > min_intensity)
    return mask, abundances


def _sample_ids(dataset: Dataset, n_samples: int) -> np.ndarray:
    """The row-aligned sample ids (fail loud if metadata is not row-aligned)."""
    ids = np.asarray(dataset.metadata.index.to_numpy(), dtype=str)
    if ids.shape[0] != n_samples:
        raise ValueError(
            f"metadata has {ids.shape[0]} rows but {n_samples} abundance samples; it "
            f"is not row-aligned."
        )
    return ids


def _build_result(
    mask: np.ndarray, abundances: np.ndarray, sample_ids: np.ndarray
) -> CompletenessResult:
    """Assemble the detection rates + MNAR correlation from the detected mask."""
    feature_detection_rate = np.asarray(mask.mean(axis=0), dtype=float)
    sample_detection_rate = np.asarray(mask.mean(axis=1), dtype=float)
    mean_log_ab = _feature_mean_log_abundance(mask, abundances)
    mnar = _mnar_correlation(mean_log_ab, feature_detection_rate)
    return CompletenessResult(
        sample_ids=sample_ids,
        feature_detection_rate=feature_detection_rate,
        sample_detection_rate=sample_detection_rate,
        feature_mean_log_abundance=mean_log_ab,
        mnar_correlation=mnar,
    )


def _feature_mean_log_abundance(mask: np.ndarray, abundances: np.ndarray) -> np.ndarray:
    """Mean log2 of each feature's positive detected values (``NaN`` if it has none)."""
    log_mask = mask & (abundances > 0.0)  # log2 is only defined on positive intensities
    counts = log_mask.sum(axis=0).astype(float)
    safe = np.where(
        log_mask, abundances, 1.0
    )  # 1.0 placeholder -> log2 == 0, then zeroed
    log_sum = np.where(log_mask, np.log2(safe), 0.0).sum(axis=0)
    out = np.full(abundances.shape[1], np.nan, dtype=float)
    np.divide(log_sum, counts, out=out, where=counts > 0)
    return out


def _mnar_correlation(mean_log_ab: np.ndarray, detection_rate: np.ndarray) -> float:
    """Pearson r between mean log abundance and detection rate (NaN if undefined)."""
    ok = np.isfinite(mean_log_ab) & np.isfinite(detection_rate)
    if int(ok.sum()) < 2:
        return float("nan")
    a = mean_log_ab[ok]
    b = detection_rate[ok]
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def plot_missingness(
    dataset: Dataset,
    *,
    color_by: str | None = None,
    min_intensity: float = 0.0,
    title: str | None = None,
    legend_title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
) -> MissingnessPlot:
    """Draw the two-panel completeness + MNAR diagnostic for one Dataset.

    Parameters
    ----------
    dataset:
        A single linear-scale :class:`Dataset` (raised otherwise). Call once per feature
        level (e.g. protein, then precursor) — unlike the cv/abundance templates this is
        not a multi-state comparison (normalization does not change presence/absence).
    color_by:
        Metadata column whose value splits the completeness curve into one class-colored
        line each, through the project color registry (capped at eight). ``None``
        (default) draws a single whole-cohort curve and no legend. Missing values are
        refused.
    min_intensity:
        Detection threshold; a feature counts as detected when finite and
        ``> min_intensity`` (default ``0.0``).
    title:
        Optional figure suptitle.
    legend_title:
        Title for the companion legend figure; defaults to ``color_by``.
    registry_path:
        Color registry JSON. Default ``state/color_registry.json``.
    persist_colors:
        Write newly assigned categorical colors back to the registry (default ``True``).

    Returns
    -------
    MissingnessPlot
        The main figure, the companion legend figure (``None`` when ``color_by`` is
        ``None``), the :class:`CompletenessResult`, and the ``{class: hex}`` color map.

    Raises
    ------
    MissingnessScaleError
        If the Dataset is not on the ``"linear"`` scale.
    ValueError
        On a non-2D / empty / non-row-aligned Dataset, or a ``color_by`` column that is
        missing or has missing values.
    CategoricalPaletteExceededError
        If ``color_by`` exceeds the palette's categorical capacity (the >8 guard).
    """
    # Validate + compute BEFORE creating the figure, so the scale / column guards (the
    # likely failures) raise with no figure to leak. The only post-figure raise is the
    # registry's >8-category guard, handled by the try/except below.
    mask, abundances = _detection_mask(dataset, min_intensity)
    sample_ids = _sample_ids(dataset, mask.shape[0])
    result = _build_result(mask, abundances, sample_ids)
    classes = _resolve_color_by(dataset, color_by)
    title_for_legend = legend_title if legend_title is not None else (color_by or "")

    color_map: dict[str, str] = {}
    legend_figure: Figure | None = None
    with publication_style():
        fig, (ax_curve, ax_mnar) = plt.subplots(
            1, 2, figsize=(15, 6), constrained_layout=True
        )
        try:
            if classes is not None:
                ordered = _ordered_unique(classes)
                color_map = assign_colors(
                    str(color_by),
                    ordered,
                    registry_path=registry_path,
                    persist=persist_colors,
                )
                _draw_completeness_by_class(ax_curve, mask, classes, ordered, color_map)
                legend_figure = _legend_figure(
                    ordered, classes, color_map, title_for_legend
                )
            else:
                _draw_completeness_single(ax_curve, mask)

            _draw_mnar(ax_mnar, fig, result)

            if title is not None:
                fig.suptitle(title, fontsize=15, weight="bold")
        except BaseException:
            plt.close(fig)
            raise

    return MissingnessPlot(
        figure=fig,
        legend_figure=legend_figure,
        result=result,
        color_map=color_map,
    )


def save_missingness(
    dataset: Dataset,
    output_dir: str | Path,
    base_name: str,
    *,
    color_by: str | None = None,
    min_intensity: float = 0.0,
    title: str | None = None,
    legend_title: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = True,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render a missingness figure (:func:`plot_missingness`) and save it.

    Writes the main figure as ``<base>.{svg,png}`` and, when ``color_by`` is set, its
    companion legend as ``<base>.legend.{svg,png}`` via
    :func:`figures.figure_io.save_figure`, and returns their paths. Both figures are
    closed even if saving fails (no leak on a bad ``base_name``/``dpi``).
    """
    plot = plot_missingness(
        dataset,
        color_by=color_by,
        min_intensity=min_intensity,
        title=title,
        legend_title=legend_title,
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
# Validation
# --------------------------------------------------------------------------- #


def _resolve_color_by(dataset: Dataset, color_by: str | None) -> np.ndarray | None:
    """Validate + extract ``color_by`` as per-sample strings (or ``None``)."""
    if color_by is None:
        return None
    if color_by not in dataset.metadata.columns:
        raise ValueError(
            f"color_by {color_by!r} is not a metadata column "
            f"{list(dataset.metadata.columns)}."
        )
    series = dataset.metadata[color_by]
    if bool(series.isna().any()):
        n_bad = int(series.isna().sum())
        raise ValueError(
            f"color_by {color_by!r} has {n_bad} missing value(s); a missing label "
            f"cannot be colored. Resolve or relabel them first."
        )
    return np.asarray(series.to_numpy(), dtype=str)


def _ordered_unique(values: np.ndarray) -> list[str]:
    """Distinct values in first-seen order (stable, study-agnostic)."""
    seen: list[str] = []
    for value in values.astype(str):
        if value not in seen:
            seen.append(value)
    return seen


# --------------------------------------------------------------------------- #
# Drawing — completeness curve
# --------------------------------------------------------------------------- #


def _completeness_curve(counts: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """(x, y): features detected in >= k of m samples, for k=1..m, x = k/m."""
    ks = np.arange(1, m + 1)
    retained = np.array([int((counts >= k).sum()) for k in ks], dtype=int)
    return ks / m, retained


def _draw_completeness_by_class(
    ax: Axes,
    mask: np.ndarray,
    classes: np.ndarray,
    ordered: list[str],
    color_map: dict[str, str],
) -> None:
    """One completeness curve per sample class, colored from the registry."""
    n_features = mask.shape[1]
    minima: list[int] = []
    for cls in ordered:
        members = classes == cls
        m = int(members.sum())
        if m < 1:
            continue
        counts = mask[members].sum(axis=0)
        x, retained = _completeness_curve(counts, m)
        ax.step(
            x,
            retained,
            where="post",
            color=color_map[cls],
            linewidth=2.2,
            label=f"{cls} (n={m})",
        )
        minima.append(int(retained.min()))
    _finish_completeness_axis(ax, n_features, minima)


def _draw_completeness_single(ax: Axes, mask: np.ndarray) -> None:
    """A single whole-cohort completeness curve (no class split)."""
    n_samples, n_features = mask.shape
    counts = mask.sum(axis=0)
    x, retained = _completeness_curve(counts, n_samples)
    ax.step(x, retained, where="post", color=DEFAULT_CURVE_COLOR, linewidth=2.4)
    _finish_completeness_axis(ax, n_features, [int(retained.min())])


def _finish_completeness_axis(ax: Axes, n_features: int, minima: list[int]) -> None:
    """Axis labels, the all-features reference line, and a drop-revealing y-limit."""
    ax.axhline(n_features, color="#999999", linestyle="--", linewidth=0.8)
    ax.text(
        0.01,
        n_features,
        f"  all {n_features} features",
        va="bottom",
        ha="left",
        fontsize=8,
        color="#666666",
    )
    # Lower bound at the steepest curve's minimum so its drop is visible (the clip fix).
    low = min(minima) if minima else 0
    pad = max(1.0, 0.03 * (n_features - low))
    ax.set_ylim(low - pad, n_features + 0.02 * n_features)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("required detection fraction (≥ this fraction of the class)")
    ax.set_ylabel("features retained")
    ax.set_title("Completeness curve", loc="left", fontsize=13, weight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)


# --------------------------------------------------------------------------- #
# Drawing — MNAR diagnostic
# --------------------------------------------------------------------------- #


def _draw_mnar(ax: Axes, fig: Figure, result: CompletenessResult) -> None:
    """Hexbin of detection rate vs mean log2 abundance + binned-median trend + r."""
    mean_log_ab = result.feature_mean_log_abundance
    rate = result.feature_detection_rate
    ok = np.isfinite(mean_log_ab) & np.isfinite(rate)
    x = mean_log_ab[ok]
    y = rate[ok]

    hexbin = ax.hexbin(x, y, gridsize=40, cmap=_HEXBIN_CMAP, bins="log", mincnt=1)
    cbar = fig.colorbar(hexbin, ax=ax)
    cbar.set_label("features per bin (log)")

    centers, medians = _binned_median(x, y, _TREND_BINS)
    if centers.size:
        ax.plot(
            centers,
            medians,
            color=_TREND_COLOR,
            linewidth=2.4,
            marker="o",
            markersize=4,
            label="binned median",
        )
        ax.legend(loc="lower right", fontsize=10, frameon=True)

    annotation = "MNAR signal undefined (no spread)"
    if np.isfinite(result.mnar_correlation):
        annotation = (
            f"Pearson r = {result.mnar_correlation:.2f}\n"
            f"(low abundance → more missing = MNAR)"
        )
    ax.text(
        0.03,
        0.05,
        annotation,
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("mean log2 abundance of detected values")
    ax.set_ylabel("detection rate (fraction of all samples)")
    ax.set_title("MNAR diagnostic", loc="left", fontsize=13, weight="bold")


def _binned_median(
    x: np.ndarray, y: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Median ``y`` within ``n_bins`` percentile bins of ``x`` (skips empty bins)."""
    if x.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    edges = np.unique(np.percentile(x, np.linspace(0.0, 100.0, n_bins + 1)))
    centers: list[float] = []
    medians: list[float] = []
    for lo, hi in pairwise(edges):
        in_bin = (x >= lo) & (x <= hi)
        if bool(in_bin.any()):
            centers.append((lo + hi) / 2.0)
            medians.append(float(np.median(y[in_bin])))
    return np.asarray(centers, dtype=float), np.asarray(medians, dtype=float)


# --------------------------------------------------------------------------- #
# Legend figure (rendered separately so it never overlaps the plot)
# --------------------------------------------------------------------------- #


def _legend_figure(
    ordered: list[str],
    classes: np.ndarray,
    color_map: dict[str, str],
    legend_title: str,
) -> Figure:
    """Standalone swatch legend: one line per sample class, with its sample count."""
    counts = {cls: int((classes == cls).sum()) for cls in ordered}
    labels = [f"{cls} (n={counts[cls]})" for cls in ordered]
    height = max(1.4, 0.35 * len(ordered) + 0.8)
    fig, ax = plt.subplots(figsize=(3.4, height))
    ax.axis("off")
    handles = [Line2D([0], [0], color=color_map[cls], linewidth=2.5) for cls in ordered]
    ax.legend(
        handles,
        labels,
        title=legend_title or None,
        loc="center",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    return fig
