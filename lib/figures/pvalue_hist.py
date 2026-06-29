"""p-value histogram — the calibration diagnostic for a univariate test.

TEMPLATE (lib/) — a *seed* for a project's p-value-histogram module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call sites per study. Held
to the correctness charter (conventions/correctness.md): **assume nothing, verify
everything, fail loud.**

What it draws (one figure): the distribution of the **raw** per-feature p-values over
``[0, 1]`` (density), with the **uniform null** at ``1.0`` for reference. It is the
honesty check on a differential-abundance run — read the *shape*:

  * **uniform body + a spike near 0** — healthy: nulls are uniform, true effects pile up
    at small p. The flat body sits near the true-null fraction (Storey's π0).
  * **U-shape / a hump near 1** — anti-conservative *or* unmodeled structure: a missing
    covariate, batch confounding, or a wrong variance model. Investigate before trusting
    the q-values.
  * **a hill rising toward 1** — conservative: the test is over-correcting (e.g. ties in
    a rank test, an over-shrunk variance), so real effects may be missed.

Pass **raw p-values, not BH q-values** (q-values cluster on the BH step plateaus and
make the histogram meaningless); a value outside ``[0, 1]`` raises — catching the common
"passed q by mistake" slip. It overlays several distributions (methods, contrasts, or
the both-ways corrected/uncorrected robustness pair) via the project color registry, and
:func:`pvalue_histogram_from_result` is the bridge that reads a
:class:`~analysis.differential_abundance.DifferentialAbundanceResult`.

Convention wiring (conventions/visualization.md): the per-distribution colors come from
the Okabe-Ito palette via :mod:`figures.colors` (the >8-category guard applies), but —
unlike the other figure templates — they are **not persisted** to the registry by
default (``persist_colors=False``): a p-value histogram's labels are figure-local
(methods, batch-handling, real-vs-null), so each figure colors from the palette head
and nothing is written back, instead of independent figures accumulating colors in one
shared namespace and polluting the registry with non-biological labels. Pass
``persist_colors=True`` + a ``category`` to make a label keep one color across figures.
The legend is rendered as a **separate** figure saved as ``<base>.legend.{svg,png}`` via
:func:`figures.figure_io.save_figure` (the main figure stays clean); each legend entry
carries the distribution's feature count and estimated π0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis.differential_abundance import DifferentialAbundanceResult
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from figures.colors import DEFAULT_REGISTRY_PATH, assign_colors
from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "pvalue-hist", "version": "0.2"},
    "kind": "module",
    "provides": [
        "PValueHistogramResult",
        "PValueHistogramPlot",
        "estimate_pi0",
        "plot_pvalue_histogram",
        "pvalue_histogram_from_result",
        "save_pvalue_histogram",
    ],
    "uses": [
        "analysis.differential_abundance",
        "figures.colors",
        "figures.figure_io",
    ],
    "seeded_from": None,
    "description": (
        "p-value histogram, the calibration diagnostic for a univariate test: the raw "
        "per-feature p-values over [0, 1] (density) with the uniform null at 1.0 for "
        "reference. Reads the shape — uniform + spike (healthy), U-shape/hump "
        "(unmodeled structure), hill toward 1 (conservative). Overlays methods / "
        "contrasts / the both-ways pair colored from the Okabe-Ito palette (>8 guard); "
        "estimates Storey's pi0 per distribution (shown in the legend). Colors are "
        "ephemeral by default (persist_colors=False) since the labels are figure-local "
        "(so independent figures neither accumulate colors nor pollute the registry); "
        "opt into cross-figure consistency with persist_colors=True. Rejects "
        "values outside [0, 1] (the 'passed q by mistake' slip). Dual-export plus a "
        "separate legend image. Study-agnostic; fail-loud."
    ),
}

# Color-registry namespace for the overlaid distributions. Unlike the CV plot's
# NormalizationState (a fixed vocabulary that SHOULD stay consistent across figures),
# a p-value histogram's labels are figure-local (methods, batch-handling, real-vs-null).
# Colors are ephemeral by default (persist_colors=False): each figure colors from the
# palette head and nothing is written back. Pass persist_colors=True + a meaningful
# category to make a label keep one color across figures.
DEFAULT_PVALUE_CATEGORY = "PValueDistribution"

# Default Storey lambda for the pi0 (true-null fraction) estimate: pi0 = #{p > lam} /
# (m * (1 - lam)). lambda=0.5 is the canonical simple choice.
_DEFAULT_PI0_LAMBDA = 0.5


@dataclass(frozen=True)
class PValueHistogramResult:
    """Per-distribution summary underlying the figure.

    Attributes
    ----------
    counts:
        ``{label: number of finite p-values}`` (the histogrammed features).
    pi0:
        ``{label: estimated true-null fraction}`` (Storey, clamped to ``(0, 1]``);
        ``NaN`` if a label had no finite p-values.
    """

    counts: dict[str, int]
    pi0: dict[str, float]


@dataclass
class PValueHistogramPlot:
    """A rendered p-value histogram plus its companion legend figure.

    Attributes
    ----------
    figure:
        The main matplotlib figure (the overlaid histograms, no baked legend).
    legend_figure:
        A standalone legend (a swatch per distribution with its ``n`` and π0), saved
        beside the main figure as ``<base>.legend.{svg,png}`` by
        :func:`save_pvalue_histogram`.
    result:
        The underlying :class:`PValueHistogramResult`.
    color_map:
        ``{label: hex}`` drawn (one per distribution).
    """

    figure: Figure
    legend_figure: Figure
    result: PValueHistogramResult
    color_map: dict[str, str]


def estimate_pi0(pvalues: np.ndarray, lam: float = _DEFAULT_PI0_LAMBDA) -> float:
    """Storey's fixed-``lam`` true-null-fraction estimate, clamped to ``(0, 1]``.

    ``pi0 = #{p > lam} / (m * (1 - lam))`` over the ``m`` finite p-values. Returns
    ``NaN`` if there are none. An estimate exceeding 1 (sampling noise on a pure
    null) is clamped to 1.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1); got {lam}.")
    p = np.asarray(pvalues, dtype=float)
    finite = p[np.isfinite(p)]
    m = finite.size
    if m == 0:
        return float("nan")
    above = float(np.sum(finite > lam))
    pi0 = above / (m * (1.0 - lam))
    return float(min(pi0, 1.0))


def _coerce_mapping(
    pvalues: Mapping[str, np.ndarray] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Accept a single array (label ``"p-values"``) or an ordered ``{label: array}``."""
    if isinstance(pvalues, Mapping):
        if len(pvalues) == 0:
            raise ValueError("pvalues mapping is empty; pass >= 1 distribution.")
        return {str(k): np.asarray(v, dtype=float) for k, v in pvalues.items()}
    return {"p-values": np.asarray(pvalues, dtype=float)}


def _validate_pvalues(label: str, p: np.ndarray) -> np.ndarray:
    """Return the finite p-values, raising if any finite value is outside ``[0, 1]``."""
    if p.ndim != 1:
        raise ValueError(f"{label!r}: p-values must be 1D; got {p.ndim}D.")
    finite = p[np.isfinite(p)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError(
            f"{label!r}: p-values must lie in [0, 1]; got "
            f"[{finite.min():.4g}, {finite.max():.4g}]. Pass raw p-values, not BH "
            f"q-values or test statistics."
        )
    return np.asarray(finite, dtype=float)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_pvalue_histogram(
    pvalues: Mapping[str, np.ndarray] | np.ndarray,
    *,
    n_bins: int = 20,
    show_pi0: bool = True,
    pi0_lambda: float = _DEFAULT_PI0_LAMBDA,
    category: str = DEFAULT_PVALUE_CATEGORY,
    title: str | None = None,
    legend_title: str = "Distribution",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = False,
) -> PValueHistogramPlot:
    """Overlay one or more raw-p-value distributions as a calibration histogram.

    Parameters
    ----------
    pvalues:
        A single ``(n_features,)`` array (labelled ``"p-values"``) or an ordered
        ``{label: array}`` to overlay (methods, contrasts, or the both-ways pair). Each
        array's ``NaN`` (untestable features) are dropped; a finite value outside
        ``[0, 1]`` raises.
    n_bins:
        Number of equal-width bins over ``[0, 1]`` (default ``20`` → width ``0.05``).
    show_pi0:
        For a **single** distribution, draw a horizontal line at the estimated π0 (the
        null floor). π0 is always reported in the legend regardless. Default ``True``.
    pi0_lambda:
        Storey λ for the π0 estimate (default ``0.5``).
    category:
        Color-registry namespace for the labels. Only relevant when
        ``persist_colors=True``. Default ``"PValueDistribution"``.
    title:
        Optional figure suptitle.
    legend_title:
        Title for the companion legend figure.
    registry_path:
        Color registry JSON (read for the palette; written only if ``persist_colors``).
    persist_colors:
        Write the assigned colors back to the registry (default ``False``). The labels
        here are figure-local, so by default each figure colors from the palette head
        and nothing is persisted; set ``True`` (with a ``category``) to keep a label's
        color consistent across figures.

    Returns
    -------
    PValueHistogramPlot
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}.")
    data = _coerce_mapping(pvalues)
    labels = list(data.keys())
    finite = {label: _validate_pvalues(label, data[label]) for label in labels}
    result = PValueHistogramResult(
        counts={label: int(finite[label].size) for label in labels},
        pi0={label: estimate_pi0(finite[label], pi0_lambda) for label in labels},
    )

    with publication_style():
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            color_map = assign_colors(
                category, labels, registry_path=registry_path, persist=persist_colors
            )
            _draw_histograms(ax, finite, labels, color_map, n_bins=n_bins)
            ax.axhline(
                1.0,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                zorder=5,
            )
            if show_pi0 and len(labels) == 1:
                _draw_pi0_line(ax, labels[0], result.pi0[labels[0]], color_map)
            if title is not None:
                fig.suptitle(title, fontsize=14, weight="bold")
            legend_figure = _legend_figure(labels, color_map, result, legend_title)
        except BaseException:
            plt.close(fig)
            raise

    return PValueHistogramPlot(
        figure=fig, legend_figure=legend_figure, result=result, color_map=color_map
    )


def pvalue_histogram_from_result(
    result: DifferentialAbundanceResult,
    *,
    terms: list[str] | None = None,
    n_bins: int = 20,
    show_pi0: bool = True,
    pi0_lambda: float = _DEFAULT_PI0_LAMBDA,
    title: str | None = None,
    legend_title: str = "Contrast term",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = False,
) -> PValueHistogramPlot:
    """p-value histogram for a :class:`DifferentialAbundanceResult`'s contrast term(s).

    ``terms`` selects which contrast coefficients to overlay (each a labelled
    distribution); ``None`` uses all contrast terms (one for a binary/continuous
    contrast, ``k-1`` for a ``k``-level factor). Raw p-values are read off the table.
    """
    chosen = _resolve_terms(result, terms)
    data: dict[str, np.ndarray] = {}
    for term in chosen:
        rows = result.table[result.table["term"] == term]
        data[term] = rows["p"].to_numpy(dtype=float)
    return plot_pvalue_histogram(
        data,
        n_bins=n_bins,
        show_pi0=show_pi0,
        pi0_lambda=pi0_lambda,
        title=title,
        legend_title=legend_title,
        registry_path=registry_path,
        persist_colors=persist_colors,
    )


def _resolve_terms(
    result: DifferentialAbundanceResult, terms: list[str] | None
) -> list[str]:
    """Validate the requested contrast terms (default: all of them)."""
    available = list(result.contrast_terms)
    if terms is None:
        return available
    unknown = [t for t in terms if t not in available]
    if unknown:
        raise ValueError(
            f"term(s) {unknown} are not contrast terms; choose from {available}."
        )
    if not terms:
        raise ValueError("terms is empty; pass >= 1 contrast term or None for all.")
    return list(terms)


def save_pvalue_histogram(
    pvalues: Mapping[str, np.ndarray] | np.ndarray,
    output_dir: str | Path,
    base_name: str,
    *,
    n_bins: int = 20,
    show_pi0: bool = True,
    pi0_lambda: float = _DEFAULT_PI0_LAMBDA,
    category: str = DEFAULT_PVALUE_CATEGORY,
    title: str | None = None,
    legend_title: str = "Distribution",
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    persist_colors: bool = False,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render a p-value histogram and save it dual-export + a separate legend image."""
    plot = plot_pvalue_histogram(
        pvalues,
        n_bins=n_bins,
        show_pi0=show_pi0,
        pi0_lambda=pi0_lambda,
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
def _draw_histograms(
    ax: Axes,
    finite: dict[str, np.ndarray],
    labels: list[str],
    color_map: dict[str, str],
    *,
    n_bins: int,
) -> None:
    """Density histograms over a shared ``[0, 1]`` grid; bar for one, step for many."""
    bins = [float(edge) for edge in np.linspace(0.0, 1.0, n_bins + 1)]
    single = len(labels) == 1
    for label in labels:
        values = finite[label]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=color_map[label],
            alpha=0.55 if single else 1.0,
            histtype="bar" if single else "step",
            linewidth=0.0 if single else 2.0,
            edgecolor="white" if single else color_map[label],
        )
    ax.set_xlabel("p-value")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, 1.0)


def _draw_pi0_line(ax: Axes, label: str, pi0: float, color_map: dict[str, str]) -> None:
    """Mark the estimated π0 (null floor) for a single distribution."""
    if not np.isfinite(pi0):
        return
    ax.axhline(pi0, color=color_map[label], linestyle=":", linewidth=1.5, zorder=6)
    ax.text(
        0.98,
        pi0,
        f" π0≈{pi0:.2f}",
        color=color_map[label],
        fontsize=11,
        va="bottom",
        ha="right",
    )


def _legend_figure(
    labels: list[str],
    color_map: dict[str, str],
    result: PValueHistogramResult,
    legend_title: str,
) -> Figure:
    """Standalone legend: a swatch per distribution with its feature count and π0."""
    height = max(1.4, 0.4 * len(labels) + 0.8)
    fig, ax = plt.subplots(figsize=(3.6, height))
    ax.axis("off")
    handles = [Patch(facecolor=color_map[label], edgecolor="none") for label in labels]
    texts = [
        f"{label} (n={result.counts[label]}, π0≈{result.pi0[label]:.2f})"
        for label in labels
    ]
    ax.legend(
        handles,
        texts,
        title=legend_title,
        loc="center",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    return fig
