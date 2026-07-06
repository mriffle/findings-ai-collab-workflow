"""Boruta importance figure — the all-relevant selection at a glance.

TEMPLATE (lib/) — a *seed* for a project's Boruta-figure module, not a finished script.
Copy it into the project's ``scripts/`` and adapt the call sites per study. Held to the
correctness charter (conventions/correctness.md): **assume nothing, verify everything,
fail loud.**

One figure reads a :class:`~analysis.boruta.BorutaResult`:

  * :func:`plot_boruta_importance` — a horizontal **box plot of per-iteration
    importance** for the top-N ranked features (box + jittered per-iteration scatter).
    **Accepted** features (Confirmed + Tentative) are colored by their **median
    importance** on ``viridis`` (a continuous gradient — no categorical color budget
    consumed); **Rejected** features are gray. A red dashed vertical line marks the
    **median shadow threshold** (``result.shadow_threshold``) — the noise floor a
    feature had to beat — so the plot reads as "which features cleared the shadows?".
    The C/T/R counts sit in the title.

It shares a *skeleton* with the classifier's coefficient plot (top-N horizontal, colored
by a per-feature stability quantity) but not its look: this shows the all-relevant
decision + importance-vs-noise-floor, not a signed minimal-optimal coefficient.

Convention wiring (conventions/visualization.md): the importance **colorbar sits beside
the axes** and the small shadow-line / rejected key sits **on-axes** (both provably
clear of the data, the documented exception) — so, like the classifier figures, this
passes **no** separate legend image to :func:`figures.figure_io.save_figure`; it still
dual-exports (SVG + 300-DPI PNG).

Feature labels: by default the raw feature name is shown (truncated if very long).
Domain prettifying — e.g. collapsing a UniProt ``sp|ACC|GENE_SPECIES`` id to its gene
symbol — is **not** baked in (a study's id scheme is unknowable a priori); pass a
``label_formatter`` callable to do it in the project copy.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analysis.boruta import BorutaResult
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from figures.figure_io import FigureArtifacts, publication_style, save_figure

__script_meta__: dict[str, object] = {
    "template": {"name": "boruta-importance", "version": "0.1"},
    "kind": "module",
    "provides": [
        "plot_boruta_importance",
        "save_boruta_importance",
    ],
    "uses": ["analysis.boruta", "figures.figure_io"],
    "seeded_from": None,
    "description": (
        "Boruta all-relevant selection figure for a BorutaResult: a horizontal box "
        "plot of per-iteration importance for the top-N ranked features (box + "
        "jittered "
        "scatter), accepted features colored by median importance on viridis and "
        "rejected features gray, with a red dashed line at the median shadow threshold "
        "(the noise floor) and the Confirmed/Tentative/Rejected counts in the title. "
        "Optional label_formatter for domain id prettifying (none baked in). The "
        "importance colorbar sits beside the axes and the shadow/rejected key on-axes "
        "(the documented legend exception), so no separate legend image; dual-export "
        "via figure-io. Uses analysis.boruta, figures.figure_io. Study-agnostic; "
        "fail-loud."
    ),
}

# House-style constants (encode no metadata category, so not color-registry slots).
_REJECTED_COLOR = "#888888"
_SHADOW_LINE_COLOR = "#D55E00"  # Okabe-Ito vermillion — the shadow-threshold reference
_CMAP_NAME = "viridis"
_MAX_LABEL_LEN = 40


def _default_label(name: str) -> str:
    """Show the raw feature name, truncated if very long (no domain assumptions)."""
    text = str(name)
    if len(text) > _MAX_LABEL_LEN:
        return text[: _MAX_LABEL_LEN - 3] + "..."
    return text


def plot_boruta_importance(
    result: BorutaResult,
    *,
    top_n: int = 30,
    title: str | None = None,
    label_formatter: Callable[[str], str] | None = None,
    jitter_seed: int = 42,
) -> Figure:
    """Box-plot the per-iteration importance history for the top-N ranked features.

    Features are ranked by Boruta ranking (then median importance within a rank) and the
    top ``top_n`` are shown, most important at the top. Accepted features are colored by
    median importance on ``viridis`` (the color scale spans **all** accepted features,
    not just the shown ones); rejected features are gray. A red dashed vertical line
    marks ``result.shadow_threshold``.

    Parameters
    ----------
    result:
        A :class:`~analysis.boruta.BorutaResult` from
        :func:`~analysis.boruta.boruta_select`.
    top_n:
        Number of top-ranked features to display (default 30). Must be positive.
    title:
        Optional suptitle; the C/T/R counts are appended below it either way.
    label_formatter:
        Optional ``name -> label`` callable for the y-axis (e.g. collapse a UniProt id
        to a gene symbol). Default shows the raw name, truncated if long.
    jitter_seed:
        Seed for the reproducible per-iteration scatter jitter.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}.")
    fmt = label_formatter if label_formatter is not None else _default_label

    history = np.asarray(result.importance_history, dtype=float)
    decisions = np.asarray(result.decision)
    medians = np.asarray(result.importance, dtype=float)
    ranking = np.asarray(result.ranking, dtype=float)
    names = np.asarray(result.feature_names)

    # Order: ranking ascending, then median importance descending.
    order = np.lexsort((-medians, ranking))
    n_show = min(top_n, order.size)
    order = order[:n_show]

    accepted = (decisions == "Confirmed") | (decisions == "Tentative")
    vmin, vmax = _accepted_color_range(medians, accepted)
    cmap = plt.get_cmap(_CMAP_NAME)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rng = np.random.default_rng(jitter_seed)

    with publication_style():
        fig, ax = plt.subplots(figsize=(11, max(6.0, 0.4 * n_show)))
        try:
            for i, feat in enumerate(order):
                color = (
                    _REJECTED_COLOR
                    if decisions[feat] == "Rejected"
                    else cmap(norm(medians[feat]))
                )
                _draw_feature_row(ax, i, history[:, feat], color, rng)
            ax.axvline(
                result.shadow_threshold,
                color=_SHADOW_LINE_COLOR,
                linestyle="--",
                linewidth=1.3,
                zorder=4,
            )
            ax.set_yticks(range(n_show))
            ax.set_yticklabels([fmt(str(names[feat])) for feat in order])
            ax.set_ylim(n_show - 0.5, -0.5)
            ax.set_xlabel("Boruta feature importance (per iteration)")
            _add_colorbar(fig, ax, cmap, norm)
            _add_legend(ax, result)
            _apply_title(fig, result, title)
        except BaseException:
            plt.close(fig)
            raise
    return fig


def _accepted_color_range(
    medians: np.ndarray, accepted: np.ndarray
) -> tuple[float, float]:
    """Color-scale bounds over all accepted features (fallback when none accepted)."""
    if not bool(accepted.any()):
        return 0.0, 1.0
    vals = medians[accepted]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax


def _draw_feature_row(
    ax: Axes,
    position: int,
    feat_history: np.ndarray,
    color: tuple[float, float, float, float] | str,
    rng: np.random.Generator,
) -> None:
    """Draw one feature's box + jittered per-iteration scatter at ``position``."""
    valid = feat_history[~np.isnan(feat_history)]
    if valid.size == 0:
        return
    box = ax.boxplot(
        valid,
        positions=[position],
        orientation="horizontal",
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "#444444", "linewidth": 0.8},
        capprops={"color": "#444444", "linewidth": 0.8},
        boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        zorder=2,
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    y_jitter = position + rng.uniform(-0.18, 0.18, valid.size)
    ax.scatter(
        valid, y_jitter, color=color, alpha=0.75, s=10, edgecolors="none", zorder=3
    )


def _add_colorbar(fig: Figure, ax: Axes, cmap: Colormap, norm: Normalize) -> None:
    """Viridis colorbar for the accepted-feature median importance (beside the axes)."""
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(np.empty(0))
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("median importance (accepted)", fontsize=9)


def _add_legend(ax: Axes, result: BorutaResult) -> None:
    """On-axes key: the shadow-threshold line and the rejected-feature swatch."""
    handles = [
        Line2D(
            [0],
            [0],
            color=_SHADOW_LINE_COLOR,
            linestyle="--",
            linewidth=1.3,
            label=f"shadow max (median): {result.shadow_threshold:.4f}",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_REJECTED_COLOR,
            markersize=10,
            label=f"rejected ({result.n_rejected})",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)


def _apply_title(fig: Figure, result: BorutaResult, title: str | None) -> None:
    counts = (
        f"Confirmed: {result.n_confirmed}  |  Tentative: {result.n_tentative}  |  "
        f"Rejected: {result.n_rejected}"
    )
    default = f"Boruta all-relevant selection — target: {result.target}"
    fig.suptitle(f"{title or default}\n{counts}", fontsize=12, weight="bold")


def save_boruta_importance(
    result: BorutaResult,
    output_dir: str | Path,
    base_name: str,
    *,
    top_n: int = 30,
    title: str | None = None,
    label_formatter: Callable[[str], str] | None = None,
    jitter_seed: int = 42,
    dpi: int = 300,
) -> FigureArtifacts:
    """Render :func:`plot_boruta_importance` and dual-export it (no separate legend)."""
    fig = plot_boruta_importance(
        result,
        top_n=top_n,
        title=title,
        label_formatter=label_formatter,
        jitter_seed=jitter_seed,
    )
    return save_figure(fig, output_dir, base_name, dpi=dpi)
