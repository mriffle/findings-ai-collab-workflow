"""Tests for the Boruta importance figure.

Built on synthetic :class:`~analysis.boruta.BorutaResult` objects (no Boruta run needed
— we test the figure, not the selection): row count vs top_n, the label formatter, dual
export with no separate legend image, and that the figure never leaks on an error path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from analysis.boruta import BorutaResult
from figures import boruta_importance as bfig
from matplotlib.figure import Figure


def _fake_result(
    n_features: int = 8,
    n_iter: int = 20,
    n_confirmed: int = 3,
    n_tentative: int = 1,
    seed: int = 0,
) -> BorutaResult:
    """A hand-built BorutaResult with consistent shapes for figure tests."""
    rng = np.random.default_rng(seed)
    names = np.array([f"P{i}" for i in range(n_features)], dtype=object)
    decision = np.array(["Rejected"] * n_features, dtype=object)
    decision[:n_confirmed] = "Confirmed"
    decision[n_confirmed : n_confirmed + n_tentative] = "Tentative"
    ranking = np.array(
        [1] * n_confirmed + [2] * n_tentative + list(range(3, 3 + n_features)),
        dtype=int,
    )[:n_features]
    importance = np.linspace(0.06, 0.004, n_features)
    history = np.abs(
        rng.normal(loc=importance[None, :], scale=0.01, size=(n_iter, n_features))
    )
    shadow = np.abs(rng.normal(loc=0.012, scale=0.002, size=n_iter))
    n_rejected = n_features - n_confirmed - n_tentative
    return BorutaResult(
        feature_names=names,
        decision=decision,
        ranking=ranking,
        importance=importance,
        importance_history=history,
        shadow_max_history=shadow,
        shadow_threshold=float(np.median(shadow)),
        target="grp",
        task="classification",
        classes=("A", "B"),
        n_confirmed=n_confirmed,
        n_tentative=n_tentative,
        n_rejected=n_rejected,
        n_iter=n_iter,
        n_samples=40,
        n_features=n_features,
        n_dropped_constant=0,
        n_dropped_missing_target=0,
        max_iter=100,
        alpha=0.05,
        perc=100.0,
        random_state=0,
    )


def test_plot_returns_figure_with_colorbar() -> None:
    fig = bfig.plot_boruta_importance(_fake_result(), top_n=8)
    assert isinstance(fig, Figure)
    # the main axes + the colorbar axes
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_top_n_caps_rows() -> None:
    fig = bfig.plot_boruta_importance(_fake_result(n_features=8), top_n=3)
    ax = fig.axes[0]
    assert len(ax.get_yticks()) == 3
    plt.close(fig)


def test_top_n_larger_than_features_shows_all() -> None:
    fig = bfig.plot_boruta_importance(_fake_result(n_features=5), top_n=30)
    assert len(fig.axes[0].get_yticks()) == 5
    plt.close(fig)


def test_top_n_nonpositive_raises() -> None:
    with pytest.raises(ValueError, match="top_n must be positive"):
        bfig.plot_boruta_importance(_fake_result(), top_n=0)


def test_label_formatter_applied() -> None:
    fig = bfig.plot_boruta_importance(
        _fake_result(n_features=4), top_n=4, label_formatter=lambda s: f"[{s}]"
    )
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert all(lbl.startswith("[") and lbl.endswith("]") for lbl in labels)
    plt.close(fig)


def test_default_label_truncates_long_names() -> None:
    res = _fake_result(n_features=1)
    res.feature_names[0] = "x" * 60
    fig = bfig.plot_boruta_importance(res, top_n=1)
    label = fig.axes[0].get_yticklabels()[0].get_text()
    assert label.endswith("...") and len(label) == bfig._MAX_LABEL_LEN
    plt.close(fig)


def test_save_dual_export_no_legend_image(tmp_path: Path) -> None:
    arts = bfig.save_boruta_importance(_fake_result(), tmp_path, "boruta_imp", top_n=6)
    assert Path(arts.svg).exists() and Path(arts.png).exists()
    # colorbar + on-axes key => no separate legend image
    assert arts.legend_svg is None
    assert arts.legend_png is None
    assert not (tmp_path / "boruta_imp.legend.png").exists()


def test_no_figure_leak_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    plt.close("all")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("drawing failed")

    monkeypatch.setattr(bfig, "_add_legend", boom)
    with pytest.raises(RuntimeError, match="drawing failed"):
        bfig.plot_boruta_importance(_fake_result(), top_n=4)
    assert plt.get_fignums() == []  # the figure was closed, not leaked
