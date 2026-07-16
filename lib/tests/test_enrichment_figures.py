"""Tests for the enrichment figures (lib/figures/enrichment.py).

Layers:
  * render — each of the three figures produces a main + a separate legend figure;
  * io — save_* writes the dual export + the legend image; a bad base_name closes both;
  * coloring — source colors come from the project registry (persisted);
  * guards — manhattan needs all_results; empty significant set -> empty-state figure;
    top_n must be positive; a broken registry closes the figure (no leak).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from analysis import enrichment as en
from figures import enrichment as ef
from matplotlib.figure import Figure

_PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "color_registry.json"
    path.write_text(
        json.dumps(
            {
                "_palette": {
                    "name": "Okabe-Ito",
                    "colors": _PALETTE,
                    "max_categorical": 8,
                }
            }
        )
    )
    return path


def _make_result(
    *, all_results: bool = True, significant: bool = True
) -> en.EnrichmentResult:
    """A small multi-source EnrichmentResult built directly (no network)."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)
    for src in ("GO:BP", "GO:MF", "GO:CC", "KEGG"):
        for k in range(5):
            p = float(10 ** (-(1 + 6 * rng.random())))
            is_sig = significant and p <= 0.05
            inter = int(3 + k * 4)
            rows.append(
                {
                    "source": src,
                    "term_id": f"{src}:{k}",
                    "term_name": f"{src} example term {k}",
                    "p_value": p if all_results else (p if is_sig else 0.5),
                    "significant": is_sig,
                    "term_size": 100 + k * 10,
                    "query_size": 50,
                    "intersection_size": inter,
                    "effective_domain_size": 8000,
                    "gene_ratio": inter / 50,
                    "recall": inter / (100 + k * 10),
                }
            )
    table = pd.DataFrame(rows)
    if not all_results:
        table = table[table["significant"]].reset_index(drop=True)
    return en.EnrichmentResult(
        table=table,
        mapping=pd.DataFrame(
            columns=[
                "original_id",
                "uniprot_accession",
                "source",
                "in_query",
                "in_background",
            ]
        ),
        organism="mmusculus",
        sources=("GO:BP", "GO:MF", "GO:CC", "KEGG"),
        correction="g_SCS",
        user_threshold=0.05,
        gprofiler_version="eTEST",
        query_label="up in test",
        n_query=50,
        n_query_mapped=50,
        n_query_recognized=48,
        n_background=8000,
        n_background_mapped=7900,
        effective_domain_size=7900,
        all_results_included=all_results,
    )


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "func",
    [ef.enrichment_dotplot, ef.enrichment_barplot, ef.enrichment_manhattan],
)
def test_render_produces_main_and_legend(func: object, registry: Path) -> None:
    result = _make_result()
    plot = func(result, registry_path=registry)  # type: ignore[operator]
    assert isinstance(plot.figure, Figure)
    assert isinstance(plot.legend_figure, Figure)
    assert plot.figure is not plot.legend_figure
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("save_func", "base"),
    [
        (ef.save_enrichment_dotplot, "enr_dot"),
        (ef.save_enrichment_barplot, "enr_bar"),
        (ef.save_enrichment_manhattan, "enr_man"),
    ],
)
def test_save_writes_figure_and_legend(
    save_func: object, base: str, registry: Path, tmp_path: Path
) -> None:
    result = _make_result()
    arts = save_func(result, tmp_path, base, registry_path=registry)  # type: ignore[operator]
    assert arts.svg.exists() and arts.png.exists()
    assert arts.legend_svg is not None and arts.legend_svg.exists()
    assert arts.legend_png is not None and arts.legend_png.exists()


def test_save_bad_base_name_closes_figures(registry: Path, tmp_path: Path) -> None:
    result = _make_result()
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="bare filename stem"):
        ef.save_enrichment_dotplot(result, tmp_path, "bad/name", registry_path=registry)
    assert set(plt.get_fignums()) == before


# --------------------------------------------------------------------------- #
# Coloring
# --------------------------------------------------------------------------- #
def test_source_colors_persisted_to_registry(registry: Path) -> None:
    result = _make_result()
    plot = ef.enrichment_barplot(result, registry_path=registry)
    saved = json.loads(registry.read_text())
    assert ef.DEFAULT_SOURCE_CATEGORY in saved
    values = saved[ef.DEFAULT_SOURCE_CATEGORY]["values"]
    assert "GO:BP" in values and values["GO:BP"] in _PALETTE
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_manhattan_requires_all_results(registry: Path) -> None:
    result = _make_result(all_results=False)
    with pytest.raises(ValueError, match="all_results=True"):
        ef.enrichment_manhattan(result, registry_path=registry)


def test_empty_significant_is_empty_state(registry: Path) -> None:
    result = _make_result(significant=False)
    plot = ef.enrichment_dotplot(result, registry_path=registry)
    # Renders a placeholder rather than crashing; both figures exist.
    assert isinstance(plot.figure, Figure)
    assert isinstance(plot.legend_figure, Figure)
    plt.close(plot.figure)
    plt.close(plot.legend_figure)


def test_top_n_must_be_positive(registry: Path) -> None:
    result = _make_result()
    with pytest.raises(ValueError, match="top_n"):
        ef.enrichment_dotplot(result, top_n=0, registry_path=registry)


def test_broken_registry_closes_figure(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}")  # no _palette
    result = _make_result()
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="_palette"):
        ef.enrichment_barplot(result, registry_path=bad)
    assert set(plt.get_fignums()) == before
