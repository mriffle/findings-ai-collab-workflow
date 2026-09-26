"""Tests for the aggregate-replicates template (replicates.py).

Layers:
  * unit/aggregation — hand-computed unit means / medians for 3-, 2-, and 1-run units,
    first-appearance order, the index / key columns, ``n_replicates``, the replicate
    histogram and run map, and the invariants (singleton rows bit-identical, the
    replicate-weighted grand mean preserved, row-order invariance, no aliasing);
  * unit/metadata — the constant-within-unit check (incl. NaN vs value), run-level
    drop, ``summarize`` numeric means + categorical fractions (hand-computed);
  * unit/weights — the precision-weight formula, the planted-truth recovery of the
    technical-variance fraction (with and without the design — the conservative
    direction), and the no-repeat pass-through;
  * unit/guards — scale refusal, non-finite abundances, missing/NaN keys, unknown /
    overlapping columns, an already-aggregated input, a rank-deficient weight design;
  * smoke — real 5xFAD proteins (git-ignored): 74 experimental runs → 52 animals.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from common import data_loading as dl
from common import replicates as rp

# Most fixtures here have unequal run counts and no weight_design; the warning that
# flags that combination is tested explicitly below.
pytestmark = pytest.mark.filterwarnings(
    "ignore::common.replicates.EmptyWeightDesignWarning"
)


def _ds(
    abundances: np.ndarray,
    metadata: dict[str, Any],
    *,
    scale: dl.Scale = "log2",
) -> dl.Dataset:
    """A Dataset (default log2) from an abundance matrix + a metadata dict."""
    n_samples, n_features = abundances.shape
    names = np.array([f"F{j}" for j in range(n_features)], dtype=str)
    return dl.Dataset(
        abundances=np.asarray(abundances, dtype=float),
        feature_names=names,
        feature_metadata=pd.DataFrame({"feature": names}),
        metadata=pd.DataFrame(metadata, index=[f"r{i}" for i in range(n_samples)]),
        scale=scale,
    )


def _three_units() -> dl.Dataset:
    """Unit A: 3 runs, unit B: 1 run, unit C: 2 runs (interleaved)."""
    ab = np.array(
        [
            [1.0, 10.0],  # A
            [5.0, 20.0],  # B
            [2.0, 11.0],  # A
            [7.0, 30.0],  # C
            [6.0, 13.0],  # A
            [9.0, 34.0],  # C
        ]
    )
    meta = {
        "animal": ["A", "B", "A", "C", "A", "C"],
        "group": ["g1", "g2", "g1", "g2", "g1", "g2"],
        "run": [1, 2, 3, 4, 5, 6],
    }
    return _ds(ab, meta)


# --------------------------------------------------------------------------- #
# Unit — aggregation (planted truth)
# --------------------------------------------------------------------------- #
def test_mean_hand_computed_and_first_appearance_order() -> None:
    res = rp.aggregate_replicates(_three_units(), "animal", run_level=["run"])
    out = res.dataset
    assert list(out.metadata.index) == ["A", "B", "C"]
    assert out.metadata.index.name == "animal"
    np.testing.assert_allclose(
        out.abundances, [[3.0, 34.0 / 3.0], [5.0, 20.0], [8.0, 32.0]]
    )
    assert list(out.metadata["animal"]) == ["A", "B", "C"]
    assert list(out.metadata["group"]) == ["g1", "g2", "g2"]
    assert "run" not in out.metadata.columns
    assert list(out.metadata[rp.N_REPLICATES]) == [3, 1, 2]
    assert out.metadata[rp.N_REPLICATES].dtype == np.int64
    assert res.n_runs_in == 6
    assert res.n_units_out == 3
    assert res.replicate_counts == {"1": 1, "2": 1, "3": 1}
    assert list(res.run_map["unit"]) == ["A", "B", "A", "C", "A", "C"]
    assert list(res.run_map["sample"]) == [f"r{i}" for i in range(6)]
    assert out.scale == "log2"


def test_median_differs_from_mean_only_at_three_runs() -> None:
    res = rp.aggregate_replicates(
        _three_units(), "animal", method="median", run_level=["run"]
    )
    # A: median(1, 2, 6) = 2, median(10, 11, 13) = 11; C (2 runs): median == mean.
    np.testing.assert_allclose(
        res.dataset.abundances, [[2.0, 11.0], [5.0, 20.0], [8.0, 32.0]]
    )


def test_singleton_rows_are_bit_identical() -> None:
    rng = np.random.default_rng(3)
    ab = rng.normal(20, 3, size=(5, 40))
    meta = {"unit": ["a", "b", "b", "c", "d"], "run": [1, 2, 3, 4, 5]}
    out = rp.aggregate_replicates(_ds(ab, meta), "unit", run_level=["run"]).dataset
    assert np.array_equal(out.abundances[0], ab[0])
    assert np.array_equal(out.abundances[2], ab[3])
    assert np.array_equal(out.abundances[3], ab[4])


def test_replicate_weighted_grand_mean_preserved() -> None:
    rng = np.random.default_rng(4)
    ab = rng.normal(0, 1, size=(9, 7))
    meta = {"unit": list("aaabbcddd"), "run": list(range(9))}
    out = rp.aggregate_replicates(_ds(ab, meta), "unit", run_level=["run"]).dataset
    n = out.metadata[rp.N_REPLICATES].to_numpy(dtype=float)
    weighted = (out.abundances * n[:, None]).sum(axis=0) / n.sum()
    np.testing.assert_allclose(weighted, ab.mean(axis=0))


def test_row_order_invariance() -> None:
    ds = _three_units()
    perm = np.array([5, 3, 1, 0, 4, 2])
    shuffled = dataclasses.replace(
        ds, abundances=ds.abundances[perm], metadata=ds.metadata.iloc[perm]
    )
    a = rp.aggregate_replicates(ds, "animal", run_level=["run"]).dataset
    b = rp.aggregate_replicates(shuffled, "animal", run_level=["run"]).dataset
    for key in ("A", "B", "C"):
        i = list(a.metadata.index).index(key)
        j = list(b.metadata.index).index(key)
        np.testing.assert_allclose(a.abundances[i], b.abundances[j])


def test_output_is_independent_of_input() -> None:
    ds = _three_units()
    out = rp.aggregate_replicates(ds, "animal", run_level=["run"]).dataset
    out.abundances[0, 0] = -99.0
    out.metadata.loc["A", "group"] = "changed"
    out.feature_names[0] = "changed"
    assert ds.abundances[0, 0] == 1.0
    assert ds.metadata["group"].iloc[0] == "g1"
    assert ds.feature_names[0] == "F0"


def test_multi_column_key_joins_with_pipe() -> None:
    ab = np.arange(8, dtype=float).reshape(4, 2)
    meta = {
        "mouse": ["m1", "m1", "m1", "m2"],
        "time": ["t0", "t0", "t1", "t0"],
        "rep": [1, 2, 1, 1],
    }
    res = rp.aggregate_replicates(_ds(ab, meta), ("mouse", "time"), run_level=["rep"])
    md = res.dataset.metadata
    assert list(md.index) == ["m1|t0", "m1|t1", "m2|t0"]
    assert md.index.name == "mouse|time"
    assert list(md["mouse"]) == ["m1", "m1", "m2"]
    np.testing.assert_allclose(res.dataset.abundances[0], [1.0, 2.0])


def test_no_repeats_passes_through_with_unit_weights() -> None:
    ab = np.arange(6, dtype=float).reshape(3, 2)
    res = rp.aggregate_replicates(_ds(ab, {"unit": ["a", "b", "c"]}), "unit")
    np.testing.assert_array_equal(res.dataset.abundances, ab)
    assert list(res.dataset.metadata[rp.PRECISION_WEIGHT]) == [1.0, 1.0, 1.0]
    assert res.technical_variance_fraction is None
    assert res.replicate_counts == {"1": 3}


# --------------------------------------------------------------------------- #
# Unit — metadata handling
# --------------------------------------------------------------------------- #
def test_varying_column_raises_with_ready_to_paste_run_level() -> None:
    with pytest.raises(ValueError, match=r"run_level=\('run',\)") as err:
        rp.aggregate_replicates(_three_units(), "animal")
    assert "['run']" in str(err.value)
    assert "pairing error" in str(err.value)


def test_design_label_disagreement_within_unit_raises() -> None:
    ds = _three_units()
    ds.metadata.iloc[2, ds.metadata.columns.get_loc("group")] = "g2"
    with pytest.raises(ValueError, match=r"\['group'\] vary within a unit"):
        rp.aggregate_replicates(ds, "animal", run_level=["run"])


def test_nan_versus_value_counts_as_varying() -> None:
    ab = np.zeros((2, 1))
    meta = {"unit": ["a", "a"], "note": ["x", None]}
    with pytest.raises(ValueError, match="note"):
        rp.aggregate_replicates(_ds(ab, meta), "unit")


def test_summarize_numeric_mean_and_categorical_fractions() -> None:
    ab = np.zeros((5, 1))
    meta = {
        "unit": ["a", "a", "a", "b", "b"],
        "order": [1.0, 2.0, 6.0, 10.0, 20.0],
        "batch": ["B1", "B2", "B2", "B1", "B3"],
    }
    res = rp.aggregate_replicates(_ds(ab, meta), "unit", summarize=["order", "batch"])
    md = res.dataset.metadata
    np.testing.assert_allclose(md["order"], [3.0, 15.0])
    assert "batch" not in md.columns
    np.testing.assert_allclose(md["batch[frac B2]"], [2 / 3, 0.0])
    np.testing.assert_allclose(md["batch[frac B3]"], [0.0, 0.5])
    assert "batch[frac B1]" not in md.columns
    assert res.summarize_reference == {"batch": "B1"}
    assert res.summarize_columns == {
        "order": ("order",),
        "batch": ("batch[frac B2]", "batch[frac B3]"),
    }


def test_summarize_single_level_raises() -> None:
    ab = np.zeros((2, 1))
    meta = {"unit": ["a", "a"], "batch": ["B1", "B1"]}
    with pytest.raises(ValueError, match="single level"):
        rp.aggregate_replicates(_ds(ab, meta), "unit", summarize=["batch"])


# --------------------------------------------------------------------------- #
# Unit — precision weights
# --------------------------------------------------------------------------- #
def _planted_variance_components(
    *, sigma_b: float, sigma_t: float, n_units: int = 240, n_features: int = 400
) -> dl.Dataset:
    """Units with 1/2/3 runs, a planted between-group effect, known vb and vt."""
    rng = np.random.default_rng(11)
    runs = np.tile([1, 2, 3], n_units // 3)
    group = np.where(np.arange(n_units) % 2 == 0, "ctrl", "case")
    effect = rng.normal(0, 3.0, size=n_features)  # large, so ignoring it matters
    rows: list[np.ndarray] = []
    meta: dict[str, list[Any]] = {"unit": [], "group": [], "run": []}
    for u in range(n_units):
        mu = rng.normal(0, sigma_b, size=n_features)
        if group[u] == "case":
            mu = mu + effect
        for r in range(runs[u]):
            rows.append(mu + rng.normal(0, sigma_t, size=n_features))
            meta["unit"].append(f"u{u}")
            meta["group"].append(group[u])
            meta["run"].append(r)
    return _ds(np.vstack(rows), meta)


@pytest.mark.parametrize("f_true", [0.2, 0.5, 0.8])
def test_weights_recover_planted_technical_fraction_with_design(f_true: float) -> None:
    ds = _planted_variance_components(
        sigma_b=float(np.sqrt(1.0 - f_true)), sigma_t=float(np.sqrt(f_true))
    )
    res = rp.aggregate_replicates(
        ds, "unit", run_level=["run"], weight_design=["group"]
    )
    assert res.technical_variance_fraction is not None
    assert res.technical_variance_fraction == pytest.approx(f_true, abs=0.02)
    assert res.weight_design == ("group",)


def test_weights_without_design_are_conservative() -> None:
    ds = _planted_variance_components(sigma_b=1.0, sigma_t=1.0)
    with_design = rp.aggregate_replicates(
        ds, "unit", run_level=["run"], weight_design=["group"]
    )
    without = rp.aggregate_replicates(ds, "unit", run_level=["run"])
    assert with_design.technical_variance_fraction is not None
    assert without.technical_variance_fraction is not None
    assert without.technical_variance_fraction < with_design.technical_variance_fraction


def test_weights_follow_formula_and_have_mean_one() -> None:
    ds = _planted_variance_components(sigma_b=1.0, sigma_t=2.0)
    res = rp.aggregate_replicates(
        ds, "unit", run_level=["run"], weight_design=["group"]
    )
    f = res.technical_variance_fraction
    assert f is not None
    md = res.dataset.metadata
    n = md[rp.N_REPLICATES].to_numpy(dtype=float)
    raw = 1.0 / ((1.0 - f) + f / n)
    np.testing.assert_allclose(md[rp.PRECISION_WEIGHT], raw / raw.mean())
    assert md[rp.PRECISION_WEIGHT].mean() == pytest.approx(1.0)
    # More runs, more weight.
    by_n = md.groupby(rp.N_REPLICATES)[rp.PRECISION_WEIGHT].first()
    assert by_n[1] < by_n[2] < by_n[3]


def test_weight_design_accepts_summarized_batch_fractions() -> None:
    rng = np.random.default_rng(5)
    ab = rng.normal(0, 1, size=(8, 60))
    meta = {
        "unit": list("aabbccdd"),
        "grp": list("xxxxyyyy"),
        "batch": ["B1", "B2", "B1", "B1", "B2", "B1", "B2", "B2"],
    }
    res = rp.aggregate_replicates(
        _ds(ab, meta), "unit", summarize=["batch"], weight_design=["grp", "batch"]
    )
    assert res.technical_variance_fraction is not None


# --------------------------------------------------------------------------- #
# Unit — fail-loud guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", ["linear", "ratio"])
def test_non_log_scale_refused(scale: dl.Scale) -> None:
    ds = dataclasses.replace(_three_units(), scale=scale)
    with pytest.raises(rp.ReplicateScaleError, match="log scale"):
        rp.aggregate_replicates(ds, "animal", run_level=["run"])


@pytest.mark.parametrize("scale", ["log2", "log10", "ln", "glog2", "zscore"])
def test_log_scales_accepted_and_preserved(scale: dl.Scale) -> None:
    ds = dataclasses.replace(_three_units(), scale=scale)
    out = rp.aggregate_replicates(ds, "animal", run_level=["run"]).dataset
    assert out.scale == scale


def test_nonfinite_abundances_raise() -> None:
    ds = _three_units()
    ds.abundances[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN/inf"):
        rp.aggregate_replicates(ds, "animal", run_level=["run"])


def test_missing_key_value_raises() -> None:
    ds = _three_units()
    ds.metadata.iloc[0, ds.metadata.columns.get_loc("animal")] = None
    with pytest.raises(ValueError, match="missing for 1 row"):
        rp.aggregate_replicates(ds, "animal", run_level=["run"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"by": "nope"}, "by column"),
        ({"by": "animal", "run_level": ["nope"]}, "run_level column"),
        ({"by": "animal", "run_level": ["animal"]}, "both by and run_level"),
        (
            {"by": "animal", "run_level": ["run"], "summarize": ["run"]},
            "both run_level and summarize",
        ),
        ({"by": "animal", "run_level": ["run", "run"]}, "more than once"),
        ({"by": []}, "at least one"),
    ],
)
def test_column_guards(kwargs: dict[str, Any], match: str) -> None:
    by = kwargs.pop("by")
    with pytest.raises(ValueError, match=match):
        rp.aggregate_replicates(_three_units(), by, **kwargs)


def test_already_aggregated_input_raises() -> None:
    once = rp.aggregate_replicates(_three_units(), "animal", run_level=["run"])
    with pytest.raises(ValueError, match="already aggregated"):
        rp.aggregate_replicates(once.dataset, "animal")


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        rp.aggregate_replicates(
            _three_units(),
            "animal",
            method="mode",  # type: ignore[arg-type]
            run_level=["run"],
        )


def test_weight_design_guards() -> None:
    ds = _three_units()
    with pytest.raises(ValueError, match="not a unit-level column"):
        rp.aggregate_replicates(ds, "animal", run_level=["run"], weight_design=["run"])
    meta = dict(ds.metadata)
    meta["group2"] = ds.metadata["group"]
    aliased = _ds(ds.abundances, {k: list(v) for k, v in meta.items()})
    with pytest.raises(ValueError, match="rank-deficient"):
        rp.aggregate_replicates(
            aliased, "animal", run_level=["run"], weight_design=["group", "group2"]
        )


# --------------------------------------------------------------------------- #
# Unit — review fixes (key collisions, warnings, categorical weight design, cache)
# --------------------------------------------------------------------------- #
def test_multi_column_keys_that_join_identically_raise() -> None:
    ab = np.zeros((2, 1))
    meta = {"a": ["x|y", "x"], "b": ["z", "y|z"]}
    with pytest.raises(ValueError, match="display as the same key"):
        rp.aggregate_replicates(_ds(ab, meta), ("a", "b"))


def test_unit_ids_differing_only_in_type_raise() -> None:
    ab = np.zeros((3, 1))
    with pytest.raises(ValueError, match="differ only in type"):
        rp.aggregate_replicates(_ds(ab, {"u": [1, "1", "2"]}), "u")


def test_empty_weight_design_warns_only_when_counts_differ() -> None:
    with pytest.warns(rp.EmptyWeightDesignWarning, match="weight_design"):
        rp.aggregate_replicates(_three_units(), "animal", run_level=["run"])
    equal = _ds(np.arange(8.0).reshape(4, 2), {"u": list("aabb"), "r": [1, 2, 1, 2]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", rp.EmptyWeightDesignWarning)
        rp.aggregate_replicates(equal, "u", run_level=["r"])  # equal counts: silent
        rp.aggregate_replicates(
            _three_units(), "animal", run_level=["run"], weight_design=["group"]
        )


def _coded_batch_effect() -> dl.Dataset:
    """A non-monotone batch effect (0, 0, +4) with the batch coded as integers 1/2/3."""
    rng = np.random.default_rng(21)
    n_units, n_features = 60, 300
    batch = np.tile([1, 2, 3], n_units // 3)
    shift = np.array([0.0, 0.0, 4.0])[batch - 1]
    rows, meta = [], {"unit": [], "batch": [], "run": []}
    for u in range(n_units):
        mu = rng.normal(0, 0.1, size=n_features) + shift[u]
        for r in range(1 + u % 2):
            rows.append(mu + rng.normal(0, 1.0, size=n_features))
            meta["unit"].append(f"u{u}")
            meta["batch"].append(int(batch[u]))
            meta["run"].append(r)
    return _ds(np.vstack(rows), meta)


def test_integer_coded_factor_in_weight_design_warns_and_categorical_fixes_it() -> None:
    ds = _coded_batch_effect()
    with pytest.warns(rp.WeightDesignCardinalityWarning, match="categorical="):
        slope = rp.aggregate_replicates(
            ds, "unit", run_level=["run"], weight_design=["batch"]
        )
    factor = rp.aggregate_replicates(
        ds, "unit", run_level=["run"], weight_design=["batch"], categorical=["batch"]
    )
    assert slope.technical_variance_fraction is not None
    assert factor.technical_variance_fraction is not None
    # vb is ~0.01 and vt 1, so the true fraction is ~0.99; the slope misfits the
    # non-monotone effect and leaves it in vb.
    assert factor.technical_variance_fraction > 0.95
    assert slope.technical_variance_fraction < factor.technical_variance_fraction - 0.2


def test_summarize_numeric_with_missing_values_raises() -> None:
    ab = np.zeros((2, 1))
    meta = {"unit": ["a", "a"], "order": [1.0, np.nan]}
    with pytest.raises(ValueError, match="missing values"):
        rp.aggregate_replicates(_ds(ab, meta), "unit", summarize=["order"])


def test_aggregation_result_round_trips_through_the_result_cache(
    tmp_path: Path,
) -> None:
    from analysis import result_io as rio

    res = rp.aggregate_replicates(
        _three_units(), "animal", run_level=["run"], weight_design=["group"]
    )
    rio.save_result(res, tmp_path / "agg")
    back = rio.load_result(tmp_path / "agg", rp.ReplicateAggregation)
    np.testing.assert_array_equal(back.dataset.abundances, res.dataset.abundances)
    pd.testing.assert_frame_equal(back.run_map, res.run_map)
    assert back.replicate_counts == res.replicate_counts
    assert back.technical_variance_fraction == res.technical_variance_fraction
    assert back.summarize_columns == res.summarize_columns
    assert back.dataset.scale == "log2"


# --------------------------------------------------------------------------- #
# Smoke — real 5xFAD proteins (git-ignored)
# --------------------------------------------------------------------------- #
_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)
_RUN_LEVEL = ("Replicate", "RunOrder", "Technical Replicate")


def _experimental_runs() -> dl.Dataset:
    """All runs (no keep-one collapse), median + log2, pools dropped, Disease added."""
    from common import normalize as norm

    ds = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    logged = norm.log2_transform(norm.normalize(ds, "median"))
    mask = logged.metadata["Genotype"].to_numpy() != "na"
    meta = logged.metadata.loc[mask].copy()
    meta["Disease"] = np.where(meta["Genotype"].to_numpy() == "5xFAD", "5xFAD", "nonAD")
    return dataclasses.replace(
        logged, abundances=logged.abundances[mask, :], metadata=meta
    )


@_skip_no_data
def test_smoke_5xfad_runs_to_animals() -> None:
    runs = _experimental_runs()
    assert runs.abundances.shape[0] == 74
    res = rp.aggregate_replicates(
        runs,
        "Sample ID",
        run_level=_RUN_LEVEL,
        weight_design=("Disease", "Gender", "Treatment", "Cohort"),
    )
    assert res.n_units_out == 52
    assert res.replicate_counts == {"1": 30, "2": 22}
    assert res.technical_variance_fraction == pytest.approx(0.516, abs=0.005)
    md = res.dataset.metadata
    # Singleton animals are their run, unchanged.
    singles = md.index[md[rp.N_REPLICATES] == 1]
    run_units = runs.metadata["Sample ID"].astype(str).to_numpy()
    for animal in singles[:5]:
        row = int(np.flatnonzero(run_units == animal)[0])
        i = list(md.index).index(animal)
        assert np.array_equal(res.dataset.abundances[i], runs.abundances[row])
    # Two-run animals weigh more than one-run animals.
    by_n = md.groupby(rp.N_REPLICATES)[rp.PRECISION_WEIGHT].first()
    assert by_n[2] > by_n[1]
