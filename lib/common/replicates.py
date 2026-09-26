"""Verified replicate aggregation: one row per independent unit, plus precision weights.

TEMPLATE (lib/) — a *seed* for a project's ``common`` replicate module, not a finished
script. Copy it into the project's ``scripts/`` and adapt the call site (which column is
the unit, which columns are run-level) per study. Held to the correctness charter
(conventions/correctness.md): **assume nothing, verify everything, fail loud.**

WHY IT EXISTS. A per-feature test treats every row as an independent observation.
Technical replicates (the same biological sample injected twice) are not independent:
testing them as separate samples inflates the degrees of freedom and makes the p-values
too small (pseudoreplication — on real 5xFAD, 86 hits at q<0.05 on all runs vs ~62 once
each animal counts once). When the contrast is a property of the unit (genotype,
treatment arm, sex), the honest analysis tests one row per unit. This template produces
that row: the **mean** (default) or median of the unit's runs, on the log scale.

WHERE IT RUNS. After per-run processing, before a unit-level test: load all runs →
``handle_missing`` → ``normalize`` / ``log2_transform`` → (ComBat, for visualization and
classifiers only) → **aggregate_replicates** → ``differential_abundance``. Normalization
is per run (each injection has its own loading), so it must precede the aggregation; QC
(id-depth, correlation, annotation concordance) is also per run, so the loaders keep
every run and this step happens only where a unit-level test needs it.

SCALE IS A HARD REFUSE on ``linear`` / ``ratio``. The log-space mean is the geometric
mean of the runs (limma ``avereps``); an arithmetic mean of linear intensities is
pulled toward the larger replicate. ``zscore`` input is accepted — the output row is
the mean of per-run z-scores and is **not** re-standardized.

METADATA. The ``by`` columns name the unit: one column (``"Sample ID"``) or several
(``("Mouse", "Timepoint")`` when technical replicates sit inside a within-subject
design). Every other column must be **constant within each unit** or be declared:

* ``run_level`` — acquisition variables that legitimately differ between a unit's runs
  (run order, replicate number, the data-file key). Dropped from the output. **Never
  list a design variable here to silence the error** — a design variable that varies
  within a unit means the unit is wrong (add it to ``by``) or the design is
  repeated-measures.
* ``summarize`` — a run-level variable that must survive as a unit-level covariate, the
  canonical case being **technical replicates spanning batches**. A numeric column
  becomes its unit mean (same name); a categorical column becomes ``k-1`` fraction
  columns ``"{col}[frac {level}]"`` (the sorted-first level dropped — the fractions sum
  to 1, so the full set is collinear with the intercept). Under an additive batch
  effect a unit mean carries exactly (per-unit batch fractions) x (batch effects), so a
  DE model with these columns as covariates stays correctly specified after averaging.

A column that varies within a unit and is neither is an error that names the columns
and prints a ready-to-paste ``run_level=(...)``. That check doubles as an integrity
check: an animal whose replicates disagree on Genotype is a pairing error, caught here.

PRECISION WEIGHTS. A unit averaged over ``n`` runs is more precise than a one-run unit:
``var(unit mean) = vb + vt/n`` (``vb`` biological, ``vt`` technical variance). When
replicate counts differ — and on real data they often track the contrast (5xFAD: 18 of
31 transgenic animals have two runs, 4 of 21 controls) — an unweighted test puts that
imbalance onto the contrast. The template estimates one **consensus technical-variance
fraction** ``f = vt / (vb + vt)`` across features and emits a per-unit weight
``1 / ((1 - f) + f/n)`` (rescaled to mean 1) as the metadata column
``precision_weight`` — one weight per unit shared by every feature, in the spirit of
limma's ``arrayWeights``, so the differential-abundance template's shared-design
vectorized core and its variance moderation apply unchanged (``weights=``).

Per feature: ``vt`` is the pooled within-unit variance (units with >= 2 runs); the unit
means are regressed on ``weight_design`` (intercept + the columns the scientist will
test); ``vb`` is recovered from the residual sum of squares by its exact expectation,
``E[RSS] = vb * sum(1 - h_u) + vt * sum((1 - h_u) / n_u)`` (``h_u`` the leverages),
clamped at 0. ``f`` is the median of the per-feature fractions. **Pass the contrast and
covariates as** ``weight_design``: an empty design leaves the contrast's effect inside
``vb``, which *underestimates* ``f`` and shrinks the weights toward equal — the
conservative direction, never an over-correction. With ``method="median"`` and >= 3
runs the ``vt/n`` law is an approximation. Variance components always use unit *means*.

SCOPE (v0.1). One consensus weight per unit (not per feature). Sample exclusions (a run
that failed QC) are a separate, explicit decision — the loader's ``ReplicateCollapse``
is a keep-one *exclusion*, not an aggregation. Designs with a between-unit contrast
*and* repeated measures within a unit need a mixed model and are not served here.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from common.data_loading import LOG_SCALES, Dataset

__script_meta__: dict[str, object] = {
    "template": {"name": "aggregate-replicates", "version": "0.1"},
    "kind": "module",
    "provides": [
        "AggregateMethod",
        "N_REPLICATES",
        "PRECISION_WEIGHT",
        "ReplicateScaleError",
        "EmptyWeightDesignWarning",
        "WeightDesignCardinalityWarning",
        "ReplicateAggregation",
        "aggregate_replicates",
    ],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Verified Dataset replicate aggregation: one row per independent unit (mean or "
        "median of its runs on a log scale), unit-level metadata carried with a "
        "constant-within-unit check (an integrity check), run-level columns dropped or "
        "summarized (numeric mean / categorical fractions — technical replicates "
        "spanning batches), an n_replicates column, and limma-arrayWeights-style "
        "precision weights from a consensus technical-variance fraction. Hard-refuses "
        "linear/ratio scale; fail-loud; returns an independent Dataset plus a "
        "provenance record (run map, replicate-count histogram, technical-variance "
        "fraction)."
    ),
}

AggregateMethod = Literal["mean", "median"]

#: Metadata column holding each unit's run count (added by aggregation).
N_REPLICATES = "n_replicates"
#: Metadata column holding each unit's precision weight (mean 1).
PRECISION_WEIGHT = "precision_weight"

# Separator joining a multi-column unit key into the output index (display only —
# units are grouped on the column values themselves, never on the joined text).
_KEY_SEP = "|"
# A numeric weight_design column with at most this many distinct integer values is
# probably a coded factor (batch 1/2/3), mirroring the differential-abundance rule.
_LOW_CARDINALITY_NUMERIC = 5
# How many example units an error message lists.
_EXAMPLES = 5


class ReplicateScaleError(ValueError):
    """Raised when aggregation is requested on a non-log scale (``linear``/``ratio``).

    Aggregate after per-run normalization and logging: the log-space mean is the
    geometric mean of the runs. ``log2_transform`` only accepts linear input, so a
    ``ratio`` matrix needs a project-local log step first.
    """


class EmptyWeightDesignWarning(UserWarning):
    """Precision weights were estimated with an empty ``weight_design``.

    Without the contrast and covariates in the design, their effects stay inside the
    between-unit variance, which underestimates the technical-variance fraction and
    shrinks the weights toward equal (conservative, but weaker than it should be).
    Pass the contrast and covariates you will test.
    """


class WeightDesignCardinalityWarning(UserWarning):
    """A low-cardinality integer column in ``weight_design`` is used as a slope.

    The likely cause is a factor coded as integers (``batch = 1, 2, 3``). Pass it via
    ``categorical=`` to encode it as a factor.
    """


@dataclass(frozen=True)
class ReplicateAggregation:
    """An aggregated :class:`Dataset` (one row per unit) plus its provenance record.

    Attributes
    ----------
    dataset:
        The new :class:`Dataset`: one row per unit in first-appearance order, indexed by
        the unit key (multi-column keys joined with ``"|"``; index name likewise),
        on the input scale. Metadata carries the ``by`` columns and every unit-level
        column, the ``summarize`` outputs, :data:`N_REPLICATES` (int), and
        :data:`PRECISION_WEIGHT` (float, mean 1). Independent of the input.
    by, method, run_level, summarize, weight_design:
        The parameters used, echoed for the provenance record.
    n_runs_in, n_units_out:
        Row counts before (runs) and after (units) aggregation.
    replicate_counts:
        Histogram ``{str(n_runs): n_units}`` (string keys so it round-trips through the
        result cache).
    summarize_reference:
        For each categorical ``summarize`` column, the level whose fraction was dropped.
    summarize_columns:
        For each ``summarize`` column, the output column name(s) it became — pass these
        as ``covariates=`` to ``differential_abundance`` (a categorical column becomes
        its ``"{col}[frac {level}]"`` columns, not ``col``).
    run_map:
        One row per input run: ``sample`` (the input metadata index) → ``unit`` (the
        output key) — which runs were averaged into which unit.
    technical_variance_fraction:
        The consensus ``f = vt / (vb + vt)`` behind the weights; ``None`` when no
        unit has ≥ 2 runs (all weights are then 1).
    """

    dataset: Dataset
    by: tuple[str, ...]
    method: AggregateMethod
    n_runs_in: int
    n_units_out: int
    replicate_counts: dict[str, int]
    run_level: tuple[str, ...]
    summarize: tuple[str, ...]
    summarize_reference: dict[str, str]
    summarize_columns: dict[str, tuple[str, ...]]
    run_map: pd.DataFrame
    technical_variance_fraction: float | None
    weight_design: tuple[str, ...]


def aggregate_replicates(
    dataset: Dataset,
    by: str | Sequence[str],
    *,
    method: AggregateMethod = "mean",
    run_level: Sequence[str] = (),
    summarize: Sequence[str] = (),
    weight_design: Sequence[str] = (),
    categorical: Sequence[str] = (),
) -> ReplicateAggregation:
    """Average each unit's runs into one row and compute per-unit precision weights.

    Parameters
    ----------
    dataset:
        Input dataset on a log scale (``LOG_SCALES``), normalized per run, with missing
        values already resolved (a ``NaN``/``inf`` raises).
    by:
        The metadata column (or columns) identifying the unit. Several columns when
        technical replicates sit inside a within-unit design, e.g.
        ``("Mouse", "Timepoint")``.
    method:
        ``"mean"`` (default; the log-space mean, limma ``avereps``) or ``"median"``
        (differs from the mean only at ≥ 3 runs).
    run_level:
        Acquisition columns that differ between a unit's runs; dropped. Never a design
        variable.
    summarize:
        Run-level columns kept as unit-level covariates: numeric → unit mean (same
        name); categorical → ``k-1`` fraction columns ``"{col}[frac {level}]"``.
    weight_design:
        Output metadata columns (unit-level, or ``summarize`` columns) the precision-
        weight estimate adjusts for — pass the contrast and covariates you will test.
        Empty is allowed and conservative (weights shrink toward equal); it warns
        (:class:`EmptyWeightDesignWarning`) when run counts differ.
    categorical:
        ``weight_design`` columns to encode as factors even though they are numeric
        (the ``batch = 1/2/3`` case). A low-cardinality integer column not listed here
        is a slope, with a :class:`WeightDesignCardinalityWarning`.

    Returns
    -------
    ReplicateAggregation

    Raises
    ------
    ReplicateScaleError
        On a ``linear`` or ``ratio`` scale.
    ValueError
        On a malformed Dataset, non-finite abundances, a missing/NaN key, an unknown or
        overlapping column, a column that varies within a unit and is not declared, an
        existing ``n_replicates``/``precision_weight`` column, or a ``weight_design``
        that is rank-deficient or leaves no residual degrees of freedom.
    """
    by_cols = (by,) if isinstance(by, str) else tuple(by)
    run_level_cols = tuple(run_level)
    summarize_cols = tuple(summarize)
    weight_cols = tuple(weight_design)
    categorical_set = frozenset(categorical)
    if method not in ("mean", "median"):
        raise ValueError(f"Unknown method {method!r}; expected 'mean' or 'median'.")

    abundances = _validate(dataset)
    metadata = dataset.metadata
    _check_columns(metadata, by_cols, run_level_cols, summarize_cols)

    codes, unit_keys = _unit_codes(metadata, by_cols)
    n_units = len(unit_keys)
    counts = np.bincount(codes, minlength=n_units)

    carried = [
        c
        for c in metadata.columns
        if c not in run_level_cols and c not in summarize_cols
    ]
    _check_constant_within_units(metadata, codes, carried, by_cols, run_level_cols)

    means = _unit_means(abundances, codes, n_units, counts)
    values = means if method == "mean" else _unit_medians(abundances, codes, n_units)

    # Codes are numbered in first-appearance order: first indices come out sorted.
    _, first_rows = np.unique(codes, return_index=True)
    out_meta, summarize_reference, summarize_columns = _build_metadata(
        metadata, codes, n_units, first_rows, carried, summarize_cols
    )
    index_name = _KEY_SEP.join(by_cols)
    out_meta.index = pd.Index(unit_keys, name=index_name)
    out_meta[N_REPLICATES] = counts.astype(np.int64)

    if not weight_cols and np.unique(counts).size > 1:
        warnings.warn(
            "Precision weights estimated with an empty weight_design while run counts "
            "differ: the contrast's effect stays in the between-unit variance, which "
            "underestimates the technical-variance fraction and shrinks the weights "
            "toward equal. Pass weight_design=(<contrast>, *covariates).",
            EmptyWeightDesignWarning,
            stacklevel=2,
        )
    fraction = _technical_fraction(
        abundances,
        codes,
        means,
        counts,
        out_meta,
        weight_cols,
        summarize_cols,
        summarize_reference,
        categorical_set,
    )
    out_meta[PRECISION_WEIGHT] = _precision_weights(counts, fraction)

    new_dataset = replace(
        dataset,
        abundances=values,
        feature_names=np.asarray(dataset.feature_names).copy(),
        feature_metadata=dataset.feature_metadata.copy(),
        metadata=out_meta,
    )
    hist = np.bincount(counts)
    replicate_counts = {str(n): int(hist[n]) for n in range(hist.size) if hist[n] > 0}
    run_map = pd.DataFrame(
        {
            "sample": [str(s) for s in metadata.index],
            "unit": [unit_keys[c] for c in codes],
        }
    )
    return ReplicateAggregation(
        dataset=new_dataset,
        by=by_cols,
        method=method,
        n_runs_in=int(abundances.shape[0]),
        n_units_out=n_units,
        replicate_counts=replicate_counts,
        run_level=run_level_cols,
        summarize=summarize_cols,
        summarize_reference=summarize_reference,
        summarize_columns=summarize_columns,
        run_map=run_map,
        technical_variance_fraction=fraction,
        weight_design=weight_cols,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate(dataset: Dataset) -> np.ndarray:
    if dataset.scale not in LOG_SCALES:
        hint = (
            " A 'ratio' matrix needs a project-local log step first (log2_transform "
            "only accepts linear input)."
            if dataset.scale == "ratio"
            else " Normalize and log2_transform the per-run matrix first."
        )
        raise ReplicateScaleError(
            f"aggregate_replicates needs a log scale {sorted(LOG_SCALES)}; got "
            f"{dataset.scale!r}. The log-space mean is the geometric mean of the runs; "
            f"an arithmetic mean of intensities is pulled toward the larger run.{hint}"
        )
    abundances = np.asarray(dataset.abundances, dtype=float)
    if abundances.ndim != 2 or abundances.shape[0] == 0 or abundances.shape[1] == 0:
        raise ValueError(
            f"abundances must be a non-empty 2D array; got {abundances.shape}."
        )
    if len(dataset.metadata) != abundances.shape[0]:
        raise ValueError(
            f"metadata has {len(dataset.metadata)} rows but abundances has "
            f"{abundances.shape[0]} samples."
        )
    if len(dataset.feature_names) != abundances.shape[1]:
        raise ValueError(
            f"feature_names length ({len(dataset.feature_names)}) must equal "
            f"n_features ({abundances.shape[1]})."
        )
    if not np.all(np.isfinite(abundances)):
        raise ValueError(
            "abundances contain NaN/inf. Resolve missing values upstream (the "
            "missing-values template, before normalization) before aggregating."
        )
    return abundances


def _check_columns(
    metadata: pd.DataFrame,
    by_cols: tuple[str, ...],
    run_level_cols: tuple[str, ...],
    summarize_cols: tuple[str, ...],
) -> None:
    if not by_cols:
        raise ValueError("by must name at least one metadata column.")
    for label, cols in (
        ("by", by_cols),
        ("run_level", run_level_cols),
        ("summarize", summarize_cols),
    ):
        missing = [c for c in cols if c not in metadata.columns]
        if missing:
            raise ValueError(f"{label} column(s) not in metadata: {missing}.")
        dups = sorted({c for c in cols if cols.count(c) > 1})
        if dups:
            raise ValueError(f"{label} lists column(s) more than once: {dups}.")
    for a_label, a_cols, b_label, b_cols in (
        ("by", by_cols, "run_level", run_level_cols),
        ("by", by_cols, "summarize", summarize_cols),
        ("run_level", run_level_cols, "summarize", summarize_cols),
    ):
        overlap = sorted(set(a_cols) & set(b_cols))
        if overlap:
            raise ValueError(
                f"column(s) {overlap} listed in both {a_label} and {b_label}."
            )
    for reserved in (N_REPLICATES, PRECISION_WEIGHT):
        if reserved in metadata.columns:
            raise ValueError(
                f"metadata already has a {reserved!r} column — the input looks already "
                f"aggregated. Aggregate the per-run Dataset, once."
            )


def _unit_codes(
    metadata: pd.DataFrame, by_cols: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    """Unit code per row (first-appearance order) and each unit's display key.

    Units are grouped on the column **values** (so ``1`` and ``"1"``, or ``("x|y",
    "z")`` and ``("x", "y|z")``, stay distinct units); the ``"|"``-joined text is only
    the display key, and it must be one-to-one with the units or this raises rather
    than index two units under one label.
    """
    for col in by_cols:
        na = metadata[col].isna().to_numpy()
        if bool(na.any()):
            examples = [str(s) for s in metadata.index[na][:_EXAMPLES]]
            raise ValueError(
                f"unit column {col!r} is missing for {int(na.sum())} row(s) (e.g. "
                f"{examples}); every run must name its unit."
            )
    grouped = metadata.groupby(list(by_cols), sort=False, dropna=False, observed=True)
    raw = grouped.ngroup().to_numpy()
    codes_raw, _ = pd.factorize(raw, sort=False)  # renumber in first-appearance order
    codes = np.asarray(codes_raw, dtype=np.intp)
    _, first_rows = np.unique(codes, return_index=True)
    keys = [
        _KEY_SEP.join(str(metadata[c].iloc[int(r)]) for c in by_cols)
        for r in first_rows
    ]
    if len(set(keys)) < len(keys):
        seen: dict[str, int] = {}
        clashes: list[str] = []
        for code, key in enumerate(keys):
            if key in seen:
                clashes.append(key)
            seen.setdefault(key, code)
        raise ValueError(
            f"distinct units display as the same key "
            f"{sorted(set(clashes))[:_EXAMPLES]} "
            f"(values that differ only in type, e.g. 1 vs '1', or that contain "
            f"{_KEY_SEP!r} across columns of by={list(by_cols)}). Make the unit ids "
            f"unambiguous before aggregating."
        )
    return codes, keys


def _check_constant_within_units(
    metadata: pd.DataFrame,
    codes: np.ndarray,
    carried: list[str],
    by_cols: tuple[str, ...],
    run_level_cols: tuple[str, ...],
) -> None:
    grouped = metadata[carried].groupby(codes, sort=False)
    nunique = grouped.nunique(dropna=False)
    varying = [c for c in carried if c not in by_cols and bool((nunique[c] > 1).any())]
    if not varying:
        return
    bad_units = nunique.index[(nunique[varying] > 1).any(axis=1)]
    examples: list[str] = []
    for code in bad_units[:_EXAMPLES]:
        rows = np.flatnonzero(codes == code)
        key = _KEY_SEP.join(str(metadata[c].iloc[rows[0]]) for c in by_cols)
        examples.append(key)
    suggestion = repr((*run_level_cols, *varying))
    raise ValueError(
        f"column(s) {varying} vary within a unit (e.g. units {examples}). Declare "
        f"acquisition columns as run-level — run_level={suggestion} — or keep "
        f"one as a unit-level covariate with summarize=. If any of these is a design "
        f"variable, the unit is wrong (add it to by=) or the design is "
        f"repeated-measures (analyze with differential_abundance(unit=...) instead "
        f"of aggregating); if a unit's runs disagree on a design label, it is a "
        f"pairing error to fix upstream."
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _unit_means(
    abundances: np.ndarray, codes: np.ndarray, n_units: int, counts: np.ndarray
) -> np.ndarray:
    sums = np.zeros((n_units, abundances.shape[1]), dtype=float)
    np.add.at(sums, codes, abundances)
    # A one-run unit is its run, bit for bit: 0.0 + x and x / 1.0 are exact.
    return np.asarray(sums / counts[:, None], dtype=float)


def _unit_medians(
    abundances: np.ndarray, codes: np.ndarray, n_units: int
) -> np.ndarray:
    out = np.empty((n_units, abundances.shape[1]), dtype=float)
    for u in range(n_units):
        out[u] = np.median(abundances[codes == u], axis=0)
    return out


def _build_metadata(
    metadata: pd.DataFrame,
    codes: np.ndarray,
    n_units: int,
    first_rows: np.ndarray,
    carried: list[str],
    summarize_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, tuple[str, ...]]]:
    """Unit-level metadata in the input column order; summarize outputs in place."""
    summarize_reference: dict[str, str] = {}
    summarize_columns: dict[str, tuple[str, ...]] = {}
    pieces: dict[str, pd.Series] = {}
    first = metadata.iloc[first_rows].reset_index(drop=True)
    existing = set(metadata.columns)
    for col in metadata.columns:
        if col in carried:
            pieces[col] = first[col]
        elif col in summarize_cols:
            series = metadata[col]
            if bool(series.isna().any()):
                raise ValueError(
                    f"summarize column {col!r} has missing values; it must be defined "
                    f"for every run."
                )
            if pd.api.types.is_numeric_dtype(series):
                vals = pd.Series(series.to_numpy(dtype=float)).groupby(codes).mean()
                pieces[col] = pd.Series(vals.to_numpy(), dtype=float)
                summarize_columns[col] = (col,)
            else:
                labels = series.astype(str).to_numpy()
                levels = sorted(set(labels.tolist()))
                if len(levels) < 2:
                    raise ValueError(
                        f"summarize column {col!r} has a single level {levels}; it "
                        f"carries no information — drop it via run_level= instead."
                    )
                summarize_reference[col] = levels[0]
                counts = np.bincount(codes, minlength=n_units).astype(float)
                summarize_columns[col] = tuple(f"{col}[frac {lv}]" for lv in levels[1:])
                for level in levels[1:]:
                    name = f"{col}[frac {level}]"
                    if name in existing:
                        raise ValueError(
                            f"summarize output column {name!r} already exists in the "
                            f"metadata; rename the input column."
                        )
                    hits = np.bincount(
                        codes,
                        weights=(labels == level).astype(float),
                        minlength=n_units,
                    )
                    pieces[name] = pd.Series(hits / counts, dtype=float)
    return pd.DataFrame(pieces), summarize_reference, summarize_columns


# --------------------------------------------------------------------------- #
# Precision weights
# --------------------------------------------------------------------------- #
def _weight_design_matrix(
    out_meta: pd.DataFrame,
    weight_cols: tuple[str, ...],
    summarize_cols: tuple[str, ...],
    summarize_reference: dict[str, str],
    categorical_set: frozenset[str],
) -> np.ndarray:
    """Intercept + the weight_design columns encoded at the unit level."""
    n_units = len(out_meta)
    columns: list[np.ndarray] = [np.ones(n_units, dtype=float)]
    for col in weight_cols:
        if col in summarize_reference:  # categorical summarize → its fraction columns
            prefix = f"{col}[frac "
            for name in out_meta.columns:
                if str(name).startswith(prefix):
                    columns.append(out_meta[name].to_numpy(dtype=float))
            continue
        if col not in out_meta.columns or col in (N_REPLICATES, PRECISION_WEIGHT):
            raise ValueError(
                f"weight_design column {col!r} is not a unit-level column of the "
                f"aggregated metadata (run-level columns are dropped; use summarize= "
                f"to keep one)."
            )
        series = out_meta[col]
        if bool(series.isna().any()):
            raise ValueError(f"weight_design column {col!r} has missing values.")
        numeric = pd.api.types.is_numeric_dtype(series)
        if col not in summarize_cols and (col in categorical_set or not numeric):
            labels = series.astype(str).to_numpy()
            levels = sorted(set(labels.tolist()))
            columns.extend((labels == lvl).astype(float) for lvl in levels[1:])
            continue
        values = series.to_numpy(dtype=float)
        n_distinct = int(np.unique(values).size)
        if (
            col not in summarize_cols
            and n_distinct <= _LOW_CARDINALITY_NUMERIC
            and bool(np.all(values == np.round(values)))
        ):
            warnings.warn(
                f"weight_design column {col!r} has only {n_distinct} distinct integer "
                f"values and is used as a slope. If it is a coded factor (batch "
                f"1/2/3), pass categorical=[{col!r}].",
                WeightDesignCardinalityWarning,
                stacklevel=4,
            )
        columns.append(values)
    design = np.column_stack(columns)
    if int(np.linalg.matrix_rank(design)) < design.shape[1]:
        raise ValueError(
            f"weight_design {list(weight_cols)} is rank-deficient at the unit level "
            f"(aliased or constant columns); drop the redundant column."
        )
    if design.shape[0] - design.shape[1] <= 0:
        raise ValueError(
            f"weight_design leaves no residual degrees of freedom ({design.shape[0]} "
            f"units, {design.shape[1]} parameters); use fewer columns."
        )
    return design


def _technical_fraction(
    abundances: np.ndarray,
    codes: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    out_meta: pd.DataFrame,
    weight_cols: tuple[str, ...],
    summarize_cols: tuple[str, ...],
    summarize_reference: dict[str, str],
    categorical_set: frozenset[str],
) -> float | None:
    """Consensus ``vt / (vb + vt)`` across features, or ``None`` (no repeats)."""
    design = _weight_design_matrix(
        out_meta, weight_cols, summarize_cols, summarize_reference, categorical_set
    )
    n_rows, n_units = abundances.shape[0], means.shape[0]
    df_within = n_rows - n_units
    if df_within <= 0:
        return None

    resid_within = abundances - means[codes]
    s_t2 = np.sum(resid_within * resid_within, axis=0) / df_within

    xtx_inv = np.linalg.inv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    coef = xtx_inv @ design.T @ means
    resid = means - design @ coef
    rss = np.sum(resid * resid, axis=0)
    df_between = n_units - design.shape[1]
    tech_share = float(np.sum((1.0 - leverage) / counts))
    sigma_b2 = np.maximum(0.0, (rss - s_t2 * tech_share) / df_between)

    total = sigma_b2 + s_t2
    valid = np.isfinite(total) & (total > 0) & np.isfinite(s_t2)
    if not bool(valid.any()):
        return None
    fraction = float(np.median(s_t2[valid] / total[valid]))
    return float(min(1.0, max(0.0, fraction)))


def _precision_weights(counts: np.ndarray, fraction: float | None) -> np.ndarray:
    if fraction is None:
        return np.ones(counts.size, dtype=float)
    raw = 1.0 / ((1.0 - fraction) + fraction / counts.astype(float))
    return np.asarray(raw / raw.mean(), dtype=float)
