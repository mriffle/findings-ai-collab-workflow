"""Verified loader for wide omics matrices (feature x sample) + sample metadata.

TEMPLATE (lib/) — a *seed* for a project's ``common`` data loader, not a finished
script. Copy it into the project's ``scripts/`` and ADAPT the marked points for the
study. It is held to the correctness charter (conventions/correctness.md): **assume
nothing, verify everything, fail loud.**

Study-agnostic by design — it hardcodes no biology. The caller supplies the column
names; the loader returns the abundance matrix plus *whatever* metadata columns the
study's file carries.

ADAPT per project (pass as arguments — do not edit the body):
  * ``join_key`` / ``id_columns`` / ``feature_id_column`` — your files' column names.
  * ``strip_suffix`` — when the metadata join key carries a suffix the data headers
    don't (proteomics: metadata ``Replicate`` ends in ``.raw``; data headers don't).
  * ``collapse_replicates`` — collapse technical replicates to one row per sample.
  * ``numeric_columns`` — metadata columns to coerce to numeric (fail-loud).

DO IN THE PROJECT COPY, NOT HERE: sample exclusions and any relabeling/identity
corrections. Those are study-specific decisions and must live in the project as
explicit, reviewed steps — **never** as silent rewrites buried in a loader (that is
exactly the common-mode error the integrity gate exists to prevent). This template
deliberately omits them.

Scale & missingness: abundances are returned on the file's own scale (proteomics:
linear intensity, where ``0`` means "not detected"); zeros are preserved exactly.
The returned :class:`Dataset` carries a ``scale`` tag recording which scale the values
are on (``"linear"`` by default). The loader does *not* infer scale — that is a Stage-2
determination — it merely **records** what the scientist established, so the downstream
normalization / batch-correction templates can refuse scale-incorrect operations (e.g.
double-logging, or running ComBat on linear intensities) at the contract boundary.
NaN/missing values are governed by the ``missing`` policy — ``"error"`` (default)
refuses to load if any value is missing, ``"preserve"`` keeps them — and ``na_values``
declares which tokens count as missing.

Orientation is *inferred* (non-id columns are taken as samples and the matrix is
transposed) and validated only *transitively* by the sample<->metadata pairing check —
a transposed file fails there, but confirming orientation is a Stage-2 step, not the
loader's job. The loader likewise does NOT police domain fidelity: identifier corruption
(gene symbols turned to dates, accessions in sci-notation) and linear-vs-log scale
confusion are caught in Stages 1-2 and by the reviewers, not here.

Downstream contract: the returned :class:`Dataset` — not this file's input format — is
the stable interface the other templates (PCA, differential abundance, figures) consume.
A study whose files don't fit this shape should get its own loader (written and tested
in Stages 2-3, with this as a guide) that returns the same ``Dataset`` structure, so the
downstream templates compose unchanged. Adapt the input; keep the output structure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# The scale an abundance matrix is on. The contract that lets the normalization and
# batch-correction templates enforce scale-correctness (no double-log; ComBat only on a
# log-ish scale) instead of trusting a convention:
#   * "linear" — raw intensity (proteomics default); ``0`` = not detected.
#   * "log2"   — ``log2(x + 1)`` of a linear matrix.
#   * "log10"  — base-10 log of a linear matrix.
#   * "ln"     — natural log of a linear matrix.
#   * "glog2"  — VSN / arsinh variance-stabilized (log2-comparable; arsinh is ~natural
#                log asymptotically, so "log2-comparable" is a pronoms approximation).
#   * "zscore" — per-sample robust (MAD) z-score in log space; centred at 0.
#   * "ratio"  — relative / size-normalized linear-family values (e.g. SILAC/TMT
#                ratios); multiplicative, NOT a log scale.
# Add a member here (and to LOG_SCALES if it is log-ish) rather than mislabelling data —
# the guards make scale-dependent decisions, so an honest tag is load-bearing.
Scale = Literal["linear", "log2", "log10", "ln", "glog2", "zscore", "ratio"]

# The subset of scales on which ComBat (Gaussian-ish per feature) may run and on which a
# log-domain median *centering* is meaningful. Shared so the normalize and batch-correct
# templates agree on one taxonomy instead of each hardcoding a set.
LOG_SCALES: frozenset[Scale] = frozenset({"log2", "log10", "ln", "glog2", "zscore"})

__script_meta__: dict[str, object] = {
    "template": {"name": "wide-data-loader", "version": "0.4"},
    "kind": "module",
    "provides": [
        "Dataset",
        "Scale",
        "LOG_SCALES",
        "ReplicateCollapse",
        "load_wide_data",
        "load_precursor_data",
    ],
    "uses": [],
    "seeded_from": None,
    "description": (
        "Verified loader for wide feature x sample omics matrices + a sample-metadata "
        "table: orientation/pairing checks, optional technical-replicate collapse, "
        "precursor charge-state collapse, zero-preserving, fail-loud, scale-tagged. "
        "Study-agnostic (column names are arguments)."
    ),
}


@dataclass(frozen=True)
class ReplicateCollapse:
    """How to collapse technical replicates to one row per biological sample.

    Within each ``group_column`` (the biological-sample id), keep the single row
    with the highest ``rank_column`` value (e.g. the highest "Technical Replicate").
    ``rank_column`` must be numeric-coercible; ties keep the last in stable order.
    """

    group_column: str
    rank_column: str


@dataclass
class Dataset:
    """A wide omics dataset aligned to its sample metadata.

    Attributes
    ----------
    abundances:
        ``(n_samples, n_features)`` float array. Row ``i`` corresponds to
        ``metadata.iloc[i]``. On the file's scale (proteomics: linear intensity;
        ``0`` = not detected, preserved — never NaN).
    feature_names:
        ``(n_features,)`` str array — the chosen feature id column, in the same
        order as the columns of ``abundances``.
    feature_metadata:
        ``(n_features, ...)`` the data file's non-sample id columns, in feature
        order (e.g. ``protein``; or ``protein``/``modifiedSequence``/
        ``precursorCharge`` for precursors).
    metadata:
        ``(n_samples, ...)`` the kept sample-metadata rows, in ``abundances`` row
        order, indexed by the sample key (``join_key`` with ``strip_suffix``
        removed). Carries every column from the source metadata file.
    scale:
        Which scale ``abundances`` is on — ``"linear"`` (default), ``"log2"``,
        ``"glog2"``, or ``"zscore"`` (see :data:`Scale`). Stamped by the loader from
        the Stage-2 determination and updated by each transform (normalization,
        log2). Downstream templates read it to refuse scale-incorrect operations.
    """

    abundances: np.ndarray
    feature_names: np.ndarray
    feature_metadata: pd.DataFrame
    metadata: pd.DataFrame
    scale: Scale = "linear"


# ---------------------------------------------------------------------------
# Internal helpers (fail loud at every boundary)
# ---------------------------------------------------------------------------


def _strip_suffix(value: str, suffix: str | None) -> str:
    """Strip a single trailing ``suffix`` from ``value`` (the join transform)."""
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def _duplicates(items: Iterable[str]) -> list[str]:
    """Return the values that appear more than once in ``items`` (sorted, unique)."""
    seen: set[str] = set()
    dups: set[str] = set()
    for item in items:
        if item in seen:
            dups.add(item)
        seen.add(item)
    return sorted(dups)


def _read_header(data_file: Path, sep: str) -> list[str]:
    """Return the raw first-line column names, before pandas de-duplicates them."""
    with data_file.open() as handle:
        first_line = handle.readline()
    return first_line.rstrip("\r\n").split(sep)


def _read_wide(
    data_file: Path,
    id_columns: Sequence[str],
    sep: str,
    na_values: Sequence[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read a wide data file; return ``(frame, sample_columns)``.

    Sample columns are every column that is not an id column. Raises if a column header
    is duplicated, an id column is missing, or no sample columns remain.

    Duplicate headers are caught on the *raw* header line because ``pandas`` silently
    renames them (``sA`` -> ``sA``, ``sA.1``), which would otherwise slip a phantom
    sample column through to a confusing pairing mismatch instead of a clear error.
    """
    raw_header = _read_header(data_file, sep)
    duplicate_headers = _duplicates(raw_header)
    if duplicate_headers:
        raise ValueError(
            f"Duplicate column headers in data file {data_file}: {duplicate_headers}"
        )
    extra_na = list(na_values) if na_values is not None else None
    frame = pd.read_csv(data_file, sep=sep, na_values=extra_na)
    id_set = set(id_columns)
    missing = [c for c in id_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Data file {data_file} is missing id column(s): {missing}")
    sample_columns = [str(c) for c in frame.columns if c not in id_set]
    if not sample_columns:
        raise ValueError(
            f"Data file {data_file} has no sample columns after removing id columns "
            f"{list(id_columns)}"
        )
    return frame, sample_columns


def _load_metadata(
    metadata_file: Path, join_key: str, sep: str, numeric_columns: Sequence[str]
) -> pd.DataFrame:
    """Read the metadata table (all columns str — no silent coercion), validated.

    Raises if ``join_key`` is absent, has duplicate values, or a requested
    ``numeric_columns`` entry is missing or not numeric-coercible.
    """
    meta = pd.read_csv(metadata_file, sep=sep, dtype=str)
    if join_key not in meta.columns:
        raise ValueError(
            f"Metadata file {metadata_file} has no join_key column {join_key!r}"
        )
    duplicated = meta[join_key].duplicated(keep=False)
    if bool(duplicated.any()):
        dup_values = sorted(meta.loc[duplicated, join_key].unique())
        raise ValueError(f"Duplicate {join_key!r} values in metadata: {dup_values}")
    for col in numeric_columns:
        if col not in meta.columns:
            raise ValueError(
                f"numeric_columns references missing metadata column {col!r}"
            )
        meta[col] = pd.to_numeric(meta[col], errors="raise")
    return meta


def _collapse(meta: pd.DataFrame, collapse: ReplicateCollapse) -> pd.DataFrame:
    """Keep one row per ``group_column`` — the highest ``rank_column`` (stable)."""
    for col in (collapse.group_column, collapse.rank_column):
        if col not in meta.columns:
            raise ValueError(
                f"ReplicateCollapse references missing metadata column {col!r}"
            )
    rank = pd.to_numeric(meta[collapse.rank_column], errors="raise")
    ordered = meta.assign(_collapse_rank=rank).sort_values(
        [collapse.group_column, "_collapse_rank"], kind="stable"
    )
    kept = ordered.groupby(collapse.group_column, as_index=False, sort=False).tail(1)
    return kept.drop(columns="_collapse_rank")


def _order(meta: pd.DataFrame, order_by: str | None) -> pd.DataFrame:
    """Order rows by ``order_by`` (numeric if coercible, else lexical); stable."""
    if order_by is None:
        return meta.reset_index(drop=True)
    if order_by not in meta.columns:
        raise ValueError(f"order_by references missing metadata column {order_by!r}")
    try:
        key = pd.to_numeric(meta[order_by], errors="raise")
    except (ValueError, TypeError):
        key = meta[order_by]
    return (
        meta.assign(_order_key=key)
        .sort_values("_order_key", kind="stable")
        .drop(columns="_order_key")
        .reset_index(drop=True)
    )


def _check_pairing(
    sample_columns: Sequence[str], metadata_keys: Sequence[str], data_file: Path
) -> None:
    """Verify an exact 1:1 between data sample columns and metadata keys."""
    data_set = set(sample_columns)
    meta_set = set(metadata_keys)
    missing_in_data = sorted(meta_set - data_set)
    if missing_in_data:
        raise ValueError(
            f"Samples in metadata but not in data file {data_file}: {missing_in_data}"
        )
    extra_in_data = sorted(data_set - meta_set)
    if extra_in_data:
        raise ValueError(
            f"Sample columns in data file {data_file} not found in metadata: "
            f"{extra_in_data}"
        )


def _build_abundances(
    frame: pd.DataFrame,
    sample_keys: Sequence[str],
    missing: Literal["error", "preserve"],
    data_file: Path,
) -> np.ndarray:
    """Build the ``(n_samples, n_features)`` float matrix, enforcing the missing policy.

    Raises a clear error on non-numeric abundance, then applies ``missing``:
    ``"error"`` refuses to load if any value is NaN; ``"preserve"`` keeps NaN.
    """
    values = frame[list(sample_keys)]
    is_numeric = pd.api.types.is_numeric_dtype
    non_numeric = [c for c in sample_keys if not is_numeric(values[c])]
    if non_numeric:
        column = values[non_numeric[0]]
        coerced = pd.to_numeric(column, errors="coerce")
        offenders = column[coerced.isna() & column.notna()]
        example = offenders.iloc[0] if not offenders.empty else "<non-numeric>"
        raise ValueError(
            f"Non-numeric abundance in data file {data_file}, sample column "
            f"{non_numeric[0]!r} (e.g. {example!r}). Declare missing-value tokens via "
            f"na_values, or fix the data."
        )
    abundances = values.to_numpy(dtype=float).T
    if missing == "error":
        nan_count = int(np.isnan(abundances).sum())
        if nan_count:
            raise ValueError(
                f"{nan_count} missing value(s) (NaN) in abundances from {data_file}. "
                f"Pass missing='preserve' to keep them (declare tokens via na_values)."
            )
    return abundances


def _assemble(
    frame: pd.DataFrame,
    sample_columns: Sequence[str],
    feature_ids: pd.DataFrame,
    feature_id_column: str,
    *,
    metadata_file: Path,
    data_file: Path,
    join_key: str,
    strip_suffix: str | None,
    numeric_columns: Sequence[str],
    collapse_replicates: ReplicateCollapse | None,
    order_by: str | None,
    metadata_sep: str,
    missing: Literal["error", "preserve"],
    require_unique_features: bool,
    expected_n_samples: int | None,
    expected_n_features: int | None,
    scale: Scale,
) -> Dataset:
    """Align ``frame`` (features x samples) to its metadata; build a Dataset."""
    meta_full = _load_metadata(metadata_file, join_key, metadata_sep, numeric_columns)
    full_keys = [
        _strip_suffix(str(v), strip_suffix) for v in meta_full[join_key].tolist()
    ]
    collisions = _duplicates(full_keys)
    if collisions:
        raise ValueError(
            f"Metadata {join_key!r} values collide after stripping {strip_suffix!r}: "
            f"{collisions}"
        )
    _check_pairing(sample_columns, full_keys, data_file)

    kept = (
        meta_full
        if collapse_replicates is None
        else _collapse(meta_full, collapse_replicates)
    )
    kept = _order(kept, order_by)
    kept_keys = [_strip_suffix(str(v), strip_suffix) for v in kept[join_key].tolist()]

    if len(feature_ids) == 0:
        raise ValueError(f"Data file {data_file} has no feature rows.")

    abundances = _build_abundances(frame, kept_keys, missing, data_file)
    feature_names = feature_ids[feature_id_column].to_numpy(dtype=str)

    if require_unique_features:
        dup_features = _duplicates([str(x) for x in feature_names])
        if dup_features:
            raise ValueError(
                f"Duplicate feature ids in column {feature_id_column!r}: "
                f"{dup_features[:5]} (pass require_unique_features=False to allow)."
            )
    if expected_n_samples is not None and abundances.shape[0] != expected_n_samples:
        raise ValueError(
            f"Expected {expected_n_samples} samples but loaded {abundances.shape[0]}."
        )
    if expected_n_features is not None and abundances.shape[1] != expected_n_features:
        raise ValueError(
            f"Expected {expected_n_features} features but loaded {abundances.shape[1]}."
        )

    metadata = kept.copy()
    metadata.index = pd.Index(kept_keys, name=join_key)
    return Dataset(
        abundances=abundances,
        feature_names=feature_names,
        feature_metadata=feature_ids.reset_index(drop=True),
        metadata=metadata,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_wide_data(
    data_file: str | Path,
    metadata_file: str | Path,
    *,
    join_key: str,
    id_columns: Sequence[str] = ("protein",),
    feature_id_column: str | None = None,
    strip_suffix: str | None = None,
    numeric_columns: Sequence[str] = (),
    collapse_replicates: ReplicateCollapse | None = None,
    order_by: str | None = None,
    missing: Literal["error", "preserve"] = "error",
    na_values: Sequence[str] | None = None,
    require_unique_features: bool = True,
    expected_n_samples: int | None = None,
    expected_n_features: int | None = None,
    data_sep: str = "\t",
    metadata_sep: str = ",",
    scale: Scale = "linear",
) -> Dataset:
    """Load a wide feature x sample matrix and align it to a sample-metadata table.

    Parameters
    ----------
    data_file:
        Wide table: leading ``id_columns``, then one sample per column. Row=feature.
    metadata_file:
        Sample metadata: one row per sample, a ``join_key`` column matching the
        data sample-column headers (after ``strip_suffix``).
    join_key:
        Metadata column whose values match the data sample-column headers.
    id_columns:
        The leading non-sample columns of the data file (default ``("protein",)``).
    feature_id_column:
        Which id column names each feature; defaults to ``id_columns[0]``.
    strip_suffix:
        Suffix to strip from each ``join_key`` value before matching data headers
        (e.g. ``".raw"``). ``None`` = exact match.
    numeric_columns:
        Metadata columns to coerce to numeric (fail-loud).
    collapse_replicates:
        If given, collapse technical replicates to one row per biological sample.
    order_by:
        Metadata column to order samples by (numeric if coercible). ``None``
        keeps the data file's column order.
    missing:
        Missing-value policy: ``"error"`` (default) refuses to load if any abundance
        is NaN; ``"preserve"`` keeps NaN. Pair with ``na_values``.
    na_values:
        Extra tokens to read as missing (e.g. ``["Filtered"]``), added to pandas'
        defaults. A non-numeric value that is not declared missing is an error.
    require_unique_features:
        If ``True`` (default), raise on duplicate ``feature_id_column`` values.
    expected_n_samples, expected_n_features:
        Optional sanity check on the loaded shape; raise on mismatch.
    scale:
        The scale the data file is on, **recorded** on the returned ``Dataset`` (not
        inferred). Default ``"linear"`` (raw proteomics intensity). Set from the
        Stage-2 determination — e.g. ``"log2"`` if the file already ships log2 values.

    Returns
    -------
    Dataset
        Abundances ``(n_samples, n_features)`` row-aligned to ``metadata``; zeros
        preserved, missing values governed by ``missing``; tagged with ``scale``.
    """
    data_path = Path(data_file)
    metadata_path = Path(metadata_file)
    fid = feature_id_column if feature_id_column is not None else id_columns[0]
    if fid not in id_columns:
        raise ValueError(
            f"feature_id_column {fid!r} is not one of id_columns {list(id_columns)}"
        )

    frame, sample_columns = _read_wide(data_path, id_columns, data_sep, na_values)
    feature_ids = frame[list(id_columns)].reset_index(drop=True)
    return _assemble(
        frame,
        sample_columns,
        feature_ids,
        fid,
        metadata_file=metadata_path,
        data_file=data_path,
        join_key=join_key,
        strip_suffix=strip_suffix,
        numeric_columns=numeric_columns,
        collapse_replicates=collapse_replicates,
        order_by=order_by,
        metadata_sep=metadata_sep,
        missing=missing,
        require_unique_features=require_unique_features,
        expected_n_samples=expected_n_samples,
        expected_n_features=expected_n_features,
        scale=scale,
    )


def _collapse_charge_states(
    frame: pd.DataFrame,
    sample_columns: Sequence[str],
    protein_column: str,
    sequence_column: str,
    charge_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse a precursor table to one row per ``sequence_column``.

    For each modified sequence, keep the charge state with the highest
    mean-over-samples of the median-over-rows abundance (ties keep the lowest charge);
    then collapse rows that still share a sequence (peptides shared across proteins)
    into one — asserting those rows agree on sample values — and join the accessions.

    Returns ``(feature_ids, values)`` aligned row-for-row, in first-appearance order.
    """
    cols = list(sample_columns)
    median_by_pair = (
        frame.groupby([sequence_column, charge_column])[cols].median().mean(axis=1)
    )
    best_pair = median_by_pair.groupby(level=sequence_column).idxmax()
    best_charge = best_pair.map(lambda pair: pair[1])  # (seq, charge) -> charge
    keep_mask = frame[charge_column] == frame[sequence_column].map(best_charge)
    kept = frame[keep_mask]

    grouped = kept.groupby(sequence_column, sort=False)

    # Rows that share a sequence at the kept charge are the same precursor under
    # different proteins, so their sample values must match — assert it (fail loud)
    # rather than silently trusting the .first() below.
    multiplicity = grouped.size()
    shared = multiplicity.index[multiplicity.to_numpy() > 1]
    if len(shared) > 0:
        shared_rows = kept[kept[sequence_column].isin(shared)]
        distinct = shared_rows.groupby(sequence_column, sort=False)[cols].nunique(
            dropna=False
        )
        disagreeing = distinct.index[(distinct > 1).any(axis=1).to_numpy()].tolist()
        if disagreeing:
            raise ValueError(
                f"{len(disagreeing)} sequence(s) have charge-collapsed rows that "
                f"disagree on sample values (expected identical duplicates): "
                f"{disagreeing[:5]}"
            )

    values = grouped[cols].first()
    sequences = values.index.to_numpy(dtype=str)
    proteins = (
        grouped[protein_column]
        .agg(lambda s: ",".join(sorted(set(s.astype(str)))))
        .to_numpy(dtype=str)
    )
    charge_values = grouped[charge_column].first()
    charges_numeric = pd.to_numeric(charge_values, errors="coerce")
    if charges_numeric.isna().any():
        bad = charge_values[charges_numeric.isna()].unique().tolist()
        raise ValueError(
            f"Non-integer values in charge column {charge_column!r}: {bad[:5]}. "
            f"Charges must be plain integers (e.g. 2, 3); strip any suffix such as '+' "
            f"in the loader call before collapsing."
        )
    charges = charges_numeric.to_numpy(dtype=int)

    feature_ids = pd.DataFrame(
        {protein_column: proteins, sequence_column: sequences, charge_column: charges}
    )
    return feature_ids, values.reset_index(drop=True)


def load_precursor_data(
    data_file: str | Path,
    metadata_file: str | Path,
    *,
    join_key: str,
    protein_column: str = "protein",
    sequence_column: str = "modifiedSequence",
    charge_column: str = "precursorCharge",
    strip_suffix: str | None = None,
    numeric_columns: Sequence[str] = (),
    collapse_replicates: ReplicateCollapse | None = None,
    order_by: str | None = None,
    missing: Literal["error", "preserve"] = "error",
    na_values: Sequence[str] | None = None,
    require_unique_features: bool = True,
    expected_n_samples: int | None = None,
    expected_n_features: int | None = None,
    data_sep: str = "\t",
    metadata_sep: str = ",",
    scale: Scale = "linear",
) -> Dataset:
    """Load a precursor (peptide) matrix, collapsing charges to one row per sequence.

    Like :func:`load_wide_data`, but first reduces a DIA precursor table (multiple
    rows per ``sequence_column`` x ``charge_column``) to one row per modified
    sequence via :func:`_collapse_charge_states`. ``feature_names`` are the modified
    sequences; the joined protein accessions and chosen charge go in
    ``feature_metadata``. The ``missing`` / ``na_values`` / ``require_unique_features``
    / ``expected_*`` / ``scale`` controls behave as in :func:`load_wide_data`.
    """
    data_path = Path(data_file)
    metadata_path = Path(metadata_file)
    id_columns = (protein_column, sequence_column, charge_column)

    frame, sample_columns = _read_wide(data_path, id_columns, data_sep, na_values)
    feature_ids, values = _collapse_charge_states(
        frame, sample_columns, protein_column, sequence_column, charge_column
    )
    reduced = pd.concat([feature_ids, values], axis=1)
    return _assemble(
        reduced,
        sample_columns,
        feature_ids,
        sequence_column,
        metadata_file=metadata_path,
        data_file=data_path,
        join_key=join_key,
        strip_suffix=strip_suffix,
        numeric_columns=numeric_columns,
        collapse_replicates=collapse_replicates,
        order_by=order_by,
        metadata_sep=metadata_sep,
        missing=missing,
        require_unique_features=require_unique_features,
        expected_n_samples=expected_n_samples,
        expected_n_features=expected_n_features,
        scale=scale,
    )
