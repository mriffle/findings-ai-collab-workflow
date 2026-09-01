"""Tests for the wide-data-loader template (lib/common/data_loading.py).

Two layers:
  * unit — hand-built synthetic fixtures, a planted-truth alignment check, and the
    integrity charter's failure modes (orientation, pairing, duplicates, missing cols);
  * smoke — the real 5xFAD data under testdata/ (git-ignored), reproducing the oracle
    ground truth. Skips cleanly when that data is absent, so the suite still passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common import data_loading as dl

# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #


def _write_protein(path: Path, frame: pd.DataFrame) -> Path:
    f = path / "data.tsv"
    frame.to_csv(f, sep="\t", index=False)
    return f


def _write_meta(path: Path, frame: pd.DataFrame) -> Path:
    f = path / "meta.csv"
    frame.to_csv(f, index=False)
    return f


@pytest.fixture
def protein_case(tmp_path: Path) -> tuple[Path, Path]:
    """A 3-feature x 3-sample protein matrix with a planted, asymmetric value pattern.

    Cell value = 100*feature_index + sample_number, so every cell is unique and the
    correct (sample, feature) placement after transpose+alignment is verifiable.
    """
    data = pd.DataFrame(
        {
            "protein": ["P1", "P2", "P3"],
            "sA": [100.0, 200.0, 300.0],  # feature base + sample 0
            "sB": [101.0, 201.0, 301.0],
            "sC": [102.0, 202.0, 302.0],
        }
    )
    meta = pd.DataFrame(
        {
            "Replicate": [
                "sB.raw",
                "sA.raw",
                "sC.raw",
            ],  # deliberately out of data order
            "RunOrder": ["2", "1", "3"],
            "Group": ["t", "c", "t"],
        }
    )
    return _write_protein(tmp_path, data), _write_meta(tmp_path, meta)


# --------------------------------------------------------------------------- #
# Unit — orientation, planted-truth alignment, zero preservation
# --------------------------------------------------------------------------- #


def test_orientation_is_samples_by_features(protein_case: tuple[Path, Path]) -> None:
    data_file, meta_file = protein_case
    ds = dl.load_wide_data(
        data_file, meta_file, join_key="Replicate", strip_suffix=".raw"
    )
    assert ds.abundances.shape == (
        3,
        3,
    )  # (n_samples, n_features), not the file's 3x3 transpose
    assert list(ds.feature_names) == ["P1", "P2", "P3"]


def test_planted_truth_alignment_with_reordering(
    protein_case: tuple[Path, Path],
) -> None:
    """The common-mode bug: a row landing against the wrong sample. Order by RunOrder
    (sA, sB, sC); each cell must equal 100*feature + sample_number."""
    data_file, meta_file = protein_case
    ds = dl.load_wide_data(
        data_file,
        meta_file,
        join_key="Replicate",
        strip_suffix=".raw",
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    assert list(ds.metadata.index) == ["sA", "sB", "sC"]
    expected = np.array(
        [
            [100.0, 200.0, 300.0],  # sample sA (number 0)
            [101.0, 201.0, 301.0],  # sample sB (number 1)
            [102.0, 202.0, 302.0],  # sample sC (number 2)
        ]
    )
    np.testing.assert_array_equal(ds.abundances, expected)
    # metadata travels with the right sample
    assert list(ds.metadata["Group"]) == ["c", "t", "t"]


def test_zeros_preserved_no_nan(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1", "P2"], "sA": [0.0, 5.0], "sB": [3.0, 0.0]})
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    ds = dl.load_wide_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
    )
    assert int((ds.abundances == 0).sum()) == 2
    assert not np.isnan(ds.abundances).any()


def test_metadata_carries_all_columns_row_aligned(
    protein_case: tuple[Path, Path],
) -> None:
    data_file, meta_file = protein_case
    ds = dl.load_wide_data(
        data_file, meta_file, join_key="Replicate", strip_suffix=".raw"
    )
    assert set(ds.metadata.columns) >= {"Replicate", "RunOrder", "Group"}
    # row i of abundances corresponds to row i of metadata
    for i, key in enumerate(ds.metadata.index):
        col_for_key = {"sA": 0, "sB": 1, "sC": 2}[str(key)]
        np.testing.assert_array_equal(
            ds.abundances[i], np.array([100.0, 200.0, 300.0]) + col_for_key
        )


# --------------------------------------------------------------------------- #
# Unit — scale tag (recorded, not inferred)
# --------------------------------------------------------------------------- #


def test_scale_defaults_to_linear(protein_case: tuple[Path, Path]) -> None:
    data_file, meta_file = protein_case
    ds = dl.load_wide_data(
        data_file, meta_file, join_key="Replicate", strip_suffix=".raw"
    )
    assert ds.scale == "linear"


def test_scale_is_recorded_when_given(protein_case: tuple[Path, Path]) -> None:
    """The loader records the Stage-2 scale verbatim; it does not transform values."""
    data_file, meta_file = protein_case
    ds = dl.load_wide_data(
        data_file, meta_file, join_key="Replicate", strip_suffix=".raw", scale="log2"
    )
    assert ds.scale == "log2"
    # recording the scale must not alter the abundances
    assert float(ds.abundances.max()) == 302.0


def test_scale_accepts_extended_log_scales(protein_case: tuple[Path, Path]) -> None:
    """The Scale taxonomy covers log10/ln/ratio, so such files get an honest tag."""
    data_file, meta_file = protein_case
    scales: tuple[dl.Scale, ...] = ("log10", "ln", "ratio")
    for sc in scales:
        ds = dl.load_wide_data(
            data_file, meta_file, join_key="Replicate", strip_suffix=".raw", scale=sc
        )
        assert ds.scale == sc


def test_precursor_scale_is_recorded(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "protein": ["PA"],
            "modifiedSequence": ["SEQ1"],
            "precursorCharge": [2],
            "sA": [10.0],
            "sB": [12.0],
        }
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    ds = dl.load_precursor_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
        scale="glog2",
    )
    assert ds.scale == "glog2"


# --------------------------------------------------------------------------- #
# Unit — technical-replicate collapse
# --------------------------------------------------------------------------- #


def test_collapse_keeps_highest_rank_per_group(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "protein": ["P1"],
            "m1_tr1": [10.0],
            "m1_tr2": [20.0],  # higher TR for sample m1 -> kept
            "m2_tr1": [30.0],
        }
    )
    meta = pd.DataFrame(
        {
            "Replicate": ["m1_tr1", "m1_tr2", "m2_tr1"],
            "SampleID": ["m1", "m1", "m2"],
            "TR": ["1", "2", "1"],
        }
    )
    ds = dl.load_wide_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
        collapse_replicates=dl.ReplicateCollapse("SampleID", "TR"),
        order_by="Replicate",
    )
    assert sorted(ds.metadata.index) == ["m1_tr2", "m2_tr1"]
    pairs = zip(ds.metadata.index, ds.abundances[:, 0], strict=True)
    kept = {str(k): float(v) for k, v in pairs}
    assert kept == {"m1_tr2": 20.0, "m2_tr1": 30.0}


# --------------------------------------------------------------------------- #
# Unit — fail-loud edge cases
# --------------------------------------------------------------------------- #


def test_raises_on_sample_in_data_missing_from_metadata(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0], "sB": [2.0], "sEXTRA": [3.0]})
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    with pytest.raises(ValueError, match="sEXTRA"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_raises_on_metadata_sample_missing_from_data(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0]})
    meta = pd.DataFrame({"Replicate": ["sA", "sMISSING"]})
    with pytest.raises(ValueError, match="sMISSING"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_raises_on_duplicate_join_key(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0]})
    meta = pd.DataFrame({"Replicate": ["sA", "sA"]})
    with pytest.raises(ValueError, match="Duplicate"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_raises_on_missing_join_key_column(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0]})
    meta = pd.DataFrame({"NotReplicate": ["sA"]})
    with pytest.raises(ValueError, match="join_key"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_raises_on_missing_id_column(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {"feat": ["P1"], "sA": [1.0]}
    )  # id col is 'feat', not 'protein'
    meta = pd.DataFrame({"Replicate": ["sA"]})
    with pytest.raises(ValueError, match="id column"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_raises_on_non_numeric_numeric_column(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0]})
    meta = pd.DataFrame({"Replicate": ["sA"], "RunOrder": ["not-a-number"]})
    with pytest.raises(ValueError):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
            numeric_columns=("RunOrder",),
        )


def test_single_sample_single_feature(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [42.0]})
    meta = pd.DataFrame({"Replicate": ["sA"]})
    ds = dl.load_wide_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
    )
    assert ds.abundances.shape == (1, 1)
    assert ds.abundances[0, 0] == 42.0


def test_raises_on_no_sample_columns(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1", "P2"]})  # id column only, no samples
    meta = pd.DataFrame({"Replicate": ["sA"]})
    with pytest.raises(ValueError, match="no sample columns"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_raises_on_duplicate_data_header(tmp_path: Path) -> None:
    # pandas would silently rename the second 'sA' to 'sA.1'; we must catch it.
    data = pd.DataFrame([["P1", 1.0, 2.0]], columns=["protein", "sA", "sA"])
    meta = pd.DataFrame({"Replicate": ["sA"]})
    with pytest.raises(ValueError, match="Duplicate column headers"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_raises_on_stripped_key_collision(tmp_path: Path) -> None:
    # "sA.raw" and "sA" are distinct raw keys but both strip to "sA".
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0]})
    meta = pd.DataFrame({"Replicate": ["sA.raw", "sA"]})
    with pytest.raises(ValueError, match="collide after stripping"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
            strip_suffix=".raw",
        )


def test_raises_on_feature_id_not_in_id_columns(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0]})
    meta = pd.DataFrame({"Replicate": ["sA"]})
    with pytest.raises(ValueError, match="feature_id_column"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
            feature_id_column="nope",
        )


def test_raises_on_collapse_missing_column(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0]})
    meta = pd.DataFrame({"Replicate": ["sA"]})
    with pytest.raises(ValueError, match="ReplicateCollapse references missing"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
            collapse_replicates=dl.ReplicateCollapse("NoGroup", "NoRank"),
        )


def test_raises_on_order_by_missing_column(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1"], "sA": [1.0]})
    meta = pd.DataFrame({"Replicate": ["sA"]})
    with pytest.raises(ValueError, match="order_by references missing"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
            order_by="NoSuchCol",
        )


# --------------------------------------------------------------------------- #
# Unit — identifier fidelity (id columns are never type-inferred)
#
# Regression suite for the defect where pandas' type inference silently rewrote
# feature ids on read: 0001 -> 1, 1001 -> 1001.0 (one decimal value anywhere in the
# column promotes the whole column to float), 1E5 -> 100000.0, TRUE -> True. The
# corrupted id lands in `feature_names` — the key every downstream result, figure,
# and finding is reported against — so these assert the on-disk bytes survive.
# Files are written raw (not via to_csv) so the fixture pins exactly what is read.
# --------------------------------------------------------------------------- #

_ID_META = "Replicate,Group\nsA,c\nsB,t\n"


def _write_raw(
    path: Path, data_text: str, meta_text: str = _ID_META
) -> tuple[Path, Path]:
    """Write a data.tsv / meta.csv pair verbatim; return (data, meta)."""
    data_file = path / "data.tsv"
    meta_file = path / "meta.csv"
    data_file.write_text(data_text)
    meta_file.write_text(meta_text)
    return data_file, meta_file


def test_numeric_looking_feature_ids_keep_their_exact_text(tmp_path: Path) -> None:
    """Leading zeros survive, and one decimal id does not float-promote its column."""
    data, meta = _write_raw(
        tmp_path,
        "protein\tsA\tsB\n0001\t10\t20\n1002\t30\t40\n1003.5\t50\t60\n",
    )
    ds = dl.load_wide_data(data, meta, join_key="Replicate")
    assert [str(x) for x in ds.feature_names] == ["0001", "1002", "1003.5"]


def test_bool_like_and_sci_notation_feature_ids_are_not_coerced(tmp_path: Path) -> None:
    data, meta = _write_raw(
        tmp_path,
        "protein\tsA\tsB\nTRUE\t10\t20\nFALSE\t30\t40\n1E5\t50\t60\n",
    )
    ds = dl.load_wide_data(data, meta, join_key="Replicate")
    assert [str(x) for x in ds.feature_names] == ["TRUE", "FALSE", "1E5"]


def test_every_id_column_keeps_string_fidelity_not_just_the_feature_id(
    tmp_path: Path,
) -> None:
    """The fidelity rule covers feature_metadata, not only feature_names."""
    data, meta = _write_raw(
        tmp_path,
        "protein\tgene_id\tsA\tsB\nP1\t0001\t10\t20\nP2\t1002\t30\t40\n",
    )
    ds = dl.load_wide_data(
        data, meta, join_key="Replicate", id_columns=("protein", "gene_id")
    )
    assert ds.feature_metadata["gene_id"].tolist() == ["0001", "1002"]


def test_numeric_ids_stay_distinct_after_the_str_read(tmp_path: Path) -> None:
    """Ids differing only in leading zeros must not collide into one feature."""
    data, meta = _write_raw(
        tmp_path,
        "protein\tsA\tsB\n0001\t10\t20\n001\t30\t40\n1\t50\t60\n",
    )
    ds = dl.load_wide_data(data, meta, join_key="Replicate")
    assert [str(x) for x in ds.feature_names] == ["0001", "001", "1"]


def test_raises_on_blank_feature_id(tmp_path: Path) -> None:
    data, meta = _write_raw(tmp_path, "protein\tsA\tsB\nP1\t10\t20\n\t30\t40\n")
    with pytest.raises(ValueError, match="Missing value"):
        dl.load_wide_data(data, meta, join_key="Replicate")


def test_raises_on_feature_id_spelling_an_na_token(tmp_path: Path) -> None:
    """An id read as NaN would otherwise become the literal feature name "nan"."""
    data, meta = _write_raw(tmp_path, "protein\tsA\tsB\nNA\t10\t20\nP1\t30\t40\n")
    with pytest.raises(ValueError, match="NA token"):
        dl.load_wide_data(data, meta, join_key="Replicate")


def test_precursor_ids_are_not_coerced_and_charges_still_parse(tmp_path: Path) -> None:
    """The precursor path keeps id text and still resolves charges to ints."""
    data, meta = _write_raw(
        tmp_path,
        "protein\tmodifiedSequence\tprecursorCharge\tsA\tsB\n"
        "0001\tPEPTIDEK\t2\t10\t20\n"
        "0001\tPEPTIDEK\t3\t1\t2\n"
        "1003\tELVISK\t2\t30\t40\n",
    )
    ds = dl.load_precursor_data(data, meta, join_key="Replicate")
    assert ds.feature_metadata["protein"].tolist() == ["0001", "1003"]
    assert ds.feature_metadata["precursorCharge"].tolist() == [2, 2]


# --------------------------------------------------------------------------- #
# Unit — precursor charge collapse
# --------------------------------------------------------------------------- #


def test_precursor_charge_selection_and_protein_join(tmp_path: Path) -> None:
    """SEQ1 has charges 2 and 3; charge 3 has the higher mean abundance -> kept. At the
    kept charge SEQ1 maps to two proteins (PA, PB) -> collapsed to one row with the
    accessions comma-joined; the charge-2 rows are dropped."""
    data = pd.DataFrame(
        {
            "protein": ["PA", "PB", "PA", "PB", "PC"],
            "modifiedSequence": ["SEQ1", "SEQ1", "SEQ1", "SEQ1", "SEQ2"],
            "precursorCharge": [2, 2, 3, 3, 2],
            "sA": [10.0, 10.0, 100.0, 100.0, 5.0],
            "sB": [12.0, 12.0, 120.0, 120.0, 6.0],
        }
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    ds = dl.load_precursor_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
    )
    assert sorted(ds.feature_names) == ["SEQ1", "SEQ2"]
    fm = ds.feature_metadata.set_index("modifiedSequence")
    assert fm.loc["SEQ1", "precursorCharge"] == 3  # higher-mean charge kept
    assert fm.loc["SEQ1", "protein"] == "PA,PB"  # both proteins kept-charge, joined
    # the kept SEQ1 row carries the charge-3 abundances (100/120), not charge-2 (10/12)
    seq1_col = list(ds.feature_names).index("SEQ1")
    np.testing.assert_array_equal(ds.abundances[:, seq1_col], np.array([100.0, 120.0]))


def test_precursor_raises_on_non_integer_charge(tmp_path: Path) -> None:
    """A charge label like '2+' must fail loud with a clear message, not a raw cast."""
    data = pd.DataFrame(
        {
            "protein": ["PA"],
            "modifiedSequence": ["SEQ1"],
            "precursorCharge": ["2+"],
            "sA": [10.0],
            "sB": [12.0],
        }
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    with pytest.raises(ValueError, match="Non-integer values in charge column"):
        dl.load_precursor_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_precursor_raises_when_shared_rows_disagree(tmp_path: Path) -> None:
    """Two proteins carry SEQ1 at the same (winning) charge but with different sample
    values — not the identical duplicates the collapse assumes. Must fail loud, not
    silently keep the first row (hardening of the .first() collapse)."""
    data = pd.DataFrame(
        {
            "protein": ["PA", "PB"],
            "modifiedSequence": ["SEQ1", "SEQ1"],
            "precursorCharge": [2, 2],
            "sA": [10.0, 999.0],  # the two rows disagree on sample sA
            "sB": [12.0, 12.0],
        }
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    with pytest.raises(ValueError, match="disagree"):
        dl.load_precursor_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


# --------------------------------------------------------------------------- #
# Unit — missing values, non-numeric, unique features, shape, charge ties
# --------------------------------------------------------------------------- #


def test_missing_default_errors_on_nan(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {"protein": ["P1", "P2"], "sA": [1.0, np.nan], "sB": [3.0, 4.0]}
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    with pytest.raises(ValueError, match="missing value"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_missing_preserve_keeps_nan(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {"protein": ["P1", "P2"], "sA": [1.0, np.nan], "sB": [3.0, 4.0]}
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    ds = dl.load_wide_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
        missing="preserve",
    )
    assert int(np.isnan(ds.abundances).sum()) == 1


def test_na_values_declares_missing_token(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {"protein": ["P1", "P2"], "sA": ["1.0", "Filtered"], "sB": ["3.0", "4.0"]}
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    ds = dl.load_wide_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
        na_values=["Filtered"],
        missing="preserve",
    )
    assert int(np.isnan(ds.abundances).sum()) == 1  # "Filtered" -> NaN; rest numeric


def test_raises_on_non_numeric_abundance(tmp_path: Path) -> None:
    # 'Filtered' is NOT declared missing here -> genuine non-numeric junk, clear error.
    data = pd.DataFrame(
        {"protein": ["P1", "P2"], "sA": ["1.0", "Filtered"], "sB": ["3.0", "4.0"]}
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    with pytest.raises(ValueError, match="Non-numeric abundance"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_duplicate_features_raise_by_default(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1", "P1"], "sA": [1.0, 2.0]})
    meta = pd.DataFrame({"Replicate": ["sA"]})
    with pytest.raises(ValueError, match="Duplicate feature ids"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_duplicate_features_allowed_when_opted_out(tmp_path: Path) -> None:
    data = pd.DataFrame({"protein": ["P1", "P1"], "sA": [1.0, 2.0]})
    meta = pd.DataFrame({"Replicate": ["sA"]})
    ds = dl.load_wide_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
        require_unique_features=False,
    )
    assert list(ds.feature_names) == ["P1", "P1"]
    assert ds.abundances.shape == (1, 2)


def test_raises_on_no_feature_rows(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {"protein": pd.Series([], dtype=str), "sA": pd.Series([], dtype=float)}
    )
    meta = pd.DataFrame({"Replicate": ["sA"]})
    with pytest.raises(ValueError, match="no feature rows"):
        dl.load_wide_data(
            _write_protein(tmp_path, data),
            _write_meta(tmp_path, meta),
            join_key="Replicate",
        )


def test_expected_shape_mismatch_raises(protein_case: tuple[Path, Path]) -> None:
    data_file, meta_file = protein_case  # 3 samples x 3 features
    with pytest.raises(ValueError, match="Expected 5 samples"):
        dl.load_wide_data(
            data_file,
            meta_file,
            join_key="Replicate",
            strip_suffix=".raw",
            expected_n_samples=5,
        )
    ds = dl.load_wide_data(
        data_file,
        meta_file,
        join_key="Replicate",
        strip_suffix=".raw",
        expected_n_samples=3,
        expected_n_features=3,
    )
    assert ds.abundances.shape == (3, 3)


def test_charge_tie_keeps_lowest_charge(tmp_path: Path) -> None:
    # SEQ1 at charge 2 and 3 have equal mean abundance -> the lower charge (2) is kept.
    data = pd.DataFrame(
        {
            "protein": ["PA", "PA"],
            "modifiedSequence": ["SEQ1", "SEQ1"],
            "precursorCharge": [2, 3],
            "sA": [10.0, 10.0],
            "sB": [20.0, 20.0],
        }
    )
    meta = pd.DataFrame({"Replicate": ["sA", "sB"]})
    ds = dl.load_precursor_data(
        _write_protein(tmp_path, data),
        _write_meta(tmp_path, meta),
        join_key="Replicate",
    )
    fm = ds.feature_metadata.set_index("modifiedSequence")
    assert fm.loc["SEQ1", "precursorCharge"] == 2


# --------------------------------------------------------------------------- #
# Smoke — real 5xFAD data (git-ignored); reproduces the oracle ground truth
# --------------------------------------------------------------------------- #

_TESTDATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _TESTDATA / "data" / "proteins_wide_unnormalized.tsv"
_PREC = _TESTDATA / "data" / "precursors_wide_unnormalized.tsv"
_META = _TESTDATA / "metadata" / "Replicates_5xFAD.csv"

_COLLAPSE = dl.ReplicateCollapse("Sample ID", "Technical Replicate")
_skip_no_data = pytest.mark.skipif(
    not (_PROT.exists() and _META.exists()),
    reason="testdata/5xFAD not present (git-ignored)",
)


@_skip_no_data
def test_smoke_protein_matches_oracle() -> None:
    ds = dl.load_wide_data(
        _PROT,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=_COLLAPSE,
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    assert ds.abundances.shape == (61, 8829)  # oracle ground truth
    assert ds.abundances.dtype == np.float64
    assert not np.isnan(ds.abundances).any()  # zeros preserved, never NaN
    assert int((ds.abundances == 0).sum()) == 25556
    # spot-reconcile one cell against the raw file
    raw = pd.read_csv(_PROT, sep="\t")
    key = str(ds.metadata.index[3])
    raw_val = float(raw.loc[raw["protein"] == ds.feature_names[0], key].iloc[0])
    assert float(ds.abundances[3, 0]) == raw_val


@_skip_no_data
@pytest.mark.slow
def test_smoke_precursor_matches_oracle() -> None:
    ds = dl.load_precursor_data(
        _PREC,
        _META,
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=_COLLAPSE,
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    assert ds.abundances.shape == (61, 74340)  # oracle ground truth
    assert len(set(ds.feature_names.tolist())) == len(
        ds.feature_names
    )  # unique sequences
    assert set(np.unique(ds.feature_metadata["precursorCharge"]).tolist()) <= {2, 3}
    assert not np.isnan(ds.abundances).any()
