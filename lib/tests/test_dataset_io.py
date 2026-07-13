"""Tests for the Dataset on-disk round-trip (common/dataset_io.py).

  * round-trip — save then load must reproduce the Dataset exactly (values, dtypes,
    index, feature names, and the scale tag);
  * fail-loud — a missing artifact, unknown format version, invalid scale, or any
    shape/pairing mismatch must raise, never yield a malformed Dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from common import data_loading as dl
from common import dataset_io as dsio


def _ds(scale: dl.Scale = "linear") -> dl.Dataset:
    """A small Dataset with mixed-dtype metadata and a named sample index."""
    abundances = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    feature_names = np.array(["P1", "P2", "P3"], dtype=str)
    feature_metadata = pd.DataFrame(
        {"protein": ["P1", "P2", "P3"], "length": [10, 20, 30]}
    )
    metadata = pd.DataFrame(
        {"group": ["A", "B"], "age": [5.0, 7.0], "batch": [1, 2]},
        index=pd.Index(["s0", "s1"], name="sample"),
    )
    return dl.Dataset(
        abundances=abundances,
        feature_names=feature_names,
        feature_metadata=feature_metadata,
        metadata=metadata,
        scale=scale,
    )


def test_round_trip_preserves_everything(tmp_path: Path) -> None:
    ds = _ds()
    dsio.save_dataset(ds, tmp_path / "ds")
    out = dsio.load_dataset(tmp_path / "ds")
    np.testing.assert_allclose(out.abundances, ds.abundances)
    np.testing.assert_array_equal(out.feature_names, ds.feature_names)
    assert out.scale == ds.scale
    pd.testing.assert_frame_equal(out.metadata, ds.metadata)
    pd.testing.assert_frame_equal(out.feature_metadata, ds.feature_metadata)


@pytest.mark.parametrize("scale", ["linear", "log2", "glog2", "zscore"])
def test_round_trip_preserves_scale_tag(tmp_path: Path, scale: dl.Scale) -> None:
    dsio.save_dataset(_ds(scale), tmp_path / "ds")
    assert dsio.load_dataset(tmp_path / "ds").scale == scale


def test_save_creates_nested_directory_and_returns_it(tmp_path: Path) -> None:
    target = tmp_path / "results" / "qc_states" / "raw_linear"
    returned = dsio.save_dataset(_ds(), target)
    assert returned == target
    assert (target / "dataset.json").is_file()


def test_load_returns_independent_object(tmp_path: Path) -> None:
    ds = _ds()
    dsio.save_dataset(ds, tmp_path / "ds")
    out = dsio.load_dataset(tmp_path / "ds")
    assert out is not ds
    out.metadata["injected"] = 1
    assert "injected" not in ds.metadata.columns


def test_load_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a save_dataset directory"):
        dsio.load_dataset(tmp_path / "nope")


def test_load_bad_format_version_raises(tmp_path: Path) -> None:
    dsio.save_dataset(_ds(), tmp_path / "ds")
    sidecar = tmp_path / "ds" / "dataset.json"
    payload = json.loads(sidecar.read_text())
    payload["format_version"] = 999
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="format version"):
        dsio.load_dataset(tmp_path / "ds")


def test_load_invalid_scale_raises(tmp_path: Path) -> None:
    dsio.save_dataset(_ds(), tmp_path / "ds")
    sidecar = tmp_path / "ds" / "dataset.json"
    payload = json.loads(sidecar.read_text())
    payload["scale"] = "bananas"
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid scale"):
        dsio.load_dataset(tmp_path / "ds")


def test_load_feature_name_count_mismatch_raises(tmp_path: Path) -> None:
    dsio.save_dataset(_ds(), tmp_path / "ds")
    sidecar = tmp_path / "ds" / "dataset.json"
    payload = json.loads(sidecar.read_text())
    payload["feature_names"] = payload["feature_names"][:-1]  # drop one
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="feature names but"):
        dsio.load_dataset(tmp_path / "ds")


def test_load_missing_artifact_raises(tmp_path: Path) -> None:
    dsio.save_dataset(_ds(), tmp_path / "ds")
    (tmp_path / "ds" / "abundances.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Missing Dataset artifact"):
        dsio.load_dataset(tmp_path / "ds")
