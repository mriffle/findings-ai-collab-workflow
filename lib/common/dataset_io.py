"""Persist a :class:`~common.data_loading.Dataset` to disk and reload it faithfully.

TEMPLATE (lib/) — a *seed* for a project's ``common`` dataset-io module, not a
finished script. Copy it into the project's ``scripts/`` and adapt only if a study
needs a different on-disk layout. Held to the correctness charter
(conventions/correctness.md): **assume nothing, verify everything, fail loud.**

Why this exists: the ``Dataset`` is an *in-memory* contract, but some steps need to
hand a prepared ``Dataset`` to a *separate process* — most importantly the **Stage-3
QC "prep-once"** pattern, where one step materializes the processing-state matrices
(raw / normalized / batch-corrected, each on the scale its plots consume) and each
context-isolated plot-family figure job **loads** the state it needs instead of
re-loading + re-normalizing + re-running ComBat. Crossing a process boundary means
serializing to disk; doing it by hand (a bare ``.npy`` + a CSV) drops the **scale
tag** and the shape/pairing invariants — exactly the footguns the ``Dataset``
contract prevents. So this round-trip is vetted, tag-preserving, and re-verifies the
contract on load.

Layout (a directory per ``Dataset``): ``abundances.npy`` (the float matrix),
``feature_metadata.parquet`` + ``metadata.parquet`` (the frames, dtypes + index
preserved), and ``dataset.json`` (the scale tag, feature names, shapes, format
version). Parquet needs no extra dependency — pandas (>= 3.0, the project baseline)
requires ``pyarrow``. Outputs go under ``results/`` (regenerable), never ``data/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd

from common.data_loading import Dataset, Scale

__script_meta__: dict[str, object] = {
    "template": {"name": "dataset-io", "version": "0.1"},
    "kind": "module",
    "provides": ["save_dataset", "load_dataset"],
    "uses": ["common.data_loading"],
    "seeded_from": None,
    "description": (
        "Faithful on-disk round-trip for the Dataset contract: save_dataset writes a "
        "directory (abundances.npy + metadata/feature_metadata parquet + a dataset.json"
        " sidecar carrying the scale tag, feature names, and shapes); load_dataset "
        "reads it back and re-verifies the shape/pairing invariants and a valid scale, "
        "fail-loud. Enables the Stage-3 QC prep-once pattern (materialize the "
        "processing-state matrices once; each isolated plot-family job loads what it "
        "needs). Parquet relies on pandas>=3.0's required pyarrow; no extra dependency."
    ),
}

_FORMAT_VERSION = 1
_ABUNDANCES = "abundances.npy"
_FEATURE_METADATA = "feature_metadata.parquet"
_METADATA = "metadata.parquet"
_SIDECAR = "dataset.json"

_VALID_SCALES: frozenset[str] = frozenset(get_args(Scale))


def save_dataset(dataset: Dataset, path: str | Path) -> Path:
    """Serialize ``dataset`` into directory ``path`` (created if needed); return it.

    Writes the four files described in the module docstring. Overwrites any existing
    files of the same names in ``path`` — the caller owns the directory. Round-trips
    losslessly through :func:`load_dataset`, including the ``scale`` tag, the metadata
    index, and every column dtype.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / _ABUNDANCES, np.asarray(dataset.abundances, dtype=float))
    dataset.feature_metadata.to_parquet(directory / _FEATURE_METADATA)
    dataset.metadata.to_parquet(directory / _METADATA)
    sidecar = {
        "format_version": _FORMAT_VERSION,
        "scale": dataset.scale,
        "feature_names": [str(name) for name in dataset.feature_names],
        "n_samples": int(dataset.abundances.shape[0]),
        "n_features": int(dataset.abundances.shape[1]),
    }
    (directory / _SIDECAR).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return directory


def load_dataset(path: str | Path) -> Dataset:
    """Reload a :class:`Dataset` from :func:`save_dataset`; re-verify the contract.

    Fails loud (``FileNotFoundError`` / ``ValueError``) on a missing artifact, an
    unknown format version, an invalid scale tag, or any shape/pairing mismatch — a
    corrupted or hand-edited directory must not silently yield a malformed ``Dataset``.
    """
    directory = Path(path)
    sidecar_path = directory / _SIDECAR
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"No Dataset sidecar at {sidecar_path}; not a save_dataset directory."
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    version = sidecar.get("format_version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Dataset format version {version!r} at {directory} is not supported "
            f"(this reader expects {_FORMAT_VERSION})."
        )
    scale = sidecar.get("scale")
    if scale not in _VALID_SCALES:
        raise ValueError(
            f"Dataset sidecar at {directory} has an invalid scale {scale!r}; "
            f"expected one of {sorted(_VALID_SCALES)}."
        )

    for name in (_ABUNDANCES, _FEATURE_METADATA, _METADATA):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing Dataset artifact {directory / name}.")

    abundances = np.asarray(np.load(directory / _ABUNDANCES), dtype=float)
    feature_metadata = pd.read_parquet(directory / _FEATURE_METADATA)
    metadata = pd.read_parquet(directory / _METADATA)
    feature_names = np.array(
        [str(name) for name in sidecar["feature_names"]], dtype=str
    )

    if abundances.ndim != 2:
        raise ValueError(
            f"abundances at {directory} must be 2-D, got shape {abundances.shape}."
        )
    n_samples, n_features = abundances.shape
    if (n_samples, n_features) != (sidecar["n_samples"], sidecar["n_features"]):
        raise ValueError(
            f"abundances shape {(n_samples, n_features)} disagrees with the sidecar "
            f"{(sidecar['n_samples'], sidecar['n_features'])} at {directory}."
        )
    if len(feature_names) != n_features:
        raise ValueError(
            f"{len(feature_names)} feature names but {n_features} abundance columns "
            f"at {directory}."
        )
    if len(feature_metadata) != n_features:
        raise ValueError(
            f"{len(feature_metadata)} feature_metadata rows but {n_features} abundance "
            f"columns at {directory}."
        )
    if len(metadata) != n_samples:
        raise ValueError(
            f"{len(metadata)} metadata rows but {n_samples} abundance rows at "
            f"{directory} (the sample pairing would be broken)."
        )

    return Dataset(
        abundances=abundances,
        feature_names=feature_names,
        feature_metadata=feature_metadata,
        metadata=metadata,
        scale=scale,
    )
