"""Persist an analysis **result** to disk and reload it — so a slow analysis is not
re-run just to tweak a figure.

TEMPLATE (lib/) — a *seed* for a project's result-caching module, not a finished
script. Copy it into the project's ``scripts/`` and adapt only if a study needs a
different layout. Held to the correctness charter (conventions/correctness.md):
**assume nothing, verify everything, fail loud.**

Why this exists: the CPU-heavy analysis templates (`classification`,
`classification-xgboost`, `regression`, `boruta`) return a **frozen dataclass** result
(nested CV, a stability loop, an opt-in label/target-shuffle **null** — the expensive
part) that the figure templates read via ``plot_*(result)``. With the result only in
memory, a naive *run-and-plot* script re-runs the whole analysis to change a figure
title. This module makes the result a **regenerable artifact on disk**: compute once,
persist, then re-render figures from the cache.

No pickle (the file-format convention + reproducibility): the result is decomposed into
its parts — ``pd.DataFrame`` -> parquet, ``np.ndarray`` -> ``.npy`` (numeric) or a JSON
list (string/object), nested dataclasses -> subdirectories, scalars/tuples/dicts -> a
JSON manifest. Reload is **type-directed** (the caller passes the result class), so no
arbitrary-class import happens. Parquet needs no extra dependency — pandas (>= 3.0, the
project baseline) requires ``pyarrow``.

Identity + caching: :func:`result_fingerprint` deterministically keys a result by
``(analysis, data_version, params, seed)`` so a re-run with identical inputs maps to the
same cache entry and any parameter change is a new one. :func:`save_cached_result`
writes the result + an immutable :class:`ResultMeta` sidecar under
``<cache_root>/<analysis>/<fingerprint>/``. The mutable *lifecycle* (which cached result
is `current`, which is superseded, pruning) lives in the project's
``results/manifest.md`` registry, not here (conventions/results-cache.md).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd

__script_meta__: dict[str, object] = {
    "template": {"name": "result-io", "version": "0.1"},
    "kind": "module",
    "provides": [
        "save_result",
        "load_result",
        "result_fingerprint",
        "ResultMeta",
        "save_cached_result",
        "load_cached_result",
        "read_result_meta",
    ],
    "uses": [],
    "seeded_from": None,
    "description": (
        "Faithful, pickle-free on-disk round-trip for a frozen-dataclass analysis "
        "result (DataFrame->parquet, ndarray->npy/json, nested dataclasses->subdirs, "
        "scalars->a JSON manifest); type-directed reload. Plus deterministic identity "
        "(result_fingerprint over analysis+data_version+params+seed) and a cache "
        "helper (save_cached_result / load_cached_result) writing an immutable "
        "ResultMeta sidecar, so a slow analysis is computed once and figures re-render "
        "from the cache. Parquet relies on pandas>=3.0's required pyarrow."
    ),
}

_FORMAT_VERSION = 1
_MANIFEST = "_result.json"
_META = "meta.json"

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
def save_result(result: object, path: str | Path) -> Path:
    """Serialize a frozen-dataclass ``result`` into directory ``path``; return it.

    Decomposes every field by type (see the module docstring) and writes a
    ``_result.json`` manifest describing how to reload it. Round-trips through
    :func:`load_result` given the same result class.
    """
    if not is_dataclass(result) or isinstance(result, type):
        raise TypeError(
            f"save_result expects a dataclass *instance*, got {type(result)!r}."
        )
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "dataclass": type(result).__qualname__,
        "fields": {
            f.name: _save_value(f.name, getattr(result, f.name), directory)
            for f in fields(result)
        },
    }
    (directory / _MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def _save_value(name: str, value: object, directory: Path) -> dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, pd.DataFrame):
        value.to_parquet(directory / f"{name}.parquet")
        return {"kind": "dataframe", "file": f"{name}.parquet"}
    if isinstance(value, np.ndarray):
        return _save_ndarray(name, value, directory)
    if is_dataclass(value) and not isinstance(value, type):
        save_result(value, directory / name)
        return {"kind": "dataclass", "dir": name}
    if isinstance(value, (list, tuple)):
        elements = list(value)
        if elements and all(
            is_dataclass(v) and not isinstance(v, type) for v in elements
        ):
            for i, element in enumerate(elements):
                save_result(element, directory / name / str(i))
            return {
                "kind": "list_dataclass",
                "dir": name,
                "count": len(elements),
                "container": "tuple" if isinstance(value, tuple) else "list",
            }
        return {
            "kind": "sequence",
            "container": "tuple" if isinstance(value, tuple) else "list",
            "value": [_scalarize(v) for v in elements],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "value": {str(k): _scalarize(v) for k, v in value.items()},
        }
    return {"kind": "scalar", "value": _scalarize(value)}


def _save_ndarray(name: str, arr: np.ndarray, directory: Path) -> dict[str, Any]:
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        # String / object arrays cannot go through np.save without allow_pickle
        # (a pickle path we refuse); store as a JSON value list instead.
        payload = {"shape": list(arr.shape), "values": arr.tolist()}
        (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        return {"kind": "ndarray_json", "file": f"{name}.json"}
    np.save(
        directory / f"{name}.npy", arr
    )  # numeric: exact, allow_pickle=False default
    return {"kind": "ndarray_npy", "file": f"{name}.npy"}


def _scalarize(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"result_io cannot serialize scalar of type {type(value)!r}; supported: "
        f"None/bool/int/float/str, ndarray, DataFrame, dataclass, and lists/tuples/"
        f"dicts of these."
    )


# --------------------------------------------------------------------------- #
# Load (type-directed by the caller's result class)
# --------------------------------------------------------------------------- #
def load_result(path: str | Path, cls: type[T]) -> T:
    """Reload a result written by :func:`save_result`, reconstructing ``cls``.

    Type-directed: field kinds come from the on-disk manifest, but nested dataclass
    types are resolved from ``cls``'s annotations (so no arbitrary class is imported).
    Fails loud on a missing manifest, an unknown format version, or a missing field.
    """
    directory = Path(path)
    manifest_path = directory / _MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No result manifest at {manifest_path}; not a save_result directory."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            f"Result format version {manifest.get('format_version')!r} at {directory} "
            f"is not supported (this reader expects {_FORMAT_VERSION})."
        )
    entries = manifest["fields"]
    hints = get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for field_info in fields(cls):  # type: ignore[arg-type]  # cls is a dataclass type
        if field_info.name not in entries:
            raise ValueError(
                f"Result at {directory} is missing field {field_info.name!r} "
                f"expected by {cls.__qualname__}."
            )
        kwargs[field_info.name] = _load_value(
            entries[field_info.name], directory, hints.get(field_info.name)
        )
    return cls(**kwargs)


def _load_value(entry: dict[str, Any], directory: Path, field_type: object) -> object:
    kind = entry["kind"]
    if kind == "none":
        return None
    if kind == "scalar":
        return entry["value"]
    if kind == "dict":
        return dict(entry["value"])
    if kind == "sequence":
        values = list(entry["value"])
        return tuple(values) if entry["container"] == "tuple" else values
    if kind == "dataframe":
        return pd.read_parquet(directory / entry["file"])
    if kind == "ndarray_npy":
        return np.load(directory / entry["file"])
    if kind == "ndarray_json":
        payload = json.loads((directory / entry["file"]).read_text(encoding="utf-8"))
        return np.asarray(payload["values"], dtype=object)
    if kind == "dataclass":
        return load_result(directory / entry["dir"], _dataclass_type(field_type))
    if kind == "list_dataclass":
        element_cls = _dataclass_type(field_type)
        items: list[object] = [
            load_result(directory / entry["dir"] / str(i), element_cls)
            for i in range(entry["count"])
        ]
        return tuple(items) if entry["container"] == "tuple" else items
    raise ValueError(f"Unknown serialized field kind {kind!r} at {directory}.")


def _dataclass_type(field_type: object) -> type:
    """Resolve the dataclass class carried by a field annotation.

    Handles a bare dataclass, ``list[X]`` / ``tuple[X, ...]``, and ``X | None``.
    """
    if field_type is None:
        raise ValueError(
            "Cannot reconstruct a dataclass field without its type hint on the result "
            "class (pass the correct class to load_result)."
        )
    candidates: list[type] = []
    if get_origin(field_type) is not None:  # list[X] / tuple[X, ...] / union (X | None)
        candidates = [
            a for a in get_args(field_type) if isinstance(a, type) and is_dataclass(a)
        ]
    elif isinstance(field_type, type) and is_dataclass(field_type):
        candidates = [field_type]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot resolve a single dataclass type from annotation {field_type!r}."
        )
    return candidates[0]


# --------------------------------------------------------------------------- #
# Identity + cache
# --------------------------------------------------------------------------- #
def result_fingerprint(
    *, analysis: str, data_version: str, params: Mapping[str, object], seed: int
) -> str:
    """Deterministic 12-hex id for a result, keyed by what produced it.

    Identical ``(analysis, data_version, params, seed)`` -> the same fingerprint (a
    re-run reuses the cache entry); any change (a different ``outcome``, features,
    ``run_null``, method, seed, or dataset version) -> a new one. ``params`` must be
    JSON-native (str/number/bool/None and lists/dicts of these) — it raises if not, so
    the identity can never silently depend on an unserialized object.
    """
    payload = {
        "analysis": analysis,
        "data_version": data_version,
        "params": dict(params),
        "seed": seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ResultMeta:
    """Immutable identity + provenance of a cached result (the ``meta.json`` sidecar).

    The *mutable* lifecycle — which cached result is `current`, which is superseded,
    pruning — lives in the project ``results/manifest.md`` registry, never here.
    """

    analysis: str
    data_version: str
    params: Mapping[str, object]
    seed: int
    fingerprint: str
    label: str | None = None
    created: str | None = None


def save_cached_result(
    result: object,
    *,
    cache_root: str | Path,
    analysis: str,
    data_version: str,
    params: Mapping[str, object],
    seed: int,
    label: str | None = None,
    created: str | None = None,
) -> ResultMeta:
    """Persist ``result`` to ``<cache_root>/<analysis>/<fingerprint>/`` (+ meta).

    Returns the :class:`ResultMeta` (with the fingerprint) so the caller can register
    it in ``results/manifest.md``. ``created`` is an optional ISO timestamp the caller
    supplies (kept an argument for reproducible tests rather than read from the clock).
    """
    fingerprint = result_fingerprint(
        analysis=analysis, data_version=data_version, params=params, seed=seed
    )
    directory = Path(cache_root) / analysis / fingerprint
    save_result(result, directory)
    meta = ResultMeta(
        analysis=analysis,
        data_version=data_version,
        params=params,
        seed=seed,
        fingerprint=fingerprint,
        label=label,
        created=created,
    )
    (directory / _META).write_text(
        json.dumps(
            {
                "analysis": meta.analysis,
                "data_version": meta.data_version,
                "params": dict(meta.params),
                "seed": meta.seed,
                "fingerprint": meta.fingerprint,
                "label": meta.label,
                "created": meta.created,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return meta


def read_result_meta(path: str | Path) -> ResultMeta:
    """Read the ``meta.json`` sidecar written by :func:`save_cached_result`."""
    meta_path = Path(path) / _META
    if not meta_path.is_file():
        raise FileNotFoundError(f"No result meta sidecar at {meta_path}.")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return ResultMeta(
        analysis=payload["analysis"],
        data_version=payload["data_version"],
        params=payload["params"],
        seed=payload["seed"],
        fingerprint=payload["fingerprint"],
        label=payload.get("label"),
        created=payload.get("created"),
    )


def load_cached_result(
    cls: type[T], *, cache_root: str | Path, analysis: str, fingerprint: str
) -> T:
    """Reload a cached result by its ``fingerprint`` (see :func:`load_result`)."""
    return load_result(Path(cache_root) / analysis / fingerprint, cls)
