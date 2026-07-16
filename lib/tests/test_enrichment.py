"""Tests for the enrichment template (lib/analysis/enrichment.py).

Layers:
  * parser — planted-truth accession extraction for every encoding form (sp|/tr|/bare/
    isoform/protein-group/contaminant), hand-verified;
  * audit — build_mapping_audit rows, flags, and the unmappable source;
  * wiring — enrich() with an INJECTED transport (no network): the request payload shape
    (custom domain scope, sources, correction, background), response parsing, version +
    recognized-count capture, the significant_table, and all_results;
  * guards — empty query/background, bad correction/threshold, query-not-subset-of-
    background, zero/low mapping, and the EnrichmentServiceError paths;
  * bridge — enrich_from_differential_abundance direction split + term selection;
  * smoke — a NETWORK-GATED real-5xFAD call (skipped offline / without testdata).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from analysis import differential_abundance as da
from analysis import enrichment as en
from common import data_loading as dl

# --------------------------------------------------------------------------- #
# A canned g:Profiler response + an injectable transport that records the payload
# --------------------------------------------------------------------------- #
_CANNED_RESULT = [
    {
        "source": "GO:BP",
        "native": "GO:0006955",
        "name": "immune response",
        "p_value": 1e-5,
        "significant": True,
        "term_size": 200,
        "query_size": 10,
        "intersection_size": 8,
        "effective_domain_size": 8000,
    },
    {
        "source": "KEGG",
        "native": "KEGG:04610",
        "name": "complement cascade",
        "p_value": 0.30,
        "significant": False,
        "term_size": 50,
        "query_size": 10,
        "intersection_size": 2,
        "effective_domain_size": 8000,
    },
]


class RecordingTransport:
    """A fake transport that records the last payload and returns a canned response."""

    def __init__(
        self, result: list[dict[str, Any]] | None = None, version: str = "eTEST_1"
    ) -> None:
        self.result = _CANNED_RESULT if result is None else result
        self.version = version
        self.payload: dict[str, Any] | None = None

    def __call__(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        return {
            "meta": {
                "version": self.version,
                "genes_metadata": {"failed": ["Q0FAKE0"]},
            },
            "result": self.result,
        }


def _valid_query() -> list[str]:
    return ["sp|P05067|A4_MOUSE", "tr|A0A075B5J9|X_MOUSE", "sp|Q9DBR1|Y_MOUSE"]


def _valid_background() -> list[str]:
    return [
        *_valid_query(),
        "sp|P68871|HBB_MOUSE",
        "sp|P0DTC2|Z_MOUSE",
        "crapola_crap|ALBU_BOVIN|ALBU_BOVIN",
    ]


# --------------------------------------------------------------------------- #
# Parser — planted truth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sp|A1L3T7|RIPR3_MOUSE", "A1L3T7"),
        ("tr|A0A075B5J9|A0A075B5J9_MOUSE", "A0A075B5J9"),
        ("P05067", "P05067"),
        ("P05067-2", "P05067"),  # isoform stripped
        ("sp|P12345|X;sp|Q9DBR1|Y", "P12345"),  # group -> first mappable
        ("crapola_crap|ALBU_BOVIN|ALBU_BOVIN", None),  # contaminant mnemonic
        ("crapola_crap|P22629|SAV_STRAV", "P22629"),  # accession-in-contaminant
        ("not_an_id", None),
        ("", None),
    ],
)
def test_parse_uniprot_accession(raw: str, expected: str | None) -> None:
    assert en.parse_uniprot_accession(raw) == expected


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def test_build_mapping_audit_rows_and_flags() -> None:
    query = ["sp|P05067|A", "crapola_crap|ALBU_BOVIN|ALBU_BOVIN"]
    background = ["sp|P05067|A", "sp|Q9DBR1|B"]
    audit = en.build_mapping_audit(query, background)
    assert set(audit.columns) == {
        "original_id",
        "uniprot_accession",
        "source",
        "in_query",
        "in_background",
    }
    assert len(audit) == 3  # three distinct ids
    row = audit[audit["original_id"] == "sp|P05067|A"].iloc[0]
    assert row["uniprot_accession"] == "P05067"
    assert bool(row["in_query"]) and bool(row["in_background"])
    contaminant = audit[audit["source"] == "unmappable"].iloc[0]
    assert contaminant["uniprot_accession"] == ""


def test_audit_source_labels() -> None:
    audit = en.build_mapping_audit(
        ["sp|P05067|A", "tr|A0A075B5J9|B", "P0DTC2"], ["sp|P05067|A"]
    )
    by_id = {r["original_id"]: r["source"] for _, r in audit.iterrows()}
    assert by_id["sp|P05067|A"] == "swissprot"
    assert by_id["tr|A0A075B5J9|B"] == "trembl"
    assert by_id["P0DTC2"] == "other"


# --------------------------------------------------------------------------- #
# Wiring — injected transport
# --------------------------------------------------------------------------- #
def test_enrich_payload_shape() -> None:
    tr = RecordingTransport()
    en.enrich(_valid_query(), _valid_background(), "mmusculus", transport=tr)
    assert tr.payload is not None
    assert tr.payload["organism"] == "mmusculus"
    assert tr.payload["domain_scope"] == "custom"
    assert tr.payload["significance_threshold_method"] == "g_SCS"
    assert tr.payload["sources"] == list(en.DEFAULT_SOURCES)
    # query accessions are a subset of the background accessions sent.
    assert set(tr.payload["query"]) <= set(tr.payload["background"])
    assert "P05067" in tr.payload["query"]


def test_enrich_parses_table_and_meta() -> None:
    tr = RecordingTransport()
    res = en.enrich(_valid_query(), _valid_background(), "mmusculus", transport=tr)
    assert res.gprofiler_version == "eTEST_1"
    assert res.n_query_mapped == 3
    assert res.n_query_recognized == 3  # none of the 3 are in the failed list
    assert res.effective_domain_size == 8000
    assert len(res.table) == 2
    assert len(res.significant_table) == 1
    top = res.significant_table.iloc[0]
    assert top["term_id"] == "GO:0006955"
    assert top["gene_ratio"] == pytest.approx(0.8)  # 8 / 10
    assert top["recall"] == pytest.approx(0.04)  # 8 / 200
    # The canned response carries no evidences/gene-map, so members are empty.
    assert top["intersecting_genes"] == ""


def test_intersecting_genes_recovered_as_input_ids() -> None:
    """The per-term member list maps g:Profiler's ENSGs back to the caller's accessions.

    ``intersections`` aligns to the recognized-gene order (``ensgs``); a non-empty entry
    means that gene is in the term. Inverting ``mapping`` (input id -> ENSG) recovers
    caller's own ids, in ``ensgs`` order.
    """

    def transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "meta": {
                "version": "eTEST_1",
                "genes_metadata": {
                    "failed": [],
                    "query": {
                        "query_1": {
                            "ensgs": ["ENS1", "ENS2", "ENS3"],
                            "mapping": {
                                "P05067": ["ENS1"],
                                "A0A075B5J9": ["ENS2"],
                                "Q9DBR1": ["ENS3"],
                            },
                        }
                    },
                },
            },
            "result": [
                {
                    "source": "GO:BP",
                    "native": "GO:1",
                    "name": "term one",
                    "p_value": 1e-4,
                    "significant": True,
                    "term_size": 100,
                    "query_size": 3,
                    "intersection_size": 2,
                    "effective_domain_size": 8000,
                    # ENS1 in, ENS2 out, ENS3 in.
                    "intersections": [["IEA"], [], ["IEA"]],
                }
            ],
        }

    res = en.enrich(
        _valid_query(), _valid_background(), "mmusculus", transport=transport
    )
    assert res.table.iloc[0]["intersecting_genes"] == "P05067;Q9DBR1"


def test_intersecting_genes_empty_on_length_mismatch() -> None:
    """A malformed intersections array (wrong length) leaves the member list empty."""

    def transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "meta": {
                "version": "eTEST_1",
                "genes_metadata": {
                    "query": {"query_1": {"ensgs": ["ENS1", "ENS2"], "mapping": {}}}
                },
            },
            "result": [
                {
                    "source": "GO:BP",
                    "native": "GO:1",
                    "name": "term one",
                    "p_value": 1e-4,
                    "significant": True,
                    "term_size": 100,
                    "query_size": 3,
                    "intersection_size": 1,
                    "effective_domain_size": 8000,
                    "intersections": [["IEA"]],  # length 1 != len(ensgs) 2
                }
            ],
        }

    res = en.enrich(
        _valid_query(), _valid_background(), "mmusculus", transport=transport
    )
    assert res.table.iloc[0]["intersecting_genes"] == ""


def test_enrich_correction_threaded() -> None:
    tr = RecordingTransport()
    en.enrich(
        _valid_query(), _valid_background(), "mmusculus", correction="fdr", transport=tr
    )
    assert tr.payload is not None
    assert tr.payload["significance_threshold_method"] == "fdr"


def test_all_results_flag_recorded() -> None:
    tr = RecordingTransport()
    res = en.enrich(
        _valid_query(),
        _valid_background(),
        "mmusculus",
        all_results=False,
        transport=tr,
    )
    assert res.all_results_included is False
    assert tr.payload is not None
    assert tr.payload["all_results"] is False


def test_recognized_count_excludes_failed() -> None:
    # P05067 is in the failed list -> recognized is one fewer than mapped.
    tr_fail = RecordingTransport()

    def transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        out = tr_fail(url, payload)
        out["meta"]["genes_metadata"]["failed"] = ["P05067"]
        return out

    res = en.enrich(
        _valid_query(), _valid_background(), "mmusculus", transport=transport
    )
    assert res.n_query_mapped == 3
    assert res.n_query_recognized == 2


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="query is empty"):
        en.enrich([], _valid_background(), "mmusculus", transport=RecordingTransport())


def test_empty_background_raises() -> None:
    with pytest.raises(ValueError, match="background is empty"):
        en.enrich(_valid_query(), [], "mmusculus", transport=RecordingTransport())


def test_bad_correction_raises() -> None:
    with pytest.raises(ValueError, match="Unknown correction"):
        en.enrich(
            _valid_query(),
            _valid_background(),
            "mmusculus",
            correction="holm",  # type: ignore[arg-type]
            transport=RecordingTransport(),
        )


def test_bad_threshold_raises() -> None:
    with pytest.raises(ValueError, match="user_threshold"):
        en.enrich(
            _valid_query(),
            _valid_background(),
            "mmusculus",
            user_threshold=1.5,
            transport=RecordingTransport(),
        )


def test_query_not_subset_of_background_raises() -> None:
    with pytest.raises(ValueError, match="not in the background"):
        en.enrich(
            ["sp|P99999|NOTBG_MOUSE"],
            _valid_background(),
            "mmusculus",
            transport=RecordingTransport(),
        )


def test_zero_mapping_raises() -> None:
    with pytest.raises(ValueError, match="No query id mapped"):
        en.enrich(
            ["contaminant_only|FOO|BAR"],
            ["contaminant_only|FOO|BAR"],
            "mmusculus",
            transport=RecordingTransport(),
        )


def test_low_mapping_warns() -> None:
    # 1 of 3 query ids map (33% < 50%) -> warn. All are in background.
    query = ["sp|P05067|A", "junk1|FOO|BAR", "junk2|BAZ|QUX"]
    background = [*query, "sp|Q9DBR1|B"]
    with pytest.warns(en.EnrichmentMappingWarning, match="mapped"):
        en.enrich(query, background, "mmusculus", transport=RecordingTransport())


# --------------------------------------------------------------------------- #
# Service errors
# --------------------------------------------------------------------------- #
def test_transport_error_propagates() -> None:
    def bad_transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise en.EnrichmentServiceError("boom")

    with pytest.raises(en.EnrichmentServiceError, match="boom"):
        en.enrich(
            _valid_query(), _valid_background(), "mmusculus", transport=bad_transport
        )


def test_response_without_result_raises() -> None:
    def no_result(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"meta": {"version": "x"}}

    with pytest.raises(en.EnrichmentServiceError, match="no 'result'"):
        en.enrich(_valid_query(), _valid_background(), "mmusculus", transport=no_result)


def test_missing_version_warns_and_defaults() -> None:
    def no_version(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"meta": {}, "result": _CANNED_RESULT}

    with pytest.warns(en.EnrichmentMappingWarning, match="no data version"):
        res = en.enrich(
            _valid_query(), _valid_background(), "mmusculus", transport=no_version
        )
    assert res.gprofiler_version == "unknown"


# --------------------------------------------------------------------------- #
# Bridge — differential-abundance
# --------------------------------------------------------------------------- #
def _de_result() -> tuple[da.DifferentialAbundanceResult, dl.Dataset]:
    """A DE result whose feature ids are valid-format UniProt accessions in sp| form."""
    rng = np.random.default_rng(0)
    n_features = 30
    features = np.array(
        [f"sp|P{i:04d}0|G{i}_MOUSE" for i in range(n_features)], dtype=str
    )
    # 6 samples, two groups; plant a strong up-shift in features 0-4 for group B.
    ab = rng.standard_normal((6, n_features)) + 10.0
    ab[3:, :5] += 5.0
    meta = pd.DataFrame({"grp": ["A", "A", "A", "B", "B", "B"]})
    ds = dl.Dataset(
        abundances=ab,
        feature_names=features,
        feature_metadata=pd.DataFrame({"feature": features}),
        metadata=meta,
        scale="log2",
    )
    res = da.differential_abundance(ds, "grp", method="ols")
    return res, ds


def test_bridge_up_direction_query() -> None:
    res, ds = _de_result()
    tr = RecordingTransport()
    en.enrich_from_differential_abundance(
        res, ds, "mmusculus", direction="up", fdr=0.5, transport=tr
    )
    assert tr.payload is not None
    # Background is every feature; query is a strict subset (the up hits).
    assert len(tr.payload["background"]) == 30
    assert 0 < len(tr.payload["query"]) <= 30


def test_bridge_no_hits_raises() -> None:
    res, ds = _de_result()
    with pytest.raises(ValueError, match="nothing to enrich"):
        # An impossibly strict gate leaves no hits.
        en.enrich_from_differential_abundance(
            res,
            ds,
            "mmusculus",
            direction="up",
            fdr=1e-12,
            transport=RecordingTransport(),
        )


def test_bridge_unknown_term_raises() -> None:
    res, ds = _de_result()
    with pytest.raises(ValueError, match="not a contrast term"):
        en.enrich_from_differential_abundance(
            res, ds, "mmusculus", term="grp[Z vs A]", transport=RecordingTransport()
        )


# --------------------------------------------------------------------------- #
# Network-gated real-data smoke
# --------------------------------------------------------------------------- #
_DATA = Path(__file__).resolve().parents[2] / "testdata" / "5xFAD"
_PROT = _DATA / "data" / "proteins_wide_unnormalized.tsv"


def _gprofiler_reachable() -> bool:
    try:
        en.enrich(
            ["P05067"],
            ["P05067", "P12345"],
            "hsapiens",
            sources=["KEGG"],
            all_results=False,
        )
    except en.EnrichmentServiceError:
        return False
    except Exception:  # any other error still means we reached the API
        return True
    return True


@pytest.mark.skipif(not _PROT.exists(), reason="5xFAD testdata not present")
def test_real_5xfad_enrichment_smoke() -> None:
    if not _gprofiler_reachable():
        pytest.skip("g:Profiler service not reachable (offline)")
    ids = pd.read_csv(_PROT, sep="\t", usecols=[0])["protein"].astype(str).tolist()
    background = ids
    accs = [a for a in (en.parse_uniprot_accession(x) for x in ids) if a]
    # Query = a small, deterministic slice of mappable proteins.
    query = [x for x in ids if en.parse_uniprot_accession(x) in set(accs[:40])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = en.enrich(query, background, "mmusculus")
    # The decode audit reproduces the ~99% mappable ratio from the preview.
    assert res.n_background_mapped / res.n_background > 0.98
    assert res.gprofiler_version != "unknown"
    assert isinstance(res.table, pd.DataFrame)
