"""GO / KEGG over-representation (ORA) of a protein list via g:Profiler.

TEMPLATE (lib/) — a *seed* for a project's enrichment script, not a finished analysis.
Copy it into the project's ``scripts/`` and adapt the call site per study (which id
column, which organism, how the ids encode UniProt accessions). Held to the correctness
charter (conventions/correctness.md) and the statistics convention
(conventions/statistics.md): **assume nothing, verify everything, fail loud.**

What it does: given a **query** set of proteins (the significant / selected features
of an upstream analysis — a differential-abundance hit list, a Boruta-confirmed set, a
classifier-selected set, a cluster) and a **background** set (the experiment's detected
proteome), it asks g:Profiler which GO terms (biological process / molecular function /
cellular component) and KEGG pathways are **over-represented** in the query relative to
the background, and returns one per-term table (corrected p + set sizes + gene ratio)
plus the id-decoding audit. The two companion figures live in :mod:`figures.enrichment`.

**Identifier handling — the load-bearing, human-signed-off step.** g:Profiler keys on
gene identifiers; this template maps each protein id to a **UniProt accession** with
:func:`parse_uniprot_accession` (``sp|A1L3T7|RIPR3_MOUSE`` -> ``A1L3T7``;
``tr|A0A075B5J9|...`` -> ``A0A075B5J9``; bare accessions pass through; isoform suffixes
are stripped; a contaminant mnemonic like ``ALBU_BOVIN`` returns ``None`` and
drops out). g:Profiler then converts the accessions to genes internally (g:Convert), so
no separate UniProt REST call is needed. **No parser can be assumed correct for an
unknown study**, so the workflow *must* show the scientist the decoding
(:func:`build_mapping_audit`) and get sign-off first — the parser is a swappable
``id_parser=`` argument, so an odd encoding is a one-function fix, not a template
edit (conventions/statistics.md; enforced by the Stage-4 checkpoint + stats-reviewer).

**Background.** The statistical background is the experiment's **detected, mappable
proteome** (``domain_scope="custom"``) — not the whole genome. This is the honest,
conservative choice: a whole-genome background trivially inflates terms that are
ubiquitous in the measured tissue. The query must be a subset of the background.

**Network + reproducibility.** This is the one ``lib/`` template that calls an external,
**versioned live service**. The g:Profiler data version (Ensembl / GO / KEGG release,
e.g. ``e114_eg62_p19_...``) is captured for provenance — a result is only
reproducible against a stated version; pin ``base_url`` to an archived release
for exact reproducibility. Every network / service failure **raises**
:class:`EnrichmentServiceError` (never a silent empty result). The call is injectable
via ``transport=`` so tests run offline; the real service is exercised only by a
network-gated smoke test.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from common.data_loading import Dataset

from analysis.differential_abundance import DifferentialAbundanceResult

# The g:Profiler correction for over-representation. ``g_SCS`` is g:Profiler's own
# graph-structure-aware method and its recommended default (it accounts for the
# correlated hierarchy of GO terms); ``fdr`` (Benjamini-Hochberg) and ``bonferroni`` are
# the familiar alternatives, both more permissive than g:SCS on this data.
Correction = Literal["g_SCS", "fdr", "bonferroni"]

# Which upstream-hit direction to enrich (the diff-abundance bridge). Directional
# ORA (up / down separately) is far more interpretable than the merged list.
Direction = Literal["up", "down", "both"]

# The HTTP layer, injectable for offline tests: ``(url, json_payload) -> parsed_json``.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]

DEFAULT_GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
DEFAULT_SOURCES: tuple[str, ...] = ("GO:BP", "GO:MF", "GO:CC", "KEGG")
_DEFAULT_TIMEOUT = 120
# Below this fraction of the query mapping to an accession, warn — the ids probably are
# not encoded the way the parser assumes, so the scientist should re-check the decoding.
_LOW_MAPPING_RATE = 0.5

# Official UniProt accession syntax (www.uniprot.org/help/accession_numbers): the two
# alternations match the 6- and 10-character accession forms. Anchored — a token is an
# accession only if it matches in full (so ``ALBU_BOVIN`` and ``RIPR3_MOUSE`` do not).
UNIPROT_ACCESSION = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)
# A trailing ``-<n>`` is a UniProt isoform suffix (``P05067-2``); strip to the canonical
# accession, which is what g:Profiler maps.
_ISOFORM_SUFFIX = re.compile(r"-\d+$")
# Protein-group / multi-id separators to split before scanning pipe-delimited fields.
_GROUP_SEPARATORS = re.compile(r"[;,\s]+")

__script_meta__: dict[str, object] = {
    "template": {"name": "enrichment", "version": "0.1"},
    "kind": "analysis",
    "provides": [
        "Correction",
        "Direction",
        "Transport",
        "EnrichmentMappingWarning",
        "EnrichmentServiceError",
        "EnrichmentResult",
        "parse_uniprot_accession",
        "build_mapping_audit",
        "enrich",
        "enrich_from_differential_abundance",
        "DEFAULT_GPROFILER_URL",
        "DEFAULT_SOURCES",
    ],
    "uses": ["common.data_loading", "analysis.differential_abundance"],
    "seeded_from": None,
    "description": (
        "GO (BP/MF/CC) + KEGG over-representation of a protein query against the "
        "experiment's detected-proteome background via g:Profiler. Maps protein ids to "
        "UniProt accessions with a generalized, swappable, human-signed-off parser "
        "(sp|/tr|/bare/isoform/protein-group; contaminant mnemonics drop out); custom "
        "background (domain_scope=custom), g:SCS default correction (fdr/bonferroni "
        "selectable). Returns a per-term table (corrected p + set sizes + gene ratio) "
        "plus the id-decode audit and the captured g:Profiler data version for "
        "reproducibility. Thin injectable urllib client (offline-testable); raises "
        "EnrichmentServiceError on any network/service failure. Study-agnostic; "
        "fail-loud. Requires network at run time."
    ),
}


class EnrichmentMappingWarning(UserWarning):
    """Few query ids mapped to a UniProt accession — the encoding is probably different.

    The fix is to check how the study encodes ids and pass a matching ``id_parser=``
    (conventions/statistics.md — the ID-encoding sign-off step).
    """


class EnrichmentServiceError(RuntimeError):
    """A g:Profiler request failed (network, HTTP, bad response, or service error).

    Enrichment depends on a live external service; a failure is surfaced loudly rather
    than returned as an empty result that would read as "nothing was enriched".
    """


@dataclass(frozen=True)
class EnrichmentResult:
    """Over-representation results for one query against one background.

    Attributes
    ----------
    table:
        One row per tested term (all terms when ``all_results`` was ``True``, else the
        significant terms only). Columns: ``source`` (GO:BP/MF/CC/KEGG/...), ``term_id``
        (the ontology native id, e.g. ``GO:0006955``), ``term_name``, ``p_value`` (the
        **corrected** p), ``significant`` (bool), ``term_size``, ``query_size``,
        ``intersection_size`` (query genes in the term), ``effective_domain_size``,
        ``gene_ratio`` (= intersection / query size), ``recall`` (= intersection / term
        size), and ``intersecting_genes`` (a ``;``-joined list of the caller's own ids
        for the query proteins in the term — *which of my proteins drove this term*).
        Sorted by ``(source, p_value)``.
    mapping:
        The id-decode audit — one row per input id: ``original_id``,
        ``uniprot_accession`` (empty when unmappable), ``source``
        (swissprot/trembl/other/unmappable), ``in_query``, ``in_background``. This
        artifact the scientist signs off on and the permanent provenance record.
    organism, sources, correction, user_threshold:
        The resolved request (``organism`` is the g:Profiler id, e.g. ``"mmusculus"``;
        ``user_threshold`` is the corrected-p significance cutoff used).
    gprofiler_version:
        The g:Profiler data version string (Ensembl/GO/KEGG release), captured for
        reproducibility; ``"unknown"`` if the service did not report one.
    query_label:
        A caller label for this query (e.g. ``"up in 5xFAD"``), for figures/provenance.
    n_query, n_query_mapped, n_query_recognized:
        Query id counts: supplied, mapped to an accession, and recognized as genes
        by g:Profiler (``n_query_recognized`` is best-effort from the service metadata).
    n_background, n_background_mapped:
        Background id counts: supplied and mapped to a UniProt accession.
    effective_domain_size:
        The universe size g:Profiler used for the hypergeometric test (the mapped
        background), or ``None`` if not reported.
    all_results_included:
        Whether the table includes non-significant terms (for the Manhattan plot).
    """

    table: pd.DataFrame
    mapping: pd.DataFrame
    organism: str
    sources: tuple[str, ...]
    correction: Correction
    user_threshold: float
    gprofiler_version: str
    query_label: str
    n_query: int
    n_query_mapped: int
    n_query_recognized: int
    n_background: int
    n_background_mapped: int
    effective_domain_size: int | None
    all_results_included: bool

    @property
    def significant_table(self) -> pd.DataFrame:
        """The significant terms of :attr:`table`, sorted by ``p_value`` ascending."""
        sub = self.table[self.table["significant"]].copy()
        return sub.sort_values("p_value", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Identifier parsing (the signed-off step)
# --------------------------------------------------------------------------- #
def parse_uniprot_accession(raw: str) -> str | None:
    """Best-effort extraction of a UniProt accession from a protein id string.

    Handles the common encodings, in order of scanning:

      * ``sp|A1L3T7|RIPR3_MOUSE``   -> ``A1L3T7``      (Swiss-Prot pipe form)
      * ``tr|A0A075B5J9|..._MOUSE`` -> ``A0A075B5J9``  (TrEMBL pipe form)
      * bare ``A1L3T7`` / ``P05067`` -> itself
      * protein groups ``P1;P2`` or ``sp|P1|..,sp|P2|..`` -> the **first** mappable
        accession (the representative; note this in provenance)
      * isoform ``P05067-2`` -> canonical ``P05067``

    Returns the accession, or ``None`` when no field matches the accession syntax
    (e.g. a contaminant mnemonic ``ALBU_BOVIN``). This is deliberately conservative — it
    recovers an *embedded* accession, it does not look anything up. Pass a custom parser
    via ``enrich(id_parser=...)`` for an id scheme this does not cover.
    """
    for member in _GROUP_SEPARATORS.split(raw.strip()):
        if not member:
            continue
        for token in member.split("|"):
            candidate = _ISOFORM_SUFFIX.sub("", token.strip())
            if UNIPROT_ACCESSION.match(candidate):
                return candidate
    return None


def _accession_source(raw: str, accession: str | None) -> str:
    """Classify an id for the audit: swissprot / trembl / other / unmappable."""
    if accession is None:
        return "unmappable"
    head = raw.split("|", 1)[0].strip().lower()
    if head == "sp":
        return "swissprot"
    if head == "tr":
        return "trembl"
    return "other"


def build_mapping_audit(
    query: Sequence[str],
    background: Sequence[str],
    *,
    id_parser: Callable[[str], str | None] = parse_uniprot_accession,
) -> pd.DataFrame:
    """Decode every id and return the audit table the scientist signs off on.

    Pure (no network). One row per **distinct** id across ``query`` + ``background``:
    ``original_id``, ``uniprot_accession`` (``""`` if unmappable), ``source``,
    ``in_query``, ``in_background``. Show this (with per-source counts / mappable %) to
    the scientist before running :func:`enrich`.
    """
    query_set = {str(x) for x in query}
    background_set = {str(x) for x in background}
    rows: list[dict[str, object]] = []
    for raw in sorted(query_set | background_set):
        accession = id_parser(raw)
        rows.append(
            {
                "original_id": raw,
                "uniprot_accession": accession or "",
                "source": _accession_source(raw, accession),
                "in_query": raw in query_set,
                "in_background": raw in background_set,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "original_id",
            "uniprot_accession",
            "source",
            "in_query",
            "in_background",
        ],
    )


def _map_accessions(
    ids: Sequence[str], id_parser: Callable[[str], str | None]
) -> list[str]:
    """Parse ids to a sorted, de-duplicated list of accessions (drops misses)."""
    seen: set[str] = set()
    for raw in ids:
        accession = id_parser(str(raw))
        if accession is not None:
            seen.add(accession)
    return sorted(seen)


# --------------------------------------------------------------------------- #
# Transport (injectable; default is stdlib urllib)
# --------------------------------------------------------------------------- #
def _http_post_json(
    url: str, payload: dict[str, Any], *, timeout: int
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the parsed object; fail loud on error."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "findings-workflow-enrichment",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise EnrichmentServiceError(
            f"g:Profiler request to {url} failed: {exc}. Check network connectivity "
            f"and that the service / base_url is reachable."
        ) from exc
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EnrichmentServiceError(
            f"g:Profiler at {url} returned a non-JSON response: {exc}."
        ) from exc
    if not isinstance(parsed, dict):
        raise EnrichmentServiceError(
            f"g:Profiler at {url} returned {type(parsed).__name__}, expected an object."
        )
    return parsed


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def enrich(
    query: Sequence[str],
    background: Sequence[str],
    organism: str,
    *,
    sources: Sequence[str] = DEFAULT_SOURCES,
    id_parser: Callable[[str], str | None] = parse_uniprot_accession,
    correction: Correction = "g_SCS",
    user_threshold: float = 0.05,
    all_results: bool = True,
    query_label: str = "query",
    base_url: str = DEFAULT_GPROFILER_URL,
    timeout: int = _DEFAULT_TIMEOUT,
    transport: Transport | None = None,
    min_mapping_rate: float = _LOW_MAPPING_RATE,
) -> EnrichmentResult:
    """Run GO/KEGG over-representation of ``query`` vs ``background`` via g:Profiler.

    Parameters
    ----------
    query:
        Protein ids of the significant / selected set (raw, as they appear upstream).
    background:
        Protein ids of the **detected proteome** — the statistical universe. ``query``
        must map into this set (enforced after parsing).
    organism:
        g:Profiler organism id (e.g. ``"mmusculus"``, ``"hsapiens"``). Study-specific.
    sources:
        Ontology sources to query (default GO:BP/MF/CC + KEGG). Passed through to
        g:Profiler, which validates them.
    id_parser:
        ``str -> accession | None``; default :func:`parse_uniprot_accession`. Swap for
        an unusual encoding (after showing the scientist :func:`build_mapping_audit`).
    correction:
        Multiple-testing method: ``"g_SCS"`` (default), ``"fdr"``, or ``"bonferroni"``.
    user_threshold:
        Significance threshold for the corrected p (default ``0.05``).
    all_results:
        When ``True`` (default) the table includes non-significant terms (needed for the
        Manhattan plot); the ``significant`` column / :attr:`significant_table` select
        the hits. ``False`` returns significant terms only (leaner).
    query_label:
        Human label for this query (e.g. ``"up in 5xFAD"``) — recorded and used by
        figures.
    base_url:
        g:Profiler gost endpoint. Pin to an archived release for exact reproducibility.
    timeout:
        Request timeout (seconds) for the default transport.
    transport:
        Optional injected HTTP function ``(url, payload) -> parsed_json`` (for offline
        testing). ``None`` uses the stdlib-urllib client.
    min_mapping_rate:
        Warn (:class:`EnrichmentMappingWarning`) if fewer than this fraction of query
        ids map to an accession (default ``0.5``); a **zero** rate raises.

    Returns
    -------
    EnrichmentResult
    """
    if not query:
        raise ValueError("query is empty; nothing to enrich.")
    if not background:
        raise ValueError(
            "background is empty; a detected-proteome background is required."
        )
    sources = tuple(str(s) for s in sources)
    if not sources:
        raise ValueError("sources is empty; pass at least one (e.g. 'GO:BP').")
    if correction not in ("g_SCS", "fdr", "bonferroni"):
        raise ValueError(
            f"Unknown correction {correction!r}; expected g_SCS/fdr/bonferroni."
        )
    if not 0.0 < user_threshold <= 1.0:
        raise ValueError(f"user_threshold must be in (0, 1]; got {user_threshold}.")

    mapping = build_mapping_audit(query, background, id_parser=id_parser)
    query_acc = _map_accessions(query, id_parser)
    background_acc = _map_accessions(background, id_parser)
    n_query = len({str(x) for x in query})
    n_background = len({str(x) for x in background})

    if not query_acc:
        raise ValueError(
            "No query id mapped to a UniProt accession. Check how the study encodes "
            "protein ids and pass a matching id_parser= (see build_mapping_audit)."
        )
    if not background_acc:
        raise ValueError(
            "No background id mapped to a UniProt accession; check the id encoding / "
            "id_parser (see build_mapping_audit)."
        )
    mapping_rate = len(query_acc) / n_query
    if mapping_rate < min_mapping_rate:
        warnings.warn(
            f"Only {len(query_acc)}/{n_query} ({mapping_rate:.0%}) of query ids mapped "
            f"to a UniProt accession. The ids may be encoded unlike the parser "
            f"assumes — check build_mapping_audit and pass a matching id_parser=.",
            EnrichmentMappingWarning,
            stacklevel=2,
        )

    background_set = set(background_acc)
    unmatched = [a for a in query_acc if a not in background_set]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} query accession(s) are not in the background "
            f"(e.g. {unmatched[:3]}). The query must be a subset of the background "
            f"background for a valid over-representation test."
        )

    payload: dict[str, Any] = {
        "organism": organism,
        "query": query_acc,
        "sources": list(sources),
        "domain_scope": "custom",
        "background": background_acc,
        "user_threshold": user_threshold,
        "significance_threshold_method": correction,
        "no_evidences": False,
        "all_results": all_results,
    }
    if transport is None:

        def _default_transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
            return _http_post_json(url, body, timeout=timeout)

        transport = _default_transport
    response = transport(base_url, payload)

    table, effective_domain = _parse_result_table(response, sources)
    version = _extract_version(response)
    n_recognized = _count_recognized(response, query_acc)

    return EnrichmentResult(
        table=table,
        mapping=mapping,
        organism=organism,
        sources=sources,
        correction=correction,
        user_threshold=user_threshold,
        gprofiler_version=version,
        query_label=query_label,
        n_query=n_query,
        n_query_mapped=len(query_acc),
        n_query_recognized=n_recognized,
        n_background=n_background,
        n_background_mapped=len(background_acc),
        effective_domain_size=effective_domain,
        all_results_included=all_results,
    )


def _parse_result_table(
    response: dict[str, Any], sources: tuple[str, ...]
) -> tuple[pd.DataFrame, int | None]:
    """Turn the gost ``result`` list into the per-term table + effective domain size."""
    result_items = response.get("result")
    if not isinstance(result_items, list):
        raise EnrichmentServiceError(
            "g:Profiler response has no 'result' list — the request may have been "
            f"rejected. Response keys: {sorted(response)}."
        )
    columns = [
        "source",
        "term_id",
        "term_name",
        "p_value",
        "significant",
        "term_size",
        "query_size",
        "intersection_size",
        "effective_domain_size",
        "gene_ratio",
        "recall",
        "intersecting_genes",
    ]
    ensgs, ensg_to_input = _query_gene_map(response)
    rows: list[dict[str, object]] = []
    effective_domain: int | None = None
    for item in result_items:
        if not isinstance(item, dict):
            continue
        query_size = int(item.get("query_size", 0))
        term_size = int(item.get("term_size", 0))
        intersection = int(item.get("intersection_size", 0))
        eff = item.get("effective_domain_size")
        if isinstance(eff, int) and effective_domain is None:
            effective_domain = eff
        rows.append(
            {
                "source": str(item.get("source", "")),
                "term_id": str(item.get("native", "")),
                "term_name": str(item.get("name", "")),
                "p_value": float(item.get("p_value", np.nan)),
                "significant": bool(item.get("significant", False)),
                "term_size": term_size,
                "query_size": query_size,
                "intersection_size": intersection,
                "effective_domain_size": int(eff) if isinstance(eff, int) else -1,
                "gene_ratio": (intersection / query_size) if query_size else np.nan,
                "recall": (intersection / term_size) if term_size else np.nan,
                "intersecting_genes": _term_members(
                    item.get("intersections"), ensgs, ensg_to_input
                ),
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    if not table.empty:
        # Order sources as requested, then by ascending corrected p within each source.
        order = {src: i for i, src in enumerate(sources)}
        table["_src_order"] = table["source"].map(lambda s: order.get(s, len(order)))
        table = (
            table.sort_values(["_src_order", "p_value"], kind="stable")
            .drop(columns="_src_order")
            .reset_index(drop=True)
        )
    return table, effective_domain


def _query_gene_map(response: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """The recognized-gene order + an ``ensg -> input id`` inverse for member recovery.

    g:Profiler's per-term ``intersections`` align to the **recognized genes**
    (``meta.genes_metadata.query.<q>.ensgs``), not the query — unmapped ids are
    dropped. ``mapping`` is ``{input_id: [ensg, ...]}``; inverting it lets each term's
    members be reported as the caller's own ids. Returns ``([], {})`` when metadata is
    absent (e.g. an evidence-free response), so members are then left empty.
    """
    meta = response.get("meta")
    if not isinstance(meta, dict):
        return [], {}
    genes_meta = meta.get("genes_metadata")
    if not isinstance(genes_meta, dict):
        return [], {}
    query = genes_meta.get("query")
    if not isinstance(query, dict):
        return [], {}
    entry = next(
        (v for v in query.values() if isinstance(v, dict) and "ensgs" in v), None
    )
    if entry is None:
        return [], {}
    raw_ensgs = entry.get("ensgs")
    if not isinstance(raw_ensgs, list):
        return [], {}
    ensgs = [str(e) for e in raw_ensgs]
    inverse: dict[str, str] = {}
    mapping = entry.get("mapping")
    if isinstance(mapping, dict):
        for input_id, ensg_list in mapping.items():
            if isinstance(ensg_list, list):
                for ensg in ensg_list:
                    inverse.setdefault(str(ensg), str(input_id))
    return ensgs, inverse


def _term_members(
    intersections: object, ensgs: list[str], ensg_to_input: dict[str, str]
) -> str:
    """The caller's ids for the query genes in one term (``;``-joined; ``""`` if none).

    ``intersections`` is the gost per-term list aligned to ``ensgs``: a non-empty entry
    (evidence codes) means that gene is in the term. Falls back to the ENSG id when an
    input id can't be recovered; returns ``""`` on any shape mismatch (fail-soft — the
    member column is a convenience, not load-bearing).
    """
    if not isinstance(intersections, list) or len(intersections) != len(ensgs):
        return ""
    members = [
        ensg_to_input.get(ensg, ensg)
        for ensg, evidence in zip(ensgs, intersections, strict=True)
        if evidence
    ]
    return ";".join(members)


def _extract_version(response: dict[str, Any]) -> str:
    """Capture the g:Profiler data version for reproducibility ('unknown' if absent)."""
    meta = response.get("meta")
    if isinstance(meta, dict):
        version = meta.get("version")
        if isinstance(version, str) and version:
            return version
    warnings.warn(
        "g:Profiler response carried no data version; the result cannot be pinned to a "
        "release. Enrichment reproducibility depends on recording the version.",
        EnrichmentMappingWarning,
        stacklevel=3,
    )
    return "unknown"


def _count_recognized(response: dict[str, Any], query_acc: list[str]) -> int:
    """Best-effort count of query accessions g:Profiler recognized (mapped - failed)."""
    meta = response.get("meta")
    if not isinstance(meta, dict):
        return len(query_acc)
    genes_meta = meta.get("genes_metadata")
    if not isinstance(genes_meta, dict):
        return len(query_acc)
    failed = genes_meta.get("failed")
    if not isinstance(failed, list):
        return len(query_acc)
    failed_set = {str(f) for f in failed}
    return sum(1 for a in query_acc if a not in failed_set)


# --------------------------------------------------------------------------- #
# Family bridge — extract query + background from a differential-abundance result
# --------------------------------------------------------------------------- #
def enrich_from_differential_abundance(
    result: DifferentialAbundanceResult,
    dataset: Dataset,
    organism: str,
    *,
    direction: Direction = "up",
    term: str | None = None,
    fdr: float = 0.05,
    effect_threshold: float | None = None,
    sources: Sequence[str] = DEFAULT_SOURCES,
    id_parser: Callable[[str], str | None] = parse_uniprot_accession,
    correction: Correction = "g_SCS",
    base_url: str = DEFAULT_GPROFILER_URL,
    timeout: int = _DEFAULT_TIMEOUT,
    transport: Transport | None = None,
) -> EnrichmentResult:
    """Enrich the significant hits of a differential-abundance contrast.

    Pulls the **query** from the contrast term's significant features (``q <= fdr``,
    optionally ``|effect| >= effect_threshold``), by ``direction`` (``"up"`` = effect
    > 0, ``"down"`` = effect < 0, ``"both"`` = either — directional ORA is more
    interpretable), and the **background** from every feature in ``dataset``. ``term``
    selects the contrast coefficient when the contrast has >1 (a ``k>2`` factor).
    """
    term_name = _resolve_term(result, term)
    rows = result.table[result.table["term"] == term_name]
    effect = np.asarray(rows["effect"].to_numpy(), dtype=float)
    q = np.asarray(rows["q"].to_numpy(), dtype=float)
    features = rows["feature"].to_numpy()

    sig = np.isfinite(q) & (q <= fdr)
    if effect_threshold is not None:
        sig = sig & (np.abs(effect) >= effect_threshold)
    if direction == "up":
        selected = sig & (effect > 0)
    elif direction == "down":
        selected = sig & (effect < 0)
    else:
        selected = sig
    query_ids = [str(f) for f in features[selected]]
    if not query_ids:
        raise ValueError(
            f"No {direction} hits at q<={fdr} for {term_name!r}; nothing to enrich."
        )
    background_ids = [str(f) for f in dataset.feature_names]
    return enrich(
        query_ids,
        background_ids,
        organism,
        sources=sources,
        id_parser=id_parser,
        correction=correction,
        query_label=f"{direction} in {term_name}",
        base_url=base_url,
        timeout=timeout,
        transport=transport,
    )


def _resolve_term(result: DifferentialAbundanceResult, term: str | None) -> str:
    """Pick the contrast term to enrich; fail loud on an ambiguous or unknown choice."""
    terms = result.contrast_terms
    if term is None:
        if len(terms) != 1:
            raise ValueError(
                f"This has {len(terms)} contrast terms {list(terms)}; pass term= "
                f"to choose which to enrich."
            )
        return terms[0]
    if term not in terms:
        raise ValueError(
            f"term {term!r} is not a contrast term; choose one of {list(terms)}."
        )
    return term
