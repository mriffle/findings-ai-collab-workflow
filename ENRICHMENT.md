# Enrichment analysis — GO / KEGG over-representation via g:Profiler — implementation plan

**What this is.** The guiding + progress-tracking document for the engine's **enrichment
layer**: the `lib/analysis/` + `lib/figures/` templates that answer *"what biology is my
list of significant/selected proteins enriched for?"* — GO (BP/MF/CC) and KEGG
over-representation (ORA) against the experiment's own detected proteome, via the
**g:Profiler** service.

It is **engine-dev planning, not user-facing** (like [`FEATURE_FINDING.md`](FEATURE_FINDING.md)
and [`QC_GAPS.md`](QC_GAPS.md)). It is the downstream companion to feature finding: enrichment
consumes a *query* set (DE hits, Boruta-confirmed, classifier-selected, a cluster) plus a
*background* (the detected proteome) — so it is a new analysis family, not a `method=` of an
existing template.

**Status: ✅ SHIPPED (v0.1, 2026-07-13).** `lib/analysis/enrichment` (`enrich` +
`enrich_from_differential_abundance` + `parse_uniprot_accession` + `build_mapping_audit`) and
`lib/figures/enrichment` (dot / bar / Manhattan) ship, strict-clean (ruff + `mypy --strict`)
with 42 tests (planted-truth parser, mocked-HTTP wiring/guards, the three figures, and a
network-gated real-5xFAD smoke). Wired into `lib/manifest.md`,
`conventions/{statistics,visualization,findings,enforcement-map}.md`, and
`commands/stage4-explore.md`. The real-data preview that de-risked it is under
`testdata/5xFAD/_enrichment_preview/`; the shipped-template example gallery is under
`testdata/5xFAD/_enrichment_examples/`. Design decisions (below) are settled.

> **Scope note:** `FEATURE_FINDING.md` and `QC_GAPS.md` currently list enrichment as
> *out of scope*. This document is the deliberate reversal of that call (the user has asked
> for it). Those two docs' scope-boundary lines get updated when the first template ships.

---

## What the real-data preview proved (2026-07-13)

Full detail in `testdata/5xFAD/_enrichment_preview/PREVIEW_NOTES.md`. Headlines:

- **g:Profiler accepts bare UniProt accessions directly** (auto-converts to genes via
  g:Convert) — no separate UniProt REST mapping call needed. Endpoint reachable from stdlib
  `urllib` (no client library strictly required).
- **The generalized ID parser hit 99.2%** on the 8829 real protein ids
  (`sp|ACC|NAME` / `tr|ACC|NAME` / bare / isoform-suffixed), correctly returning `None` for
  the 75 `crapola_crap|MNEMONIC|MNEMONIC` contaminants — **and** correctly decoding a
  contaminant that carries a real accession (`crapola_crap|P22629|SAV_STRAV` → `P22629`). So
  **accession parsing and contaminant filtering are two separate concerns.**
- **91% of the query accessions were recognized** as mouse genes by g:Profiler.
- **The custom detected-proteome background is the correct, conservative choice** — whole-genome
  background inflated to 250 "significant" terms dominated by *trivial* ones (synapse,
  cytoplasm, protein binding); the detected-proteome background is honest.
- **Directional ORA (up/down separately) is far more interpretable** than the merged list, and
  produced biologically coherent 5xFAD signatures (extracellular / ion-channel / GAG up;
  contractile / ribosomal down).
- **Every response carries a data version** (`e114_eg62_p19_27110d83`) → recordable for
  reproducibility.

---

## Proposed templates

Two templates, mirroring the established analysis+figure pairing (e.g.
`differential-abundance` + `volcano`):

### `lib/analysis/enrichment` (`enrich(...)`)

```python
enrich(
    query,                       # Sequence[str] — raw feature ids of the sig/selected set
    background,                  # Sequence[str] — raw feature ids of the detected proteome
    organism="mmusculus",       # g:Profiler organism id (REQUIRED — study-specific)
    sources=("GO:BP", "GO:MF", "GO:CC", "KEGG"),
    *,
    id_parser=parse_uniprot_accession,   # str -> str|None; default the generalized parser
    correction="g_SCS",         # g_SCS | fdr | bonferroni (see Open Decision #2)
    user_threshold=0.05,
    base_url=DEFAULT_GPROFILER,  # override to a pinned archive for reproducibility
    timeout=120,
) -> EnrichmentResult
```

`EnrichmentResult` (frozen dataclass, round-trippable via `result-io`):
- `table: pd.DataFrame` — one row per significant term: `source, term_id, term_name,
  p_value` (already corrected), `term_size, query_size, intersection_size,
  effective_domain_size, gene_ratio (=intersection/query), recall (=intersection/term),
  intersecting_genes` (the query genes in the term).
- `mapping: pd.DataFrame` — the **decode audit**: `original_id, uniprot_accession, source
  (swissprot/trembl/bare/unmappable), recognized_by_gprofiler (bool)`. This is the sign-off
  artifact (below).
- Run metadata: `organism, sources, correction, gprofiler_version, n_query, n_query_mapped,
  n_background, n_background_mapped, query_label` (for provenance + the cache fingerprint).

**Design contract:**
- **`Dataset` in the loose sense** — enrichment doesn't take a `Dataset`; it takes id lists
  derived from one (the analysis-result feature column + the `Dataset.feature_names` for the
  background). A thin bridge (`enrich_from_result(da_result, dataset, direction=…)`) pulls the
  query (sig hits, optionally direction-split) + background from a `DifferentialAbundanceResult`
  / `BorutaResult` / classifier result, mirroring `volcano_from_result`.
- **The query is enforced ⊆ background** (statistical validity) — fail-loud if not.
- **Fail loud** at every boundary: empty query after mapping → raise; network / HTTP / service
  error → raise with a clear message (no silent empty result); a low mapping rate → **warn**
  (like the classifier's `FeatureListWarning`), a zero mapping rate → raise.
- **The g:Profiler client is a thin internal helper over stdlib `urllib`** (fully typed, no new
  dependency, no stub friction — the engine venv is deliberately stub-free). Full control over
  fail-loud error handling + version capture. (See Open Decision #1 — vs the `gprofiler-official`
  client.)
- **`base_url` overridable to a pinned g:Profiler archive** for reproducibility; the live
  `version` string is always captured into the result + provenance regardless.

### `lib/figures/enrichment` (`enrichment_dotplot` / `enrichment_barplot`)

Publication-quality result figures on the existing figure foundation (`figure-io` dual-export +
separate legend; source colors from the Okabe–Ito registry — 4 sources is well within the >8
guard). Two proven styles (previewed on real data):
- **Faceted dot plot** — one panel per source; y = term, x = gene ratio, dot **size** =
  intersection size (# query genes), **color** = −log10 adjusted p (viridis, continuous → no
  categorical budget). The richest single view. **Layout settled in the preview** (from user
  feedback): the source name is an **in-panel** label (top-left, colored) — never a title in the
  inter-panel gap that could collide with the tick labels of the panel above; and the **whole key
  (colour scale + dot-size legend) goes in the companion legend image**, not baked into the plot
  area (the `figure-io.save_figure` `<base>.legend.{svg,png}` convention), so nothing overlaps a
  data point.
- **Grouped horizontal bar chart** — top-N terms across all sources, x = −log10 adjusted p,
  bar color = source, gene counts inline, dashed line at the significance threshold. Cleanest
  for a slide/paper.

Which is the default / whether we ship both → Open Decision #3.

---

## The ID-encoding sign-off checkpoint (the user's core requirement)

**Non-negotiable:** enrichment must **never** run before the scientist confirms how their
protein ids encode UniProt accessions. This is a Stage-4 collaborative checkpoint, the analysis
analogue of the Stage-2 normalization / control-detection decisions (enforced by the
orchestrator + stats-reviewer, **not** a hook — "did you confirm the encoding?" isn't decidable
from a tool event).

The flow, before any g:Profiler call:
1. Run `id_parser` over the whole detected proteome and **show the scientist**: the per-source
   counts, the overall mappable %, and **worked examples of each decoding**
   (`sp|A1L3T7|RIPR3_MOUSE → A1L3T7`, `tr|... → ...`, `contaminant → (none)`), plus the list of
   **unmappable** ids so nothing is silently dropped.
2. The scientist **signs off** on the decoding (or supplies guidance / a custom `id_parser` if
   their scheme is unusual — the parser is a swappable argument precisely so an odd encoding is
   a one-function fix, not a template edit).
3. Only then is enrichment run; the decode audit (`mapping` table) is written to disk and
   attached to the finding's provenance as the permanent record of what was mapped and how.

The generalized `parse_uniprot_accession` (validated in the preview) is the default and handles
`sp|`/`tr|`/bare/isoform/protein-group forms. It is **best-effort and explicitly reviewable** —
the sign-off exists because no parser can be assumed correct a priori for an unknown study.

---

## Statistics & conventions

Routed through `conventions/statistics.md` (stats-reviewer enforced):
- **Background = the experiment's detected, mappable proteome** (the user's requirement; the
  preview shows why — whole-genome inflates trivially). `domain_scope="custom"` + the mapped
  background list.
- **Experimental subset only** for the background (controls excluded upstream, as everywhere).
- **No bare p** — every term reports the **corrected** p + effect (gene ratio / fold
  enrichment) + set sizes. **Report all significant terms** (the full table to CSV), not a
  hand-picked few.
- **Correction is g:Profiler's g:SCS by default** (GO-structure-aware; g:Profiler's own
  recommendation) with `fdr` / `bonferroni` available — Open Decision #2.
- **Directional ORA** (up / down separately) is offered as the standard, since it is far more
  interpretable than the merged list (a finding usually records both directions).
- Enrichment is an **exploratory→validated** result like any other: a finding cites the upstream
  DE/selection finding it enriches (`relates_to`), and inherits its exploratory/validated status.

---

## Outputs written to disk (the user's requirement)

- **Per-term results CSV** (one file, `source` column distinguishes GO:BP/MF/CC/KEGG; or one per
  source — Open Decision #4) with the full `table` columns above.
- **Decode-audit CSV** (`original_id → accession → source → recognized`) — the sign-off record.
- **Figures** dual-exported (SVG + 300-DPI PNG) + separate legend image via `figure-io`.
- The finding embeds the figure(s) inline (per the 2026-07-13 inline-figures decision) and pins
  `provenance` (the producing script + the g:Profiler version + params).

---

## Reproducibility (a first-of-its-kind concern for `lib/`)

g:Profiler is a **live, versioned external service** — the first `lib/` template to depend on
the network (every existing template is pure-compute + offline + deterministic). Consequences
the plan must handle:
- **Capture `meta.version`** (e.g. `e114_eg62_p19_27110d83`) into the result + finding provenance
  every run — the analogue of recording a random seed. A result is only reproducible against a
  stated g:Profiler/Ensembl/GO/KEGG version.
- **Allow pinning** `base_url` to an archived g:Profiler version for exact reproducibility.
- **CI stays offline:** unit tests **mock** the HTTP layer (inject a fake transport); the
  real-service smoke test is **network-gated** (`skipif` when unreachable), exactly as the
  real-data smokes are `skipif` on absent testdata. No live call in CI.
- **Fail loud** on network/service errors — never a silent empty enrichment.

---

## Testing strategy (to the `lib/` bar, per `lib/AUTHORING.md`)

- **ID parser: planted-truth + edge cases** — every encoding form (`sp|`/`tr|`/bare/isoform/
  group/contaminant-mnemonic/contaminant-with-accession), hand-verified expected accessions.
- **Client wiring: mocked transport** — request shape (organism, sources, `domain_scope=custom`,
  background, correction) is threaded correctly; response parsing; version capture; the
  query⊆background guard; fail-loud on HTTP error / empty query / zero mapping; the
  low-mapping-rate warning.
- **Figures:** render without leaking a figure on error paths; dual-export; source colors from
  the registry.
- **Network-gated real-5xFAD smoke** — reproduce the preview's pinned numbers within tolerance
  (mappable %, recognized %, ≥1 known term), `skipif` offline.

---

## Workflow wiring (when it ships)

- `commands/stage4-explore.md` — an **Enrichment** step: after a DE/selection result, (1) run
  the decode audit + **ID-encoding sign-off**, (2) choose query (direction) + confirm background
  + correction, (3) `enrich(...)`, (4) render figure + CSVs, (5) findings-manager records the
  finding (cites the upstream finding, pins the g:Profiler version).
- `conventions/statistics.md` + `conventions/visualization.md` — the background/correction/
  directional rules + the two figure styles + the sign-off checkpoint (enforcement map row:
  stats-reviewer for the stats + sign-off, figure-reviewer for the figures).
- `conventions/findings.md` — enrichment evidence kind (term + corrected p + set sizes + gene
  ratio; no per-feature q — it is a term-level ORA statistic).
- `lib/manifest.md`, `commands/setup-env.md` (only if a new dep is adopted — Open Decision #1),
  `CLAUDE.md` *Next* section, and the scope-boundary lines in `FEATURE_FINDING.md` / `QC_GAPS.md`.

---

## Design decisions — SETTLED (2026-07-13, with the user)

1. **g:Profiler access — ✅ thin internal `urllib` client.** Zero new dependency, fully typed
   (no stub friction on the deliberately stub-free engine venv), full control over fail-loud +
   version capture. (The official `gprofiler-official` package was the alternative; declined to
   avoid a dep + `mypy` override.) The preview used stdlib `urllib` successfully — that client
   is lifted into the template.
2. **Default multiple-testing correction — ✅ g:SCS.** g:Profiler's GO-structure-aware method
   and its own recommendation; the honest, conservative choice (the preview showed FDR is far
   more permissive). `fdr` / `bonferroni` remain selectable via `correction=`.
3. **Figure styles — ✅ ship all three:** faceted **dot plot** (default rich view), grouped
   **bar chart** (slide/paper), **and g:Profiler-style Manhattan plot** (terms by source on x,
   −log10 p on y — familiar to g:Profiler users). All three on the `figure-io` foundation.
4. **Results CSV layout — one combined file** (a `source` column), simplest to consume; a
   per-source split is trivial to add. *(Adopted; not contentious.)*
5. **Directionality — directional is the standard** (run up and down as separate queries — the
   preview shows it is much more interpretable), with the merged list available via the bridge's
   `direction=` argument. *(Adopted; not contentious.)*

---

## Scope boundaries (deliberately not built now)

- **GSEA / rank-based enrichment** (whole-ranked-list, no significance cutoff) — a different
  method family (fgsea-style); ORA is the ask. Possible later template.
- **Other enrichment services / offline gene-set DBs** (Enrichr, DAVID, local GMT + hypergeometric)
  — g:Profiler is the ask; the `id_parser` + result contract would generalize if added later.
- **Network / STRING / pathway-topology visualization** — downstream of enrichment, its own scope.
- **Non-UniProt id schemes** (Ensembl/Entrez/symbol input) — the `id_parser` argument already
  makes this a one-function swap; g:Profiler accepts them natively, but UniProt is the stated case.
