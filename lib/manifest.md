---
schema_version: 1
generated: 2026-06-23
---

# lib/ template registry

Derived index of the vetted **template scripts** in `lib/` — source of truth is each template's
`__script_meta__` header (regenerate by scanning them, like the findings manifest). Templates are
*seeds*: copied into a project's `scripts/` and adapted (`conventions/script-registry.md`). A
project records which template + version it seeded from in its findings' `provenance.seeded_from`.

**The contract is the data structure, not the file format.** A loader template encodes the
*shape* of the in-memory object the rest of the templates consume (e.g. `Dataset`). A study whose
files don't match a template's input may need its own loader — written and tested for that data,
with the template as a guide — but it should return the **same structure** so downstream templates
(analysis, figures) compose unchanged.

| Template | Version | Path | Kind | Provides | Description |
|----------|---------|------|------|----------|-------------|
| wide-data-loader | 0.2 | lib/common/data_loading.py | module | `Dataset`, `Scale`, `ReplicateCollapse`, `load_wide_data`, `load_precursor_data` | Verified loader for wide feature×sample omics matrices + a sample-metadata table: orientation/pairing checks, optional technical-replicate collapse, precursor charge-state collapse, zero-preserving, fail-loud. Study-agnostic (column names are arguments); returns the standard `Dataset` contract, tagged with a recorded `scale` (`linear`/`log2`/`glog2`/`zscore`). |
