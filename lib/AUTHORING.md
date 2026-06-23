# Authoring a `lib/` template

**Engine-dev guide — read this first when adding a template to `lib/`.** It is the durable
authoring contract. `lib/README.md` explains *what* `lib/` is (the model); this explains
*how* to add to it; `CLAUDE.md` → *Next: building `lib/`* tracks only current status +
priorities. This file is **not user-facing** — it never ships into a project.

## What a `lib/` template is

A template is a *seed*: a study-agnostic, maximally-vetted script the workflow copies into
a project's `scripts/` and adapts (`lib/README.md`). It is held to a higher bar than
ordinary code because a careless copy can reintroduce a hazard, and because every finding
records which template + version it seeded from. The interface a template honors is the
in-memory **`Dataset`** (abundances `(n_samples, n_features)` + `feature_names` +
`feature_metadata` + a row-aligned `metadata` table, tagged with a `scale`): a template
**consumes and/or returns that structure** so the others compose unchanged.
*Adapt the input; keep the output structure.*

## The build workflow (repeat for each template)

1. **Mine an oracle — don't start from a blank file.** Begin from a real, already-working
   implementation and use it to capture ground truth. For the current templates the oracle
   is the source project `/home/mriffle/vscode/johnson-5xFAD-lecanemab-mice-AD`: its `src/`
   is generalized seed code; its `scripts/` is project-specific and is **not** template
   material. Read the relevant source, run it against the real data, and record the
   ground-truth numbers (shapes, key values) you will test against.
2. **Generalize.** Make it study-agnostic — column names and options become **arguments**,
   no hardcoded biology — and **strip every dataset-specific exception** (sample
   exclusions, relabelings, identity/tube-swap corrections). Those live in the project
   copy, never the template. Consume/return `Dataset`.
3. **Make it strict-clean.** `./.venv/bin/ruff check` + `ruff format` + `mypy --strict`
   against the strict bar in `conventions/coding.md`. No bare `# noqa` / `# type: ignore`
   (rule/code-scoped + justified only).
4. **Test it** — see *The testing bar* below.
5. **Register it** — set the `__script_meta__` header and add the row to `lib/manifest.md`.
6. **Verify** — `claude plugin validate .` and run the full `lib/` suite green.

## Layout & naming

- `lib/common/` — shared modules (these seed a project's `scripts/promoted/common/`).
  Add `lib/analysis/` and `lib/figures/` for entry-point templates as they arrive.
- `lib/tests/` — the tests, centralized (mirrors the module structure).
- `lib/manifest.md` — the **derived** registry; regenerate by scanning `__script_meta__`
  headers (the headers are the source of truth, like the findings manifest).

## The `__script_meta__` header & versioning

Every template carries a module-level `__script_meta__` dict (copy the shape from an
existing one, e.g. `lib/common/data_loading.py`):

- **`template: {name, version}`** — the **source of truth** for the template's identity.
  `lib/manifest.md` indexes it, and a finding's `provenance.seeded_from` references the
  same pair: `seeded_from.template` = this `template.name`, `seeded_from.version` = this
  `template.version` (`conventions/script-registry.md`, `conventions/findings.md`).
- **`kind`** (`"module"` | `"analysis"`), **`provides`** (exported symbols), **`uses`**
  (project modules imported, e.g. `["common.data_loading"]`), **`seeded_from: None`**
  (templates are roots), **`description`** (keep it consistent with the manifest row — the
  manifest is derived from it).
- **Versioning:** `0.x` while pre-first-use; **bump the minor on any logic or signature
  change**, and update `lib/manifest.md` in lockstep.

## The contract: `Dataset` in / `Dataset` out

- A loader **returns** a `Dataset`; a transform **takes one and returns a new one**
  (via `dataclasses.replace`, **copying** the carried `metadata`/`feature_metadata`/
  `feature_names` so the result shares no mutable state — see the `_independent` helper in
  `normalize.py`).
- **Never force a file format.** The workflow examines a study's actual files (Stages 1–2)
  and writes a loader for *them* using the template as a guide; format details are
  arguments. What's constant is the returned `Dataset`.
- **The `scale` tag is load-bearing.** A loader *records* the Stage-2-determined scale
  (`linear`/`log2`/`log10`/`ln`/`glog2`/`zscore`/`ratio`) — it does not infer it.
  Transforms read it to refuse scale-incorrect steps (double-logging; ComBat on linear
  data). `LOG_SCALES` (in `data_loading.py`) is the shared taxonomy; add a scale member
  there rather than mislabelling data.

## The testing bar

The strict promotion bar — unit + property/invariant + planted-truth + edge cases;
`mypy --strict`; strict `ruff` — is defined in `conventions/coding.md`. **Reference it; do
not restate or weaken it here.** On top of that bar, an *engine template* adds:

- **Planted-truth with hand-computed expected values.** The common-mode bug is a wrong
  number that looks plausible, so pin exact values (e.g. the median-normalization identity,
  the `log2(x+1)` zero-preservation), not just shapes.
- **Real-data smoke that reproduces the oracle ground truth** captured in step 1 (shapes +
  key values within tolerance), `@pytest.mark.skipif` when the data is absent so the suite
  still passes. `testdata/5xFAD/` is git-ignored real data; smoke tests skip cleanly
  without it.
- **Test the wrapper, not the upstream library.** Verify *our* wiring — arguments threaded,
  `scale` stamped, shape/contract preserved, guards fire, fail-loud messages are clear —
  **not** the published correctness of a third-party method. ComBat (pycombat) and the
  normalizers (pronoms) own their own algorithms; re-proving them wastes effort and couples
  our tests to their internals.

**Dual-domain note (do not conflate):** the above is **engine-dev** template testing,
enforced by us + the dev tooling in this repo. The separate **user-project** script-testing
rules (what a scientist's promoted scripts must pass) live in `conventions/coding.md` +
`correctness.md`, enforced by the code-reviewer + promotion hook in the scientist's
session. Don't add engine-template testing language to the user-facing conventions.

## Standing rules (honor these)

- **Never force a user's data format** — adapt the input, keep the `Dataset` output.
- **Strip dataset-specific exceptions** out of templates — they belong in the project copy.
- **Encode the dangerous structure safely** and mark it: leakage-safe CV inside folds, the
  figure dual-export / color-registry / >8-category guards, scale guards, batch-label-only
  ComBat. A scientific decision baked into a template (e.g. ComBat gets the batch label
  only) is documented in-code as deliberate, with the reasoning.
- **Python only; fail loud** — validate-and-raise at every boundary, no silent NA coercion
  (`conventions/correctness.md`).
- **Reference bundled files via `${CLAUDE_PLUGIN_ROOT}/lib/...`**, never absolute paths.
- **Dependencies:** a new *shipping* dependency goes into `commands/setup-env.md`'s baseline
  (pin it if the template is version-sensitive, e.g. `pronoms==0.4.0`, `pycombat==0.20`),
  and into this repo's `./.venv` for development.

## Dev environment

`./.venv` (git-ignored) carries the toolchain — `numpy pandas pandas-stubs ruff mypy
pytest hypothesis` plus template deps (`pronoms`, `pycombat`). The root `pyproject.toml`
carries the strict bar (`mypy --strict` + the strict ruff ruleset, mirroring `setup-env`)
and `pythonpath=["lib"]` / `mypy_path="lib"`, so tests import `from common import …`
exactly as a project would. Run everything via `./.venv/bin/...`.

## Pre-commit checklist

- [ ] Study-agnostic; no hardcoded biology; dataset-specific exceptions stripped.
- [ ] Consumes/returns `Dataset` (or documents why not); transforms return an independent copy.
- [ ] `__script_meta__` set (`template:{name,version}`, `kind`, `provides`, `uses`, `description`); version bumped if logic/signature changed.
- [ ] `ruff check` + `ruff format` clean; `mypy --strict` clean.
- [ ] Tests: unit + planted-truth (hand-verified) + edge + real-data smoke vs oracle; tests pin our wrapper, not the library.
- [ ] Registered in `lib/manifest.md`, matching the header (incl. description).
- [ ] Any new shipping dependency added to `commands/setup-env.md` (pinned if version-sensitive).
- [ ] `claude plugin validate .` passes; the full `lib/` suite is green.

## Worked example

The loader → normalize → batch-correct chain (`lib/common/{data_loading,normalize,
batch_correct}.py` + their `lib/tests/` suites) is the reference implementation:
study-agnostic, `Dataset`-contracted, scale-tagged, strict-clean, planted-truth +
real-5xFAD smoke, with batch-label-only ComBat as a documented scientific decision. The
adversarial audit that hardened them — and the by-decision scope (don't re-verify ComBat;
missing-value handling is a Stage-2 scientist choice) — is logged in the memory
`findings-workflow-build.md`.
