# Results cache & registry

*How a CPU-heavy analysis result is cached, identified, tracked, and selected for visualization — so a slow analysis (nested CV + the permutation null) is computed **once** and figures re-render without re-running it. Substrate: the vetted `lib/analysis/result-io` template (`save_cached_result` / `load_cached_result` / `result_fingerprint` / `ResultMeta`). Enforced by the stats-reviewer + figure-reviewer + orchestrator behavior (no clean hook — a tool event can't tell "reuse vs recompute").*

## The problem

The heavy analysis templates (`classification`, `classification-xgboost`, `regression`, `boruta`) return a rich result object — fold predictions, the tuning grid, and the opt-in **label/target-shuffle null** (the expensive part) — that the figure templates read via `plot_*(result)`. Held only in memory, tweaking a figure title re-runs the whole analysis. So the **result is cached to disk** and figures render from the cache.

## Compute once, render many

Split the Stage-4 work for a heavy analysis into two scripts:

- **A compute script** runs the analysis and calls `result_io.save_cached_result(result, cache_root="results", analysis=…, data_version=…, params=…, seed=…, label=…, created=…)`, then has the result **registered** in `results/manifest.md`.
- **Figure scripts** `load_cached_result(ResultClass, cache_root="results", analysis=…, fingerprint=…)` and render. A title / color / label tweak re-runs only the (cheap) figure script — never the analysis.

(Light univariate analyses whose whole result is a table — e.g. `differential-abundance` — don't need this; save the table to `results/` and re-render from it. The cache is for the CPU-heavy ML results.)

## Identity — the fingerprint

A cached result is keyed by `result_fingerprint(analysis, data_version, params, seed)`, a deterministic 12-hex id. **Identical inputs → the same fingerprint** (a re-run maps to the same cache entry); **any change** — `outcome`, `binarize`, `covariates`, `feature_list`, `run_null`, method, `seed`, or the dataset's `data_version` — → a **new** result. So `params` must capture *every* knob that affects the numbers. An optional human **label** rides alongside for readability; the fingerprint is the identity.

**Canonicalize set-like params.** A value that is semantically a *set* — notably a `feature_list` (all three predictive templates accept one; `conventions/statistics.md`) — must be recorded in `params` in a **canonical form: sorted and de-duplicated**. The fingerprint hashes list *order* literally, so `["A","B"]` and `["B","A"]` would otherwise mint two entries for an identical model. Canonicalizing means two independent runs requesting the same feature set reliably hit one cache slot.

## The registry — `results/manifest.md`

A Markdown index (like `findings/manifest.md`), one row per cached result: **id (fingerprint)**, analysis, label, `data_version`, key params, `run_null?`, **status**, created, path, and **referencing finding(s)**. The **findings-manager is its only writer** (it already owns the findings manifest + provenance linkage). It is a *derived* index — regenerable from the per-result `meta.json` sidecars (the source of truth).

`status` ∈ **current** | **superseded** | **archived**:
- the most recent result for a given analysis+problem is **current** — the default for visualization;
- a prior result of the same analysis+problem that a newer run replaces is **superseded** — kept on disk, off the default view;
- **archived** — explicitly retired by the scientist.

## Intent & lifecycle (keep-all; never auto-delete)

When the scientist asks to run a heavy analysis, the orchestrator fingerprints the requested params against the current `data_version` and checks the registry:

- **Fingerprint already cached** → say so ("this exact result already exists") and **reuse it** (skip the recompute) unless the scientist asks to force-regenerate.
- **New params (new fingerprint)** → a **new result**. **Keep existing results by default**: mark the prior of the same problem `superseded`, add the new as `current`. **Never delete automatically.**
- **Removal is explicit** — the scientist prunes on request. A result **referenced by a finding is protected**: refuse to delete it until the finding is repointed or retired. Provenance integrity outranks disk savings.

## Selection for visualization

A figure request **names the result** (fingerprint or label); if unspecified, use the analysis's **current** result. The chosen id is recorded in the figure's provenance **and** in the finding's `provenance.result_id`, so a finding pins the exact result it was built from (and a blinded verifier can reload it).

## Enforcement (enforcement-map.md)

| Rule | Enforced by |
|---|---|
| A CPU-heavy result is persisted via `save_cached_result` + registered in `results/manifest.md`; the compute/figure split is used | **Stats-reviewer** |
| A re-run with identical params reuses the cache (fingerprint match), not a silent recompute; a new result is a genuine param/data change | **Stats-reviewer** + orchestrator behavior |
| Result figures load a **named / current** cached result (not a fresh recompute); the figure records which id | **Figure-reviewer** |
| The finding pins the result it was built from (`provenance.result_id`) | **Stats-reviewer** + findings-manager |
| Keep-all: no result auto-deleted; a finding-referenced result is protected from pruning | **Orchestrator behavior** + findings-manager |
