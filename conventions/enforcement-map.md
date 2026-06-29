# Convention — The Enforcement Map

*Spec source: doc 05.5. The system's organizing principle is **"a convention is only real if something checks it."** This is the master index: every rule maps to a concrete enforcer. Where a check can be made deterministic and safe it is a **hook**; otherwise it is a **reviewer-agent** check, an **orchestrator behavior** (the project `CLAUDE.md`), or a **human checkpoint**.*

## Enforcer types

- **Hook** — fires deterministically on a tool-use/lifecycle event; does not depend on the model remembering. The highest-stakes, cleanly-decidable gates. Implemented in `hooks/`.
- **Reviewer agent** — a generator/reviewer pair where the reviewer checks the *artifact* (code, statistics, research, figures, reports). Used where correctness is judgment-heavy.
- **Orchestrator behavior** — standing instruction in the project `CLAUDE.md`; guidance, not a hard gate.
- **Human checkpoint** — an explicit point where the scientist decides.

## The map

| Rule | Spec | Enforced by | Mechanism |
|---|---|---|---|
| Raw data read-only | 05.1 | **Hook** | `guard_readonly_data.py` — blocks Write/Edit under `data/`; best-effort Bash block |
| No `integrity_signoff: true` / `validated` before the integrity gate passes | 02.3, 05.4 | **Hook** + human sign-off | `guard_findings.py` reads `state/workflow.json .integrity_gate.passed`; gate flipped only in `stage3-loaders` after sign-off |
| No exploratory analysis before the gate (stage ordering) | 02.3 | **Orchestrator behavior** + command precondition | `stage4-explore` hard-refuses unless the gate passed; project `CLAUDE.md` ordering rule |
| Usable Python ≥ 3.11 project env before any code | 05.2 | **Command precondition** + `setup-env` | `stage1-metadata` live-verifies a working interpreter (not the stored flag) — it is the first stage that runs code (validity checks, cohort characterization, confounding stats); `stage3-loaders` re-verifies at the integrity gate; `setup-env` detects/installs project-local Python with transparent consent. Not a hook — "is this env usable?" needs a live probe, like stage ordering |
| Finding links only to a promoted script | 03, 05 | **Hook** + findings-manager | `guard_findings.py` blocks `validated` linking to `scripts/scratch/`; findings-manager enforces at promotion |
| Script not promoted until **strict lint + types** pass | 05.4 | **Hook** | `guard_promotion.py` — `ruff check` (strict rule set) + `mypy` strict on `scripts/promoted/*.py`; strictness from the project config (PreToolUse blocks; PostToolUse warns) |
| Script not promoted until **tests** pass | 05.4 | **Code-reviewer** | tests aren't run in-hook (slow/unsafe); reviewer verifies unit/property/planted-truth/edge |
| One analysis script per task; shared code reused, no duplicates | project rule | **Code-reviewer** | script registry (`scripts/manifest.md`) + per-script `__script_meta__` header |
| Promoted script imports only promoted modules | project rule | **Code-reviewer** | a `scripts/promoted/` script may import other promoted modules only — never scratch/unreviewed code, so a finding's computation never silently depends on unreviewed code (`conventions/script-registry.md`) |
| Record-the-finding during exploration | 03.9 | **Orchestrator behavior** | project `CLAUDE.md` always-on recording; findings-manager assigns/writes |
| Loader tested + load verified + pairing exact | 05.4 | **Code-reviewer** + integrity-gate checklist + human sign-off | `stage3-loaders` checklist; scientist signs off |
| Assumptions tested in code; results recorded | 05.4 | **Orchestrator behavior** + code/stats reviewers | Stages 1–2 test inferences rather than assert them |
| No bare p; correction named; effect+CI present | 05.3 | **Stats-reviewer** | checks every analysis against `conventions/statistics.md` |
| No leakage; CV matched to target; label-shuffle null | 05.3 | **Stats-reviewer** | preprocessing inside folds; group folds; permutation null |
| Normalization method recorded; scale respected (no double-log) | 05.3 | **Stats-reviewer** + the `normalize` template's runtime scale guard | reviewer checks the method is recorded in provenance; the template's `_require_linear` / scale-tag refuses double-logging at runtime — but only if the adapted project copy keeps the guard (see honest note 5) |
| Batch correction is batch-label-only; corrected **and** uncorrected both reported | 05.3 | **Stats-reviewer** | `batch-correct-combat` passes ComBat the batch label only (`X=None`, by design); reviewer checks the key analysis is reported both ways and prefers batch-as-covariate for testing |
| Perfect/strong batch↔covariate confounding surfaced + signed off before correcting | 05.3 | **Stats-reviewer** + orchestrator behavior + human sign-off | `assess_batch_confounding` emits a graded `BatchConfoundingWarning`; Stage 3/4 surfaces it; the scientist signs off before correction |
| Canonical tests; moderated models for DE | 05.3 | **Stats-reviewer** | seed from `lib/` moderated-model template (*pending phase E*; until then, stats-reviewer only) |
| All tests run are reported (→ exploration log) | 05.3, 03.6 | **Stats-reviewer** + orchestrator behavior | exploration log appended in Stage 4 |
| Small-n / confounded → exploratory | 05.3, 03.6 | **Stats-reviewer** + findings-manager | `phase` field |
| Cohort characterized; material imbalance/skew/confound recorded as a caveat finding | 02.1, 05.3 | **Human checkpoint** (Stage 1) + **Stats-reviewer** + **Report-reviewer** + findings-manager | Stage 1 characterizes the metadata and records `kind: caveat` findings; the scientist confirms at the Stage 1 checkpoint; the stats-reviewer checks a confounded contrast is modelled (covariate/stratify) not ignored; the report-reviewer verifies caveats propagate into the report. No clean hook — "did you *notice* the imbalance?" isn't decidable from a tool event |
| Control samples identified; experimental/control split confirmed and certified | 02.1, 02.8 | **Human checkpoint** (Stage 1) + integrity-gate sign-off (Stage 3) | Stage 1 hunts for control samples (pools/references/standards/blanks) from an explicit role column or naming/`no-group` inference, documents the binary split + rule in `state/METADATA.md`, and confirms it at the Stage 1 checkpoint; Stage 3 carries the label, reconciles the per-class counts, and certifies the experimental subset at the integrity gate. No clean hook — "did you *find* the controls?" isn't decidable from a tool event |
| Analysis runs on the experimental subset; control samples excluded and the exclusion recorded | 05.3 | **Stats-reviewer** | reviewer checks every biological contrast is computed on the experimental subset (the Stage-1 split) and that `provenance.params` records the analyzed sample set / excluded-control count |
| QC/descriptive figures render control samples separately from experimental | 06 | **Figure-reviewer** | reviewer confirms controls are shown in their own panels or visibly distinct, not silently pooled into the experimental distributions. Two documented exceptions show them together *because labeled*: the `sample-correlation` heatmap (Sample Type stripe) and the `id-depth` bar chart (bar color) — the cross-class comparison is the deliverable |
| Every reference exists and supports its claim | 04.5 | **Research-reviewer** | fact-checks each reference before it enters the corpus |
| Figure rendered, reviewed, dual-exported, legend present | 06 | **Figure-reviewer** + the `figure-io` template | `figure-io`'s `save_figure` dual-exports the figure (`<base>.{svg,png}`) and a companion legend image (`<base>.legend.{svg,png}`) kept out of the plot (shipped); the figure-reviewer reviews the PNG render and confirms the figure + legend image. Mechanical only for a project script that *uses* `save_figure` (honest note 4) |
| Okabe–Ito; category colors from the registry; consistency | 06.5 | **Figure-reviewer** + the `okabe-ito-colors` template | `okabe-ito-colors` reads/extends `state/color_registry.json` so a `(category, value)` keeps one Okabe–Ito color across figures (shipped); the figure-reviewer verifies a script routes its colors through it |
| ≤8 categorical colors; explicit strategy beyond | 06.6 | **Figure-reviewer** + the `okabe-ito-colors` guard | `okabe-ito-colors.assign_colors` raises `CategoricalPaletteExceededError` past 8 categories (shipped) — the deterministic enforcer; the figure-reviewer verifies the script uses it and that the chosen alternative encoding (facet / second channel / grouped tail / sequential) is sound |
| No finding `validated` without independent validation | 03.5 | **Verifier** + findings-manager | blinded re-derivation; manager enforces the full `validated` bar |
| Report claims map to findings; status/caveats propagate; no invented refs | 07.5 | **Report-reviewer** | claim-source check at compile time |

## Human checkpoints (the scientist decides)

| Point | Spec |
|---|---|
| Confirm metadata understanding — incl. surfaced imbalances/confounds (end Stage 1) | 02.8 |
| Confirm the experimental/control sample split + the rule deriving it (end Stage 1) | 02.8 |
| Confirm sample↔metadata pairing (esp. fuzzy matches) | 02.8 |
| Integrity-gate sign-off (end Stage 3) | 02.8 |
| Accept a finding's promotion to `validated` | 02.8 |
| Select which findings a report is about | 02.8, 07.4 |

## Honest notes on hook scope

- The hooks enforce what is **deterministic and safe** to decide from a single tool-use event: read-only `data/`, the integrity precondition on finding writes, the promoted-link rule, and static promotion checks (lint/types). They are **scoped to initialized projects** (`state/workflow.json` present) and **fail open** — on a malformed event, a missing project interpreter, or (for promotion) absent `ruff`/`mypy` — so they never wedge a session over their own dependencies. The guards are **Python** (stdlib-only to run; no `bash`/`jq`), invoked via the project `./.venv` interpreter so they run identically on Windows, macOS, and Linux; see `hooks/README.md`.
- They do **not** attempt to detect "is this Bash command exploratory analysis?" — that can't be cleanly separated from legitimate Stage 3 loader/QC work by a path or string match, so stage ordering is carried by command preconditions + orchestrator behavior instead. This split is deliberate: hooks for the crisp invariants, reviewers/behavior for the judgment calls.
- YAML-field checks in `guard_findings.py` are line-oriented heuristics; the **findings-manager is the authoritative enforcer** of the finding schema and the `validated` bar. The hook is a deterministic backstop, not the whole gate.
- Some rows credit a **`lib/`-seeded guard** as a deterministic co-enforcer. The **figure machinery has shipped** (phase E in progress): `okabe-ito-colors` (the >8-category raising guard + the color-registry consistency) and `figure-io` (dual export + a separate legend image) are built, so the figure rows no longer carry a *pending phase E* marker. Still pending phase E: the leakage-safe CV and moderated differential-abundance template defaults — until they ship, the **reviewer agent named in those rows is the sole active enforcer** and they keep the *pending phase E* marker so the map never overstates coverage.
- The shipped templates' **runtime guards** (the `normalize` scale-tag check; the `batch-correct-combat` log-scale / batch-label-only / confounding checks; the `okabe-ito-colors` `CategoricalPaletteExceededError`; the `figure-io` no-legend / dual-export trio) are real enforcers, but of a fourth kind: they live *inside the template code*, so they only protect a project that **keeps the guard when adapting the copy** and actually routes its work through it (a figure script that hand-rolls `savefig` or its own colors gets neither). That is why the reviewer agent is still named as co-enforcer on those rows — the reviewer is what catches a project that strips or bypasses the guard. A runtime guard is not a substitute for the review.
