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
| Raw data read-only | 05.1 | **Hook** | `guard_readonly_data.sh` — blocks Write/Edit under `data/`; best-effort Bash block |
| No `integrity_signoff: true` / `validated` before the integrity gate passes | 02.3, 05.4 | **Hook** + human sign-off | `guard_findings.sh` reads `state/workflow.json .integrity_gate.passed`; gate flipped only in `stage3-loaders` after sign-off |
| No exploratory analysis before the gate (stage ordering) | 02.3 | **Orchestrator behavior** + command precondition | `stage4-explore` hard-refuses unless the gate passed; project `CLAUDE.md` ordering rule |
| Usable Python ≥ 3.11 project env before any analysis | 05.2 | **Command precondition** + `setup-env` | `stage3-loaders` live-verifies a working interpreter (not the stored flag); `setup-env` detects/installs project-local Python with transparent consent. Not a hook — "is this env usable?" needs a live probe, like stage ordering |
| Finding links only to a promoted script | 03, 05 | **Hook** + findings-manager | `guard_findings.sh` blocks `validated` linking to `scripts/scratch/`; findings-manager enforces at promotion |
| Script not promoted until **lint + types** pass | 05.4 | **Hook** | `guard_promotion.sh` — ruff + mypy on `scripts/promoted/*.py` (PreToolUse blocks; PostToolUse warns) |
| Script not promoted until **tests** pass | 05.4 | **Code-reviewer** | tests aren't run in-hook (slow/unsafe); reviewer verifies unit/property/planted-truth/edge |
| One analysis script per task; shared code reused, no duplicates | project rule | **Code-reviewer** | script registry (`scripts/manifest.md`) + per-script `__script_meta__` header |
| Promoted script imports only promoted modules | project rule | **Code-reviewer** | a `scripts/promoted/` script may import other promoted modules only — never scratch/unreviewed code, so a finding's computation never silently depends on unreviewed code (`conventions/script-registry.md`) |
| Record-the-finding during exploration | 03.9 | **Orchestrator behavior** | project `CLAUDE.md` always-on recording; findings-manager assigns/writes |
| Loader tested + load verified + pairing exact | 05.4 | **Code-reviewer** + integrity-gate checklist + human sign-off | `stage3-loaders` checklist; scientist signs off |
| Assumptions tested in code; results recorded | 05.4 | **Orchestrator behavior** + code/stats reviewers | Stages 1–2 test inferences rather than assert them |
| No bare p; correction named; effect+CI present | 05.3 | **Stats-reviewer** | checks every analysis against `conventions/statistics.md` |
| No leakage; CV matched to target; label-shuffle null | 05.3 | **Stats-reviewer** | preprocessing inside folds; group folds; permutation null |
| Canonical tests; moderated models for DE | 05.3 | **Stats-reviewer** | seed from `lib/` moderated-model template (*pending phase E*; until then, stats-reviewer only) |
| All tests run are reported (→ exploration log) | 05.3, 03.6 | **Stats-reviewer** + orchestrator behavior | exploration log appended in Stage 4 |
| Small-n / confounded → exploratory | 05.3, 03.6 | **Stats-reviewer** + findings-manager | `phase` field |
| Every reference exists and supports its claim | 04.5 | **Research-reviewer** | fact-checks each reference before it enters the corpus |
| Figure rendered, reviewed, dual-exported, legend present | 06 | **Figure-reviewer** (+ optional dual-export hook) | reviews the PNG render; checks SVG+PNG+legend trio |
| ≤8 categorical colors; explicit strategy beyond | 06.6 | **figure-reviewer** (shared-module guard *pending phase E*) | the seeded figure template will detect overflow once `lib/` ships; until then the figure-reviewer is the only check |
| No finding `validated` without independent validation | 03.5 | **Verifier** + findings-manager | blinded re-derivation; manager enforces the full `validated` bar |
| Report claims map to findings; status/caveats propagate; no invented refs | 07.5 | **Report-reviewer** | claim-source check at compile time |

## Human checkpoints (the scientist decides)

| Point | Spec |
|---|---|
| Confirm metadata understanding (end Stage 1) | 02.8 |
| Confirm sample↔metadata pairing (esp. fuzzy matches) | 02.8 |
| Integrity-gate sign-off (end Stage 3) | 02.8 |
| Accept a finding's promotion to `validated` | 02.8 |
| Select which findings a report is about | 02.8, 07.4 |

## Honest notes on hook scope

- The hooks enforce what is **deterministic and safe** to decide from a single tool-use event: read-only `data/`, the integrity precondition on finding writes, the promoted-link rule, and static promotion checks (lint/types). They are **scoped to initialized projects** (`state/workflow.json` present) and **fail open** if their tooling (jq/ruff/mypy) is absent, so they never wedge a session over their own dependencies.
- They do **not** attempt to detect "is this Bash command exploratory analysis?" — that can't be cleanly separated from legitimate Stage 3 loader/QC work by a path or string match, so stage ordering is carried by command preconditions + orchestrator behavior instead. This split is deliberate: hooks for the crisp invariants, reviewers/behavior for the judgment calls.
- YAML-field checks in `guard_findings.sh` are line-oriented heuristics; the **findings-manager is the authoritative enforcer** of the finding schema and the `validated` bar. The hook is a deterministic backstop, not the whole gate.
- Several rows credit a **`lib/`-seeded guard** (the >8-category raising guard; the leakage-safe / moderated-model template defaults) as a deterministic co-enforcer. **`lib/` is not built yet — it is phase-E work.** Until those templates ship, the **reviewer agent named in the row is the sole active enforcer**; the rows carry a *pending phase E* marker so the map never overstates current coverage. This is the one place the map's enforcement is aspirational rather than live.
