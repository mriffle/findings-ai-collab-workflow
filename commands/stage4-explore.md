---
name: stage4-explore
description: "Stage 4 — Explore ⇄ record findings (the heart). Run boilerplate analysis to spark discussion, investigate with the scientist, and capture every substantive insight as a finding the moment it emerges."
---

# Stage 4 — Explore ⇄ record findings  (the heart)

**Precondition (hard):** `state/workflow.json` shows `integrity_gate.passed: true`. **If it is not true, refuse** — no analysis before the integrity gate passes (doc 02.3). Tell the scientist to complete `stage3-loaders`. (A hook also enforces this.)

This is the open loop the whole system exists to capture. With understanding established and the read verified, explore the data *with* the scientist and turn insight into durable findings.

## The loop

1. **Spark discussion with vetted boilerplate.** Run standard analyses from `lib/` (differential abundance via moderated models, nonparametric tests, feature finding, leakage-safe classifiers, regression, dimensionality reduction) to give the scientist something to react to. Prefer calling `lib/` over generating fresh statistics code — models get assumptions and missingness handling wrong in ways that look fine.
2. **Go back and forth.** A plot looks interesting, a protein group invites a question, the scientist points at a heatmap cluster. Investigate by composing or writing analysis — held to the coding and statistical conventions (no bare p-values; effect size + CI + a named correction; report all tests run; no leakage; CV matched to the generalization target; label-shuffle null for classifiers; small-n is exploratory). Dispatch the **statistician**/**coder** and their reviewers where available.
3. **Capture every substantive insight as a finding — the moment it emerges.** Dispatch the **findings-manager** to record it. *What counts:* a tangible, specific, evidence-bearing observation — if it has an effect, a statistic, or a concrete claim someone might later cite, record it. **Bias toward capturing too much**; capture is cheap and low-bar (`candidate`). Give the scientist a brief, non-disruptive notice ("recorded as finding 0042"). Do not stop the flow to apply rigor — rigor is applied at promotion (Stage 5).
4. **Keep the exploration log honest.** Append what was looked at and discarded to `findings/exploration-log.md` — this is the multiplicity context that informs each finding's caveats. Mark findings `exploratory` by default; confirmatory requires held-out/orthogonal data (Stage 5).

## Skepticism calibration

**Generous here.** Capture freely as candidates. A main agent that doubts everything in real time kills the exploration the system exists to capture. The ruthless skepticism lives at promotion (Stage 5), not at every breath.

## Provenance discipline (so findings stay regenerable)

Every finding's numbers must be regenerable: pin `data_version`, the script (path + commit), params, the locked environment, the `lib/` version, and any seed. A finding may be **promoted/validated only against a script in `scripts/promoted/`** — scratch work is fine for candidates, but promotion requires the script to pass tests/types/lint and move to `promoted/`.

## When candidates mature

Run `stage5-validate` to put a finding through independent validation. Validation runs continuously as candidates mature — you don't have to finish exploring first.
