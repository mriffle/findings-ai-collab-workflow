---
name: stage5-validate
description: "Stage 5 — Independent validation. Promote a candidate finding toward validated through the blinded independent-validation gate: build the answer-stripped task, dispatch the verifier, check pre-specified concordance, and record the outcome."
argument-hint: "[finding id, e.g. 42]"
---

# Stage 5 — Independent validation

**Precondition:** `state/workflow.json` shows `integrity_gate.passed: true` (validation *assumes* the integrity gate — it does not substitute for it). Target finding: `$ARGUMENTS` (a finding id), or ask which finding to validate.

Promote a candidate toward `validated` through independent re-derivation. The bar for `validated` is defined in `conventions/findings.md` §4 and is fixed as: **integrity sign-off + computational reproduction + blinded analytic replication (under a pre-specified concordance criterion) + the phase bar** — with data replication required only to claim `confirmatory`.

## Procedure

1. **Pick the mode.** `analytic_replication` (same data; the lower bar; finding stays `exploratory`) or `data_replication` (held-out/orthogonal data; required to claim `confirmatory`).
2. **Build the blind task** with the **verification-task-builder** skill: derive the verification task mechanically from the finding's structured fields, with `verdict`, `evidence`, and `summary` **stripped**, phrased as a neutral question ("what is the effect of X on Y? report effect size, CI, BH-corrected p") that never reveals direction or magnitude.
3. **Pre-specify the concordance criterion — before the verifier runs** — and record it in the finding's `validation.<mode>.concordance_criterion` (via the findings-manager). Deciding "close enough" after seeing the result reintroduces forking paths.
4. **Dispatch the `verifier` agent** with the blind task. It has no conversation history (the clean context is the blind) and re-derives the result from scratch. It must never be given or seek the recorded answer.
5. **Compute computational reproduction** as the cheap precondition: re-run the exact promoted script on the exact data; expect identical numbers. (Necessary, not sufficient — it is not independent.)
6. **Judge concordance yourself** (the verifier doesn't): compare the verifier's independent result to the recorded `evidence` under the pre-specified criterion.
7. **Record the outcome** via the findings-manager into `validation.<mode>` (`status`, `by`, `date`, `concordance_criterion`, `concordance_result`).

## Promotion to `validated`

The findings-manager enforces the full bar before writing `status: validated`:

- `integrity_signoff: true` for the finding's `data_version`;
- computational reproduction passed;
- **blinded analytic replication passed** under the pre-specified criterion;
- the phase bar satisfied (if `phase: confirmatory`, data replication passed against a disjoint dataset);
- the finding links to a **promoted** script.

**Promotion is never silent and requires the scientist's acceptance** (doc 02.8). When the bar is met, surface it; on acceptance, finalize `validated`.

## On failure

If the verifier's result is discordant, do not promote. Record `failed`, consider moving the finding toward `invalidated` (the findings-manager will cascade re-review to dependents), and discuss with the scientist. A discordant result is information, not a setback.

## Workflow state

The first time validation runs for a project, raise `state/workflow.json` `current_stage` to **5** (highest stage reached) and bump `updated`. Validation is a continuous loop, so there is no per-stage `*_done` flag — raising `current_stage` keeps `status` from perpetually rendering "Stage 4" once validation is underway.

## Then

Close with the concrete next step (project `CLAUDE.md`, *Leave the scientist with a next step*) — validation is a **within-loop** step, so the thread returns to exploration:

- **On a pass + acceptance** — say the finding is now `validated`, then name the next thing: the next matured candidate to validate (give its id from the manifest), or back to `stage4-explore` for the analysis this result invites.
- **On a discordant result** — the next step is the *reconciliation*, not a retreat: discuss the divergence with the scientist, and name the concrete follow-up (re-examine the analysis, consider `invalidated`, and re-review the dependents the findings-manager just cascaded a flag to).

**Do not suggest moving on to reporting** — Stage 4 is closed by the scientist, not by you (the Stage 4 exception in the project `CLAUDE.md`).
