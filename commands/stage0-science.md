---
name: stage0-science
description: "Stage 0 — State the science. Interview the scientist about the study and write state/PROJECT.md, the framing that shapes everything downstream."
---

# Stage 0 — State the science

Capture the research framing before any data is touched. This is the first stage; it has no precondition beyond an initialized project (`state/workflow.json` exists — if not, run `init` first).

## Do this

Interview the scientist — **one question at a time, never a form dump**. Ask, wait for the
answer, then ask the next; each answer shapes what is worth asking next. This stage is
mostly *elicitation*, so the questions are open-ended prose, not option lists (see *How to
ask* in the project `CLAUDE.md`). Establish:

- **Domain** — what field/system this is, in enough depth to interpret findings.
- **What is being examined and why** — the biological/scientific question motivating the study.
- **Experimental design** — conditions, groups, replication, timepoints, batches; what was manipulated and what was measured.
- **Scientific goals** — what we want to find out. Be concrete about the hypotheses or discovery aims.

Where a question *does* resolve to a small set of known options (a design is one of a few
standard shapes; the goal is discovery vs hypothesis-testing), offer them with
**`AskUserQuestion`**, recommended option first — but never force an open-ended answer into
buttons.

Write **`state/PROJECT.md`** capturing all of the above in clear prose. This is a living document — note that it will be updated as understanding deepens.

## Guard against motivated reasoning

Stating goals up front improves relevance but **invites motivated reasoning**. Record the goals faithfully, but do not let stated hopes bias what you later report. The skepticism gates (generous in exploration, ruthless at promotion) and independent validation exist partly to counter this. Note this tension briefly in `PROJECT.md` so it stays visible.

## Then

- Update `state/workflow.json`: set `science_done: true`, `current_stage: 1`, bump `updated`.
- Tell the scientist the next step: **Stage 1 — Understand the metadata** (`stage1-metadata`). You'll need to know where the metadata file is. Stage 1 is the first stage that runs code (validity checks, cohort characterization, confounding statistics), so it needs a working Python ≥ 3.11 — if the project environment isn't set up yet, run `setup-env` before (or at the start of) Stage 1.

Do not proceed to examine data. Understanding precedes analysis. And do not start Stage 1 on your own — Then stop.** Naming the next stage is where your turn ends — the scientist starts it, by running the command or asking you to. Recording `current_stage` is bookkeeping, not permission (project `CLAUDE.md`, *Stages advance on the scientist's word, not yours*).
