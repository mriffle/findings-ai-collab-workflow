# 04 — Agents, Skills, and the Research Subsystem

## 4.1 Skill vs agent

- A **skill** is a reusable procedure — *how to do something well* (e.g. how to research a protein properly, how to run a leakage-safe classifier, how to render a publication-ready figure). Skills are model-invoked by description and shipped in `skills/`.
- An **agent** is a context-isolated worker that invokes skills to do a job. Context isolation keeps each worker's reasoning clean and keeps the orchestrator's context from filling with research detail, code, or raw data.

The two combine: an agent (e.g. the protein researcher) invokes a skill (the protein-research procedure) to produce an artifact (a research finding).

## 4.2 Agent roster and contracts

Each agent has a defined job, inputs, outputs, and isolation rationale. Outputs are files or structured returns, never just chat.

| Agent | Job | Inputs | Outputs |
|---|---|---|---|
| **Orchestrator** | The main agent collaborating with the scientist; drives the workflow; dispatches subagents | Scientist dialogue, project state | Workflow progression, dispatched tasks |
| **Findings manager** | Owns the findings graph and manifest (doc 03.8) | New/updated findings, manifest | Finding files, manifest, consistency + staleness flags |
| **Verifier** | Blind independent validation (doc 03.5) | Mechanically-derived task (answer stripped), data | Validation verdict + concordance result |
| **Researcher(s)** | Thoroughly research a bounded topic | A topic + scope from the librarian | A research-finding doc with references |
| **Librarian** | Controls research information; dispatches researchers | Research questions, research corpus | Answers-with-references; research dispatch decisions |
| **Research reviewer** | Independently checks research for factual accuracy; verifies every reference | A research finding | Accept / revise verdict; reference-check report |
| **Coder** | Write Python analysis scripts | A question, project state, `lib/` | A script (initially `scratch/`) |
| **Code reviewer** | Review scripts for correctness, reproducibility hygiene, data-handling bugs | A script | Pass/fail against coding + correctness conventions |
| **Statistician** | Perform statistical analysis | A question, data | Results + the analysis script |
| **Stats reviewer** | Review analysis against statistical conventions | An analysis | Pass/fail against statistical conventions (doc 05) |
| **Figure generator** | Produce publication-ready figures | A spec, data, color registry | SVG + 300 DPI PNG + legend doc (doc 06) |
| **Figure reviewer** | Review the rendered PNG for accuracy and standards | The rendered figure | Pass/fail; required corrections |
| **Writer(s)** | Draft report sections from findings | Selected findings, outline | A report section |
| **Report reviewer / editor** | Claim-source check + coherence pass | Draft sections | Verified, coherent report (doc 07) |

Notes:

- **Context isolation is the point.** Research, code, and figure work happen in subagents so the orchestrator's context stays focused on the science and the dialogue.
- **Generator/reviewer pairing** is used wherever an artifact's correctness matters: code, statistics, research, figures, reports. The reviewer checks the *artifact*, not just the intent.
- The **verifier** is deliberately starved of history (doc 03.5).

## 4.3 Skills

- **Research skills:** scientific-publication researcher; protein researcher (UniProt, PDB/AlphaFold, STRING, GO, with verification); source-code researcher (grounding methodological claims in what tools such as DIA-NN or limma actually compute — methods sections routinely misdescribe this, so this is independently valuable).
- **Statistical boilerplate (the standard library, in `lib/`):** vetted, tested functions for the analyses we already use — differential abundance (limma/MSstats-style moderated models), nonparametric tests, feature selection, classifiers with leakage-safe cross-validation, regression, dimensionality reduction. Agents **prefer calling these over generating fresh statistics code**, because models get test assumptions and missingness handling wrong in ways that look fine. Flip side: a wrong default here is wrong everywhere, so `lib/` is itself scientifically reviewed and version-recorded, and findings record which version they used.
- **Visualization library (in `lib/`):** the standard descriptive and QC plots and the publication-ready figure machinery (doc 06).
- **Finding skills:** the finding template and the mechanical verification-task builder (doc 03).
- **Report skills:** writing conventions, report structure, the assembly/coherence procedure (doc 07).

## 4.4 The research subsystem

Research is isolated so it does not fill analytic agents' context, and controlled so it is not redundant or unverified.

- The **librarian** is the controller. It knows which research findings exist, judges whether they already cover the current question, determines what research is still needed, and dispatches researchers with a bounded scope. It answers questions by drawing on the research corpus.
- **Researchers** thoroughly explore one bounded topic (a disease, protein, gene, pathway, publication, or a piece of software's source) and produce a **research-finding document**.
- The **research reviewer** independently checks each research finding for factual accuracy and **verifies every reference** before it enters the corpus.

### Research-finding document

Stored in `research/`, structured like a finding but for external knowledge: topic, summary, detailed findings, and a **mandatory references section** (specific papers, with identifiers, and/or specific web sources). A research finding without verified references is not accepted into the corpus.

## 4.5 The references invariant (hard rule)

References are required wherever an external-knowledge or interpretive claim is made:

1. **Any research material a researcher saves** must include references to its sources.
2. **Any data the librarian provides** from research must include references.
3. **The background/interpretive content of any finding** (doc 03) must include references.

And, by the generator/reviewer principle, **every reference is fact-checked by the research reviewer** — both that it exists and that it supports the claim attributed to it. Hallucinated citations are a known failure mode; this invariant plus review is the countermeasure. Software/tool citations (with versions) are drawn for free from the locked environment (doc 05) and the source-code researcher.
