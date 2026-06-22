# 01 — Overview and Principles

## 1.1 What we are doing

We are building a formalized, repository-encoded workflow in which a scientist and an AI agent (Claude Code) analyze a scientific dataset together, and in which the durable output is a curated, interconnected, independently validated set of **findings** organized as a graph. The workflow is packaged as a Claude Code plugin so that any user can install it with a single command and run it against their own data (see doc 08).

The system covers the full arc of an analysis: understanding the metadata, understanding the data, building and verifying data loaders, framing the science, exploring the data in collaboration with the scientist, capturing findings as they emerge, validating them independently, and compiling them into disseminable reports.

Proteomics — sample-by-feature quantitative matrices with structured missingness — is the proving ground because it is the domain we know best and exercises every hard part of the design. Nothing in the architecture is proteomics-specific by intent; it generalizes to any data-analytic discipline.

## 1.2 Why we are doing it — the problem

Exploratory analysis is where scientific insight is created and where it is most quietly compromised. Four hazards recur, and all four are normally invisible:

1. **Insights are lost.** Observations surface mid-conversation and are forgotten; in long sessions early discoveries are buried, and across sessions they are gone.
2. **Multiplicity is hidden.** Every "I wonder if protein X is involved" is a new, data-dependent test. The analyst remembers the paths taken, not the ones tried and abandoned, so false-discovery risk inflates invisibly — the garden of forking paths.
3. **Results are not reproducible.** A figure in a slide deck rarely carries the data version, code, parameters, and environment that produced it; data and code drift, and results silently go stale.
4. **Speculative and solid claims blur.** Nothing distinguishes "confirmed against held-out data" from "looked interesting once," and the distinction is lost entirely by the time of write-up.

The contribution of this project is to convert each implicit hazard into an explicit, tracked, auditable safeguard. The findings paradigm is the mechanism that does the converting.

## 1.3 The paradigm shift

An AI assistant is normally used as a *transient analyst*: the unit of value is an answer in a chat, and it evaporates. We invert this. The AI becomes a *curator of trustworthy knowledge*: it continuously captures, structures, validates, and curates the findings produced during analysis. The conversation stays ephemeral; the findings are durable, regenerable, reviewable, and connected. This reframing — and the rigor machinery that makes the captured findings trustworthy — is the project's reason to exist.

## 1.4 Design principles

These are the cross-cutting commitments the rest of the spec serves. Every concrete rule downstream should trace to one of these.

1. **Code is the source of truth; conversation is ephemeral.** A finding is never "the agent observed X." It is a script, a data version, and a result. Figures and result tables are caches; the script that produced them is the artifact.
2. **Every finding is regenerable.** Each pins data version (hash), script (commit), parameters, and the locked environment, so it can be re-run by anyone, later, to the same numbers.
3. **State is persistent and structured, not remembered.** The understanding built during the workflow lives in files any agent rehydrates from (doc 02), not in a conversation that ends.
4. **Correctness is foundational and upstream of everything.** Data-loading fidelity is tested and verified before any analysis (doc 05). A silent read error is a common-mode failure that defeats independent validation, so it cannot be delegated to the validation gate.
5. **No finding is validated without independent validation.** Promotion to *validated* requires a check performed independently of the path that produced the finding (doc 03).
6. **Skepticism lives in the gates, calibrated by phase.** Generous during exploration so candidate findings are captured freely; ruthless at promotion. A main agent that doubts everything in real time kills the exploration the system exists to capture.
7. **Multiplicity is made honest.** The system tracks what was explored and separates exploratory (hypothesis-generating) findings from confirmatory ones, so the false-discovery burden is visible.
8. **The findings form a graph.** Findings are nodes; typed relationships are edges. Normalized entity references make the graph queryable (doc 03).
9. **Conventions are an enforcement spec.** A convention is only real if something checks it — preferably a deterministic hook (doc 05, 08).
10. **Humans decide the science.** The agent proposes, captures, validates, and curates; the scientist decides what matters and what crosses the bar, at defined checkpoints (doc 02).

## 1.5 Novelty and prior art (for the eventual publication)

The novelty is the paradigm, not the domain: a workflow in which a scientist and an AI collaborator turn exploratory analysis into a curated, provenance-bound, independently validated, multiplicity-honest findings graph, distributed so others can adopt and reproduce it.

Prior art to situate against:

- **Nanopublications and micropublications** model a claim-plus-evidence-plus-provenance unit (as RDF triples that aggregate into knowledge graphs). Our finding is closely analogous, in a lightweight, markdown-native, AI-collaborative form. This is the clearest lineage to cite.
- **W3C PROV** supplies the standard vocabulary for provenance; our provenance fields are a pragmatic specialization.
- **FAIR** principles (and FAIR-for-software/workflows) motivate findability, accessibility, interoperability, reuse — entity normalization and the manifest serve these.
- **Research Object / RO-Crate** packaging is the natural comparison for bundling analysis + outputs + metadata; the plugin + project layout is a lighter, executable cousin.

What none of these solved is the interactive, AI-collaborative, exploratory version this project targets, and most are far heavier.

**Evaluation needs (decide early, since it shapes what the system records):** reproducibility of findings (re-run to identical numbers); false-discovery behavior under the exploration tracking; the effect of independent validation on finding survival; finding quality judged by domain experts; and one or more worked case studies on real proteomics datasets. Distribution-via-plugin is an adoption/reproducibility contribution, not the scientific novelty; frame it as "how others adopt and reproduce this."

## 1.6 Glossary

- **Finding** — a first-class, numbered, structured document capturing one substantive insight, with evidence, provenance, caveats, status, and links (doc 03).
- **Candidate finding** — a captured insight not yet promoted; cheap to record, low bar.
- **Validated finding** — a finding that has cleared the independent-validation gate and the evidentiary bar for its status.
- **Findings graph** — the graph of findings (nodes) and their typed epistemic relationships (edges).
- **Knowledge graph** — the findings graph integrated with normalized domain entities (proteins, genes, pathways, diseases) that findings reference; the queryable layer (doc 03).
- **Entity normalization** — referencing domain entities by canonical identifier (UniProt, HGNC, Reactome, MONDO, etc.) rather than free text.
- **Promoted script** — a reviewed, tested script in `scripts/promoted/`; the only kind a finding may link to.
- **Project state** — `PROJECT.md`, `METADATA.md`, `DATA_DESCRIPTION.md`, and registries that persist the agent's verified understanding (doc 02).
- **Gate** — a precondition that must pass before the workflow proceeds; ideally a deterministic hook.
- **Independent validation** — re-derivation of a finding by a process (typically a clean-context subagent) independent of the one that produced it (doc 03).
