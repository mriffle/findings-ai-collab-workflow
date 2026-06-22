# skills/

Reusable, model-invoked procedures — *how to do something well* (spec doc 04). Each skill is a directory containing `SKILL.md` (YAML frontmatter `name` + `description`, then markdown instructions) and optional supporting files.

A skill is invoked by its description when relevant; an agent invokes a skill to produce an artifact (e.g. the protein-researcher agent invokes the protein-research skill to produce a research finding).

Skills (doc 04.3):

| Skill | Built | Procedure |
|---|---|---|
| `verification-task-builder` | ✅ | Build a blind, answer-stripped validation task + pre-specify concordance (doc 03). |
| `statistical-analysis` | ✅ | Call `lib/` and apply the leakage-safe, multiplicity-honest stats patterns. |
| `figure-generation` | ✅ | Render publication figures: dual export, legend doc, color registry, >8-category rule. |
| `research-publications` | ✅ | Research the literature; cite verifiable DOI/PMID sources. |
| `research-protein` | ✅ | Research a protein/gene via UniProt/PDB/AlphaFold/STRING/GO, with ID verification. |
| `research-source-code` | ✅ | Ground a methodological claim in what a tool (DIA-NN, limma, …) actually computes. |
| report skills | ⬜ | Writing conventions, report structure, assembly/coherence (reporting phase, doc 07). |

The finding template lives in `templates/finding.md`; the conventions live in `conventions/`.
