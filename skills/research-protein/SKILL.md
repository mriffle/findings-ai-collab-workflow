---
name: research-protein
description: >-
  How to research a protein or gene for a Findings Workflow project using the
  authoritative databases (UniProt, PDB/AlphaFold, STRING, GO), with identifier
  and fact verification. Use when a finding involves a specific protein/gene and
  needs grounded background. Output feeds a research-finding.
---

# Researching a protein / gene

Goal: grounded, normalized, verifiable background on a specific protein or gene. Every claim is sourced; every identifier is canonical and confirmed.

## Authoritative sources

- **UniProt** — the canonical protein record: accession, recommended name, gene name, function, subcellular location, PTMs, sequence features. Use the **UniProt accession** as the canonical id (`db: uniprot`).
- **HGNC** — the canonical human gene symbol (`db: hgnc`). Resolve symbol↔accession explicitly; do not assume.
- **PDB / AlphaFold** — experimental and predicted structures (cite the PDB id / AlphaFold entry).
- **STRING** — functional interaction partners and evidence channels (treat as evidence-weighted, not ground truth).
- **GO** — function/process/component annotations, with evidence codes (`db: go`); note the evidence code, since IEA (electronic) ≠ experimental.
- **Reactome** — pathway membership (`db: reactome`).

## Procedure

1. **Pin the identity first.** Confirm the exact UniProt accession and gene symbol for the right organism/isoform. Identifier integrity matters — the wrong accession poisons everything downstream.
2. **Gather function/role** from UniProt + GO + Reactome, recording the source for each claim and the GO evidence code where relevant.
3. **Interactions/structure** from STRING / PDB / AlphaFold as the question needs, noting confidence/evidence.
4. **Verify** each claim against its source before recording it. Distinguish experimentally supported from predicted/electronic annotation.

## Output

A research-finding (`templates/research-finding.md`) with the entity normalized to canonical IDs (verified), each claim sourced (database record or paper), and a non-empty `references` list. It goes to the research-reviewer, which re-checks the IDs and every reference. Never attach an unverifiable accession or citation — record the claim as unverified instead.
