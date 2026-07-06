# testdata/

Real example datasets used for **local** testing of the `lib/` template scripts during engine
development. This is engine test data — it vets the templates *here*; it is **not** a user's
study data and never enters a user's project.

## What's here

- `5xFAD/` — proteomics data from the Johnson 5xFAD / lecanemab mouse AD study, copied verbatim
  from `/home/mriffle/vscode/johnson-5xFAD-lecanemab-mice-AD`. **Git-ignored** (see root
  `.gitignore`): the precursor TSV is ~100 MB and these are real study data, so they are not
  committed. Re-copy from the source project if missing.

  | File | Shape | Notes |
  |------|-------|-------|
  | `data/proteins_wide_unnormalized.tsv`   | 8,830 proteins × 90 sample cols | rows=features, cols=samples; col 1 = `protein` |
  | `data/precursors_wide_unnormalized.tsv` | 96,425 precursors × 90 sample cols | id cols = `protein`, `modifiedSequence`, `precursorCharge` |
  | `metadata/Replicates_5xFAD.csv`         | 90 samples × 10 vars | rows=samples; join key `Replicate` (`.raw`) ↔ data col headers (`.raw` stripped) |
  | `METADATA.source.md`                    | — | the study's own metadata reference, for context |

  Abundances are **linear** intensity; `0` = not detected (preserved, never NaN-substituted).

- `trex/` — proteomics data from the TEI-REX Phase-2a Pelt radiation study, copied from
  `/home/mriffle/vscode/manuscript-trex-phase2a/te-phase2a-pelt`. **Git-ignored.** This is the
  **regression** template's oracle dataset: it carries a genuine *continuous* outcome
  (radiation `Dose_cGy`, 0–100 across graded levels) that 5xFAD lacks. Re-copy from the source
  project if missing (see `_regression_preview/` for the staging step).

  | File | Shape | Notes |
  |------|-------|-------|
  | `data/proteins_unnormalized_wide.tsv` | 10,568 proteins × 936 sample cols | rows=features, cols=samples; col 1 = `protein` (UniProt `sp\|ACC\|GENE_MOUSE`). Already **log2** (0 = not detected). |
  | `metadata/metadata_trex.tsv` | 936 samples × vars | derived from the study's `unblinded-augmented-metadata_wide.tsv`; adds `replicate_key` (source `replicate` with `-`→`.`) so it joins the R-mangled data headers. Target `Dose_cGy`; ComBat batch `Plate`. |

  Abundances are **log2** (not linear like 5xFAD); the experimental dose-labeled subset is the
  492 samples with a numeric `Dose_cGy` (412 T&E "Skin Punch" + 80 UW reference).

## Why git-ignored, and what ships

For now the loader template is exercised **locally** against this real data (per the project
decision to test against real example data). Tests that depend on it **skip** when it is absent,
so the committed suite still passes on a fresh clone. A small, committed, synthetic canonical
fixture (with a documented planted ground truth) will be added later as the portable smoke-test
input; this real dataset stays local.
