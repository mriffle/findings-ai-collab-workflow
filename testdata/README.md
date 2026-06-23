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

## Why git-ignored, and what ships

For now the loader template is exercised **locally** against this real data (per the project
decision to test against real example data). Tests that depend on it **skip** when it is absent,
so the committed suite still passes on a fresh clone. A small, committed, synthetic canonical
fixture (with a documented planted ground truth) will be added later as the portable smoke-test
input; this real dataset stays local.
