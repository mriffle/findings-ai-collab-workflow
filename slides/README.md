# Workflow step slides

Presentation slides for the Findings Workflow — **install & initialize, then
Stages 0–6**. One self-contained HTML page per slide, all sharing a single
stylesheet so the deck reads as one set.

Open **`index.html`** for a clickable contact sheet, or open any slide directly.
Each slide's progress rail links to the others.

| File | Slide |
|------|-------|
| `00-install-init.html` | Install & initialize (plugin · `setup-env` · `init`) |
| `01-stage0-science.html` | Stage 0 — State the science → `state/PROJECT.md` |
| `02-stage1-metadata.html` | Stage 1 — Understand the metadata → `state/METADATA.md` |
| `03-stage2-data.html` | Stage 2 — Understand the data → `state/DATA_DESCRIPTION.md` |
| `04-stage3-loaders-qc.html` | Stage 3 — Loaders, pairing & QC **[integrity gate]** |
| `05-stage4-explore.html` | Stage 4 — Explore ⇄ record findings → `findings/` |
| `06-stage5-validate.html` | Stage 5 — Independent validation → `validated` findings |
| `07-stage6-report.html` | Stage 6 — Reporting → `reports/` |

## Design

- **`assets/slides.css`** is the shared design system: a per-stage accent color
  (set via the `<body>` class, e.g. `stage-3`), subagent chips colored by each
  agent's real plugin color, the progress rail, and print styles. Edit the look
  in one place; every slide follows.
- Each slide card is a **16:9 minimum that grows** if a slide is content-dense,
  so text is never clipped; all type scales with the card width.
- Each stage slide carries an **inline-SVG example visualization** of what that
  stage produces — metadata distributions (1), id-depth bars (2), PCA + CV (3),
  volcano + ROC (4), a concordance scatter (5), a report PDF (6), and a project
  brief (0). They are illustrative, self-contained, and theme-aware (they inherit
  the stage accent via `var(--accent)`); the data is fictional.
- Colors are consistent with the poster schematic figures in `../prompts/`.

## Export / print

- **PDF:** open a slide and Print → Save as PDF (landscape). The print stylesheet
  drops the backdrop and shadow for a clean page.
- **Images (for the poster):** render each page with a headless browser, e.g.
  `chromium --headless --screenshot=05.png --window-size=1600,900 05-stage4-explore.html`.
