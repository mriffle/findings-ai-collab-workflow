# Compact workflow step slides

A briefer, **half-height** variant of the workflow step slides (`../`). Same
width, ~half the height, less text — for a denser layout (e.g. stacking several
stages on one poster panel). One self-contained HTML page per slide.

Open any slide directly, or use the rendered PNGs in `exports/`.

| File | Slide |
|------|-------|
| `00-install-init.html` | Install & initialize (plugin · `setup-env` · `init`) |
| `01-stage0-science.html` | Stage 0 — State the science |
| `02-stage1-metadata.html` | Stage 1 — Understand the metadata |
| `03-stage2-data.html` | Stage 2 — Understand the data |
| `04-stage3-loaders-qc.html` | Stage 3 — Loaders, pairing & QC **[integrity gate]** |
| `05-stage4-explore.html` | Stage 4 — Explore ⇄ record findings |
| `06-stage5-validate.html` | Stage 5 — Independent validation |
| `07-stage6-report.html` | Stage 6 — Reporting |

## How it differs from the full deck (`../`)

Same design system, **same fonts and colors** — it links `../assets/slides.css`
and then `compact.css` for a few spacing/layout overrides only. The changes are:

- **No top navigation rail** and **no lede** — stripped for height.
- **Briefer "What happens"** — 3–4 one-line bullets instead of numbered steps.
- **Scaled-down figure** — the same example visualization, smaller.
- **Full-width bottom strip** (`.metafoot`) carrying the **Run** command and the
  **agents** involved on one line (moved out of the side column so it stays compact).
- **~Half-height card** (`compact.css` sets the deck `min-height` to half).

Each slide still includes the more important information: title, a brief what-happens,
the command that gets run, the agents involved, and a sample output figure.

## Export / print

The PNGs in `exports/` are **6144×1728** (6K, half-height) — exactly half the
height of the full deck's `../exports/` (6144×3456), same width. They are rendered
edge-to-edge (no page backdrop) by injecting a fill-the-viewport override, e.g.:

```bash
INJECT='<style>html,body{background:#fff}body.slide{padding:0;margin:0;display:block;min-height:0}.deck{width:100vw;min-height:100vh;margin:0;border-radius:0;box-shadow:none}</style>'
sed "s|</head>|$INJECT</head>|" 04-stage3-loaders-qc.html > .t.html
google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=6144,1728 --default-background-color=FFFFFFFF \
  --screenshot=exports/04-stage3-loaders-qc.png "file://$PWD/.t.html"
rm -f .t.html
```

`exports/` is git-ignored (PNGs are regenerable); the HTML + `compact.css` are the source.
