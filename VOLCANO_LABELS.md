# Volcano label placement — ✅ RESOLVED (volcano v0.2)

> **Status: shipped.** `annotate_top=` now places labels **collision-free via `textalloc`**
> (`lib/figures/volcano.py` `_annotate_top`, v0.2): labels repelled off each other and the
> full point cloud, one leader line per point, clamped inside the axes. Chosen after a
> parallel bake-off of all four candidate approaches below (adjustText, **textalloc**, a
> `dynamic_range` port, and a from-scratch repel) rendered against the A–F stress cases on
> real 5xFAD data. **textalloc** won: it is a trusted library that *also* ships `py.typed`
> (passes `mypy --strict` with no override, unlike adjustText) and is fast (<1s/fig). New
> shipping dep `textalloc==1.2.3` (added to `commands/setup-env.md` + `requirements-dev.txt`);
> a planted-dense-cluster **no-overlap bbox invariant** added to `lib/tests/test_volcano.py`;
> `lib/manifest.md` bumped to v0.2. The plan below is kept as the bake-off record / rationale.

**What this is.** A scoped plan (now executed) to improve point-label placement in
the volcano template (`lib/figures/volcano.py`). The optional `annotate_top=` labels
were drawn with a fixed pixel offset and **no collision avoidance**, so when several top hits
sat close together the labels overprinted each other and became unreadable. This doc records
the problem, the in-repo prior art mined, the candidate approaches, and **exactly how the
failing volcano plots + the conditions were regenerated** to iterate against real data.
It is engine-dev planning, not user-facing.

Companion to [`FEATURE_FINDING.md`](FEATURE_FINDING.md) (which tracks the differential-
abundance family the volcano belongs to) and [`QC_GAPS.md`](QC_GAPS.md). Follow
[`lib/AUTHORING.md`](lib/AUTHORING.md) when shipping the change.

---

## The problem

`plot_volcano(..., annotate_top=N)` labels the `N` most-significant hits. The current
`_annotate_top` (in `lib/figures/volcano.py`) is deliberately minimal:

```python
for i in order[:annotate_top]:
    ax.annotate(str(names[i]), (eff[i], neg_log_q[i]),
                textcoords="offset points", xytext=(4, 2), fontsize=7, color="black")
```

A fixed `(+4, +2)` point offset, no leader lines, no de-collision, no axis clamp. It is
documented as a v0.1 simplification (default `annotate_top=0`).

**The concrete failure** (seen in the examples, `testdata/5xFAD/_diffabund_examples/F1_*`):
the disease contrast's two strongest hits — **APP** (`sp|P05067|5xFADA4_HUMAN`) and its mouse
ortholog (`sp|P12023|A4_MOUSE`) — sit at almost the same `(log2FC, −log10 q)` against the
q-underflow ceiling, so their labels render as the unreadable mash **`A45xFADA4`**. Several
of the next-ranked hits (TICN1/TICN2/CLUS) also crowd. This is the canonical hard case: a
tight cluster of co-significant features at the top of the plot.

## In-repo prior art (mine this first — no new dependency)

The **dynamic-range** template already solved a very similar leader-line placement problem:
`lib/figures/dynamic_range._place_labels` (+ `_annotate`) does greedy **vertical
de-collision** with **leader lines** — head points get a label column on one side (one leader
per point, unambiguous ownership), tail points are labelled below in open space, both stacks
are spread by a fixed gap, and the y-limits are padded so **no label leaves the axes** (and
none collides with the title). That is the reference implementation to adapt; it ships, is
strict-clean, and adds no dependency. The volcano differs (labels can go on either side of a
central x=0, and the y-axis is significance not rank), so the geometry needs rethinking, but
the de-collide-and-lead approach transfers.

## Candidate approaches to explore (the menu to iterate on)

1. **Port the dynamic-range greedy de-collision + leader lines.** Adapt `_place_labels` to the
   volcano: sort hits by `−log10 q`, place labels in de-collided stacks (left stack for
   `effect < 0`, right stack for `effect > 0`), one leader line each, clamp inside the axes.
   *Pros:* no new dep, consistent with the engine, strict-clean. *Cons:* hand-tuned geometry.
2. **`adjustText`** (force-directed repel, the popular choice). *Pros:* good results with little
   code. *Cons:* a **new shipping dependency**, **untyped** (needs an `ignore_missing_imports`
   override against the stub-free venv — see `conventions/coding.md`), and a moving target.
   Only with explicit sign-off (it would go in `commands/setup-env.md`'s baseline).
3. **Minimal force-directed repel, reimplemented.** A small label-vs-label + label-vs-point
   repulsion loop with leader lines. *Pros:* full control, strict-clean, no dep. *Cons:* real
   geometry to get right and test.
4. **Smarter selection instead of placement.** Cap labels by a minimum `(x, y)` separation
   (skip a label whose anchor is within ε of an already-placed one) and/or merge co-located
   orthologs (`APP`/`A4`) into one label. Cheaper; pairs well with 1 or 3.

Likely answer: **(1) or (3)** — keep it dependency-free and consistent with `dynamic-range`,
possibly with (4) as a pre-filter. Decide with the user after seeing renders.

## How to generate volcano plots to iterate on  ← the core of this plan

**Data.** The git-ignored real 5xFAD proteins under `testdata/5xFAD/` (skip cleanly if
absent), the same data every preview/example uses. Load → median-normalize → `log2` →
experimental subset (drop `Genotype == "na"`, the pools) → a binary `Disease` column. The
exact recipe is already written in `testdata/5xFAD/_diffabund_examples/make_examples.py` and
`_diffabund_preview/make_diffabund_preview.py` — **copy the setup block from there.**

**Analysis.** One call gives the table the volcano reads:
`differential_abundance(exp, "Disease", covariates=["Gender","Treatment","Cohort"],
reference={"Disease":"nonAD", ...}, method="moderated")`. Render via
`volcano_from_result(res, annotate_top=N, labels=…)` or the array core `plot_volcano(...)`.

**Conditions to compare** (render each variant side by side into a new git-ignored
`testdata/5xFAD/_volcano_labels_preview/` dir, then review with the user — the established
render → review → iterate loop):

| # | Stress case | How to produce it | What it tests |
|---|---|---|---|
| A | **Dense top cluster** (the headline failure) | disease contrast, `annotate_top=8` | the APP/A4 co-located pair + TICN/CLUS crowd at the q-ceiling |
| B | **Crowded** | disease contrast, `annotate_top=20`–`30` | does the method scale / when to stop labelling |
| C | **Bidirectional** | the `Genotype[5xFAD vs WT]` term (`reference={"Genotype":"WT"}`), `annotate_top=12` | labels on **both** sides of x=0 (strong down hit `ZFAN5` + up hits) |
| D | **Sparse / easy** | a near-null term (`Genotype[C57BL/6j vs WT]`), `annotate_top=6` | the method must not over-engineer the easy case |
| E | **Axis edges** | disease contrast (has a `log2FC≈12` far-right outlier + points at the top y-limit) | leader lines + **clamp labels inside the axes**, no title collision |
| F | **Label length** | render A & C with both the raw accession `sp|P05067|5xFADA4_HUMAN` **and** the collapsed gene symbol `APP` | width drives collisions; the example's `short()` helper collapses to the gene |

**Knobs to vary while iterating:** `annotate_top` (8 / 20 / 30), label source (accession vs
gene symbol), `figsize` (collisions are resolution-dependent), and the placement algorithm
itself. Keep `fdr`/`effect_threshold` fixed so only the labelling changes between renders.

**Starter skeleton** (a future session can drop this into the preview dir and extend):

```python
# testdata/5xFAD/_volcano_labels_preview/make_preview.py  (git-ignored, throwaway)
# 1. copy the load+experimental-subset+Disease setup from _diffabund_examples/make_examples.py
# 2. res = da.differential_abundance(exp, "Disease",
#        covariates=["Gender","Treatment","Cohort"],
#        reference={"Disease":"nonAD","Gender":"F","Treatment":"ISO","Cohort":"1"},
#        method="moderated")
# 3. for each condition A–F: vol.volcano_from_result(res, term=…, annotate_top=N,
#        registry_path=<tmp palette>, title=…)  -> save_figure(...)
#    (for the prototype placement algorithm, render with the WIP _annotate_top variant and
#     compare against the current one in the same figure dir)
```

## How to judge (acceptance)

This is a **visual-quality** task, so the loop is render → review-with-the-user → iterate
(same as every figure template's design history). Targets: no overlapping label text; the
co-located APP/A4 pair both legible; leader lines clearly own their points; every label inside
the axes and clear of the title; the easy/sparse case (D) stays clean and uncluttered. A
unit-testable invariant to add once an algorithm is chosen: **no two label bounding boxes
overlap** on a planted cluster (render to a canvas, compare `Text.get_window_extent()` boxes).

## Constraints (the `lib/` bar — keep these)

- **Optional + back-compatible.** `annotate_top=0` (no labels) stays the default and unchanged.
- **No new *required* shipping dependency without sign-off.** Prefer the dependency-free
  port of `dynamic-range`'s placement; `adjustText` only with explicit user approval (untyped,
  goes in `setup-env`'s baseline).
- **Strict bar.** `ruff` strict + `mypy --strict` + tests; never leak a figure on an error path
  (the `try/except BaseException: plt.close(fig); raise` guard); route colors through the
  registry as today. **Bump `volcano` to v0.2** and update `lib/manifest.md` when it ships.
- Mine `lib/figures/dynamic_range.py` (`_place_labels`/`_annotate`) before writing new geometry.

## Pointers

- Failing artifact: `testdata/5xFAD/_diffabund_examples/F1_volcano_disease_moderated.png`
  (and its `make_examples.py`).
- Template + starting point: `lib/figures/volcano.py` (`_annotate_top`).
- Prior art: `lib/figures/dynamic_range.py` (`_place_labels`, `_annotate`).
- Data recipe: `testdata/5xFAD/_diffabund_examples/make_examples.py` setup block.
