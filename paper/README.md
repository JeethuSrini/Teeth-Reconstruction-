# Paper source

Overleaf: https://www.overleaf.com/5873254836rkmbfrhztsxv#03b194

LaTeX source for *Statistical Shape Reconstruction of Worn Enamel–Dentine
Junction Surfaces in Primate Molars*.

| File | Purpose |
|---|---|
| `main.tex` | the paper |
| `references.bib` | bibliography used by `main.tex` |
| `Makefile` | build helper |
| `paper_draft.md` | earlier markdown draft |
| `results_section.tex` | earlier results-section draft |
| `references_earlier.bib` | bibliography that draft was written against |
| `Tooth_Reconstruction_Presentation.pptx` | talk slides |

## Build

```bash
cd paper
latexmk -pdf main.tex        # preferred
# or
make                          # pdflatex → bibtex → pdflatex ×2
```

No LaTeX toolchain was available on the machine where this was drafted, so the
document has **not been compile-tested**. It uses only standard TeX Live
packages: `inputenc`, `fontenc`, `lmodern`, `babel`, `amsmath`, `amssymb`, `bm`,
`booktabs`, `array`, `multirow`, `float`, `graphicx`, `caption`, `geometry`,
`microtype`, `xcolor`, `hyperref`, `natbib`.

## Layout

Two-column, 10pt, A4, 1.9cm margins. The title, author block and abstract span
both columns via the standard `\twocolumn[\begin{@twocolumnfalse}...]` idiom.

Table rules in two-column mode:

| Kind | Environment | Placement |
|---|---|---|
| Narrow (fits one column) | `table` + `[H]` | pinned where written |
| Wide (spans both columns) | `table*` | top of page only |

`[H]` is deliberately **not** used on `table*`. Starred floats in two-column
LaTeX can only be placed at the top of a page — that is a limitation of the
format, not an oversight, and forcing `[H]` on them breaks the build.

Narrow tables bind to `\columnwidth`, spanning tables to `\textwidth`, both via
`tabularx`, so neither can overrun its measure regardless of cell content.

Six tables: two spanning `table*` for the results, one for the literature
comparison, one for the metric layers, one for the parameter reference, and one
in-column table for the specimen inventory.

## Structure

| Section | Covers |
|---|---|
| 1 Introduction | why EDJ, what wear destroys, limits of existing methods |
| 2 Background | SSM machinery in accessible terms — correspondence, shape space, PCA, fitting a partial shape |
| 3 Related work | anthropological baselines, dental SSM, learning-based methods |
| 4 Materials | both datasets, why artificial wear, Molnar staging |
| 5 Methods | correspondence, the three reconstruction strategies, surface output |
| 6 Evaluation protocol | shared wear mask, the three metric layers, metric definitions |
| 7 Results | all three layers, per-layer reading guidance, wear-level behaviour |
| 8 Discussion | physical meaning, literature comparison, why the neighborhood model gains so little, severe-wear failure |
| 9 Limitations | six items |
| 10 Library | design principles, parameter guidance, intended workflow |
| 11 Conclusion | |

## Citation checks before submission

`references.bib` marks each entry `[VERIFIED]` or `[CHECK]`:

- **`cuspml2026`** — title, venue, year and DOI confirmed; the author list could
  not be retrieved (publisher returned 403) and is a placeholder. **Fill this in
  before submitting.**
- Two entries in the older bibliography, now `references_earlier.bib`, had
  incorrect authors and have been corrected here: TeethDreamer (Xu et al., not
  He et al.) and Occudent (Park, Kim, Song & Baek, not Wu et al.).

## Open items in the text

- **Author list and affiliations are placeholders** — `[Author Two]`,
  `[Author Three]`, `[Author Four]` and both `[Department, Institution, ...]`
  lines near the top of `main.tex`. Add or delete entries as needed; the
  superscripts index the affiliation list.
- There is no table of contents, matching the web version.
- **All results come from the corrected pipeline** (Sept 2026). Three errors
  found after the first draft were fixed and everything was rerun:
  1. Stage 1 orientation search (`correspondence_pipeline.py`): it allowed mirror
     images and chose by a saturating ICP score, leaving 51 of 90 ULM training
     teeth and 5 of 8 test specimens registered half a turn out. The ULM series
     was re-registered (`output/correspondence_v5_10k_fixed`); EDJ training
     tooth n0307 failed and is excluded (`output/correspondence_all_100k_fixed`).
  2. Missing-region estimate (`reconstruction_pipeline.py`,
     `neighborhood_reconstruction.py`): now `--proxy-mask-method scan-distance`.
  3. Evaluation wear mask (`evaluate_paper_protocol.py`): now density-robust
     (`--wear-mask surface`); the old mask covered about 80% of the ULM crown.
  4. Mapping outputs back to millimetres (`inverse_correspondence_transform` in
     `reconstruction_pipeline.py`): now uses the exact transform between each
     tooth's `sampled_raw.ply` and `icp_aligned.ply`. The stored normalisation of
     the EDJ Stage 1 run was off by 1 to 17 degrees; the EDJ `*_in_input_space.ply`
     files were regenerated and the EDJ evaluation rerun (only level 7 changed).
  `ssm_pipeline/run_fixed_pipeline.sh` reproduces the Stage 2 outputs and the CSVs
  `eval_paper_old_fixed.csv` / `eval_paper_v5_fixed.csv`; the first-draft outputs
  and CSVs are left untouched for comparison. The headline changed: on the
  held-out ULM series the global shape model is best (0.267 mm) and the non-rigid
  fit is worst (0.361 mm); on the dense EDJ series the non-rigid fit is best.
- The repository-level `results.md` and `README.md` still describe the
  first-draft numbers and have not been updated.
- Figures live in `figures/` and are loaded through `\graphicspath{{figures/}}`.
  **Upload the `figures/` folder to Overleaf**, or the build fails. They are
  copies of outputs from `data_analysis/benchmark_analysis.py`
  (`data_analysis/plots_benchmark/`); rerun that script and copy them again if
  the data change.
  - Main text: Figure 1 `fig_pipeline_edj.pdf` (§5, Methods), Figure 2
    `fig_error_vs_wear.png` (§7, Results) and Figure 3
    `fig_shape_methods_edj.png` (§7, Recovery in shape space).
  - `fig_pipeline_edj.pdf` (used) and `fig_pipeline.pdf` (ULM version, not used)
    are built by `data_analysis/pipeline_figure/render_assets.py --series EDJ|ULM`
    then `compose.py --series EDJ|ULM` (needs `pyvista`, `rtree`, `open3d`). Every
    tooth image is rendered from the pipeline's own files: EDJ specimen TEST1 at
    wear level 5, or ULM specimen N459 at wear level 4.
  - Main text: `fig_realworn.pdf` (§7, Naturally worn teeth), built by
    `data_analysis/realworn_figure/render_realworn.py` then `compose_realworn.py`
    from `ssm_pipeline/run_realworn_edj.sh` outputs. Only the five naturally worn
    teeth that passed the registration check are shown (01, 02, 04, 08, 09); 03,
    06 and 07 are registered in the wrong orientation and 05 fits poorly, and all
    four need re-registration on the GPU cluster at the EDJ settings.
  - Supplementary material (one-column pages after the references): Figures S1
    and S2 `fig_shape_methods_ulm_part1/2.png`, S3 and S4
    `fig_shape_overview_edj/ulm.png`.

