# Scoring review

Helen, this branch is yours. It starts from current `main` (`e36f2ca`). Please do the review here. Do not commit onto `main`.

The job is to check how the reconstruction scores are calculated, and to say whether the numbers in the README are produced by a sound pipeline. Look for logical errors and implementation errors. You can rerun code and try experiments on this branch.

If you find a problem, add two files on this branch:

- `findings.md` — what is wrong, which file and function, and which published number it affects
- `recommendations.md` — what we should change next

If a check turns out to be fine, say that in `findings.md` too, so we know it was looked at.

## Where the published scores are

Start with these. They are the numbers we are currently standing behind.

| What you are reading | Path |
|---|---|
| Short tables | `README.md` |
| Full write-up of the same numbers | `results.md` |
| Paper table (25-tooth Global vs Neighborhood only) | `results_section.tex` |
| 19-tooth comparison vs the raw worn scan | `ssm_pipeline/output/eval_flagship_19teeth.csv` |
| 16-case comparison vs the true original (TEST1 / TEST2) | `ssm_pipeline/output/eval_old_dataset.csv` |
| 64-case comparison vs the true original (8 specimens × 8 wear levels) | `ssm_pipeline/output/eval_v5_dataset.csv` |

Per-case files from the reconstruction runs are the `evaluation.json` files under `ssm_pipeline/output/`. The runs that feed the tables are:

- Global SSM: `recon_all_v3` (old), `recon_all_v5` (v5)
- Neighborhood SSM: `recon_neighborhood_v4` (old), `recon_neighborhood_v5` (v5)
- Patch / Hole Filling: `recon_holes_final` (old), `recon_holes_v5` (v5)
- GPMM: `recon_gpmm_test` (old), `recon_gpmm_v5` (v5)

## Which code computes them

`ssm_pipeline/evaluate_all_methods.py` builds the old and v5 CSVs.

- `chamfer_rmse_hausdorff` and `directed_error` are the metric functions.
- `eval_old_dataset` scores TEST1 and TEST2 against the unworn originals `n0245` and `n0257`.
- `_ground_truth_wear_mask` decides which points count as worn on the old dataset.
- `eval_v5_dataset` scores the eight v5 specimens. Index-paired methods use a shared mask. Patch / Hole Filling uses its own `patch_mask.npy`.

The 19-tooth and 25-tooth tables in the README are not produced by that script. `results.md` says the 19-tooth CSV came from a one-off script, and that script is not in the repo. Please find it or say that those two tables cannot be recomputed from this commit.

The old-dataset ground-truth meshes are loaded from a hardcoded path:

`/Users/jeethu_srini/Downloads/Original models`

A rerun will fail on any other machine until that path is fixed.

## What to check

Please answer each of these. A short yes or no, with the file you looked at, is enough when the code is fine.

1. **Do the README tables match the CSVs?**
   Average `eval_flagship_19teeth.csv`, `eval_old_dataset.csv`, and `eval_v5_dataset.csv` yourself and compare with `README.md` and `results.md`. On the v5 table, the “full” column is RMSE for Global SSM, Neighborhood SSM, and GPMM, and Chamfer for Patch / Hole Filling. Those are not the same metric. Confirm the README is using the column it claims to use.

2. **Is the worn-region mask measuring wear?**
   On v5, `eval_v5_dataset` marks a point worn when the original and the worn input differ by more than 1.5 times the median point spacing. The published CSV marks about 8,508 of 10,000 points worn (about 85%). `results.md` already says this mask is broader than intended.
   On the old dataset, `eval_old_dataset.csv` reports about 98,590 worn points per case. That is roughly 82% of the 120,000-point ground-truth sample. `results.md` says the old dataset does not have this problem. Check whether that claim is true. If the mask includes intact surface, the worn-region RMSE (about 0.069 mm for Global SSM) is diluted and should not be quoted as restoration error.

3. **Is the same mask applied to every method?**
   The old-dataset code builds one mask, then uses `patch_mask.npy` instead whenever that file exists. Patch / Hole Filling is therefore scored on a different set of points than the other methods. On v5 the same split exists: index-paired methods share one mask, and Patch / Hole Filling uses its own. Say whether the README comparison is still fair.

4. **Do the metric directions match their names?**
   In `chamfer_rmse_hausdorff`, the value saved as old-dataset `full_rmse` is one direction only (ground truth toward the reconstruction), not a symmetric RMSE. On v5, index-paired `full_chamfer` is also one direction, while Patch / Hole Filling’s Chamfer is symmetric. Worn-region error is “reconstruction points to nearest original” for the old dataset, and exact pointwise distance for v5 index-paired methods. Please say whether any published label describes a different calculation than the code.

5. **Are the 25-tooth and 19-tooth tables the same experiment?**
   The first README table is 25 teeth scored against the raw worn scan (Neighborhood Chamfer 0.0866 mm). The next table is 19 of those teeth (Chamfer about 0.070 mm). Six real teeth were left out because the local scans had corrupted vertices. `results_section.tex` uses the 25-tooth numbers. There is no CSV in the repo for that 25-tooth table. Please either locate the source run or mark the table as not reproducible from this commit.
   Coverage is written as a fraction in the 25-tooth table (0.411) and as a percent in the 19-tooth table (45.51). The LaTeX file uses percent. These should not be mixed.
   On Hausdorff, the 25-tooth mean favors Global SSM (0.665 mm vs 0.674 mm), but the win count favors Neighborhood SSM (16 of 25 teeth). The README verdict says “Global better.” Please say which of those the paper should use.

6. **Do Blend and GPMM-kernel belong in the CSVs?**
   Both are still computed in `evaluate_all_methods.py` and still have rows in the old and v5 CSVs. They were removed from the README. On the old CSV, GPMM and GPMM-kernel match to five decimals. No need to put them back in the table. The code, the CSV, and the write-up should not look like three different experiments.

## What to leave alone

Do not treat files that are only on someone’s laptop as the scores to review. In particular, ignore any local `eval_paper_*.csv` files and any rewritten `results.md` that is not on this branch. Those are a later protocol and are not what the README currently reports.

Experiments and edits stay on `helen/score-review`.
