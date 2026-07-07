# Reconstruction Results, Full Detail

This file holds every numeric result behind the summary shown in [README.md](README.md). It covers all four reconstruction methods (Global SSM, Neighborhood SSM, Patch / Hole Filling, GPMM), scored on three independent comparisons: against the raw worn tooth scan (the flagship, 19 of the original 25 old dataset teeth), and against the true original tooth on two datasets (the 16 case old dataset and the 64 case v5 dataset). See the README for method descriptions and images; this file is numbers only.

Raw, per case data lives in [`ssm_pipeline/output/eval_flagship_19teeth.csv`](ssm_pipeline/output/eval_flagship_19teeth.csv), [`ssm_pipeline/output/eval_old_dataset.csv`](ssm_pipeline/output/eval_old_dataset.csv), and [`ssm_pipeline/output/eval_v5_dataset.csv`](ssm_pipeline/output/eval_v5_dataset.csv), produced by [`ssm_pipeline/evaluate_all_methods.py`](ssm_pipeline/evaluate_all_methods.py) (the two datasets vs. the true original) and a one off script (the flagship comparison vs. the raw worn scan).

## How These Numbers Were Computed

The flagship comparison, further down this file, follows the project's original methodology: reconstruction scored against the raw worn tooth scan itself, since 9 of the 25 old dataset teeth are real worn specimens with no known original. The other two comparisons instead score against the true, original, unworn tooth, which is only available for the synthetic wear series (old dataset TEST1/TEST2) and the v5 dataset's real specimens (whose corresponded, unworn original was kept aside during data preparation). Those two are scored two ways.

**Full surface metric.** The reconstruction is compared to the original over every point in the cloud. This rewards a method for keeping the real surface wherever it is intact, so on its own it can make a method that barely changes anything look artificially good.

**Worn region metric.** The same comparison, but restricted only to the points that were genuinely worn away. This is the fairer test of restoration quality, since a method cannot score well here just by leaving most of the tooth untouched.

For the old dataset, the worn region mask is ground truth anchored and shared by Global SSM, Neighborhood SSM, and GPMM. It is built from where the real worn scan sits recessed relative to the true original mesh, so no single method's own opinion of what counts as worn biases the comparison. Patch/Holes appends new points rather than replacing points in place, so it is scored separately using its own graft mask, meaning exactly the points it decided to fill in.

For the v5 dataset, ground truth is the corresponded original tooth in the same point ordering as the worn input, so no ICP alignment or nearest neighbor search is needed for the paired methods (Global SSM, Neighborhood SSM, GPMM). Their worn region mask comes from where the true original deviates from the worn input by more than one and a half times the median point spacing. Patch/Holes again appends points and is scored with its own graft mask against the nearest point in the true original.

**Known limitation.** On the v5 dataset the shared mask (used by three of the four methods) flags roughly eighty five percent of all points as worn, on average, which is broader than intended and likely reflects real specimen correspondence noise rather than only genuine wear. Patch/Holes' own graft mask is much smaller and highly variable case to case (ranging from about five percent to one hundred percent of points). This means the Patch/Holes numbers on v5 are not perfectly apples to apples with the other three methods there. The old dataset does not have this issue, since its ground truth anchored mask is more reliable.

---

## Flagship Comparison Against the Raw Worn Tooth (19 of 25 Old Dataset Teeth)

This is the full detail behind the "Adding GPMM and Patch / Hole Filling to the Comparison" table in the README. Every reconstruction is scored against its own raw worn tooth scan rather than the true original, which is the only ground truth available for the 9 real worn teeth (they have no known unworn original).

Real teeth 01, 02, 03, 04, 05, and 09 are excluded from this table. Each had a local copy of its raw scan with genuine read corruption (thousands of vertex read errors for six of them, and a smaller but still meaningful 1597 corrupted vertices for tooth 03), detected by checking for vertices far from the cloud's own median center after loading. Rather than publish numbers computed against a partially corrupted file, those six teeth were left out. The 19 teeth used are TEST1 levels 0 through 7, TEST2 levels 0 through 7, and real teeth 06, 07, and 08, all verified to have zero corrupted vertices.

As a sanity check, the Global and Neighborhood SSM numbers recomputed on this 19 tooth set are close to the historical 25 tooth numbers in the README's first table (R squared 99.89 percent here versus 99.74 percent there, Chamfer 0.070 mm here versus 0.088 mm there), which is the amount of difference expected from a smaller subset of teeth scored against a separately sourced copy of the same raw scans, not a sign of a methodology problem.

Source data: [`ssm_pipeline/output/eval_flagship_19teeth.csv`](ssm_pipeline/output/eval_flagship_19teeth.csv).

### Mean Across the 19 Teeth

| Method | R squared (%) | Chamfer (mm) | Hausdorff (mm) | RMSE worn to recon (mm) | RMSE recon to worn (mm) | MAE worn to recon (mm) | MAE recon to worn (mm) | Coverage 1x (%) | Coverage 2x (%) | Coverage 5x (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global SSM | 99.89 | 0.0698 | 0.584 | 0.0990 | 0.0845 | 0.0731 | 0.0665 | 15.65 | 45.51 | 82.50 |
| Neighborhood SSM | 99.89 | 0.0705 | 0.588 | 0.1013 | 0.0853 | 0.0741 | 0.0670 | 15.57 | 45.32 | 82.22 |
| GPMM | 99.56 | 0.1483 | 0.731 | 0.1910 | 0.1748 | 0.1537 | 0.1430 | 6.90 | 21.40 | 52.54 |
| Patch / Hole Filling | 99.63 | 0.1376 | 0.774 | 0.1720 | 0.1671 | 0.1381 | 0.1371 | 8.54 | 26.28 | 58.63 |

### Best Method Per Metric, Count Out of 19 Teeth

| Metric | Global SSM | Neighborhood SSM | GPMM | Patch / Hole Filling |
|---|---:|---:|---:|---:|
| R squared | 9 | 6 | 0 | 4 |
| Chamfer | 11 | 5 | 0 | 3 |
| Hausdorff | 10 | 1 | 3 | 5 |
| RMSE worn to recon | 9 | 6 | 0 | 4 |
| RMSE recon to worn | 11 | 5 | 0 | 3 |
| MAE worn to recon | 10 | 6 | 0 | 3 |
| MAE recon to worn | 9 | 7 | 0 | 3 |
| Coverage 1x | 9 | 8 | 0 | 2 |
| Coverage 2x | 7 | 10 | 0 | 2 |
| Coverage 5x | 9 | 6 | 0 | 4 |

### Full Detail, All 19 Teeth x All 4 Methods (76 rows)

| Tooth | Method | R squared (%) | Chamfer (mm) | Hausdorff (mm) | RMSE worn to recon (mm) | RMSE recon to worn (mm) | MAE worn to recon (mm) | MAE recon to worn (mm) | Coverage 1x | Coverage 2x | Coverage 5x |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tooth_06_wear_real | Global SSM | 99.67 | 0.1483 | 0.980 | 0.2034 | 0.1869 | 0.1525 | 0.1442 | 4.92 | 18.09 | 49.65 |
| tooth_06_wear_real | Neighborhood SSM | 99.69 | 0.1457 | 0.980 | 0.1988 | 0.1858 | 0.1485 | 0.1430 | 5.05 | 18.70 | 50.97 |
| tooth_06_wear_real | GPMM | 98.48 | 0.3320 | 1.751 | 0.4369 | 0.4107 | 0.3445 | 0.3195 | 2.06 | 6.86 | 18.31 |
| tooth_06_wear_real | Patch / Hole Filling | 98.65 | 0.3436 | 1.718 | 0.4110 | 0.4582 | 0.3250 | 0.3622 | 1.87 | 6.53 | 18.90 |
| tooth_07_wear_real | Global SSM | 99.51 | 0.1893 | 0.968 | 0.2506 | 0.2408 | 0.1923 | 0.1864 | 3.41 | 12.95 | 36.94 |
| tooth_07_wear_real | Neighborhood SSM | 99.49 | 0.1909 | 1.052 | 0.2569 | 0.2417 | 0.1956 | 0.1863 | 3.52 | 13.00 | 37.62 |
| tooth_07_wear_real | GPMM | 97.73 | 0.4161 | 1.913 | 0.5405 | 0.4791 | 0.4359 | 0.3962 | 0.96 | 3.53 | 11.14 |
| tooth_07_wear_real | Patch / Hole Filling | 98.33 | 0.3963 | 1.980 | 0.4639 | 0.5096 | 0.3821 | 0.4105 | 1.16 | 4.32 | 12.89 |
| tooth_08_wear_real | Global SSM | 99.96 | 0.0482 | 0.620 | 0.0644 | 0.0617 | 0.0490 | 0.0474 | 17.83 | 53.99 | 91.47 |
| tooth_08_wear_real | Neighborhood SSM | 99.96 | 0.0471 | 0.620 | 0.0635 | 0.0603 | 0.0480 | 0.0462 | 18.48 | 55.37 | 92.05 |
| tooth_08_wear_real | GPMM | 99.61 | 0.1753 | 0.749 | 0.2098 | 0.2118 | 0.1745 | 0.1760 | 3.16 | 11.85 | 33.13 |
| tooth_08_wear_real | Patch / Hole Filling | 99.63 | 0.1743 | 0.841 | 0.2044 | 0.2158 | 0.1701 | 0.1786 | 2.96 | 11.23 | 33.93 |
| tooth_TEST1_wear_level0 | Global SSM | 99.97 | 0.0496 | 0.297 | 0.0611 | 0.0588 | 0.0506 | 0.0486 | 15.82 | 48.14 | 89.25 |
| tooth_TEST1_wear_level0 | Neighborhood SSM | 99.97 | 0.0491 | 0.297 | 0.0607 | 0.0583 | 0.0502 | 0.0481 | 15.78 | 48.78 | 89.38 |
| tooth_TEST1_wear_level0 | GPMM | 99.79 | 0.1227 | 0.497 | 0.1544 | 0.1476 | 0.1251 | 0.1202 | 4.94 | 17.69 | 47.80 |
| tooth_TEST1_wear_level0 | Patch / Hole Filling | 99.83 | 0.1127 | 0.507 | 0.1366 | 0.1399 | 0.1116 | 0.1138 | 4.86 | 18.83 | 52.75 |
| tooth_TEST1_wear_level1 | Global SSM | 99.96 | 0.0502 | 0.346 | 0.0631 | 0.0597 | 0.0515 | 0.0489 | 15.56 | 48.42 | 88.38 |
| tooth_TEST1_wear_level1 | Neighborhood SSM | 99.96 | 0.0517 | 0.346 | 0.0647 | 0.0614 | 0.0530 | 0.0505 | 15.22 | 46.80 | 87.34 |
| tooth_TEST1_wear_level1 | GPMM | 99.83 | 0.1104 | 0.447 | 0.1379 | 0.1312 | 0.1127 | 0.1081 | 5.16 | 19.16 | 52.07 |
| tooth_TEST1_wear_level1 | Patch / Hole Filling | 99.88 | 0.0941 | 0.430 | 0.1153 | 0.1148 | 0.0943 | 0.0938 | 5.64 | 23.40 | 61.25 |
| tooth_TEST1_wear_level2 | Global SSM | 99.96 | 0.0525 | 0.380 | 0.0679 | 0.0617 | 0.0545 | 0.0505 | 15.20 | 47.06 | 86.12 |
| tooth_TEST1_wear_level2 | Neighborhood SSM | 99.96 | 0.0534 | 0.380 | 0.0685 | 0.0626 | 0.0553 | 0.0515 | 14.39 | 45.69 | 85.89 |
| tooth_TEST1_wear_level2 | GPMM | 99.83 | 0.1127 | 0.462 | 0.1396 | 0.1314 | 0.1156 | 0.1098 | 4.25 | 17.26 | 49.93 |
| tooth_TEST1_wear_level2 | Patch / Hole Filling | 99.88 | 0.0963 | 0.473 | 0.1171 | 0.1151 | 0.0966 | 0.0960 | 7.64 | 25.28 | 57.93 |
| tooth_TEST1_wear_level3 | Global SSM | 99.95 | 0.0543 | 0.465 | 0.0732 | 0.0631 | 0.0571 | 0.0516 | 14.61 | 45.72 | 85.15 |
| tooth_TEST1_wear_level3 | Neighborhood SSM | 99.95 | 0.0560 | 0.465 | 0.0745 | 0.0651 | 0.0586 | 0.0534 | 13.50 | 43.47 | 84.15 |
| tooth_TEST1_wear_level3 | GPMM | 99.81 | 0.1161 | 0.475 | 0.1458 | 0.1366 | 0.1193 | 0.1129 | 5.30 | 18.89 | 48.24 |
| tooth_TEST1_wear_level3 | Patch / Hole Filling | 99.85 | 0.1037 | 0.609 | 0.1284 | 0.1240 | 0.1045 | 0.1029 | 6.44 | 23.41 | 55.15 |
| tooth_TEST1_wear_level4 | Global SSM | 99.94 | 0.0571 | 0.596 | 0.0848 | 0.0670 | 0.0614 | 0.0528 | 15.36 | 46.13 | 83.47 |
| tooth_TEST1_wear_level4 | Neighborhood SSM | 99.93 | 0.0590 | 0.596 | 0.0863 | 0.0686 | 0.0633 | 0.0547 | 14.61 | 43.80 | 82.16 |
| tooth_TEST1_wear_level4 | GPMM | 99.83 | 0.1067 | 0.509 | 0.1370 | 0.1246 | 0.1107 | 0.1027 | 5.63 | 20.96 | 53.51 |
| tooth_TEST1_wear_level4 | Patch / Hole Filling | 99.87 | 0.0929 | 0.713 | 0.1206 | 0.1101 | 0.0949 | 0.0909 | 6.80 | 22.96 | 63.21 |
| tooth_TEST1_wear_level5 | Global SSM | 99.93 | 0.0596 | 0.720 | 0.0901 | 0.0696 | 0.0642 | 0.0551 | 14.11 | 44.14 | 82.42 |
| tooth_TEST1_wear_level5 | Neighborhood SSM | 99.93 | 0.0612 | 0.720 | 0.0912 | 0.0707 | 0.0658 | 0.0566 | 13.76 | 42.04 | 81.20 |
| tooth_TEST1_wear_level5 | GPMM | 99.82 | 0.1155 | 0.637 | 0.1413 | 0.1353 | 0.1181 | 0.1128 | 4.24 | 15.65 | 48.30 |
| tooth_TEST1_wear_level5 | Patch / Hole Filling | 99.85 | 0.1037 | 0.867 | 0.1319 | 0.1175 | 0.1069 | 0.1005 | 4.80 | 16.60 | 51.96 |
| tooth_TEST1_wear_level6 | Global SSM | 99.66 | 0.0844 | 1.313 | 0.1950 | 0.0905 | 0.1052 | 0.0637 | 14.90 | 42.66 | 75.12 |
| tooth_TEST1_wear_level6 | Neighborhood SSM | 99.66 | 0.0857 | 1.313 | 0.1953 | 0.0914 | 0.1064 | 0.0650 | 13.88 | 40.93 | 74.56 |
| tooth_TEST1_wear_level6 | GPMM | 99.54 | 0.1604 | 0.886 | 0.2261 | 0.1854 | 0.1731 | 0.1476 | 4.36 | 14.91 | 38.06 |
| tooth_TEST1_wear_level6 | Patch / Hole Filling | 99.42 | 0.1635 | 1.407 | 0.2546 | 0.1931 | 0.1753 | 0.1517 | 5.75 | 18.15 | 40.77 |
| tooth_TEST1_wear_level7 | Global SSM | 99.90 | 0.0628 | 0.840 | 0.1065 | 0.0742 | 0.0696 | 0.0560 | 15.52 | 46.83 | 80.79 |
| tooth_TEST1_wear_level7 | Neighborhood SSM | 99.90 | 0.0659 | 0.840 | 0.1085 | 0.0782 | 0.0724 | 0.0595 | 14.44 | 44.28 | 79.24 |
| tooth_TEST1_wear_level7 | GPMM | 98.97 | 0.2594 | 1.097 | 0.3403 | 0.3025 | 0.2710 | 0.2478 | 2.52 | 8.96 | 24.64 |
| tooth_TEST1_wear_level7 | Patch / Hole Filling | 99.03 | 0.2435 | 1.194 | 0.3293 | 0.2741 | 0.2652 | 0.2218 | 2.74 | 8.20 | 23.41 |
| tooth_TEST2_wear_level0 | Global SSM | 99.96 | 0.0541 | 0.374 | 0.0653 | 0.0658 | 0.0539 | 0.0542 | 19.14 | 52.90 | 92.86 |
| tooth_TEST2_wear_level0 | Neighborhood SSM | 99.96 | 0.0537 | 0.317 | 0.0655 | 0.0650 | 0.0539 | 0.0536 | 19.14 | 53.79 | 92.51 |
| tooth_TEST2_wear_level0 | GPMM | 99.94 | 0.0664 | 0.333 | 0.0810 | 0.0784 | 0.0670 | 0.0659 | 14.85 | 41.69 | 86.23 |
| tooth_TEST2_wear_level0 | Patch / Hole Filling | 99.97 | 0.0503 | 0.184 | 0.0571 | 0.0574 | 0.0499 | 0.0507 | 19.10 | 51.95 | 98.57 |
| tooth_TEST2_wear_level1 | Global SSM | 99.96 | 0.0542 | 0.374 | 0.0655 | 0.0656 | 0.0542 | 0.0543 | 19.15 | 52.87 | 92.61 |
| tooth_TEST2_wear_level1 | Neighborhood SSM | 99.96 | 0.0540 | 0.309 | 0.0661 | 0.0652 | 0.0542 | 0.0537 | 19.32 | 53.84 | 92.31 |
| tooth_TEST2_wear_level1 | GPMM | 99.95 | 0.0612 | 0.310 | 0.0742 | 0.0722 | 0.0617 | 0.0607 | 16.20 | 44.91 | 90.40 |
| tooth_TEST2_wear_level1 | Patch / Hole Filling | 99.98 | 0.0418 | 0.205 | 0.0472 | 0.0478 | 0.0413 | 0.0424 | 24.97 | 63.96 | 99.64 |
| tooth_TEST2_wear_level2 | Global SSM | 99.96 | 0.0553 | 0.377 | 0.0675 | 0.0668 | 0.0555 | 0.0551 | 19.00 | 51.79 | 91.92 |
| tooth_TEST2_wear_level2 | Neighborhood SSM | 99.96 | 0.0554 | 0.321 | 0.0687 | 0.0670 | 0.0559 | 0.0549 | 19.30 | 52.71 | 91.44 |
| tooth_TEST2_wear_level2 | GPMM | 99.95 | 0.0619 | 0.364 | 0.0746 | 0.0737 | 0.0621 | 0.0617 | 15.68 | 43.83 | 89.82 |
| tooth_TEST2_wear_level2 | Patch / Hole Filling | 99.98 | 0.0401 | 0.248 | 0.0445 | 0.0460 | 0.0391 | 0.0410 | 25.32 | 67.42 | 99.47 |
| tooth_TEST2_wear_level3 | Global SSM | 99.96 | 0.0552 | 0.386 | 0.0678 | 0.0669 | 0.0555 | 0.0549 | 18.92 | 52.37 | 91.95 |
| tooth_TEST2_wear_level3 | Neighborhood SSM | 99.96 | 0.0553 | 0.337 | 0.0682 | 0.0667 | 0.0556 | 0.0549 | 19.24 | 52.70 | 91.34 |
| tooth_TEST2_wear_level3 | GPMM | 99.93 | 0.0754 | 0.429 | 0.0900 | 0.0899 | 0.0753 | 0.0755 | 12.48 | 35.11 | 80.97 |
| tooth_TEST2_wear_level3 | Patch / Hole Filling | 99.96 | 0.0599 | 0.306 | 0.0667 | 0.0692 | 0.0588 | 0.0611 | 10.24 | 41.18 | 95.22 |
| tooth_TEST2_wear_level4 | Global SSM | 99.96 | 0.0559 | 0.369 | 0.0695 | 0.0672 | 0.0566 | 0.0553 | 18.76 | 50.95 | 91.27 |
| tooth_TEST2_wear_level4 | Neighborhood SSM | 99.96 | 0.0554 | 0.402 | 0.0687 | 0.0674 | 0.0559 | 0.0549 | 19.50 | 52.31 | 91.62 |
| tooth_TEST2_wear_level4 | GPMM | 99.91 | 0.0841 | 0.426 | 0.0999 | 0.0981 | 0.0846 | 0.0837 | 9.12 | 28.55 | 75.59 |
| tooth_TEST2_wear_level4 | Patch / Hole Filling | 99.95 | 0.0684 | 0.322 | 0.0761 | 0.0790 | 0.0671 | 0.0697 | 9.45 | 31.63 | 90.47 |
| tooth_TEST2_wear_level5 | Global SSM | 99.96 | 0.0561 | 0.394 | 0.0705 | 0.0683 | 0.0566 | 0.0556 | 19.41 | 51.80 | 91.05 |
| tooth_TEST2_wear_level5 | Neighborhood SSM | 99.96 | 0.0552 | 0.379 | 0.0690 | 0.0674 | 0.0556 | 0.0548 | 19.64 | 53.15 | 91.63 |
| tooth_TEST2_wear_level5 | GPMM | 99.87 | 0.1005 | 0.482 | 0.1201 | 0.1166 | 0.1015 | 0.0994 | 7.85 | 23.19 | 63.32 |
| tooth_TEST2_wear_level5 | Patch / Hole Filling | 99.90 | 0.0916 | 0.502 | 0.1068 | 0.1065 | 0.0904 | 0.0928 | 8.50 | 24.86 | 70.26 |
| tooth_TEST2_wear_level6 | Global SSM | 99.89 | 0.0693 | 0.670 | 0.1106 | 0.0838 | 0.0747 | 0.0639 | 17.19 | 48.78 | 84.83 |
| tooth_TEST2_wear_level6 | Neighborhood SSM | 99.89 | 0.0693 | 0.670 | 0.1104 | 0.0854 | 0.0743 | 0.0643 | 18.17 | 49.23 | 84.51 |
| tooth_TEST2_wear_level6 | GPMM | 99.69 | 0.1377 | 0.854 | 0.1858 | 0.1612 | 0.1452 | 0.1302 | 7.44 | 19.67 | 49.65 |
| tooth_TEST2_wear_level6 | Patch / Hole Filling | 99.71 | 0.1369 | 1.031 | 0.1799 | 0.1657 | 0.1370 | 0.1369 | 8.61 | 24.62 | 51.83 |
| tooth_TEST2_wear_level7 | Global SSM | 99.90 | 0.0696 | 0.621 | 0.1051 | 0.0878 | 0.0740 | 0.0652 | 18.58 | 49.16 | 82.31 |
| tooth_TEST2_wear_level7 | Neighborhood SSM | 99.83 | 0.0757 | 0.829 | 0.1382 | 0.0930 | 0.0847 | 0.0666 | 18.94 | 50.45 | 82.26 |
| tooth_TEST2_wear_level7 | GPMM | 99.23 | 0.2037 | 1.266 | 0.2940 | 0.2342 | 0.2216 | 0.1859 | 4.90 | 14.02 | 37.18 |
| tooth_TEST2_wear_level7 | Patch / Hole Filling | 99.31 | 0.2003 | 1.172 | 0.2768 | 0.2316 | 0.2129 | 0.1878 | 5.49 | 14.84 | 36.28 |
---

## Dataset A: Old Dataset (TEST1 vs n0245, TEST2 vs n0257)

Sixteen cases total: two synthetic wear series (TEST1 and TEST2), each with eight progressive wear levels (Molnar stages three through six), each compared against its own true original specimen (n0245 for TEST1, n0257 for TEST2).

### Mean Across All 16 Cases

| Method | Full Chamfer (mm) | Full RMSE (mm) | Full Hausdorff (mm) | Full Coverage at 2x spacing | Worn region RMSE (mm) | Worn region MAE (mm) |
|---|---:|---:|---:|---:|---:|---:|
| Global SSM | 0.0572 | 0.0833 | 0.5306 | 0.3038 | 0.0688 | 0.0538 |
| Neighborhood SSM | 0.0582 | 0.0854 | 0.5225 | 0.2948 | 0.0696 | 0.0545 |
| Patch/Holes | 0.0393 | 0.0631 | 0.4928 | 0.7746 | 0.0911 | 0.0693 |
| GPMM | 0.0709 | 0.0952 | 0.4621 | 0.2661 | 0.0864 | 0.0684 |

Lower is better for every column except Coverage, where higher is better.

Global SSM and Neighborhood SSM are effectively tied and are the best restorers of the group once the metric is fairly restricted to the worn region. Patch/Holes has by far the best full surface score and the best coverage, since it changes the fewest points of any method and keeps the real scan almost everywhere, but this is also why it has the worst worn region score of the four. GPMM sits in between on full surface accuracy and close to Patch/Holes on worn region accuracy, somewhat behind plain SSM.

### Full Detail, All 16 Cases x All 4 Methods (64 rows)

| Set | Level | Method | Full Chamfer (mm) | Full RMSE (mm) | Full Hausdorff (mm) | Full Coverage at 2x | Worn RMSE (mm) | Worn MAE (mm) | Worn N points |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TEST1 | 0 | Global SSM | 0.049 | 0.0601 | 0.2952 | 0.3333 | 0.0586 | 0.0481 | 98227 |
| TEST1 | 0 | Neighborhood SSM | 0.0486 | 0.0597 | 0.2998 | 0.3337 | 0.0581 | 0.0476 | 98227 |
| TEST1 | 0 | Patch/Holes | 0.0171 | 0.0198 | 0.2025 | 0.8998 | 0.0363 | 0.0284 | 5940 |
| TEST1 | 0 | GPMM | 0.0577 | 0.0712 | 0.4049 | 0.2789 | 0.0696 | 0.057 | 98227 |
| TEST1 | 1 | Global SSM | 0.0496 | 0.0619 | 0.3247 | 0.334 | 0.0593 | 0.0484 | 98253 |
| TEST1 | 1 | Neighborhood SSM | 0.0511 | 0.0634 | 0.3213 | 0.3202 | 0.061 | 0.0499 | 98253 |
| TEST1 | 1 | Patch/Holes | 0.0174 | 0.0209 | 0.2479 | 0.8945 | 0.0489 | 0.0374 | 5752 |
| TEST1 | 1 | GPMM | 0.056 | 0.0693 | 0.3594 | 0.2902 | 0.0677 | 0.0553 | 98253 |
| TEST1 | 2 | Global SSM | 0.052 | 0.0663 | 0.3547 | 0.312 | 0.0613 | 0.0502 | 98478 |
| TEST1 | 2 | Neighborhood SSM | 0.0527 | 0.0666 | 0.3502 | 0.298 | 0.062 | 0.0511 | 98478 |
| TEST1 | 2 | Patch/Holes | 0.0204 | 0.0255 | 0.3681 | 0.8699 | 0.059 | 0.0476 | 12366 |
| TEST1 | 2 | GPMM | 0.0588 | 0.0728 | 0.3671 | 0.277 | 0.0705 | 0.0577 | 98478 |
| TEST1 | 3 | Global SSM | 0.0537 | 0.0714 | 0.4446 | 0.307 | 0.0627 | 0.0512 | 98447 |
| TEST1 | 3 | Neighborhood SSM | 0.0552 | 0.0722 | 0.4393 | 0.2793 | 0.064 | 0.0529 | 98447 |
| TEST1 | 3 | Patch/Holes | 0.0218 | 0.0322 | 0.478 | 0.8617 | 0.0665 | 0.0527 | 14103 |
| TEST1 | 3 | GPMM | 0.0596 | 0.075 | 0.3718 | 0.2755 | 0.0715 | 0.0583 | 98447 |
| TEST1 | 4 | Global SSM | 0.0563 | 0.0824 | 0.5771 | 0.316 | 0.0662 | 0.0523 | 98463 |
| TEST1 | 4 | Neighborhood SSM | 0.0579 | 0.0828 | 0.5655 | 0.2944 | 0.0669 | 0.0539 | 98463 |
| TEST1 | 4 | Patch/Holes | 0.0268 | 0.0542 | 0.5952 | 0.8584 | 0.109 | 0.0825 | 17949 |
| TEST1 | 4 | GPMM | 0.0596 | 0.0781 | 0.3707 | 0.2747 | 0.0714 | 0.0576 | 98463 |
| TEST1 | 5 | Global SSM | 0.0586 | 0.0873 | 0.7002 | 0.2965 | 0.0688 | 0.0544 | 98345 |
| TEST1 | 5 | Neighborhood SSM | 0.0597 | 0.0872 | 0.6885 | 0.2812 | 0.0689 | 0.0555 | 98345 |
| TEST1 | 5 | Patch/Holes | 0.0264 | 0.0538 | 0.6369 | 0.8486 | 0.0874 | 0.0666 | 20423 |
| TEST1 | 5 | GPMM | 0.0606 | 0.0771 | 0.3687 | 0.2692 | 0.0729 | 0.059 | 98345 |
| TEST1 | 6 | Global SSM | 0.0839 | 0.1937 | 1.2901 | 0.298 | 0.0896 | 0.0632 | 98551 |
| TEST1 | 6 | Neighborhood SSM | 0.085 | 0.1939 | 1.2924 | 0.2841 | 0.0906 | 0.0644 | 98551 |
| TEST1 | 6 | Patch/Holes | 0.0576 | 0.1556 | 1.1835 | 0.7411 | 0.199 | 0.1508 | 34949 |
| TEST1 | 6 | GPMM | 0.0801 | 0.1267 | 0.6856 | 0.245 | 0.0988 | 0.0726 | 98551 |
| TEST1 | 7 | Global SSM | 0.0618 | 0.1039 | 0.8185 | 0.3266 | 0.0731 | 0.0553 | 98335 |
| TEST1 | 7 | Neighborhood SSM | 0.0648 | 0.1052 | 0.8105 | 0.2988 | 0.0768 | 0.0587 | 98335 |
| TEST1 | 7 | Patch/Holes | 0.1965 | 0.2795 | 0.9332 | 0.101 | 0.1532 | 0.1101 | 41792 |
| TEST1 | 7 | GPMM | 0.2133 | 0.2836 | 0.9003 | 0.0803 | 0.2619 | 0.2065 | 98335 |
| TEST2 | 0 | Global SSM | 0.0518 | 0.0633 | 0.4059 | 0.3003 | 0.0647 | 0.0517 | 98571 |
| TEST2 | 0 | Neighborhood SSM | 0.0516 | 0.0633 | 0.3309 | 0.2949 | 0.0636 | 0.0512 | 98571 |
| TEST2 | 0 | Patch/Holes | 0.0164 | 0.0195 | 0.1736 | 0.9001 | 0.0465 | 0.0357 | 2774 |
| TEST2 | 0 | GPMM | 0.0496 | 0.0608 | 0.2925 | 0.3084 | 0.0601 | 0.0491 | 98571 |
| TEST2 | 1 | Global SSM | 0.0519 | 0.0635 | 0.3981 | 0.3007 | 0.0645 | 0.0517 | 98714 |
| TEST2 | 1 | Neighborhood SSM | 0.0518 | 0.0639 | 0.32 | 0.2978 | 0.0637 | 0.0513 | 98714 |
| TEST2 | 1 | Patch/Holes | 0.0168 | 0.0207 | 0.2377 | 0.8981 | 0.0521 | 0.0391 | 4593 |
| TEST2 | 1 | GPMM | 0.0496 | 0.0609 | 0.2822 | 0.3093 | 0.0599 | 0.049 | 98714 |
| TEST2 | 2 | Global SSM | 0.053 | 0.0654 | 0.4085 | 0.2964 | 0.0656 | 0.0526 | 98756 |
| TEST2 | 2 | Neighborhood SSM | 0.0533 | 0.0664 | 0.3245 | 0.2961 | 0.0655 | 0.0525 | 98756 |
| TEST2 | 2 | Patch/Holes | 0.0176 | 0.0216 | 0.2587 | 0.8921 | 0.0563 | 0.0439 | 7922 |
| TEST2 | 2 | GPMM | 0.0506 | 0.0622 | 0.3057 | 0.3085 | 0.0612 | 0.0499 | 98756 |
| TEST2 | 3 | Global SSM | 0.0527 | 0.0657 | 0.4013 | 0.3012 | 0.0657 | 0.0522 | 98713 |
| TEST2 | 3 | Neighborhood SSM | 0.0531 | 0.0662 | 0.3577 | 0.2967 | 0.0655 | 0.0525 | 98713 |
| TEST2 | 3 | Patch/Holes | 0.0186 | 0.023 | 0.2524 | 0.8844 | 0.0655 | 0.051 | 10070 |
| TEST2 | 3 | GPMM | 0.0507 | 0.0631 | 0.3062 | 0.3084 | 0.0614 | 0.0499 | 98713 |
| TEST2 | 4 | Global SSM | 0.0533 | 0.0671 | 0.3717 | 0.2962 | 0.0656 | 0.0525 | 98678 |
| TEST2 | 4 | Neighborhood SSM | 0.0531 | 0.0666 | 0.3907 | 0.2972 | 0.0658 | 0.0523 | 98678 |
| TEST2 | 4 | Patch/Holes | 0.0199 | 0.0258 | 0.2352 | 0.8772 | 0.0747 | 0.0589 | 12998 |
| TEST2 | 4 | GPMM | 0.0519 | 0.0648 | 0.3003 | 0.302 | 0.0627 | 0.0509 | 98678 |
| TEST2 | 5 | Global SSM | 0.054 | 0.0683 | 0.4114 | 0.295 | 0.0671 | 0.0533 | 98906 |
| TEST2 | 5 | Neighborhood SSM | 0.0533 | 0.0668 | 0.3738 | 0.291 | 0.066 | 0.0526 | 98906 |
| TEST2 | 5 | Patch/Holes | 0.0232 | 0.035 | 0.3983 | 0.8623 | 0.0849 | 0.066 | 20097 |
| TEST2 | 5 | GPMM | 0.0512 | 0.0637 | 0.3103 | 0.309 | 0.0624 | 0.0505 | 98906 |
| TEST2 | 6 | Global SSM | 0.0673 | 0.1088 | 0.6693 | 0.2606 | 0.0826 | 0.0617 | 99043 |
| TEST2 | 6 | Neighborhood SSM | 0.0672 | 0.108 | 0.6687 | 0.266 | 0.0837 | 0.062 | 99043 |
| TEST2 | 6 | Patch/Holes | 0.0359 | 0.073 | 0.7295 | 0.7925 | 0.1253 | 0.095 | 34265 |
| TEST2 | 6 | GPMM | 0.0631 | 0.0955 | 0.6832 | 0.2729 | 0.0785 | 0.0593 | 99043 |
| TEST2 | 7 | Global SSM | 0.067 | 0.1033 | 0.6183 | 0.2876 | 0.0861 | 0.0623 | 98963 |
| TEST2 | 7 | Neighborhood SSM | 0.0727 | 0.1344 | 0.8255 | 0.287 | 0.0907 | 0.0638 | 98963 |
| TEST2 | 7 | Patch/Holes | 0.0969 | 0.1491 | 0.9541 | 0.2125 | 0.1932 | 0.1434 | 44277 |
| TEST2 | 7 | GPMM | 0.1222 | 0.1985 | 1.0852 | 0.1479 | 0.152 | 0.1118 | 98963 |

---

## Dataset B: v5 Dataset (8 Real Specimens x 8 Molnar Wear Levels)

Sixty four cases total. Eight real specimens (N1063, N332, N4, N459, N705, N726, N728, N891), each with eight progressive Molnar wear levels, each compared against its own corresponded original tooth. Ground truth is exact and index paired for Global SSM, Neighborhood SSM, and GPMM, so no ICP alignment is needed for those three. Patch/Holes is scored via nearest neighbor search since it appends new points. See the known limitation note above regarding mask comparability on this dataset.

### Mean Across All 64 Cases

| Method | Full Metric (mm) | Worn Region RMSE (mm) | Worn Region MAE (mm) | Mean Worn Points out of 10000 |
|---|---:|---:|---:|---:|
| Global SSM | 0.02633 | 0.02716 | 0.02462 | 8508 |
| Neighborhood SSM | 0.02626 | 0.02712 | 0.02460 | 8508 |
| GPMM | 0.02688 | 0.02767 | 0.02488 | 8508 |
| Patch/Holes | 0.01251 | 0.01663 | 0.01385 | 5462 |

The full metric column is full RMSE for Global SSM, Neighborhood SSM, and GPMM, computed exactly since those methods are index paired against the original. For Patch/Holes it is full Chamfer, since that method's point cloud is not index paired.

Neighborhood SSM and Global SSM are essentially tied for best among the three index paired methods, with GPMM close behind. Patch/Holes shows the lowest numbers of all four, but read this together with the known limitation note above: its worn region is self selected and roughly forty five percent smaller on average than the shared mask used by the other three, so part of its advantage here comes from evaluating a smaller, easier region rather than a strictly fair comparison.

### Full Detail, Per Specimen, All 8 Levels x All 4 Methods

For each specimen below, the first table gives the full metric per level per method (full RMSE for the three index paired methods, full Chamfer for Patch/Holes). The second table gives the worn region RMSE per level per method.

#### Specimen N1063

Full metric (mm): full RMSE for Global/Neighborhood SSM and GPMM; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01625 | 0.01576 | 0.01711 | 0.01011 |
| 1 | 0.00782 | 0.00774 | 0.00862 | 0.01011 |
| 2 | 0.0169 | 0.01707 | 0.01739 | 0.01142 |
| 3 | 0.02531 | 0.02518 | 0.02619 | 0.01225 |
| 4 | 0.03392 | 0.03367 | 0.0354 | 0.01331 |
| 5 | 0.04063 | 0.04137 | 0.04446 | 0.01466 |
| 6 | 0.03995 | 0.04022 | 0.0422 | 0.01477 |
| 7 | 0.04284 | 0.04272 | 0.0456 | 0.01313 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01748 | 0.01694 | 0.01841 | 0.01098 |
| 1 | 0.01068 | 0.01058 | 0.01158 | 0.01056 |
| 2 | 0.01818 | 0.01835 | 0.01871 | 0.013 |
| 3 | 0.02556 | 0.02543 | 0.02645 | 0.01563 |
| 4 | 0.03411 | 0.03385 | 0.0356 | 0.01823 |
| 5 | 0.04089 | 0.04164 | 0.04474 | 0.02228 |
| 6 | 0.04072 | 0.041 | 0.04297 | 0.02011 |
| 7 | 0.04293 | 0.04281 | 0.0457 | 0.017 |

#### Specimen N332

Full metric (mm): full RMSE for Global/Neighborhood SSM and GPMM; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.0102 | 0.01041 | 0.01142 | 0.01182 |
| 1 | 0.02279 | 0.02293 | 0.02424 | 0.01302 |
| 2 | 0.02666 | 0.02688 | 0.02879 | 0.01332 |
| 3 | 0.01562 | 0.01522 | 0.01728 | 0.01194 |
| 4 | 0.0208 | 0.02059 | 0.0229 | 0.01271 |
| 5 | 0.02249 | 0.02274 | 0.02498 | 0.01262 |
| 6 | 0.02958 | 0.02916 | 0.03141 | 0.01342 |
| 7 | 0.02964 | 0.0296 | 0.03241 | 0.0128 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01108 | 0.01127 | 0.0125 | 0.01058 |
| 1 | 0.02293 | 0.02308 | 0.0244 | 0.01705 |
| 2 | 0.02685 | 0.02706 | 0.02899 | 0.01658 |
| 3 | 0.01623 | 0.01581 | 0.01796 | 0.01389 |
| 4 | 0.0217 | 0.02148 | 0.02391 | 0.01662 |
| 5 | 0.02266 | 0.02291 | 0.02517 | 0.01529 |
| 6 | 0.02977 | 0.02935 | 0.03162 | 0.01791 |
| 7 | 0.03043 | 0.03039 | 0.03328 | 0.01838 |

#### Specimen N4

Full metric (mm): full RMSE for Global/Neighborhood SSM and GPMM; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01204 | 0.01054 | 0.00999 | 0.01103 |
| 1 | 0.01397 | 0.01295 | 0.01268 | 0.01129 |
| 2 | 0.0089 | 0.00742 | 0.00606 | 0.0108 |
| 3 | 0.01373 | 0.01217 | 0.01148 | 0.01105 |
| 4 | 0.02006 | 0.01968 | 0.019 | 0.01195 |
| 5 | 0.01961 | 0.01902 | 0.01885 | 0.01156 |
| 6 | 0.0503 | 0.05034 | 0.03275 | 0.01834 |
| 7 | 0.0494 | 0.04873 | 0.03146 | 0.01846 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01386 | 0.01201 | 0.01142 | 0.01284 |
| 1 | 0.01498 | 0.01387 | 0.01357 | 0.01141 |
| 2 | 0.00871 | 0.00882 | 0.00875 | 0.0121 |
| 3 | 0.01469 | 0.01298 | 0.01238 | 0.01171 |
| 4 | 0.02055 | 0.02022 | 0.01958 | 0.01574 |
| 5 | 0.01997 | 0.01985 | 0.01971 | 0.01488 |
| 6 | 0.0503 | 0.05034 | 0.03275 | 0.02846 |
| 7 | 0.04953 | 0.04886 | 0.03153 | 0.02823 |

#### Specimen N459

Full metric (mm): full RMSE for Global/Neighborhood SSM and GPMM; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.00874 | 0.00824 | 0.00932 | 0.01036 |
| 1 | 0.0125 | 0.01233 | 0.01262 | 0.01115 |
| 2 | 0.02374 | 0.02353 | 0.02499 | 0.012 |
| 3 | 0.0229 | 0.02274 | 0.02417 | 0.01239 |
| 4 | 0.04884 | 0.04958 | 0.05122 | 0.01872 |
| 5 | 0.05325 | 0.05356 | 0.05534 | 0.01876 |
| 6 | 0.05548 | 0.05589 | 0.05845 | 0.01939 |
| 7 | 0.05346 | 0.05406 | 0.05663 | 0.01845 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01077 | 0.01009 | 0.01156 | 0.0114 |
| 1 | 0.01312 | 0.01299 | 0.0133 | 0.01577 |
| 2 | 0.02403 | 0.02382 | 0.0253 | 0.0164 |
| 3 | 0.0235 | 0.02334 | 0.02481 | 0.0171 |
| 4 | 0.04892 | 0.04967 | 0.05131 | 0.02782 |
| 5 | 0.05339 | 0.0537 | 0.05548 | 0.02972 |
| 6 | 0.05567 | 0.05607 | 0.05864 | 0.02709 |
| 7 | 0.05371 | 0.05431 | 0.05688 | 0.02581 |

#### Specimen N705

Full metric (mm): full RMSE for Global/Neighborhood SSM and GPMM; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.0092 | 0.00823 | 0.00698 | 0.00806 |
| 1 | 0.00707 | 0.00635 | 0.00568 | 0.00828 |
| 2 | 0.01379 | 0.01382 | 0.01388 | 0.00835 |
| 3 | 0.01075 | 0.0102 | 0.01088 | 0.00857 |
| 4 | 0.0435 | 0.04395 | 0.04482 | 0.0196 |
| 5 | 0.02023 | 0.02085 | 0.02135 | 0.00939 |
| 6 | 0.02763 | 0.03042 | 0.0329 | 0.01247 |
| 7 | 0.03165 | 0.03328 | 0.0366 | 0.01314 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01066 | 0.01207 | 0.01013 | 0.00842 |
| 1 | 0.01071 | 0.00888 | 0.00854 | 0.01349 |
| 2 | 0.01506 | 0.01506 | 0.01525 | 0.01064 |
| 3 | 0.01287 | 0.01243 | 0.01311 | 0.01487 |
| 4 | 0.04352 | 0.04397 | 0.04485 | 0.02483 |
| 5 | 0.02103 | 0.02175 | 0.02229 | 0.01512 |
| 6 | 0.02796 | 0.0308 | 0.03332 | 0.02213 |
| 7 | 0.03176 | 0.03341 | 0.03674 | 0.02177 |

#### Specimen N726

Full metric (mm): full RMSE for Global/Neighborhood SSM and GPMM; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.0104 | 0.00941 | 0.01053 | 0.01074 |
| 1 | 0.00872 | 0.00827 | 0.00952 | 0.01032 |
| 2 | 0.0288 | 0.02786 | 0.02951 | 0.01389 |
| 3 | 0.03834 | 0.03782 | 0.03899 | 0.01539 |
| 4 | 0.03094 | 0.03038 | 0.03219 | 0.01318 |
| 5 | 0.04424 | 0.04344 | 0.04591 | 0.01523 |
| 6 | 0.0435 | 0.04349 | 0.04554 | 0.01601 |
| 7 | 0.05441 | 0.05436 | 0.05679 | 0.01929 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01134 | 0.01065 | 0.0112 | 0.01021 |
| 1 | 0.01154 | 0.011 | 0.01114 | 0.01075 |
| 2 | 0.02923 | 0.02837 | 0.03005 | 0.01989 |
| 3 | 0.0388 | 0.03828 | 0.03946 | 0.01998 |
| 4 | 0.03169 | 0.03114 | 0.033 | 0.02088 |
| 5 | 0.04433 | 0.04353 | 0.046 | 0.02254 |
| 6 | 0.04353 | 0.04352 | 0.04557 | 0.02335 |
| 7 | 0.05448 | 0.05443 | 0.05686 | 0.02616 |

#### Specimen N728

Full metric (mm): full RMSE for Global/Neighborhood SSM and GPMM; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.00658 | 0.00607 | 0.00641 | 0.00741 |
| 1 | 0.01079 | 0.0109 | 0.0115 | 0.00757 |
| 2 | 0.01317 | 0.01433 | 0.01562 | 0.00781 |
| 3 | 0.01802 | 0.01905 | 0.01991 | 0.00823 |
| 4 | 0.02369 | 0.02514 | 0.01825 | 0.00854 |
| 5 | 0.02739 | 0.02841 | 0.02967 | 0.00801 |
| 6 | 0.03791 | 0.03904 | 0.04082 | 0.01075 |
| 7 | 0.03977 | 0.0401 | 0.04272 | 0.01139 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01003 | 0.00902 | 0.00953 | 0.00688 |
| 1 | 0.01196 | 0.01225 | 0.01295 | 0.00804 |
| 2 | 0.01395 | 0.0152 | 0.01659 | 0.00995 |
| 3 | 0.0185 | 0.01956 | 0.02045 | 0.01042 |
| 4 | 0.02405 | 0.02551 | 0.01833 | 0.01166 |
| 5 | 0.02766 | 0.02869 | 0.02996 | 0.0114 |
| 6 | 0.0382 | 0.03934 | 0.04113 | 0.01485 |
| 7 | 0.03983 | 0.04016 | 0.04278 | 0.01904 |

#### Specimen N891

Full metric (mm): full RMSE for Global/Neighborhood SSM and GPMM; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01 | 0.00828 | 0.00792 | 0.00774 |
| 1 | 0.0097 | 0.00842 | 0.00799 | 0.00755 |
| 2 | 0.01092 | 0.01046 | 0.01657 | 0.00747 |
| 3 | 0.01425 | 0.01353 | 0.01432 | 0.00754 |
| 4 | 0.05055 | 0.05067 | 0.05213 | 0.02101 |
| 5 | 0.01447 | 0.01441 | 0.01705 | 0.00783 |
| 6 | 0.05238 | 0.05291 | 0.05463 | 0.02139 |
| 7 | 0.05416 | 0.05447 | 0.05728 | 0.02112 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Patch/Holes |
|---|---:|---:|---:|---:|
| 0 | 0.01276 | 0.01044 | 0.00986 | 0.01066 |
| 1 | 0.01292 | 0.01071 | 0.01035 | 0.00841 |
| 2 | 0.01348 | 0.01353 | 0.01515 | 0.00844 |
| 3 | 0.01567 | 0.01512 | 0.01603 | 0.0098 |
| 4 | 0.05061 | 0.05073 | 0.05219 | 0.02575 |
| 5 | 0.0159 | 0.01592 | 0.01698 | 0.01055 |
| 6 | 0.05238 | 0.05291 | 0.05463 | 0.02691 |
| 7 | 0.05422 | 0.05453 | 0.05734 | 0.02583 |

---

## File Pointers

| File | Contents |
|---|---|
| `ssm_pipeline/output/eval_flagship_19teeth.csv` | Raw 76 row table behind the flagship comparison above, one row per tooth and method, scored against the raw worn tooth scan. |
| `ssm_pipeline/output/eval_old_dataset.csv` | Raw 64 row table behind the Dataset A tables above (scored against the true original), one row per set, level, and method, with every column including full RMSE, full Hausdorff, and full Coverage at 2x spacing that are summarized here. |
| `ssm_pipeline/output/eval_v5_dataset.csv` | Raw 256 row table behind the Dataset B tables above. |
| `ssm_pipeline/evaluate_all_methods.py` | The script that produced the old and v5 dataset CSVs (the ones scored against the true original). Run with `--dataset old`, `--dataset v5`, or `--dataset both`. |
| `ssm_pipeline/output/archive/patch_method_chamfer.csv` | An earlier, smaller evaluation pass kept for history. It used a different, less rigorous worn region mask and does not include GPMM. Superseded by the tables in this file. |

## Reproducing These Numbers

```bash
cd ssm_pipeline
conda activate teeth
python3 evaluate_all_methods.py --dataset both
```

This reads reconstructions that already exist under `output/`, so the correspondence and reconstruction pipelines (Stage 1 and Stage 2) must have already been run for both datasets before this command will produce results. See the Quick Start section of the README for those commands.

The flagship comparison against the raw worn tooth scan is not produced by `evaluate_all_methods.py`; it requires a local, verified copy of the original raw worn meshes for the old dataset's real teeth and TEST1/TEST2 levels, matched against each method's `reconstructed_in_input_space.ply` or equivalent.
