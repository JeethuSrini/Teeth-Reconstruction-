# Reconstruction Results, Full Detail

This file holds every numeric result behind the summary shown in [README.md](README.md). It covers all four reconstruction methods (Global SSM, Neighborhood SSM, Blend, Patch/Hole Filling) plus the GPMM posterior variant, scored on two independent test sets. See the README for method descriptions and images; this file is numbers only.

Raw, per case data lives in [`ssm_pipeline/output/eval_old_dataset.csv`](ssm_pipeline/output/eval_old_dataset.csv) and [`ssm_pipeline/output/eval_v5_dataset.csv`](ssm_pipeline/output/eval_v5_dataset.csv), produced by [`ssm_pipeline/evaluate_all_methods.py`](ssm_pipeline/evaluate_all_methods.py).

## How These Numbers Were Computed

Every reconstruction is compared against the true, original, unworn tooth in two ways.

**Full surface metric.** The reconstruction is compared to the original over every point in the cloud. This rewards a method for keeping the real surface wherever it is intact, so on its own it can make a method that barely changes anything look artificially good.

**Worn region metric.** The same comparison, but restricted only to the points that were genuinely worn away. This is the fairer test of restoration quality, since a method cannot score well here just by leaving most of the tooth untouched.

For the old dataset, the worn region mask is ground truth anchored and shared by every method except Patch/Holes. It is built from where the real worn scan sits recessed relative to the true original mesh, so no single method's own opinion of what counts as worn biases the comparison. Patch/Holes appends new points rather than replacing points in place, so it is scored separately using its own graft mask, meaning exactly the points it decided to fill in.

For the v5 dataset, ground truth is the corresponded original tooth in the same point ordering as the worn input, so no ICP alignment or nearest neighbor search is needed for the paired methods (Global SSM, Neighborhood SSM, GPMM, Blend). Their worn region mask comes from where the true original deviates from the worn input by more than one and a half times the median point spacing. Patch/Holes again appends points and is scored with its own graft mask against the nearest point in the true original.

**Known limitation.** On the v5 dataset the shared mask (used by four of the five methods) flags roughly eighty five percent of all points as worn, on average, which is broader than intended and likely reflects real specimen correspondence noise rather than only genuine wear. Patch/Holes' own graft mask is much smaller and highly variable case to case (ranging from about five percent to one hundred percent of points). This means the Patch/Holes numbers on v5 are not perfectly apples to apples with the other four methods there. The old dataset does not have this issue, since its ground truth anchored mask is more reliable.

---

## Dataset A: Old Dataset (TEST1 vs n0245, TEST2 vs n0257)

Sixteen cases total: two synthetic wear series (TEST1 and TEST2), each with eight progressive wear levels (Molnar stages three through six), each compared against its own true original specimen (n0245 for TEST1, n0257 for TEST2).

### Mean Across All 16 Cases

| Method | Full Chamfer (mm) | Full RMSE (mm) | Full Hausdorff (mm) | Full Coverage at 2x spacing | Worn region RMSE (mm) | Worn region MAE (mm) |
|---|---:|---:|---:|---:|---:|---:|
| Global SSM | 0.0572 | 0.0833 | 0.5306 | 0.3038 | 0.0688 | 0.0538 |
| Neighborhood SSM | 0.0582 | 0.0854 | 0.5225 | 0.2948 | 0.0696 | 0.0545 |
| Blend | 0.0703 | 0.1017 | 0.5586 | 0.2776 | 0.0839 | 0.0659 |
| Patch/Holes | 0.0393 | 0.0631 | 0.4928 | 0.7746 | 0.0911 | 0.0693 |
| GPMM | 0.0709 | 0.0952 | 0.4621 | 0.2661 | 0.0864 | 0.0684 |
| GPMM (kernel augmented) | 0.0709 | 0.0952 | 0.4621 | 0.2660 | 0.0864 | 0.0684 |

Lower is better for every column except Coverage, where higher is better.

Global SSM and Neighborhood SSM are effectively tied and are the best restorers of the group once the metric is fairly restricted to the worn region. Patch/Holes has by far the best full surface score and the best coverage, since it changes the fewest points of any method and keeps the real scan almost everywhere, but this is also why it has the worst worn region score of the six. Blend and both GPMM variants sit in between on full surface accuracy and roughly tie each other on worn region accuracy, somewhat behind plain SSM.

### Full Detail, All 16 Cases x All 6 Methods (96 rows)

| Set | Level | Method | Full Chamfer (mm) | Full RMSE (mm) | Full Hausdorff (mm) | Full Coverage at 2x | Worn RMSE (mm) | Worn MAE (mm) | Worn N points |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TEST1 | 0 | Global SSM | 0.049 | 0.0601 | 0.2952 | 0.3333 | 0.0586 | 0.0481 | 98227 |
| TEST1 | 0 | Neighborhood SSM | 0.0486 | 0.0597 | 0.2998 | 0.3337 | 0.0581 | 0.0476 | 98227 |
| TEST1 | 0 | Blend | 0.0487 | 0.0602 | 0.2943 | 0.3394 | 0.0581 | 0.0476 | 98227 |
| TEST1 | 0 | Patch/Holes | 0.0171 | 0.0198 | 0.2025 | 0.8998 | 0.0363 | 0.0284 | 5940 |
| TEST1 | 0 | GPMM | 0.0577 | 0.0712 | 0.4049 | 0.2789 | 0.0696 | 0.057 | 98227 |
| TEST1 | 0 | GPMM (kernel) | 0.0577 | 0.0713 | 0.405 | 0.2788 | 0.0696 | 0.057 | 98227 |
| TEST1 | 1 | Global SSM | 0.0501 | 0.0617 | 0.3231 | 0.325 | 0.0605 | 0.0495 | 98227 |
| TEST1 | 1 | Neighborhood SSM | 0.0511 | 0.0631 | 0.3241 | 0.3196 | 0.0615 | 0.0503 | 98227 |
| TEST1 | 1 | Blend | 0.0501 | 0.0619 | 0.3221 | 0.3283 | 0.0605 | 0.0495 | 98227 |
| TEST1 | 1 | Patch/Holes | 0.0176 | 0.0203 | 0.2049 | 0.8969 | 0.0369 | 0.0289 | 5988 |
| TEST1 | 1 | GPMM | 0.0592 | 0.0725 | 0.4106 | 0.276 | 0.0709 | 0.0582 | 98227 |
| TEST1 | 1 | GPMM (kernel) | 0.0592 | 0.0725 | 0.4102 | 0.2762 | 0.0709 | 0.0582 | 98227 |
| TEST1 | 2 | Global SSM | 0.0532 | 0.0654 | 0.3474 | 0.313 | 0.0645 | 0.0523 | 98227 |
| TEST1 | 2 | Neighborhood SSM | 0.0527 | 0.0648 | 0.3499 | 0.3163 | 0.0638 | 0.0517 | 98227 |
| TEST1 | 2 | Blend | 0.0534 | 0.0656 | 0.3502 | 0.3121 | 0.0647 | 0.0525 | 98227 |
| TEST1 | 2 | Patch/Holes | 0.0225 | 0.0267 | 0.2379 | 0.8543 | 0.0451 | 0.0353 | 8010 |
| TEST1 | 2 | GPMM | 0.0621 | 0.0761 | 0.4231 | 0.2691 | 0.0745 | 0.0611 | 98227 |
| TEST1 | 2 | GPMM (kernel) | 0.0621 | 0.0761 | 0.4232 | 0.2691 | 0.0745 | 0.0611 | 98227 |
| TEST1 | 3 | Global SSM | 0.0557 | 0.0691 | 0.3711 | 0.3016 | 0.068 | 0.0547 | 98227 |
| TEST1 | 3 | Neighborhood SSM | 0.0552 | 0.0684 | 0.3675 | 0.3049 | 0.0674 | 0.0542 | 98227 |
| TEST1 | 3 | Blend | 0.0567 | 0.0704 | 0.3778 | 0.297 | 0.0693 | 0.0559 | 98227 |
| TEST1 | 3 | Patch/Holes | 0.024 | 0.0287 | 0.2517 | 0.8265 | 0.0479 | 0.0373 | 8971 |
| TEST1 | 3 | GPMM | 0.065 | 0.0798 | 0.4373 | 0.2606 | 0.078 | 0.0641 | 98227 |
| TEST1 | 3 | GPMM (kernel) | 0.065 | 0.0798 | 0.4374 | 0.2607 | 0.078 | 0.0641 | 98227 |
| TEST1 | 4 | Global SSM | 0.0583 | 0.0724 | 0.3927 | 0.2917 | 0.0713 | 0.0568 | 98227 |
| TEST1 | 4 | Neighborhood SSM | 0.0578 | 0.0719 | 0.3902 | 0.2946 | 0.0708 | 0.0564 | 98227 |
| TEST1 | 4 | Blend | 0.0587 | 0.073 | 0.3945 | 0.2894 | 0.0716 | 0.0572 | 98227 |
| TEST1 | 4 | Patch/Holes | 0.0296 | 0.0357 | 0.2673 | 0.7938 | 0.0521 | 0.0405 | 10998 |
| TEST1 | 4 | GPMM | 0.0682 | 0.0836 | 0.4507 | 0.2521 | 0.0817 | 0.0673 | 98227 |
| TEST1 | 4 | GPMM (kernel) | 0.0682 | 0.0836 | 0.4507 | 0.2521 | 0.0817 | 0.0673 | 98227 |
| TEST1 | 5 | Global SSM | 0.0605 | 0.0752 | 0.4088 | 0.2842 | 0.0743 | 0.0588 | 98227 |
| TEST1 | 5 | Neighborhood SSM | 0.0604 | 0.0752 | 0.4046 | 0.2838 | 0.0741 | 0.0587 | 98227 |
| TEST1 | 5 | Blend | 0.0604 | 0.0752 | 0.4085 | 0.2846 | 0.0742 | 0.0588 | 98227 |
| TEST1 | 5 | Patch/Holes | 0.0298 | 0.0359 | 0.2735 | 0.7842 | 0.053 | 0.0412 | 11298 |
| TEST1 | 5 | GPMM | 0.0716 | 0.0879 | 0.462 | 0.2455 | 0.0855 | 0.0704 | 98227 |
| TEST1 | 5 | GPMM (kernel) | 0.0716 | 0.0879 | 0.4618 | 0.2455 | 0.0855 | 0.0704 | 98227 |
| TEST1 | 6 | Global SSM | 0.0836 | 0.1149 | 0.6067 | 0.2233 | 0.0921 | 0.0722 | 98227 |
| TEST1 | 6 | Neighborhood SSM | 0.0848 | 0.1163 | 0.6034 | 0.2219 | 0.0932 | 0.0731 | 98227 |
| TEST1 | 6 | Blend | 0.0897 | 0.1213 | 0.6118 | 0.2124 | 0.0968 | 0.0762 | 98227 |
| TEST1 | 6 | Patch/Holes | 0.0575 | 0.0698 | 0.4106 | 0.5296 | 0.0798 | 0.0605 | 21524 |
| TEST1 | 6 | GPMM | 0.0799 | 0.101 | 0.5245 | 0.2224 | 0.0937 | 0.0776 | 98227 |
| TEST1 | 6 | GPMM (kernel) | 0.08 | 0.1011 | 0.5241 | 0.2222 | 0.0938 | 0.0777 | 98227 |
| TEST1 | 7 | Global SSM | 0.0617 | 0.0928 | 0.7051 | 0.313 | 0.0708 | 0.0504 | 98227 |
| TEST1 | 7 | Neighborhood SSM | 0.0648 | 0.0994 | 0.6913 | 0.2934 | 0.0754 | 0.0538 | 98227 |
| TEST1 | 7 | Blend | 0.2166 | 0.2778 | 0.9542 | 0.1063 | 0.2367 | 0.1889 | 98227 |
| TEST1 | 7 | Patch/Holes | 0.1973 | 0.2531 | 0.8918 | 0.1224 | 0.2145 | 0.1717 | 46934 |
| TEST1 | 7 | GPMM | 0.2153 | 0.2757 | 0.9364 | 0.1082 | 0.2352 | 0.1878 | 98227 |
| TEST1 | 7 | GPMM (kernel) | 0.2153 | 0.2761 | 0.9401 | 0.1081 | 0.2357 | 0.1882 | 98227 |
| TEST2 | 0 | Global SSM | 0.0518 | 0.0611 | 0.3095 | 0.3222 | 0.0906 | 0.0709 | 99332 |
| TEST2 | 0 | Neighborhood SSM | 0.0516 | 0.0608 | 0.3038 | 0.3226 | 0.0894 | 0.07 | 99332 |
| TEST2 | 0 | Blend | 0.0491 | 0.0589 | 0.313 | 0.3382 | 0.0721 | 0.0562 | 99332 |
| TEST2 | 0 | Patch/Holes | 0.0164 | 0.0189 | 0.1889 | 0.9033 | 0.0466 | 0.0353 | 5231 |
| TEST2 | 0 | GPMM | 0.0496 | 0.0587 | 0.2994 | 0.3287 | 0.0624 | 0.049 | 99332 |
| TEST2 | 0 | GPMM (kernel) | 0.0496 | 0.0587 | 0.2996 | 0.3286 | 0.0623 | 0.0489 | 99332 |
| TEST2 | 1 | Global SSM | 0.052 | 0.0616 | 0.3131 | 0.318 | 0.0888 | 0.0699 | 99332 |
| TEST2 | 1 | Neighborhood SSM | 0.0518 | 0.0613 | 0.3081 | 0.3184 | 0.0879 | 0.0692 | 99332 |
| TEST2 | 1 | Blend | 0.0492 | 0.0592 | 0.3216 | 0.3363 | 0.0724 | 0.0563 | 99332 |
| TEST2 | 1 | Patch/Holes | 0.017 | 0.0197 | 0.1979 | 0.897 | 0.0472 | 0.0359 | 5559 |
| TEST2 | 1 | GPMM | 0.0498 | 0.059 | 0.3005 | 0.3241 | 0.0621 | 0.0486 | 99332 |
| TEST2 | 1 | GPMM (kernel) | 0.0498 | 0.059 | 0.3011 | 0.3241 | 0.0621 | 0.0487 | 99332 |
| TEST2 | 2 | Global SSM | 0.0538 | 0.0637 | 0.3315 | 0.3068 | 0.0876 | 0.0692 | 99332 |
| TEST2 | 2 | Neighborhood SSM | 0.0538 | 0.0637 | 0.3355 | 0.3065 | 0.0876 | 0.0692 | 99332 |
| TEST2 | 2 | Blend | 0.0512 | 0.0616 | 0.3452 | 0.3244 | 0.0728 | 0.0568 | 99332 |
| TEST2 | 2 | Patch/Holes | 0.019 | 0.0224 | 0.2181 | 0.8783 | 0.0491 | 0.0378 | 6742 |
| TEST2 | 2 | GPMM | 0.0508 | 0.0605 | 0.3122 | 0.3128 | 0.0625 | 0.0489 | 99332 |
| TEST2 | 2 | GPMM (kernel) | 0.0508 | 0.0605 | 0.3135 | 0.3128 | 0.0624 | 0.0489 | 99332 |
| TEST2 | 3 | Global SSM | 0.0553 | 0.0658 | 0.3435 | 0.2967 | 0.0862 | 0.0682 | 99332 |
| TEST2 | 3 | Neighborhood SSM | 0.0554 | 0.0659 | 0.3446 | 0.2956 | 0.0864 | 0.0684 | 99332 |
| TEST2 | 3 | Blend | 0.0533 | 0.064 | 0.3573 | 0.3111 | 0.0733 | 0.0575 | 99332 |
| TEST2 | 3 | Patch/Holes | 0.0209 | 0.0247 | 0.2384 | 0.8558 | 0.0508 | 0.0394 | 7906 |
| TEST2 | 3 | GPMM | 0.0524 | 0.0626 | 0.3238 | 0.3013 | 0.0631 | 0.0495 | 99332 |
| TEST2 | 3 | GPMM (kernel) | 0.0524 | 0.0626 | 0.3247 | 0.3013 | 0.0632 | 0.0496 | 99332 |
| TEST2 | 4 | Global SSM | 0.0569 | 0.0678 | 0.3563 | 0.2882 | 0.0851 | 0.0674 | 99332 |
| TEST2 | 4 | Neighborhood SSM | 0.057 | 0.068 | 0.3567 | 0.2871 | 0.0854 | 0.0677 | 99332 |
| TEST2 | 4 | Blend | 0.0553 | 0.0664 | 0.3691 | 0.3005 | 0.074 | 0.0582 | 99332 |
| TEST2 | 4 | Patch/Holes | 0.023 | 0.0273 | 0.2588 | 0.8318 | 0.0527 | 0.0409 | 9074 |
| TEST2 | 4 | GPMM | 0.0541 | 0.0648 | 0.3352 | 0.2915 | 0.0639 | 0.0502 | 99332 |
| TEST2 | 4 | GPMM (kernel) | 0.0541 | 0.0648 | 0.3357 | 0.2915 | 0.0639 | 0.0502 | 99332 |
| TEST2 | 5 | Global SSM | 0.0587 | 0.0698 | 0.3693 | 0.2799 | 0.0838 | 0.0665 | 99332 |
| TEST2 | 5 | Neighborhood SSM | 0.0587 | 0.0698 | 0.3695 | 0.2799 | 0.0839 | 0.0666 | 99332 |
| TEST2 | 5 | Blend | 0.0577 | 0.0691 | 0.3814 | 0.2892 | 0.0748 | 0.0589 | 99332 |
| TEST2 | 5 | Patch/Holes | 0.025 | 0.0298 | 0.2793 | 0.8081 | 0.0546 | 0.0424 | 10307 |
| TEST2 | 5 | GPMM | 0.0561 | 0.0672 | 0.3477 | 0.2811 | 0.0648 | 0.0509 | 99332 |
| TEST2 | 5 | GPMM (kernel) | 0.0561 | 0.0672 | 0.3488 | 0.2811 | 0.0648 | 0.0509 | 99332 |
| TEST2 | 6 | Global SSM | 0.0673 | 0.0817 | 0.4477 | 0.2504 | 0.0923 | 0.0716 | 99332 |
| TEST2 | 6 | Neighborhood SSM | 0.0729 | 0.0908 | 0.4735 | 0.2247 | 0.1006 | 0.0776 | 99332 |
| TEST2 | 6 | Blend | 0.0671 | 0.0821 | 0.4776 | 0.2554 | 0.0813 | 0.0642 | 99332 |
| TEST2 | 6 | Patch/Holes | 0.036 | 0.0439 | 0.3479 | 0.7192 | 0.0651 | 0.0503 | 16043 |
| TEST2 | 6 | GPMM | 0.0632 | 0.0763 | 0.3966 | 0.2634 | 0.0719 | 0.0563 | 99332 |
| TEST2 | 6 | GPMM (kernel) | 0.0632 | 0.0763 | 0.3972 | 0.2634 | 0.0719 | 0.0563 | 99332 |
| TEST2 | 7 | Global SSM | 0.0672 | 0.0994 | 0.6653 | 0.2919 | 0.0746 | 0.0559 | 99332 |
| TEST2 | 7 | Neighborhood SSM | 0.0729 | 0.1101 | 0.6738 | 0.2734 | 0.0907 | 0.0672 | 99332 |
| TEST2 | 7 | Blend | 0.1237 | 0.1791 | 0.798 | 0.1837 | 0.1492 | 0.1155 | 99332 |
| TEST2 | 7 | Patch/Holes | 0.0969 | 0.1372 | 0.7013 | 0.2159 | 0.1932 | 0.148 | 25890 |
| TEST2 | 7 | GPMM | 0.122 | 0.1745 | 0.7521 | 0.1876 | 0.152 | 0.1174 | 99332 |
| TEST2 | 7 | GPMM (kernel) | 0.1222 | 0.1746 | 0.752 | 0.1874 | 0.152 | 0.1175 | 99332 |

---

## Dataset B: v5 Dataset (8 Real Specimens x 8 Molnar Wear Levels)

Sixty four cases total. Eight real specimens (N1063, N332, N4, N459, N705, N726, N728, N891), each with eight progressive Molnar wear levels, each compared against its own corresponded original tooth. Ground truth is exact and index paired for Global SSM, Neighborhood SSM, GPMM, and Blend, so no ICP alignment is needed for those four. Patch/Holes is scored via nearest neighbor search since it appends new points. See the known limitation note above regarding mask comparability on this dataset.

### Mean Across All 64 Cases

| Method | Full Metric (mm) | Worn Region RMSE (mm) | Worn Region MAE (mm) | Mean Worn Points out of 10000 |
|---|---:|---:|---:|---:|
| Global SSM | 0.02633 | 0.02716 | 0.02462 | 8508 |
| Neighborhood SSM | 0.02626 | 0.02712 | 0.02460 | 8508 |
| GPMM | 0.02688 | 0.02767 | 0.02488 | 8508 |
| Blend | 0.02710 | 0.02800 | 0.02526 | 8508 |
| Patch/Holes | 0.01251 | 0.01663 | 0.01385 | 5462 |

The full metric column is full RMSE for Global SSM, Neighborhood SSM, GPMM, and Blend, computed exactly since those methods are index paired against the original. For Patch/Holes it is full Chamfer, since that method's point cloud is not index paired.

Neighborhood SSM and Global SSM are essentially tied for best among the four index paired methods, with GPMM close behind and Blend slightly behind that. Patch/Holes shows the lowest numbers of all five, but read this together with the known limitation note above: its worn region is self selected and roughly forty five percent smaller on average than the shared mask used by the other four, so part of its advantage here comes from evaluating a smaller, easier region rather than a strictly fair comparison.

### Full Detail, Per Specimen, All 8 Levels x All 5 Methods

For each specimen below, the first table gives the full metric per level per method (full RMSE for the four index paired methods, full Chamfer for Patch/Holes). The second table gives the worn region RMSE per level per method.

#### Specimen N1063

Full metric (mm): full RMSE for Global/Neighborhood SSM, GPMM, Blend; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01625 | 0.01576 | 0.01711 | 0.01662 | 0.01011 |
| 1 | 0.00782 | 0.00774 | 0.00862 | 0.00806 | 0.01011 |
| 2 | 0.0169 | 0.01707 | 0.01739 | 0.01709 | 0.01142 |
| 3 | 0.02531 | 0.02518 | 0.02619 | 0.02557 | 0.01225 |
| 4 | 0.03392 | 0.03367 | 0.0354 | 0.03449 | 0.01331 |
| 5 | 0.04063 | 0.04137 | 0.04446 | 0.04351 | 0.01466 |
| 6 | 0.03995 | 0.04022 | 0.0422 | 0.04114 | 0.01477 |
| 7 | 0.04284 | 0.04272 | 0.0456 | 0.04562 | 0.01313 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01748 | 0.01694 | 0.01841 | 0.01787 | 0.01098 |
| 1 | 0.01068 | 0.01058 | 0.01158 | 0.01123 | 0.01056 |
| 2 | 0.01818 | 0.01835 | 0.01871 | 0.01838 | 0.013 |
| 3 | 0.02556 | 0.02543 | 0.02645 | 0.02583 | 0.01563 |
| 4 | 0.03411 | 0.03385 | 0.0356 | 0.03468 | 0.01823 |
| 5 | 0.04089 | 0.04164 | 0.04474 | 0.04379 | 0.02228 |
| 6 | 0.04072 | 0.041 | 0.04297 | 0.04193 | 0.02011 |
| 7 | 0.04293 | 0.04281 | 0.0457 | 0.04572 | 0.017 |

#### Specimen N332

Full metric (mm): full RMSE for Global/Neighborhood SSM, GPMM, Blend; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0102 | 0.01041 | 0.01142 | 0.01113 | 0.01182 |
| 1 | 0.02279 | 0.02293 | 0.02424 | 0.02358 | 0.01302 |
| 2 | 0.02666 | 0.02688 | 0.02879 | 0.02859 | 0.01332 |
| 3 | 0.01562 | 0.01522 | 0.01728 | 0.01695 | 0.01194 |
| 4 | 0.0208 | 0.02059 | 0.0229 | 0.02248 | 0.01271 |
| 5 | 0.02249 | 0.02274 | 0.02498 | 0.0248 | 0.01262 |
| 6 | 0.02958 | 0.02916 | 0.03141 | 0.03011 | 0.01342 |
| 7 | 0.02964 | 0.0296 | 0.03241 | 0.03114 | 0.0128 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01108 | 0.01127 | 0.0125 | 0.01217 | 0.01058 |
| 1 | 0.02293 | 0.02308 | 0.0244 | 0.02373 | 0.01705 |
| 2 | 0.02685 | 0.02706 | 0.02899 | 0.02879 | 0.01658 |
| 3 | 0.01623 | 0.01581 | 0.01796 | 0.01761 | 0.01389 |
| 4 | 0.0217 | 0.02148 | 0.02391 | 0.02347 | 0.01662 |
| 5 | 0.02266 | 0.02291 | 0.02517 | 0.02499 | 0.01529 |
| 6 | 0.02977 | 0.02935 | 0.03162 | 0.03031 | 0.01791 |
| 7 | 0.03043 | 0.03039 | 0.03328 | 0.03197 | 0.01838 |

#### Specimen N4

Full metric (mm): full RMSE for Global/Neighborhood SSM, GPMM, Blend; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01204 | 0.01054 | 0.00999 | 0.01041 | 0.01103 |
| 1 | 0.01397 | 0.01295 | 0.01268 | 0.01288 | 0.01129 |
| 2 | 0.0089 | 0.00742 | 0.00606 | 0.00704 | 0.0108 |
| 3 | 0.01373 | 0.01217 | 0.01148 | 0.01198 | 0.01105 |
| 4 | 0.02006 | 0.01968 | 0.019 | 0.01953 | 0.01195 |
| 5 | 0.01961 | 0.01902 | 0.01885 | 0.01909 | 0.01156 |
| 6 | 0.0503 | 0.05034 | 0.03275 | 0.05138 | 0.01834 |
| 7 | 0.0494 | 0.04873 | 0.03146 | 0.04946 | 0.01846 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01386 | 0.01201 | 0.01142 | 0.01189 | 0.01284 |
| 1 | 0.01498 | 0.01387 | 0.01357 | 0.0138 | 0.01141 |
| 2 | 0.00871 | 0.00882 | 0.00875 | 0.00872 | 0.0121 |
| 3 | 0.01469 | 0.01298 | 0.01238 | 0.01281 | 0.01171 |
| 4 | 0.02055 | 0.02022 | 0.01958 | 0.0201 | 0.01574 |
| 5 | 0.01997 | 0.01985 | 0.01971 | 0.01995 | 0.01488 |
| 6 | 0.0503 | 0.05034 | 0.03275 | 0.05138 | 0.02846 |
| 7 | 0.04953 | 0.04886 | 0.03153 | 0.04959 | 0.02823 |

#### Specimen N459

Full metric (mm): full RMSE for Global/Neighborhood SSM, GPMM, Blend; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.00874 | 0.00824 | 0.00932 | 0.00918 | 0.01036 |
| 1 | 0.0125 | 0.01233 | 0.01262 | 0.01255 | 0.01115 |
| 2 | 0.02374 | 0.02353 | 0.02499 | 0.02498 | 0.012 |
| 3 | 0.0229 | 0.02274 | 0.02417 | 0.02414 | 0.01239 |
| 4 | 0.04884 | 0.04958 | 0.05122 | 0.05099 | 0.01872 |
| 5 | 0.05325 | 0.05356 | 0.05534 | 0.05487 | 0.01876 |
| 6 | 0.05548 | 0.05589 | 0.05845 | 0.0582 | 0.01939 |
| 7 | 0.05346 | 0.05406 | 0.05663 | 0.05598 | 0.01845 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01077 | 0.01009 | 0.01156 | 0.01136 | 0.0114 |
| 1 | 0.01312 | 0.01299 | 0.0133 | 0.01323 | 0.01577 |
| 2 | 0.02403 | 0.02382 | 0.0253 | 0.02529 | 0.0164 |
| 3 | 0.0235 | 0.02334 | 0.02481 | 0.02478 | 0.0171 |
| 4 | 0.04892 | 0.04967 | 0.05131 | 0.05109 | 0.02782 |
| 5 | 0.05339 | 0.0537 | 0.05548 | 0.05501 | 0.02972 |
| 6 | 0.05567 | 0.05607 | 0.05864 | 0.0584 | 0.02709 |
| 7 | 0.05371 | 0.05431 | 0.05688 | 0.05623 | 0.02581 |

#### Specimen N705

Full metric (mm): full RMSE for Global/Neighborhood SSM, GPMM, Blend; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0092 | 0.00823 | 0.00698 | 0.00698 | 0.00806 |
| 1 | 0.00707 | 0.00635 | 0.00568 | 0.00532 | 0.00828 |
| 2 | 0.01379 | 0.01382 | 0.01388 | 0.01383 | 0.00835 |
| 3 | 0.01075 | 0.0102 | 0.01088 | 0.01065 | 0.00857 |
| 4 | 0.0435 | 0.04395 | 0.04482 | 0.04467 | 0.0196 |
| 5 | 0.02023 | 0.02085 | 0.02135 | 0.02122 | 0.00939 |
| 6 | 0.02763 | 0.03042 | 0.0329 | 0.0338 | 0.01247 |
| 7 | 0.03165 | 0.03328 | 0.0366 | 0.03671 | 0.01314 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01066 | 0.01207 | 0.01013 | 0.01011 | 0.00842 |
| 1 | 0.01071 | 0.00888 | 0.00854 | 0.00851 | 0.01349 |
| 2 | 0.01506 | 0.01506 | 0.01525 | 0.01519 | 0.01064 |
| 3 | 0.01287 | 0.01243 | 0.01311 | 0.01303 | 0.01487 |
| 4 | 0.04352 | 0.04397 | 0.04485 | 0.04469 | 0.02483 |
| 5 | 0.02103 | 0.02175 | 0.02229 | 0.02213 | 0.01512 |
| 6 | 0.02796 | 0.0308 | 0.03332 | 0.03423 | 0.02213 |
| 7 | 0.03176 | 0.03341 | 0.03674 | 0.03685 | 0.02177 |

#### Specimen N726

Full metric (mm): full RMSE for Global/Neighborhood SSM, GPMM, Blend; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0104 | 0.00941 | 0.01053 | 0.00903 | 0.01074 |
| 1 | 0.00872 | 0.00827 | 0.00952 | 0.00789 | 0.01032 |
| 2 | 0.0288 | 0.02786 | 0.02951 | 0.02851 | 0.01389 |
| 3 | 0.03834 | 0.03782 | 0.03899 | 0.03869 | 0.01539 |
| 4 | 0.03094 | 0.03038 | 0.03219 | 0.03114 | 0.01318 |
| 5 | 0.04424 | 0.04344 | 0.04591 | 0.04432 | 0.01523 |
| 6 | 0.0435 | 0.04349 | 0.04554 | 0.04527 | 0.01601 |
| 7 | 0.05441 | 0.05436 | 0.05679 | 0.05516 | 0.01929 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01134 | 0.01065 | 0.0112 | 0.01025 | 0.01021 |
| 1 | 0.01154 | 0.011 | 0.01114 | 0.01063 | 0.01075 |
| 2 | 0.02923 | 0.02837 | 0.03005 | 0.02904 | 0.01989 |
| 3 | 0.0388 | 0.03828 | 0.03946 | 0.03916 | 0.01998 |
| 4 | 0.03169 | 0.03114 | 0.033 | 0.03192 | 0.02088 |
| 5 | 0.04433 | 0.04353 | 0.046 | 0.04441 | 0.02254 |
| 6 | 0.04353 | 0.04352 | 0.04557 | 0.04529 | 0.02335 |
| 7 | 0.05448 | 0.05443 | 0.05686 | 0.05523 | 0.02616 |

#### Specimen N728

Full metric (mm): full RMSE for Global/Neighborhood SSM, GPMM, Blend; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.00658 | 0.00607 | 0.00641 | 0.00613 | 0.00741 |
| 1 | 0.01079 | 0.0109 | 0.0115 | 0.01119 | 0.00757 |
| 2 | 0.01317 | 0.01433 | 0.01562 | 0.01537 | 0.00781 |
| 3 | 0.01802 | 0.01905 | 0.01991 | 0.0196 | 0.00823 |
| 4 | 0.02369 | 0.02514 | 0.01825 | 0.02583 | 0.00854 |
| 5 | 0.02739 | 0.02841 | 0.02967 | 0.02938 | 0.00801 |
| 6 | 0.03791 | 0.03904 | 0.04082 | 0.04024 | 0.01075 |
| 7 | 0.03977 | 0.0401 | 0.04272 | 0.04155 | 0.01139 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01003 | 0.00902 | 0.00953 | 0.00938 | 0.00688 |
| 1 | 0.01196 | 0.01225 | 0.01295 | 0.01259 | 0.00804 |
| 2 | 0.01395 | 0.0152 | 0.01659 | 0.01632 | 0.00995 |
| 3 | 0.0185 | 0.01956 | 0.02045 | 0.02014 | 0.01042 |
| 4 | 0.02405 | 0.02551 | 0.01833 | 0.02621 | 0.01166 |
| 5 | 0.02766 | 0.02869 | 0.02996 | 0.02966 | 0.0114 |
| 6 | 0.0382 | 0.03934 | 0.04113 | 0.04055 | 0.01485 |
| 7 | 0.03983 | 0.04016 | 0.04278 | 0.04161 | 0.01904 |

#### Specimen N891

Full metric (mm): full RMSE for Global/Neighborhood SSM, GPMM, Blend; full Chamfer for Patch/Holes.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01 | 0.00828 | 0.00792 | 0.00792 | 0.00774 |
| 1 | 0.0097 | 0.00842 | 0.00799 | 0.0081 | 0.00755 |
| 2 | 0.01092 | 0.01046 | 0.01657 | 0.01065 | 0.00747 |
| 3 | 0.01425 | 0.01353 | 0.01432 | 0.01413 | 0.00754 |
| 4 | 0.05055 | 0.05067 | 0.05213 | 0.05189 | 0.02101 |
| 5 | 0.01447 | 0.01441 | 0.01705 | 0.01475 | 0.00783 |
| 6 | 0.05238 | 0.05291 | 0.05463 | 0.05436 | 0.02139 |
| 7 | 0.05416 | 0.05447 | 0.05728 | 0.05643 | 0.02112 |

Worn region RMSE (mm), restricted to the points that were actually worn away.

| Level | Global SSM | Neighborhood SSM | GPMM | Blend | Patch/Holes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.01276 | 0.01044 | 0.00986 | 0.00994 | 0.01066 |
| 1 | 0.01292 | 0.01071 | 0.01035 | 0.01031 | 0.00841 |
| 2 | 0.01348 | 0.01353 | 0.01515 | 0.01384 | 0.00844 |
| 3 | 0.01567 | 0.01512 | 0.01603 | 0.0158 | 0.0098 |
| 4 | 0.05061 | 0.05073 | 0.05219 | 0.05194 | 0.02575 |
| 5 | 0.0159 | 0.01592 | 0.01698 | 0.0163 | 0.01055 |
| 6 | 0.05238 | 0.05291 | 0.05463 | 0.05436 | 0.02691 |
| 7 | 0.05422 | 0.05453 | 0.05734 | 0.05649 | 0.02583 |

---

## File Pointers

| File | Contents |
|---|---|
| `ssm_pipeline/output/eval_old_dataset.csv` | Raw 96 row table behind the Dataset A tables above, one row per set, level, and method, with every column including full RMSE, full Hausdorff, and full Coverage at 2x spacing that are summarized here. |
| `ssm_pipeline/output/eval_v5_dataset.csv` | Raw 320 row table behind the Dataset B tables above. |
| `ssm_pipeline/evaluate_all_methods.py` | The script that produced both CSVs. Run with `--dataset old`, `--dataset v5`, or `--dataset both`. |
| `ssm_pipeline/output/archive/patch_method_chamfer.csv` | An earlier, smaller evaluation pass kept for history. It used a different, less rigorous worn region mask and does not include GPMM. Superseded by the tables in this file. |

## Reproducing These Numbers

```bash
cd ssm_pipeline
conda activate teeth
python3 evaluate_all_methods.py --dataset both
```

This reads reconstructions that already exist under `output/`, so the correspondence and reconstruction pipelines (Stage 1 and Stage 2) must have already been run for both datasets before this command will produce results. See the Quick Start section of the README for those commands.
