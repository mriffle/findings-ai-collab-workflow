# Shrinkage-LDA preview — magnet

- data: n = 40, p = 2343 (constant features dropped), scale = log2
- binary outcome: `Disease` = ADD (10) vs rest (30)

## Seed 0 — same 5x5 outer folds as the shipped templates

- elastic net (first pass): mean AUC 0.870 ± 0.180, wall 474s
- linear SVM (first pass):  mean AUC 0.963 ± 0.067, wall 2s
- shrinkage LDA (dual):     mean AUC 0.950 ± 0.108, 1 ms per fold fit, 25 folds 0.0s

| paired vs | LDA mean | other mean | mean diff | W/L/T | corrected t | p (corrected t) | p Wilcoxon (indicative) |
|---|---|---|---|---|---|---|---|
| elastic net | 0.950 | 0.870 | +0.080 | 10/0/15 | 0.97 | 0.341 | 0.005 |
| linear SVM | 0.950 | 0.963 | -0.013 | 1/3/21 | -0.43 | 0.670 | 0.197 |

## Seed noise — LDA and SVM at extra seeds (elastic net seed 0 only)

| seed | LDA mean AUC | SVM mean AUC | LDA−SVM | W/L/T | p (corrected t) |
|---|---|---|---|---|---|
| 0 | 0.950 | 0.963 | -0.013 | 1/3/21 | 0.670 |
| 1 | 0.970 | 0.970 | +0.000 | 3/4/18 | 1.000 |
| 2 | 0.980 | 0.960 | +0.020 | 7/1/17 | 0.544 |

## Label-shuffle null (LDA)

- observed mean AUC over 3x5 folds: 0.933; null 0.493 ± 0.122 (max 0.878); empirical p = 0.0010; 1000 permutations in 32s

## Stability read and overlap (top-20 by |weight|)

- all-data fit at p = 2343: 1 ms; per-class Ledoit-Wolf shrinkage = [0.25, 0.503]
- LDA top-20 mean fold-membership frequency: 0.45 (≥0.8 in 2 of 20)
- overlap with elastic-net top-20 (selected): 8; with SVM top-20: 11; EN∩SVM: 9

| feature | weight | membership |
|---|---|---|
| sp|P29350|PTN6_HUMAN | -6.641 | 1.00 |
| sp|O60763|USO1_HUMAN | -6.571 | 0.92 |
| sp|Q16822|PCKGM_HUMAN | -6.120 | 0.72 |
| sp|Q6ZU65|UBN2_HUMAN | +5.826 | 0.72 |
| sp|Q9P253|VPS18_HUMAN | +5.642 | 0.68 |
| sp|P62333|PRS10_HUMAN | -5.282 | 0.56 |
| sp|Q8N543|OGFD1_HUMAN | +5.240 | 0.52 |
| sp|O43242|PSMD3_HUMAN | -4.982 | 0.56 |
| sp|A8MWD9|RUXGL_HUMAN / sp|P62308|RUXG_HUMAN | -4.757 | 0.40 |
| sp|P19404|NDUV2_HUMAN | -4.731 | 0.36 |
| sp|Q68D91|MBLC2_HUMAN | -4.697 | 0.32 |
| sp|Q53T59|H1BP3_HUMAN | +4.684 | 0.16 |
| sp|Q9BX93|PG12B_HUMAN | +4.640 | 0.28 |
| sp|P43686|PRS6B_HUMAN | -4.579 | 0.32 |
| sp|P37235|HPCL1_HUMAN | -4.566 | 0.20 |
| sp|Q9H9S4|CB39L_HUMAN | -4.534 | 0.24 |
| sp|Q13618|CUL3_HUMAN | +4.488 | 0.44 |
| sp|Q9Y371|SHLB1_HUMAN | -4.457 | 0.16 |
| sp|Q9HC98|NEK6_HUMAN / sp|Q8TDX7|NEK7_HUMAN | -4.449 | 0.12 |
| sp|P42224|STAT1_HUMAN | -4.443 | 0.24 |

## Multiclass-native — `Condition` {'ADD': 10, 'HCN': 10, 'PDCN': 10, 'PDD': 10}

- 5x5 stratified CV: macro OvR AUC 0.957 ± 0.027; balanced accuracy 0.790 ± 0.104 (chance 0.250)

## Timing

- synthetic 100 x 20,000 fit: 0.04s (dual form; sklearn would form a 20,000² covariance)
