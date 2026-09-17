# Shrinkage-LDA preview — 5xfad

- data: n = 52, p = 8828 (constant features dropped), scale = log2
- binary outcome: `Disease` = 5xFAD (31) vs rest (21)

## Seed 0 — same 5x5 outer folds as the shipped templates

- elastic net (first pass): mean AUC 0.905 ± 0.139, wall 90s
- linear SVM (first pass):  mean AUC 0.796 ± 0.174, wall 7s
- shrinkage LDA (dual):     mean AUC 0.848 ± 0.140, 4 ms per fold fit, 25 folds 0.1s

| paired vs | LDA mean | other mean | mean diff | W/L/T | corrected t | p (corrected t) | p Wilcoxon (indicative) |
|---|---|---|---|---|---|---|---|
| elastic net | 0.848 | 0.905 | -0.057 | 5/13/7 | -0.81 | 0.426 | 0.036 |
| linear SVM | 0.848 | 0.796 | +0.052 | 15/4/6 | 0.97 | 0.344 | 0.011 |

## Seed noise — LDA and SVM at extra seeds (elastic net seed 0 only)

| seed | LDA mean AUC | SVM mean AUC | LDA−SVM | W/L/T | p (corrected t) |
|---|---|---|---|---|---|
| 0 | 0.848 | 0.796 | +0.052 | 15/4/6 | 0.344 |
| 1 | 0.869 | 0.807 | +0.062 | 12/3/10 | 0.247 |
| 2 | 0.829 | 0.780 | +0.049 | 15/1/9 | 0.098 |

## Label-shuffle null (LDA)

- observed mean AUC over 3x5 folds: 0.841; null 0.502 ± 0.102 (max 0.810); empirical p = 0.0010; 1000 permutations in 90s

## Stability read and overlap (top-20 by |weight|)

- all-data fit at p = 8828: 7 ms; per-class Ledoit-Wolf shrinkage = [0.472, 0.243]
- LDA top-20 mean fold-membership frequency: 0.83 (≥0.8 in 13 of 20)
- overlap with elastic-net top-20 (selected): 17; with SVM top-20: 14; EN∩SVM: 12

| feature | weight | membership |
|---|---|---|
| sp|P12023|A4_MOUSE | +8.277 | 1.00 |
| sp|P05067|5xFADA4_HUMAN | +8.243 | 1.00 |
| sp|P12025|MK_MOUSE | +7.413 | 1.00 |
| sp|P08226|APOE_MOUSE | +7.266 | 1.00 |
| sp|Q62288|TICN1_MOUSE | +6.997 | 1.00 |
| sp|Q9ER58|TICN2_MOUSE | +6.709 | 1.00 |
| sp|Q06890|CLUS_MOUSE | +6.265 | 1.00 |
| sp|O89098|CYTF_MOUSE | +5.387 | 0.88 |
| sp|Q61790|LAG3_MOUSE | -5.278 | 0.76 |
| sp|Q9QZF2|GPC1_MOUSE | +5.060 | 1.00 |
| sp|P05555|ITAM_MOUSE | +4.878 | 0.96 |
| sp|Q8R066|C1QT4_MOUSE | +4.629 | 1.00 |
| sp|Q8VCC9|SPON1_MOUSE | +4.107 | 0.92 |
| sp|P29788|VTNC_MOUSE | +4.082 | 0.88 |
| sp|O35709|ENC1_MOUSE | +4.000 | 0.36 |
| sp|P98086|C1QA_MOUSE | +3.911 | 0.72 |
| sp|P14106|C1QB_MOUSE | +3.871 | 0.56 |
| sp|Q02105|C1QC_MOUSE | +3.769 | 0.56 |
| sp|Q9QYM9|TEFF2_MOUSE | +3.655 | 0.52 |
| sp|P63089|PTN_MOUSE | +3.586 | 0.40 |

## Multiclass-native — `Genotype` {'5xFAD': 31, 'C57BL/6j': 14, 'WT': 7}

- 5x5 stratified CV: macro OvR AUC 0.725 ± 0.130; balanced accuracy 0.492 ± 0.107 (chance 0.333)

## Timing

- synthetic 100 x 20,000 fit: 0.04s (dual form; sklearn would form a 20,000² covariance)
