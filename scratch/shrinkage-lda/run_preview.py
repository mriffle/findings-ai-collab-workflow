"""Shrinkage-LDA preview on the two local classification datasets (5xFAD, MagNet-EV-ADD).

Scratch — imports the shipped templates read-only, changes nothing under lib/.

    PYTHONPATH=lib:scratch/shrinkage-lda ./.venv/bin/python scratch/shrinkage-lda/run_preview.py \
        --dataset 5xfad|magnet [--n-permutations 1000] [--seeds 0 1 2]

What it tests (the plan from the session):
  1. Same outer folds as ``classify`` (elastic net) and ``classify_svm`` at seed 0 —
     asserted on fold identity — so the comparison is **paired per fold**.
  2. LDA fit in the dual form (no p x p covariance), in-fold StandardScaler, no inner CV
     (Ledoit-Wolf shrinkage is analytic — nothing to tune).
  3. Paired statistics vs each template: per-fold differences, win/loss/tie, the
     **corrected resampled t-test** (Nadeau & Bengio 2003) and a Wilcoxon p (indicative).
  4. Seed noise: LDA and SVM re-run at extra seeds (both cheap); the elastic net at seed 0
     only (its nested grid is the expensive part).
  5. Label-shuffle null for LDA (row shuffle; 1000 permutations x 3 CV repeats x 5 folds).
  6. Stability read for the dense weights: top-20 membership frequency across the 25 fold
     models; overlap of the all-data top-20 with the elastic net's and the SVM's.
  7. Multiclass-native run on the full categorical outcome (Genotype: 3 levels;
     Condition: 4 levels) — macro one-vs-rest AUC + balanced accuracy, 5x5 stratified CV.
  8. Timing: one all-data fit at full width, and a synthetic 100 x 20,000 fit.

Outputs: scratch/shrinkage-lda/out/<dataset>/{summary.md, paired_folds.csv, top20.csv}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from analysis import classification as clf
from analysis import classification_svm as svm
from common import data_loading as dl
from common import normalize as norm
from dual_lda import DualShrinkageLDA
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTDATA = ROOT / "testdata"


# --------------------------------------------------------------------------- data
def load_5xfad() -> tuple[dl.Dataset, str, str, str]:
    base = TESTDATA / "5xFAD"
    ds = dl.load_wide_data(
        base / "data" / "proteins_wide_unnormalized.tsv",
        base / "metadata" / "Replicates_5xFAD.csv",
        join_key="Replicate",
        strip_suffix=".raw",
        collapse_replicates=dl.ReplicateCollapse("Sample ID", "Technical Replicate"),
        order_by="RunOrder",
        numeric_columns=("RunOrder",),
    )
    logged = norm.log2_transform(norm.normalize(ds, "median"))
    mask = logged.metadata["Genotype"].to_numpy() != "na"
    meta = logged.metadata.loc[mask].reset_index(drop=True).copy()
    meta["Disease"] = np.where(meta["Genotype"].to_numpy() == "5xFAD", "5xFAD", "nonAD")
    exp = dataclasses.replace(logged, abundances=logged.abundances[mask, :], metadata=meta)
    return exp, "Disease", "5xFAD", "Genotype"


def load_magnet() -> tuple[dl.Dataset, str, str, str]:
    base = TESTDATA / "MagNet-EV-ADD"
    meta_exp = base / "metadata.experimental.csv"
    if not meta_exp.exists():
        meta = pd.read_csv(base / "metadata.txt", dtype=str)
        ctrl = {"Plasma Reference", "TPAD plasma pooled reference"}
        meta.loc[~meta["Condition"].isin(ctrl)].to_csv(meta_exp, index=False)
    ds = dl.load_wide_data(
        base / "proteins.txt",
        meta_exp,
        join_key="SampleID",
        id_columns=("Unnamed: 0",),
        data_sep=",",
        metadata_sep=",",
        numeric_columns=("Age", "Run Order"),
        expected_n_samples=40,
        expected_n_features=2343,
        scale="log2",
    )
    meta = ds.metadata.copy()
    meta["Disease"] = np.where(meta["Condition"].to_numpy() == "ADD", "ADD", "rest")
    return dataclasses.replace(ds, metadata=meta), "Disease", "ADD", "Condition"


# --------------------------------------------------------------------------- helpers
def drop_constant(x: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = x.std(axis=0) > 0
    return x[:, keep], names[keep]


def lda_fold_scores(
    x: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[list[float], list[np.ndarray], list[float]]:
    aucs, coefs, times = [], [], []
    for train, test in folds:
        sc = StandardScaler().fit(x[train])
        t0 = time.perf_counter()
        m = DualShrinkageLDA().fit(sc.transform(x[train]), y[train])
        times.append(time.perf_counter() - t0)
        aucs.append(float(roc_auc_score(y[test], m.decision_function(sc.transform(x[test])))))
        coefs.append(m.coef_.ravel())
    return aucs, coefs, times


def corrected_resampled_t(diff: np.ndarray, n_train: float, n_test: float) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test over J dependent CV folds."""
    j = len(diff)
    var = float(np.var(diff, ddof=1))
    if var == 0:
        return float("nan"), float("nan")
    se = np.sqrt(var * (1.0 / j + n_test / n_train))
    t = float(diff.mean() / se)
    return t, float(2 * stats.t.sf(abs(t), df=j - 1))


def paired(name: str, lda: list[float], other: list[float], n_train: float, n_test: float) -> dict:
    d = np.asarray(lda) - np.asarray(other)
    t, p_t = corrected_resampled_t(d, n_train, n_test)
    try:
        p_w = float(stats.wilcoxon(lda, other).pvalue)
    except ValueError:
        p_w = float("nan")
    return {
        "vs": name,
        "lda_mean": float(np.mean(lda)),
        "other_mean": float(np.mean(other)),
        "mean_diff": float(d.mean()),
        "win": int((d > 1e-12).sum()),
        "loss": int((d < -1e-12).sum()),
        "tie": int((np.abs(d) <= 1e-12).sum()),
        "corrected_t": t,
        "p_corrected_t": p_t,
        "p_wilcoxon_indicative": p_w,
    }


def fold_identity(res: object) -> list[tuple[int, int, tuple[int, ...]]]:
    return [(f.repeat, f.fold, tuple(int(i) for i in f.test_indices)) for f in res.fold_predictions]  # type: ignore[attr-defined]


def folds_from(res: object, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for f in res.fold_predictions:  # type: ignore[attr-defined]
        test = np.asarray(f.test_indices, dtype=int)
        train = np.setdiff1d(np.arange(n), test)
        out.append((train, test))
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["5xfad", "magnet"], required=True)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    out = HERE / "out" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# Shrinkage-LDA preview — {args.dataset}", ""]

    ds, outcome, positive, multi_col = load_5xfad() if args.dataset == "5xfad" else load_magnet()
    x_all, names = drop_constant(ds.abundances, np.asarray(ds.feature_names))
    y_bin = (ds.metadata[outcome].to_numpy() == positive).astype(int)
    n, p = x_all.shape
    lines.append(f"- data: n = {n}, p = {p} (constant features dropped), scale = {ds.scale}")
    lines.append(f"- binary outcome: `{outcome}` = {positive} ({y_bin.sum()}) vs rest ({n - y_bin.sum()})")

    # ---- shipped templates at seed 0 (first pass, no null), same folds
    en_kw: dict = {}
    if args.dataset == "5xfad":  # reduced grid, as the SVM preview used, for speed
        en_kw = {"c_grid": [0.1, 1.0], "l1_ratios": [0.5, 0.9], "stability_repeats": 3,
                 "max_iter": 1500, "tol": 1e-3}
    t0 = time.time()
    en = clf.classify(ds, outcome, positive_class=positive, n_jobs=-1, random_state=0, **en_kw)
    t_en = time.time() - t0
    t0 = time.time()
    sv = svm.classify_svm(ds, outcome, positive_class=positive, n_jobs=-1, random_state=0)
    t_sv = time.time() - t0
    assert fold_identity(en) == fold_identity(sv), "outer splits differ between templates"
    folds0 = folds_from(sv, n)
    en_f = [float(roc_auc_score(f.y_true, f.y_prob)) for f in en.fold_predictions]
    sv_f = [float(roc_auc_score(f.y_true, f.y_score)) for f in sv.fold_predictions]
    lda_f, lda_coefs, lda_times = lda_fold_scores(x_all, y_bin, folds0)
    n_test = float(np.mean([len(t) for _, t in folds0]))
    n_train = n - n_test
    pd.DataFrame({
        "repeat": [f.repeat for f in sv.fold_predictions], "fold": [f.fold for f in sv.fold_predictions],
        "lda": lda_f, "elastic_net": en_f, "svm": sv_f,
    }).to_csv(out / "paired_folds.csv", index=False)

    lines += ["", "## Seed 0 — same 5x5 outer folds as the shipped templates", ""]
    lines.append(f"- elastic net (first pass): mean AUC {np.mean(en_f):.3f} ± {np.std(en_f):.3f}, wall {t_en:.0f}s")
    lines.append(f"- linear SVM (first pass):  mean AUC {np.mean(sv_f):.3f} ± {np.std(sv_f):.3f}, wall {t_sv:.0f}s")
    lines.append(f"- shrinkage LDA (dual):     mean AUC {np.mean(lda_f):.3f} ± {np.std(lda_f):.3f}, "
                 f"{np.mean(lda_times) * 1000:.0f} ms per fold fit, 25 folds {sum(lda_times):.1f}s")
    lines += ["", "| paired vs | LDA mean | other mean | mean diff | W/L/T | corrected t | p (corrected t) | p Wilcoxon (indicative) |",
              "|---|---|---|---|---|---|---|---|"]
    comp = [paired("elastic net", lda_f, en_f, n_train, n_test), paired("linear SVM", lda_f, sv_f, n_train, n_test)]
    for c in comp:
        lines.append(f"| {c['vs']} | {c['lda_mean']:.3f} | {c['other_mean']:.3f} | {c['mean_diff']:+.3f} | "
                     f"{c['win']}/{c['loss']}/{c['tie']} | {c['corrected_t']:.2f} | {c['p_corrected_t']:.3f} | {c['p_wilcoxon_indicative']:.3f} |")

    # ---- extra seeds: LDA + SVM (cheap), paired per seed
    lines += ["", "## Seed noise — LDA and SVM at extra seeds (elastic net seed 0 only)", "",
              "| seed | LDA mean AUC | SVM mean AUC | LDA−SVM | W/L/T | p (corrected t) |", "|---|---|---|---|---|---|"]
    for seed in args.seeds:
        if seed == 0:
            s_f, l_f = sv_f, lda_f
        else:
            s = svm.classify_svm(ds, outcome, positive_class=positive, n_jobs=-1, random_state=seed)
            s_f = [float(roc_auc_score(f.y_true, f.y_score)) for f in s.fold_predictions]
            l_f, _, _ = lda_fold_scores(x_all, y_bin, folds_from(s, n))
        c = paired("svm", l_f, s_f, n_train, n_test)
        lines.append(f"| {seed} | {np.mean(l_f):.3f} | {np.mean(s_f):.3f} | {c['mean_diff']:+.3f} | "
                     f"{c['win']}/{c['loss']}/{c['tie']} | {c['p_corrected_t']:.3f} |")

    # ---- label-shuffle null for LDA (row shuffle; 3 CV repeats x 5 folds per permutation)
    rng = np.random.default_rng(0)
    cv_null = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    t0 = time.time()
    null_folds = list(cv_null.split(x_all, y_bin))
    obs = float(np.mean(lda_fold_scores(x_all, y_bin, null_folds)[0]))
    null_stats = []
    for _ in range(args.n_permutations):
        yp = rng.permutation(y_bin)
        null_stats.append(float(np.mean(lda_fold_scores(x_all, yp, list(cv_null.split(x_all, yp)))[0])))
    null_arr = np.asarray(null_stats)
    p_null = (1 + int((null_arr >= obs).sum())) / (1 + len(null_arr))
    t_null = time.time() - t0
    lines += ["", "## Label-shuffle null (LDA)", "",
              f"- observed mean AUC over 3x5 folds: {obs:.3f}; null {null_arr.mean():.3f} ± {null_arr.std():.3f} "
              f"(max {null_arr.max():.3f}); empirical p = {p_null:.4f}; {args.n_permutations} permutations in {t_null:.0f}s"]

    # ---- stability read + overlap with the two templates
    k = 20
    fold_top = [set(np.argsort(-np.abs(c))[:k]) for c in lda_coefs]
    sc = StandardScaler().fit(x_all)
    t0 = time.perf_counter()
    m_all = DualShrinkageLDA().fit(sc.transform(x_all), y_bin)
    t_fit_all = time.perf_counter() - t0
    w = m_all.coef_.ravel()
    order = np.argsort(-np.abs(w))[:k]
    top = pd.DataFrame({
        "feature": names[order], "weight": w[order], "abs_weight": np.abs(w[order]),
        "top20_membership": [np.mean([i in s for s in fold_top]) for i in order],
    })
    top.to_csv(out / "top20.csv", index=False)
    en_top = set(en.coefficients.loc[en.coefficients["abs_coef"] > 0].head(k)["feature"])
    sv_top = set(sv.coefficients.head(k)["feature"])
    lda_top = set(top["feature"])
    lines += ["", "## Stability read and overlap (top-20 by |weight|)", "",
              f"- all-data fit at p = {p}: {t_fit_all * 1000:.0f} ms; per-class Ledoit-Wolf shrinkage = "
              f"{np.round(m_all.shrinkage_, 3).tolist()}",
              f"- LDA top-20 mean fold-membership frequency: {top['top20_membership'].mean():.2f} "
              f"(≥0.8 in {(top['top20_membership'] >= 0.8).sum()} of 20)",
              f"- overlap with elastic-net top-20 (selected): {len(lda_top & en_top)}; with SVM top-20: {len(lda_top & sv_top)}; "
              f"EN∩SVM: {len(en_top & sv_top)}",
              "", "| feature | weight | membership |", "|---|---|---|"]
    for r in top.itertuples():
        lines.append(f"| {r.feature} | {r.weight:+.3f} | {r.top20_membership:.2f} |")

    # ---- multiclass-native
    y_multi = ds.metadata[multi_col].to_numpy()
    classes, y_idx = np.unique(y_multi, return_inverse=True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    aucs, baccs = [], []
    for train, test in cv.split(x_all, y_idx):
        s = StandardScaler().fit(x_all[train])
        m = DualShrinkageLDA().fit(s.transform(x_all[train]), y_idx[train])
        proba = m.predict_proba(s.transform(x_all[test]))
        present = np.unique(y_idx[test])
        aucs.append(float(roc_auc_score(y_idx[test], proba[:, present] if len(present) < len(classes) else proba,
                                        multi_class="ovr", average="macro", labels=present)))
        baccs.append(float(balanced_accuracy_score(y_idx[test], m.predict(s.transform(x_all[test])))))
    counts = dict(zip(classes.tolist(), np.bincount(y_idx).tolist()))
    lines += ["", f"## Multiclass-native — `{multi_col}` {counts}", "",
              f"- 5x5 stratified CV: macro OvR AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}; "
              f"balanced accuracy {np.mean(baccs):.3f} ± {np.std(baccs):.3f} (chance {1 / len(classes):.3f})"]

    # ---- timing at the user's width
    xs = np.random.default_rng(1).standard_normal((100, 20_000))
    ys = np.arange(100) % 2
    t0 = time.perf_counter()
    DualShrinkageLDA().fit(xs, ys)
    t20k = time.perf_counter() - t0
    lines += ["", "## Timing", "", f"- synthetic 100 x 20,000 fit: {t20k:.2f}s (dual form; sklearn would form a 20,000² covariance)"]

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "paired_stats.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
