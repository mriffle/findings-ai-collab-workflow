"""Dual-form Ledoit-Wolf shrinkage LDA — a prototype, NOT a shipped template.

Reproduces scikit-learn's ``LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")``
*exactly* (same per-class Ledoit-Wolf shrinkage, same pooled covariance, same
coefficients / intercepts / decision scores) without ever forming the p x p covariance.

Why: sklearn materializes the shrunk pooled covariance (p x p) and solves it, which is
O(p^3) time and O(p^2) memory — measured at ~26 s per fit at p = 2,000 on the 5xFAD
matrix and unfinished after 10 min at p = 6,186; at p = 20,000 (the user's routine
width) that is 3.2 GB per fit and hours. But the shrunk covariance is

    S = diag(a) + U^T U,        U : (n x p), rank <= n,

(sklearn's per-class Ledoit-Wolf shrinks toward the diagonal of the class variances, so
the "identity" part is a diagonal, which changes nothing), and with A = diag(a)

    S^{-1} v = A^{-1} v - A^{-1} U^T (I_n + U A^{-1} U^T)^{-1} U A^{-1} v   (Woodbury),

an n x n solve costing O(n^2 p). The Ledoit-Wolf shrinkage intensity is likewise a
function of the n x n Gram matrix only. Everything is exact; the test file pins it to
sklearn at machine precision on the real data.

sklearn's recipe (LinearDiscriminantAnalysis._solve_lsqr, shrinkage="auto"):
  * priors_k = n_k / n; means_k = class means
  * cov = sum_k priors_k * ledoit_wolf(X_k)      (per-class LW on the class-centered
    block, 1/n_k normalisation, target mu_k * I with mu_k = trace(S_k)/p)
  * coef = solve(cov, means^T)^T ; intercept_k = -0.5 * means_k . coef_k + log priors_k
  * binary: coef = coef_1 - coef_0, intercept = intercept_1 - intercept_0
  * decision_function = X @ coef^T + intercept; predict_proba = sigmoid / softmax.
"""

from __future__ import annotations

import numpy as np


def ledoit_wolf_shrinkage_gram(x_centered: np.ndarray) -> float:
    """Ledoit-Wolf shrinkage intensity from the Gram matrix (exact sklearn formula).

    sklearn's ``ledoit_wolf_shrinkage`` accumulates two scalars over p x p blocks:
    ``beta_ = sum(X2^T X2)`` = sum_s ||x_s||^4 and ``delta_ = ||X^T X||_F^2``. Both are
    Gram-matrix quantities: with G = X X^T (n x n), ``beta_ = sum_s G_ss^2`` and
    ``delta_ = ||G||_F^2``. The rest of the formula is scalar algebra.
    """
    n, p = x_centered.shape
    g = x_centered @ x_centered.T  # (n, n)
    trace_s = float(np.trace(g)) / n  # trace of the empirical covariance
    mu = trace_s / p
    beta_ = float(np.sum(np.diag(g) ** 2))
    delta_ = float(np.sum(g**2)) / n**2
    beta = 1.0 / (p * n) * (beta_ / n - delta_)
    delta = (delta_ - 2.0 * mu * trace_s + p * mu**2) / p
    beta = min(beta, delta)
    return 0.0 if beta == 0 else beta / delta


class DualShrinkageLDA:
    """LDA with per-class Ledoit-Wolf shrinkage, solved in the n-dimensional dual.

    Mirrors ``sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver="lsqr",
    shrinkage="auto")`` — same attributes ``classes_``, ``priors_``, ``means_``,
    ``coef_``, ``intercept_``, ``decision_function``, ``predict_proba``, ``predict`` —
    plus ``shrinkage_`` (per-class intensities) and ``c_`` (the identity weight of the
    pooled shrunk covariance), which are diagnostics the shipped template would report.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray
        self.priors_: np.ndarray
        self.means_: np.ndarray
        self.coef_: np.ndarray
        self.intercept_: np.ndarray
        self.shrinkage_: np.ndarray
        self.c_: float

    def fit(self, x: np.ndarray, y: np.ndarray) -> DualShrinkageLDA:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        n, p = x.shape
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        if k < 2:
            msg = "need at least two classes"
            raise ValueError(msg)
        counts = np.bincount(y_idx, minlength=k)
        if np.any(counts < 2):
            msg = "every class needs at least two samples for a covariance"
            raise ValueError(msg)
        self.priors_ = counts / n
        self.means_ = np.vstack([x[y_idx == j].mean(axis=0) for j in range(k)])

        # sklearn's _cov(X, "auto"): standardize the class block (biased std; a zero-std
        # feature keeps scale 1), run Ledoit-Wolf on the standardized block, rescale.
        # So cov_k = (1 - lam_k) S_k + lam_k * diag(scale_k^2)  — the target is the
        # diagonal of the class variances, and the pooled shrunk covariance is
        #     S = diag(a) + U^T U,   a = sum_k priors_k lam_k scale_k^2,
        # still identity-plus-low-rank in the Woodbury sense (diagonal instead of scalar).
        a = np.zeros(p)
        blocks: list[np.ndarray] = []
        lam = np.empty(k)
        for j in range(k):
            xc = x[y_idx == j] - self.means_[j]
            n_j = counts[j]
            scale = np.sqrt(np.sum(xc**2, axis=0) / n_j)
            scale[scale == 0.0] = 1.0
            lam[j] = ledoit_wolf_shrinkage_gram(xc / scale)
            a += self.priors_[j] * lam[j] * scale**2
            blocks.append(np.sqrt(self.priors_[j] * (1.0 - lam[j]) / n_j) * xc)
        if np.any(a == 0.0):
            msg = "a feature is constant within every class; drop constant features first"
            raise ValueError(msg)
        self.shrinkage_ = lam
        self.c_ = float(a.mean())
        u = np.vstack(blocks)  # (n, p)

        # coef_j = S^{-1} means_j via Woodbury with A = diag(a):
        #   S^{-1} m = A^{-1} m - A^{-1} U^T (I_n + U A^{-1} U^T)^{-1} U A^{-1} m
        m = self.means_.T  # (p, k)
        a_inv = 1.0 / a
        ua = u * a_inv  # (n, p) = U A^{-1}
        inner = np.eye(n) + ua @ u.T  # (n, n)
        am = m * a_inv[:, None]  # (p, k) = A^{-1} m
        coef = am - ua.T @ np.linalg.solve(inner, u @ am)  # (p, k)
        coef = coef.T  # (k, p)
        intercept = -0.5 * np.einsum("kp,kp->k", self.means_, coef) + np.log(self.priors_)
        if k == 2:
            coef = (coef[1] - coef[0])[None, :]
            intercept = np.array([intercept[1] - intercept[0]])
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        scores = np.asarray(x, dtype=np.float64) @ self.coef_.T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        d = self.decision_function(x)
        if d.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-d))
            return np.column_stack([1.0 - p1, p1])
        d = d - d.max(axis=1, keepdims=True)
        e = np.exp(d)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = self.decision_function(x)
        idx = (d > 0).astype(int) if d.ndim == 1 else d.argmax(axis=1)
        return self.classes_[idx]
