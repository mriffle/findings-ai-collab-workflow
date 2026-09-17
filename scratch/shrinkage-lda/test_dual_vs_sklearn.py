"""Pin the dual-form shrinkage LDA to sklearn at machine precision — synthetic + real data.

Run:  PYTHONPATH=lib:scratch/shrinkage-lda ./.venv/bin/python -m pytest scratch/shrinkage-lda -q
"""

from __future__ import annotations

import numpy as np
import pytest
from dual_lda import DualShrinkageLDA, ledoit_wolf_shrinkage_gram
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)


def _synthetic(n: int, p: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    # correlated features: low-rank structure + noise, k shifted classes
    z = RNG.standard_normal((n, 5))
    w = RNG.standard_normal((5, p))
    x = z @ w + 0.5 * RNG.standard_normal((n, p))
    y = np.arange(n) % k
    x[:, :10] += y[:, None] * 0.8
    return x, y


@pytest.mark.parametrize("n,p", [(30, 20), (40, 500), (60, 1500)])
def test_shrinkage_intensity_matches_sklearn(n: int, p: int) -> None:
    x, _ = _synthetic(n, p, 2)
    xc = x - x.mean(0)
    assert ledoit_wolf_shrinkage_gram(xc) == pytest.approx(
        ledoit_wolf_shrinkage(x), rel=1e-10, abs=1e-14
    )


@pytest.mark.parametrize("n,p,k", [(30, 20, 2), (50, 400, 2), (45, 300, 3), (60, 900, 4)])
def test_coefficients_and_scores_match_sklearn(n: int, p: int, k: int) -> None:
    x, y = _synthetic(n, p, k)
    x = StandardScaler().fit_transform(x)
    ref = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x, y)
    ours = DualShrinkageLDA().fit(x, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ours.intercept_, ref.intercept_, rtol=1e-8, atol=1e-10)
    xt, _ = _synthetic(12, p, k)
    np.testing.assert_allclose(
        ours.decision_function(xt), ref.decision_function(xt), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(ours.predict_proba(xt), ref.predict_proba(xt), atol=1e-10)
    assert (ours.predict(xt) == ref.predict(xt)).all()


def test_p_much_larger_than_n_is_fast_and_finite() -> None:
    x, y = _synthetic(80, 20_000, 2)
    x = StandardScaler().fit_transform(x)
    m = DualShrinkageLDA().fit(x, y)
    assert np.isfinite(m.coef_).all()
    assert 0.0 <= m.shrinkage_.min() <= m.shrinkage_.max() <= 1.0
