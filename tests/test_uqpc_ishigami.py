import numpy as np
from pbg_uq.uqpc import fit_pce_and_sobol


def _ishigami(X, a=7.0, b=0.1):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)).reshape(-1, 1)


def test_ishigami_sobol_recovered():
    rng = np.random.default_rng(0)
    bounds = np.array([[-np.pi, np.pi]] * 3)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(800, 3))
    Y = _ishigami(X)

    result = fit_pce_and_sobol(X, Y, bounds, polynomial_order=8, regression="lsq")
    s1 = result.sobol.first_order
    st = result.sobol.total_order

    assert abs(s1[0] - 0.314) < 0.05
    assert abs(s1[1] - 0.442) < 0.05
    assert s1[2] < 0.05
    assert abs(st[2] - 0.244) < 0.06          # x3 only matters via interaction
    assert result.relerr_train[0] < 0.05      # surrogate fits well
