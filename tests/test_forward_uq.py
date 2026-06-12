import numpy as np
import pytest
from pbg_uq import ForwardUQ, CallableAdapter


def _ishigami(X, a=7.0, b=0.1):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)).reshape(-1, 1)


def test_forward_uq_end_to_end(tmp_path):
    bounds = np.array([[-np.pi, np.pi]] * 3)
    sim = CallableAdapter(_ishigami, ["x1", "x2", "x3"], bounds)
    uq = ForwardUQ(sim=sim, n_samples=600, n_test=100, order=8,
                   seed=0, cache_dir=str(tmp_path / "cache"))
    uq.sample()
    assert (tmp_path / "cache" / "X.npy").exists()

    result = uq.quantify()
    s1 = result.sobol.first_order
    assert abs(s1[0] - 0.314) < 0.06
    assert abs(s1[1] - 0.442) < 0.06
    assert s1[2] < 0.06
    assert result.relerr_test is not None and result.relerr_test[0] < 0.1

    # report() requires Task 9 (pbg_uq.report); skip if not present yet
    pytest.importorskip("pbg_uq.report")
    out = tmp_path / "report.html"
    uq.report(str(out))
    assert out.exists() and out.stat().st_size > 1000
