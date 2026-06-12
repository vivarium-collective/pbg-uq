import numpy as np
from pbg_uq.adapters.callable import CallableAdapter


def test_callable_adapter_evaluates_and_names():
    fn = lambda X: np.column_stack([X[:, 0] ** 2, X[:, 1]])
    ad = CallableAdapter(fn, param_names=["a", "b"],
                         bounds=np.array([[0.0, 1.0], [0.0, 2.0]]))
    X = np.array([[2.0, 3.0], [1.0, 1.0]])
    Y = ad.evaluate_batch(X)
    assert Y.shape == (2, 2)
    np.testing.assert_allclose(Y[:, 0], [4.0, 1.0])
    assert ad.obs_names == ["y0", "y1"]
