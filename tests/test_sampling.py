import numpy as np
from pbg_uq.sampling import draw_samples


def test_draw_samples_in_bounds_and_shape():
    bounds = np.array([[0.0, 10.0], [-1.0, 1.0]])
    X, X_test = draw_samples(bounds, n_samples=50, n_test=10, seed=1)
    assert X.shape == (50, 2)
    assert X_test.shape == (10, 2)
    assert np.all(X[:, 0] >= 0) and np.all(X[:, 0] <= 10)
    assert np.all(X[:, 1] >= -1) and np.all(X[:, 1] <= 1)


def test_draw_samples_reproducible():
    bounds = np.array([[0.0, 1.0]])
    a, _ = draw_samples(bounds, n_samples=20, n_test=0, seed=7)
    b, _ = draw_samples(bounds, n_samples=20, n_test=0, seed=7)
    np.testing.assert_array_equal(a, b)
