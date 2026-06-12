import numpy as np
from pbg_uq.results import SobolIndices, PCESurrogate


def test_sobol_select_top_k():
    s = SobolIndices(
        first_order=np.array([0.1, 0.5, 0.2]),
        total_order=np.array([0.15, 0.6, 0.25]),
        parameter_names=["a", "b", "c"],
    )
    top = s.select(n=2)
    assert top[0][0] == "b"
    assert top[1][0] == "c"


def test_pce_surrogate_export_roundtrip(tmp_path):
    surr = PCESurrogate(
        coefficients=np.array([1.0, 2.0, 3.0]),
        multi_indices=np.array([[0], [1], [2]]),
        polynomial_order=2, input_dim=1, output_dim=1,
        input_bounds=np.array([[0.0, 1.0]]),
    )
    p = tmp_path / "surr"
    surr.export(p)
    loaded = PCESurrogate.from_export(p)
    np.testing.assert_array_equal(loaded.coefficients, surr.coefficients)
    np.testing.assert_array_equal(loaded.multi_indices, surr.multi_indices)
