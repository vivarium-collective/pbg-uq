import numpy as np
from pbg_uq.report import generate_html_report
from pbg_uq.results import SobolIndices, PCESurrogate, UQPCResult


def _fake_result():
    sob = SobolIndices(first_order=np.array([0.3, 0.5]),
                       total_order=np.array([0.35, 0.6]),
                       parameter_names=["a", "b"])
    surr = PCESurrogate(coefficients=np.array([1.0, 0.2, 0.1]),
                        multi_indices=np.array([[0, 0], [1, 0], [0, 1]]),
                        polynomial_order=2, input_dim=2, output_dim=1,
                        input_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]))
    return UQPCResult(
        sobol=sob, surrogate=surr, pcrv=None, linregs=[],
        germ_train=np.zeros((4, 2)), X_train=np.zeros((4, 2)), Y_train=np.zeros((4, 1)),
        Y_train_pc=np.zeros((4, 1)), Y_train_pc_std=np.zeros((4, 1)),
        relerr_train=np.array([0.02]),
    )


def test_report_emits_html(tmp_path):
    html = generate_html_report(_fake_result(), parameter_names=["a", "b"],
                                observable_names=["y0"])
    assert "<html" in html.lower()
    assert "Sobol" in html
    assert "a" in html and "b" in html


def test_report_writes_file(tmp_path):
    html = generate_html_report(_fake_result(), parameter_names=["a", "b"])
    p = tmp_path / "r.html"
    p.write_text(html)
    assert p.stat().st_size > 1000
