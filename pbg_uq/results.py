from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pytuq.rv.pcrv import PCRV

from pbg_uq.io import DataclassIO


@dataclass
class SobolIndices:
    """
    Container for Sobol sensitivity indices.

    Attributes:
        first_order: First-order (main effect) indices, shape (n_params,) or (n_outputs, n_params)
        total_order: Total-order indices, shape (n_params,) or (n_outputs, n_params)
        second_order: Second-order interaction indices, shape (n_params, n_params) or (n_outputs, n_params, n_params)
        parameter_names: Names of the input parameters
        output_names: Names of the outputs (if multiple)
        confidence_intervals: Optional confidence intervals for indices
    """

    first_order: np.ndarray
    total_order: np.ndarray
    second_order: Optional[np.ndarray] = None
    parameter_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    confidence_intervals: Optional[dict[str, np.ndarray]] = None

    @property
    def selection_size(self) -> int:
        return 5

    @property
    def selections(
        self,
    ) -> list[tuple[str, float]]:
        return self.select(n=self.selection_size)

    def select(self, n: int = 5, index_type: str = "total") -> list[tuple[str, float]]:
        """
        Get the most influential parameters.

        Args:
            n: Number of top parameters to return
            index_type: Type of index to use ('first' or 'total')

        Returns:
            List of (parameter_name, index_value) tuples
        """
        if index_type == "first":
            indices = self.first_order
        else:
            indices = self.total_order

        # Handle multi-output case by averaging
        if indices.ndim > 1:
            indices = np.mean(indices, axis=0)

        sorted_idx = np.argsort(indices)[::-1][:n]
        return [(self.parameter_names[i], indices[i]) for i in sorted_idx]


@dataclass
class PCESurrogate:
    """
    Polynomial Chaos Expansion surrogate model.

    Wraps a fitted PyTUQ PCE object for prediction and uncertainty estimation.
    When the live PyTUQ object is available (normal usage), prediction delegates
    to PyTUQ's evaluate(). When loaded from disk (via from_export), the PyTUQ
    object is reconstructed from stored coefficients and multi-indices.

    Attributes:
        coefficients: PCE coefficients, shape (n_terms,) or (n_terms, n_outputs)
        multi_indices: Multi-index matrix for polynomial terms, shape (n_terms, n_params)
        basis_type: Type of polynomial basis used ('legendre' or 'hermite')
        polynomial_order: Maximum polynomial order
        input_dim: Number of input parameters
        output_dim: Number of outputs
        r_squared: Coefficient of determination
        input_bounds: Optional bounds for input normalization, shape (n_params, 2)
    """

    coefficients: np.ndarray
    multi_indices: np.ndarray
    basis_type: str = "legendre"
    polynomial_order: int = 3
    input_dim: int = 0
    output_dim: int = 0
    r_squared: float = 0.0
    input_bounds: Optional[np.ndarray] = None

    def __post_init__(self):
        # Live PyTUQ PCE object — set via set_pytuq_pce() or built lazily
        self._pytuq_pce = None

    def set_pytuq_pce(self, pce) -> None:
        """Attach a fitted PyTUQ PCE object (with lreg state from build())."""
        self._pytuq_pce = pce

    def _get_pytuq_pce(self):
        """Get or reconstruct a PyTUQ PCE for evaluation.

        If a live fitted object exists (from set_pytuq_pce), use it directly.
        Otherwise, reconstruct from stored coefficients by creating a PCE,
        setting training data from a minimal identity, and building.
        """
        if self._pytuq_pce is not None:
            return self._pytuq_pce

        from pytuq.surrogates.pce import PCE as PyTUQ_PCE

        pc_type = {"legendre": "LU", "hermite": "HG"}.get(self.basis_type, "LU")
        pce = PyTUQ_PCE(self.input_dim, self.polynomial_order, pc_type)
        # Build with minimal synthetic data so lreg is initialized
        n_terms = len(self.multi_indices)
        n_train = max(n_terms + 1, 2 * n_terms)
        rng = np.random.default_rng(42)
        X_train = rng.uniform(-1, 1, (n_train, self.input_dim))
        pce.set_training_data(X_train, np.zeros(n_train))
        pce.build(regression="lsq")
        # Overwrite with our actual coefficients via PyTUQ's setCfs API
        pce.pcrv.setCfs(self.coefficients.copy())
        self._pytuq_pce = pce
        return pce

    def _to_germ_space(self, X: np.ndarray) -> np.ndarray:
        """Scale physical inputs to PyTUQ germ space [-1, 1]."""
        if self.input_bounds is None:
            return X
        from pytuq.utils.maps import scaleDomTo01

        X01 = scaleDomTo01(X, self.input_bounds)
        return 2.0 * X01 - 1.0

    def export(self, path: str | Path):
        return DataclassIO.save(instance=self, path=path)

    @classmethod
    def from_export(cls, path: str | Path) -> "PCESurrogate":
        return DataclassIO.load(path=path, _class=PCESurrogate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs using PyTUQ's PCE evaluate.

        Args:
            X: Input array of shape (n_samples, n_params) or (n_params,)

        Returns:
            Predicted outputs
        """
        squeeze = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze = True

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input has {X.shape[1]} features, expected {self.input_dim}")

        X_germ = self._to_germ_space(X)
        pce = self._get_pytuq_pce()
        result = pce.evaluate(X_germ)
        predictions = result["Y_eval"]

        return predictions.squeeze() if squeeze else predictions

    def predict_with_uncertainty(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict outputs with uncertainty estimate.

        When the PCE was built with analytical regression, PyTUQ provides
        posterior predictive variance. Otherwise uses coefficient-based estimate.

        Args:
            X: Input array of shape (n_samples, n_params)
            return_std: If True, return std; otherwise return variance

        Returns:
            Tuple of (predictions, uncertainty)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_germ = self._to_germ_space(X)
        pce = self._get_pytuq_pce()
        result = pce.evaluate(X_germ)
        predictions = result["Y_eval"]

        if result.get("Y_eval_var") is not None:
            variance = result["Y_eval_var"]
        else:
            coeffs = self.coefficients.reshape(-1, 1) if self.coefficients.ndim == 1 else self.coefficients
            variance = np.sum(coeffs[1:] ** 2, axis=0)

        uncertainty = np.sqrt(variance) if return_std else variance

        if predictions.ndim == 1:
            return predictions, uncertainty
        return predictions, np.broadcast_to(uncertainty, predictions.shape)


@dataclass
class UQPCResult:
    """Complete output of the UQPC workflow for one aggregation strategy.

    Mirrors the ``results.pk`` dict from ``uq_pc.py``, but uses RFC006
    domain types and stores per-strategy diagnostics.

    Attributes:
        sobol: Sobol sensitivity indices (main, total, joint).
        surrogate: Exportable PCE surrogate with coefficients.
        pcrv: The fitted ``PCRV`` object (PyTUQ internal repr).
        linregs: Per-output linear regression objects from ``pc_fit``.
        germ_train: Training samples in germ space [-1, 1].
        X_train: Training samples in physical space.
        Y_train: Training outputs.
        Y_train_pc: PCE predictions at training points.
        Y_train_pc_std: Prediction std dev at training points.
        relerr_train: Per-output relative error at training points.
        germ_test: Test samples in germ space (None if no test set).
        X_test: Test samples in physical space (None if no test set).
        Y_test: Test outputs (None if no test set).
        Y_test_pc: PCE predictions at test points (None if no test set).
        Y_test_pc_std: Prediction std dev at test points.
        relerr_test: Per-output relative error at test points.
        relerr_cv: Per-output relative error from k-fold cross-validation
            (used in place of relerr_test when no independent test set is
            available — BYO mode or default mode with ``--n-test 0``).
        cv_n_folds: Number of folds used to compute relerr_cv (0 = not run).
        sobol_first_order_ci: Empirical 95% CI on first-order Sobol indices
            from bootstrap, shape (n_params, 2) — columns [low, high].
            None when bootstrap not run.
        sobol_total_order_ci: Empirical 95% CI on total-order Sobol indices,
            shape (n_params, 2). None when bootstrap not run.
        n_bootstrap: Number of bootstrap resamples actually used (0 = off).
    """

    sobol: SobolIndices
    surrogate: PCESurrogate
    pcrv: PCRV
    linregs: list[Any]
    germ_train: np.ndarray
    X_train: np.ndarray
    Y_train: np.ndarray
    Y_train_pc: np.ndarray
    Y_train_pc_std: np.ndarray
    relerr_train: np.ndarray
    germ_test: np.ndarray | None = None
    X_test: np.ndarray | None = None
    Y_test: np.ndarray | None = None
    Y_test_pc: np.ndarray | None = None
    Y_test_pc_std: np.ndarray | None = None
    relerr_test: np.ndarray | None = None
    relerr_cv: np.ndarray | None = None
    cv_n_folds: int = 0
    sobol_first_order_ci: np.ndarray | None = None
    sobol_total_order_ci: np.ndarray | None = None
    n_bootstrap: int = 0
