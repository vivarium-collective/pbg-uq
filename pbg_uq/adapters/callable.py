from __future__ import annotations
from typing import Callable
import numpy as np


class CallableAdapter:
    """Wrap a raw f(X)->Y for non-pbg models (and golden tests)."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray],
                 param_names: list[str], bounds: np.ndarray,
                 obs_names: list[str] | None = None):
        self.fn = fn
        self.param_names = list(param_names)
        self.bounds = np.asarray(bounds, dtype=float)
        self._obs_names = obs_names

    def evaluate_batch(self, X: np.ndarray, max_workers: int | None = None) -> np.ndarray:
        Y = np.asarray(self.fn(X), dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if self._obs_names is None:
            self._obs_names = [f"y{j}" for j in range(Y.shape[1])]
        return Y

    @property
    def obs_names(self) -> list[str]:
        if self._obs_names is None:
            raise RuntimeError("Run evaluate_batch first.")
        return self._obs_names
