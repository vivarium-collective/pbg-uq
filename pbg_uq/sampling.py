"""PCRV germ sampling + affine map ξ∈[-1,1]^d → physical [lb, ub] (Legendre)."""
from __future__ import annotations
import numpy as np
from pbg_uq.uqpc import _setup_input_pc, _generate_training_samples, _generate_test_samples


def draw_samples(
    bounds: np.ndarray, n_samples: int, n_test: int = 0, seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw training (+ optional held-out test) samples in physical space.

    Uses PyTUQ's PCRV germ measure (uniform → Legendre), mapped affinely
    onto the per-parameter [lb, ub] bounds.

    Returns (X_train (n_samples, d), X_test (n_test, d) or None).
    """
    pc, _, in_pcdim = _setup_input_pc(bounds)
    _, X_train = _generate_training_samples(pc, n_samples, in_pcdim, seed=seed)
    X_test = None
    if n_test > 0:
        test_seed = None if seed is None else seed + 1
        _, X_test = _generate_test_samples(pc, n_test, seed=test_seed)
    return X_train, X_test
