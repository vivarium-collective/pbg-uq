"""SimulationAdapter protocol + path resolution for arbitrary pbg state/documents.

Generalizes uqEcoli's uq/v2ecoli_bridge.py (which hard-coded v2ecoli paths)
to any nested-dict composite state. Paths are "/"-joined strings.
"""
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
import numpy as np


def _split(path: str | tuple) -> list[str]:
    if isinstance(path, (tuple, list)):
        return [str(p) for p in path]
    return [p for p in path.split("/") if p != ""]


def resolve_path(d: dict, path: str | tuple, default: Any = None) -> Any:
    """Walk a nested dict by a '/'-joined path. Returns default if missing."""
    target: Any = d
    for key in _split(path):
        if isinstance(target, dict) and key in target:
            target = target[key]
        else:
            return default
    return target


def set_path(d: dict, path: str | tuple, value: Any) -> None:
    """Set a leaf at a '/'-joined path, creating intermediate dicts as needed."""
    keys = _split(path)
    target = d
    for key in keys[:-1]:
        nxt = target.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            target[key] = nxt
        target = nxt
    target[keys[-1]] = value


def read_observable(state: dict, path: str | tuple) -> Any:
    """Read an observable from composite state. Scalar -> float; array -> ndarray.

    Returns 0.0 when the path is missing (matching uqEcoli behavior).
    """
    val = resolve_path(state, path, default=None)
    if val is None:
        return 0.0
    if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
        arr = np.asarray(val, dtype=np.float64)
        return arr.ravel() if arr.ndim > 1 else arr
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


@runtime_checkable
class SimulationAdapter(Protocol):
    """What ForwardUQ requires of a simulator."""

    param_names: list[str]
    bounds: np.ndarray            # (d, 2)

    def evaluate_batch(self, X: np.ndarray, max_workers: int | None = None) -> np.ndarray:
        """Map (n, d) physical-space samples -> (n, m) aggregated observables."""
        ...

    @property
    def obs_names(self) -> list[str]:
        """Observable column names (known after first evaluate_batch)."""
        ...
