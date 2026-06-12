"""On-disk cache of (X, Y) sample pairs for UQ analysis.

Lifted from uqEcoli/libuq/sampling.py (PrecomputedCache only),
decoupled from libuq, scipy, and XSpace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PrecomputedCache:
    """On-disk cache of (X, Y) sample pairs.

    Attributes:
        cache_dir: Directory containing the cached files.
        X: Input (training) samples, shape (n_samples, n_params).
        Y: Aggregated training outputs, shape (n_samples, n_outputs).
        parameter_names: Names of the input parameters.
        metadata: Additional metadata (bounds, polynomial_order, etc.).
        Y_timeseries: Per-sample raw timeseries for Phase 2.
            List of arrays, each (n_timesteps, n_obs). None if not cached.
        Y_timeseries_meta: Per-sample row-level metadata for aggregation
            strategies 2 (by generation) and 3 (by lineage seed).
            List of dicts, each ``{"generation": array, "lineage_seed": array}``
            matching the rows of the corresponding ``Y_timeseries[i]``.
            None if not available (e.g. old caches or synthetic data).
        X_test: Optional held-out validation inputs, shape (n_test, n_params).
            Maps to the PyTUQ UQPC ``--ntst`` flag.  Used by
            ``uq.workflow.quantify`` to compute test relative errors.
        Y_test: Optional held-out validation outputs, shape (n_test, n_outputs).
        Y_test_timeseries: Optional per-test-sample raw timeseries.
        Y_test_timeseries_meta: Optional per-test-sample metadata.
    """

    cache_dir: Path
    X: np.ndarray
    Y: np.ndarray
    parameter_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    Y_timeseries: list[np.ndarray] | None = None
    Y_timeseries_meta: list[dict[str, np.ndarray]] | None = None
    X_test: np.ndarray | None = None
    Y_test: np.ndarray | None = None
    Y_test_timeseries: list[np.ndarray] | None = None
    Y_test_timeseries_meta: list[dict[str, np.ndarray]] | None = None

    def save(self) -> None:
        """Save X.npy, Y.npy, metadata.json (and optional timeseries) to cache_dir."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.cache_dir / "X.npy", self.X)
        np.save(self.cache_dir / "Y.npy", self.Y)

        meta = {
            "parameter_names": self.parameter_names,
            "n_samples": int(self.X.shape[0]),
            "n_params": int(self.X.shape[1]),
            "n_outputs": int(self.Y.shape[1]),
            **self.metadata,
        }
        (self.cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        # Save per-sample timeseries if available (for Phase 2)
        if self.Y_timeseries is not None:
            ts_dir = self.cache_dir / "timeseries"
            ts_dir.mkdir(exist_ok=True)
            for i, ts in enumerate(self.Y_timeseries):
                np.save(ts_dir / f"sample_{i:04d}.npy", ts)

        # Save per-sample metadata (generation/seed labels) if available
        if self.Y_timeseries_meta is not None:
            ts_dir = self.cache_dir / "timeseries"
            ts_dir.mkdir(exist_ok=True)
            for i, meta_dict in enumerate(self.Y_timeseries_meta):
                np.savez(
                    ts_dir / f"sample_{i:04d}_meta.npz",
                    **meta_dict,
                )

        # Save held-out test set (UQPC ``--ntst`` validation samples)
        if self.X_test is not None and self.Y_test is not None:
            np.save(self.cache_dir / "X_test.npy", self.X_test)
            np.save(self.cache_dir / "Y_test.npy", self.Y_test)
        if self.Y_test_timeseries is not None:
            ts_test_dir = self.cache_dir / "timeseries_test"
            ts_test_dir.mkdir(exist_ok=True)
            for i, ts in enumerate(self.Y_test_timeseries):
                np.save(ts_test_dir / f"sample_{i:04d}.npy", ts)
        if self.Y_test_timeseries_meta is not None:
            ts_test_dir = self.cache_dir / "timeseries_test"
            ts_test_dir.mkdir(exist_ok=True)
            for i, meta_dict in enumerate(self.Y_test_timeseries_meta):
                np.savez(
                    ts_test_dir / f"sample_{i:04d}_meta.npz",
                    **meta_dict,
                )

    @classmethod
    def load(cls, cache_dir: str | Path) -> PrecomputedCache:
        """Load from cache_dir."""
        cache_dir = Path(cache_dir)
        X = np.load(cache_dir / "X.npy")
        Y = np.load(cache_dir / "Y.npy")
        meta = json.loads((cache_dir / "metadata.json").read_text())
        parameter_names = meta.pop("parameter_names")

        # Load timeseries if available
        ts_dir = cache_dir / "timeseries"
        Y_timeseries = None
        Y_timeseries_meta = None
        if ts_dir.exists():
            n_samples = X.shape[0]
            Y_timeseries = []
            has_meta = (ts_dir / "sample_0000_meta.npz").exists()
            if has_meta:
                Y_timeseries_meta = []
            for i in range(n_samples):
                ts_path = ts_dir / f"sample_{i:04d}.npy"
                if ts_path.exists():
                    Y_timeseries.append(np.load(ts_path))
                meta_path = ts_dir / f"sample_{i:04d}_meta.npz"
                if has_meta and meta_path.exists():
                    npz = np.load(meta_path)
                    Y_timeseries_meta.append({k: npz[k] for k in npz.files})

        # Load optional held-out test set (UQPC --ntst)
        X_test: np.ndarray | None = None
        Y_test: np.ndarray | None = None
        X_test_path = cache_dir / "X_test.npy"
        Y_test_path = cache_dir / "Y_test.npy"
        if X_test_path.exists() and Y_test_path.exists():
            X_test = np.load(X_test_path)
            Y_test = np.load(Y_test_path)

        Y_test_timeseries: list[np.ndarray] | None = None
        Y_test_timeseries_meta: list[dict[str, np.ndarray]] | None = None
        ts_test_dir = cache_dir / "timeseries_test"
        if ts_test_dir.exists() and X_test is not None:
            n_test = X_test.shape[0]
            Y_test_timeseries = []
            has_test_meta = (ts_test_dir / "sample_0000_meta.npz").exists()
            if has_test_meta:
                Y_test_timeseries_meta = []
            for i in range(n_test):
                ts_path = ts_test_dir / f"sample_{i:04d}.npy"
                if ts_path.exists():
                    Y_test_timeseries.append(np.load(ts_path))
                meta_path = ts_test_dir / f"sample_{i:04d}_meta.npz"
                if has_test_meta and meta_path.exists():
                    npz = np.load(meta_path)
                    Y_test_timeseries_meta.append({k: npz[k] for k in npz.files})

        return cls(
            cache_dir=cache_dir,
            X=X,
            Y=Y,
            parameter_names=parameter_names,
            metadata=meta,
            Y_timeseries=Y_timeseries,
            Y_timeseries_meta=Y_timeseries_meta,
            X_test=X_test,
            Y_test=Y_test,
            Y_test_timeseries=Y_test_timeseries,
            Y_test_timeseries_meta=Y_test_timeseries_meta,
        )
