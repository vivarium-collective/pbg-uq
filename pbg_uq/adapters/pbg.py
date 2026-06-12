"""PbgAdapter: run any process-bigraph composite under forward-UQ.

Public API
----------
PbgAdapter(*, observables, duration, document, core, params, ...)
    Document-path mode: deepcopy document, inject params, attach XArray or poll
    emitter, run, aggregate.

PbgAdapter.from_builder(*, build_composite, param_bounds, ...)
    Build-hook mode: caller supplies a factory function; adapter calls it per sample.

Notes
-----
* XArray mode uses a **warmup step** (one interval without the emitter) to
  initialize observable values to their process steady-state before the emitter
  run begins.  This eliminates the initial-state bias that would otherwise appear
  when the emitter fires once before any process update (process-bigraph fires
  steps that are "ready" at the start of each ``comp.run()`` before the process
  loop).  After the warmup the pre-initialized observable values are overwritten
  in the document state, so all emitter fires see the correct steady-state value
  and the time-mean converges exactly.

* Poll mode reads ``comp.state`` after each ``comp.run()`` call, so it naturally
  sees post-process values.  With the warmup applied to xarray mode both paths
  agree to floating-point precision for constant processes (and agree in
  expectation for stochastic ones).

* Parallelism: ``max_workers > 1`` is accepted but runs sequentially because
  ``core`` and ``document`` may not pickle cleanly across processes.  A future
  Task-11 follow-up can add multiprocessing for build-hook mode where the builder
  is picklable.
"""

from __future__ import annotations

import copy
import gc
import os
import shutil
import tempfile
from typing import Any, Callable

import numpy as np

from pbg_uq.adapters.base import read_observable, set_path
from pbg_uq.emit import (
    attach_xarray_emitter,
    close_xarray_emitter,
    read_run,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scope(state: dict) -> dict:
    """Return the observable scope from composite state.

    For composites that wrap processes under ``agents/0`` (v2ecoli style),
    redirect to that sub-dict.  Otherwise use the root state.
    """
    agents = state.get("agents")
    if isinstance(agents, dict) and "0" in agents:
        return agents["0"]
    return state


def _flatten_agg(
    agg: dict[str, Any], observables: list[str]
) -> tuple[list[float], list[str]]:
    """Flatten observable aggregates to a flat row + column names.

    Scalar observables → one column named ``obs``.
    Array observables (ndarray with size > 1) → columns ``obs_0 … obs_{k-1}``.
    """
    row: list[float] = []
    names: list[str] = []
    for obs in observables:
        val = agg.get(obs, 0.0)
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size > 1:
            for i, v in enumerate(arr):
                row.append(float(v))
                names.append(f"{obs}_{i}")
        else:
            row.append(float(arr[0]) if arr.size == 1 else 0.0)
            names.append(obs)
    return row, names


# ---------------------------------------------------------------------------
# PbgAdapter
# ---------------------------------------------------------------------------


class PbgAdapter:
    """Run a process-bigraph composite under forward-UQ.

    Supports two construction modes:

    * **Document-path mode** (constructor with ``document`` + ``params``):
      deepcopy the document, inject parameter values at specified paths, then
      run with the configured emitter.

    * **Build-hook mode** (:meth:`from_builder`):
      caller supplies a factory ``build_composite(sample, out_uri=None) ->
      Composite``; adapter calls it once per sample.

    Parameters
    ----------
    observables:
        State-tree paths to measure (top-level keys for document mode).
    duration:
        Simulation duration (number of ``interval``-size steps).
    document:
        Full composite document ``{"state": {...}}``.  Required for
        document-path mode.
    core:
        ``bigraph_schema.Core`` instance (required for document mode;
        optional for builder mode but needed if the adapter attaches an emitter).
    params:
        ``{path: (lo, hi)}`` dict for document-path mode.  Paths are
        relative to ``document["state"]``.
    build_composite:
        Factory for build-hook mode; signature
        ``build_composite(sample: dict[str,float], out_uri: str | None) ->
        Composite``.
    param_bounds:
        ``{name: (lo, hi)}`` dict for build-hook mode.
    interval:
        Step size.  Each ``comp.run(interval)`` is one "tick".
    max_workers:
        Accepted but currently ignored (sequential execution; see module
        Notes for rationale).
    emitter:
        ``"xarray"`` (default) or ``"poll"``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        observables: list[str],
        duration: float,
        document: dict | None = None,
        core: Any = None,
        params: dict | None = None,
        build_composite: Callable | None = None,
        param_bounds: dict | None = None,
        interval: float = 1.0,
        max_workers: int | None = None,
        emitter: str = "xarray",
    ) -> None:
        # Determine mode from provided arguments
        if document is not None and params is not None:
            self._mode = "document"
            self._document = document
            self._core = core
            _param_dict = params
        elif build_composite is not None and param_bounds is not None:
            self._mode = "builder"
            self._build_composite_fn = build_composite
            self._core = core
            _param_dict = param_bounds
        else:
            raise ValueError(
                "Provide either (document, params) for document-path mode "
                "or (build_composite, param_bounds) for build-hook mode."
            )

        self.param_names: list[str] = list(_param_dict.keys())
        self.bounds: np.ndarray = np.array(
            [list(v) for v in _param_dict.values()], dtype=float
        )

        self.observables: list[str] = list(observables)
        self._duration = float(duration)
        self._interval = float(interval)
        self._emitter = emitter
        self._max_workers = max_workers

        self._obs_names: list[str] | None = None

    @classmethod
    def from_builder(
        cls,
        *,
        build_composite: Callable,
        param_bounds: dict,
        observables: list[str],
        duration: float,
        core: Any = None,
        interval: float = 1.0,
        max_workers: int | None = None,
        emitter: str = "xarray",
    ) -> "PbgAdapter":
        """Build-hook constructor.

        Parameters mirror the main constructor; ``build_composite`` and
        ``param_bounds`` are required; ``document``/``params`` are absent.
        """
        return cls(
            observables=observables,
            duration=duration,
            build_composite=build_composite,
            param_bounds=param_bounds,
            core=core,
            interval=interval,
            max_workers=max_workers,
            emitter=emitter,
        )

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def obs_names(self) -> list[str]:
        """Observable column names.  Available after first :meth:`evaluate_batch`."""
        if self._obs_names is None:
            raise RuntimeError(
                "obs_names is not yet known; call evaluate_batch first."
            )
        return self._obs_names

    # ------------------------------------------------------------------
    # Internal: xarray run (document-path mode)
    # ------------------------------------------------------------------

    def _run_doc_xarray(self, sample: dict[str, float], out_uri: str) -> dict:
        """Document-path + xarray emitter run for one sample.

        A warmup step is performed without the emitter so that observable
        state keys hold their process steady-state values before the emitter
        attaches.  This ensures the emitter's initial-state fires record the
        correct value rather than the document's pre-initialized placeholder.
        """
        from process_bigraph import Composite  # lazy import to keep module light

        # --- warmup: run one interval without emitter ---
        doc_w = copy.deepcopy(self._document)
        state_w = doc_w.get("state", doc_w)
        for path, val in sample.items():
            set_path(state_w, path, val)
        comp_w = Composite(doc_w, core=self._core)
        comp_w.run(self._interval)
        warmup_obs = {obs: read_observable(comp_w.state, obs) for obs in self.observables}
        del comp_w
        gc.collect()

        # --- emitter run: deepcopy + inject params + set warmup values ---
        doc = copy.deepcopy(self._document)
        state = doc.get("state", doc)
        for path, val in sample.items():
            set_path(state, path, val)
        # Override initial observable values with warmup steady-state
        for obs, val in warmup_obs.items():
            v = val
            if isinstance(v, np.ndarray):
                state[obs] = v
            else:
                state[obs] = float(v)

        new_state = attach_xarray_emitter(state, self.observables, out_uri, self._core)
        if "state" in doc:
            doc["state"] = new_state
        else:
            doc = new_state

        comp = Composite(doc, core=self._core)
        nsteps = max(1, round(self._duration / self._interval))
        for _ in range(nsteps):
            comp.run(self._interval)
        close_xarray_emitter(comp)
        agg = read_run(out_uri, self.observables)
        del comp
        gc.collect()
        return agg

    # ------------------------------------------------------------------
    # Internal: poll run (document-path mode)
    # ------------------------------------------------------------------

    def _run_doc_poll(self, sample: dict[str, float]) -> dict:
        """Document-path + poll emitter run for one sample."""
        from process_bigraph import Composite

        doc = copy.deepcopy(self._document)
        state = doc.get("state", doc)
        for path, val in sample.items():
            set_path(state, path, val)

        comp = Composite(doc, core=self._core)
        nsteps = max(1, round(self._duration / self._interval))
        history: dict[str, list] = {obs: [] for obs in self.observables}

        for _ in range(nsteps):
            comp.run(self._interval)
            scope = _scope(comp.state)
            for obs in self.observables:
                history[obs].append(read_observable(scope, obs))

        agg: dict[str, Any] = {}
        for obs in self.observables:
            vals = history[obs]
            if not vals:
                agg[obs] = 0.0
            else:
                v0 = vals[0]
                if isinstance(v0, np.ndarray) and v0.size > 1:
                    agg[obs] = np.mean(np.stack(vals), axis=0)
                else:
                    agg[obs] = float(np.mean([float(v) for v in vals]))

        del comp
        gc.collect()
        return agg

    # ------------------------------------------------------------------
    # Internal: builder-mode runs
    # ------------------------------------------------------------------

    def _run_builder_xarray(self, sample: dict[str, float], out_uri: str) -> dict:
        """Build-hook + xarray emitter run.

        The adapter passes ``out_uri`` to the builder so it can wire its own
        XArray emitter.  The adapter calls ``close_xarray_emitter`` and
        ``read_run`` after the run.
        """
        comp = self._build_composite_fn(sample, out_uri=out_uri)
        nsteps = max(1, round(self._duration / self._interval))
        for _ in range(nsteps):
            comp.run(self._interval)
        close_xarray_emitter(comp)
        agg = read_run(out_uri, self.observables)
        del comp
        gc.collect()
        return agg

    def _run_builder_poll(self, sample: dict[str, float]) -> dict:
        """Build-hook + poll emitter run."""
        comp = self._build_composite_fn(sample)
        nsteps = max(1, round(self._duration / self._interval))
        history: dict[str, list] = {obs: [] for obs in self.observables}

        for _ in range(nsteps):
            comp.run(self._interval)
            scope = _scope(comp.state)
            for obs in self.observables:
                history[obs].append(read_observable(scope, obs))

        agg: dict[str, Any] = {}
        for obs in self.observables:
            vals = history[obs]
            if not vals:
                agg[obs] = 0.0
            else:
                v0 = vals[0]
                if isinstance(v0, np.ndarray) and v0.size > 1:
                    agg[obs] = np.mean(np.stack(vals), axis=0)
                else:
                    agg[obs] = float(np.mean([float(v) for v in vals]))

        del comp
        gc.collect()
        return agg

    # ------------------------------------------------------------------
    # Internal: dispatch per sample
    # ------------------------------------------------------------------

    def _run_sample(self, sample: dict[str, float]) -> tuple[list[float], list[str]]:
        """Run one parameter sample; return (row, obs_names)."""
        if self._emitter == "xarray":
            tmp_dir = tempfile.mkdtemp()
            out_uri = os.path.join(tmp_dir, "run.zarr")
            try:
                if self._mode == "document":
                    agg = self._run_doc_xarray(sample, out_uri)
                else:
                    agg = self._run_builder_xarray(sample, out_uri)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:  # poll
            if self._mode == "document":
                agg = self._run_doc_poll(sample)
            else:
                agg = self._run_builder_poll(sample)

        return _flatten_agg(agg, self.observables)

    # ------------------------------------------------------------------
    # Public: evaluate_batch
    # ------------------------------------------------------------------

    def evaluate_batch(
        self, X: np.ndarray, max_workers: int | None = None
    ) -> np.ndarray:
        """Map (n, d) parameter samples to (n, m) observable aggregates.

        Parameters
        ----------
        X:
            ``(n, d)`` array of physical-space parameter values.
        max_workers:
            Accepted but currently ignored (sequential; see module Notes).

        Returns
        -------
        np.ndarray
            ``(n, m)`` array of time-mean observable values.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]

        rows: list[list[float]] = []
        cached_names: list[str] | None = None

        for i in range(n):
            sample = {
                name: float(X[i, j]) for j, name in enumerate(self.param_names)
            }
            row, obs_names = self._run_sample(sample)
            if cached_names is None:
                cached_names = obs_names
            rows.append(row)

        if self._obs_names is None and cached_names:
            self._obs_names = cached_names

        return np.array(rows, dtype=float)
