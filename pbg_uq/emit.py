"""XArray-backed observable collector for process-bigraph composites.

Public API
----------
attach_xarray_emitter(document, observables, out_uri, core) -> dict
    Deep-copies *document*, registers a ``_PbgXArrayEmitter`` step on *core*,
    and inserts it into the returned document copy.  The step wraps the
    composite's plain state into the ``agents/<agent_id>/`` shape expected by
    ``XArrayEmitter`` and writes to a zarr store at *out_uri*.

read_run(out_uri, observables) -> dict[str, float]
    Opens the zarr store written by the attached emitter.  Returns the
    time-mean float for each observable.  Observables not found in the store
    are silently returned as 0.0.

Notes
-----
* ``XArrayEmitter`` stores data under a partition prefix
  ``experiment_id=<id>.variant=0.lineage_seed=0.<obs>``.
  ``read_run`` bridges this by scanning ``RunReader.observables()`` for
  entries that end with ``.<obs>`` (or equal ``<obs>``).
* The XArrayEmitter buffer flushes to disk when it fills up (``buf_size=3``);
  the zarr store is readable after a minimum of 4 composite steps.  To ensure
  all data is written, call ``close_xarray_emitter(composite)`` before
  ``read_run`` when the number of steps may be small.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from process_bigraph.emitter import Emitter

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_EMITTER_KEY = "uq_emitter"
_EXPERIMENT_ID = "pbg_uq"
_AGENT_ID = "1"
_BUF_SIZE = 3  # minimum valid for XarrayTransducer (must be > 2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_view(observables: list[str]) -> list[dict]:
    """Build the XArrayEmitter ``view`` config list for *observables*.

    Uses an empty root path ``[]`` so each observable is resolved directly
    as ``agents/<agent_id>/<obs>`` without a ``listeners/`` prefix.
    """
    return [
        {
            "root": [],  # empty: agents/<agent_id>/<obs> directly
            "variables": {
                obs: [{"path": obs, "dtype": "<f8"}]
                for obs in observables
            },
        }
    ]


def _build_inner_config(observables: list[str], out_uri: str) -> dict:
    """Return a fully-populated XArrayEmitter config for *observables*."""
    return {
        "emit": {},
        "out_uri": out_uri,
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": _BUF_SIZE},
        },
        "view": _build_view(observables),
        "writer": {
            "backend": "zarr",
            "store": out_uri,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {
            "experiment_id": _EXPERIMENT_ID,
            "agent_id": _AGENT_ID,
            "variant": 0,
            "lineage_seed": 0,
        },
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }


def _make_emitter_class(observables: list[str], inner_config: dict) -> type:
    """Return a new ``Emitter`` subclass bound to *observables* and *inner_config*.

    The class is created fresh per ``attach_xarray_emitter`` call so each run
    gets its own inner ``XArrayEmitter`` instance (and hence its own zarr store).
    The inner config is embedded as a class attribute to avoid bigraph-schema
    type-checking issues with complex nested dicts in the document.
    """

    class _PbgXArrayEmitter(Emitter):
        """Thin adapter: maps plain composite state into the ``agents/`` shape
        required by ``XArrayEmitter``, then writes each step to zarr."""

        config_schema = {"emit": "schema"}

        # Class-level attributes set by the factory:
        _OBSERVABLES: list[str] = list(observables)
        _INNER_CFG: dict = inner_config

        def __init__(self, config: dict, core: Any) -> None:
            super().__init__(config, core)
            from pbg_emitters.xarray_emitter import XArrayEmitter  # lazy import

            self._observables: list[str] = self._OBSERVABLES
            self._inner: XArrayEmitter = XArrayEmitter(
                config=self._INNER_CFG, core=core
            )
            self._step: int = 0

        def update(self, state: dict) -> dict:
            # process-bigraph fires every step-type emitter once *before* the
            # first process update runs (pre-update placeholder frame).  For
            # the document-path mode that frame contains the document's initial
            # placeholder value (e.g. y=0.0) rather than any real simulation
            # output, so it must not be included in the time-mean.
            #
            # Empirically: each comp.run(interval) call triggers exactly two
            # emitter fires — one before and one after the process update.
            # _step==0 is always the pre-simulation placeholder; all subsequent
            # calls (including the pre-update fires on step 2+ where y already
            # holds the post-update value from the previous interval) are valid
            # simulation observations and should be recorded.
            if self._step == 0:
                self._step += 1
                return {}
            t = float(self._step)
            self._step += 1
            wrapped = {
                "time": t,
                "agents": {
                    _AGENT_ID: {obs: state[obs] for obs in self._observables}
                },
            }
            self._inner.update(wrapped)
            return {}

        def close(self) -> None:
            """Flush the final buffer to zarr.  Idempotent."""
            inner = getattr(self, "_inner", None)
            if inner is not None and not getattr(inner, "_closed", True):
                inner.close()

        def __del__(self) -> None:  # pragma: no cover
            try:
                self.close()
            except Exception:
                pass

    _PbgXArrayEmitter.__name__ = "_PbgXArrayEmitter"
    _PbgXArrayEmitter.__qualname__ = "_PbgXArrayEmitter"
    return _PbgXArrayEmitter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_xarray_emitter(
    document: dict,
    observables: list[str],
    out_uri: str,
    core: Any,
) -> dict:
    """Deep-copy *document* and attach an XArray (zarr) emitter.

    Registers a ``_PbgXArrayEmitter`` step type on *core* and inserts it at
    key ``"_uq_emitter"`` in the returned document copy.  The step wires each
    observable in *observables* to the corresponding top-level state path.

    Args:
        document:    process-bigraph composite document dict.  Not mutated.
        observables: State-tree path keys to capture, e.g. ``["y"]``.
                     Only top-level single-segment paths are supported.
        out_uri:     Zarr store path, e.g. ``"/tmp/run.zarr"``.
        core:        ``bigraph_schema.Core`` instance.  Modified in-place to
                     register the emitter type.

    Returns:
        New document dict (deep copy) with the emitter step inserted under
        key ``"_uq_emitter"``.
    """
    doc = copy.deepcopy(document)
    inner_config = _build_inner_config(observables, out_uri)
    emitter_cls = _make_emitter_class(observables, inner_config)

    # Register on core so ``local:_PbgXArrayEmitter`` resolves correctly.
    core.register_link("_PbgXArrayEmitter", emitter_cls)

    # Build the emitter node.
    emit_schema = {obs: "node" for obs in observables}
    inputs_map = {obs: [obs] for obs in observables}  # single-level wire

    doc[_EMITTER_KEY] = {
        "_type": "step",
        "address": "local:_PbgXArrayEmitter",
        "config": {"emit": emit_schema},
        "inputs": inputs_map,
    }
    return doc


def close_xarray_emitter(composite: Any) -> None:
    """Flush and close the XArray emitter attached by :func:`attach_xarray_emitter`.

    Must be called before :func:`read_run` when the composite may have run
    fewer steps than needed to auto-flush the final buffer (buffer size is
    ``_BUF_SIZE = 3``).  Safe to call multiple times (idempotent).

    Args:
        composite: A process-bigraph ``Composite`` instance that was built
                   from a document produced by :func:`attach_xarray_emitter`.
    """
    from bigraph_schema import get_path

    node = get_path(composite.state, [_EMITTER_KEY])
    if node is None:
        return
    instance = node.get("instance") if isinstance(node, dict) else None
    if instance is not None and hasattr(instance, "close"):
        instance.close()


def _find_observable(reader: Any, obs: str) -> str | None:
    """Map a user observable name to the full ``RunReader`` observable path.

    ``XArrayEmitter`` stores observables under a partition prefix, e.g.
    ``experiment_id=pbg_uq.variant=0.lineage_seed=0.y``.  We find a match
    by scanning ``reader.observables()`` for entries equal to *obs* or that
    end with ``".{obs}"``.

    Args:
        reader: Open ``RunReader`` instance.
        obs:    User-facing observable name (e.g. ``"y"``).

    Returns:
        Matching full observable name, or ``None`` if not found.
    """
    target_suffix = obs.replace("/", ".")
    for full_name in reader.observables():
        if full_name == target_suffix or full_name.endswith("." + target_suffix):
            return full_name
    return None


def read_run(out_uri: str, observables: list[str]) -> dict[str, float]:
    """Read per-observable time-means from a zarr store.

    Opens the zarr store at *out_uri* (written by the emitter attached via
    :func:`attach_xarray_emitter`) using ``RunReader``, extracts the value
    series for each observable, and returns the time-mean as a float.

    Args:
        out_uri:     Zarr store path, same value passed to
                     :func:`attach_xarray_emitter`.
        observables: Observable names to read back, e.g. ``["y"]``.

    Returns:
        Dict mapping each observable name to its time-mean.  If an observable
        is not present in the store (e.g. not yet flushed), its value is
        ``0.0``.
    """
    from pbg_emitters.run_reader import RunReader

    reader = RunReader.open(out_uri, kind="xarray")
    result: dict[str, float] = {}

    for obs in observables:
        full_name = _find_observable(reader, obs)
        if full_name is None:
            result[obs] = 0.0
            continue
        try:
            s = reader.series(full_name)
            vals = s["value"].to_numpy()
            result[obs] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        except Exception:
            result[obs] = 0.0

    return result
