"""Integration test for pbg_uq.emit: attach → run → read via XArrayEmitter."""

import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

from process_bigraph.composite import Process, Composite
from bigraph_schema import allocate_core

from pbg_uq.emit import attach_xarray_emitter, read_run, close_xarray_emitter


# ---------------------------------------------------------------------------
# Minimal synthetic process
# ---------------------------------------------------------------------------

class LinearSim(Process):
    """Always returns y = 2*gain + offset (overwrite, not delta)."""

    config_schema = {
        "gain":   {"_type": "float", "_default": 1.0},
        "offset": {"_type": "float", "_default": 0.0},
    }

    def inputs(self):
        return {}

    def outputs(self):
        # ``overwrite[float]`` so the output replaces (not adds to) ``y``.
        return {"y": "overwrite[float]"}

    def update(self, state, interval):
        return {"y": 2.0 * self.config["gain"] + self.config["offset"]}


def _core():
    core = allocate_core()
    core.register_link("LinearSim", LinearSim)
    return core


def _document():
    """
    Composite state with LinearSim and y initialised to the steady-state
    value (7.0) so every emitter fire — including the t=0 initialisation
    emit — records the correct value.
    """
    return {
        "state": {
            "sim": {
                "_type": "process",
                "address": "local:LinearSim",
                "config": {"gain": 3.0, "offset": 1.0},
                "inputs": {},
                "outputs": {"y": ["y"]},
            },
            # y = 2*3+1 = 7.0.  Pre-initialise so the t=0 emitter fire also
            # sees 7.0 (process outputs are overwrite; initial matters).
            "y": 7.0,
        }
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_attach_run_read(tmp_path):
    """attach_xarray_emitter + run + read_run returns time-mean y=7.0."""
    core = _core()
    out_uri = str(tmp_path / "run.zarr")

    # attach_xarray_emitter works on the nested "state" dict
    base_doc = _document()
    base_doc["state"] = attach_xarray_emitter(
        base_doc["state"],
        observables=["y"],
        out_uri=out_uri,
        core=core,
    )

    comp = Composite(base_doc, core=core)

    # Run enough steps to trigger at least one automatic buffer flush
    # (XArrayEmitter buf_size=3 flushes after 4 calls to update).
    for _ in range(5):
        comp.run(1.0)

    # Flush any remaining partial buffer before reading.
    close_xarray_emitter(comp)

    agg = read_run(out_uri, observables=["y"])

    # y = 2*gain + offset = 2*3 + 1 = 7.0 every tick → time-mean 7.0
    assert "y" in agg, f"'y' missing from read_run result: {agg}"
    assert abs(agg["y"] - 7.0) < 1e-4, (
        f"Expected time-mean y=7.0, got {agg['y']}"
    )


def test_missing_observable_returns_zero(tmp_path):
    """read_run returns 0.0 for observables absent from the store."""
    core = _core()
    out_uri = str(tmp_path / "run.zarr")

    base_doc = _document()
    base_doc["state"] = attach_xarray_emitter(
        base_doc["state"],
        observables=["y"],
        out_uri=out_uri,
        core=core,
    )
    comp = Composite(base_doc, core=core)
    for _ in range(5):
        comp.run(1.0)
    close_xarray_emitter(comp)

    agg = read_run(out_uri, observables=["y", "nonexistent"])
    assert agg["nonexistent"] == 0.0


def test_attach_does_not_mutate_original(tmp_path):
    """attach_xarray_emitter must not modify the original document."""
    core = _core()
    out_uri = str(tmp_path / "run.zarr")
    orig = _document()
    orig_state_copy = dict(orig["state"])

    attach_xarray_emitter(orig["state"], observables=["y"], out_uri=out_uri, core=core)

    assert list(orig["state"].keys()) == list(orig_state_copy.keys()), (
        "attach_xarray_emitter mutated the original document"
    )
