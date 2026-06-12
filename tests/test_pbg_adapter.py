"""Integration tests for PbgAdapter: document-path + build-hook modes, xarray + poll emitters."""

import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

from process_bigraph.composite import Process, Composite
from bigraph_schema import allocate_core

from pbg_uq.adapters.pbg import PbgAdapter


# ---------------------------------------------------------------------------
# Minimal synthetic process (copied from tests/test_emit.py)
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
        return {"y": "overwrite[float]"}

    def update(self, state, interval):
        return {"y": 2.0 * self.config["gain"] + self.config["offset"]}


def _core():
    core = allocate_core()
    core.register_link("LinearSim", LinearSim)
    return core


def _document():
    """Document with LinearSim; default gain/offset will be overridden by params."""
    return {
        "state": {
            "sim": {
                "_type": "process",
                "address": "local:LinearSim",
                "config": {"gain": 3.0, "offset": 1.0},
                "inputs": {},
                "outputs": {"y": ["y"]},
            },
            # Pre-init y to steady-state for default params (y=2*3+1=7.0).
            # The adapter's warmup run overrides this for other param values,
            # ensuring xarray-mode initial fires also see the correct value.
            "y": 7.0,
        }
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adapter(emitter: str) -> PbgAdapter:
    return PbgAdapter(
        document=_document(),
        core=_core(),
        params={
            "sim/config/gain":   (0.0, 5.0),
            "sim/config/offset": (0.0, 10.0),
        },
        observables=["y"],
        duration=3.0,
        interval=1.0,
        emitter=emitter,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pbg_adapter_xarray_mode():
    """XArray-mode adapter returns correct time-mean y = 2*gain + offset."""
    ad = _adapter("xarray")
    X = np.array([[3.0, 1.0], [2.0, 0.0]])  # y = 7.0, y = 4.0
    Y = ad.evaluate_batch(X)
    assert Y.shape == (2, 1), f"Expected (2,1), got {Y.shape}"
    np.testing.assert_allclose(Y[:, 0], [7.0, 4.0], atol=1e-4)
    assert ad.obs_names == ["y"]


def test_pbg_adapter_poll_mode():
    """Poll-mode adapter returns correct time-mean y = 2*gain + offset."""
    ad = _adapter("poll")
    X = np.array([[3.0, 1.0], [2.0, 0.0]])  # y = 7.0, y = 4.0
    Y = ad.evaluate_batch(X)
    assert Y.shape == (2, 1), f"Expected (2,1), got {Y.shape}"
    np.testing.assert_allclose(Y[:, 0], [7.0, 4.0], atol=1e-6)
    assert ad.obs_names == ["y"]


def test_pbg_adapter_poll_matches_xarray():
    """Both emitter modes return the same value for a known sample."""
    X = np.array([[2.0, 0.0]])   # y = 4.0
    Yx = _adapter("xarray").evaluate_batch(X)
    Yp = _adapter("poll").evaluate_batch(X)
    np.testing.assert_allclose(Yx, Yp, atol=1e-4)
    np.testing.assert_allclose(Yp[:, 0], [4.0], atol=1e-6)


def test_pbg_adapter_obs_names_cached():
    """obs_names is set after first evaluate_batch and stays consistent."""
    ad = _adapter("poll")
    X = np.array([[1.0, 0.0]])
    ad.evaluate_batch(X)
    assert ad.obs_names == ["y"]
    # Second call — obs_names unchanged
    ad.evaluate_batch(X)
    assert ad.obs_names == ["y"]


def test_pbg_adapter_param_names_and_bounds():
    """param_names and bounds are derived from the params dict."""
    ad = _adapter("poll")
    assert ad.param_names == ["sim/config/gain", "sim/config/offset"]
    np.testing.assert_array_equal(ad.bounds, [[0.0, 5.0], [0.0, 10.0]])


def test_pbg_adapter_protocol():
    """PbgAdapter satisfies the SimulationAdapter protocol."""
    from pbg_uq.adapters.base import SimulationAdapter
    ad = _adapter("poll")
    assert isinstance(ad, SimulationAdapter)


def test_pbg_adapter_from_builder():
    """from_builder classmethod works; poll mode returns correct y."""

    def build_composite(sample, out_uri=None):
        core = _core()
        doc = _document()
        doc["state"]["sim"]["config"]["gain"] = sample["gain"]
        doc["state"]["sim"]["config"]["offset"] = sample["offset"]
        # For poll mode the builder ignores out_uri; that's fine.
        return Composite(doc, core=core)

    ad = PbgAdapter.from_builder(
        build_composite=build_composite,
        param_bounds={"gain": (0.0, 5.0), "offset": (0.0, 10.0)},
        observables=["y"],
        duration=3.0,
        interval=1.0,
        emitter="poll",
        core=_core(),
    )
    X = np.array([[2.0, 0.0]])  # y = 4.0
    Y = ad.evaluate_batch(X)
    assert Y.shape == (1, 1)
    np.testing.assert_allclose(Y[:, 0], [4.0], atol=1e-6)
