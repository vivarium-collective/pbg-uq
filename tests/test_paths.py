import numpy as np
from pbg_uq.adapters.base import resolve_path, set_path, read_observable


def test_resolve_nested_scalar():
    state = {"agents": {"0": {"listeners": {"mass": {"dry_mass": 3.5}}}}}
    assert resolve_path(state, "agents/0/listeners/mass/dry_mass") == 3.5


def test_resolve_missing_returns_default():
    assert resolve_path({"a": {}}, "a/b/c", default=0.0) == 0.0


def test_set_path_mutates_leaf():
    d = {"p": {"config": {"k": 1.0}}}
    set_path(d, "p/config/k", 9.0)
    assert d["p"]["config"]["k"] == 9.0


def test_read_observable_scalar_and_array():
    state = {"listeners": {"mass": {"dry_mass": 2.0},
                           "counts": np.array([1.0, 2.0, 3.0])}}
    assert read_observable(state, "listeners/mass/dry_mass") == 2.0
    arr = read_observable(state, "listeners/counts")
    np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))
