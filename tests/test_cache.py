import numpy as np
from pbg_uq.cache import PrecomputedCache


def test_cache_roundtrip(tmp_path):
    X = np.arange(12, dtype=float).reshape(6, 2)
    Y = np.arange(18, dtype=float).reshape(6, 3)
    ts = [np.ones((4, 3)) * i for i in range(6)]
    meta = [{"generation": np.array([0, 0, 1, 1]),
             "lineage_seed": np.array([i, i, i, i])} for i in range(6)]
    cache = PrecomputedCache(
        cache_dir=tmp_path, X=X, Y=Y, parameter_names=["a", "b"],
        metadata={"order": 2}, Y_timeseries=ts, Y_timeseries_meta=meta,
        X_test=X[:2], Y_test=Y[:2],
    )
    cache.save()

    loaded = PrecomputedCache.load(tmp_path)
    np.testing.assert_array_equal(loaded.X, X)
    np.testing.assert_array_equal(loaded.Y, Y)
    assert loaded.parameter_names == ["a", "b"]
    assert loaded.metadata["order"] == 2
    assert len(loaded.Y_timeseries) == 6
    assert loaded.Y_timeseries_meta[2]["generation"].tolist() == [0, 0, 1, 1]
    np.testing.assert_array_equal(loaded.X_test, X[:2])
