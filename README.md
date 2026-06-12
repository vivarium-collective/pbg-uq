# pbg-uq

Portable forward uncertainty quantification (Polynomial Chaos Expansion + Sobol sensitivity indices) for [process-bigraph](https://github.com/vivarium-collective/process-bigraph) composites and plain Python callables. Lifted from uqEcoli and decoupled from vEcoli/libuq so it can wrap any pbg composite — or any `f(X)->Y` — without modification. Observable collection defaults to the XArray emitter (zarr, via pbg-emitters).

## Install

```bash
uv pip install -e .
```

Dependencies: `pytuq`, `process-bigraph`, `bigraph-schema`, `pbg-emitters[xarray]`.

## Quickstart — PbgAdapter

Wrap a process-bigraph document with parameter ranges; `ForwardUQ` handles sampling, PCE fitting, and reporting.

```python
from pbg_uq import ForwardUQ
from pbg_uq.adapters.pbg import PbgAdapter

sim = PbgAdapter(
    document=_document(),       # {"state": {...}}
    core=_core(),               # bigraph_schema Core with processes registered
    params={
        "toy/config/k1": (0.5, 2.0),
        "toy/config/k2": (0.5, 2.0),
    },
    observables=["signal"],
    duration=3.0,
    interval=1.0,
    emitter="xarray",           # default; "poll" also accepted
)

uq = ForwardUQ(sim=sim, n_samples=40, n_test=10, order=3, cache_dir="./uq_cache")
uq.sample()
result = uq.quantify()
uq.report("out.html")
```

`result.sobol.select(n=2)` returns the top-n `(name, total_order_index)` pairs.

See `examples/synthetic_demo.py` for a complete runnable version.

## Quickstart — CallableAdapter

Wrap any `f(X) -> Y` (X shape `(n, d)`, Y shape `(n, m)`) without a pbg composite:

```python
import numpy as np
from pbg_uq import ForwardUQ, CallableAdapter

def ishigami(X, a=7.0, b=0.1):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)).reshape(-1, 1)

bounds = np.array([[-np.pi, np.pi]] * 3)
sim = CallableAdapter(ishigami, param_names=["x1", "x2", "x3"], bounds=bounds)

uq = ForwardUQ(sim=sim, n_samples=600, n_test=100, order=8, cache_dir="./uq_cache")
uq.sample()
result = uq.quantify()
```

## Bring your own simulator

`PbgAdapter` has two construction modes:

**Document-path mode** (default, shown above): the adapter deepcopies the document, injects each sampled parameter at the given `"/"` -joined state-tree path (e.g. `"toy/config/k1"`), attaches an emitter, builds the `Composite`, and runs it. Use this when all parameters are leaves in the bigraph document.

**Build-hook mode** (`from_builder`): supply a factory `build_composite(sample, out_uri)` that builds and returns a ready `Composite`. The adapter calls it once per sample, passing the sampled parameter dict and the zarr `out_uri` for the XArray emitter. Use this when parameters live outside the document — for example, in a cached simulation bundle like v2ecoli's ParCa output:

```python
def build_composite(sample: dict, out_uri: str | None = None):
    overrides = {"ecoli-polypeptide-elongation.gtpPerElongation": sample["gtp"]}
    doc = baseline(core=core, bundle=bundle, config_overrides=overrides)
    # wire your own XArray emitter using out_uri, return Composite(doc, core=core)
    ...

sim = PbgAdapter.from_builder(
    build_composite=build_composite,
    param_bounds={"gtp": (2.5, 6.5)},
    observables=["instantaneous_growth_rate"],
    duration=60.0,
)
```

Set `emitter="poll"` if you prefer reading `comp.state` after each step instead of zarr.

## Observable collection

The default emitter is `"xarray"` (pbg-emitters `XArrayEmitter`): a zarr store is written per sample to a temp directory, then read back as time-means before cleanup. Observable names are `"/"` -joined state-tree paths relative to the composite root (e.g. `"signal"` or `"listeners/mass/dry_mass"`). For array-valued observables each element becomes a separate column (`obs_0`, `obs_1`, …).

## Examples

- `examples/synthetic_demo.py` — toy pbg composite, verifies k1 dominates k2 in Sobol total-order indices.
- `examples/v2ecoli_demo.py` — whole-cell v2ecoli; run in v2ecoli's venv from its repo root (`cd /path/to/v2ecoli && .venv/bin/python /path/to/pbg-uq/examples/v2ecoli_demo.py`). Expected result: `gtpPerElongation ≈ 0.88` dominates `instantaneous_growth_rate`.
