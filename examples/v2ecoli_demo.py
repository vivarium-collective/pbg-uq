"""v2ecoli forward-UQ acceptance demo.

Demonstrates the full pbg-uq pipeline against live v2ecoli whole-cell
simulations:
  sample → PCE surrogate → Sobol indices → HTML report

Three translation-related parameters from the PolypeptideElongation process
are varied; three mass/growth observables are extracted via the XArray emitter
(run_multigen_xarray → zarr store → xarray DataTree) after each run.

Run with v2ecoli's venv from the v2ecoli repo root::

    cd /Users/eranagmon/code/v2ecoli
    time .venv/bin/python /Users/eranagmon/code/pbg-uq/examples/v2ecoli_demo.py

Expected result: ``gtpPerElongation`` and ``basal_elongation_rate`` dominate
``ribosomeElongationRate`` for the ``instantaneous_growth_rate`` output
(GTP cost has a strong inverse effect on growth; elongation rate sets the
production ceiling).

Observable collection: XArray emitter (run_multigen_xarray + xarray DataTree).
Data is written to a per-sample temp zarr store, read back as time-means, then
the store is deleted.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Silence non-essential warnings from the large composite build
# ---------------------------------------------------------------------------
logging.getLogger("v2ecoli").setLevel(logging.WARNING)
logging.getLogger("process_bigraph").setLevel(logging.WARNING)
os.environ.setdefault("POLARS_MAX_THREADS", "1")

# ---------------------------------------------------------------------------
# v2ecoli setup — pre-load once at module level (lru_cache reuse)
# ---------------------------------------------------------------------------
_CACHE_DIR = "/Users/eranagmon/code/v2ecoli/out/cache"
_N_STEPS = 60   # simulation seconds per sample; chunk=10 → 6 emits per run
_CHUNK = 10     # ticks between emitter updates; must be < _N_STEPS

print("[v2ecoli_demo] Loading v2ecoli modules…", flush=True)
_t_import = time.perf_counter()

from v2ecoli.core import build_core, load_cache_bundle          # noqa: E402
from v2ecoli.composites.baseline import baseline                 # noqa: E402
from v2ecoli.composites._helpers import set_null_emitter_override  # noqa: E402
from v2ecoli.library.xarray_run import view_from_emit_paths, run_multigen_xarray  # noqa: E402
from process_bigraph import Composite                             # noqa: E402

print(f"[v2ecoli_demo] Loading cache bundle from {_CACHE_DIR}…", flush=True)
_bundle = load_cache_bundle(_CACHE_DIR)       # memoised by lru_cache
_core = build_core()                          # shared across all samples

print(f"[v2ecoli_demo] Setup done ({time.perf_counter() - _t_import:.1f}s).",
      flush=True)

# ---------------------------------------------------------------------------
# pbg-uq imports
# ---------------------------------------------------------------------------
from pbg_uq import ForwardUQ, CallableAdapter   # noqa: E402

# ---------------------------------------------------------------------------
# UQ parameter space (3 PolypeptideElongation scalars)
# ---------------------------------------------------------------------------
PARAM_NAMES = [
    "ribosomeElongationRate",   # aa/s; default ~17.4
    "basal_elongation_rate",    # aa/s; default 22.0
    "gtpPerElongation",         # GTP molecules per amino acid; default 4.2
]

BOUNDS = np.array([
    [12.0, 23.0],   # ribosomeElongationRate
    [15.0, 28.0],   # basal_elongation_rate
    [2.5,   6.5],   # gtpPerElongation
])

OBS_NAMES = ["dry_mass", "cell_mass", "instantaneous_growth_rate"]

# XArray view — built once from dotted emit paths (listeners.mass.*)
_VIEW = view_from_emit_paths([
    "listeners.mass.dry_mass",
    "listeners.mass.cell_mass",
    "listeners.mass.instantaneous_growth_rate",
])

# Fixed metadata for the zarr partition; store_path is per-sample so no clash
_METADATA_BASE = {
    "experiment_id": "uq_sample",
    "variant": 0,
    "lineage_seed": 0,
    "time_step": 1.0,
    "max_duration": float(_N_STEPS),
}


# ---------------------------------------------------------------------------
# Helper: read time-means from a run_multigen_xarray zarr store
# ---------------------------------------------------------------------------
_OBS_LEAVES = ["dry_mass", "cell_mass", "instantaneous_growth_rate"]


def _read_store_means(store_path: str) -> np.ndarray:
    """Time-mean of each mass observable from the run_multigen_xarray zarr store.

    Delegates to ``pbg_uq.emit.read_run`` (RunReader-based, with suffix matching
    onto the emitter's partition-prefixed names). RunReader now resolves time
    coords at the partition path, so this works directly (pbg-emitters PR #10).
    """
    from pbg_uq.emit import read_run

    agg = read_run(store_path, _OBS_LEAVES)
    return np.array([agg[o] for o in _OBS_LEAVES])


# ---------------------------------------------------------------------------
# Single-sample evaluator
# ---------------------------------------------------------------------------
def _run_one(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Build and run one v2ecoli composite; return observable vector.

    Uses ``config_overrides`` on ``baseline()`` so the parameter mutation is
    applied to the pre-loaded cache bundle without any disk I/O on the ParCa
    side.  The simulation data is collected via an external XArrayEmitter
    (run_multigen_xarray) written to a per-sample temp zarr store.
    """
    config_overrides = {
        "ecoli-polypeptide-elongation.ribosomeElongationRate": float(x[0]),
        "ecoli-polypeptide-elongation.basal_elongation_rate":  float(x[1]),
        "ecoli-polypeptide-elongation.gtpPerElongation":       float(x[2]),
    }

    # Suppress default ParquetEmitter; the null-emitter flag is read by
    # baseline() during state construction — restore it after the build.
    set_null_emitter_override(True)
    try:
        doc = baseline(core=_core, seed=seed, bundle=_bundle,
                       config_overrides=config_overrides)
    finally:
        set_null_emitter_override(False)

    composite = Composite(doc, core=_core)

    # Per-sample temp zarr store — deleted after reading
    tmp_dir = tempfile.mkdtemp(prefix="v2ecoli_uq_")
    store_path = os.path.join(tmp_dir, "run.zarr")
    try:
        run_multigen_xarray(
            composite,
            store_path=store_path,
            view=_VIEW,
            metadata_base=_METADATA_BASE,
            max_steps=_N_STEPS,
            max_generations=1,
            chunk=_CHUNK,
        )
        return _read_store_means(store_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Batch evaluator (called by CallableAdapter)
# ---------------------------------------------------------------------------
def evaluate(X: np.ndarray) -> np.ndarray:
    """Run X.shape[0] v2ecoli simulations; return (n, 3) observable matrix.

    Catches per-sample failures gracefully: failed rows are filled with
    the last successful row (or zeros) so PCE fitting can continue.
    """
    n = X.shape[0]
    Y = np.zeros((n, len(OBS_NAMES)), dtype=float)
    last_good: np.ndarray | None = None

    for i, x in enumerate(X):
        t0 = time.perf_counter()
        try:
            y = _run_one(x, seed=i)
            Y[i] = y
            last_good = y.copy()
            print(f"  sample {i+1}/{n} done in {time.perf_counter()-t0:.1f}s  "
                  f"dry={y[0]:.2f} cell={y[1]:.2f} igr={y[2]:.5g}", flush=True)
        except Exception as exc:
            fallback = last_good if last_good is not None else np.zeros(len(OBS_NAMES))
            Y[i] = fallback
            print(f"  sample {i+1}/{n} FAILED ({exc!r}); using fallback",
                  flush=True)

    return Y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()

    sim = CallableAdapter(
        evaluate,
        param_names=PARAM_NAMES,
        bounds=BOUNDS,
        obs_names=OBS_NAMES,
    )

    uq = ForwardUQ(
        sim=sim,
        n_samples=8,
        n_test=2,
        order=2,
        cache_dir="./uq_cache_v2ecoli",
    )

    # ── Stage 1: sample ────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Stage 1: Sampling (8 train + 2 test) …", flush=True)
    t1 = time.perf_counter()
    uq.sample()
    print(f"Sampling done in {time.perf_counter()-t1:.1f}s", flush=True)

    # ── Stage 2: PCE + Sobol ───────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Stage 2: PCE fitting + Sobol indices …", flush=True)
    t2 = time.perf_counter()
    result = uq.quantify()
    print(f"PCE done in {time.perf_counter()-t2:.1f}s", flush=True)

    # ── Print top Sobol indices ────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Top-3 Sobol total-order indices:")
    top3 = result.sobol.select(n=3)
    for name, val in top3:
        print(f"  {name}: {val:.4f}")

    # ── Stage 3: HTML report ───────────────────────────────────────────────
    report_path = "v2ecoli_uq_report.html"
    uq.report(report_path)
    size = os.path.getsize(report_path)
    print(f"\nReport → {report_path}  ({size:,} bytes)")

    t_total = time.perf_counter() - t_start
    print(f"\nTotal wall time: {t_total/60:.1f} min  ({t_total:.0f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
