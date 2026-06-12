"""Synthetic process-bigraph forward-UQ demo.

Proves the full pipeline (sample → quantify → report) on a tiny composite
using the default XArray emitter.

Expected result: k1 ranks above k2 in Sobol total-order indices because
  signal = k1**2 + 0.1*k2
so k1 drives the output strongly and k2 only weakly.
"""
from __future__ import annotations

from process_bigraph.composite import Process, Composite
from bigraph_schema import allocate_core

from pbg_uq.adapters.pbg import PbgAdapter
from pbg_uq.core import ForwardUQ


# ---------------------------------------------------------------------------
# Tiny synthetic process
# ---------------------------------------------------------------------------

class ToySim(Process):
    """signal = k1**2 + 0.1*k2  (overwrite; no delta)."""

    config_schema = {
        "k1": {"_type": "float", "_default": 1.0},
        "k2": {"_type": "float", "_default": 1.0},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {"signal": "overwrite[float]"}

    def update(self, state, interval):
        k1 = self.config["k1"]
        k2 = self.config["k2"]
        return {"signal": k1 ** 2 + 0.1 * k2}


# ---------------------------------------------------------------------------
# Document / core builders
# ---------------------------------------------------------------------------

def _core():
    core = allocate_core()
    core.register_link("ToySim", ToySim)
    return core


def _document():
    return {
        "state": {
            "toy": {
                "_type": "process",
                "address": "local:ToySim",
                "config": {"k1": 1.0, "k2": 1.0},
                "inputs": {},
                "outputs": {"signal": ["signal"]},
            },
            # Initial value; the emitter skips the t=0 fire so this
            # placeholder is never included in the time-mean.
            "signal": 0.0,
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sim = PbgAdapter(
        document=_document(),
        core=_core(),
        params={
            "toy/config/k1": (0.5, 2.0),
            "toy/config/k2": (0.5, 2.0),
        },
        observables=["signal"],
        duration=3.0,
        interval=1.0,
        emitter="xarray",  # default
    )

    uq = ForwardUQ(
        sim=sim,
        n_samples=40,
        n_test=10,
        order=3,
        cache_dir="./uq_cache_synth",
    )

    print("Sampling (40 runs) …")
    uq.sample()

    print("Quantifying (PCE + Sobol) …")
    result = uq.quantify()

    top2 = result.sobol.select(n=2)
    print("\nTop-2 Sobol total-order indices:")
    for name, val in top2:
        print(f"  {name}: {val:.4f}")

    report_path = "synthetic_uq_report.html"
    uq.report(report_path)

    import os
    size = os.path.getsize(report_path)
    print(f"\nReport written to {report_path!r}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
