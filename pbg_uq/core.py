"""ForwardUQ — sample → quantify → report over an injected SimulationAdapter."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from pbg_uq.cache import PrecomputedCache
from pbg_uq.results import UQPCResult
from pbg_uq.sampling import draw_samples
from pbg_uq.uqpc import fit_pce_and_sobol

logger = logging.getLogger(__name__)


class ForwardUQ:
    def __init__(self, sim, *, n_samples: int, n_test: int = 0, order: int = 2,
                 regression: str = "lsq", seed: int = 42,
                 cache_dir: str = "./uq_cache"):
        self.sim = sim
        self.n_samples = n_samples
        self.n_test = n_test
        self.order = order
        self.regression = regression
        self.seed = seed
        self.cache_dir = Path(cache_dir)
        self._cache: PrecomputedCache | None = None
        self._result: UQPCResult | None = None

    def sample(self) -> PrecomputedCache:
        """Expensive stage: draw PCRV samples, run the simulator, cache (X, Y)."""
        X, X_test = draw_samples(self.sim.bounds, self.n_samples, self.n_test, self.seed)
        Y = self.sim.evaluate_batch(X)
        Y_test = self.sim.evaluate_batch(X_test) if X_test is not None else None
        cache = PrecomputedCache(
            cache_dir=self.cache_dir,
            X=X,
            Y=Y,
            parameter_names=list(self.sim.param_names),
            metadata={
                "order": self.order,
                "regression": self.regression,
                "observable_names": list(self.sim.obs_names),
                "bounds": self.sim.bounds.tolist(),
            },
            X_test=X_test,
            Y_test=Y_test,
        )
        cache.save()
        self._cache = cache
        logger.info("Sampled %d (+%d test) -> %s", self.n_samples, self.n_test, self.cache_dir)
        return cache

    def _load_cache(self) -> PrecomputedCache:
        if self._cache is None:
            self._cache = PrecomputedCache.load(self.cache_dir)
        return self._cache

    def quantify(self, bootstrap: int = 0) -> UQPCResult:
        """Cheap stage: fit PCE + Sobol from the cache."""
        cache = self._load_cache()
        bounds = np.array(cache.metadata["bounds"], float)
        result = fit_pce_and_sobol(
            cache.X,
            cache.Y,
            bounds,
            polynomial_order=self.order,
            regression=self.regression,
            parameter_names=cache.parameter_names,
            X_test=cache.X_test,
            Y_test=cache.Y_test,
            bootstrap=bootstrap,
            seed=self.seed,
        )
        self._result = result
        return result

    def report(self, path: str) -> str:
        """Render the quantify result to a self-contained HTML file."""
        if self._result is None:
            raise RuntimeError("Call quantify() before report().")
        from pbg_uq.report import generate_html_report
        cache = self._load_cache()
        obs_names = cache.metadata.get("observable_names")
        html = generate_html_report(
            self._result,
            parameter_names=cache.parameter_names,
            observable_names=obs_names,
        )
        Path(path).write_text(html)
        logger.info("Report written to %s", path)
        return path
