"""Pure PCE/Sobol math core — lifted from uqEcoli/uq/workflow.py.

No vEcoli or simulator dependencies. All functions operate on plain
numpy arrays.  Public entry point: ``fit_pce_and_sobol``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pytuq.lreg.anl import anl  # type: ignore[import-untyped]
from pytuq.lreg.bcs import bcs  # type: ignore[import-untyped]
from pytuq.lreg.lreg import lsq  # type: ignore[import-untyped]
from pytuq.rv.pcrv import PCRV  # type: ignore[import-untyped]
from pytuq.utils.mindex import get_mi  # type: ignore[import-untyped]

from pbg_uq.results import SobolIndices, PCESurrogate, UQPCResult

logger = logging.getLogger(__name__)


# ── Step 1: Input PC setup ──────────────────────────────────────────


def _setup_input_pc(
    bounds: np.ndarray,
) -> tuple[PCRV, np.ndarray, int]:
    """Set up the input PC object from parameter bounds.

    Equivalent to ``uq_pc.py`` lines that handle ``--pdom``:
    uniform parameters → Legendre (LU) basis, order 1.

    The PC coefficients encode the affine map from germ space [-1, 1]
    to the physical domain [lb, ub]:

        x_phys = midpoint + half_range * xi

    where xi ∈ [-1, 1] is the germ variable.

    Args:
        bounds: Parameter bounds, shape (n_params, 2).

    Returns:
        Tuple of (PCRV object, PC coefficient matrix, stochastic dim).
    """
    n_params = bounds.shape[0]
    in_pcdim = n_params
    in_pcord = 1
    pc_type = "LU"  # Legendre — uniform priors (RFC006 §4)

    # Build PC coefficients: row 0 = midpoints, rows 1..n = diag(half_ranges)
    midpoints = 0.5 * (bounds[:, 1] + bounds[:, 0])
    half_ranges = 0.5 * (bounds[:, 1] - bounds[:, 0])
    pcf_all = np.vstack((midpoints, np.diag(half_ranges)))

    # Construct PCRV
    mi = get_mi(in_pcord, in_pcdim)
    pc = PCRV(in_pcdim, n_params, pc_type, mi=mi, cfs=pcf_all.T)

    return pc, pcf_all, in_pcdim


# ── Step 2: Generate / load training samples ────────────────────────


def _generate_training_samples(
    pc: PCRV,
    n_samples: int,
    in_pcdim: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate training samples in germ and physical spaces via PCRV.sampleGerm().

    Equivalent to ``uq_pc.py`` random sampling path (``--sampl rand``).
    Uses PyTUQ-native random sampling from the germ measure — no quadrature.

    Args:
        pc: Input PCRV object.
        n_samples: Number of training points.
        in_pcdim: Stochastic dimensionality.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (germ_train, X_train) — germ space and physical space.
    """
    if seed is not None:
        np.random.seed(seed)
    germ_train = pc.sampleGerm(n_samples)
    X_train = pc.evalPC(germ_train)

    logger.info(
        "Generated %d training samples (%d germ dims, %d physical params)",
        n_samples,
        in_pcdim,
        X_train.shape[1],
    )
    return germ_train, X_train


def _generate_test_samples(
    pc: PCRV,
    n_test: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random test samples for surrogate validation.

    Args:
        pc: Input PCRV object.
        n_test: Number of test points.
        seed: Random seed.

    Returns:
        Tuple of (germ_test, X_test).
    """
    if seed is not None:
        np.random.seed(seed)
    germ_test = pc.sampleGerm(n_test)
    X_test = pc.evalPC(germ_test)
    return germ_test, X_test


def _physical_to_germ(
    X: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Scale physical-space samples to germ space [-1, 1].

    Used when loading precomputed samples from ``PrecomputedCache``
    (offline regime), which are stored in physical space.

    Args:
        X: Samples in physical space, shape (n, d).
        bounds: Parameter bounds, shape (d, 2).

    Returns:
        Samples in germ space [-1, 1], shape (n, d).
    """
    lb, ub = bounds[:, 0], bounds[:, 1]
    span = ub - lb
    span = np.where(span > 0, span, 1.0)
    return 2.0 * (X - lb) / span - 1.0  # type: ignore[no-any-return]


# ── Step 4: Construct PC surrogates ─────────────────────────────────


def _fit_surrogate(
    germ_train: np.ndarray,
    Y_train: np.ndarray,
    polynomial_order: int,
    regression: str = "lsq",
    tolerance: float = 1e-3,
) -> tuple[PCRV, list[Any]]:
    """Fit PCE surrogate following the ``pc_fit`` workflow.

    This is the core of the UQPC workflow (step 4 in ``uq_pc.py``).
    We replicate ``pytuq.workflows.fits.pc_fit`` here so that we can
    retain the per-output linear regression objects (``linregs``) for
    prediction variance estimation — ``pc_fit`` itself only returns
    the PCRV.

    The workflow:
      1. Build multi-index for the output polynomial order
      2. Construct PCRV with Legendre (LU) basis
      3. Evaluate basis matrix at training points
      4. Per-output: instantiate regressor, fit coefficients
      5. Sync multi-indices and coefficients into PCRV
      6. Set PCRV evaluation function

    Args:
        germ_train: Training samples in germ space, shape (n_train, d).
        Y_train: Training outputs, shape (n_train, n_out).
        polynomial_order: Output PCE order.
        regression: Fitting method — 'lsq', 'bcs', or 'anl'.
        tolerance: BCS tolerance (only used when regression='bcs').

    Returns:
        Tuple of (output_pcrv, linregs) — the fitted PCRV and per-output
        linear regression objects.
    """
    n_train, n_dim = germ_train.shape
    n_out = Y_train.shape[1]

    logger.info(
        "Fitting PCE surrogate: order=%d, method=%s, n_train=%d, n_outputs=%d",
        polynomial_order,
        regression,
        n_train,
        n_out,
    )

    # Step 1-2: Multi-index + PCRV
    mindex = get_mi(polynomial_order, n_dim)
    pcrv = PCRV(n_out, n_dim, "LU", mi=mindex)

    # Step 3: Basis matrix
    Amat = pcrv.evalBases(germ_train, 0)

    # Step 4: Per-output fitting
    mindices_list: list[np.ndarray] = []
    cfs_list: list[np.ndarray] = []
    linregs: list[Any] = []

    for j in range(n_out):
        logger.debug("Fitting output %d / %d", j + 1, n_out)

        if regression == "bcs":
            lreg_obj = bcs(eta=tolerance)
        elif regression == "anl":
            lreg_obj = anl()
        elif regression == "lsq":
            lreg_obj = lsq()
        else:
            raise ValueError(f"Unknown regression method: {regression!r}. Must be 'lsq', 'bcs', or 'anl'.")

        lreg_obj.fita(Amat, Y_train[:, j])
        mindices_list.append(mindex[lreg_obj.used, :])
        cfs_list.append(lreg_obj.cf)
        linregs.append(lreg_obj)

    # Step 5-6: Sync coefficients and set evaluation function
    pcrv.setMiCfs(mindices_list, cfs_list)
    pcrv.setFunction()

    return pcrv, linregs


def _predict_and_variance(
    output_pcrv: PCRV,
    linregs: list[Any],
    germ: np.ndarray,
    n_outputs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict at given germ points and compute prediction variance.

    Equivalent to ``uq_pc.py`` lines after ``pc_fit`` call.

    Args:
        output_pcrv: Fitted output PCRV.
        linregs: Per-output linear regression objects.
        germ: Germ-space samples, shape (n, d).
        n_outputs: Number of output columns.

    Returns:
        Tuple of (Y_pc, Y_pc_std) — predictions and std deviations.
    """
    n = germ.shape[0]
    Y_pc = output_pcrv.function(germ)

    Y_pc_var = np.empty((n, n_outputs))
    for j, lreg in enumerate(linregs):
        basis_mat = output_pcrv.evalBases(germ, j)
        Y_pc_var[:, j] = lreg.predicta(basis_mat, msc=1)[1]

    return Y_pc, np.sqrt(Y_pc_var)


# ── Step 5: Compute relative errors ────────────────────────────────


def _compute_relative_errors(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> np.ndarray:
    """Per-output relative L2 error between true and predicted values.

    Equivalent to ``uq_pc.py`` step 5.

    Args:
        Y_true: Ground truth, shape (n, n_out).
        Y_pred: PCE predictions, shape (n, n_out).

    Returns:
        Relative errors, shape (n_out,).
    """
    norms = np.linalg.norm(Y_true, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return np.linalg.norm(Y_true - Y_pred, axis=0) / norms  # type: ignore[no-any-return]


def _k_fold_cv_error(
    germ: np.ndarray,
    Y: np.ndarray,
    polynomial_order: int,
    regression: str = "lsq",
    tolerance: float = 1e-3,
    n_folds: int = 5,
    seed: int | None = 42,
) -> tuple[np.ndarray, int]:
    """Per-output relative L2 error from k-fold cross-validation.

    Used as a generalization measure when no independent held-out test
    set is available (BYO mode, or default mode with ``--n-test 0``).
    PyTUQ-idiomatic: same fit machinery as `_fit_surrogate`, just looped
    over folds — basis matrix per fold, regression backend per output,
    predict on held-out rows, accumulate residuals.

    Skips (returns zero-length array, ``n_folds=0``) when ``germ.shape[0]``
    is too small to leave any rows out for validation:
    ``n_samples < 2 * basis_size`` is the standard cutoff (under that, the
    in-fold training set is itself under-fit and the CV error is noise).

    Returns:
        (relerr_cv, n_folds_actually_run) — per-output mean CV relative
        error across folds, plus the fold count actually used (0 if skipped).
    """
    n_samples, n_dim = germ.shape
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_out = Y.shape[1]

    mindex = get_mi(polynomial_order, n_dim)
    basis_size = mindex.shape[0]

    # Don't run CV when even a single fold's training set is too small to
    # honestly fit the basis. Cutoff: 2× basis_size for the training set —
    # so n_samples > 2*basis_size * n_folds/(n_folds-1). For 5-fold that's
    # roughly 2.5 × basis_size as a floor on n_samples.
    min_samples = int(np.ceil(2.0 * basis_size * n_folds / (n_folds - 1)))
    if n_samples < min_samples:
        return np.array([], dtype=float), 0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[: n_samples % n_folds] += 1

    Y_pred_oof = np.zeros_like(Y, dtype=float)
    start = 0
    for fold_size in fold_sizes:
        test_idx = perm[start : start + fold_size]
        train_idx = np.setdiff1d(perm, test_idx, assume_unique=False)
        start += fold_size

        germ_tr, Y_tr = germ[train_idx], Y[train_idx]
        germ_te = germ[test_idx]

        pcrv_fold = PCRV(n_out, n_dim, "LU", mi=mindex)
        Amat_tr = pcrv_fold.evalBases(germ_tr, 0)

        mindices_list: list[np.ndarray] = []
        cfs_list: list[np.ndarray] = []
        for j in range(n_out):
            if regression == "bcs":
                lreg_obj = bcs(eta=tolerance)
            elif regression == "anl":
                lreg_obj = anl()
            else:  # lsq (default)
                lreg_obj = lsq()
            lreg_obj.fita(Amat_tr, Y_tr[:, j])
            mindices_list.append(mindex[lreg_obj.used, :])
            cfs_list.append(lreg_obj.cf)

        pcrv_fold.setMiCfs(mindices_list, cfs_list)
        pcrv_fold.setFunction()
        Y_pred_oof[test_idx] = pcrv_fold.function(germ_te)

    return _compute_relative_errors(Y, Y_pred_oof), n_folds


# ── Step 6: Compute Sobol indices ───────────────────────────────────


def _compute_sobol(
    output_pcrv: PCRV,
    parameter_names: list[str],
    Y_train: np.ndarray,
) -> SobolIndices:
    """Compute Sobol indices from the fitted PCE.

    Equivalent to ``uq_pc.py`` step 6.  Uses PCRV's analytical Sobol
    computation (Sudret 2008) — no additional model evaluations needed.

    Main (first-order) indices measure each parameter's independent
    contribution to output variance.  Total-order indices include all
    interaction terms involving that parameter.

    For multi-output models, Sobol indices are variance-weighted across
    outputs to produce a single set of parameter importances.

    Args:
        output_pcrv: Fitted output PCRV with synced coefficients.
        parameter_names: Parameter names for labeling.
        Y_train: Training outputs (for variance weighting).

    Returns:
        SobolIndices with main, total, and joint indices.
    """
    allsens_main = output_pcrv.computeSens()  # (n_out, n_params)
    allsens_total = output_pcrv.computeTotSens()  # (n_out, n_params)
    allsens_joint = output_pcrv.computeJointSens()  # (n_out, n_params, n_params)

    n_outputs = Y_train.shape[1]

    logger.info(
        "Sobol indices computed: sum(main)=%s, sum(total)=%s",
        np.sum(allsens_main, axis=1),
        np.sum(allsens_total, axis=1),
    )

    # Variance-weighted aggregation across outputs
    if n_outputs == 1:
        first_order = allsens_main[0]
        total_order = allsens_total[0]
        second_order = allsens_joint[0]
    else:
        output_vars = np.var(Y_train, axis=0)
        total_var = output_vars.sum()
        if total_var > 0:
            weights = output_vars / total_var
        else:
            weights = np.ones(n_outputs) / n_outputs

        first_order = np.zeros(allsens_main.shape[1])
        total_order = np.zeros(allsens_total.shape[1])
        second_order = np.zeros(allsens_joint.shape[1:])
        for j in range(n_outputs):
            first_order += weights[j] * allsens_main[j]
            total_order += weights[j] * allsens_total[j]
            second_order += weights[j] * allsens_joint[j]

    return SobolIndices(
        first_order=first_order,
        total_order=total_order,
        second_order=second_order,
        parameter_names=parameter_names,
    )


def _build_surrogate(
    output_pcrv: PCRV,
    polynomial_order: int,
    n_params: int,
    n_outputs: int,
    bounds: np.ndarray,
) -> PCESurrogate:
    """Build an exportable PCESurrogate from the fitted PCRV.

    Args:
        output_pcrv: Fitted output PCRV.
        polynomial_order: PCE polynomial order.
        n_params: Number of input parameters.
        n_outputs: Number of outputs.
        bounds: Parameter bounds, shape (n_params, 2).

    Returns:
        PCESurrogate ready for export / prediction.
    """
    coefficients = output_pcrv.coefs[0] if output_pcrv.coefs else np.zeros(1)
    multi_indices = output_pcrv.mindices[0] if output_pcrv.mindices else np.zeros((1, n_params), dtype=int)

    return PCESurrogate(
        coefficients=coefficients,
        multi_indices=multi_indices,
        basis_type="legendre",
        polynomial_order=polynomial_order,
        input_dim=n_params,
        output_dim=n_outputs,
        input_bounds=bounds,
    )


# ── Public entry point ───────────────────────────────────────────────


def fit_pce_and_sobol(
    X: np.ndarray,
    Y: np.ndarray,
    bounds: np.ndarray,
    polynomial_order: int = 2,
    regression: str = "lsq",
    parameter_names: list[str] | None = None,
    X_test: np.ndarray | None = None,
    Y_test: np.ndarray | None = None,
    bootstrap: int = 0,
    seed: int = 42,
) -> UQPCResult:
    """Fit a PCE surrogate to (X, Y) and compute analytic Sobol indices.

    Pure UQPC math (PyTUQ). No simulator. X (n,d) physical-space samples,
    Y (n,m) outputs, bounds (d,2) per-parameter [low, high].
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_params = bounds.shape[0]
    n_outputs = Y.shape[1]
    if parameter_names is None:
        parameter_names = [f"x{i}" for i in range(n_params)]

    germ_train = _physical_to_germ(X, bounds)
    output_pcrv, linregs = _fit_surrogate(
        germ_train, Y, polynomial_order, regression=regression
    )

    Y_train_pc, Y_train_pc_std = _predict_and_variance(
        output_pcrv, linregs, germ_train, n_outputs
    )
    relerr_train = _compute_relative_errors(Y, Y_train_pc)

    germ_test = relerr_test = Y_test_pc = Y_test_pc_std = None
    relerr_cv = None
    cv_n_folds = 0
    if X_test is not None and Y_test is not None:
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)
        germ_test = _physical_to_germ(X_test, bounds)
        Y_test_pc, Y_test_pc_std = _predict_and_variance(
            output_pcrv, linregs, germ_test, n_outputs
        )
        relerr_test = _compute_relative_errors(Y_test, Y_test_pc)
    else:
        relerr_cv, cv_n_folds = _k_fold_cv_error(
            germ_train, Y, polynomial_order, regression=regression, seed=seed
        )

    sobol = _compute_sobol(output_pcrv, parameter_names, Y)
    surrogate = _build_surrogate(
        output_pcrv, polynomial_order, n_params, n_outputs, bounds
    )

    first_ci = total_ci = None
    # bootstrap CI skipped: _bootstrap_sobol_cis in source has incompatible
    # signature (no bounds arg, uses n_bootstrap not n_resamples) — fallback

    return UQPCResult(
        sobol=sobol, surrogate=surrogate, pcrv=output_pcrv, linregs=linregs,
        germ_train=germ_train, X_train=X, Y_train=Y,
        Y_train_pc=Y_train_pc, Y_train_pc_std=Y_train_pc_std,
        relerr_train=relerr_train,
        germ_test=germ_test, X_test=X_test, Y_test=Y_test,
        Y_test_pc=Y_test_pc, Y_test_pc_std=Y_test_pc_std,
        relerr_test=relerr_test, relerr_cv=relerr_cv, cv_n_folds=cv_n_folds,
        sobol_first_order_ci=first_ci, sobol_total_order_ci=total_ci,
        n_bootstrap=bootstrap,
    )
