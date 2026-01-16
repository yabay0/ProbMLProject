from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, brute
from kernels_np import rbf_k


def chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve (L L^T) x = b for x, where L is lower-triangular Cholesky factor.
    """
    return np.linalg.solve(L.T, np.linalg.solve(L, b))

def logdet_from_chol(L: np.ndarray) -> float:
    """Compute log det(L L^T) from Cholesky factor L."""
    return 2.0 * float(np.sum(np.log(np.diag(L))))

def mll_standard(y, X, ell, sigma_f, sigma_y, jitter=1e-8) -> float:
    """
    Log marginal likelihood for standard GP regression:
      y ~ N(0, K + sigma_y^2 I)
    """
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X).reshape(-1)
    n = y.shape[0]

    K = rbf_k(X, X, ell, sigma_f)
    Ky = K + (sigma_y**2 + jitter) * np.eye(n)

    try:
        L = np.linalg.cholesky(Ky)
    except np.linalg.LinAlgError:
        print("Cholesky failed inside mll_standard.")
        return -np.inf

    alpha = chol_solve(L, y)
    mll = -0.5 * float(y.T @ alpha) - 0.5 * logdet_from_chol(L) - 0.5 * n * np.log(2.0*np.pi)
    return mll

def fit_b1(
    X, 
    y, 
    sigma_y_fixed: float | None, 
    x0=(0.2, 1.0, 0.05),
):
    """
    Maximize MLL in log-space.

    If sigma_y_fixed is None:
      optimize (ell, sigma_f, sigma_y) 
    Else:
      optimize (ell, sigma_f) with sigma_y fixed.

    Returns dict with fitted parameters (stds).
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if sigma_y_fixed is None:
        def obj(logp):
            ell, sigma_f, sigma_y = np.exp(logp)
            return -mll_standard(y, X, ell, sigma_f, sigma_y)
        
        res = minimize(
            obj,
            np.log(np.array(x0)),
            method="L-BFGS-B",
        )
        ell, sigma_f, sigma_y = np.exp(res.x)
    else:
        sigma_y = float(sigma_y_fixed)

        def obj(logp):
            ell, sigma_f = np.exp(logp)
            return -mll_standard(y, X, ell, sigma_f, sigma_y)
        
        res = minimize(
            obj,
            np.log(np.array([x0[0], x0[1]])),
            method="L-BFGS-B",
        )
        ell, sigma_f = np.exp(res.x)

    best_mll = mll_standard(y, X, float(ell), float(sigma_f), float(sigma_y))

    return {
        "ell": float(ell), 
        "sigma_f": float(sigma_f), 
        "sigma_y": float(sigma_y), 
        "mll": float(best_mll),
        "opt": res
    }

def grid_search_b1(
    X,
    y,
    sigma_y_fixed: float | None,
    *,
    ell_range=(1e-2, 2.0),
    sigma_f_range=(1e-2, 3.0),
    sigma_y_range=(1e-3, 1.0),
    grid_sizes=(35, 35, 25),
    jitter=1e-8,
):
    """
    Coarse grid search (SciPy brute) over hyperparameters in log-space.
    Intended as a sanity check for L-BFGS-B.

    If sigma_y_fixed is None:
      grid over (ell, sigma_f, sigma_y)
    Else:
      grid over (ell, sigma_f) with sigma_y fixed.

    grid_sizes = (n_ell, n_sigma_f, n_sigma_y). If sigma_y fixed, n_sigma_y ignored.
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)

    log_ell_min, log_ell_max = np.log(ell_range[0]), np.log(ell_range[1])
    log_sf_min, log_sf_max = np.log(sigma_f_range[0]), np.log(sigma_f_range[1])
    log_sy_min, log_sy_max = np.log(sigma_y_range[0]), np.log(sigma_y_range[1])

    n_ell, n_sf, n_sy = grid_sizes

    if sigma_y_fixed is None:
        def obj(logp):
            ell, sigma_f, sigma_y = np.exp(logp)
            return -mll_standard(y, X, ell, sigma_f, sigma_y, jitter=jitter)

        rranges = (
            slice(log_ell_min, log_ell_max, complex(n_ell)),
            slice(log_sf_min, log_sf_max, complex(n_sf)),
            slice(log_sy_min, log_sy_max, complex(n_sy)),
        )
        xopt, fval, *_ = brute(obj, rranges, full_output=True, finish=None)
        ell, sigma_f, sigma_y = np.exp(xopt)
    else:
        sigma_y = float(sigma_y_fixed)

        def obj(logp):
            ell, sigma_f = np.exp(logp)
            return -mll_standard(y, X, ell, sigma_f, sigma_y, jitter=jitter)

        rranges = (
            slice(log_ell_min, log_ell_max, complex(n_ell)),
            slice(log_sf_min, log_sf_max, complex(n_sf)),
        )
        xopt, fval, *_ = brute(obj, rranges, full_output=True, finish=None)
        ell, sigma_f = np.exp(xopt)

    return {
        "ell": float(ell),
        "sigma_f": float(sigma_f),
        "sigma_y": float(sigma_y),
        "mll": float(-fval),
        "grid_sizes": tuple(grid_sizes),
        "ranges": {"ell": ell_range, "sigma_f": sigma_f_range, "sigma_y": sigma_y_range},
    }

def posterior_latent_f(X_train, y_train, X_test, ell, sigma_f, sigma_y, jitter=1e-8):
    """
    Posterior over latent f(X_test) (NOT y*).
    So we do NOT add sigma_y^2 to the predictive variance.
    """
    X_train = np.asarray(X_train).reshape(-1)
    y_train = np.asarray(y_train).reshape(-1, 1)
    X_test = np.asarray(X_test).reshape(-1)

    n = len(X_train)
    K = rbf_k(X_train, X_train, ell, sigma_f)
    Ky = K + (sigma_y**2 + jitter) * np.eye(n)
    L = np.linalg.cholesky(Ky)

    K_s = rbf_k(X_train, X_test, ell, sigma_f)
    K_ss = rbf_k(X_test, X_test, ell, sigma_f)

    alpha = chol_solve(L, y_train)
    mu = (K_s.T @ alpha).ravel()

    v = np.linalg.solve(L, K_s)
    cov = K_ss - v.T @ v
    var = np.clip(np.diag(cov), 0.0, np.inf)
    return mu, var
