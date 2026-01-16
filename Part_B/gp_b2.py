from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from kernels_np import rbf_k, rbf_dk_dx1, rbf_d2k_dx1dx2


def chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve (L L^T) x = b for x, where L is lower-triangular Cholesky factor.
    """
    return np.linalg.solve(L.T, np.linalg.solve(L, b))

def logdet_from_chol(L: np.ndarray) -> float:
    """Compute log det(L L^T) from Cholesky factor L."""
    return 2.0 * float(np.sum(np.log(np.diag(L))))

def C_delta_np(X, delta, ell, sigma_f, sigma_y, jitter=1e-8):
    """ Construct C_Δ matrix for noisy-input GP regression."""
    X = np.asarray(X).reshape(-1)
    delta = np.asarray(delta).reshape(-1)
    n = len(X)

    K  = rbf_k(X, X, ell, sigma_f)
    K1 = rbf_dk_dx1(X, X, ell, sigma_f)         
    K2 = rbf_d2k_dx1dx2(X, X, ell, sigma_f)

    D = np.diag(delta)

    C = K - D @ K1 - K1.T @ D + D @ K2 @ D + (sigma_y**2 + jitter) * np.eye(n)
    # ensure symmetry numerically
    C = 0.5 * (C + C.T)
    return C

def mll_b2(y, X, delta, ell, sigma_f, sigma_y, jitter=1e-8):
    """
    Log marginal likelihood for noisy-input GP regression:
      y ~ N(0, C_Δ)
    """
    y = np.asarray(y).reshape(-1, 1)
    C = C_delta_np(X, delta, ell, sigma_f, sigma_y, jitter=jitter)
    n = y.shape[0]

    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        print("Cholesky failed inside mll_b2.")
        return -np.inf
    
    alpha = chol_solve(L, y)
    mll = -0.5 * float(y.T @ alpha) - 0.5 * logdet_from_chol(L) - 0.5 * n * np.log(2.0*np.pi)
    return mll

def fit_b2(
    X, y, delta,
    x0=(0.2, 1.0, 0.05),
):
    """
    Maximize MLL in log-space. 
    Returns dict with fitted parameters.
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    delta = np.asarray(delta).reshape(-1)


    def obj(logp):
        ell, sigma_f, sigma_y = np.exp(logp)
        return -mll_b2(y, X, delta, ell, sigma_f, sigma_y)

    res = minimize(
        obj,
        np.log(np.array(x0)),
        method="L-BFGS-B",
    )

    ell, sigma_f, sigma_y = np.exp(res.x)

    best_mll = -res.fun
    return {
        "ell": float(ell), 
        "sigma_f": float(sigma_f), 
        "sigma_y": float(sigma_y), 
        "mll": float(best_mll),
        "opt": res}

from scipy.optimize import brute
import numpy as np

def grid_search_b2(
    X,
    y,
    delta,
    *,
    ell_range=(1e-2, 2.0),
    sigma_f_range=(1e-2, 3.0),
    sigma_y_range=(1e-3, 1.0),
    grid_sizes=(35, 35, 20),
    jitter=1e-8,
):
    """
    Coarse grid search for B.2 hyperparameters (ell, sigma_f, sigma_y),
    conditioning on fixed Delta.
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    delta = np.asarray(delta).reshape(-1)

    log_ell_min, log_ell_max = np.log(ell_range[0]), np.log(ell_range[1])
    log_sf_min, log_sf_max = np.log(sigma_f_range[0]), np.log(sigma_f_range[1])
    log_sy_min, log_sy_max = np.log(sigma_y_range[0]), np.log(sigma_y_range[1])

    n_ell, n_sf, n_sy = grid_sizes

    def obj(logp):
        ell, sigma_f, sigma_y = np.exp(logp)
        return -mll_b2(
            y, X, delta,
            ell=ell,
            sigma_f=sigma_f,
            sigma_y=sigma_y,
            jitter=jitter,
        )

    rranges = (
        slice(log_ell_min, log_ell_max, complex(n_ell)),
        slice(log_sf_min, log_sf_max, complex(n_sf)),
        slice(log_sy_min, log_sy_max, complex(n_sy)),
    )

    xopt, fval, *_ = brute(obj, rranges, full_output=True, finish=None)
    ell, sigma_f, sigma_y = np.exp(xopt)

    return {
        "ell": float(ell),
        "sigma_f": float(sigma_f),
        "sigma_y": float(sigma_y),
        "mll": float(-fval),
        "grid_sizes": tuple(grid_sizes),
    }

def posterior_f_b2(X, y, delta, X_star, ell, sigma_f, sigma_y, jitter=1e-8):
    """
    Posterior over latent f(X_star) under noisy-input likelihood.
    Uses: Cov(f_*, y) = K(X_*,X) - Cov(f_*, f') D
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1, 1)
    delta = np.asarray(delta).reshape(-1)
    X_star = np.asarray(X_star).reshape(-1)

    n = len(X)
    D = np.diag(delta)

    # Build C_Δ
    C = C_delta_np(X, delta, ell, sigma_f, sigma_y, jitter=jitter)
    L = np.linalg.cholesky(C)

    # Compute Cov(f_*, y)
    K_starX = rbf_k(X_star, X, ell, sigma_f)          
    K1_X_Xstar = rbf_dk_dx1(X, X_star, ell, sigma_f)     
    Cov_fstar_fprime = K1_X_Xstar.T                  
    Cov_fstar_y = K_starX - Cov_fstar_fprime @ D      

    # Compute posterior mean and variance
    alpha = chol_solve(L, y)
    mu = (Cov_fstar_y @ alpha).ravel()

    K_starstar = rbf_k(X_star, X_star, ell, sigma_f)
    V = np.linalg.solve(L, Cov_fstar_y.T)
    Cov = K_starstar - V.T @ V

    # Ensure numerical stability
    var = np.clip(np.diag(Cov), 0.0, np.inf)
    return mu, var

def marginal_posterior_from_delta_samples(X, y, delta_samples, X_star, ell, sigma_f, sigma_y,
                                         max_samples=300, seed=0):
    """
    Compute marginal posterior over f(X_star) by averaging over delta samples.
    """
    ds = np.asarray(delta_samples)
    rng = np.random.default_rng(seed)
    S = min(max_samples, ds.shape[0])
    idx = rng.choice(ds.shape[0], size=S, replace=False)
    ds = ds[idx]

    mus = []
    vars_ = []
    for d in ds:
        mu, var = posterior_f_b2(X, y, d, X_star, ell, sigma_f, sigma_y)
        mus.append(mu)
        vars_.append(var)

    mus = np.stack(mus, axis=0)     
    vars_ = np.stack(vars_, axis=0)

    mu_marg = mus.mean(axis=0)
    var_marg = vars_.mean(axis=0) + mus.var(axis=0, ddof=1)  
    return mu_marg, var_marg

