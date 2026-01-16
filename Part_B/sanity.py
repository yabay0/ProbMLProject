from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from kernels_np import rbf_k, rbf_dk_dx1, rbf_d2k_dx1dx2

def prior_sanity_check(out_png="prior_sanity_check.png", ell=0.35, sigma_f=1.0, n_grid=100, seed=0, dpi=200):
    """Sample from the joint prior over f and f' and compare f' to finite differences."""
    # Set random seed
    rng = np.random.default_rng(seed)
    # Generate grid
    X = np.linspace(-1, 1, n_grid)

    # Build joint covariance
    K  = rbf_k(X, X, ell, sigma_f)
    K1 = rbf_dk_dx1(X, X, ell, sigma_f)
    K2 = rbf_d2k_dx1dx2(X, X, ell, sigma_f)

    top = np.concatenate([K, K1.T], axis=1)
    bot = np.concatenate([K1, K2], axis=1)
    Sigma = np.concatenate([top, bot], axis=0)

    # Ensure symmetry and add jitter
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(2*n_grid)
    # Cholesky
    L = np.linalg.cholesky(Sigma)
    # Sample from prior
    z = L @ rng.standard_normal(2*n_grid)

    # Split f and f'
    f = z[:n_grid]
    fp = z[n_grid:]

    # Finite differences
    fd = np.full_like(fp, np.nan)
    fd[1:-1] = (f[2:] - f[:-2]) / (X[2:] - X[:-2])

    # Plot results
    plt.figure(figsize=(8, 4.5))
    plt.plot(X, fp, linewidth=2, label=r"Sampled $f'(x)$")
    plt.plot(X, fd, "--", linewidth=2, label=r"Central diff $\tilde f'(x)$")
    plt.xlabel("$x$")
    plt.ylabel(r"$f'(x)$")
    plt.title("Prior sanity check for kernel derivatives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()

    # Compute mean absolute error
    err = np.nanmean(np.abs(fp - fd))
    return float(err)

