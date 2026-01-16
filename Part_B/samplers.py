from __future__ import annotations
import numpy as np
import torch
import pyro
from pyro.infer.mcmc import NUTS, MCMC
from kernels_torch import rbf_k, rbf_dk_dx1, rbf_d2k_dx1dx2


def build_C_delta_torch(X, delta, ell, sigma_f, sigma_y, jitter=1e-8):
    """Build the covariance matrix C_delta for the noisy-input GP model."""
    K  = rbf_k(X, X, ell, sigma_f)
    K1 = rbf_dk_dx1(X, X, ell, sigma_f)
    K2 = rbf_d2k_dx1dx2(X, X, ell, sigma_f)
    D = torch.diag(delta)
    n = X.shape[0]
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    C = K - D @ K1 - K1.T @ D + D @ K2 @ D + (sigma_y**2 + jitter) * I
    return 0.5 * (C + C.T)

def logpost_delta_torch(X, y, delta, ell, sigma_f, sigma_y, sigma_x, jitter=1e-8):
    """Compute the log posterior over delta in the noisy-input GP model."""
    # prior: N(0, sigma_x^2 I)
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=X.dtype, device=X.device))
    n = X.shape[0]
    log_prior = -0.5 * torch.sum((delta / sigma_x)**2) - n*torch.log(sigma_x) - 0.5*n*log2pi

    # likelihood: N(y;0,C_delta)
    C = build_C_delta_torch(X, delta, ell, sigma_f, sigma_y, jitter=jitter)
    L = torch.linalg.cholesky(C)  # may fail
    alpha = torch.cholesky_solve(y[:, None], L)
    quad = (y[None, :] @ alpha)[0, 0]
    logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
    log_like = -0.5*quad - 0.5*logdet - 0.5*n*log2pi

    return log_like + log_prior

class PotentialFn:
    """
    Potential function for NUTS sampling over delta in the noisy-input GP model.
    """
    def __init__(self, x_np, y_np, ell, sigma_f, sigma_y, sigma_x, jitter=1e-8):
        self.X = torch.tensor(x_np, dtype=torch.float64)
        self.y = torch.tensor(y_np, dtype=torch.float64)
        self.ell = torch.tensor(float(ell), dtype=torch.float64)
        self.sigma_f = torch.tensor(float(sigma_f), dtype=torch.float64)
        self.sigma_y = torch.tensor(float(sigma_y), dtype=torch.float64)
        self.sigma_x = torch.tensor(float(sigma_x), dtype=torch.float64)
        self.jitter = float(jitter)

    def __call__(self, params):
        delta = params["delta"]
        try:
            lp = logpost_delta_torch(self.X, self.y, delta, self.ell, self.sigma_f, self.sigma_y, self.sigma_x, jitter=self.jitter)
            return -lp
        except RuntimeError:
            print("Cholesky failed inside NUTS potential function.")
            return torch.tensor(float("inf"), dtype=torch.float64)

def run_nuts_delta(x_np, y_np, ell, sigma_f, sigma_y, sigma_x, num_chains=4, warmup=1500, samples=1500, seed=0, init_scale=0.2, target_accept=0.85, jitter=1e-8):
    """Run NUTS to sample from the posterior over delta in the noisy-input GP model."""
    pyro.set_rng_seed(seed)
    torch.set_default_dtype(torch.float64)
    n = len(x_np)
    init = (init_scale * torch.randn(num_chains, n, dtype=torch.float64)) * float(sigma_x)

    kernel = NUTS(
        potential_fn=PotentialFn(x_np, y_np, ell, sigma_f, sigma_y, sigma_x, jitter=jitter),
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        target_accept_prob=target_accept,
    )
    mcmc = MCMC(
        kernel, 
        num_samples=samples, 
        warmup_steps=warmup, 
        num_chains=num_chains, 
        initial_params={"delta": init}
    )
    mcmc.run()
    return mcmc

