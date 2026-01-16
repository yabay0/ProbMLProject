from __future__ import annotations
import numpy as np

#---------------------------
# RBF kernel and derivatives
#---------------------------
def rbf_k(x1, x2, ell: float, sigma_f: float) -> np.ndarray:
    """
    Squared Exponential (RBF) kernel:
      k(x,x') = sigma_f^2 * exp(-(x-x')^2 / (2 ell^2))

    for all pairs (x_i, x'_j) with x_i ∈ x1 and x'_j ∈ x2.
    """
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(1, -1)
    r = x1 - x2
    return (sigma_f**2) * np.exp(-0.5 * (r**2) / (ell**2))

def rbf_dk_dx1(x1, x2, ell: float, sigma_f: float) -> np.ndarray:
    """"
    First derivative of RBF kernel w.r.t. first argument:
        ∂/∂x k(x,x') = k(x,x') * (x' - x) / ell^2
    
    evaluated at all pairs (x_i, x'_j).
    """
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(1, -1)
    r = x1 - x2
    K = (sigma_f**2) * np.exp(-0.5 * (r**2) / (ell**2))
    return K * ((x2 - x1) / (ell**2))

def rbf_d2k_dx1dx2(x1, x2, ell: float, sigma_f: float) -> np.ndarray:
    """
    Second derivative of RBF kernel w.r.t. first and second arguments:
        ∂^2/∂x∂x' k(x,x') = k(x,x') * [1/ell^2 - (x - x')^2 / ell^4]
    
    evaluated at all pairs (x_i, x'_j).
    """
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(1, -1)
    r = x1 - x2
    K = (sigma_f**2) * np.exp(-0.5 * (r**2) / (ell**2))
    return K * (1.0/(ell**2) - (r**2)/(ell**4))

