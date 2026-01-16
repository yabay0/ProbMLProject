import torch

# ----------------------------
# RBF kernel
# ----------------------------

def rbf_k(x1, x2, ell, sigma_f):
    """Radial Basis Function (RBF) kernel."""
    r = x1[:, None] - x2[None, :]
    return (sigma_f**2) * torch.exp(-0.5 * (r**2) / (ell**2))

def rbf_dk_dx1(x1, x2, ell, sigma_f):
    """Derivative of RBF kernel with respect to x1."""
    r = x1[:, None] - x2[None, :]
    K = (sigma_f**2) * torch.exp(-0.5 * (r**2) / (ell**2))
    return K * ((x2[None, :] - x1[:, None]) / (ell**2))

def rbf_d2k_dx1dx2(x1, x2, ell, sigma_f):
    """Second derivative of RBF kernel with respect to x1 and x2."""
    r = x1[:, None] - x2[None, :]
    K = (sigma_f**2) * torch.exp(-0.5 * (r**2) / (ell**2))
    return K * (1.0/(ell**2) - (r**2)/(ell**4))

