from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from data import f_true

def plot_raw_data(x, y, delta=None, highlight_idx=None, out_png="raw_data_partB.png"):
    """
    Scatter plot of the raw Part B observations.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(x, y, s=18, alpha=0.8, label="Data")

    plt.xlim([-1, 1])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Raw observations")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_b1_three_panel(x, y, x_truth, x_grid, mu_a, var_a, mu_b, var_b, mu_c, var_c, out_png="B1_output.png"):
    """
    Plot three panels for B.1: (a) noisy x, learned sigma_y; (b) noisy x, fixed sigma_y; (c) true x, fixed sigma_y.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    def panel(ax, xs, title, mu, var, legend=False):
        ax.plot(x_grid, f_true(x_grid), linewidth=2, label="True $f(x)$")
        ax.plot(x_grid, mu, linewidth=2, label="Posterior mean")
        std = np.sqrt(np.clip(var, 0, np.inf))
        ax.fill_between(x_grid, mu-1.96*std, mu+1.96*std, alpha=0.2, label="95% CI (latent $f$)")
        ax.scatter(xs, y, s=14, alpha=0.7, label="Data")
        ax.set_title(title); ax.set_xlim([-1,1]); ax.set_xlabel("$x$")
        if legend: ax.legend(loc="best", fontsize=8) 

    panel(axes[0], x, "(a) noisy $x$, learned $\\sigma_y^2$", mu_a, var_a, legend=True)
    panel(axes[1], x, "(b) noisy $x$, fixed $\\sigma_y^2$", mu_b, var_b, legend=False)
    panel(axes[2], x_truth, "(c) true $x_{\\mathrm{truth}}$, fixed $\\sigma_y^2$", mu_c, var_c, legend=False)

    axes[0].set_ylabel("$f(x)$")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_b2_single(x, y, x_grid, mu, var, out_png="B2_output.png"):
    """ Plot single panel for B.2: noisy-input GP (Taylor)."""
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(x_grid, f_true(x_grid), linewidth=2, label="True $f(x)$")
    plt.plot(x_grid, mu, linewidth=2, label="B.2 posterior mean")
    std = np.sqrt(np.clip(var, 0, np.inf))
    plt.fill_between(x_grid, mu-1.96*std, mu+1.96*std, alpha=0.2, label="95% CI (latent $f$)")
    plt.scatter(x, y, s=14, alpha=0.7, label="Data")
    plt.xlim([-1,1]); plt.xlabel("$x$"); plt.ylabel("$f(x)$")
    plt.title("B.2 Noisy-input GP (Taylor)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_delta_scatter(delta_samples, i=8, j=9, true_pair=(-0.25, 0.25), out_png="delta_scatter.png"):
    """ Scatter plot of posterior samples of (Delta_i, Delta_j). """
    ds = np.asarray(delta_samples)
    plt.figure(figsize=(5,5))
    plt.scatter(ds[:, i], ds[:, j], s=10, alpha=0.35, label="Samples")
    plt.scatter([true_pair[0]], [true_pair[1]], s=70, marker="x", label="True")
    plt.xlabel(rf"$\Delta_{{{i+1}}}$"); plt.ylabel(rf"$\Delta_{{{j+1}}}$")
    plt.title(rf"Posterior samples of $(\Delta_{{{i+1}}},\Delta_{{{j+1}}})$")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
