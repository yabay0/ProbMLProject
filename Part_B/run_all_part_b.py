# ------------------------------------------------------------
# Runs ALL of Part B (B.1 + B.2) end-to-end:
#  - B.1: three GP baselines
#  - B.2.1: prior sanity check
#  - B.2.2: noisy-input GP fit + posterior plot
#  - B.2.3: NUTS over Delta (4 chains) + ArviZ diagnostics + trace plot + scatter
#  - Saves all figures + a results .npz bundle
# ------------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from data import load_part_b_csv, make_grid
from gp_b1 import fit_b1, posterior_latent_f, grid_search_b1
from gp_b2 import fit_b2, posterior_f_b2, marginal_posterior_from_delta_samples, grid_search_b2
from sanity import prior_sanity_check
from samplers import run_nuts_delta
from plotting import (
    plot_raw_data,
    plot_b1_three_panel,
    plot_b2_single,
    plot_delta_scatter,
)

# ----------------------------
# Helper functions for logging
# ----------------------------
def banner(msg: str):
    line = "=" * len(msg)
    print("\n" + line)
    print(msg)
    print(line + "\n")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def fmt_params(d):
    ell = d["ell"]
    sigma_f = d["sigma_f"]
    sigma_y = d["sigma_y"]
    return (
        f"ell={ell:.6g}," 
        f"sigma_f={sigma_f:.6g}"
        f"(sigma_f^2={sigma_f**2:.6g})" 
        f"sigma_y={sigma_y:.6g} "
        f"(sigma_y^2={sigma_y**2:.6g})"
    )

def grid_then_refine_b1(X, y, sigma_y_fixed, *, grid_kwargs=None, fit_kwargs=None):
    """
    1) Coarse brute grid-search in log-space
    2) Refine using L-BFGS-B starting from grid optimum
    Returns (grid_result, refined_fit).
    """
    grid_kwargs = grid_kwargs or {}
    fit_kwargs = fit_kwargs or {}

    gs = grid_search_b1(X, y, sigma_y_fixed=sigma_y_fixed, **grid_kwargs)

    if sigma_y_fixed is None:
        x0 = (gs["ell"], gs["sigma_f"], gs["sigma_y"])
    else:
        # x0 still expects 3-tuple; sigma_y ignored when fixed, but we keep it sane
        x0 = (gs["ell"], gs["sigma_f"], gs["sigma_y"])

    refined = fit_b1(X, y, sigma_y_fixed=sigma_y_fixed, x0=x0, **fit_kwargs)
    return gs, refined

def grid_then_refine_b2(X, y, delta, *, grid_kwargs=None, fit_kwargs=None):
    grid_kwargs = grid_kwargs or {}
    fit_kwargs = fit_kwargs or {}

    gs = grid_search_b2(X, y, delta, **grid_kwargs)

    x0 = (gs["ell"], gs["sigma_f"], gs["sigma_y"])

    refined = fit_b2(X, y, delta, x0=x0, **fit_kwargs)
    return gs, refined

# -------------------------------------
# Main function for running full Part B
# -------------------------------------
def main():
    outdir = "outputs_partB"
    ensure_dir(outdir)

    banner("PART B is now running")
    print("This script runs B.1 + B.2 including sanity checks and Bayesian inference (NUTS).")
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {outdir}")

    # -------------------------
    # Load data + grid
    # -------------------------
    ds = load_part_b_csv("data_part_B.csv")
    x_grid = make_grid(100)

    print(f"Loaded data_part_B.csv with n={len(ds.x)} observations.")
    print("CSV columns assumed: x, y, Delta")
    df = pd.DataFrame({
        "x": ds.x,
        "y": ds.y,
        "delta": ds.delta,
    })
    df.index.name = "i" 
    print("Data preview:")
    print(df)
    print()

    # Plot raw data
    raw_fig = os.path.join(outdir, "raw_data_partB.png")
    plot_raw_data(ds.x, ds.y, delta=ds.delta, highlight_idx=[9, 10], out_png=raw_fig)
    print(f"Saved raw data reference figure: {raw_fig}")

    # -------------------------
    # Part B.1
    # -------------------------
    banner("Part B.1 — Standard GP baselines (ignoring input noise)")
    print("Computing maximum marginal likelihood estimates (MLL)\n")

    sigma_y2_fixed = 0.0025
    sigma_y_fixed = np.sqrt(sigma_y2_fixed)

    # (a) noisy x, sigma_y learned
    fit_a = fit_b1(ds.x, ds.y, sigma_y_fixed=None)
    gs_a, fit_a_ref = grid_then_refine_b1(ds.x, ds.y, sigma_y_fixed=None,
                                      grid_kwargs=dict(grid_sizes=(35, 35, 25)))

    print("(a) noisy x, sigma_y learned")
    print("    L-BFGS-B (default init):", fmt_params(fit_a), f"(mll={fit_a['mll']:.4f})")
    print("    Grid (coarse):          ", fmt_params(gs_a), f"(mll={gs_a['mll']:.4f})")
    print("    L-BFGS-B (from grid):   ", fmt_params(fit_a_ref), f"(mll={fit_a_ref['mll']:.4f})")

    # (b) noisy x, sigma_y fixed
    fit_b = fit_b1(ds.x, ds.y, sigma_y_fixed=sigma_y_fixed)
    gs_b, fit_b_ref = grid_then_refine_b1(ds.x, ds.y, sigma_y_fixed=sigma_y_fixed,
                                      grid_kwargs=dict(grid_sizes=(35, 35, 10)))

    print("(b) noisy x, sigma_y fixed to sqrt(0.0025)")
    print("    L-BFGS-B (default init):", fmt_params(fit_b), f"(mll={fit_b['mll']:.4f})")
    print("    Grid (coarse):          ", fmt_params(gs_b), f"(mll={gs_b['mll']:.4f})")
    print("    L-BFGS-B (from grid):   ", fmt_params(fit_b_ref), f"(mll={fit_b_ref['mll']:.4f})")

    # (c) true x_truth, sigma_y fixed
    fit_c = fit_b1(ds.x_truth, ds.y, sigma_y_fixed=sigma_y_fixed)
    gs_c, fit_c_ref = grid_then_refine_b1(ds.x_truth, ds.y, sigma_y_fixed=sigma_y_fixed)

    print("(c) true x_truth, sigma_y fixed to sqrt(0.0025)")
    print("    L-BFGS-B (default init):", fmt_params(fit_c), f"(mll={fit_c['mll']:.4f})")
    print("    Grid (coarse):          ", fmt_params(gs_c), f"(mll={gs_c['mll']:.4f})")
    print("    L-BFGS-B (from grid):   ", fmt_params(fit_c_ref), f"(mll={fit_c_ref['mll']:.4f})")


    # Posterior plots (latent f)
    mu_a, var_a = posterior_latent_f(ds.x, ds.y, x_grid, **{k: fit_a[k] for k in ["ell","sigma_f","sigma_y"]})
    mu_b, var_b = posterior_latent_f(ds.x, ds.y, x_grid, **{k: fit_b[k] for k in ["ell","sigma_f","sigma_y"]})
    mu_c, var_c = posterior_latent_f(ds.x_truth, ds.y, x_grid, **{k: fit_c[k] for k in ["ell","sigma_f","sigma_y"]})

    b1_fig = os.path.join(outdir, "B1_output.png")
    plot_b1_three_panel( 
        ds.x, ds.y, ds.x_truth,
        x_grid,
        mu_a, var_a,
        mu_b, var_b,
        mu_c, var_c,
        out_png=b1_fig,
    )
    print(f"\nSaved B.1 figure using L-BFGS-B: {b1_fig}")

    # Posterior plots (latent f) using grid search params for comparison
    mu_a_1, var_a_1 = posterior_latent_f(ds.x, ds.y, x_grid, **{k: gs_a[k] for k in ["ell","sigma_f","sigma_y"]})
    mu_b_1, var_b_1 = posterior_latent_f(ds.x, ds.y, x_grid, **{k: gs_b[k] for k in ["ell","sigma_f","sigma_y"]})
    mu_c_1, var_c_1 = posterior_latent_f(ds.x_truth, ds.y, x_grid, **{k: gs_c[k] for k in ["ell","sigma_f","sigma_y"]})

    b1_fig = os.path.join(outdir, "B1_output_grid.png")
    plot_b1_three_panel( 
        ds.x, ds.y, ds.x_truth,
        x_grid,
        mu_a_1, var_a_1,
        mu_b_1, var_b_1,
        mu_c_1, var_c_1,
        out_png=b1_fig,
    )
    print(f"\nSaved B.1 figure using Grid search: {b1_fig}")

    # -------------------------
    # Part B.2.1 — sanity check
    # -------------------------
    banner("Part B.2.1 — Prior sanity check for kernel derivatives")
    sanity_png = os.path.join(outdir, "prior_sanity_check.png")

    print("Sampling jointly from the GP prior over (f, f') and comparing to finite differences.")

    # Uses ell/sigma_f from B.1(c) by default (often a reasonable smooth prior)
    err = prior_sanity_check(
        out_png=sanity_png,
        ell=fit_c["ell"],
        sigma_f=fit_c["sigma_f"],
        n_grid=100,
        seed=1,
    )
    print(f"Saved sanity check: {sanity_png}")
    print(f"Mean absolute error (interior points): {err:.4g}")
    
    # -------------------------------------------------------------
    # Part B.2.2 — fit noisy-input GP conditioned on provided Delta
    # -------------------------------------------------------------
    banner("Part B.2.2 — Noisy-input GP (conditioning on provided $\Delta$)")

    fit2 = fit_b2(ds.x, ds.y, ds.delta)
    gs2, fit2_ref = grid_then_refine_b2(ds.x, ds.y, ds.delta)

    print("B.2 noisy-input GP (conditioning on Δ)")
    print("    L-BFGS-B (default):", fmt_params(fit2), f"(mll={fit2['mll']:.4f})")
    print("    Grid (coarse):     ", fmt_params(gs2), f"(mll={gs2['mll']:.4f})")
    print("    L-BFGS-B (refined):", fmt_params(fit2_ref), f"(mll={fit2_ref['mll']:.4f})\n")

    mu2, var2 = posterior_f_b2(ds.x, ds.y, ds.delta, x_grid, fit2["ell"], fit2["sigma_f"], fit2["sigma_y"])
    b2_fig = os.path.join(outdir, "B2_output.png")
    plot_b2_single(ds.x, ds.y, x_grid, mu2, var2, out_png=b2_fig)
    print(f"\nSaved B.2 figure: {b2_fig}")

    # -------------------------------------------------------------
    # Part B.2.3 — NUTS for Delta + ArviZ diagnostics + trace plots
    # -------------------------------------------------------------
    banner("Part B.2.3 — Bayesian inference over Δ using NUTS (4 chains)")

    sigma_x = float(np.sqrt(0.01))

    print("Running NUTS ...")
    print("    chains=4, warmup=1500, samples=1500")
    print(f"    Using fitted params from B.2.2: ell={fit2['ell']:.6g}, sigma_f={fit2['sigma_f']:.6g}, sigma_y={fit2['sigma_y']:.6g}")
    print(f"    Prior: Δ ~ N(0, sigma_x^2 I), sigma_x={sigma_x:.6g}\n")

    mcmc = run_nuts_delta(
        ds.x, ds.y,
        ell=fit2["ell"],
        sigma_f=fit2["sigma_f"],
        sigma_y=fit2["sigma_y"],
        sigma_x=sigma_x,
        num_chains=4,
        warmup=1500,
        samples=1500,
        seed=0,
        target_accept=0.85,
        jitter=1e-8,
    )

    delta_nuts = mcmc.get_samples(group_by_chain=False)["delta"].detach().cpu().numpy()
    print("NUTS finished.")

    mu_marg, var_marg = marginal_posterior_from_delta_samples(
        ds.x, ds.y, delta_nuts, x_grid,
        fit2["ell"], fit2["sigma_f"], fit2["sigma_y"],
        max_samples=300, seed=0
    )

    b2_marg_fig = os.path.join(outdir, "B2_marginal_over_delta.png")
    plot_b2_single(ds.x, ds.y, x_grid, mu_marg, var_marg, out_png=b2_marg_fig)
    print(f"Saved marginal B.2 plot: {b2_marg_fig}")

    print(f"Posterior mean of (Δ10, Δ11) (1-indexed): {delta_nuts[:,9].mean():.4f}, {delta_nuts[:,10].mean():.4f}")
    print(f"Posterior std  of (Δ10, Δ11) (1-indexed): {delta_nuts[:,9].std():.4f}, {delta_nuts[:,10].std():.4f}")

    # Scatter plot (Delta_10, Delta_11) => Python indices 9, 10
    scatter_fig = os.path.join(outdir, "delta_scatter.png")
    plot_delta_scatter(delta_nuts, i=9, j=10, true_pair=(-0.25, 0.25), out_png=scatter_fig)
    print(f"\nSaved NUTS scatter: {scatter_fig}")    

    # ArviZ diagnostics
    print("\nComputing ArviZ diagnostics (R-hat, ESS, trace plots)")
    idata = None
    try:
        idata = az.from_pyro(mcmc)

        # Summary table
        summ = az.summary(idata, var_names=["delta"], round_to=4)
        print("\nArviZ summary (first 12 Delta components):")
        print(summ)

        summ_csv = os.path.join(outdir, "nuts_arviz_summary.csv")
        summ.to_csv(summ_csv)
        print(f"\nSaved ArviZ summary CSV: {summ_csv}")

        trace_png = os.path.join(outdir, "nuts_trace_selected.png")
        az.plot_trace(idata, var_names=["delta"], coords={"delta_dim_0":[9,10]})
        
        plt.tight_layout()
        plt.savefig(trace_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved trace plot: {trace_png}")

    except Exception as e:
        print(f"ArviZ diagnostics failed with: {repr(e)}")

    # ------------------
    # Save all results
    # ------------------
    banner("Saving results bundle")

    results_path = os.path.join(outdir, "results_partB.npz")
    np.savez(
        results_path,
        # B.1 fits (ell, sigma_f, sigma_y)
        b1_a=np.array([fit_a["ell"], fit_a["sigma_f"], fit_a["sigma_y"]], dtype=np.float64),
        b1_b=np.array([fit_b["ell"], fit_b["sigma_f"], fit_b["sigma_y"]], dtype=np.float64),
        b1_c=np.array([fit_c["ell"], fit_c["sigma_f"], fit_c["sigma_y"]], dtype=np.float64),
        # B.2 fit
        b2=np.array([fit2["ell"], fit2["sigma_f"], fit2["sigma_y"]], dtype=np.float64),
        sanity_err=np.array(err, dtype=np.float64),
        # samples
        delta_nuts=delta_nuts.astype(np.float64),
        delta_true_from_file=ds.delta.astype(np.float64),
        # convenience
        x=ds.x.astype(np.float64),
        y=ds.y.astype(np.float64),
        x_truth=ds.x_truth.astype(np.float64),
        x_grid=x_grid.astype(np.float64),
        b1_mu_a=mu_a.astype(np.float64),
        b1_var_a=var_a.astype(np.float64),
        b1_mu_b=mu_b.astype(np.float64),
        b1_var_b=var_b.astype(np.float64),
        b1_mu_c=mu_c.astype(np.float64),
        b1_var_c=var_c.astype(np.float64),
        b2_mu=mu2.astype(np.float64),
        b2_var=var2.astype(np.float64),
    )

    print(f"Saved results: {results_path}")

    banner("DONE — Files written")
    print("Figures:")
    print(f"  - {b1_fig}")
    print(f"  - {b2_fig}")
    print(f"  - {sanity_png}")
    print(f"  - {scatter_fig}")
    print(f"  - {b2_marg_fig}")
    trace_png = os.path.join(outdir, "nuts_trace_selected.png")
    summ_csv = os.path.join(outdir, "nuts_arviz_summary.csv")
    if os.path.exists(trace_png):
        print(f"  - {trace_png}")
    if os.path.exists(summ_csv):
        print(f"  - {summ_csv}")

    print("\nResults bundle:")
    print(f"  - {results_path}")

    print("\nKey fitted parameters:")
    print("  B.1 (a):", fmt_params(fit_a))
    print("  B.1 (b):", fmt_params(fit_b))
    print("  B.1 (c):", fmt_params(fit_c))
    print("  B.2.2  :", fmt_params(fit2))

    print("\nHighlighted Δ recovery (NUTS):")
    print(f"  mean(Δ10,Δ11) = ({delta_nuts[:,9].mean():.4f}, {delta_nuts[:,10].mean():.4f})")
    print(f"  std (Δ10,Δ11) = ({delta_nuts[:,9].std():.4f}, {delta_nuts[:,10].std():.4f})")

if __name__ == "__main__":
    main()
