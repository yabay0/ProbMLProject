# Probabilistic Machine Learning Project

**University of Copenhagen**

**Group:**
- Benjamin Baadsager (npr151@alumni.ku.dk)
- Christian Clasen (nsw337@alumni.ku.dk)
- Yasin Baysal (cmv882@alumni.ku.dk)

---

This repository contains the code used for the final project in Probabilistic Machine Learning, consisting of two main parts: diffusion-based generative models (Part A) and function fitting with noisy inputs (Part B).

## Project Overview

### Part A: Diffusion-Based Generative Models for MNIST
Implementation of diffusion models for image generation on the MNIST dataset. This part explores:
- Baseline diffusion model implementation
- Four directional improvements to the baseline model with comparisons to DDPM and failure mode analyses
- Quantitative comparisons of different approaches using different metrics

### Part B: Function Fitting with Noisy Inputs
Gaussian process regression with input noise modeling, including:
- **B.1**: Standard GP baselines (ignoring input noise)
- **B.2**: Bayesian inference for noisy-input GP using NUTS sampling
  - Prior sanity checks
  - Posterior inference with noisy observations
  - MCMC diagnostics

## Repository Structure

```
ProbMLProject/
├── README.md
├── Part_A/
│   └── Part_A_code.ipynb          # Jupyter notebook for diffusion models
├── Part_B/
│   ├── data_part_B.csv            # Dataset for Part B
│   ├── run_all_part_b.py          # Main script to run all of Part B
│   ├── data.py                    # Data loading utilities
│   ├── gp_b1.py                   # Standard GP baselines
│   ├── gp_b2.py                   # Noisy-input GP implementation
│   ├── kernels_np.py              # Kernel functions (NumPy)
│   ├── kernels_torch.py           # Kernel functions (PyTorch)
│   ├── samplers.py                # NUTS sampler implementation
│   ├── sanity.py                  # Prior sanity checks
│   ├── plotting.py                # Visualization utilities
│   └── outputs_partB/             # Generated outputs and results
```

## How to Run

### Part A: Diffusion-Based Generative Models

Part A is implemented as a Jupyter notebook with sections:
1. Baseline diffusion model
2. Four directional improvements
3. Quantitative comparisons

**To run**:
1. Open `Part_A/Part_A_code.ipynb`
2. Run the cells sequentially from top to bottom
3. Each section builds on the previous one, so maintain the execution order

### Part B: Function Fitting with Noisy Inputs

Part B is organized as modular Python files with a single entry point.

**To run**:
```bash
cd Part_B
python run_all_part_b.py
```

This script will:
1. Load the data from `data_part_B.csv`
2. Run Part B.1: Three GP baseline approaches (standard MLE)
3. Run Part B.2: Bayesian inference with NUTS sampling
   - Prior sanity checks
   - Posterior inference for noisy-input GP
   - 4-chain NUTS sampling (parallelized)
   - ArviZ diagnostics and trace plots
4. Generate all figures and save results to `outputs_partB/`

> Part B uses Python files rather than a notebook like in Part A because the NUTS sampler runs 4 parallel chains for robust MCMC inference. This multi-chain parallelization is better handled by Python scripts, which allow better control over multiprocessing and avoid potential kernel conflicts in the notebook environments.

## Outputs

### Part A
- Generated images from diffusion models
- Training curves and metrics
- Comparison plots between different approaches and failure mode analyses

### Part B
All outputs are saved to `Part_B/outputs_partB/`:
- `raw_data_partB.png`: Visualization of input data with noisy observations
- `B1_output.png`: Standard GP posterior (single baseline)
- `B1_output_grid.png`: Grid search results for baseline GP
- `B2_output.png`: Noisy-input GP posterior predictions
- `B2_marginal_over_delta.png`: Marginal posterior over Delta samples
- `prior_sanity_check.png`: Prior predictive samples sanity check
- `nuts_trace_selected.png`: MCMC trace plots for selected parameters
- `delta_scatter.png`: Parameter posterior scatter plots
- `nuts_arviz_summary.csv`: MCMC diagnostics summary
- `results_partB.npz`: Final results saved in NumPy format

## Project Documentation

This repository includes two main documents:
- `PML2025_final_project.pdf` - Project description
- `exam_project_probabilistic_ML.pdf` - Our solution

---