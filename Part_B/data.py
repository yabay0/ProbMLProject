from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetB:
    x: np.ndarray
    y: np.ndarray
    delta: np.ndarray
    x_truth: np.ndarray

def load_part_b_csv(path: str = "data_part_B.csv", delimiter: str = ",") -> DatasetB:
    """
    CSV columns: x, y, Delta.
    Returns:
      x        : noisy inputs
      y        : outputs
      delta    : input noise realizations (Delta_i)
      x_truth  : x - delta
    """
    data = np.loadtxt(path, delimiter=delimiter)
    x = data[:, 0].astype(np.float64)
    y = data[:, 1].astype(np.float64)
    delta = data[:, 2].astype(np.float64)
    return DatasetB(x=x, y=y, delta=delta, x_truth=x - delta)

def make_grid(n: int = 100, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Creates a 1D grid of n points between lo and hi."""
    return np.linspace(lo, hi, n, dtype=np.float64)

def f_true(x: np.ndarray) -> np.ndarray:
    """Ground-truth function."""
    x = np.asarray(x, dtype=np.float64)
    return -x**2 + 2.0 / (1.0 + np.exp(-10.0 * x))

