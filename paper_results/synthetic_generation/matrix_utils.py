"""Shared grid-distance and seeding utilities.

Kept separate from offgrid_matrix.py so other scripts can import it without
duplicating.
"""

import hashlib

import numpy as np

# Training grid (training_files/create_contours_args6.py, main(): create_contour_dataset(
#   angle, roughness, 0, 1, 11, 1, 4, 7, sampling_window_size=50, ...))
BO_GRID = np.linspace(0, 1, 11)  # step 0.10
SCALER_GRID = np.geomspace(1, 4, 7)  # log step ln(4)/6 = 0.23105
ROUGH_POS = np.array([0, 0.002, 0.004, 0.006, 0.008, 0.01])  # step 0.002
ROUGH_NEG = np.array([-0.01, -0.02, -0.03, -0.04, -0.05, -0.06])  # step 0.01


def grid_distance(bond: float, scaler: float, roughness: float) -> float:
    """Distance from (bond, scaler, roughness) to the nearest training grid
    point, combined in quadrature across per-axis grid-step units.

    Angle is not included: it is matched exactly elsewhere, not distance-
    weighted (the training grid is dense enough in angle — 0.05 deg steps —
    that any chosen angle is effectively on-grid).
    """
    d_bo = np.min(np.abs(BO_GRID - bond)) / 0.10
    d_sc = np.min(np.abs(np.log(SCALER_GRID) - np.log(scaler))) / (np.log(4) / 6)
    grid_r = ROUGH_POS if roughness >= 0 else ROUGH_NEG
    step_r = 0.002 if roughness >= 0 else 0.01
    d_r = np.min(np.abs(grid_r - roughness)) / step_r
    return float(np.sqrt(d_bo**2 + d_sc**2 + d_r**2))


def roughness_regime(roughness: float) -> str:
    """Categorical regime for a roughness value — positive = random surface
    jitter, negative = reflective baseline. Never compare/match across regimes."""
    return "positive" if roughness >= 0 else "negative"


def cell_seed(angle: float, bond: float, scaler: float, roughness: float) -> int:
    """Deterministic seed for a parameter tuple, for use with np.random.seed()
    immediately before a draw_drop() call (draw_drop perturbs roughness with
    the *global* np.random.uniform, so a Generator instance won't do)."""
    key = f"{angle:.4f}|{bond:.4f}|{scaler:.4f}|{roughness:.4f}"
    return int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16)
