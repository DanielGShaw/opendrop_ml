"""Contour-level (no-image-pipeline) prediction (SYNTHETIC_DATA_REANALYSIS.md, A.7).

Reproduces the exact training-format preprocessing — analytic contact
point, no CPID, no isolate_contour — so it tests the model's shape->angle
mapping in isolation. Used by:
  - generate_offgrid_synthetic.py, for the Phase-1 GT-discretisation-fidelity
    check (A.5): compares a dpi=64 render's prediction to dpi=256's.
  - contour_level_sweep.py, for the Phase-2A sweep (the plan's primary
    endpoint).

Gotchas (see A.7 in the plan doc for the full list):
  - prepare_contour() mutates its arguments in place — never reuse a
    contour/CPs array across calls.
  - sampling() can silently return a wrong/empty window if the CP index is
    too close to the start of the contour (negative-index wraparound) —
    hence the shape assert below.
  - window_shift_start=0, window_shift_end=0 gives exactly one window;
    the +/-5 shift in create_contour_dataset() is a training-time
    augmentation only, never used at evaluation.
  - Right side only, matching training's right_only=True.
  - No output clipping to [0, 180] here (unlike experimental_pred), so
    raw extrapolation behaviour at the 20/180 deg endpoints stays visible.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_files"))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from create_contours_args6 import (  # noqa: E402
    draw_drop,
    extract_edges_CV,
    normalize_ds,
    prepare_contour,
    sampling,
)

from config import ML_MODEL_DIR  # noqa: E402
from matrix_utils import cell_seed  # noqa: E402

_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.saved_model.load(ML_MODEL_DIR)
    return _model


def render_contour(angle, bond, scaler, roughness, dpi=64, seed=None):
    """Render + extract + prepare the right-side contour and its contact
    point, at the given dpi. Reimplements create_contour()'s body (rather
    than calling it) so dpi is adjustable for the GT-fidelity check."""
    if seed is None:
        seed = cell_seed(angle, bond, scaler, roughness)
    np.random.seed(seed)  # immediately before draw_drop — see A.3

    img, CPs_img = draw_drop(angle, bond, scaler, roughness, 5000, dpi=dpi)
    contour = extract_edges_CV(img)
    CPs = [list(CPs_img[0]), list(CPs_img[1])]
    right, CPs_prepared = prepare_contour(contour, CPs, given_input_len=None, right_only=True)
    return right, CPs_prepared


def predict_contour_level(angle, bond, scaler, roughness, dpi=64, seed=None, model=None):
    """Exact-CP, training-format prediction. No image pipeline. Returns the
    raw (unclipped) predicted angle for the right side."""
    right, CPs = render_contour(angle, bond, scaler, roughness, dpi=dpi, seed=seed)

    # sampling() re-parses its dict keys with eval(), so this format is load-bearing:
    # angle_Bo_scaler_roughness_CPx_CPy_0_0
    key = f"{angle}_{bond}_{scaler}_{roughness}_{CPs[1][0]}_{CPs[1][1]}_0_0"

    windows = sampling(
        {key: right},
        window_size=50,
        window_shift_start=0,
        window_shift_end=0,
        surface_points=10,
        display=False,
    )
    ds = normalize_ds(windows)

    window = np.array(list(ds.values())[0], dtype=np.float32)
    assert window.shape == (50, 2), f"bad window shape {window.shape}"

    if model is None:
        model = get_model()
    out = model.signatures["serving_default"](
        conv1d_input=tf.convert_to_tensor(window[None, ...])  # (1, 50, 2)
    )
    return float(list(out.values())[0].numpy().ravel()[0])
