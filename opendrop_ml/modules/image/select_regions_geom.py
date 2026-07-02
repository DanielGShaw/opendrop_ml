#!/usr/bin/env python
# coding=utf-8
"""Pure geometry helpers for the USER_SELECTED baseline workflow.

These were lifted out of ``select_regions.user_line`` so they can be unit
tested without driving an OpenCV window. All inputs and outputs use image
pixel coordinates (origin at top-left, +y downward) to match the rest of
the image pipeline.
"""

from typing import Tuple

import numpy as np


def points_above_baseline(
    contour: np.ndarray, ix: int, iy: int, fx: int, fy: int
) -> np.ndarray:
    """Return contour points on the positive-cross-product side of the baseline.

    The baseline is the directed segment from ``(ix, iy)`` to ``(fx, fy)``.
    For each contour point ``(cx, cy)`` the 2D cross product of the baseline
    direction with the vector from the baseline start to the point is

        xp = (ix - fx) * (cy - iy) - (iy - fy) * (cx - ix)

    In image coordinates (y-axis flipped relative to math convention) ``xp > 0``
    selects the visually-above side, i.e. the drop. Replaces a Python loop over
    thousands of points with a single numpy expression.

    Args:
        contour: ``(N, 2)`` array of ``(x, y)`` points.
        ix, iy: baseline start endpoint.
        fx, fy: baseline end endpoint.

    Returns:
        ``(M, 2)`` subset of ``contour`` (same dtype) with ``M <= N``.
    """
    if contour.size == 0:
        return contour
    xp = (ix - fx) * (contour[:, 1] - iy) - (iy - fy) * (contour[:, 0] - ix)
    return contour[xp > 0]


def initial_baseline_seed(
    contour: np.ndarray, canvas_width: int, canvas_height: int
) -> Tuple[int, int, int, int]:
    """Return ``(ix, iy, fx, fy)`` for the starting baseline.

    A horizontal line one pixel below the lowest contour point spans the full
    canvas width. Clamped to stay at least 5 px above the canvas bottom so it
    remains visible after window scaling. Replaces the previous
    polyfit-through-first-and-last-points seed, which produced near-vertical
    lines on tilted-surface images and forced the user to rotate hundreds of
    times before the baseline became usable.

    Args:
        contour: ``(N, 2)`` array of ``(x, y)`` points; only the y column is read.
        canvas_width: width of the cropped image in pixels.
        canvas_height: height of the cropped image in pixels.

    Returns:
        Tuple ``(ix, iy, fx, fy)`` of integer pixel coordinates.
    """
    fallback_y = min(canvas_height - 5, int(np.max(contour[:, 1])) + 1)
    return 0, fallback_y, canvas_width, fallback_y


def longest_contour_arc_above_baseline(
    contour: np.ndarray, ix: int, iy: int, fx: int, fy: int
) -> np.ndarray:
    """Return the longest contiguous arc of ``contour`` lying above the baseline.

    For a drop sitting on a rough/spiky surface, the cross-product mask of
    ``points_above_baseline`` returns the drop arc *plus* every spike-tip that
    pokes above the baseline. The drop is the *longest contiguous run* of
    above-baseline points in contour-traversal order; the spikes are short
    runs separated by below-baseline gaps. Selecting just the longest run
    drops the spikes.

    The contour is treated as cyclic, so the longest run may wrap from the
    last index back to index 0. Assumes the input order is a meaningful
    boundary traversal (true for ``cv2.findContours`` output, which
    ``cluster_OPTICS`` preserves).

    Args:
        contour: ``(N, 2)`` array of ``(x, y)`` points in traversal order.
        ix, iy: baseline start endpoint.
        fx, fy: baseline end endpoint.

    Returns:
        ``(M, 2)`` subset of ``contour`` with ``M <= N``. May be a slice or a
        concatenation across the wrap. Empty when no point is above the line.
    """
    if contour.size == 0:
        return contour

    xp = (ix - fx) * (contour[:, 1] - iy) - (iy - fy) * (contour[:, 0] - ix)
    mask = xp > 0
    n = len(mask)
    if not mask.any():
        return contour[:0]
    if mask.all():
        return contour

    # Detect rising / falling edges. Padding with False on both ends turns the
    # boundary cases into ordinary transitions.
    diff = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.where(diff == 1)[0]  # inclusive start of each run
    ends = np.where(diff == -1)[0]  # exclusive end of each run

    runs = list(zip(starts.tolist(), ends.tolist()))

    # Wrap-around: a run touching index 0 and a run touching index n are
    # adjacent on the cyclic contour and should compete as one merged run.
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n:
        wrapped = (runs[-1][0], runs[0][1])  # end < start signals wrap
        runs = runs[1:-1] + [wrapped]

    def run_length(run):
        s, e = run
        return (e - s) if e > s else (n - s) + e

    s, e = max(runs, key=run_length)
    if e > s:
        return contour[s:e]
    return np.concatenate([contour[s:], contour[:e]])


def rotate_baseline(
    ix: float, iy: float, fx: float, fy: float, theta_deg: float
) -> Tuple[float, float, float, float]:
    """Rotate baseline endpoints about their midpoint by ``theta_deg``.

    Sign convention: positive ``theta_deg`` rotates the baseline visually
    clockwise in the image (matches the ``D`` / ``E`` key bindings in
    ``user_line``), negative rotates counter-clockwise (matches ``A`` /
    ``Q``). The midpoint is preserved exactly; endpoints are returned as
    floats so callers can accumulate sub-pixel rotations (a 0.1° press on
    a typical 500-px baseline shifts each end by ~0.4 px, which would
    round away if the helper int-rounded internally — the caller renders
    with an ``int`` cast at draw time instead).

    Args:
        ix, iy: baseline start endpoint.
        fx, fy: baseline end endpoint.
        theta_deg: rotation angle in degrees.

    Returns:
        Tuple ``(ix', iy', fx', fy')`` of rotated float endpoints.
    """
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    x0 = np.array([ix, iy], dtype=float)
    x1 = np.array([fx, fy], dtype=float)
    xc = 0.5 * (x0 + x1)

    x0r = rotation @ (x0 - xc) + xc
    x1r = rotation @ (x1 - xc) + xc

    return float(x0r[0]), float(x0r[1]), float(x1r[0]), float(x1r[1])
