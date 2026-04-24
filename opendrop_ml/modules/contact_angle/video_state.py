"""Persist the per-run analysis state so diagnostic figures can be
regenerated without re-running the full video pipeline.

The video-CA pipelines (``_auto_two_pass``, ``_needle_in_two_pass``,
``_single_pass_with_tilt``, ``_auto_two_pass_with_tilt_correction``)
return a result dict with the numeric bits needed to draw the
``cp_scatter`` and ``timeseries`` figures (collected CPs, fitted
contact lines, needle x-intercepts, median tilt angle) plus a
``rotated_first_frame`` image for tilted runs.

This module serialises that dict to:

  {stem}_state.json  -- scalars, lists, dicts, source_path provenance
  {stem}_state.png   -- rotated_first_frame (only when non-None)

The JSON does *not* store the sidecar's filename — the relationship is
1:1 with the same stem, so ``load_state`` derives the PNG path from the
JSON's own filename. Rename the bundle and nothing internal breaks.

Provides ``regenerate_figures(state_json_path)`` to redraw the diagnostic
PNGs against the existing CSV + state bundle.

State-JSON schema
-----------------
The keys listed in ``_STATE_KEYS`` are the stable fields of the saved
state. The ``"phase"`` field tags the pipeline variant that produced
the state and takes one of the following values:

  * ``"pass_1"`` / ``"pass_2"``  — emitted by ``_auto_two_pass`` and
    ``_needle_in_two_pass`` (non-tilted two-pass pipelines).
  * ``"single_pass"``             — emitted by ``_single_pass_with_tilt``.
  * ``"two_pass_tilted"``         — emitted by
    ``_auto_two_pass_with_tilt_correction``.

These values are part of the on-disk schema — downstream tooling may
branch on them, so treat additions as schema changes rather than
silent renames.
"""

import json
import os

import cv2
import numpy as np


_STATE_KEYS = (
    "status",
    "phase",
    "frames_processed",
    "frames_skipped",
    "median_tilt_angle",
    "pass1_cps",
    "left_contact_line",
    "right_contact_line",
    "left_needle_x",
    "right_needle_x",
)


def _to_jsonable(value):
    """Convert numpy scalars/arrays to plain Python so json can encode them."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _sidecar_path(state_json_path):
    """Derive the PNG sidecar path from the JSON filename."""
    base = os.path.basename(state_json_path)
    if base.endswith("_state.json"):
        stem = base[: -len("_state.json")]
    else:
        stem = os.path.splitext(base)[0]
    return os.path.join(os.path.dirname(state_json_path), "{}_state.png".format(stem))


def save_state(result, out_dir, stem, source_path=None):
    """Serialise the pipeline return dict to JSON (+ PNG sidecar).

    Args:
        result: Pipeline return dict (as produced by ``_auto_two_pass`` /
            ``_needle_in_two_pass`` / ``_single_pass_with_tilt``).
        out_dir: Directory to write into. Must already exist.
        stem: Basename stem; files are ``{stem}_state.json`` and
            ``{stem}_state.png``.
        source_path: Original video file or image directory that the run
            analysed. Stored in the JSON as provenance so a renamed bundle
            can still trace back to its inputs. Absolute path is recorded.

    Returns:
        Path to the written JSON file.
    """
    payload = {k: _to_jsonable(result.get(k)) for k in _STATE_KEYS}

    rotated = result.get("rotated_first_frame")
    has_rotated = rotated is not None
    if has_rotated:
        sidecar_path = os.path.join(out_dir, "{}_state.png".format(stem))
        cv2.imwrite(sidecar_path, rotated)
    payload["has_rotated_first_frame"] = has_rotated
    payload["source_path"] = os.path.abspath(source_path) if source_path else None

    json_path = os.path.join(out_dir, "{}_state.json".format(stem))
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return json_path


def load_state(state_json_path):
    """Reconstruct a result-dict-shaped object from disk.

    The rotated-frame sidecar path is derived from the JSON filename
    (``{stem}_state.png`` next to ``{stem}_state.json``) so renames of
    either file do not invalidate the reference, as long as both get the
    same new stem.
    """
    with open(state_json_path, "r") as f:
        payload = json.load(f)

    state = {k: payload.get(k) for k in _STATE_KEYS}
    state["source_path"] = payload.get("source_path")

    sidecar_path = _sidecar_path(state_json_path)
    if payload.get("has_rotated_first_frame") and os.path.isfile(sidecar_path):
        state["rotated_first_frame"] = cv2.imread(sidecar_path)
    else:
        state["rotated_first_frame"] = None
    return state


def regenerate_figures(state_json_path, source_path=None, output_dir=None):
    """Redraw cp_scatter and timeseries PNGs from a saved state bundle.

    Expects the unified CSV to sit alongside the state JSON with the same
    stem (``{stem}_results.csv`` / ``{stem}_state.json``).

    Args:
        state_json_path: Path to the ``{stem}_state.json`` written by
            ``save_state``.
        source_path: Original video file or image directory. When omitted,
            falls back to the ``source_path`` recorded in the JSON. Only
            needed for non-tilted runs; tilted runs use the PNG sidecar.
        output_dir: Where to write the regenerated PNGs. Defaults to the
            directory containing the state JSON.

    Returns:
        ``(timeseries_path, cp_scatter_path)``.
    """
    from opendrop_ml.modules.contact_angle.video_figures import (
        generate_cp_scatter_figure,
        generate_timeseries_figure,
    )

    state_dir = os.path.dirname(state_json_path)
    stem = os.path.basename(state_json_path)[: -len("_state.json")]
    if output_dir is None:
        output_dir = state_dir

    state = load_state(state_json_path)
    if source_path is None:
        source_path = state.get("source_path")

    csv_path = os.path.join(state_dir, "{}_results.csv".format(stem))
    ts_path = os.path.join(output_dir, "{}_timeseries.png".format(stem))
    scatter_path = os.path.join(output_dir, "{}_cp_scatter.png".format(stem))

    generate_timeseries_figure(csv_path, ts_path)

    generate_cp_scatter_figure(
        source_path=source_path,
        pass1_cps=state.get("pass1_cps"),
        contact_lines={
            "left": state.get("left_contact_line"),
            "right": state.get("right_contact_line"),
        },
        output_path=scatter_path,
        left_needle_x=state.get("left_needle_x"),
        right_needle_x=state.get("right_needle_x"),
        rotated_first_frame=state.get("rotated_first_frame"),
    )
    return ts_path, scatter_path
