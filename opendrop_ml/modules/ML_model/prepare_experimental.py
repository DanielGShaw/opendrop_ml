#!/usr/bin/env python
# coding=utf-8

import numpy as np
import cv2
import tensorflow as tf
import time  # for recording timings


def experimental_pred(pred_ds, model, side="both", cluster=True, display=False):
    """Takes an input experimental image, and outputs the predicted contact
    angle based on the contour input model found in this folder:
    './modules/ML_model/'
    """

    start_time = time.time()

    if side == "left":
        # Convert to float32
        input_left = np.array([pred_ds[0, 0]], dtype=np.float32)
        ML_prediction_start_time = time.time()

        # Use signatures approach instead of predict
        input_tensor = tf.convert_to_tensor(input_left)
        prediction_left = model.signatures["serving_default"](
            **{"conv1d_input": input_tensor}
        )
        # Extract from result dictionary if needed
        if isinstance(prediction_left, dict):
            prediction_left = list(prediction_left.values())[0]
        # Convert tensor to numpy if needed
        if hasattr(prediction_left, "numpy"):
            prediction_left = prediction_left.numpy()

        ML_prediction_time = time.time() - ML_prediction_start_time
        analysis_time = time.time() - start_time

        timings = {}
        timings["fit time"] = ML_prediction_time
        timings["analysis time"] = analysis_time

        if prediction_left > 180:
            prediction_left = 180
        elif prediction_left < 0:
            prediction_left = 0

        return prediction_left, timings

    if side == "right":
        # Convert to float32
        input_right = np.array([pred_ds[0, 1]], dtype=np.float32)
        ML_prediction_start_time = time.time()

        # Use signatures approach instead of predict
        input_tensor = tf.convert_to_tensor(input_right)
        prediction_right = model.signatures["serving_default"](
            **{"conv1d_input": input_tensor}
        )
        # Extract from result dictionary if needed
        if isinstance(prediction_right, dict):
            prediction_right = list(prediction_right.values())[0]
        # Convert tensor to numpy if needed
        if hasattr(prediction_right, "numpy"):
            prediction_right = prediction_right.numpy()

        ML_prediction_time = time.time() - ML_prediction_start_time
        analysis_time = time.time() - start_time

        timings = {}
        timings["fit time"] = ML_prediction_time
        timings["analysis time"] = analysis_time

        if prediction_right > 180:
            prediction_right = 180
        elif prediction_right < 0:
            prediction_right = 0

        return prediction_right, timings

    if side == "both":
        # Convert to float32
        pred_ds_float32 = pred_ds.astype(np.float32)
        ML_prediction_start_time = time.time()

        # Use signatures approach instead of predict
        input_tensor = tf.convert_to_tensor(pred_ds_float32)
        predictions = model.signatures["serving_default"](
            **{"conv1d_input": input_tensor}
        )
        # Extract from result dictionary if needed
        if isinstance(predictions, dict):
            predictions = list(predictions.values())[0]
        # Convert tensor to numpy if needed
        if hasattr(predictions, "numpy"):
            predictions = predictions.numpy()

        ML_prediction_time = time.time() - ML_prediction_start_time
        analysis_time = time.time() - start_time

        timings = {}
        timings["fit time"] = ML_prediction_time
        timings["analysis time"] = analysis_time

        if predictions[0] > 180:
            predictions[0] = 180
        elif predictions[0] < 0:
            predictions[0] = 0
        if predictions[1] > 180:
            predictions[1] = 180
        elif predictions[1] < 0:
            predictions[1] = 0

        return predictions, timings


def normalize_ds(ds):
    """Normalize contour windows so the drop apex is at z=1.

    For each contour window: flip y (image coords → Cartesian), divide by
    the apex height, then translate so the minimum x value is 0.
    This matches the normalization used during model training.
    """
    output = []
    for contour in ds:
        contour = contour.astype(float)
        contour[:, 1] = -contour[:, 1] + max(contour[:, 1])
        maxz = abs(max(contour[:, 1]))
        if maxz == 0:
            raise ValueError(
                "Drop contour has zero height after normalization. "
                "The contact angle may be below the model's supported range "
                "(20°–180°), or preprocessing failed to extract the drop "
                "profile correctly."
            )
        normalised = contour / maxz
        if normalised[0, 0] != 0:
            minx = min(normalised[:, 0])
            new = normalised.copy()
            new[:, 0] -= minx
            output.append(new)
        else:
            output.append(normalised)
    return np.array(output)


def resize_image(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Downsample *image* by *scale* using INTER_AREA interpolation."""
    if scale <= 0:
        raise ValueError("Scale factor must be positive.")
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return np.clip(resized, 0, 255).astype(np.uint8)


def prepare_for_model(
    img, input_len=50, n_surface=10, display=False, blur=True, return_cps=False
):
    """Prepare experimental image for the all-angle ML model.

    Uses CPID contact-point identification to extract a 50-point window
    around each CP from the actual contour: 10 real surface contour points
    on the substrate side of the CP, and 40 drop contour points on the
    drop side. Left and right sides are extracted independently (not by
    mirroring), matching the format the model was trained on.
    Normalisation is applied via normalize_ds().

    A resolution check is applied before window extraction: if the drop
    profile is above the resolution plateau threshold (>974 points), or falls
    in the angle-dependent range (565–974) and fails the m*angle+c test, the
    image is downsampled by 0.5x and preprocessing is re-run.

    :param img: BGR image as a numpy array (from cv2.imread)
    :param input_len: Total window size fed to the model (default 50)
    :param n_surface: Surface-side points in the window (default 10)
    :param display: If True, show intermediate plots
    :param blur: If True (default), apply Gaussian blur in edge detection.
                 Set to False to skip blur (legacy behaviour).
    :param return_cps: If True, also return the CPID contact points identified
                        on the original (full-resolution) image, in image pixel
                        coordinates — used to annotate the prediction visually.
    :return: pred_ds of shape (2, input_len, 2) float32 — [left, right]. If
             return_cps is True, returns (pred_ds, contact_points) instead.
    :raises ValueError: If contact point identification fails or profile too short
    """
    from opendrop_ml.modules.preprocessing.preprocessing import preprocess
    from opendrop_ml.modules.fitting.circular_fit import circular_fit

    n_drop = input_len - n_surface  # 40 drop points nearest the CP

    def _run_preprocessing(image):
        img_proc, cps, contour = preprocess(image, display=display, blur=blur)
        if cps is None or len(cps) < 2:
            raise ValueError(
                "All-angle preprocessing failed to identify contact points. "
                "Check image quality or consider manual preprocessing."
            )
        contour = contour.copy()
        idx_l = np.where((contour == cps[0]).all(axis=1))[0][0]
        idx_r = np.where((contour == cps[1]).all(axis=1))[0][0]
        if idx_l > idx_r:
            idx_l, idx_r = idx_r, idx_l
        profile = contour[idx_l : idx_r + 1]
        return img_proc, cps, contour, profile, idx_l, idx_r

    # --- First preprocessing pass ---
    img_processed, CPs_chosen, longest_contour, profile, idx_leftCP, idx_rightCP = (
        _run_preprocessing(img)
    )
    # Contact points on the original, full-resolution image — kept for annotation
    # purposes even if a later resize pass re-derives CPs_chosen on a downsampled
    # copy for the model input windows (those would be in different pixel scale).
    original_CPs = CPs_chosen

    # --- Resolution check ---
    _RESOLUTION_SLOPE = 6.383333333333334
    _RESOLUTION_INTERCEPT = 208.0
    _PROFILE_HIGH = 974
    _PROFILE_MID_LOW = 565
    _PROFILE_WARNING = 234

    profile_len = len(profile)
    needs_resize = False

    if profile_len > _PROFILE_HIGH:
        needs_resize = True
    elif _PROFILE_MID_LOW < profile_len < _PROFILE_HIGH:
        try:
            circ_angles, _, _, _, _, _ = circular_fit(profile)
            threshold = _RESOLUTION_SLOPE * min(circ_angles) + _RESOLUTION_INTERCEPT
            if profile_len > threshold:
                needs_resize = True
        except Exception:
            pass  # circular_fit failed — skip the angle-dependent check
    elif profile_len < _PROFILE_WARNING:
        print(
            f"WARNING: Profile length ({profile_len}) is low. "
            "Predictions may be unreliable."
        )

    if needs_resize:
        img_small = resize_image(img, scale=0.5)
        img_processed, CPs_chosen, longest_contour, profile, idx_leftCP, idx_rightCP = (
            _run_preprocessing(img_small)
        )

    # --- Extract independent left and right windows from the actual contour ---
    # Left side: n_surface points before leftCP (surface) + n_drop points after (drop)
    if idx_leftCP < n_surface:
        left_side = longest_contour[:input_len]
    else:
        left_side = longest_contour[idx_leftCP - n_surface : idx_leftCP + n_drop]

    # Right side: n_drop points before rightCP (drop) + n_surface points after (surface)
    if idx_rightCP > len(longest_contour) - n_surface:
        right_side = longest_contour[-input_len:]
    else:
        right_side = longest_contour[idx_rightCP - n_drop : idx_rightCP + n_surface]

    if len(left_side) != input_len or len(right_side) != input_len:
        raise ValueError(
            f"Contour window extraction failed — expected {input_len} points "
            f"per side, got left={len(left_side)}, right={len(right_side)}. "
            "Profile may be too short near contact points."
        )

    # Validate CP position is within the model's trained range (n_surface ± 5)
    min_surface = n_surface - 5  # minimum 5 surface points
    if idx_leftCP < n_surface and idx_leftCP < min_surface:
        raise ValueError(
            f"Left contact point has only {idx_leftCP} surface points "
            f"(minimum {min_surface} required). The contact point is too "
            "close to the edge of the contour for a valid model input."
        )
    right_surface = len(longest_contour) - 1 - idx_rightCP
    if right_surface < n_surface and right_surface < min_surface:
        raise ValueError(
            f"Right contact point has only {right_surface} surface points "
            f"(minimum {min_surface} required). The contact point is too "
            "close to the edge of the contour for a valid model input."
        )

    # Reverse left side (start from drop contour) and reflect x (training convention)
    left_side = left_side[::-1].copy()
    left_side[:, 0] = -left_side[:, 0]

    pred_ds = np.array([left_side, right_side])
    pred_ds = normalize_ds(pred_ds)

    if return_cps:
        return pred_ds.astype(np.float32), original_CPs
    return pred_ds.astype(np.float32)
