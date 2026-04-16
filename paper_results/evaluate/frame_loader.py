"""Unified frame loading from video files or image directories.

Provides a single interface for iterating over frames regardless of
whether the source is a video file (.avi/.mp4/.mov) or a directory
of images (.bmp/.png/.jpg etc.).

Based on extract_frames() and get_middle_frame() logic from
files_from_old_computer/video_analysis/isaac/*/needle_in_clean.ipynb
"""

import os
import re

import cv2
import numpy as np


# Supported formats
IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif")
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov")


def natural_sort_key(s):
    """Sort key that handles embedded numbers naturally.

    'img2.png' sorts before 'img10.png'.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))]


def load_frames(source_path, start_sec=0, end_sec=-1, frame_step=1):
    """Load frames from a video file or directory of images.

    Args:
        source_path: Path to .avi/.mp4/.mov file, or directory of images.
        start_sec: Start time in seconds (video only, ignored for directories).
        end_sec: End time in seconds, -1 for whole video (video only).
        frame_step: Process every Nth frame.

    Yields:
        (frame, frame_number, metadata) where:
            frame: np.ndarray (BGR image)
            frame_number: int (0-based sequential index)
            metadata: dict with 'fps' (video) or 'filename' (directory)
    """
    if os.path.isdir(source_path):
        yield from _load_from_directory(source_path, frame_step)
    elif os.path.isfile(source_path):
        ext = os.path.splitext(source_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            yield from _load_from_video(source_path, start_sec, end_sec, frame_step)
        else:
            raise ValueError(
                "Unsupported file format '{}'. Expected one of: {}".format(
                    ext, VIDEO_EXTENSIONS
                )
            )
    else:
        raise FileNotFoundError("Source path not found: {}".format(source_path))


def get_middle_frame(source_path):
    """Return the middle frame from a video or image directory for preview.

    Args:
        source_path: Path to video file or image directory.

    Returns:
        (frame, frame_number, metadata) tuple.
    """
    if os.path.isdir(source_path):
        filenames = sorted(
            (f for f in os.listdir(source_path)
             if f.lower().endswith(IMAGE_EXTENSIONS)),
            key=natural_sort_key,
        )
        if not filenames:
            raise ValueError("No images found in {}".format(source_path))
        mid_idx = len(filenames) // 2
        img_path = os.path.join(source_path, filenames[mid_idx])
        frame = cv2.imread(img_path)
        if frame is None:
            raise IOError("Failed to read image: {}".format(img_path))
        return frame, mid_idx, {"filename": filenames[mid_idx]}

    # Video file
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: {}".format(source_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        mid = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if not ret:
            raise IOError("Failed to read frame {} from video".format(mid))
        return frame, mid, {"fps": fps}
    finally:
        cap.release()


def get_frame_count(source_path):
    """Return total number of frames (respecting frame_step=1).

    Args:
        source_path: Path to video file or image directory.

    Returns:
        int: Total frame count.
    """
    if os.path.isdir(source_path):
        return len([
            f for f in os.listdir(source_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: {}".format(source_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_from_directory(directory, frame_step):
    """Yield frames from a directory of images."""
    filenames = sorted(
        (f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS)),
        key=natural_sort_key,
    )
    if not filenames:
        raise ValueError("No images found in {}".format(directory))

    for i, fname in enumerate(filenames):
        if i % frame_step != 0:
            continue
        img_path = os.path.join(directory, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print("  WARNING: Failed to read image: {}".format(img_path))
            continue
        yield frame, i, {"filename": fname}


def _load_from_video(video_path, start_sec, end_sec, frame_step):
    """Yield frames from a video file within the given time range."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: {}".format(video_path))

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_sec * fps) if start_sec > 0 else 0
        if end_sec < 0:
            end_frame = total_frames
        else:
            end_frame = min(int(end_sec * fps), total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        yielded_count = 0
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_idx - start_frame) % frame_step == 0:
                yield frame, frame_idx, {"fps": fps}
                yielded_count += 1
            frame_idx += 1
    finally:
        cap.release()
