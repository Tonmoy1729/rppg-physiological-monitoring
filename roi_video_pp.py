from __future__ import annotations

from pathlib import Path
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import resample


mp_face_mesh = mp.solutions.face_mesh

# Ordered MediaPipe face-oval vertices. The mask covers the full facial skin
# region and the two eye polygons are removed from that mask.
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def _points(landmarks, width: int, height: int, indices: list[int]) -> np.ndarray:
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        x = int(np.clip(lm.x * width, 0, width - 1))
        y = int(np.clip(lm.y * height, 0, height - 1))
        pts.append((x, y))
    return np.asarray(pts, dtype=np.int32)


def _resize(frame_bgr: np.ndarray, max_width: int | None) -> np.ndarray:
    if max_width is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    scale = max_width / float(w)
    return cv2.resize(
        frame_bgr,
        (max_width, int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _skin_mean_rgb(frame_bgr: np.ndarray, face_pts: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray):
    if face_pts.shape[0] < 3:
        return None

    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [face_pts], 255)
    if left_eye.shape[0] >= 3:
        cv2.fillPoly(mask, [left_eye], 0)
    if right_eye.shape[0] >= 3:
        cv2.fillPoly(mask, [right_eye], 0)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pixels = frame_rgb[mask > 0]
    if pixels.size == 0:
        return None
    return pixels.mean(axis=0).astype(np.float32)


def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() == 0:
        raise RuntimeError("A complete RGB channel is missing after face extraction.")
    if good.sum() == 1:
        return np.full_like(x, x[good][0], dtype=np.float32)
    return np.interp(idx, idx[good], x[good]).astype(np.float32)


def clean_trace(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32)
    out = np.empty_like(trace)
    for c in range(trace.shape[1]):
        out[:, c] = fill_nan_1d(trace[:, c])
    return out


def resample_trace(trace: np.ndarray, original_fs: float, target_fs: float = 30.0) -> np.ndarray:
    if original_fs <= 0:
        raise ValueError("original_fs must be positive")
    duration = len(trace) / float(original_fs)
    target_len = int(round(duration * target_fs))
    if target_len < 1:
        raise RuntimeError("Invalid resampled trace length.")
    return resample(trace, target_len, axis=0).astype(np.float32)


def extract_face_rgb_trace(
    video_path: str | Path,
    max_seconds: float | None = None,
    frame_stride: int = 1,
    resize_width: int | None = 384,
    detect_every: int = 5,
    min_frames: int = 300,
):
    """Extract one eye-excluded full-face RGB mean per processed frame.

    Returns
    -------
    trace_raw : ndarray [T, 3]
        Raw RGB means in R,G,B order.
    face_ok : ndarray [T]
        Whether a valid facial mask was available at that sample.
    metadata : dict
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    max_original_frames = None
    if max_seconds is not None:
        max_original_frames = int(round(max_seconds * fps))

    trace = []
    face_ok = []
    last_polys = None
    processed_count = 0
    frame_i = 0
    original_seen = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_original_frames is not None and frame_i >= max_original_frames:
                break

            original_seen += 1
            if frame_i % frame_stride != 0:
                frame_i += 1
                continue

            frame_bgr = _resize(frame_bgr, resize_width)
            h, w = frame_bgr.shape[:2]
            processed_count += 1

            if last_polys is None or processed_count % detect_every == 1:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(frame_rgb)
                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0]
                    last_polys = (
                        _points(lm, w, h, FACE_OVAL_IDX),
                        _points(lm, w, h, LEFT_EYE_IDX),
                        _points(lm, w, h, RIGHT_EYE_IDX),
                    )
                else:
                    last_polys = None

            rgb = None
            if last_polys is not None:
                rgb = _skin_mean_rgb(frame_bgr, *last_polys)

            if rgb is None:
                trace.append([np.nan, np.nan, np.nan])
                face_ok.append(False)
            else:
                trace.append(rgb)
                face_ok.append(True)

            frame_i += 1

    cap.release()

    if len(trace) < min_frames:
        raise RuntimeError(f"Too few processed frames: {len(trace)}; need at least {min_frames}.")

    trace = np.asarray(trace, dtype=np.float32)
    face_ok = np.asarray(face_ok, dtype=bool)
    trace_fs = fps / float(frame_stride)

    metadata = {
        "video_path": str(video_path),
        "source_fps": fps,
        "trace_fps": trace_fs,
        "frame_stride": int(frame_stride),
        "detect_every": int(detect_every),
        "resize_width": -1 if resize_width is None else int(resize_width),
        "num_original_frames_seen": int(original_seen),
        "num_trace_samples": int(len(trace)),
        "face_detection_rate": float(face_ok.mean()),
        "roi": "full facial skin excluding both eye polygons",
    }
    return trace, face_ok, metadata


def make_windows(
    trace_30hz: np.ndarray,
    target_fs: float = 30.0,
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
):
    """Return raw, non-normalized RGB windows [N, 900, 3]."""
    window_len = int(round(target_fs * window_sec))
    stride_len = int(round(target_fs * stride_sec))
    if len(trace_30hz) < window_len:
        raise RuntimeError(
            f"Video too short after resampling: need {window_len} samples "
            f"({window_sec:.0f} s), got {len(trace_30hz)}."
        )

    windows, starts = [], []
    for start in range(0, len(trace_30hz) - window_len + 1, stride_len):
        windows.append(trace_30hz[start : start + window_len].astype(np.float32))
        starts.append(start)
    return np.asarray(windows, dtype=np.float32), np.asarray(starts, dtype=np.int64)


def prepare_video_to_windows(
    video_path: str | Path,
    out_dir: str | Path,
    target_fs: float = 30.0,
    window_sec: float = 30.0,
    stride_sec: float = 30.0,
    max_seconds: float | None = 90.0,
    max_frame_width: int | None = 384,
    frame_stride: int = 1,
    detect_every: int = 5,
    min_face_detection_rate: float = 0.60,
):
    """Revised app preprocessing: video -> eye-excluded RGB -> 30 Hz -> 30-s raw RGB windows.

    The saved ``x_windows.npy`` is [N, 900, 3] and is intentionally raw.
    Endpoint-specific normalization is performed later by ``infer_from_windows.py``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, face_ok, metadata = extract_face_rgb_trace(
        video_path=video_path,
        max_seconds=max_seconds,
        frame_stride=frame_stride,
        resize_width=max_frame_width,
        detect_every=detect_every,
        min_frames=max(300, int(round(window_sec * target_fs * 0.5))),
    )

    if metadata["face_detection_rate"] < min_face_detection_rate:
        raise RuntimeError(
            f"Face detected in only {metadata['face_detection_rate'] * 100:.1f}% of processed frames. "
            "Use a stable frontal video with clearer lighting."
        )

    clean = clean_trace(raw)
    trace_30hz = resample_trace(clean, metadata["trace_fps"], target_fs)
    windows, starts = make_windows(trace_30hz, target_fs, window_sec, stride_sec)

    np.save(out_dir / "x_windows.npy", windows.astype(np.float32))
    np.save(out_dir / "trace_30hz.npy", trace_30hz.astype(np.float32))
    np.save(out_dir / "face_ok.npy", face_ok.astype(bool))

    metadata.update(
        {
            "target_fs": float(target_fs),
            "window_sec": float(window_sec),
            "stride_sec": float(stride_sec),
            "window_shape": list(windows.shape),
            "window_starts": starts.tolist(),
            "num_windows": int(len(windows)),
            "normalization": "none here; endpoint-specific preprocessing occurs during inference",
        }
    )

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return windows, metadata
