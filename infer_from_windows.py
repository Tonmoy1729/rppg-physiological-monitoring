from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import signal

from model_defs import HRResTCNAttention, RRRecordingNet, HybridBPNet


FS = 30.0
CARDIAC_BAND = (0.7, 3.0)
RESP_BAND = (0.10, 0.50)

_MODEL_CACHE = {}


def load_json(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_dicts(obj):
    """Yield nested dictionaries from a checkpoint metadata object."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            if isinstance(value, dict):
                yield from _iter_dicts(value)


def _normalization_from_checkpoint(meta: dict, prefix: str, model_path: str | Path):
    """Read target mean/std from a revised checkpoint.

    The revision scripts may store normalization values under slightly different
    key names. This helper accepts the common variants and also supports an
    optional target_norm.json beside the model as a fallback.
    """
    prefix = prefix.lower()
    candidates = [
        (f"{prefix}_mean", f"{prefix}_std"),
        (f"{prefix}_m", f"{prefix}_s"),
        ("target_mean", "target_std"),
        ("y_mean", "y_std"),
        ("mean", "std"),
    ]

    for d in _iter_dicts(meta):
        for mean_key, std_key in candidates:
            if mean_key in d and std_key in d:
                mean = float(d[mean_key])
                std = float(d[std_key])
                if np.isfinite(mean) and np.isfinite(std) and std > 0:
                    return mean, std

    # Optional fallback. The app no longer requires this file, but if an older
    # export left one beside the model we can still use it.
    sidecar = Path(model_path).with_name("target_norm.json")
    if sidecar.exists():
        norm = load_json(sidecar)
        for mean_key, std_key in candidates:
            if mean_key in norm and std_key in norm:
                return float(norm[mean_key]), float(norm[std_key])

    visible_keys = sorted(meta.keys()) if isinstance(meta, dict) else []
    raise KeyError(
        f"Could not find {prefix.upper()} target normalization inside {Path(model_path).name}. "
        f"Checkpoint top-level keys: {visible_keys}. "
        "The model weights loaded, but inference also needs the training-target mean/std. "
        "If this error appears, send me the printed checkpoint keys and I will map the exact fields."
    )

def _state_from_checkpoint(obj):
    if isinstance(obj, dict):
        for key in ("model", "model_state", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], obj
        # A plain state_dict is also a dict.
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj, {}
    raise RuntimeError("Unsupported PyTorch checkpoint format.")


def _load_state(model, path: str | Path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    state, meta = _state_from_checkpoint(obj)
    # Remove DataParallel prefix when necessary.
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {path.name}. "
            "Use the revised checkpoint produced by the matching revision script.\n\n"
            f"Original error:\n{exc}"
        ) from exc
    model.to(device).eval()
    return model, meta


def _interp_window(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32).copy()
    if x.shape != (900, 3):
        raise ValueError(f"Expected one raw RGB window [900,3], got {x.shape}")
    idx = np.arange(len(x))
    for c in range(3):
        good = np.isfinite(x[:, c])
        if good.sum() < 10:
            raise ValueError("Too few finite RGB values in a window.")
        if not good.all():
            x[~good, c] = np.interp(idx[~good], idx[good], x[good, c])
    return x


def _relative_detrended(rgb: np.ndarray, percentage: bool = False) -> np.ndarray:
    x = _interp_window(rgb)
    scale = 100.0 if percentage else 1.0
    x = scale * (x / (x.mean(axis=0, keepdims=True) + 1e-8) - 1.0)
    return signal.detrend(x, axis=0, type="linear").astype(np.float32)


def prepare_hr_windows(raw_windows: np.ndarray) -> np.ndarray:
    """Revised HR input: relative RGB -> detrend -> per-channel standardization."""
    out = []
    for rgb in raw_windows:
        x = _relative_detrended(rgb, percentage=False)
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
        out.append(x.T)
    return np.asarray(out, dtype=np.float32)  # [N,3,900]


def _sos_bandpass(low: float, high: float, order: int = 3):
    return signal.butter(order, [low, high], btype="bandpass", fs=FS, output="sos")


def prepare_rr_windows(raw_windows: np.ndarray) -> np.ndarray:
    """RR-v2 input: normalized RGB + 0.10-0.50 Hz RGB + temporal gradient."""
    sos = _sos_bandpass(*RESP_BAND, order=3)
    out = []
    for rgb in raw_windows:
        base = _relative_detrended(rgb, percentage=False)
        base = (base - base.mean(axis=0, keepdims=True)) / (base.std(axis=0, keepdims=True) + 1e-6)
        resp = signal.sosfiltfilt(sos, base, axis=0).astype(np.float32)
        grad = np.diff(base, axis=0, prepend=base[:1]).astype(np.float32)
        x = np.concatenate([base, resp, grad], axis=1)
        out.append(x.T)
    return np.asarray(out, dtype=np.float32)  # [N,9,900]


def prepare_bp_rgb_windows(raw_windows: np.ndarray) -> np.ndarray:
    """Amplitude-preserving BP CNN branch: 100*(RGB/mean-1) + detrending."""
    return np.asarray(
        [_relative_detrended(rgb, percentage=True).T for rgb in raw_windows],
        dtype=np.float32,
    )  # [N,3,900]


def _channel_features(raw: np.ndarray, pulse: np.ndarray) -> list[float]:
    raw = np.asarray(raw, float)
    pulse = np.asarray(pulse, float)

    raw_std = np.std(raw)
    raw_rms = np.sqrt(np.mean(raw ** 2))
    raw_p2p = np.ptp(raw)

    pulse_std = np.std(pulse)
    pulse_rms = np.sqrt(np.mean(pulse ** 2))
    pulse_p2p = np.ptp(pulse)

    env = np.abs(signal.hilbert(pulse))
    env_mean = np.mean(env)
    env_std = np.std(env)
    env_cv = env_std / (env_mean + 1e-8)
    env_p90 = np.percentile(env, 90)

    f, pxx = signal.welch(pulse, fs=FS, nperseg=min(256, len(pulse)))
    band = (f >= CARDIAC_BAND[0]) & (f <= CARDIAC_BAND[1])
    fb = f[band]
    pb = np.maximum(pxx[band], 0.0)
    if len(fb) == 0 or np.sum(pb) <= 0:
        peak_frequency = 0.0
        band_power = 0.0
        spectral_centroid = 0.0
        spectral_entropy = 0.0
        peak_fraction = 0.0
    else:
        peak_frequency = float(fb[np.argmax(pb)])
        band_power = float(np.trapezoid(pb, fb)) if len(fb) > 1 else float(pb[0])
        total = float(np.sum(pb)) + 1e-12
        spectral_centroid = float(np.sum(fb * pb) / total)
        prob = pb / total
        nz = prob > 0
        spectral_entropy = float(-np.sum(prob[nz] * np.log(prob[nz])) / np.log(max(2, len(prob))))
        peak_fraction = float(100.0 * np.max(pb) / total)

    return [
        raw_std,
        raw_rms,
        raw_p2p,
        pulse_std,
        pulse_rms,
        pulse_p2p,
        env_mean,
        env_std,
        env_cv,
        env_p90,
        peak_frequency,
        band_power,
        spectral_centroid,
        spectral_entropy,
        peak_fraction,
    ]


def extract_bp_session_features(raw_windows: np.ndarray) -> np.ndarray:
    """Return the 96-D session feature vector used by the robust BP branch.

    Per 30-s window: 15 features/channel x 3 channels + 3 zero-lag pulse
    correlations = 48. Session aggregation: feature-wise mean + std = 96.
    """
    sos = _sos_bandpass(*CARDIAC_BAND, order=3)
    rows = []
    for rgb in raw_windows:
        rel = _relative_detrended(rgb, percentage=True)  # [900,3]
        pulse = signal.sosfiltfilt(sos, rel, axis=0)
        feat = []
        for c in range(3):
            feat.extend(_channel_features(rel[:, c], pulse[:, c]))

        corr = np.corrcoef(pulse.T)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            v = corr[i, j]
            feat.append(float(v) if np.isfinite(v) else 0.0)
        rows.append(feat)

    rows = np.asarray(rows, dtype=np.float64)
    if rows.shape[1] != 48:
        raise RuntimeError(f"Expected 48 BP features per window, got {rows.shape}")
    session = np.concatenate([np.mean(rows, axis=0), np.std(rows, axis=0)], axis=0)
    if session.shape != (96,):
        raise RuntimeError(f"Expected 96 session features, got {session.shape}")
    return session.astype(np.float64)


def _apply_bp_feature_bundle(features_96: np.ndarray, bundle: dict):
    required = ["train_median", "clip_low", "clip_high", "scaler", "sbp_model"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(f"BP feature bundle missing keys: {missing}")

    x = np.asarray(features_96, dtype=float).reshape(1, -1)
    med = np.asarray(bundle["train_median"], dtype=float)
    lo = np.asarray(bundle["clip_low"], dtype=float)
    hi = np.asarray(bundle["clip_high"], dtype=float)
    if x.shape[1] != len(med):
        raise ValueError(f"BP feature count mismatch: app={x.shape[1]}, bundle={len(med)}")

    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.broadcast_to(med, x.shape)[bad]
    x = np.clip(x, lo, hi)
    x_scaled = bundle["scaler"].transform(x)
    sbp = float(bundle["sbp_model"].predict(x_scaled)[0])
    return x_scaled.astype(np.float32), sbp


def _load_bundle(
    hr_model_path,
    rr_model_path,
    bp_hybrid_model_path,
    bp_feature_bundle_path,
    device,
):
    key = tuple(map(str, [
        hr_model_path,
        rr_model_path,
        bp_hybrid_model_path,
        bp_feature_bundle_path,
        device,
    ]))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    hr_model, hr_meta = _load_state(HRResTCNAttention(), hr_model_path, device)
    rr_model, rr_meta = _load_state(RRRecordingNet(), rr_model_path, device)
    bp_model, bp_meta = _load_state(HybridBPNet(96), bp_hybrid_model_path, device)

    hr_mean, hr_std = _normalization_from_checkpoint(hr_meta, "hr", hr_model_path)
    rr_mean, rr_std = _normalization_from_checkpoint(rr_meta, "rr", rr_model_path)

    bp_feature_bundle_path = Path(bp_feature_bundle_path)
    if not bp_feature_bundle_path.exists():
        raise FileNotFoundError(
            f"BP feature bundle not found: {bp_feature_bundle_path}\n"
            "Run export_bp_bundle.py once to create it."
        )
    bp_features = joblib.load(bp_feature_bundle_path)

    bundle = {
        "hr_model": hr_model,
        "hr_meta": hr_meta,
        "hr_mean": hr_mean,
        "hr_std": hr_std,
        "rr_model": rr_model,
        "rr_meta": rr_meta,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "bp_model": bp_model,
        "bp_meta": bp_meta,
        "bp_features": bp_features,
    }
    _MODEL_CACHE[key] = bundle
    return bundle

def run_all_models_on_windows(
    x_windows_path,
    hr_model_path,
    rr_model_path,
    bp_hybrid_model_path,
    bp_feature_bundle_path,
    batch_size=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_windows = np.load(x_windows_path).astype(np.float32)
    if raw_windows.ndim != 3 or raw_windows.shape[1:] != (900, 3):
        raise ValueError(
            f"Revised app expects raw windows [N,900,3]; got {raw_windows.shape}. "
            "Regenerate the video with the revised roi_video_pp.py."
        )

    b = _load_bundle(
        hr_model_path,
        rr_model_path,
        bp_hybrid_model_path,
        bp_feature_bundle_path,
        device,
    )

    # ---------------- HR: window predictions -> recording mean ----------------
    hr_x = torch.from_numpy(prepare_hr_windows(raw_windows))
    hr_preds_n = []
    with torch.no_grad():
        for i in range(0, len(hr_x), batch_size):
            hr_preds_n.append(b["hr_model"](hr_x[i:i + batch_size].to(device)).cpu().numpy())
    hr_preds_n = np.concatenate(hr_preds_n).reshape(-1)
    hr_preds = hr_preds_n * b["hr_std"] + b["hr_mean"]
    hr_recording = float(np.mean(hr_preds))

    # ---------------- RR: recording-level model ----------------
    rr_x = torch.from_numpy(prepare_rr_windows(raw_windows)).to(device)
    with torch.no_grad():
        rr_n = float(b["rr_model"](rr_x).cpu().numpy().reshape(-1)[0])
    rr_recording = rr_n * b["rr_std"] + b["rr_mean"]

    # ---------------- BP: robust SBP + hybrid DBP ----------------
    bp_feat_raw = extract_bp_session_features(raw_windows)
    bp_feat_scaled, sbp_robust = _apply_bp_feature_bundle(bp_feat_raw, b["bp_features"])

    bp_rgb = torch.from_numpy(prepare_bp_rgb_windows(raw_windows)).to(device)
    bp_feat_tensor = torch.from_numpy(bp_feat_scaled).to(device)
    with torch.no_grad():
        bp_n = b["bp_model"](bp_rgb, bp_feat_tensor).cpu().numpy().reshape(-1)

    meta = b["bp_meta"]
    for k in ("sbp_mean", "sbp_std", "dbp_mean", "dbp_std"):
        if k not in meta:
            raise KeyError(
                f"Hybrid BP checkpoint is missing '{k}'. Use best_bp_hybrid.pt saved by Script 17."
            )
    hybrid_sbp = float(bp_n[0] * float(meta["sbp_std"]) + float(meta["sbp_mean"]))
    dbp_hybrid = float(bp_n[1] * float(meta["dbp_std"]) + float(meta["dbp_mean"]))

    pred_df = pd.DataFrame(
        {
            "window_index": np.arange(len(raw_windows)),
            "hr_pred": hr_preds,
        }
    )

    summary = {
        "num_windows": int(len(raw_windows)),
        "hr_mean": hr_recording,
        "hr_window_std": float(np.std(hr_preds, ddof=0)),
        "rr_mean": float(rr_recording),
        "sbp_mean": float(sbp_robust),
        "dbp_mean": float(dbp_hybrid),
        "hybrid_sbp_internal": hybrid_sbp,
        "bp_strategy": "SBP=robust feature Ridge; DBP=Hybrid BP-v2",
        "device": str(device),
    }
    return pred_df, summary
