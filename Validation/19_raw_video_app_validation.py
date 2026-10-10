from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import signal


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"D:\mcd_rppg\PythonProject"
)

REV_ROOT = (
    PROJECT_ROOT
    / "Revision work"
)

APP_ROOT = Path(
    r"D:\mcd_rppg\hf_rppg_app"
)


# ------------------------------------------------------------
# REVISED JOURNAL SPLIT
# ------------------------------------------------------------

MANIFEST_CSV = (
    REV_ROOT
    / "preprocessed_journal_split"
    / "manifest.csv"
)


# ------------------------------------------------------------
# REVISED FULL-FACE / NO-EYES RGB NPZ FILES
# ------------------------------------------------------------

RAW_NPZ_DIR = (
    REV_ROOT
    / "npz_face_no_eyes_30s"
)


# ------------------------------------------------------------
# NEW CLEAN OUTPUT DIRECTORY
# ------------------------------------------------------------

OUT_DIR = (
    REV_ROOT
    / "runs_app_revision"
    / "raw_npz_validation_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# MODEL FILES
# ------------------------------------------------------------

HR_MODEL = (
    APP_ROOT
    / "models"
    / "hr"
    / "best_hr_model.pt"
)

RR_MODEL = (
    APP_ROOT
    / "models"
    / "rr"
    / "best_rr_model.pt"
)

BP_HYBRID_MODEL = (
    APP_ROOT
    / "models"
    / "bp"
    / "best_bp_hybrid.pt"
)

BP_FEATURE_BUNDLE = (
    APP_ROOT
    / "models"
    / "bp"
    / "bp_feature_bundle.joblib"
)


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

PRED_CSV = (
    OUT_DIR
    / "raw_npz_predictions.csv"
)

FAIL_CSV = (
    OUT_DIR
    / "raw_npz_failures.csv"
)

METRICS_CSV = (
    OUT_DIR
    / "recording_level_metrics.csv"
)

BOOT_CSV = (
    OUT_DIR
    / "subject_cluster_bootstrap_ci.csv"
)

CAMERA_CSV = (
    OUT_DIR
    / "metrics_by_camera.csv"
)

STATE_CSV = (
    OUT_DIR
    / "metrics_by_state.csv"
)

BP_SESSION_CSV = (
    OUT_DIR
    / "bp_session_level_metrics.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "validation_summary.json"
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

BATCH_SIZE = 64

# ------------------------------------------------------------
# FIRST RUN:
# ------------------------------------------------------------
# Keep this as 5.
#
# After we inspect the first five recordings,
# change to:
#
# LIMIT = None
#
# for all 537 held-out test recordings.
# ------------------------------------------------------------

LIMIT = None


# ------------------------------------------------------------
# BOOTSTRAP
# ------------------------------------------------------------

N_BOOT = 5000

BOOT_SEED = 42


# ------------------------------------------------------------
# PPG / HR SETTINGS
# ------------------------------------------------------------

PPG_FS = 30.0

HR_LOW_HZ = 0.7

HR_HIGH_HZ = 3.0

HR_FILTER_ORDER = 3


# ============================================================
# IMPORT THE EXACT APP INFERENCE CODE
# ============================================================

sys.path.insert(
    0,
    str(APP_ROOT),
)

from infer_from_windows import run_all_models_on_windows


# ============================================================
# GENERAL HELPERS
# ============================================================

def scalar_float(x):

    try:

        arr = np.asarray(
            x
        ).reshape(-1)

        if len(arr) == 0:

            return np.nan

        value = float(
            arr[0]
        )

        if not np.isfinite(
            value
        ):

            return np.nan

        return value

    except Exception:

        return np.nan


def find_column(
    df: pd.DataFrame,
    candidates,
):

    column_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for name in candidates:

        key = str(
            name
        ).lower()

        if key in column_map:

            return column_map[
                key
            ]

    return None


def append_row(
    row: dict,
    path: Path,
):

    pd.DataFrame(
        [row]
    ).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

def check_required_files():

    required = [
        MANIFEST_CSV,
        RAW_NPZ_DIR,
        HR_MODEL,
        RR_MODEL,
        BP_HYBRID_MODEL,
        BP_FEATURE_BUNDLE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required file/folder:\n\n"
            + "\n".join(
                missing
            )
        )


# ============================================================
# RAW NPZ INDEX
# ============================================================

def build_raw_npz_index():

    print(
        "\nIndexing revised raw RGB NPZ files:"
    )

    print(
        RAW_NPZ_DIR
    )

    files = sorted(
        RAW_NPZ_DIR.glob(
            "*.npz"
        )
    )

    print(
        f"NPZ files found: "
        f"{len(files)}"
    )

    if len(files) == 0:

        raise RuntimeError(
            f"No NPZ files found in:\n"
            f"{RAW_NPZ_DIR}"
        )

    index = {}

    for path in files:

        recording_id = (
            path.stem
        )

        index[
            recording_id
        ] = path

    print(
        f"Recording IDs represented: "
        f"{len(index)}"
    )

    return index


# ============================================================
# LOAD EXACT REVISED windows_raw
# ============================================================

def load_raw_recording_npz(
    recording_id: str,
    raw_index: dict,
):

    npz_path = raw_index.get(
        str(recording_id)
    )

    if npz_path is None:

        target = str(
            recording_id
        ).lower()

        for key, path in raw_index.items():

            if key.lower() == target:

                npz_path = path
                break

    if npz_path is None:

        raise FileNotFoundError(
            "No revised raw NPZ found for:\n"
            f"{recording_id}"
        )

    z = np.load(
        npz_path,
        allow_pickle=True,
    )

    if "windows_raw" not in z.files:

        raise KeyError(
            "'windows_raw' not found in:\n"
            f"{npz_path}\n\n"
            f"Available keys:\n"
            f"{z.files}"
        )

    windows = np.asarray(
        z[
            "windows_raw"
        ],
        dtype=np.float32,
    )

    if windows.ndim != 3:

        raise RuntimeError(
            "windows_raw must be 3-dimensional.\n"
            f"Got shape: {windows.shape}"
        )

    if windows.shape[1:] != (
        900,
        3,
    ):

        raise RuntimeError(
            "Expected windows_raw shape "
            "[N, 900, 3].\n"
            f"Got: {windows.shape}"
        )

    metadata = {}

    scalar_keys = [
        "original_fps",
        "resample_fs",
        "window_sec",
        "window_stride_sec",
        "frame_stride",
        "resize_width",
        "detect_every",
        "detection_rate",
    ]

    for key in scalar_keys:

        if key in z.files:

            metadata[
                key
            ] = scalar_float(
                z[key]
            )

    if "window_starts" in z.files:

        metadata[
            "window_starts"
        ] = np.asarray(
            z[
                "window_starts"
            ]
        ).astype(
            int
        )

    if "roi_type" in z.files:

        metadata[
            "roi_type"
        ] = str(
            z[
                "roi_type"
            ].item()
        )

    if "channel_order" in z.files:

        metadata[
            "channel_order"
        ] = [
            str(x)
            for x in np.asarray(
                z[
                    "channel_order"
                ]
            )
        ]

    return (
        windows,
        npz_path,
        metadata,
    )


# ============================================================
# PPG -> HR TARGET
# ============================================================

def ppg_to_hr_bpm(
    ppg,
    fs=PPG_FS,
):

    """
    Derive HR from the synchronized 30-Hz contact PPG window.

    Processing:
        detrend
        0.7-3.0 Hz Butterworth bandpass
        Welch PSD
        dominant spectral peak
        Hz -> beats/minute
    """

    x = np.asarray(
        ppg,
        dtype=np.float64,
    ).reshape(-1)

    good = np.isfinite(
        x
    )

    if np.sum(
        good
    ) < 100:

        return np.nan

    if not np.all(
        good
    ):

        idx = np.arange(
            len(x)
        )

        x = np.interp(
            idx,
            idx[good],
            x[good],
        )

    x = signal.detrend(
        x,
        type="linear",
    )

    nyquist = (
        fs / 2.0
    )

    b, a = signal.butter(
        HR_FILTER_ORDER,
        [
            HR_LOW_HZ
            / nyquist,

            HR_HIGH_HZ
            / nyquist,
        ],
        btype="bandpass",
    )

    try:

        xf = signal.filtfilt(
            b,
            a,
            x,
        )

    except Exception:

        return np.nan

    nperseg = min(
        900,
        len(xf),
    )

    freqs, psd = signal.welch(
        xf,
        fs=fs,
        nperseg=nperseg,
        detrend=False,
        scaling="density",
    )

    band = (
        (freqs >= HR_LOW_HZ)
        & (freqs <= HR_HIGH_HZ)
    )

    if not np.any(
        band
    ):

        return np.nan

    band_freqs = (
        freqs[band]
    )

    band_psd = (
        psd[band]
    )

    if (
        len(band_psd) == 0
        or not np.any(
            np.isfinite(
                band_psd
            )
        )
    ):

        return np.nan

    peak_index = int(
        np.nanargmax(
            band_psd
        )
    )

    peak_hz = float(
        band_freqs[
            peak_index
        ]
    )

    hr_bpm = (
        peak_hz
        * 60.0
    )

    return float(
        hr_bpm
    )


# ============================================================
# LOAD REFERENCE LABELS FROM PROCESSED WINDOW NPZS
# ============================================================

def load_reference_data(
    manifest_group: pd.DataFrame,
):

    if "output" not in manifest_group.columns:

        raise KeyError(
            "Manifest does not contain "
            "'output' column."
        )

    hr_values = []

    pulse_values = []

    rr_values = []

    sbp_values = []

    dbp_values = []

    ppg_window_count = 0

    processed_paths = []

    for _, row in manifest_group.iterrows():

        processed_path = Path(
            str(
                row[
                    "output"
                ]
            )
        )

        if not processed_path.exists():

            raise FileNotFoundError(
                "Processed reference NPZ "
                "does not exist:\n"
                f"{processed_path}"
            )

        processed_paths.append(
            processed_path
        )

        z = np.load(
            processed_path,
            allow_pickle=True,
        )

        # ----------------------------------------
        # HR PRIMARY REFERENCE:
        # synchronized contact PPG
        # ----------------------------------------

        if "ppg" in z.files:

            ppg = np.asarray(
                z[
                    "ppg"
                ],
                dtype=np.float64,
            )

            hr = ppg_to_hr_bpm(
                ppg,
                fs=PPG_FS,
            )

            if np.isfinite(
                hr
            ):

                hr_values.append(
                    hr
                )

                ppg_window_count += 1

        # ----------------------------------------
        # Metadata pulse retained only as
        # secondary comparison / sanity check.
        # ----------------------------------------

        if "pulse" in z.files:

            value = scalar_float(
                z[
                    "pulse"
                ]
            )

            if np.isfinite(
                value
            ):

                pulse_values.append(
                    value
                )

        # ----------------------------------------
        # RR
        # ----------------------------------------

        if "respiratory" in z.files:

            value = scalar_float(
                z[
                    "respiratory"
                ]
            )

            if np.isfinite(
                value
            ):

                rr_values.append(
                    value
                )

        # ----------------------------------------
        # SBP
        # ----------------------------------------

        if "sbp" in z.files:

            value = scalar_float(
                z[
                    "sbp"
                ]
            )

            if np.isfinite(
                value
            ):

                sbp_values.append(
                    value
                )

        # ----------------------------------------
        # DBP
        # ----------------------------------------

        if "dbp" in z.files:

            value = scalar_float(
                z[
                    "dbp"
                ]
            )

            if np.isfinite(
                value
            ):

                dbp_values.append(
                    value
                )

    # --------------------------------------------
    # HR:
    # recording reference = mean PPG-derived
    # window HR
    # --------------------------------------------

    hr_true = (
        float(
            np.mean(
                hr_values
            )
        )
        if hr_values
        else np.nan
    )

    # --------------------------------------------
    # Pulse:
    # recording metadata scalar
    # --------------------------------------------

    pulse_true = (
        float(
            np.median(
                pulse_values
            )
        )
        if pulse_values
        else np.nan
    )

    # --------------------------------------------
    # RR/SBP/DBP:
    # repeated recording/session scalar
    # --------------------------------------------

    rr_true = (
        float(
            np.median(
                rr_values
            )
        )
        if rr_values
        else np.nan
    )

    sbp_true = (
        float(
            np.median(
                sbp_values
            )
        )
        if sbp_values
        else np.nan
    )

    dbp_true = (
        float(
            np.median(
                dbp_values
            )
        )
        if dbp_values
        else np.nan
    )

    return {
        "hr_true": hr_true,
        "pulse_reference": pulse_true,
        "rr_true": rr_true,
        "sbp_true": sbp_true,
        "dbp_true": dbp_true,
        "ppg_reference_windows": ppg_window_count,
        "processed_reference_paths": processed_paths,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    valid = (
        np.isfinite(
            y_true
        )
        & np.isfinite(
            y_pred
        )
    )

    y_true = y_true[
        valid
    ]

    y_pred = y_pred[
        valid
    ]

    n = len(
        y_true
    )

    if n == 0:

        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "r": np.nan,
            "R2": np.nan,
            "slope": np.nan,
            "bias": np.nan,
            "true_mean": np.nan,
            "pred_mean": np.nan,
            "true_sd": np.nan,
            "pred_sd": np.nan,
        }

    error = (
        y_pred
        - y_true
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    bias = float(
        np.mean(
            error
        )
    )

    if (
        n >= 2
        and np.std(
            y_true
        ) > 0
        and np.std(
            y_pred
        ) > 0
    ):

        r = float(
            np.corrcoef(
                y_true,
                y_pred,
            )[0, 1]
        )

        slope = float(
            np.polyfit(
                y_true,
                y_pred,
                1,
            )[0]
        )

    else:

        r = np.nan

        slope = np.nan

    sst = float(
        np.sum(
            (
                y_true
                - np.mean(
                    y_true
                )
            ) ** 2
        )
    )

    if sst > 0:

        r2 = float(
            1.0
            - (
                np.sum(
                    error ** 2
                )
                / sst
            )
        )

    else:

        r2 = np.nan

    return {
        "N": int(
            n
        ),
        "MAE": mae,
        "RMSE": rmse,
        "r": r,
        "R2": r2,
        "slope": slope,
        "bias": bias,
        "true_mean": float(
            np.mean(
                y_true
            )
        ),
        "pred_mean": float(
            np.mean(
                y_pred
            )
        ),
        "true_sd": float(
            np.std(
                y_true,
                ddof=0,
            )
        ),
        "pred_sd": float(
            np.std(
                y_pred,
                ddof=0,
            )
        ),
    }


def calculate_all_endpoint_metrics(
    df: pd.DataFrame,
):

    mapping = {
        "HR": (
            "hr_true",
            "hr_pred",
        ),

        "RR": (
            "rr_true",
            "rr_pred",
        ),

        "SBP": (
            "sbp_true",
            "sbp_pred",
        ),

        "DBP": (
            "dbp_true",
            "dbp_pred",
        ),
    }

    rows = []

    for endpoint, (
        true_column,
        pred_column,
    ) in mapping.items():

        result = calculate_metrics(
            df[
                true_column
            ],
            df[
                pred_column
            ],
        )

        rows.append(
            {
                "endpoint": endpoint,
                **result,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUBJECT-CLUSTER BOOTSTRAP
# ============================================================

def subject_cluster_bootstrap(
    df: pd.DataFrame,
):

    if "patient_id" not in df.columns:

        return pd.DataFrame()

    subjects = (
        df[
            "patient_id"
        ]
        .astype(str)
        .unique()
    )

    if len(
        subjects
    ) < 2:

        return pd.DataFrame()

    subject_rows = {
        subject:
            df.index[
                df[
                    "patient_id"
                ].astype(str)
                == subject
            ].to_numpy()

        for subject in subjects
    }

    rng = (
        np.random.default_rng(
            BOOT_SEED
        )
    )

    mapping = {
        "HR": (
            "hr_true",
            "hr_pred",
        ),

        "RR": (
            "rr_true",
            "rr_pred",
        ),

        "SBP": (
            "sbp_true",
            "sbp_pred",
        ),

        "DBP": (
            "dbp_true",
            "dbp_pred",
        ),
    }

    metric_names = [
        "MAE",
        "RMSE",
        "r",
        "R2",
        "slope",
        "bias",
    ]

    storage = {
        endpoint: {
            metric: []
            for metric
            in metric_names
        }

        for endpoint
        in mapping
    }

    for _ in range(
        N_BOOT
    ):

        sampled_subjects = (
            rng.choice(
                subjects,
                size=len(
                    subjects
                ),
                replace=True,
            )
        )

        indices = np.concatenate(
            [
                subject_rows[
                    subject
                ]
                for subject
                in sampled_subjects
            ]
        )

        boot_df = df.loc[
            indices
        ]

        for endpoint, (
            true_column,
            pred_column,
        ) in mapping.items():

            result = calculate_metrics(
                boot_df[
                    true_column
                ],
                boot_df[
                    pred_column
                ],
            )

            for metric in metric_names:

                value = result[
                    metric
                ]

                if np.isfinite(
                    value
                ):

                    storage[
                        endpoint
                    ][
                        metric
                    ].append(
                        value
                    )

    rows = []

    for endpoint in mapping:

        for metric in metric_names:

            values = np.asarray(
                storage[
                    endpoint
                ][
                    metric
                ],
                dtype=float,
            )

            if len(
                values
            ) > 0:

                low, high = (
                    np.percentile(
                        values,
                        [
                            2.5,
                            97.5,
                        ],
                    )
                )

            else:

                low = np.nan

                high = np.nan

            rows.append(
                {
                    "endpoint":
                        endpoint,

                    "metric":
                        metric,

                    "lower_95":
                        float(
                            low
                        ),

                    "upper_95":
                        float(
                            high
                        ),

                    "n_boot_valid":
                        int(
                            len(
                                values
                            )
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CAMERA / STATE METRICS
# ============================================================

def grouped_metrics(
    df: pd.DataFrame,
    group_column: str,
):

    if group_column not in df.columns:

        return pd.DataFrame()

    result_frames = []

    for group_value, group_df in df.groupby(
        group_column,
        dropna=False,
    ):

        current = (
            calculate_all_endpoint_metrics(
                group_df
            )
        )

        current.insert(
            0,
            group_column,
            group_value,
        )

        result_frames.append(
            current
        )

    if not result_frames:

        return pd.DataFrame()

    return pd.concat(
        result_frames,
        ignore_index=True,
    )


# ============================================================
# BP SESSION AGGREGATION
# ============================================================

def calculate_bp_session_metrics(
    df: pd.DataFrame,
):

    if (
        "patient_id"
        not in df.columns
        or
        "state"
        not in df.columns
    ):

        return pd.DataFrame()

    session_df = (
        df.groupby(
            [
                "patient_id",
                "state",
            ],
            dropna=False,
        )
        .agg(
            sbp_true=(
                "sbp_true",
                "mean",
            ),

            sbp_pred=(
                "sbp_pred",
                "mean",
            ),

            dbp_true=(
                "dbp_true",
                "mean",
            ),

            dbp_pred=(
                "dbp_pred",
                "mean",
            ),

            camera_count=(
                "recording_id",
                "count",
            ),
        )
        .reset_index()
    )

    rows = []

    sbp_metrics = (
        calculate_metrics(
            session_df[
                "sbp_true"
            ],
            session_df[
                "sbp_pred"
            ],
        )
    )

    rows.append(
        {
            "endpoint":
                "SBP",

            **sbp_metrics,
        }
    )

    dbp_metrics = (
        calculate_metrics(
            session_df[
                "dbp_true"
            ],
            session_df[
                "dbp_pred"
            ],
        )
    )

    rows.append(
        {
            "endpoint":
                "DBP",

            **dbp_metrics,
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    check_required_files()

    manifest = pd.read_csv(
        MANIFEST_CSV
    )


    # --------------------------------------------------------
    # REQUIRED MANIFEST COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "patient_id",
        "recording_id",
        "window_id",
        "split",
        "camera",
        "view",
        "state",
        "output",
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in manifest.columns
    ]

    if missing_columns:

        raise KeyError(
            "Manifest missing columns:\n"
            + "\n".join(
                missing_columns
            )
        )


    # --------------------------------------------------------
    # CRITICAL:
    # revised journal split is authoritative.
    #
    # Do NOT use the physical train/test folder
    # from the old conference preprocessing path.
    # --------------------------------------------------------

    test_df = manifest[
        manifest[
            "split"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "test"
        )
    ].copy()


    test_subjects = (
        test_df[
            "patient_id"
        ].nunique()
    )

    test_recordings = (
        test_df[
            "recording_id"
        ].nunique()
    )


    print(
        "\n"
        + "=" * 90
    )

    print(
        "REVISED RAW-RGB TRACE APPLICATION VALIDATION"
    )

    print(
        "=" * 90
    )

    print(
        f"Manifest test windows : "
        f"{len(test_df)}"
    )

    print(
        f"Test subjects         : "
        f"{test_subjects}"
    )

    print(
        f"Test recordings       : "
        f"{test_recordings}"
    )

    print(
        f"Raw NPZ folder        : "
        f"{RAW_NPZ_DIR}"
    )

    print(
        f"Output folder         : "
        f"{OUT_DIR}"
    )


    # --------------------------------------------------------
    # INDEX RAW REVISED NPZS
    # --------------------------------------------------------

    raw_index = (
        build_raw_npz_index()
    )


    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    completed = set()

    if PRED_CSV.exists():

        previous = pd.read_csv(
            PRED_CSV
        )

        if (
            "recording_id"
            in previous.columns
        ):

            completed = set(
                previous[
                    "recording_id"
                ].astype(str)
            )

            print(
                "\nResume mode:"
            )

            print(
                f"{len(completed)} "
                f"recordings already completed."
            )


    # --------------------------------------------------------
    # GROUP WINDOWS BY RECORDING
    # --------------------------------------------------------

    recording_groups = list(
        test_df.groupby(
            "recording_id",
            sort=True,
        )
    )

    pending = [
        (
            recording_id,
            group.copy(),
        )

        for recording_id, group
        in recording_groups

        if str(
            recording_id
        )
        not in completed
    ]


    if LIMIT is not None:

        pending = pending[
            :int(
                LIMIT
            )
        ]


    print(
        f"\nRecordings to process "
        f"this run: {len(pending)}"
    )


    # ========================================================
    # RECORDING LOOP
    # ========================================================

    for i, (
        recording_id,
        group,
    ) in enumerate(
        pending,
        start=1,
    ):

        group = group.sort_values(
            "window_id"
        )

        first_row = (
            group.iloc[
                0
            ]
        )

        patient_id = (
            first_row[
                "patient_id"
            ]
        )

        camera = (
            first_row[
                "camera"
            ]
        )

        view = (
            first_row[
                "view"
            ]
        )

        state = (
            first_row[
                "state"
            ]
        )


        print(
            "\n"
            + "-" * 90
        )

        print(
            f"[{i}/{len(pending)}] "
            f"{recording_id}"
        )

        print(
            f"subject={patient_id} | "
            f"camera={camera} | "
            f"view={view} | "
            f"state={state}"
        )


        start_time = (
            time.time()
        )


        try:

            # ------------------------------------------------
            # EXACT RGB WINDOWS FROM REVISED RAW NPZ
            # ------------------------------------------------

            (
                windows,
                raw_npz_path,
                raw_meta,
            ) = load_raw_recording_npz(
                recording_id,
                raw_index,
            )


            manifest_window_count = (
                len(
                    group
                )
            )

            raw_window_count = (
                len(
                    windows
                )
            )


            print(
                f"windows_raw shape : "
                f"{windows.shape}"
            )

            print(
                f"manifest windows  : "
                f"{manifest_window_count}"
            )


            if (
                raw_window_count
                !=
                manifest_window_count
            ):

                raise RuntimeError(
                    "Window-count mismatch.\n"
                    f"Recording: "
                    f"{recording_id}\n"
                    f"windows_raw: "
                    f"{raw_window_count}\n"
                    f"manifest: "
                    f"{manifest_window_count}"
                )


            if (
                "window_starts"
                in raw_meta
            ):

                print(
                    "window starts     : "
                    f"{raw_meta['window_starts'].tolist()}"
                )


            if (
                "resample_fs"
                in raw_meta
            ):

                print(
                    "resample fs       : "
                    f"{raw_meta['resample_fs']}"
                )


            if (
                "detection_rate"
                in raw_meta
            ):

                print(
                    "detection rate    : "
                    f"{raw_meta['detection_rate']:.3f}"
                )


            # ------------------------------------------------
            # REFERENCE VALUES
            # ------------------------------------------------

            reference = (
                load_reference_data(
                    group
                )
            )


            print(
                "reference          : "
                f"HR={reference['hr_true']:.2f}, "
                f"pulse={reference['pulse_reference']:.2f}, "
                f"RR={reference['rr_true']:.2f}, "
                f"SBP={reference['sbp_true']:.1f}, "
                f"DBP={reference['dbp_true']:.1f}"
            )


            # ------------------------------------------------
            # APP INFERENCE
            #
            # run_all_models_on_windows currently reads
            # [N,900,3] from an NPY path.
            # ------------------------------------------------

            with tempfile.TemporaryDirectory() as temp_dir:

                x_path = (
                    Path(
                        temp_dir
                    )
                    / "x_windows.npy"
                )

                np.save(
                    x_path,
                    windows.astype(
                        np.float32
                    ),
                )


                _, model_summary = (
                    run_all_models_on_windows(
                        x_windows_path=x_path,

                        hr_model_path=HR_MODEL,

                        rr_model_path=RR_MODEL,

                        bp_hybrid_model_path=BP_HYBRID_MODEL,

                        bp_feature_bundle_path=BP_FEATURE_BUNDLE,

                        batch_size=BATCH_SIZE,
                    )
                )


            elapsed = (
                time.time()
                - start_time
            )


            # ------------------------------------------------
            # OUTPUT ROW
            # ------------------------------------------------

            result = {

                "recording_id":
                    str(
                        recording_id
                    ),

                "patient_id":
                    patient_id,

                "camera":
                    camera,

                "view":
                    view,

                "state":
                    state,

                "raw_npz_path":
                    str(
                        raw_npz_path
                    ),

                "num_windows":
                    int(
                        raw_window_count
                    ),

                "manifest_windows":
                    int(
                        manifest_window_count
                    ),

                "ppg_reference_windows":
                    int(
                        reference[
                            "ppg_reference_windows"
                        ]
                    ),


                # ----------------------------
                # TRUE / PREDICTED
                # ----------------------------

                "hr_true":
                    reference[
                        "hr_true"
                    ],

                "hr_pred":
                    float(
                        model_summary[
                            "hr_mean"
                        ]
                    ),

                "pulse_reference":
                    reference[
                        "pulse_reference"
                    ],

                "rr_true":
                    reference[
                        "rr_true"
                    ],

                "rr_pred":
                    float(
                        model_summary[
                            "rr_mean"
                        ]
                    ),

                "sbp_true":
                    reference[
                        "sbp_true"
                    ],

                "sbp_pred":
                    float(
                        model_summary[
                            "sbp_mean"
                        ]
                    ),

                "dbp_true":
                    reference[
                        "dbp_true"
                    ],

                "dbp_pred":
                    float(
                        model_summary[
                            "dbp_mean"
                        ]
                    ),


                # ----------------------------
                # BP INTERNAL DIAGNOSTIC
                # ----------------------------

                "hybrid_sbp_internal":
                    float(
                        model_summary.get(
                            "hybrid_sbp_internal",
                            np.nan,
                        )
                    ),


                # ----------------------------
                # RAW PROCESSING METADATA
                # ----------------------------

                "original_fps":
                    raw_meta.get(
                        "original_fps",
                        np.nan,
                    ),

                "resample_fs":
                    raw_meta.get(
                        "resample_fs",
                        np.nan,
                    ),

                "window_sec":
                    raw_meta.get(
                        "window_sec",
                        np.nan,
                    ),

                "window_stride_sec":
                    raw_meta.get(
                        "window_stride_sec",
                        np.nan,
                    ),

                "detection_rate":
                    raw_meta.get(
                        "detection_rate",
                        np.nan,
                    ),

                "roi_type":
                    raw_meta.get(
                        "roi_type",
                        "",
                    ),

                "elapsed_seconds":
                    float(
                        elapsed
                    ),
            }


            append_row(
                result,
                PRED_CSV,
            )


            print(
                "OK | "
                f"windows={result['num_windows']} | "
                f"HR {result['hr_true']:.2f}"
                f" -> {result['hr_pred']:.2f} | "
                f"RR {result['rr_true']:.2f}"
                f" -> {result['rr_pred']:.2f} | "
                f"SBP {result['sbp_true']:.1f}"
                f" -> {result['sbp_pred']:.1f} | "
                f"DBP {result['dbp_true']:.1f}"
                f" -> {result['dbp_pred']:.1f} | "
                f"{elapsed:.2f}s"
            )


        except Exception as exc:

            elapsed = (
                time.time()
                - start_time
            )

            print(
                "FAILED:",
                type(
                    exc
                ).__name__,
                str(
                    exc
                ),
            )


            append_row(
                {
                    "recording_id":
                        recording_id,

                    "patient_id":
                        patient_id,

                    "camera":
                        camera,

                    "view":
                        view,

                    "state":
                        state,

                    "error_type":
                        type(
                            exc
                        ).__name__,

                    "error":
                        str(
                            exc
                        ),

                    "elapsed_seconds":
                        elapsed,

                    "traceback":
                        traceback.format_exc(),
                },

                FAIL_CSV,
            )


    # ========================================================
    # IF NO SUCCESSFUL RESULTS
    # ========================================================

    if not PRED_CSV.exists():

        print(
            "\nNo successful predictions."
        )

        return


    # ========================================================
    # LOAD CURRENT RESULTS
    # ========================================================

    predictions = pd.read_csv(
        PRED_CSV
    )


    predictions = (
        predictions
        .drop_duplicates(
            "recording_id",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )


    predictions.to_csv(
        PRED_CSV,
        index=False,
    )


    # ========================================================
    # MAIN RECORDING-LEVEL METRICS
    # ========================================================

    main_metrics = (
        calculate_all_endpoint_metrics(
            predictions
        )
    )


    main_metrics.to_csv(
        METRICS_CSV,
        index=False,
    )


    # ========================================================
    # CAMERA METRICS
    # ========================================================

    camera_metrics = (
        grouped_metrics(
            predictions,
            "camera",
        )
    )


    if len(
        camera_metrics
    ):

        camera_metrics.to_csv(
            CAMERA_CSV,
            index=False,
        )


    # ========================================================
    # STATE METRICS
    # ========================================================

    state_metrics = (
        grouped_metrics(
            predictions,
            "state",
        )
    )


    if len(
        state_metrics
    ):

        state_metrics.to_csv(
            STATE_CSV,
            index=False,
        )


    # ========================================================
    # BP SESSION-LEVEL DIAGNOSTIC
    # ========================================================

    bp_session = (
        calculate_bp_session_metrics(
            predictions
        )
    )


    if len(
        bp_session
    ):

        bp_session.to_csv(
            BP_SESSION_CSV,
            index=False,
        )


    # ========================================================
    # SUBJECT-CLUSTER BOOTSTRAP
    #
    # Do not interpret CI from LIMIT=5 scientifically.
    # Final CI is meaningful after complete 90-subject run.
    # ========================================================

    bootstrap = (
        subject_cluster_bootstrap(
            predictions
        )
    )


    if len(
        bootstrap
    ):

        bootstrap.to_csv(
            BOOT_CSV,
            index=False,
        )


    # ========================================================
    # SUMMARY JSON
    # ========================================================

    failure_count = 0

    if FAIL_CSV.exists():

        failure_df = pd.read_csv(
            FAIL_CSV
        )

        failure_count = (
            len(
                failure_df
            )
        )


    summary = {

        "protocol":
            "Held-out revised raw RGB-trace "
            "application validation",

        "journal_split_source":
            str(
                MANIFEST_CSV
            ),

        "raw_rgb_source":
            str(
                RAW_NPZ_DIR
            ),

        "test_subjects":
            int(
                test_subjects
            ),

        "expected_test_recordings":
            int(
                test_recordings
            ),

        "successful_recordings":
            int(
                predictions[
                    "recording_id"
                ].nunique()
            ),

        "failure_rows":
            int(
                failure_count
            ),

        "rgb_input":
            "windows_raw",

        "rgb_shape":
            "[N, 900, 3]",

        "resample_fs":
            30.0,

        "window_seconds":
            30.0,

        "window_stride_seconds":
            30.0,

        "hr_reference":
            (
                "Synchronized contact PPG; "
                "linear detrend; 0.7-3.0 Hz "
                "Butterworth bandpass; Welch "
                "dominant frequency; Hz x 60"
            ),

        "rr_reference":
            "respiratory field in processed NPZ",

        "sbp_reference":
            "sbp field in processed NPZ",

        "dbp_reference":
            "dbp field in processed NPZ",

        "models": {

            "hr":
                str(
                    HR_MODEL
                ),

            "rr":
                str(
                    RR_MODEL
                ),

            "bp_hybrid":
                str(
                    BP_HYBRID_MODEL
                ),

            "bp_feature_bundle":
                str(
                    BP_FEATURE_BUNDLE
                ),
        },
    }


    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CURRENT RECORDING-LEVEL METRICS"
    )

    print(
        "=" * 90
    )

    print(
        main_metrics.to_string(
            index=False
        )
    )


    print(
        "\nSuccessful recordings:",
        predictions[
            "recording_id"
        ].nunique(),
    )


    print(
        "Failures:",
        failure_count,
    )


    print(
        "\nResults saved in:"
    )

    print(
        OUT_DIR
    )


    if LIMIT is not None:

        print(
            "\n"
            + "=" * 90
        )

        print(
            "SMOKE TEST MODE"
        )

        print(
            "=" * 90
        )

        print(
            f"Only {LIMIT} recordings "
            f"were processed."
        )

        print(
            "\nInspect the predictions first."
        )

        print(
            "If everything is correct, "
            "change:"
        )

        print(
            "\nLIMIT = None"
        )

        print(
            "\nand rerun for all "
            "537 held-out recordings."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()