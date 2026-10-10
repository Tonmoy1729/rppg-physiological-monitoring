import os
import html
import shutil
import tempfile
import traceback
from pathlib import Path

os.environ["GRADIO_SSR_MODE"] = "False"

import gradio as gr
import spaces

from roi_video_pp import prepare_video_to_windows
from infer_from_windows import run_all_models_on_windows


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

HR_MODEL = MODELS / "hr" / "best_hr_model.pt"
RR_MODEL = MODELS / "rr" / "best_rr_model.pt"
BP_MODEL = MODELS / "bp" / "best_bp_hybrid.pt"
BP_BUNDLE = MODELS / "bp" / "bp_feature_bundle.joblib"


# ============================================================
# PIPELINE SETTINGS
# ============================================================

TARGET_FS = 30.0
WINDOW_SEC = 30.0
STRIDE_SEC = 30.0
MAX_SECONDS = 90.0

FRAME_STRIDE = 1
DETECT_EVERY = 5
MAX_FRAME_WIDTH = 384
MIN_FACE_RATE = 0.60

BATCH_SIZE = 64


# ============================================================
# ASSET CHECK
# ============================================================

def check_assets():

    required = [
        HR_MODEL,
        RR_MODEL,
        BP_MODEL,
        BP_BUNDLE,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model assets:\n"
            + "\n".join(missing)
        )


# ============================================================
# ZEROGPU INFERENCE
# ============================================================

@spaces.GPU(duration=30)
def run_models_on_zerogpu(x_windows_path: str):

    _, summary = run_all_models_on_windows(
        x_windows_path=Path(x_windows_path),
        hr_model_path=HR_MODEL,
        rr_model_path=RR_MODEL,
        bp_hybrid_model_path=BP_MODEL,
        bp_feature_bundle_path=BP_BUNDLE,
        batch_size=BATCH_SIZE,
    )

    return summary


# ============================================================
# RESULT HTML
# ============================================================

def result_cards(
    hr=None,
    rr=None,
    sbp=None,
    dbp=None,
    face_rate=None,
    windows=None,
):

    def value(x, digits=1):
        return "--" if x is None else f"{x:.{digits}f}"

    face_text = (
        "--"
        if face_rate is None
        else f"{face_rate * 100:.1f}%"
    )

    window_text = (
        "--"
        if windows is None
        else str(windows)
    )

    return f"""
    <div class="vital-grid">

        <div class="vital-card hr-card">
            <div class="card-head">
                <span class="icon">♥</span>
                <span class="badge primary">PRIMARY</span>
            </div>

            <div class="title">Heart Rate</div>

            <div class="value hr">
                {value(hr)}
                <span>bpm</span>
            </div>

            <div class="note">
                Primary physiological endpoint
            </div>
        </div>


        <div class="vital-card rr-card">
            <div class="card-head">
                <span class="icon">◉</span>
                <span class="badge secondary">SECONDARY</span>
            </div>

            <div class="title">
                Respiratory Rate
            </div>

            <div class="value rr">
                {value(rr)}
                <span>breaths/min</span>
            </div>

            <div class="note">
                Secondary research endpoint
            </div>
        </div>


        <div class="vital-card sbp-card">
            <div class="card-head">
                <span class="icon">↑</span>
                <span class="badge exploratory">
                    EXPLORATORY
                </span>
            </div>

            <div class="title">
                Systolic Blood Pressure
            </div>

            <div class="value sbp">
                {value(sbp)}
                <span>mmHg</span>
            </div>

            <div class="note">
                Robust feature-based estimate
            </div>
        </div>


        <div class="vital-card dbp-card">
            <div class="card-head">
                <span class="icon">↓</span>
                <span class="badge exploratory">
                    EXPLORATORY
                </span>
            </div>

            <div class="title">
                Diastolic Blood Pressure
            </div>

            <div class="value dbp">
                {value(dbp)}
                <span>mmHg</span>
            </div>

            <div class="note">
                Hybrid BP-v2 estimate
            </div>
        </div>

    </div>


    <div class="quality-bar">

        <div>
            <small>FACE TRACKING</small>
            <strong>{face_text}</strong>
        </div>

        <div>
            <small>VALID WINDOWS</small>
            <strong>{window_text}</strong>
        </div>

        <div>
            <small>WINDOW LENGTH</small>
            <strong>30 s</strong>
        </div>

    </div>
    """


def status_html(message, kind="info"):

    safe = html.escape(str(message))

    return f"""
    <div class="status {kind}">
        {safe}
    </div>
    """


# ============================================================
# VIDEO PREDICTION
# ============================================================

def predict_video(video_input):

    try:

        check_assets()

        if not video_input:
            raise ValueError(
                "Please upload or record a facial video."
            )

        video_path = Path(str(video_input))

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        with tempfile.TemporaryDirectory() as temp:

            temp = Path(temp)

            local_video = (
                temp
                / f"input{video_path.suffix or '.mp4'}"
            )

            shutil.copy(
                video_path,
                local_video,
            )

            output_dir = (
                temp
                / "processed"
            )


            # --------------------------------------------
            # VIDEO -> RGB WINDOWS
            # --------------------------------------------

            _, metadata = prepare_video_to_windows(
                video_path=local_video,
                out_dir=output_dir,
                target_fs=TARGET_FS,
                window_sec=WINDOW_SEC,
                stride_sec=STRIDE_SEC,
                max_seconds=MAX_SECONDS,
                max_frame_width=MAX_FRAME_WIDTH,
                frame_stride=FRAME_STRIDE,
                detect_every=DETECT_EVERY,
                min_face_detection_rate=MIN_FACE_RATE,
            )


            # --------------------------------------------
            # ZEROGPU MODEL INFERENCE
            # --------------------------------------------

            summary = run_models_on_zerogpu(
                str(
                    output_dir
                    / "x_windows.npy"
                )
            )


            hr = float(
                summary["hr_mean"]
            )

            rr = float(
                summary["rr_mean"]
            )

            sbp = float(
                summary["sbp_mean"]
            )

            dbp = float(
                summary["dbp_mean"]
            )

            windows = int(
                summary["num_windows"]
            )

            face_rate = float(
                metadata.get(
                    "face_detection_rate",
                    0.0,
                )
            )


            cards = result_cards(
                hr=hr,
                rr=rr,
                sbp=sbp,
                dbp=dbp,
                face_rate=face_rate,
                windows=windows,
            )


            msg = (
                f"Analysis complete. "
                f"{windows} valid 30-second "
                f"window(s) processed."
            )

            if face_rate < 0.80:
                msg += (
                    " Face tracking was below 80%; "
                    "interpret results cautiously."
                )

            return (
                cards,
                status_html(
                    msg,
                    "success",
                ),
            )


    except Exception as exc:

        traceback.print_exc()

        return (
            result_cards(),

            status_html(
                f"{type(exc).__name__}: {exc}",
                "error",
            ),
        )


def reset_ui():

    return (
        None,
        result_cards(),
        status_html(
            "Upload or record a facial video, "
            "then click Analyze Video."
        ),
    )


# ============================================================
# WEBCAM TIMER
# ============================================================

TIMER_JS = """
() => {

    if (window.rppgTimerReady) return;
    window.rppgTimerReady = true;

    let interval = null;
    let remaining = 30;

    function timer() {
        return document.getElementById(
            "record-countdown"
        );
    }

    function update() {

        const el = timer();

        if (!el) return;

        el.textContent =
            "00:" +
            String(
                Math.max(0, remaining)
            ).padStart(2, "0");

        el.style.color =
            remaining <= 5
            ? "#ff617d"
            : "#ffffff";
    }

    function text(btn) {
        return (
            btn.innerText ||
            btn.getAttribute("aria-label") ||
            ""
        ).toLowerCase();
    }

    function start() {

        clearInterval(interval);

        remaining = 30;
        update();

        interval = setInterval(
            () => {

                remaining--;
                update();

                if (remaining <= 0) {

                    clearInterval(interval);

                    const stop =
                        [...document.querySelectorAll(
                            "button"
                        )].find(
                            b =>
                                text(b).includes(
                                    "stop"
                                )
                                &&
                                text(b).includes(
                                    "record"
                                )
                        );

                    if (stop) stop.click();
                }

            },
            1000
        );
    }

    function reset() {

        clearInterval(interval);

        remaining = 30;

        update();
    }

    function attach() {

        document
        .querySelectorAll("button")
        .forEach(btn => {

            if (btn.dataset.rppgTimer)
                return;

            const t = text(btn);

            if (
                t.includes("record")
                &&
                !t.includes("stop")
            ) {
                btn.dataset.rppgTimer = "1";
                btn.addEventListener(
                    "click",
                    () => setTimeout(
                        start,
                        300
                    )
                );
            }

            if (
                t.includes("clear")
                ||
                t.includes("remove")
            ) {
                btn.dataset.rppgTimer = "1";
                btn.addEventListener(
                    "click",
                    reset
                );
            }
        });
    }

    new MutationObserver(
        attach
    ).observe(
        document.body,
        {
            childList: true,
            subtree: true
        }
    );

    attach();
    update();
}
"""


# ============================================================
# CSS
# ============================================================

CSS = """
html, body {
    background: #070c15 !important;
}

.gradio-container {
    max-width: 1220px !important;
    margin: auto !important;
    padding: 18px !important;
    background: transparent !important;
}

#header {
    text-align: center;
    margin-bottom: 18px;
}

#header h1 {
    margin: 0;
    color: #f8fafc;
    font-size: 32px;
    font-weight: 800;
}

#header p {
    margin-top: 7px;
    color: #94a3b8;
    font-size: 14px;
}

#main-row {
    gap: 18px !important;
    align-items: stretch !important;
}

.panel {
    background: #0b1220;
    border: 1px solid #26344a;
    border-radius: 17px;
    padding: 15px;
}

.section-title {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 750;
    margin-bottom: 4px;
}

.section-note {
    color: #8491a4;
    font-size: 12px;
    margin-bottom: 11px;
}

#video-box {
    border-radius: 13px !important;
    overflow: hidden !important;
    border: 1px solid #334155 !important;
    background: #030711 !important;
}

#video-box video {
    height: 340px !important;
    object-fit: contain !important;
    background: #030711 !important;
}

#timer {
    margin-top: 10px;
    padding: 10px 13px;
    display: flex;
    justify-content: space-between;
    background: #101827;
    border: 1px solid #26344a;
    border-radius: 10px;
    color: #94a3b8;
    font-size: 12px;
}

#record-countdown {
    color: white;
    font-size: 19px;
    font-weight: 800;
}

#analyze {
    height: 48px !important;
    margin-top: 10px !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 750 !important;
}

#reset {
    margin-top: 7px !important;
    border-radius: 9px !important;
}

.vital-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 11px;
}

.vital-card {
    min-height: 165px;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #2b394d;
    background:
        linear-gradient(
            145deg,
            #111a2a,
            #090f1b
        );
}

.hr-card { border-top: 3px solid #ff607d; }
.rr-card { border-top: 3px solid #5b91ff; }
.sbp-card { border-top: 3px solid #f3bb4b; }
.dbp-card { border-top: 3px solid #43cbaa; }

.card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.icon {
    color: #e2e8f0;
    font-size: 18px;
}

.badge {
    font-size: 8px;
    font-weight: 800;
    padding: 4px 7px;
    border-radius: 20px;
    letter-spacing: .6px;
}

.primary {
    color: #ff94aa;
    background: rgba(255,96,125,.09);
}

.secondary {
    color: #91b4ff;
    background: rgba(91,145,255,.09);
}

.exploratory {
    color: #e8bd66;
    background: rgba(243,187,75,.08);
}

.title {
    margin-top: 12px;
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 650;
}

.value {
    margin-top: 7px;
    font-size: 42px;
    font-weight: 850;
    line-height: 1;
}

.value span {
    margin-left: 5px;
    font-size: 12px;
    font-weight: 700;
}

.hr { color: #ff6985; }
.rr { color: #689cff; }
.sbp { color: #f3bd52; }
.dbp { color: #47d0b0; }

.note {
    margin-top: 11px;
    color: #738095;
    font-size: 10px;
}

.quality-bar {
    margin-top: 11px;
    padding: 10px;
    display: flex;
    justify-content: space-around;
    background: #101827;
    border: 1px solid #26344a;
    border-radius: 10px;
}

.quality-bar div {
    text-align: center;
}

.quality-bar small {
    display: block;
    color: #778499;
    font-size: 8px;
    letter-spacing: .5px;
}

.quality-bar strong {
    display: block;
    margin-top: 3px;
    color: #e2e8f0;
    font-size: 13px;
}

.status {
    margin-top: 11px;
    padding: 10px 12px;
    border-radius: 9px;
    font-size: 11px;
    line-height: 1.45;
}

.info {
    color: #aab5c5;
    background: #101827;
    border: 1px solid #26344a;
}

.success {
    color: #bfe3d3;
    background: rgba(40,120,88,.14);
    border: 1px solid rgba(67,203,170,.3);
}

.error {
    color: #fecaca;
    background: rgba(150,30,45,.16);
    border: 1px solid rgba(239,68,68,.4);
}

#warning {
    margin-top: 13px;
    padding: 10px 13px;
    text-align: center;
    color: #7f8a9d;
    background: #090f1b;
    border: 1px solid #202c3e;
    border-radius: 10px;
    font-size: 10px;
    line-height: 1.5;
}

@media(max-width: 900px) {
    #main-row {
        flex-direction: column !important;
    }
}

@media(max-width: 600px) {
    .vital-grid {
        grid-template-columns: 1fr;
    }

    #video-box video {
        height: 240px !important;
    }

    #header h1 {
        font-size: 25px;
    }
}
"""


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Contactless Physiological Monitoring",
    css=CSS,
) as demo:

    gr.HTML(
        """
        <div id="header">

            <h1>
                Contactless Physiological Monitoring
            </h1>

            <p>
                Facial-video physiological estimation
                using a revised rPPG research pipeline
            </p>

        </div>
        """
    )


    with gr.Row(
        elem_id="main-row",
        equal_height=True,
    ):

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        with gr.Column(
            scale=6,
            min_width=420,
            elem_classes="panel",
        ):

            gr.HTML(
                """
                <div class="section-title">
                    Facial Video
                </div>

                <div class="section-note">
                    Upload a 30–90 second frontal video
                    or record 30 seconds using your webcam.
                </div>
                """
            )

            video = gr.Video(
                label="Upload or Record Video",
                sources=[
                    "upload",
                    "webcam",
                ],
                format="mp4",
                height=340,
                elem_id="video-box",
            )

            gr.HTML(
                """
                <div id="timer">

                    <span>
                        Webcam recording guide
                    </span>

                    <span id="record-countdown">
                        00:30
                    </span>

                </div>
                """
            )

            analyze = gr.Button(
                "Analyze Video",
                variant="primary",
                elem_id="analyze",
            )

            reset = gr.Button(
                "Reset",
                elem_id="reset",
            )


        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        with gr.Column(
            scale=6,
            min_width=450,
            elem_classes="panel",
        ):

            gr.HTML(
                """
                <div class="section-title">
                    Physiological Estimates
                </div>

                <div class="section-note">
                    Results appear after video analysis.
                </div>
                """
            )

            results = gr.HTML(
                value=result_cards()
            )

            status = gr.HTML(
                value=status_html(
                    "Upload or record a facial video, "
                    "then click Analyze Video."
                )
            )


    gr.HTML(
        """
        <div id="warning">

            <b>Research prototype only.</b>

            Heart rate is the primary endpoint.
            Respiratory rate is secondary.
            Systolic and diastolic blood-pressure
            estimates are exploratory.

            This software is not a medical device
            and must not be used for diagnosis,
            treatment decisions, emergency decisions,
            or clinical monitoring.

        </div>
        """
    )


    analyze.click(
        fn=predict_video,
        inputs=video,
        outputs=[
            results,
            status,
        ],
    )


    reset.click(
        fn=reset_ui,
        inputs=None,
        outputs=[
            video,
            results,
            status,
        ],
    )


    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js=TIMER_JS,
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=1,
        max_size=10,
    ).launch(
        ssr_mode=False
    )