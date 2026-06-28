# -*- coding: utf-8 -*-
"""
3-Cam YOLO (ONNX) Detector — Raspberry Pi 5 (Linux/V4L2)

This version runs an exported ONNX model (best.onnx) via Ultralytics + ONNX Runtime
on a Raspberry Pi 5. It keeps your original sequential "cold open → warm → snap → close"
per camera to avoid multi-camera contention.

New in this build:
  • A dedicated Camera Calibration page at /calibrate
    - Live MJPEG stream of the selected camera (one-at-a-time)
    - Hardware zoom control using `v4l2-ctl -c zoom_absolute=...` and readback
    - Switching cameras stops any previous stream
  • /send_mes route for Tulip MES integration

Key changes vs Windows/PT:
  • Uses CAP_V4L2 and Linux device paths (e.g., /dev/video0 or /dev/v4l/by-id/…)
  • Accepts string camera device paths, not just numeric indices
  • Loads ONNX weights (best.onnx) via Ultralytics; requires `onnxruntime` installed
  • Small safety tweaks and Linux-friendly probing

Run:
  python app_onnx_pi.py
Then visit:
  http://127.0.0.1:5000/           (detector)
  http://127.0.0.1:5000/calibrate  (camera calibration page)
"""

import os
import re
import cv2
import uuid
import yaml
import atexit
import base64
import time
import shutil
import threading
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Optional, Union

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, Response
from ultralytics import YOLO

# Load .env file before reading any env vars
load_dotenv()

# =========================
# CONFIG — EDIT THESE
# =========================
# Place your files in a convenient directory on the Pi, e.g., /home/thelostbong/testapp
WEIGHTS_PATH   = os.environ.get("WEIGHTS_PATH", "/home/thelostbong/testapp/best.onnx")
DATA_YAML_PATH = os.environ.get("DATA_YAML_PATH", "/home/thelostbong/testapp/data.yaml")
STATION_ID     = os.environ.get("STATION_ID", "Q-Check-01")

# Prefer stable by-id symlinks; fill these with your actual paths from `ls -l /dev/v4l/by-id/`
# If left empty, the app will fall back to numeric /dev/video indices [0,1,2].
WEBCAM_DEVICES: List[Union[str,int]] = [
    os.environ.get("CAM0", "/dev/video0"),
    os.environ.get("CAM1", "/dev/video2"),
    os.environ.get("CAM2", "/dev/video4"),
]
# TIP: You can also set CAM0/1/2 to by-id paths:
#   /dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_...-video-index0

IMG_SIZE       = int(os.environ.get("IMG_SIZE", 640))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.25))
HOST           = os.environ.get("HOST", "0.0.0.0")
PORT           = int(os.environ.get("PORT", 5000))

# Camera tuning (Logitech UVC cams are happy with MJPG at 720p/30 on Pi 5)
FRAME_WIDTH      = int(os.environ.get("FRAME_WIDTH", 1280))
FRAME_HEIGHT     = int(os.environ.get("FRAME_HEIGHT", 720))
FRAME_FPS        = int(os.environ.get("FRAME_FPS", 30))
FOURCC           = os.environ.get("FOURCC", "MJPG")  # try "MJPG" first; fallback to "YUYV" if needed
WARMUP_GRABS     = int(os.environ.get("WARMUP_GRABS", 8))
MAX_READ_RETRY   = int(os.environ.get("MAX_READ_RETRY", 30))
RETRY_SLEEP_S    = float(os.environ.get("RETRY_SLEEP_S", 0.015))
RELEASE_COOLDOWN = float(os.environ.get("RELEASE_COOLDOWN", 0.20))
BUFFERSIZE_TRY   = int(os.environ.get("BUFFERSIZE_TRY", 1))

# Make sure we never open two cams at once
capture_lock = threading.Lock()

# Modest thread/env tuning on ARM
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

# v4l2-ctl presence
V4L2_CTL = shutil.which("v4l2-ctl") or "/usr/bin/v4l2-ctl"

# =========================
# Load YOLO (ONNX) model
# =========================
if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(f"Model file not found: {WEIGHTS_PATH}")

# NOTE: Ultralytics will pick ONNX Runtime if available. Install: `pip install onnxruntime`
model = YOLO(WEIGHTS_PATH)

# =========================
# Optional custom class names from data.yaml
# =========================
CUSTOM_NAMES_DICT = None
if os.path.isfile(DATA_YAML_PATH):
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}
    names = data_cfg.get("names")
    if isinstance(names, dict):
        try:
            names_list = [names[k] for k in sorted(names, key=lambda x: int(x))]
        except Exception:
            names_list = list(names.values())
        CUSTOM_NAMES_DICT = {i: n for i, n in enumerate(names_list)}
    elif isinstance(names, list):
        CUSTOM_NAMES_DICT = {i: n for i, n in enumerate(names)}

# =========================
# Canonical feature names
# =========================
REQ_COMMON = ["closed cabin", "light bar", "front wheel cover", "steering wheel"]
FRONT_VARIANTS = ["front wheel roller", "front wheel tire"]
REAR_VARIANTS  = ["rear wheel roller",  "rear wheel tire"]
TOP_VARIANTS   = ["top light one", "top light two"]
SEAT_VARIANTS  = ["correct seat one", "correct seat two"]

ALIASES = {
    "frontwheelroller": "front wheel roller",
    "front wheel-roller": "front wheel roller",
    "frontroller": "front wheel roller",

    "frontwheeltire": "front wheel tire",
    "front wheel-tyre": "front wheel tire",
    "fronttyre": "front wheel tire",

    "rearwheelroller": "rear wheel roller",
    "rearroller": "rear wheel roller",

    "rearwheeltire": "rear wheel tire",
    "rear wheel-tyre": "rear wheel tire",
    "reartyre": "rear wheel tire",

    "toplight1": "top light one",
    "top light 1": "top light one",
    "top-light-one": "top light one",

    "toplight2": "top light two",
    "top light 2": "top light two",
    "top-light-two": "top light two",

    "seat1": "correct seat one",
    "seat one": "correct seat one",
    "correct seat 1": "correct seat one",

    "seat2": "correct seat two",
    "seat two": "correct seat two",
    "correct seat 2": "correct seat two",

    "cabin closed": "closed cabin",
    "closed-cabin": "closed cabin",

    "lightbar": "light bar",
    "frontwheelcover": "front wheel cover",
    "steeringwheel": "steering wheel",
}


def canonize(name: str) -> str:
    k = name.strip().lower().replace("_", " ").replace("-", " ").replace("  ", " ")
    k2 = k.replace(" ", "")
    if k2 in ALIASES:
        return ALIASES[k2]
    if k in ALIASES:
        return ALIASES[k]
    return k

# =========================
# Helpers
# =========================
def _encode_jpeg_b64(img_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

def overlay_big_label(img, text: str, bad: bool = False):
    h, w = img.shape[:2]
    pad = 16
    scale = max(0.8, min(1.6, w / 1000.0))
    thickness = int(2 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = text if text else "Unknown"
    size, _ = cv2.getTextSize(label, font, scale, thickness)
    box_w = size[0] + 2 * pad
    box_h = size[1] + 2 * pad
    x = max(10, (w - box_w) // 2)
    y = 20 + box_h
    bar = (50, 50, 200) if bad else (70, 180, 80)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 6, y - box_h - 6), (x + box_w + 6, y + 6), bar, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.putText(img, label, (x, y - 8), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# -------------------------
# V4L2 capture utilities
# -------------------------
BackendLinux = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)


def _device_exists(dev: Union[str,int]) -> bool:
    if isinstance(dev, int):
        return os.path.exists(f"/dev/video{dev}")
    return os.path.exists(str(dev))


def _try_open(dev: Union[str,int]) -> Optional[cv2.VideoCapture]:
    """Try to open a Linux V4L2 device (string path or /dev/video index)."""
    cap = cv2.VideoCapture(dev, BackendLinux)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FRAME_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFERSIZE_TRY)
    except Exception:
        pass
    # Warmup flush
    for _ in range(WARMUP_GRABS):
        cap.grab()
    return cap


def capture_once(dev: Union[str,int]):
    """Open → warm → read one → release (always)."""
    if isinstance(dev, str) and not os.path.exists(dev):
        time.sleep(RELEASE_COOLDOWN)
        return False, None, "device path not found"
    cap = _try_open(dev)
    if cap is None:
        time.sleep(RELEASE_COOLDOWN)
        return False, None, "open failed"
    ok, frame = False, None
    try:
        retries = 0
        while retries < MAX_READ_RETRY:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            retries += 1
            time.sleep(RETRY_SLEEP_S)
        if not ok or frame is None:
            return False, None, "read failed"
        return True, frame, "ok"
    finally:
        try:
            cap.release()
        except Exception:
            pass
        time.sleep(RELEASE_COOLDOWN)


def quick_probe(dev: Union[str,int]) -> bool:
    if isinstance(dev, str) and not os.path.exists(dev):
        return False
    cap = _try_open(dev)
    if cap is None:
        return False
    ok, _ = cap.read()
    try:
        cap.release()
    except Exception:
        pass
    time.sleep(0.05)
    return bool(ok)


def dev_to_v4l2_arg(dev: Union[str, int]) -> str:
    """Return a safe device string for v4l2-ctl's -d argument."""
    if isinstance(dev, int):
        return f"/dev/video{dev}"
    return str(dev)


def v4l2_get_zoom(dev: Union[str,int]) -> Tuple[bool, Optional[int], str]:
    """Read current zoom_absolute using v4l2-ctl -C."""
    if not shutil.which("v4l2-ctl") and not os.path.exists(V4L2_CTL):
        return False, None, "v4l2-ctl not found"
    d = dev_to_v4l2_arg(dev)
    try:
        r = subprocess.run([V4L2_CTL, "-d", d, "-C", "zoom_absolute"],
                           capture_output=True, text=True, timeout=3)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return False, None, out.strip()
        # Expected: "zoom_absolute: 160"
        m = re.search(r"(-?\d+)", out)
        if m:
            return True, int(m.group(1)), out.strip()
        return False, None, out.strip()
    except Exception as e:
        return False, None, str(e)


def v4l2_set_zoom(dev: Union[str,int], zoom_value: int) -> Tuple[bool, Optional[int], str]:
    """Set zoom_absolute using v4l2-ctl -c, then read back."""
    if not shutil.which("v4l2-ctl") and not os.path.exists(V4L2_CTL):
        return False, None, "v4l2-ctl not found"
    d = dev_to_v4l2_arg(dev)
    # Clamp to a sane range often seen on Logitech cams (100..500)
    z = int(max(1, min(10000, zoom_value)))
    try:
        r1 = subprocess.run([V4L2_CTL, "-d", d, "-c", f"zoom_absolute={z}"],
                            capture_output=True, text=True, timeout=3)
        out = (r1.stdout or "") + (r1.stderr or "")
        if r1.returncode != 0:
            return False, None, out.strip()
        ok, zread, msg = v4l2_get_zoom(dev)
        if not ok:
            return False, None, msg
        return True, zread, f"set_ok; readback={zread}"
    except Exception as e:
        return False, None, str(e)

# =========================
# Inference
# =========================

def detect_features(frame) -> Tuple[Set[str], any]:
    """Run YOLO (ONNX) on a single frame; return (found_features_set, plotted_bgr)."""
    results = model.predict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
    r = results[0]
    if CUSTOM_NAMES_DICT is not None:
        try:
            r.names = CUSTOM_NAMES_DICT  # per-result override
        except Exception:
            pass

    found: Set[str] = set()
    if r.boxes is not None and r.boxes.cls is not None and len(r.boxes.cls) > 0:
        for c in r.boxes.cls:
            try:
                cls_name = r.names[int(c)]
                found.add(canonize(cls_name))
            except Exception:
                continue

    plotted = r.plot()  # BGR
    return found, plotted


# =========================
# Classification logic (unchanged)
# =========================

def classify_from_found(found_all: Set[str]) -> Tuple[str, List[str]]:
    missing: List[str] = [n for n in REQ_COMMON if n not in found_all]

    # front
    if FRONT_VARIANTS[0] in found_all:
        front = "R"
    elif FRONT_VARIANTS[1] in found_all:
        front = "W"
    else:
        front = None
        missing.append("front wheel roller/tire")

    # rear
    if REAR_VARIANTS[0] in found_all:
        rear = "R"
    elif REAR_VARIANTS[1] in found_all:
        rear = "W"
    else:
        rear = None
        missing.append("rear wheel roller/tire")

    # top
    if TOP_VARIANTS[0] in found_all:
        top = "1"
    elif TOP_VARIANTS[1] in found_all:
        top = "2"
    else:
        top = None
        missing.append("top light one/two")

    # seat
    if SEAT_VARIANTS[0] in found_all:
        seat = "1"
    elif SEAT_VARIANTS[1] in found_all:
        seat = "2"
    else:
        seat = None
        missing.append("correct seat one/two")

    if missing:
        return "Unknown", missing

    label = f"{front}{rear}{top}{seat}C"
    valid = {
        "RR11C", "RR12C", "RR21C", "RR22C",
        "RW11C", "RW12C", "RW21C", "RW22C",
        "WR11C", "WR12C", "WR21C", "WR22C",
        "WW11C", "WW12C", "WW21C", "WW22C",
    }
    if label in valid:
        return label, []
    return "Unknown", ["combination not in whitelist"]


def close_all():
    cv2.destroyAllWindows()

atexit.register(close_all)

# =========================
# Flask app
# =========================
app = Flask(__name__)

# Internal mapping: UI index 0..N-1 → actual device path/index
# Validate devices and build a clean list
_valid_devices: List[Union[str,int]] = []
for d in WEBCAM_DEVICES:
    if isinstance(d, int):
        _valid_devices.append(d)
    else:
        if _device_exists(d):
            _valid_devices.append(d)
# Fallback to /dev/video[0..2]
if not _valid_devices:
    _valid_devices = [0, 1, 2]

# ---- Calibration streaming state (single camera at a time) ----
_calib_idx_lock = threading.Lock()
_calib_idx: Optional[int] = None  # which UI index is streaming (None = no stream)

def _set_calib_idx(idx: Optional[int]):
    global _calib_idx
    with _calib_idx_lock:
        _calib_idx = idx

def _get_calib_idx() -> Optional[int]:
    with _calib_idx_lock:
        return _calib_idx

# ---------------- Routes: Detector page ----------------
@app.route("/")
def index():
    cams_status = {}
    for i, dev in enumerate(_valid_devices):
        try:
            cams_status[i] = quick_probe(dev)
        except Exception:
            cams_status[i] = False
    return render_template(
        "index3.html",
        model_name=os.path.basename(WEIGHTS_PATH),
        cams=list(range(len(_valid_devices))),  # UI indices 0..N-1
        cams_status=cams_status,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
    )

@app.route("/detect_cam/<int:idx>", methods=["POST"])
def detect_cam(idx: int):
    """Capture one frame from cam (by UI index), run YOLO, overlay per-image label, return b64 image + details."""
    if idx < 0 or idx >= len(_valid_devices):
        return jsonify({"ok": False, "error": f"Camera index {idx} out of range"}), 400
    dev = _valid_devices[idx]

    with capture_lock:
        ok, frame, msg = capture_once(dev)
        if not ok or frame is None:
            return jsonify({
                "ok": False,
                "idx": idx,
                "image": "",
                "found": [],
                "label": "Unknown",
                "missing": ["no frame"],
                "msg": msg,
            }), 200

        found, plotted = detect_features(frame)
        label, missing = classify_from_found(found)
        overlay_big_label(plotted, label, bad=(label == "Unknown"))
        final_img_b64 = _encode_jpeg_b64(plotted)

        return jsonify({
            "ok": True,
            "idx": idx,
            "image": final_img_b64,
            "found": sorted(list(found)),
            "label": label,
            "missing": missing,
            "ts": time.time(),
        })

@app.route("/classify", methods=["POST"])
def classify_union():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        found_list = payload.get("found", [])
        found_all = {canonize(x) for x in found_list if isinstance(x, str)}
        label, missing = classify_from_found(found_all)
        return jsonify({"label": label, "missing": missing, "found": sorted(found_all)})
    except Exception as e:
        return jsonify({"label": "Unknown", "missing": ["server error"], "error": str(e)}), 500

@app.route("/health")
def health():
    status = {}
    for i, dev in enumerate(_valid_devices):
        try:
            ok = quick_probe(dev)
            status[i] = {"opened": ok, "has_frame": ok, "failures": 0}
        except Exception:
            status[i] = {"opened": False, "has_frame": False, "failures": 0}
    return jsonify({"camera_status": status, "imgsz": IMG_SIZE, "conf": CONF_THRESHOLD})


# ---------------- Routes: Camera Calibration ----------------
@app.route("/calibrate")
def calibrate_page():
    cams_status = {}
    for i, dev in enumerate(_valid_devices):
        try:
            cams_status[i] = quick_probe(dev)
        except Exception:
            cams_status[i] = False
    return render_template(
        "camera_calibration.html",
        cams=list(range(len(_valid_devices))),
        cams_status=cams_status
    )

@app.route("/calib/start/<int:idx>", methods=["POST", "GET"])
def calib_start(idx: int):
    if idx < 0 or idx >= len(_valid_devices):
        return jsonify({"ok": False, "error": f"Camera index {idx} out of range"}), 400
    _set_calib_idx(idx)
    return jsonify({"ok": True, "idx": idx})

@app.route("/calib/stop", methods=["POST", "GET"])
def calib_stop():
    _set_calib_idx(None)
    return jsonify({"ok": True})

def _mjpeg_generator(idx: int):
    """Yield multipart JPEG frames from the selected camera, stopping if selection changes."""
    if idx < 0 or idx >= len(_valid_devices):
        return
    dev = _valid_devices[idx]

    # Single cam at a time across the whole app
    with capture_lock:
        cap = _try_open(dev)
        if cap is None:
            return
        try:
            # Stream loop
            while True:
                # Stop if the selected camera changed
                current = _get_calib_idx()
                if current != idx:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    time.sleep(0.01)
                    continue
                chunk = buf.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n")
                time.sleep(max(0.0, 1.0 / max(5, FRAME_FPS)))
        finally:
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(RELEASE_COOLDOWN)

@app.route("/calib/stream/<int:idx>")
def calib_stream(idx: int):
    # If user hits stream URL directly, set the selection
    if idx < 0 or idx >= len(_valid_devices):
        return "bad index", 400
    _set_calib_idx(idx)
    return Response(_mjpeg_generator(idx),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/calib/zoom", methods=["GET", "POST"])
def calib_zoom():
    """
    GET  /calib/zoom?idx=0           -> read zoom_absolute
    POST /calib/zoom {idx, zoom}     -> set zoom_absolute, then read back
    """
    try:
        if request.method == "GET":
            idx = int(request.args.get("idx", "-1"))
            if idx < 0 or idx >= len(_valid_devices):
                return jsonify({"ok": False, "error": "bad idx"}), 400
            ok, val, msg = v4l2_get_zoom(_valid_devices[idx])
            return jsonify({"ok": ok, "zoom": val, "msg": msg})

        data = request.get_json(force=True, silent=True) or {}
        idx = int(data.get("idx", -1))
        zoom = int(data.get("zoom", -1))
        if idx < 0 or idx >= len(_valid_devices):
            return jsonify({"ok": False, "error": "bad idx"}), 400
        if zoom < 0:
            return jsonify({"ok": False, "error": "bad zoom"}), 400
        ok, val, msg = v4l2_set_zoom(_valid_devices[idx], zoom)
        return jsonify({"ok": ok, "zoom": val, "msg": msg})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------- Route: MES / Tulip submission ----------------
@app.route("/send_mes", methods=["POST"])
def send_mes():
    """
    Receive detection results from the frontend and submit a QC record to Tulip MES.

    Expected JSON body:
      {
        "found":    ["closed cabin", "light bar", ...],   // list of detected feature strings
        "label":    "RR11C",                              // final assembly label
        "missing":  [],                                   // list of missing feature strings
        "start_ts": 1700000000.0                          // Unix timestamp (float) when inspection started
      }

    Returns:
      { "ok": true,  "qc_id": "QC-XXXXXXXX", "label": "RR11C" }
      { "ok": false, "error": "..." }
    """
    try:
        from tulip_client import TulipClient, build_record_from_domain, TULIP_TABLE_UID

        payload      = request.get_json(force=True, silent=True) or {}
        found_list   = payload.get("found", [])
        label        = payload.get("label", "Unknown")
        missing_list = payload.get("missing", [])
        start_ts     = float(payload.get("start_ts", time.time()))

        found_set = {canonize(x) for x in found_list if isinstance(x, str)}

        # --- Derive inspection boolean fields from detected features ---
        status_motor_cover_fixed       = "front wheel cover" in found_set
        status_function_of_roller      = ("front wheel roller" in found_set or
                                          "rear wheel roller"  in found_set)
        status_function_of_cabin       = "closed cabin" in found_set
        status_position_of_light       = ("top light one" in found_set or
                                          "top light two" in found_set)
        status_tires_correct_assembled = ("front wheel tire"   in found_set or
                                          "rear wheel tire"    in found_set or
                                          "front wheel roller" in found_set or
                                          "rear wheel roller"  in found_set)
        status_all_parts_attached      = len(missing_list) == 0
        overall_result                 = (label != "Unknown") and status_all_parts_attached

        # --- Tire type strings ---
        front_tire_type = (
            "roller" if "front wheel roller" in found_set else
            "tire"   if "front wheel tire"   in found_set else
            "unknown"
        )
        rear_tire_type = (
            "roller" if "rear wheel roller" in found_set else
            "tire"   if "rear wheel tire"   in found_set else
            "unknown"
        )

        # --- Top light mode string ---
        top_light_mode = (
            "one" if "top light one" in found_set else
            "two" if "top light two" in found_set else
            "unknown"
        )

        # --- Timestamps ---
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt   = datetime.now(tz=timezone.utc)

        # --- Generate unique QC ID ---
        qc_id = "QC-" + str(uuid.uuid4())[:8].upper()

        # --- Build and submit Tulip record ---
        client = TulipClient()
        record = build_record_from_domain(
            id_value=qc_id,
            station_id=STATION_ID,
            start_dt=start_dt,
            end_dt=end_dt,
            final_model_label=label,
            status_motor_cover_fixed=status_motor_cover_fixed,
            status_function_of_roller=status_function_of_roller,
            status_function_of_cabin=status_function_of_cabin,
            status_position_of_light=status_position_of_light,
            top_light_mode=top_light_mode,
            status_tires_correct_assembled=status_tires_correct_assembled,
            front_tire_type=front_tire_type,
            rear_tire_type=rear_tire_type,
            status_all_parts_attached=status_all_parts_attached,
            overall_result=overall_result,
            found_features=sorted(found_set),
            missing_features=missing_list,
        )
        client.create_record(TULIP_TABLE_UID, record)

        return jsonify({"ok": True, "qc_id": qc_id, "label": label})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
