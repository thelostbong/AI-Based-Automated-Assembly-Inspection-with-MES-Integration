# AI-Based Automated Assembly Inspection with MES Integration

A computer-vision quality-control system that classifies LEGO assembly variants in real time using a Raspberry Pi 5, three USB cameras, and a custom-trained YOLOv12 model — with results reported automatically to a **Tulip MES** (Manufacturing Execution System) table for traceability.

This project replicates and extends a Master's thesis at **Deggendorf Institute of Technology** on automated assembly inspection, moving the original detection pipeline from a development laptop onto an edge device (Raspberry Pi 5) and wiring it into a live MES.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Configuration (.env)](#configuration-env)
- [Running the App](#running-the-app)
- [MES Integration](#mes-integration)
- [Camera Calibration & Diagnostics](#camera-calibration--diagnostics)
- [Known Issues & Troubleshooting](#known-issues--troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

A LEGO model (a small vehicle kit) moves through 16 valid assembly variants depending on which interchangeable parts are fitted — front/rear wheel type (roller vs. tire), top light configuration, and seat variant. The system:

1. Captures one frame from each of three fixed-position USB cameras.
2. Runs a YOLO object-detection model (exported to ONNX) on each frame to find assembly features (cabin closed, light bar, wheel type, seat type, etc.).
3. Unions the features found across all three camera views.
4. Applies a deterministic rule set to resolve the union into one of 16 valid variant codes (e.g. `RW21C`), or `Unknown` if the combination doesn't match any valid variant.
5. Posts the result — along with timestamps, duration, and per-feature status — to a Tulip MES table as a new record, for downstream dashboards and traceability.

The whole pipeline runs locally on the Pi; only the final structured result is sent over the network to Tulip.

## How It Works

```
   Cam 0          Cam 1          Cam 2
     │              │              │
     ▼              ▼              ▼
  capture()      capture()      capture()      (sequential open→warm→snap→close
     │              │              │            to avoid USB bus contention)
     ▼              ▼              ▼
  YOLOv12s ONNX inference (per camera)
     │              │              │
     ▼              ▼              ▼
  features found (canonicalized class names)
     │              │              │
     └──────────────┴──────────────┘
                    ▼
         union of all features found
                    ▼
        deterministic classification
        (Front)(Rear)(Top)(Seat)C
                    ▼
        valid code?  ──No──▶  "Unknown" + missing/ambiguous parts
                    │
                   Yes
                    ▼
     POST record to Tulip MES table
     (station, timestamps, duration,
      label, per-feature booleans,
      found/missing feature lists)
```

The detector runs as a Flask web app. The browser-side UI calls `/detect_cam/<idx>` for each camera in turn, then posts the union of detected features to `/classify` to get the final variant code. Once a result is produced, the frontend automatically submits it to the MES via `/send_mes` — no manual confirmation button required.

## Hardware

| Component | Spec |
|---|---|
| Compute | Raspberry Pi 5 (16 GB RAM) |
| Cameras | 3× Logitech C922 Pro Stream Webcam (USB, MJPG @ 1280×720) |
| Model | YOLOv12s, trained on a custom 20-class dataset, exported to ONNX (`best.onnx`) |
| MES | Tulip (cloud), reached over HTTPS via REST API |

## Repository Structure

```
.
├── app_onnx_pi.py        # Main Flask app: camera capture, YOLO inference,
│                          # classification, calibration UI, MES submission
├── cam_check.py           # CLI utility — probe /dev/video* devices, save a test snapshot
├── cam_control.py         # Standalone MJPEG streaming/zoom test server (dev/debug tool)
├── tulip_client.py        # Tulip REST API client + payload builder (build_record_from_domain)
├── roller_detector.py     # Marker-based rolling/motion detector (auxiliary feature)
├── data.yaml              # YOLO class names / dataset config used by the model
├── best.onnx               # Exported YOLOv12s model weights (not tracked — see below)
├── templates/              # Flask Jinja2 templates (detector UI, calibration page)
├── static/                 # CSS/JS assets for the web UI
├── .env.example             # Template for required environment variables
└── .gitignore
```

> **Note:** `best.onnx` is a large binary model file. If you're cloning this repo, either fetch it separately (see [Setup](#setup)) or use [Git LFS](https://git-lfs.github.com/) if you choose to track it in version control.

## Setup

### 1. Flash and prep the Raspberry Pi

- Flash Raspberry Pi OS (Bookworm) using Raspberry Pi Imager.
- Enable SSH and set the hostname/credentials during imaging, or via `raspi-config` afterward.
- Connect to Wi-Fi using `nmcli`. **If your SSID contains special characters like `!` (e.g. `FRITZ!Box`), wrap it in single quotes** to avoid shell interpretation issues:
  ```bash
  sudo nmcli device wifi connect 'FRITZ!Box 7520 LP' password 'your-password'
  ```
  Editing `network-config` directly on the `bootfs` partition is **not reliable** on Bookworm, since NetworkManager + cloud-init only honor it on first boot. If Wi-Fi config breaks, reflashing the SD card is usually faster than trying to patch it live.

### 2. Connect cameras

Plug in all three USB webcams, then confirm device paths:

```bash
ls /dev/video*
ls -l /dev/v4l/by-id/
```

Note which `/dev/videoN` index corresponds to which physical camera — these are needed for the `CAM0`/`CAM1`/`CAM2` environment variables below. You can sanity-check each camera with:

```bash
python cam_check.py --devices /dev/video0 /dev/video2 /dev/video4
```

This opens each device, grabs a frame, and saves a test snapshot (`snapshot_<device>.jpg`) so you can visually confirm the right camera is mapped to the right index.

### 3. Set up the Python environment

```bash
cd ~/testapp
python3 -m venv piyolo
source piyolo/bin/activate
pip install -r requirements.txt   # see below if you don't have one yet
```

If you don't already have a `requirements.txt`, install the core dependencies directly:

```bash
pip install flask opencv-python ultralytics onnxruntime pyyaml python-dotenv requests
```

> **Recreate the venv after restoring from a backup.** Copying a venv's `bin/` directory (e.g. from a Windows backup) loses the symlinks Python relies on. Always create a fresh venv on the Pi and reinstall dependencies rather than copying one over.

### 4. Place model & config files

Copy `best.onnx` and `data.yaml` into your project directory (e.g. `~/testapp/`), matching the paths you'll set in `.env`.

## Configuration (.env)

This project loads all device-specific and secret configuration from a `.env` file — **never commit this file**. Copy the template and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `WEIGHTS_PATH` | Absolute path to `best.onnx` |
| `DATA_YAML_PATH` | Absolute path to `data.yaml` |
| `CAM0`, `CAM1`, `CAM2` | Device paths for each camera (e.g. `/dev/video0`); prefer stable `/dev/v4l/by-id/...` symlinks over numeric indices if available |
| `IMG_SIZE`, `CONF_THRESHOLD` | YOLO inference image size and confidence threshold |
| `HOST`, `PORT` | Flask bind address/port (default `0.0.0.0:5000`) |
| `FRAME_WIDTH`, `FRAME_HEIGHT`, `FRAME_FPS`, `FOURCC` | Camera capture settings |
| `STATION_ID` | Identifier reported to Tulip for this inspection station |
| `TULIP_BASE` | Tulip API base URL, e.g. `https://<tenant>.tulip.co/api/v3` |
| `TULIP_AUTH_HEADER` | HTTP Basic auth header for your **own** Tulip API key (`Basic <base64>`) |
| `TULIP_TABLE_UID` | UID of the Tulip table records are written to |
| `MACHINE_ID`, `UID_ATTRIBUTE_ID`, `TABLE_ID_ATTRIBUTE_ID` | Tulip machine/attribute IDs, if using attribute reporting |

**⚠️ Security note:** generate your own Tulip API key (Tulip → Settings → API Tokens) rather than reusing one shared by a teammate or found elsewhere — API keys are tied to an individual account and should not be shared or committed to source control. Double-check `.env` is listed in `.gitignore` before your first commit, and rotate any key that may have been exposed.

## Running the App

```bash
source ~/testapp/piyolo/bin/activate
cd ~/testapp
python app_onnx_pi.py
```

Then visit, from any device on the same network:

- `http://<pi-ip-address>:5000/` — main detector UI
- `http://<pi-ip-address>:5000/calibrate` — camera calibration page

Or locally on the Pi itself:

```bash
python app_onnx_pi.py &
chromium http://localhost:5000
```

### Key Flask routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Main detector UI |
| `/detect_cam/<idx>` | POST | Capture + run YOLO on one camera, return annotated image and detected features |
| `/classify` | POST | Resolve the union of detected features into a final variant code |
| `/send_mes` | POST | Build a Tulip record from the final result and submit it |
| `/health` | GET | Quick camera probe + current runtime settings |
| `/calibrate` | GET | Camera calibration UI (live stream + zoom control) |
| `/calib/stream/<idx>` | GET | MJPEG live stream for the selected camera |
| `/calib/zoom` | GET/POST | Read or set hardware zoom via `v4l2-ctl` |

## MES Integration

Detection results are sent to a Tulip table via `tulip_client.py`. Each record includes:

- Station ID, start/end timestamps (UTC, RFC3339), duration (ms)
- Final variant label (e.g. `RW21C`) or `Unknown`
- Per-feature boolean statuses (motor cover fixed, roller function, cabin function, light position, tires correctly assembled, all parts attached, overall pass/fail)
- Top light mode, front/rear tire type
- Found features list + count, missing features list

On success, the UI shows an inline MES status and a toast notification with the generated record ID (`QC-XXXXXXXX` format).

**Before your own deployment goes live:** create a personal Tulip API key rather than relying on one issued to a previous team member, since access tied to someone who has since left the project is fragile and not appropriate to keep using.

## Camera Calibration & Diagnostics

- **`/calibrate`** in the main app streams one camera at a time (MJPEG) and lets you adjust hardware zoom (`v4l2-ctl -c zoom_absolute=...`) with live readback. Switching the selected camera automatically stops the previous stream so only one camera is ever open at once.
- **`cam_control.py`** is a separate, standalone Flask server for lower-level camera testing — start/stop a capture session, adjust software zoom, and view a raw MJPEG stream or single snapshot, independent of the YOLO pipeline.
- **`cam_check.py`** is a one-off CLI sanity check: lists detected `/dev/video*` devices and `by-id` symlinks, opens each requested device, confirms it can read a frame at the requested resolution/FPS, and saves a snapshot.

## Known Issues & Troubleshooting

- **Wi-Fi SSIDs with `!`** (e.g. `FRITZ!Box`) can silently fail in `nmcli` and bash if not single-quoted.
- **`network-config` on the `bootfs` partition** is only applied on first boot under Bookworm's NetworkManager/cloud-init setup — don't rely on editing it post-setup; reflash instead.
- **`load_dotenv()` ordering matters** — it must be called before any other code reads environment-dependent variables, or those values will silently fall back to defaults.
- **Route order matters in Flask** — `app.run()` must be the last line; any `@app.route` defined after it will never be registered.
- **Backing up the venv naively breaks it** — copying `piyolo/bin/` across machines (e.g. from a Windows backup tool) loses symlinks. Recreate the venv from scratch after any restore.
- **Hardcoded paths**: some scripts may still reference an older `/home/lego/testapp` path from earlier development — update these (e.g. via `sed`) to match your actual install path (e.g. `/home/<your-username>/testapp`).
- **Backgrounding Flask**: when working over SSH and also opening a local browser, launch Flask with `&` first so the terminal session isn't blocked.

## Roadmap

- [ ] Replace the inherited Tulip API credentials with a personal API key
- [ ] Warm-up / cooldown timing optimization for camera capture cycles
- [ ] Finalize university presentation materials

## License

MIT License — see [LICENSE](LICENSE).

---

*This repository accompanies a Master's thesis project at Deggendorf Institute of Technology. Camera images, model weights, and Tulip workspace details are specific to the original lab setup and will need to be adapted for any other deployment.*
