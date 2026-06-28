#!/usr/bin/env python3
# cam_control.py — Pi 5 MJPEG streamer with zoom & multi-endpoints

import os, time, threading, glob
from typing import Optional, Union, List, Dict

import cv2
from flask import Flask, jsonify, request, Response, make_response

app = Flask(__name__)

# --------------------------
# Helpers
# --------------------------
def apply_zoom_center(frame, zoom: float):
    """Software (digital) zoom around the center; zoom >= 1.0."""
    try:
        z = float(zoom)
    except Exception:
        z = 1.0
    if z <= 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w = max(1, int(w / z))
    new_h = max(1, int(h / z))
    x1 = max(0, (w - new_w) // 2)
    y1 = max(0, (h - new_h) // 2)
    roi = frame[y1:y1 + new_h, x1:x1 + new_w]
    if roi.size == 0:
        return frame
    return cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)

def device_exists(dev: Union[str, int]) -> bool:
    if isinstance(dev, int):
        return os.path.exists(f"/dev/video{dev}")
    return os.path.exists(str(dev))

def list_by_id() -> List[Dict[str, str]]:
    out = []
    byid_dir = "/dev/v4l/by-id"
    if os.path.isdir(byid_dir):
        for p in sorted(glob.glob(os.path.join(byid_dir, "*"))):
            try:
                target = os.path.realpath(p)
                out.append({"id_path": p, "target": target})
            except Exception:
                continue
    return out

def list_video_nodes() -> List[str]:
    return sorted(glob.glob("/dev/video[0-9]*"))

# --------------------------
# Camera session
# --------------------------
class CameraSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None
        self.device: Optional[Union[str, int]] = None
        self.width = 1280
        self.height = 720
        self.fps = 30
        self.fourcc = "MJPG"  # or "YUYV"
        self.running = False
        self.zoom = 1.0
        self.backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        self.buffersize = 1
        self.warmup_grabs = 8
        self.max_retry = 60
        self.retry_sleep = 0.01

    def _configure(self):
        # Try FOURCC, then dimensions/fps, buffer size
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        except Exception:
            pass
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffersize)
        except Exception:
            pass
        # Warmup (grab discards)
        for _ in range(self.warmup_grabs):
            self.cap.grab()

    def open(self, device: Union[str, int], width: int, height: int, fps: int, fourcc: str = "MJPG") -> bool:
        with self.lock:
            # Close previous if any
            self._close_unlocked()

            self.device = device
            self.width = int(width)
            self.height = int(height)
            self.fps = int(fps)
            self.fourcc = fourcc if fourcc in ("MJPG", "YUYV") else "MJPG"

            # Sanity
            if isinstance(device, str) and not os.path.exists(device):
                self.device = None
                return False

            cap = cv2.VideoCapture(device, self.backend)
            if not cap.isOpened():
                try: cap.release()
                except Exception: pass
                self.cap = None
                self.device = None
                return False

            self.cap = cap
            self._configure()
            self.running = True
            return True

    def _close_unlocked(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.running = False

    def close(self):
        with self.lock:
            self._close_unlocked()

    def set_zoom(self, z: float):
        z = max(1.0, min(3.0, float(z)))
        with self.lock:
            self.zoom = z

    def read(self):
        """Return a frame (BGR) or None."""
        with self.lock:
            cap = self.cap
            running = self.running
            zoom = self.zoom
        if cap is None or not running:
            time.sleep(0.02)
            return None

        tries = 0
        ok, frame = False, None
        while tries < self.max_retry:
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            tries += 1
            time.sleep(self.retry_sleep)

        if not ok or frame is None:
            return None

        if zoom > 1.0:
            frame = apply_zoom_center(frame, zoom)
        return frame

# Global session
SESSION = CameraSession()

# --------------------------
# Routes
# --------------------------
@app.get("/")
def index():
    # Minimal test UI
    html = f"""
    <!doctype html><html><head><meta charset="utf-8"><title>Cam Stream</title></head>
    <body style="background:#0b0b0f;color:#e8edf2;font-family:ui-sans-serif">
      <h2>Cam Stream</h2>
      <p>/devices → choose a device id or /dev/videoN</p>
      <form method="post" action="/start">
        <label>Device: <input name="device" value="/dev/video0" /></label>
        <label>Width: <input name="width" value="1280" size="6"/></label>
        <label>Height: <input name="height" value="720" size="6"/></label>
        <label>FPS: <input name="fps" value="30" size="4"/></label>
        <label>FOURCC: <input name="fourcc" value="MJPG" size="6"/></label>
        <button type="submit">Start</button>
      </form>
      <form method="post" action="/stop" style="margin-top:8px">
        <button type="submit">Stop</button>
      </form>
      <form method="post" action="/set_zoom" style="margin-top:8px">
        <label>Zoom: <input name="zoom" value="{SESSION.zoom:.1f}" size="4"/></label>
        <button type="submit">Set Zoom</button>
      </form>
      <div style="margin-top:12px">
        <img src="/stream" style="max-width:90vw;border:1px solid #333;border-radius:8px"/>
      </div>
    </body></html>
    """
    return make_response(html, 200)

@app.get("/favicon.ico")
def favicon():
    return ("", 204)

@app.get("/devices")
def devices():
    byid = list_by_id()
    videos = list_video_nodes()
    return jsonify({
        "by_id": byid,
        "video_nodes": videos
    })

@app.post("/start")
def start():
    # Accept JSON or form
    payload = request.get_json(silent=True) or request.form or {}
    dev = payload.get("device", "/dev/video0")
    # Allow int device like 0 as well
    try:
        dev_cast: Union[str,int] = int(dev) if isinstance(dev, str) and dev.isdigit() else dev
    except Exception:
        dev_cast = dev

    width  = int(payload.get("width", 1280))
    height = int(payload.get("height", 720))
    fps    = int(payload.get("fps", 30))
    fourcc = str(payload.get("fourcc", "MJPG")).upper()
    if fourcc not in ("MJPG", "YUYV"):
        fourcc = "MJPG"

    if not device_exists(dev_cast):
        return jsonify({"ok": False, "error": f"device not found: {dev_cast}"}), 400

    ok = SESSION.open(dev_cast, width, height, fps, fourcc)
    return jsonify({
        "ok": ok,
        "device": dev_cast,
        "width": width, "height": height, "fps": fps, "fourcc": fourcc
    }), (200 if ok else 500)

@app.post("/stop")
def stop():
    SESSION.close()
    return jsonify({"ok": True})

@app.post("/set_zoom")
def set_zoom():
    payload = request.get_json(silent=True) or request.form or {}
    try:
        z = float(payload.get("zoom", 1.0))
    except Exception:
        z = 1.0
    SESSION.set_zoom(z)
    return jsonify({"ok": True, "zoom": SESSION.zoom})

@app.get("/snapshot")
def snapshot():
    frame = SESSION.read()
    if frame is None:
        return jsonify({"ok": False, "error": "no frame"}), 503
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return jsonify({"ok": False, "error": "jpeg encode failed"}), 500
    resp = Response(jpg.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp

def mjpeg_generator():
    # Stream until client disconnects; never touches an undefined local cap
    while True:
        frame = SESSION.read()
        if frame is None:
            # If not running, idle a bit
            time.sleep(0.02)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            # Skip frame on encode failure
            continue
        data = jpg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
               data + b"\r\n")

@app.get("/stream")
def stream():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})

if __name__ == "__main__":
    # Light ARM threading caps (optional)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
