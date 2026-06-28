#!/usr/bin/env python3
# cam_check.py � quick sanity check for V4L2 webcams on Linux (Pi 5)
import argparse, os, glob, time
import cv2

def list_devices():
    devs = sorted(glob.glob("/dev/video[0-9]*"))
    byid = sorted(glob.glob("/dev/v4l/by-id/*"))
    print("Detected /dev/video* :", ", ".join(devs) or "none")
    if byid:
        print("By-ID symlinks:")
        for p in byid:
            try:
                target = os.path.realpath(p)
            except Exception:
                target = "?"
            print(f"  {p} -> {target}")
    print("-" * 50)
    return devs

def try_set_fourcc(cap, fourcc_str):
    try:
        return cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
    except Exception:
        return False

def probe_device(dev, width, height, fps, warmup=8, retries=30, sleep_s=0.02):
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    print(f"[{dev}] opening�")
    cap = cv2.VideoCapture(dev, backend)
    if not cap.isOpened():
        print(f"[{dev}] ERROR: open failed")
        return False

    # Try MJPG first, then fallback to YUYV
    ok_fourcc = try_set_fourcc(cap, "MJPG") or try_set_fourcc(cap, "YUYV")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    # Warmup flush (clears stale buffers)
    for _ in range(warmup):
        cap.grab()

    ok, frame = False, None
    n = 0
    while n < retries:
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        n += 1
        time.sleep(sleep_s)

    # Read back what we actually got
    got_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got_fps = cap.get(cv2.CAP_PROP_FPS)

    # Cleanup
    try:
        cap.release()
    except Exception:
        pass
    time.sleep(0.1)

    if not ok or frame is None:
        print(f"[{dev}] ERROR: read failed after {retries} tries")
        return False

    print(f"[{dev}] OK: {frame.shape[1]}x{frame.shape[0]} (requested {width}x{height})  ~FPS={got_fps:.1f}  FOURCC_set={bool(ok_fourcc)}")

    # Save a quick snapshot
    out_name = f"snapshot_{os.path.basename(str(dev))}.jpg"
    ok_jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[0]
    if ok_jpg:
        with open(out_name, "wb") as f:
            f.write(cv2.imencode(".jpg", frame)[1])
        print(f"[{dev}] saved {out_name}")
    else:
        print(f"[{dev}] WARN: could not encode JPEG")
    return True

def main():
    parser = argparse.ArgumentParser(description="Simple V4L2 camera check (Pi 5).")
    parser.add_argument("--devices", nargs="*", help="Device paths (e.g., /dev/video0 /dev/video2). If omitted, auto-detect.")
    parser.add_argument("--width",  type=int, default=1280, help="Request width (default 1280)")
    parser.add_argument("--height", type=int, default=720,  help="Request height (default 720)")
    parser.add_argument("--fps",    type=int, default=30,   help="Request FPS (default 30)")
    args = parser.parse_args()

    devices = args.devices if args.devices else list_devices()
    if not devices:
        print("No /dev/video* devices found.")
        return

    ok_all = True
    for dev in devices:
        ok = probe_device(dev, args.width, args.height, args.fps)
        ok_all &= ok

    print("-" * 50)
    print("RESULT:", "ALL OK" if ok_all else "Some devices failed")

if __name__ == "__main__":
    main()
