# capture_object_3cam.py
"""
Capture one object image from each Azure Kinect camera.

Use case:
  - Static object capture for Gaussian Splatting / reconstruction.
  - Preview all available cameras.
  - Manually adjust exposure/gain/white balance from keyboard.
  - Press SPACE to save one image per camera with the same timestamp.

Outputs:
  ObjectCapture/cam0/object_<timestamp>.png
  ObjectCapture/cam1/object_<timestamp>.png
  ObjectCapture/cam2/object_<timestamp>.png

Controls:
  SPACE  save one image from each active camera
  ESC    exit

  Exposure:
    1 / 2  decrease / increase exposure by 1000 us
    3 / 4  decrease / increase exposure by 5000 us

  Gain:
    5 / 6  decrease / increase gain

  White balance:
    7 / 8  decrease / increase white balance
    0      disable manual WB setting for next restart only if commented manually

Notes:
  - Azure Kinect exposure is in microseconds.
  - Typical exposure values: 8000, 12000, 15000, 20000, 30000.
  - Typical gain values: 32, 64, 128.
  - Typical WB values: 4500, 5500, 6500.
"""

import os
import sys
from datetime import datetime

import cv2

try:
    from pyk4a import PyK4A, Config, ColorResolution, DepthMode, WiredSyncMode
except ImportError:
    print("ERROR: pyk4a not installed. Run: pip install pyk4a")
    sys.exit(1)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "ObjectCapture")

# Change this if you only want two cameras, e.g. [0, 1]
CAMERA_IDS = [0, 1]

# For object capture, standalone is usually fine.
# If you want wired sync, set:
#   cam0 MASTER, others SUBORDINATE
USE_WIRED_SYNC = False

COLOR_RESOLUTION = ColorResolution.RES_1080P
# Options:
#   ColorResolution.RES_720P
#   ColorResolution.RES_1080P
#   ColorResolution.RES_1440P
#   ColorResolution.RES_2160P

# Initial manual settings
EXPOSURE_US = 30000
GAIN = 64
WHITEBALANCE = 5500

# Set to False if colors look weird/orange and you want auto WB
USE_MANUAL_WHITEBALANCE = False


def make_config(device_id: int) -> Config:
    if USE_WIRED_SYNC:
        sync_mode = WiredSyncMode.MASTER if device_id == 0 else WiredSyncMode.SUBORDINATE
    else:
        sync_mode = WiredSyncMode.STANDALONE

    return Config(
        color_resolution=COLOR_RESOLUTION,
        depth_mode=DepthMode.OFF,
        synchronized_images_only=False,
        wired_sync_mode=sync_mode,
    )


def set_camera_controls(cam: PyK4A, exposure_us: int, gain: int, whitebalance: int | None):
    """
    pyk4a 1.5.0 exposes controls as properties:
      cam.exposure
      cam.gain
      cam.whitebalance

    Some installations may not support one of them; we catch and print warnings.
    """
    try:
        cam.exposure = int(exposure_us)
    except Exception as e:
        print(f"WARNING: could not set exposure: {e}")

    try:
        cam.gain = int(gain)
    except Exception as e:
        print(f"WARNING: could not set gain: {e}")

    if whitebalance is not None:
        try:
            cam.whitebalance = int(whitebalance)
        except Exception as e:
            print(f"WARNING: could not set whitebalance: {e}")


def color_to_bgr(color):
    """
    Azure Kinect color frames normally arrive as BGRA in pyk4a.
    If your image has weird colors, try replacing this with:
        return color[:, :, :3].copy()
    """
    return cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)


def draw_status(frame, label, exposure_us, gain, whitebalance):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 92), (20, 20, 20), -1)

    wb_txt = "AUTO/unchanged" if whitebalance is None else str(whitebalance)
    lines = [
        f"{label}",
        f"Exposure: {exposure_us} us   Gain: {gain}   WB: {wb_txt}",
        "SPACE save | ESC exit | 1/2 exp +/-1000 | 3/4 exp +/-5000 | 5/6 gain | 7/8 WB",
    ]

    y = 24
    for i, line in enumerate(lines):
        color = (0, 220, 80) if i == 0 else (220, 220, 220)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 28

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    exposure_us = EXPOSURE_US
    gain = GAIN
    whitebalance = WHITEBALANCE if USE_MANUAL_WHITEBALANCE else None

    cameras = []

    try:
        print("Initializing cameras...")
        print(f"Camera IDs: {CAMERA_IDS}")
        print(f"Resolution: {COLOR_RESOLUTION}")
        print(f"Wired sync: {USE_WIRED_SYNC}")
        print()

        # If wired sync is enabled, start master first.
        ids_to_start = CAMERA_IDS
        if USE_WIRED_SYNC and 0 in CAMERA_IDS:
            ids_to_start = [0] + [i for i in CAMERA_IDS if i != 0]

        for device_id in ids_to_start:
            print(f"Starting device {device_id}...")
            cam = PyK4A(make_config(device_id), device_id=device_id)
            cam.start()
            set_camera_controls(cam, exposure_us, gain, whitebalance)
            cameras.append((device_id, cam))

            win = f"cam{device_id}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 960, 540)

        print("\nCameras ready.")
        print("SPACE = save one image per camera")
        print("ESC = exit\n")

        latest_frames = {}

        while True:
            for device_id, cam in cameras:
                cap = cam.get_capture()
                if cap.color is None:
                    continue

                frame = color_to_bgr(cap.color)
                latest_frames[device_id] = frame

                disp = draw_status(frame, f"cam{device_id}", exposure_us, gain, whitebalance)
                cv2.imshow(f"cam{device_id}", disp)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("Exiting.")
                break

            elif key == 32:  # SPACE
                if len(latest_frames) == 0:
                    print("No frames available yet.")
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                print(f"\nSaving object capture {ts}")

                for device_id, frame in sorted(latest_frames.items()):
                    cam_dir = os.path.join(OUT_DIR, f"cam{device_id}")
                    os.makedirs(cam_dir, exist_ok=True)

                    path = os.path.join(cam_dir, f"object_{ts}.png")
                    cv2.imwrite(path, frame)
                    print(f"  Saved cam{device_id}: {path}")

                print()

            elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8")]:
                if key == ord("1"):
                    exposure_us = max(500, exposure_us - 1000)
                elif key == ord("2"):
                    exposure_us += 1000
                elif key == ord("3"):
                    exposure_us = max(500, exposure_us - 5000)
                elif key == ord("4"):
                    exposure_us += 5000
                elif key == ord("5"):
                    gain = max(0, gain - 16)
                elif key == ord("6"):
                    gain += 16
                elif key == ord("7"):
                    if whitebalance is None:
                        whitebalance = WHITEBALANCE
                    whitebalance = max(2500, whitebalance - 250)
                elif key == ord("8"):
                    if whitebalance is None:
                        whitebalance = WHITEBALANCE
                    whitebalance += 250

                for _, cam in cameras:
                    set_camera_controls(cam, exposure_us, gain, whitebalance)

                print(f"Settings -> exposure={exposure_us} us, gain={gain}, whitebalance={whitebalance}")

    finally:
        for _, cam in cameras:
            try:
                cam.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
