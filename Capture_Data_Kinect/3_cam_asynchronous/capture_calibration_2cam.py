# capture_calibration_2cam.py
"""
Captures synchronized image pairs from two Azure Kinect cameras for calibration.

It was not possible to conect 3 cameras at the same time. Capture with two cameras first, and then with the other two. 
Master should be the same camera in both sessions. 

At startup, select which pair you are capturing:
  [1] cam0 (master) + cam1 (subordinate)  → Session 1
  [2] cam0 (master) + cam2 (subordinate)  → Session 2

Controls:
  SPACE BAR → save pair (only when both cameras detect the board)
  ESC       → exit

Images are saved to:
  Calibration/cam0_imgs/   (always master)
  Calibration/cam1_imgs/   (session 1 subordinate)
  Calibration/cam2_imgs/   (session 2 subordinate)
"""

import cv2
import os
import sys
from datetime import datetime
import config

try:
    from pyk4a import PyK4A, Config, ColorControlCommand, ColorControlMode, ColorResolution, DepthMode, WiredSyncMode
except ImportError:
    print("ERROR: pyk4a not installed: pip install pyk4a")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_output_dirs():
    print("=" * 50)
    print("  Which camera pair are you capturing?")
    print("  [1] cam0 (master) + cam1 (subordinate)  — Session 1")
    print("  [2] cam0 (master) + cam2 (subordinate)  — Session 2")
    print("=" * 50)
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            sub_dir  = os.path.join(BASE_DIR, "Calibration", "cam1_imgs")
            sub_label = "cam1"
            break
        elif choice == "2":
            sub_dir  = os.path.join(BASE_DIR, "Calibration", "cam2_imgs")
            sub_label = "cam2"
            break
        else:
            print("Invalid input, enter 1 or 2.")

    mas_dir = os.path.join(BASE_DIR, "Calibration", "cam0_imgs")
    return mas_dir, sub_dir, sub_label


def main():
    mas_dir, sub_dir, sub_label = get_output_dirs()

    os.makedirs(mas_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    print(f"\ncam0  → {mas_dir}")
    print(f"{sub_label} → {sub_dir}")
    print("SPACE BAR = capture   ESC = exit\n")

    config_sub = Config(
        color_resolution=ColorResolution.RES_1080P,
        depth_mode=DepthMode.OFF,
        synchronized_images_only=False,
        wired_sync_mode=WiredSyncMode.SUBORDINATE,
    )
    config_mas = Config(
        color_resolution=ColorResolution.RES_1080P,
        depth_mode=DepthMode.OFF,
        synchronized_images_only=False,
        wired_sync_mode=WiredSyncMode.MASTER,
    )

    print("Initializing subordinate (device 1)...")
    k4a_sub = PyK4A(config_sub, device_id=0)
    k4a_sub.start()

    print("Initializing master (device 0)...")
    k4a_mas = PyK4A(config_mas, device_id=1)
    k4a_mas.start()

    k4a_mas.exposure = 20000
    k4a_sub.exposure = 20000

    k4a_mas.gain = 64
    k4a_sub.gain = 64

    #k4a_mas.whitebalance = 6500
    #k4a_sub.whitebalance = 6500

    print("Cameras ready.\n")

    count = 0
    win_mas = f"Master — cam0"
    win_sub = f"Subordinate — {sub_label}"
    cv2.namedWindow(win_mas, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_sub, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_mas, 960, 540)
    cv2.resizeWindow(win_sub, 960, 540)

    try:
        while True:
            cap_mas = k4a_mas.get_capture()
            cap_sub = k4a_sub.get_capture()

            if cap_mas.color is None or cap_sub.color is None:
                continue

            #frame0 = cap_mas.color[:, :, :3]
            #frame1 = cap_mas.color[:, :, :3] 
            frame0 = cv2.cvtColor(cap_mas.color, cv2.COLOR_BGRA2BGR)
            frame1 = cv2.cvtColor(cap_sub.color, cv2.COLOR_BGRA2BGR)

            gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            _, ch0, ids0, n0 = config.detect_charuco(gray0)
            _, ch1, ids1, n1 = config.detect_charuco(gray1)

            disp0 = config.draw_overlay(frame0, ch0, ids0, n0, "cam0")
            disp1 = config.draw_overlay(frame1, ch1, ids1, n1, sub_label)

            both_ok = ch0 is not None and ch1 is not None
            hint    = f"SPACE BAR to capture  |  Saved: {count}" if both_ok else f"Board not detected in both cameras  |  Saved: {count}"
            color   = (0, 220, 80) if both_ok else (0, 100, 255)

            for disp in [disp0, disp1]:
                h, w = disp.shape[:2]
                cv2.putText(disp, hint, (12, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow(win_mas, disp0)
            cv2.imshow(win_sub, disp1)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:   # ESC
                print("Exiting.")
                break

            if key == 32:   # SPACE BAR
                if both_ok:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    f0 = os.path.join(mas_dir, f"calib_{ts}.png")
                    f1 = os.path.join(sub_dir,  f"calib_{ts}.png")
                    cv2.imwrite(f0, frame0)
                    cv2.imwrite(f1, frame1)
                    count += 1
                    print(f"[{count:02d}] Saved  cam0:{n0} corners  {sub_label}:{n1} corners")
                else:
                    print("     ⚠  Board not detected in both cameras — try again")

    finally:
        k4a_mas.stop()
        k4a_sub.stop()
        cv2.destroyAllWindows()
        print(f"\nTotal pairs saved: {count}")
        if count < 15:
            print("Recommendation: capture at least 15 pairs per session.")
        else:
            print("Next: python Calibration/calib_stereo.py")

if __name__ == "__main__":
    main()
