# config.py
import cv2 as cv
import sys
import os
import glob
from collections import Counter


# Charuco board parameters
SQUARE = 0.007      # meters
MARKER = 0.005      # meters
SX, SY = 8, 8       # cols, rows

DICT_CANDIDATES = [
    cv.aruco.DICT_4X4_50,
    cv.aruco.DICT_4X4_100,
    cv.aruco.DICT_4X4_250,
    cv.aruco.DICT_4X4_1000
]

MIN_MARKERS = 4
MIN_CORNERS = 6


# Helpers

def make_board(dict_id):
    adict = cv.aruco.getPredefinedDictionary(dict_id)
    board = cv.aruco.CharucoBoard((SX, SY), SQUARE, MARKER, adict)
    return adict, board

def detect_charuco(gray):
    """Returns (ch_corners, ch_ids, n_corners) or (None, None, 0)."""
    for d in DICT_CANDIDATES:
        adict, board = make_board(d)
        corners, ids, _ = cv.aruco.detectMarkers(gray, adict)
        if ids is None or len(ids) < MIN_MARKERS:
            continue
        ret, ch_corners, ch_ids = cv.aruco.interpolateCornersCharuco(
            corners, ids, gray, board)

        if ret and ch_ids is not None and len(ch_ids) >= MIN_CORNERS:
            return d, ch_corners, ch_ids, len(ch_ids)
        
    return None, None, None, 0 


def draw_overlay(frame, ch_corners, ch_ids, n_corners, label):
    overlay = frame.copy()

    if ch_corners is not None and ch_ids is not None and len(ch_corners) == len(ch_ids):
        cv.aruco.drawDetectedCornersCharuco(overlay, ch_corners, ch_ids)
        color  = (0, 220, 80)
        status = f"Board OK  ({n_corners} corners)"
        hint   = "SPACE BAR to capture"
    else:
        color  = (0, 100, 255)
        status = "Searching Board..."
        hint   = "Searching Board..."

    h, w = overlay.shape[:2]
    cv.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    frame_out = cv.addWeighted(overlay, 0.85, frame, 0.15, 0)

    cv.putText(frame_out, f"{label} — {status}", (12, 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv.putText(frame_out, hint, (12, 46),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return frame_out

def calibrate_camera(img_dir, label):
    """Monocular calibration. Returns K, dist, rvecs, tvecs, img_size, picked_dict, valid_files, board, adict."""
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not img_files:
        print(f"ERROR: No input images found in {img_dir}")
        sys.exit(1)

    print(f"\n{label}: {len(img_files)} input images")

    # First pass: find dominant dictionary
    all_charuco = []
    img_size    = None
    for fn in img_files:
        img = cv.imread(fn)
        if img_size is None:
            img_size = img.shape[1], img.shape[0]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        d, ch_c, ch_ids,_ = detect_charuco(gray)
        if d is not None:
            all_charuco.append((d, ch_c, ch_ids, fn))

    if not all_charuco:
        print(f"ERROR: No board detected in {img_dir}")
        sys.exit(1)

    picked = Counter([d for d, _, _, _ in all_charuco]).most_common(1)[0][0]
    adict, board = make_board(picked)
    print(f"Dictionary detected: {picked}  ({len(all_charuco)}/{len(img_files)} valid images)")

    # Second pass: collect corners with picked dictionary
    ch_corners_list = []
    ch_ids_list     = []
    valid_files     = []

    for fn in img_files:
        gray = cv.cvtColor(cv.imread(fn), cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, adict)
        if ids is None:
            continue
        ret, ch_c, ch_ids = cv.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if ret and ch_ids is not None and len(ch_ids) >= MIN_CORNERS:
            ch_corners_list.append(ch_c)
            ch_ids_list.append(ch_ids)
            valid_files.append(fn)

    print(f"Images used for calibration: {len(ch_corners_list)}")

    if len(ch_corners_list) == 0:
        print(f"ERROR: No valid frames after filtering in {img_dir}")
        sys.exit(1)

    flags = 0
    ret, K, dist, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(
        charucoCorners=ch_corners_list,
        charucoIds=ch_ids_list,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags)

    print(f"RPE {label}: {ret:.4f} px")
    return K, dist, rvecs, tvecs, img_size, picked, valid_files, board, adict
