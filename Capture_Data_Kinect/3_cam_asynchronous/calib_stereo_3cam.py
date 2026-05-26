"""
Calibración monocular (cam0, cam1, cam2) + estéreo (cam0-cam1, cam0-cam2)
y cálculo de la relación cam1-cam2 vía cam0.

Estructura esperada:
  3_cam_asynchronous/
    config.py
    calib_stereo_3cam.py
    Calibration/
      cam0_imgs/   # imágenes de sesión 1 y sesión 2 combinadas
      cam1_imgs/   # sesión 1
      cam2_imgs/   # sesión 2

Outputs:
  Calibration/cam0_intrinsics.json
  Calibration/cam1_intrinsics.json
  Calibration/cam2_intrinsics.json
  Calibration/stereo_cam0_cam1.json
  Calibration/stereo_cam0_cam2.json
  Calibration/stereo_cam1_cam2.json

Uso:
  python calib_stereo_3cam.py
"""

import cv2 as cv
import numpy as np
import glob
import json
import os
import sys
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "Calibration")


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def camera_json(K, dist, dict_id, img_size):
    return {
        "K": K.tolist(),
        "dist": dist.squeeze().tolist(),
        "dict": int(dict_id),
        "SX": config.SX,
        "SY": config.SY,
        "square_m": config.SQUARE,
        "marker_m": config.MARKER,
        "img_size": img_size,
    }


def get_stereo_points(paired_a, paired_b, adict, board, label_a="camA", label_b="camB"):
    """
    Para cada par sincronizado, detecta Charuco en ambas cámaras y conserva SOLO
    los corner IDs comunes. Esto evita errores tipo 18 object points vs 26 image points.
    """
    obj_pts_stereo = []
    img_pts_a_stereo = []
    img_pts_b_stereo = []

    chess_corners = board.getChessboardCorners()

    for fn_a, fn_b in zip(paired_a, paired_b):
        img_a = cv.imread(fn_a)
        img_b = cv.imread(fn_b)

        if img_a is None or img_b is None:
            continue

        gray_a = cv.cvtColor(img_a, cv.COLOR_BGR2GRAY)
        gray_b = cv.cvtColor(img_b, cv.COLOR_BGR2GRAY)

        corners_a, ids_a, _ = cv.aruco.detectMarkers(gray_a, adict)
        corners_b, ids_b, _ = cv.aruco.detectMarkers(gray_b, adict)

        if ids_a is None or ids_b is None:
            continue

        ret_a, ch_c_a, ch_ids_a = cv.aruco.interpolateCornersCharuco(
            corners_a, ids_a, gray_a, board
        )
        ret_b, ch_c_b, ch_ids_b = cv.aruco.interpolateCornersCharuco(
            corners_b, ids_b, gray_b, board
        )

        if (not ret_a) or (not ret_b) or ch_ids_a is None or ch_ids_b is None:
            continue

        ids_a_flat = ch_ids_a.flatten()
        ids_b_flat = ch_ids_b.flatten()

        common_ids = sorted(set(ids_a_flat) & set(ids_b_flat))
        if len(common_ids) < config.MIN_CORNERS:
            continue

        idx_a = [int(np.where(ids_a_flat == cid)[0][0]) for cid in common_ids]
        idx_b = [int(np.where(ids_b_flat == cid)[0][0]) for cid in common_ids]

        img_a_pts = ch_c_a[idx_a].astype(np.float32)
        img_b_pts = ch_c_b[idx_b].astype(np.float32)
        obj_pts = chess_corners[common_ids].reshape(-1, 1, 3).astype(np.float32)

        obj_pts_stereo.append(obj_pts)
        img_pts_a_stereo.append(img_a_pts)
        img_pts_b_stereo.append(img_b_pts)

        print(f"  {os.path.basename(fn_a)} -> common Charuco IDs: {len(common_ids)}")

    return obj_pts_stereo, img_pts_a_stereo, img_pts_b_stereo


def stereo_calibrate_pair(K_a, dist_a, valid_a, K_b, dist_b, valid_b, adict, board, img_size, label):
    """Stereo calibration using pairs matched by filename and common Charuco IDs."""
    print(f"\n── Stereo {label} ──")

    names_a = {os.path.basename(f): f for f in valid_a}
    names_b = {os.path.basename(f): f for f in valid_b}
    common = sorted(set(names_a.keys()) & set(names_b.keys()))

    print(f"Synchronized filename pairs: {len(common)}")
    if len(common) < 5:
        print(f"ERROR: Need at least 5 synchronized pairs for {label}, found {len(common)}")
        sys.exit(1)

    paired_a = [names_a[n] for n in common]
    paired_b = [names_b[n] for n in common]

    obj_pts, img_pts_a, img_pts_b = get_stereo_points(
        paired_a, paired_b, adict, board, label_a=label.split('-')[0], label_b=label.split('-')[1]
    )

    print(f"Valid stereo pairs after common-ID filtering: {len(obj_pts)}")
    if len(obj_pts) < 5:
        print(f"ERROR: Need at least 5 valid stereo pairs for {label}, found {len(obj_pts)}")
        sys.exit(1)

    rms, K_a_s, dist_a_s, K_b_s, dist_b_s, R, T, E, F = cv.stereoCalibrate(
        objectPoints=obj_pts,
        imagePoints1=img_pts_a,
        imagePoints2=img_pts_b,
        cameraMatrix1=K_a,
        distCoeffs1=dist_a,
        cameraMatrix2=K_b,
        distCoeffs2=dist_b,
        imageSize=img_size,
        flags=cv.CALIB_FIX_INTRINSIC,
    )

    print(f"Stereo RPE {label}: {rms:.4f} px")
    return R, T, E, F, rms


def has_images(img_dir):
    return os.path.isdir(img_dir) and len(glob.glob(os.path.join(img_dir, "*.png"))) > 0


def main():
    cam0_dir = os.path.join(CALIB_DIR, "cam0_imgs")
    cam1_dir = os.path.join(CALIB_DIR, "cam1_imgs")
    cam2_dir = os.path.join(CALIB_DIR, "cam2_imgs")

    if not has_images(cam0_dir):
        print(f"ERROR: No cam0 images found in {cam0_dir}")
        sys.exit(1)

    has_cam1 = has_images(cam1_dir)
    has_cam2 = has_images(cam2_dir)

    if not has_cam1 and not has_cam2:
        print("ERROR: No images found in cam1_imgs or cam2_imgs.")
        sys.exit(1)

    # cam0: master común para ambas sesiones
    K0, dist0, _, _, img_size0, dict0, valid0, board0, adict0 = config.calibrate_camera(cam0_dir, "cam0")
    save_json(os.path.join(CALIB_DIR, "cam0_intrinsics.json"), camera_json(K0, dist0, dict0, img_size0))

    R01 = T01 = None
    R02 = T02 = None

    # cam1 + stereo cam0-cam1
    if has_cam1:
        K1, dist1, _, _, img_size1, dict1, valid1, _, _ = config.calibrate_camera(cam1_dir, "cam1")
        save_json(os.path.join(CALIB_DIR, "cam1_intrinsics.json"), camera_json(K1, dist1, dict1, img_size1))

        if img_size1 != img_size0:
            print(f"ERROR: cam1 image size {img_size1} != cam0 image size {img_size0}")
            sys.exit(1)

        R01, T01, E01, F01, rms01 = stereo_calibrate_pair(
            K0, dist0, valid0, K1, dist1, valid1, adict0, board0, img_size0, "cam0-cam1"
        )
        save_json(os.path.join(CALIB_DIR, "stereo_cam0_cam1.json"), {
            "R": R01.tolist(),
            "T": T01.tolist(),
            "E": E01.tolist(),
            "F": F01.tolist(),
            "rms": rms01,
            "cam0_K": K0.tolist(),
            "cam0_dist": dist0.squeeze().tolist(),
            "cam1_K": K1.tolist(),
            "cam1_dist": dist1.squeeze().tolist(),
            "img_size": img_size0,
            "note": "Transform maps points from cam0 coordinates to cam1 coordinates as estimated by OpenCV stereoCalibrate."
        })

    # cam2 + stereo cam0-cam2
    if has_cam2:
        K2, dist2, _, _, img_size2, dict2, valid2, _, _ = config.calibrate_camera(cam2_dir, "cam2")
        save_json(os.path.join(CALIB_DIR, "cam2_intrinsics.json"), camera_json(K2, dist2, dict2, img_size2))

        if img_size2 != img_size0:
            print(f"ERROR: cam2 image size {img_size2} != cam0 image size {img_size0}")
            sys.exit(1)

        R02, T02, E02, F02, rms02 = stereo_calibrate_pair(
            K0, dist0, valid0, K2, dist2, valid2, adict0, board0, img_size0, "cam0-cam2"
        )
        save_json(os.path.join(CALIB_DIR, "stereo_cam0_cam2.json"), {
            "R": R02.tolist(),
            "T": T02.tolist(),
            "E": E02.tolist(),
            "F": F02.tolist(),
            "rms": rms02,
            "cam0_K": K0.tolist(),
            "cam0_dist": dist0.squeeze().tolist(),
            "cam2_K": K2.tolist(),
            "cam2_dist": dist2.squeeze().tolist(),
            "img_size": img_size0,
            "note": "Transform maps points from cam0 coordinates to cam2 coordinates as estimated by OpenCV stereoCalibrate."
        })

    # cam1-cam2 vía cam0
    if R01 is not None and T01 is not None and R02 is not None and T02 is not None:
        print("\n── Computing cam1-cam2 via cam0 ──")

        # OpenCV stereoCalibrate returns approximately:
        #   X_cam1 = R01 * X_cam0 + T01
        #   X_cam2 = R02 * X_cam0 + T02
        # Then:
        #   X_cam2 = R02 * R01.T * (X_cam1 - T01) + T02
        #          = R12 * X_cam1 + T12
        R12 = R02 @ R01.T
        T12 = T02 - R12 @ T01

        save_json(os.path.join(CALIB_DIR, "stereo_cam1_cam2.json"), {
            "R": R12.tolist(),
            "T": T12.tolist(),
            "note": "Derived transform mapping points from cam1 coordinates to cam2 coordinates via cam0. Not directly stereo-calibrated. Formula: R12 = R02 @ R01.T, T12 = T02 - R12 @ T01."
        })

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
