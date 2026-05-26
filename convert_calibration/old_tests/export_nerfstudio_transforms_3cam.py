"""
export_nerfstudio_transforms_3cam.py

Generate a Nerfstudio-compatible transforms.json from a calibrated 3-camera Azure Kinect rig.

Expected calibration files:
  Calibration/cam0_intrinsics.json
  Calibration/cam1_intrinsics.json
  Calibration/cam2_intrinsics.json
  Calibration/stereo_cam0_cam1.json
  Calibration/stereo_cam0_cam2.json

Expected images:
  By default, this script tries to find one image per camera in:
    ObjectCapture/
    object_images/
    images/
    Calibration/object_images/

  Accepted filename patterns include:
    cam0.png, cam1.png, cam2.png
    cam0_*.png, cam1_*.png, cam2_*.png
    *_cam0.png, *_cam1.png, *_cam2.png
    .jpg/.jpeg also supported.

Usage:
  python export_nerfstudio_transforms_3cam.py

Optional:
  python export_nerfstudio_transforms_3cam.py --image-dir ObjectCapture --out-dir nerfstudio_dataset
  python export_nerfstudio_transforms_3cam.py --no-opencv-to-opengl

Output:
  nerfstudio_dataset/
    transforms.json
    images/
      cam0.png
      cam1.png
      cam2.png
    camera_debug.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def as_np(x, dtype=np.float64):
    return np.array(x, dtype=dtype)


def get_dist_coeffs(dist):
    """
    OpenCV commonly stores:
      [k1, k2, p1, p2, k3, k4, k5, k6, ...]
    Nerfstudio accepts common OPENCV coefficients.
    """
    d = np.array(dist, dtype=float).reshape(-1).tolist()
    out = {
        "k1": d[0] if len(d) > 0 else 0.0,
        "k2": d[1] if len(d) > 1 else 0.0,
        "p1": d[2] if len(d) > 2 else 0.0,
        "p2": d[3] if len(d) > 3 else 0.0,
    }
    if len(d) > 4:
        out["k3"] = d[4]
    if len(d) > 5:
        out["k4"] = d[5]
    return out


def intrinsics_to_frame_fields(intr: dict) -> dict:
    K = as_np(intr["K"])
    img_size = intr.get("img_size", None)
    if img_size is None:
        raise ValueError("Intrinsics JSON must include img_size")

    w, h = int(img_size[0]), int(img_size[1])

    fields = {
        "camera_model": "OPENCV",
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "w": w,
        "h": h,
    }
    fields.update(get_dist_coeffs(intr.get("dist", [])))
    return fields


def make_w2c_from_stereo(stereo: dict) -> np.ndarray:
    """
    OpenCV stereoCalibrate returns R, T such that:
      X_camB = R * X_camA + T

    If cam0 is the world/reference frame, then:
      W2C_cam0 = I
      W2C_cam1 = [R01 | T01]
      W2C_cam2 = [R02 | T02]
    """
    R = as_np(stereo["R"])
    T = as_np(stereo["T"]).reshape(3, 1)

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3:4] = T
    return w2c


def w2c_opencv_to_c2w_nerfstudio(w2c_opencv: np.ndarray, opencv_to_opengl: bool = True) -> np.ndarray:
    """
    Convert OpenCV W2C to Nerfstudio transform_matrix.

    Step 1:
      invert W2C -> C2W in OpenCV camera convention

    Step 2:
      convert camera axes from OpenCV convention:
        +X right, +Y down, +Z forward
      to OpenGL/Nerfstudio-style camera convention:
        +X right, +Y up, -Z forward / camera looks down -Z

      This is done by right-multiplying the C2W matrix by diag(1, -1, -1, 1),
      which flips the camera Y and Z axes.
    """
    c2w = np.linalg.inv(w2c_opencv)

    if opencv_to_opengl:
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        c2w = c2w @ flip

    # Clean tiny numerical noise.
    c2w[np.abs(c2w) < 1e-12] = 0.0
    return c2w


def find_image_for_cam(image_dir: Path, cam: str) -> Path:
    patterns = [
        f"{cam}.png", f"{cam}.jpg", f"{cam}.jpeg",
        f"{cam}_*.png", f"{cam}_*.jpg", f"{cam}_*.jpeg",
        f"*_{cam}.png", f"*_{cam}.jpg", f"*_{cam}.jpeg",
        f"*{cam}*.png", f"*{cam}*.jpg", f"*{cam}*.jpeg",
    ]

    matches = []
    for pat in patterns:
        matches.extend(sorted(image_dir.glob(pat)))

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for p in matches:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)

    if not unique:
        raise FileNotFoundError(
            f"Could not find image for {cam} in {image_dir}. "
            f"Rename it to {cam}.png or pass --cam{cam[-1]} explicitly."
        )

    # Use the newest matching image, useful if you captured multiple times.
    unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return unique[0]


def find_default_image_dir(root: Path) -> Path:
    candidates = [
        root / "ObjectCapture",
    ]

    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        try:
            find_image_for_cam(d, "cam0")
            find_image_for_cam(d, "cam1")
            find_image_for_cam(d, "cam2")
            return d
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "Could not auto-detect image directory. "
        "Use --image-dir or --cam0/--cam1/--cam2."
    )


def copy_image(src: Path, dst_dir: Path, out_name: str) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise ValueError(f"Unsupported image extension: {src}")

    dst = dst_dir / f"{out_name}{ext}"
    shutil.copy2(src, dst)

    # Nerfstudio paths are relative to the transforms.json file.
    return f"images/{dst.name}"


def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="Project root containing Calibration/")
    
    # CORREGIDO: Añadida la 'r' antes de las comillas para evitar problemas con '\3' y '\P'
    parser.add_argument("--calib-dir", type=str, 
                        default=r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\3_cam_asynchronous\Calibration", 
                        help="Calibration folder")
    parser.add_argument("--image-dir", type=str, 
                        default=r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\ObjectCapture", 
                        help="Folder containing object images")
                        
    parser.add_argument("--out-dir", type=str, default="nerfstudio_dataset", help="Output dataset folder")
    parser.add_argument("--cam0", type=str, default=None, help="Path to cam0 object image")
    parser.add_argument("--cam1", type=str, default=None, help="Path to cam1 object image")
    parser.add_argument("--cam2", type=str, default=None, help="Path to cam2 object image")
    parser.add_argument(
        "--no-opencv-to-opengl",
        action="store_true",
        help="Do not flip OpenCV camera convention to OpenGL/Nerfstudio convention",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    calib_dir = Path(args.calib_dir).resolve() if args.calib_dir else root / "Calibration"
    out_dir = Path(args.out_dir).resolve()
    out_images_dir = out_dir / "images"

    intr0 = load_json(calib_dir / "cam0_intrinsics.json")
    intr1 = load_json(calib_dir / "cam1_intrinsics.json")
    intr2 = load_json(calib_dir / "cam2_intrinsics.json")
    st01 = load_json(calib_dir / "stereo_cam0_cam1.json")
    st02 = load_json(calib_dir / "stereo_cam0_cam2.json")

    if args.image_dir:
        image_dir = Path(args.image_dir).resolve()
    else:
        image_dir = find_default_image_dir(root)

    cam_images = {
        "cam0": Path(args.cam0).resolve() if args.cam0 else find_image_for_cam(image_dir, "cam0"),
        "cam1": Path(args.cam1).resolve() if args.cam1 else find_image_for_cam(image_dir, "cam1"),
        "cam2": Path(args.cam2).resolve() if args.cam2 else find_image_for_cam(image_dir, "cam2"),
    }

    w2c = {
        "cam0": np.eye(4, dtype=np.float64),
        "cam1": make_w2c_from_stereo(st01),
        "cam2": make_w2c_from_stereo(st02),
    }

    opencv_to_opengl = not args.no_opencv_to_opengl
    c2w = {
        cam: w2c_opencv_to_c2w_nerfstudio(mat, opencv_to_opengl=opencv_to_opengl)
        for cam, mat in w2c.items()
    }

    intr = {"cam0": intr0, "cam1": intr1, "cam2": intr2}

    frames = []
    debug = {
        "notes": [
            "cam0 is treated as the world/reference frame.",
            "stereo_cam0_cam1 and stereo_cam0_cam2 are OpenCV W2C transforms relative to cam0.",
            "Nerfstudio transform_matrix is C2W.",
            "If opencv_to_opengl=true, camera Y and Z axes are flipped for Nerfstudio/OpenGL convention.",
        ],
        "opencv_to_opengl": opencv_to_opengl,
        "image_sources": {k: str(v) for k, v in cam_images.items()},
        "w2c_opencv": {k: v.tolist() for k, v in w2c.items()},
        "c2w_nerfstudio": {k: v.tolist() for k, v in c2w.items()},
    }

    for cam in ["cam0", "cam1", "cam2"]:
        file_path = copy_image(cam_images[cam], out_images_dir, cam)
        frame = {
            "file_path": file_path,
            "transform_matrix": c2w[cam].tolist(),
        }
        frame.update(intrinsics_to_frame_fields(intr[cam]))
        frames.append(frame)

    transforms = {
        "camera_model": "OPENCV",
        "frames": frames,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)

    with open(out_dir / "camera_debug.json", "w") as f:
        json.dump(debug, f, indent=2)

    print(f"Image directory used: {image_dir}")
    print("Images:")
    for cam, path in cam_images.items():
        print(f"  {cam}: {path}")

    print(f"\nSaved Nerfstudio dataset:")
    print(f"  {out_dir / 'transforms.json'}")
    print(f"  {out_dir / 'camera_debug.json'}")
    print(f"  {out_images_dir}")

    print("\nSuggested Nerfstudio command:")
    print(f"  ns-train splatfacto --data {out_dir}")
    print("\nIf the camera rig looks flipped/inverted, regenerate with:")
    print("  python export_nerfstudio_transforms_3cam.py --no-opencv-to-opengl")


if __name__ == "__main__":
    main()
