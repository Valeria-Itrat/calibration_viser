from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


DEFAULT_CALIB_DIR = Path(
    r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\3_cam_asynchronous\Calibration"
)
DEFAULT_IMAGE_DIR = Path(
    r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\ObjectCapture"
)
DEFAULT_OUT_DIR = Path(r"C:\Projects\Itrat_Valeria_Gaussian\kinect_3cam_dataset")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def as_matrix4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t.reshape(3)
    return out


def opencv_w2c_to_blender_c2w(w2c: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(w2c)
    # Blender/NeRF transforms use an OpenGL camera: +Y up, camera looks along -Z.
    return c2w @ np.diag([1.0, -1.0, -1.0, 1.0])


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    # COLMAP image text format wants qw qx qy qz for world-to-camera rotation.
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array(
        [
            [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ],
        dtype=np.float64,
    )
    K /= 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def fov_x_from_intrinsics(K: np.ndarray, width: int) -> float:
    return 2.0 * math.atan(width / (2.0 * K[0, 0]))


def find_image(image_dir: Path, cam_name: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{cam_name}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(image_dir.glob(f"*{cam_name}*"))
    matches = [p for p in matches if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not matches:
        raise FileNotFoundError(f"No image found for {cam_name} in {image_dir}")
    return matches[0]


def make_seed_points(path: Path, count: int, radius: float, z_center: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(loc=[0.0, 0.0, z_center], scale=[radius, radius, radius], size=(count, 3))
    colors = np.full((count, 3), 128, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {count}, mean track length: 0\n")
        for idx, (p, c) in enumerate(zip(xyz, colors), start=1):
            f.write(
                f"{idx} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} 1.0\n"
            )
    ply_path = path.with_suffix(".ply")
    with ply_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, colors):
            f.write(
                f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f} "
                f"0.0 0.0 0.0 {int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Blender transforms and COLMAP text files from 3-camera OpenCV calibration."
    )
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scale", type=float, default=1.0, help="Scale camera translations/seed points.")
    parser.add_argument("--seed-points", type=int, default=2000)
    parser.add_argument("--seed-radius", type=float, default=0.08)
    parser.add_argument("--seed-z", type=float, default=0.35)
    parser.add_argument("--random-seed", type=int, default=7)
    args = parser.parse_args()

    calib_dir = args.calib_dir.resolve()
    image_dir = args.image_dir.resolve()
    out_dir = args.out_dir.resolve()

    intrinsics = {
        f"cam{i}": load_json(calib_dir / f"cam{i}_intrinsics.json")
        for i in range(3)
    }
    st01 = load_json(calib_dir / "stereo_cam0_cam1.json")
    st02 = load_json(calib_dir / "stereo_cam0_cam2.json")

    w2c = {
        "cam0": np.eye(4, dtype=np.float64),
        "cam1": as_matrix4(np.array(st01["R"], dtype=np.float64), np.array(st01["T"], dtype=np.float64) * args.scale),
        "cam2": as_matrix4(np.array(st02["R"], dtype=np.float64), np.array(st02["T"], dtype=np.float64) * args.scale),
    }
    blender_c2w = {name: opencv_w2c_to_blender_c2w(mat) for name, mat in w2c.items()}

    images_out = out_dir / "images"
    train_out = out_dir / "train"
    sparse_out = out_dir / "sparse" / "0"
    for folder in (images_out, train_out, sparse_out):
        folder.mkdir(parents=True, exist_ok=True)

    image_paths: dict[str, str] = {}
    for cam in ("cam0", "cam1", "cam2"):
        src = find_image(image_dir, cam)
        dst_name = f"{cam}{src.suffix.lower()}"
        shutil.copy2(src, images_out / dst_name)
        shutil.copy2(src, train_out / dst_name)
        image_paths[cam] = dst_name

    first_K = np.array(intrinsics["cam0"]["K"], dtype=np.float64)
    width, height = intrinsics["cam0"]["img_size"]
    transforms = {
        "camera_angle_x": fov_x_from_intrinsics(first_K, int(width)),
        "frames": [],
    }
    for cam in ("cam0", "cam1", "cam2"):
        transforms["frames"].append(
            {
                "file_path": f"./train/{Path(image_paths[cam]).stem}",
                "rotation": 0.0,
                "transform_matrix": blender_c2w[cam].tolist(),
            }
        )
    write_json(out_dir / "transforms_train.json", transforms)
    write_json(out_dir / "transforms_test.json", transforms)

    with (sparse_out / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for idx, cam in enumerate(("cam0", "cam1", "cam2"), start=1):
            K = np.array(intrinsics[cam]["K"], dtype=np.float64)
            w, h = intrinsics[cam]["img_size"]
            f.write(
                f"{idx} PINHOLE {int(w)} {int(h)} "
                f"{K[0, 0]:.12g} {K[1, 1]:.12g} {K[0, 2]:.12g} {K[1, 2]:.12g}\n"
            )

    with (sparse_out / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, cam in enumerate(("cam0", "cam1", "cam2"), start=1):
            R = w2c[cam][:3, :3]
            t = w2c[cam][:3, 3]
            q = rotmat_to_qvec(R)
            f.write(
                f"{idx} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {idx} {image_paths[cam]}\n\n"
            )

    make_seed_points(
        sparse_out / "points3D.txt",
        count=args.seed_points,
        radius=args.seed_radius * args.scale,
        z_center=args.seed_z * args.scale,
        seed=args.random_seed,
    )

    debug = {
        "calib_dir": str(calib_dir),
        "image_dir": str(image_dir),
        "scale": args.scale,
        "notes": [
            "COLMAP files keep OpenCV/COLMAP world-to-camera convention; no OpenGL axis flip is applied there.",
            "Blender transforms invert W2C to C2W and then flip camera Y/Z axes for OpenGL convention.",
            "cam0 is the world frame because stereo JSON maps cam0 coordinates into cam1/cam2.",
        ],
        "w2c_opencv_colmap": {k: v.tolist() for k, v in w2c.items()},
        "c2w_blender": {k: v.tolist() for k, v in blender_c2w.items()},
    }
    write_json(out_dir / "camera_debug.json", debug)

    print(f"Dataset written to: {out_dir}")
    print(f"COLMAP/INRIA path: {out_dir}")
    print(f"Blender transforms: {out_dir / 'transforms_train.json'}")


if __name__ == "__main__":
    main()
