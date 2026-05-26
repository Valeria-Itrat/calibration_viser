from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tyro
import viser
from PIL import Image

from example_viser import load_ply_file, load_splat_file


DEFAULT_SPLAT = Path(r"C:\Projects\Itrat_Valeria_Gaussian\point_cloud.ply")
DEFAULT_DATASET = Path(r"C:\Projects\Itrat_Valeria_Gaussian\kinect_3cam_dataset")


class CameraPose(NamedTuple):
    name: str
    c2w: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


def rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array(
        [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    )


def load_image(path: Path, max_width: int = 640) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.width > max_width:
        scale = max_width / image.width
        image = image.resize((max_width, int(image.height * scale)), Image.Resampling.LANCZOS)
    return np.asarray(image)


def fovx_to_fovy(fovx: float, aspect: float) -> float:
    return 2.0 * math.atan(math.tan(fovx / 2.0) / aspect)


def infer_model_dir(splat_path: Path) -> Path | None:
    parts = splat_path.resolve().parts
    for idx, part in enumerate(parts):
        if part == "point_cloud" and idx > 0:
            return Path(*parts[:idx])
    return None


def load_colmap_intrinsics(dataset_dir: Path) -> dict[str, dict[str, float]]:
    cameras_txt = dataset_dir / "sparse" / "0" / "cameras.txt"
    images_txt = dataset_dir / "sparse" / "0" / "images.txt"

    camera_by_id = {}
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        elems = line.split()
        camera_id = int(elems[0])
        width, height = int(elems[2]), int(elems[3])
        fx, fy, cx, cy = map(float, elems[4:8])
        camera_by_id[camera_id] = {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

    intrinsics = {}
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        elems = line.split()
        if len(elems) < 10:
            continue
        camera_id = int(elems[8])
        intrinsics[elems[9]] = camera_by_id[camera_id]
    return intrinsics


def load_dataset_camera_poses(dataset_dir: Path) -> list[CameraPose]:
    intrinsics = load_colmap_intrinsics(dataset_dir)
    images_txt = dataset_dir / "sparse" / "0" / "images.txt"
    poses = []
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        elems = line.split()
        if len(elems) < 10:
            continue
        qvec = np.array(tuple(map(float, elems[1:5])), dtype=np.float64)
        tvec = np.array(tuple(map(float, elems[5:8])), dtype=np.float64)
        image_name = elems[9]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec_to_rotmat(qvec)
        w2c[:3, 3] = tvec
        c2w = np.linalg.inv(w2c)
        intr = intrinsics[image_name]
        poses.append(CameraPose(image_name, c2w, **intr))
    return poses


def load_model_camera_poses(model_dir: Path, dataset_dir: Path) -> list[CameraPose] | None:
    cameras_json = model_dir / "cameras.json"
    if not cameras_json.exists():
        return None

    intrinsics = load_colmap_intrinsics(dataset_dir)
    poses = []
    for entry in json.loads(cameras_json.read_text(encoding="utf-8")):
        name = entry["img_name"]
        intr = intrinsics.get(name)
        if intr is None:
            continue
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.array(entry["rotation"], dtype=np.float64)
        c2w[:3, 3] = np.array(entry["position"], dtype=np.float64)
        poses.append(CameraPose(name, c2w, **intr))
    return poses or None


def nearest_point_on_ray(
    points: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    max_points: int = 250_000,
) -> tuple[int, np.ndarray, float]:
    if len(points) > max_points:
        sample_idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        candidates = points[sample_idx]
    else:
        sample_idx = None
        candidates = points

    direction = ray_direction / np.linalg.norm(ray_direction)
    rel = candidates - ray_origin[None, :]
    depth = rel @ direction
    valid = depth > 0.0
    if not np.any(valid):
        valid = np.ones(len(candidates), dtype=bool)
    closest = ray_origin[None, :] + depth[:, None] * direction[None, :]
    dist2 = np.sum((candidates - closest) ** 2, axis=1)
    dist2[~valid] = np.inf
    local_idx = int(np.argmin(dist2))
    global_idx = int(sample_idx[local_idx]) if sample_idx is not None else local_idx
    return global_idx, points[global_idx], float(math.sqrt(dist2[local_idx]))


def project_point(point: np.ndarray, camera: CameraPose) -> tuple[float, float, float, bool]:
    w2c = np.linalg.inv(camera.c2w)
    point_cam = w2c[:3, :3] @ point + w2c[:3, 3]
    z = float(point_cam[2])
    if z <= 1e-8:
        return math.nan, math.nan, z, False
    u = camera.fx * float(point_cam[0]) / z + camera.cx
    v = camera.fy * float(point_cam[1]) / z + camera.cy
    inside = 0.0 <= u < camera.width and 0.0 <= v < camera.height
    return u, v, z, inside


def main(
    splat_path: Path = DEFAULT_SPLAT,
    dataset_dir: Path = DEFAULT_DATASET,
    transforms_name: str = "transforms_train.json",
    center_splat: bool = False,
    port: int = 8080,
) -> None:
    server = viser.ViserServer(port=port)
    server.scene.world_axes.visible = False

    server.scene.add_frame(
        "/origin",
        axes_length=0.15,
        axes_radius=0.002,
    )

    info = server.gui.add_markdown(
        "Click on imagen/frustum to get information."
    )

    if splat_path.exists():
        if splat_path.suffix.lower() == ".ply":
            splat = load_ply_file(splat_path, center=center_splat)
        elif splat_path.suffix.lower() == ".splat":
            splat = load_splat_file(splat_path, center=center_splat)
        else:
            raise ValueError("splat_path must be .ply or .splat")

        server.scene.add_gaussian_splats(
            "/splat",
            centers=splat["centers"],
            rgbs=np.clip(splat["rgbs"], 0.0, 1.0),
            opacities=np.clip(splat["opacities"], 0.0, 1.0),
            covariances=splat["covariances"],
        )
    else:
        info.content = f"No encontré el splat: `{splat_path}`. Mostrando solo cámaras."

    transforms_path = dataset_dir / transforms_name
    if not transforms_path.exists():
        raise FileNotFoundError(
            f"No encontré {transforms_path}. Genera primero el dataset con build_kinect_3cam_dataset.py."
        )

    meta = json.loads(transforms_path.read_text(encoding="utf-8"))
    fovx = float(meta["camera_angle_x"])
    model_dir = infer_model_dir(splat_path)
    camera_poses = (
        load_model_camera_poses(model_dir, dataset_dir)
        if model_dir is not None
        else None
    )
    if camera_poses is None:
        camera_poses = load_dataset_camera_poses(dataset_dir)

    marker_handle = None

    for idx, camera in enumerate(camera_poses):
        image_stem = Path(camera.name).stem
        image_path = dataset_dir / "images" / camera.name
        if not image_path.exists():
            matches = sorted((dataset_dir / "images").glob(f"{image_stem}.*"))
            if not matches:
                raise FileNotFoundError(f"No image found for {camera.name}")
            image_path = matches[0]

        image = load_image(image_path)
        aspect = image.shape[1] / image.shape[0]
        fovy = fovx_to_fovy(fovx, aspect)
        position = camera.c2w[:3, 3]
        wxyz = rotmat_to_wxyz(camera.c2w[:3, :3])
        wxyz /= np.linalg.norm(wxyz)

        server.scene.add_camera_frustum(
            f"/cameras/{image_stem}",
            fov=fovy,
            aspect=aspect,
            scale=0.05,
            line_width=1.0,
            image=image,
            wxyz=wxyz,
            position=position,
            color=((255, 80, 80), (80, 180, 255), (90, 220, 140))[idx],
        )

    @server.scene.on_click()
    def _on_scene_click(event) -> None:
        nonlocal marker_handle
        if not splat_path.exists():
            return

        ray_origin = np.array(event.ray_origin, dtype=np.float64)
        ray_direction = np.array(event.ray_direction, dtype=np.float64)
        point_idx, point, ray_distance = nearest_point_on_ray(
            splat["centers"], ray_origin, ray_direction
        )

        if marker_handle is not None:
            marker_handle.remove()
        marker_handle = server.scene.add_icosphere(
            "/selected_point",
            radius=0.01,
            color=(255, 255, 0),
            position=point,
        )

        lines = [
            f"**Selected point:** gaussian `{point_idx}`",
            f"**XYZ:** `[{point[0]:.5f}, {point[1]:.5f}, {point[2]:.5f}]`",
            f"**Distance to ray:** `{ray_distance:.5f}`",
            "",
            "**Camera projections:**",
        ]
        for camera in camera_poses:
            u, v, z, inside = project_point(point, camera)
            if z <= 1e-8:
                status = "behind the camera"
                coord = "no pixel"
            else:
                status = "inside" if inside else "outside"
                coord = f"pixel `({u:.1f}, {v:.1f})`, z `{z:.4f}`"
            lines.append(f"- `{camera.name}`: {status}, {coord}")
        info.content = "  \n".join(lines)
        print(f"Click splat point {point_idx}: {point}")

    print(f"Viser abierto en http://localhost:{port}")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
