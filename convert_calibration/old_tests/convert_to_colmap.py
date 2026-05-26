import os
import json
import numpy as np
from pathlib import Path
import shutil
import cv2

BASE_DIR = Path(r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\3_cam_asynchronous\folder")
OUTPUT_DIR = BASE_DIR / "inria_dataset_pure_opencv"

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

SPARSE_DIR = OUTPUT_DIR / "sparse" / "0"
SPARSE_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

# 1. CARGAR DATOS ORIGINALES
with open(BASE_DIR / "cam0_intrinsics.json", "r") as f: cam0_data = json.load(f)
with open(BASE_DIR / "cam1_intrinsics.json", "r") as f: cam1_data = json.load(f)
with open(BASE_DIR / "cam2_intrinsics.json", "r") as f: cam2_data = json.load(f)

with open(BASE_DIR / "stereo_cam0_cam1.json", "r") as f: stereo_01 = json.load(f)
with open(BASE_DIR / "stereo_cam0_cam2.json", "r") as f: stereo_02 = json.load(f)

# 2. ESCRIBIR CAMERAS.TXT (Sin distorsión para el rasterizador estable)
with open(SPARSE_DIR / "cameras.txt", "w") as f:
    f.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n")
    f.write(f"1 PINHOLE 1920 1080 {cam0_data['K'][0][0]} {cam0_data['K'][1][1]} {cam0_data['K'][0][2]} {cam0_data['K'][1][2]}\n")
    f.write(f"2 PINHOLE 1920 1080 {cam1_data['K'][0][0]} {cam1_data['K'][1][1]} {cam1_data['K'][0][2]} {cam1_data['K'][1][2]}\n")
    f.write(f"3 PINHOLE 1920 1080 {cam2_data['K'][0][0]} {cam2_data['K'][1][1]} {cam2_data['K'][0][2]} {cam2_data['K'][1][2]}\n")

def rotation_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        return [0.25 * S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S]
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            return [(R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S]
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            return [(R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S]
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            return [(R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S]

# Matrices OpenCV extrínsecas crudas (Mundo a Cámara)
cams_raw = [
    {"id": 1, "R": np.eye(3), "T": np.zeros((3,1)), "name": "cam0.png"},
    {"id": 2, "R": np.array(stereo_01["R"]), "T": np.array(stereo_01["T"]), "name": "cam1.png"},
    {"id": 3, "R": np.array(stereo_02["R"]), "T": np.array(stereo_02["T"]), "name": "cam2.png"}
]

# Matriz P de cambio de base estricta OpenCV -> OpenGL (COLMAP)
# Invierte eje Y (arriba/abajo) y eje Z (adelante/atrás) de forma matricial limpia
P = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# FACTOR DE ESCALA: Mantiene las cámaras separadas en el espacio COLMAP
SCALE_FACTOR = 10.0

with open(SPARSE_DIR / "images.txt", "w") as f:
    for cam in cams_raw:
        R_orig = cam["R"]
        T_orig = cam["T"]
        
        # 1. Obtener la posición de la cámara en el mundo real (metros)
        C_world = -R_orig.T @ T_orig
        # Escalar la posición para darle volumen tridimensional
        C_world_scaled = C_world * SCALE_FACTOR
        
        # 2. Reconstruir la matriz Extrínseca OpenCV con la nueva escala
        R_scaled = R_orig
        T_scaled = -R_orig @ C_world_scaled
        
        # 3. Aplicar el cambio de base a COLMAP usando multiplicación matricial pura
        R_colmap = P @ R_scaled
        T_colmap = P @ T_scaled
        
        q = rotation_to_quat(R_colmap)
        f.write(f"{cam['id']} {q[0]} {q[1]} {q[2]} {q[3]} {T_colmap[0,0]} {T_colmap[1,0]} {T_colmap[2,0]} {cam['id']} {cam['name']}\n\n")

# 3. COPIAR IMÁGENES
img_paths = {}
for cam in cams_raw:
    for ext in ["png", "jpg"]:
        src_img = BASE_DIR / "images" / f"cam{cam['id']-1}.{ext}"
        if not src_img.exists(): src_img = BASE_DIR / f"cam{cam['id']-1}.{ext}"
        if src_img.exists():
            shutil.copy(src_img, OUTPUT_DIR / "images" / f"cam{cam['id']-1}.png")
            img_paths[cam['id']] = OUTPUT_DIR / "images" / f"cam{cam['id']-1}.png"
            break

# 4. GENERAR NUBE DE PUNTOS INICIAL BASADA EN PIXELES REALES
# Como el fondo negro es físico, leeremos los píxeles brillantes de cam0
img0 = cv2.imread(str(img_paths[1]))
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# Detectar el juguete (píxeles con valor mayor a 15)
_, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
y_indices, x_indices = np.where(mask > 0)

max_pts = 16000
if len(x_indices) > max_pts:
    idx = np.random.choice(len(x_indices), max_pts, replace=False)
    x_indices, y_indices = x_indices[idx], y_indices[idx]

fx, fy = cam0_data['K'][0][0], cam0_data['K'][1][1]
cx, cy = cam0_data['K'][0][2], cam0_data['K'][1][2]

with open(SPARSE_DIR / "points3D.txt", "w") as f:
    for i in range(len(x_indices)):
        u, v = x_indices[i], y_indices[i]
        b, g, r = img0[v, u]
        
        # Retroproyectar en el eje óptico correcto de Cam0
        z_base = np.random.uniform(1.2, 1.5) 
        x = (u - cx) * z_base / fx
        y = (v - cy) * z_base / fy
        z = z_base
        
        # Indicar visibilidad en las 3 cámaras para ligar gradientes
        f.write(f"{i+1} {x} {y} {z} {r} {g} {b} 0.0 1 1 2 1 3 1\n")

print("💥 Dataset COLMAP con corrección matricial de ejes completado.")