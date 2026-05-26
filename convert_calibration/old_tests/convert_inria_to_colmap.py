import os
import json
import numpy as np
from pathlib import Path
import shutil

# Configuración de rutas
ORIGINAL_DATA_DIR = Path(r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\3_cam_asynchronous\nerfstudio_dataset")
OUTPUT_DIR = Path(r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\3_cam_asynchronous\inria_dataset")

# Limpiar estructura previa para evitar conflictos con el formato Blender
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# Crear la estructura exacta que espera el cargador COLMAP de Inria
SPARSE_DIR = OUTPUT_DIR / "sparse" / "0"
SPARSE_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

# 1. Copiar imágenes reales de forma limpia
for cam_idx in range(3):
    src_img = ORIGINAL_DATA_DIR / "images" / f"cam{cam_idx}.png"
    dst_img = OUTPUT_DIR / "images" / f"cam{cam_idx}.png"
    if src_img.exists():
        shutil.copy(src_img, dst_img)

# 2. Generar el archivo 'cameras.txt' con los intrínsecos de tus cámaras OpenCV
# Formato COLMAP PINHOLE: CAMERA_ID MODEL WIDTH HEIGHT f_x f_y c_x c_y
with open(SPARSE_DIR / "cameras.txt", "w") as f:
    f.write("# Camera list with generic pinhole parameters\n")
    # Cam 0
    f.write("1 PINHOLE 1920 1080 912.809627 916.190136 943.291534 561.102244\n")
    # Cam 1
    f.write("2 PINHOLE 1920 1080 902.821555 903.015729 897.398056 570.535069\n")
    # Cam 2
    f.write("3 PINHOLE 1920 1080 914.482927 909.293153 993.425561 553.751929\n")

# 3. Generar el archivo 'images.txt' aplicando las matrices w2c (Mundo a Cámara) de OpenCV
# COLMAP requiere Quaternions (qw, qx, qy, qz) y Traslación (tx, ty, tz) de la transformación mundo-a-cámara.
# Cambiamos los signos de los ejes Y y Z para adaptar OpenCV al formato de visualización interna de Inria.

def matrix_to_quaternion(R):
    t = np.trace(R)
    if t > 0:
        M = 0.5 / np.sqrt(t + 1.0)
        return [0.25 / M, (R[2,1] - R[1,2]) * M, (R[0,2] - R[2,0]) * M, (R[1,0] - R[0,1]) * M]
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            M = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            return [(R[2,1] - R[1,2]) / M, 0.25 * M, (R[0,1] + R[1,0]) / M, (R[0,2] + R[2,0]) / M]
        elif R[1,1] > R[2,2]:
            M = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            return [(R[0,2] - R[2,0]) / M, (R[0,1] + R[1,0]) / M, 0.25 * M, (R[1,2] + R[2,1]) / M]
        else:
            M = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            return [(R[1,0] - R[0,1]) / M, (R[0,2] + R[2,0]) / M, (R[1,2] + R[2,1]) / M, 0.25 * M]

# Extraemos las rotaciones y traslaciones base directamente de tus archivos estéreo unificados
# Matriz base identidad para Cam0 (Origen)
R0 = np.eye(3); T0 = np.zeros(3)

# Datos extraídos de tus JSON estéreo de OpenCV
R1 = np.array([[0.38702399, 0.42132225, -0.82018290], [-0.55959623, 0.81428735, 0.15423415], [0.73284684, 0.39927894, 0.55091908]])
T1 = np.array([0.10025764, -0.04622760, 0.14326097])

R2 = np.array([[0.23365888, -0.34292098, 0.90983995], [0.62183194, 0.77206582, 0.13129894], [-0.74748149, 0.53508837, 0.39363925]])
T2 = np.array([-0.15844144, -0.04409940, 0.04726844])

cams_data = [(1, R0, T0, "cam0.png"), (2, R1, T1, "cam1.png"), (3, R2, T2, "cam2.png")]

with open(SPARSE_DIR / "images.txt", "w") as f:
    f.write("# Image list with rigid transformations\n")
    for cam_id, R, T, img_name in cams_data:
        # Cambio estricto de signo en los ejes Y y Z para forzar la convergencia de los rayos al centro
        R_mod = R.copy()
        R_mod[1, :] *= -1  # Inversión de eje Y
        R_mod[2, :] *= -1  # Inversión de eje Z
        T_mod = T.copy()
        T_mod[1] *= -1
        T_mod[2] *= -1
        
        q = matrix_to_quaternion(R_mod)
        # Formato COLMAP: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        f.write(f"{cam_id} {q[0]} {q[1]} {q[2]} {q[3]} {T_mod[0]} {T_mod[1]} {T_mod[2]} {cam_id} {img_name}\n\n")

# 4. Crear una nube de puntos central que sirva como anclaje
# Generamos 5,000 puntos concentrados esféricamente en el centro exacto de la escena (0,0,0)
num_pts = 5000
with open(SPARSE_DIR / "points3D.txt", "w") as f:
    f.write("# 3D point list acting as anchor\n")
    for i in range(num_pts):
        xyz = np.random.normal(0, 0.2, size=3)
        # Formato COLMAP: POINT3D_ID X Y Z R G B ERROR TRACK_LIST[...]
        f.write(f"{i+1} {xyz[0]} {xyz[1]} {xyz[2]} 128 128 128 0.0 {i+1} 1\n")

print("🎉 ¡Estructura COLMAP artificial creada de manera exitosa!")
print("Las cámaras ahora apuntarán estrictamente convergentes hacia el centro del volumen.")