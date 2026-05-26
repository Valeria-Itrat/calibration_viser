import json
import os
import numpy as np
from pathlib import Path

ORIGINAL_DATA_DIR = Path(r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\3_cam_asynchronous\nerfstudio_dataset")
OUTPUT_DIR = Path(r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\3_cam_asynchronous\inria_dataset")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)

# 1. Copiar imágenes de forma limpia
import shutil
for cam_idx in range(3):
    src_img = ORIGINAL_DATA_DIR / "images" / f"cam{cam_idx}.png"
    dst_img = OUTPUT_DIR / "images" / f"cam{cam_idx}.png"
    if src_img.exists():
        shutil.copy(src_img, dst_img)

# 2. Cargar tu transforms.json original
with open(ORIGINAL_DATA_DIR / "transforms.json", "r") as f:
    orig_data = json.load(f)

w = orig_data["frames"][0]["w"]
fl_x = orig_data["frames"][0]["fl_x"]
camera_angle_x = 2 * np.arctan(w / (2 * fl_x))

inria_json = {
    "camera_angle_x": float(camera_angle_x),
    "frames": []
}

# Matriz estricta para pasar el sistema de coordenadas de OpenCV a OpenGL
cv_to_gl = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=np.float32)

for frame in orig_data["frames"]:
    c2w_opencv = np.array(frame["transform_matrix"], dtype=np.float32)
    
    # TRUCO GEOMÉTRICO: Aplicamos el cambio de base multiplicando por la izquierda y la derecha
    # Esto rota los ejes de la cámara Y de la traslación simultáneamente para que apunten al centro
    c2w_opengl = cv_to_gl @ c2w_opencv @ cv_to_gl
    
    # Asegurar que el elemento de escala de la matriz homogénea sea 1.0
    c2w_opengl[3, 3] = 1.0
    
    file_name = Path(frame["file_path"]).stem
    
    new_frame = {
        "file_path": f"./images/{file_name}",
        "rotation": 0.0,
        "transform_matrix": c2w_opengl.tolist()
    }
    inria_json["frames"].append(new_frame)

# Guardar los JSON correspondientes
with open(OUTPUT_DIR / "transforms_train.json", "w") as f:
    json.dump(inria_json, f, indent=2)
with open(OUTPUT_DIR / "transforms_test.json", "w") as f:
    json.dump(inria_json, f, indent=2)

# 3. CREAR UNA NUBE DE PUNTOS INICIAL EN EL ORIGEN (0,0,0)
# Esto actúa como un "imán" para que los Gaussians se queden en el centro del círculo
num_pts = 2000
# Generamos puntos aleatorios concentrados en el centro de la escena
xyz = np.random.normal(0, 0.1, size=(num_pts, 3))
rgb = np.ones((num_pts, 3)) * 128 # Color gris neutro

# Escribir un archivo PLY básico que el lector de Inria pueda absorber opcionalmente
ply_path = OUTPUT_DIR / "points3D.ply"
with open(ply_path, "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {num_pts}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for i in range(num_pts):
        f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")

print("🎯 Dataset re-estructurado con corrección cruzada de matrices y anclaje central.")
