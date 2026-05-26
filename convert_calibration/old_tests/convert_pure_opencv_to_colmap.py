import os
import json
import numpy as np
from pathlib import Path
import shutil

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
# Pon aquí la carpeta donde tienes guardados tus archivos JSON originales de OpenCV y tus imágenes
BASE_DIR = Path(r"C:\Projects\Itrat_Valeria_Gaussian\Capture_Data_Kinect\3_cam_asynchronous\folder")

# Directorio de salida limpio para el formato Inria/COLMAP
OUTPUT_DIR = BASE_DIR / "inria_dataset_pure_opencv"

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

SPARSE_DIR = OUTPUT_DIR / "sparse" / "0"
SPARSE_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

# ==========================================
# CONFIGURACIÓN GEOMÉTRICA
# ==========================================
# Factor multiplicador para expandir el espacio de metros reales de laboratorio 
# a un volumen donde el optimizador de Gaussians pueda converger holgadamente.
SCALE_FACTOR = 20.0 

# Cargar Intrínsecos desde tus archivos originales
with open(BASE_DIR / "cam0_intrinsics.json", "r") as f:
    cam0_data = json.load(f)
with open(BASE_DIR / "cam1_intrinsics.json", "r") as f:
    cam1_data = json.load(f)
with open(BASE_DIR / "cam2_intrinsics.json", "r") as f:
    cam2_data = json.load(f)

# Cargar Extrínsecos Estéreo desde tus archivos originales
with open(BASE_DIR / "stereo_cam0_cam1.json", "r") as f:
    stereo_01 = json.load(f)
with open(BASE_DIR / "stereo_cam0_cam2.json", "r") as f:
    stereo_02 = json.load(f)

# 1. ESCRIBIR CAMERAS.TXT (Nativos PINHOLE planos sin distorsión para evitar artefactos en Inria)
# Estructura: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
with open(SPARSE_DIR / "cameras.txt", "w") as f:
    f.write("# Camera list: CAMERA_ID MODEL WIDTH HEIGHT f_x f_y c_x c_y\n")
    f.write(f"1 PINHOLE 1920 1080 {cam0_data['K'][0][0]} {cam0_data['K'][1][1]} {cam0_data['K'][0][2]} {cam0_data['K'][1][2]}\n")
    f.write(f"2 PINHOLE 1920 1080 {cam1_data['K'][0][0]} {cam1_data['K'][1][1]} {cam1_data['K'][0][2]} {cam1_data['K'][1][2]}\n")
    f.write(f"3 PINHOLE 1920 1080 {cam2_data['K'][0][0]} {cam2_data['K'][1][1]} {cam2_data['K'][0][2]} {cam2_data['K'][1][2]}\n")

# Función auxiliar matemática para convertir matriz de rotación a Cuaternión (Formatos COLMAP)
def rotation_matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return [qw, qx, qy, qz]

# EXTRAER MATRICES CRUDAS DE OPENCV
# Cam0 es nuestro centro de coordenadas original
R0 = np.eye(3)
T0 = np.zeros((3, 1))

# Cam1 respecto a Cam0
R1 = np.array(stereo_01["R"])
T1 = np.array(stereo_01["T"])

# Cam2 respecto a Cam0
R2 = np.array(stereo_02["R"])
T2 = np.array(stereo_02["T"])

cameras_setup = [
    {"id": 1, "R": R0, "T": T0, "name": "cam0.png"},
    {"id": 2, "R": R1, "T": T1, "name": "cam1.png"},
    {"id": 3, "R": R2, "T": T2, "name": "cam2.png"}
]

# 2. ESCRIBIR IMAGES.TXT APLICANDO RE-ESCALADO TRIDIMENSIONAL
with open(SPARSE_DIR / "images.txt", "w") as f:
    for cam in cameras_setup:
        R_opencv = cam["R"]
        T_opencv = cam["T"]
        
        # Inversa estricta para obtener la posición de la cámara en coordenadas del Mundo (Cam0-centric)
        camera_center_world = -R_opencv.T @ T_opencv
        
        # Aplicamos el escalado físico al vector de posición en el mundo
        camera_center_world_scaled = camera_center_world * SCALE_FACTOR
        
        # Volvemos a proyectar a matriz extrínseca de mundo a cámara reconstruida
        R_scaled = R_opencv
        T_scaled = -R_opencv @ camera_center_world_scaled
        
        # Cambio de convención de ejes: Convertir el sistema OpenCV a OpenGL/COLMAP
        # Multiplicamos las filas correspondientes a Y y Z por -1 para invertir los vectores ópticos
        R_colmap = R_scaled.copy()
        R_colmap[1, :] *= -1
        R_colmap[2, :] *= -1
        
        T_colmap = T_scaled.copy()
        T_colmap[1] *= -1
        T_colmap[2] *= -1
        
        # Obtener cuaternión de la matriz adaptada
        q = rotation_matrix_to_quaternion(R_colmap)
        
        # Escribir fila de la imagen
        f.write(f"{cam['id']} {q[0]} {q[1]} {q[2]} {q[3]} {T_colmap[0][0]} {T_colmap[1][0]} {T_colmap[2][0]} {cam['id']} {cam['name']}\n\n")

# 3. CREAR NUBE DE PUNTOS INICIAL SINTÉTICA REPOSITORIO (points3D.txt) - OPTIMIZADA PARA FONDO NEGRO
# En lugar de un cono gigante, creamos una esfera densa en la zona focal estimada del objeto
num_points = 10000
with open(SPARSE_DIR / "points3D.txt", "w") as f:
    for i in range(num_points):
        # Generar una bola densa de puntos en el centro del espacio de las cámaras
        x = np.random.normal(0.0, 0.2)
        y = np.random.normal(0.0, 0.2)
        z = np.random.normal(1.2, 0.2) # Ajustado a la distancia focal promedio de una Kinect en metros (~1.2m)
        
        # Color gris oscuro/neutro para que no choque fuertemente con el fondo negro
        r, g, b = 64, 64, 64
        f.write(f"{i+1} {x} {y} {z} {r} {g} {b} 0.0 {i+1} 1\n")

# 4. COPIAR IMÁGENES ORIGINALES (Asegúrate de tener tus imágenes en la carpeta esperada)
for cam in cameras_setup:
    # Buscar tanto extensiones png como jpg para evitar fallos de copia
    for ext in ["png", "jpg"]:
        src_img = BASE_DIR / "images" / f"cam{cam['id']-1}.{ext}"
        if not src_img.exists() and ext == "png": # Probar en la raíz si no hay carpeta images
            src_img = BASE_DIR / f"cam{cam['id']-1}.png"
        if not src_img.exists() and ext == "jpg":
            src_img = BASE_DIR / f"cam{cam['id']-1}.jpg"
            
        if src_img.exists():
            shutil.copy(src_img, OUTPUT_DIR / "images" / f"cam{cam['id']-1}.png")
            break

print("✅ Pipeline completado.")
print(f"Dataset generado con éxito directo desde OpenCV en: {OUTPUT_DIR}")
print(f"Cámaras posicionadas y separadas artificialmente con escala x{SCALE_FACTOR}.")