import json
import math
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS DE ENTRADA ---
DIR_CALIB = Path(r"C:\Projects\Itrat_Valeria_Gaussian\convert_calibration\input")  # Cambia por la carpeta donde están tus JSON
CAM0_IN = DIR_CALIB / "cam0_intrinsics.json"
CAM1_IN = DIR_CALIB / "cam1_intrinsics.json"
CAM2_IN = DIR_CALIB / "cam2_intrinsics.json"
STEREO_01 = DIR_CALIB / "stereo_cam0_cam1.json"
STEREO_02 = DIR_CALIB / "stereo_cam0_cam2.json"

# --- CONFIGURACIÓN DE SALIDA ---
# Nombre de los archivos de imagen reales que tienes en tu carpeta de dataset
# AJUSTA ESTOS NOMBRES si tus imágenes se llaman de otra forma (ej: "cam0.png")
FILE_NAMES = ["cam0.png", "cam1.png", "cam2.png"] 
OUTPUT_JSON = DIR_CALIB / "transforms_train.json"

def calcular_fov_x(fx, width):
    """Calcula el camera_angle_x (Horizontal FOV) que requiere el formato Blender."""
    return 2 * math.atan(width / (2 * fx))

def opencv_to_opengl(matrix_c2w):
    """
    MATEMÁTICA CRUCIAL: Invierte los ejes Y y Z para pasar del idioma
    OpenCV (Kinect) al idioma OpenGL/Blender (Inria 3DGS).
    """
    flip_axes = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])
    return matrix_c2w @ flip_axes

def construir_matriz_4x4(R, T):
    """Construye una matriz de transformación homogénea de 4x4."""
    mat = np.eye(4)
    mat[:3, :3] = np.array(R)
    mat[:3, 3] = np.array(T).flatten()
    return mat

def main():
    print("🧠 Procesando calibraciones de OpenCV...")

    # 1. Cargar datos intrínsecos de Cam 0 para obtener las dimensiones y fov
    with open(CAM0_IN, "r") as f:
        cam0_data = json.load(f)
    fx = cam0_data["K"][0][0]
    w, h = cam0_data["img_size"]
    camera_angle_x = calcular_fov_x(fx, w)

    # 2. Inicializar las matrices extrínsecas relativas (Camera-to-World antes del flip)
    # Como Cam 0 es la referencia original de OpenCV, su pose relativa es la Identidad.
    c2w_cam0 = np.eye(4)

    # Cargar Stereo Cam 0 -> Cam 1 (Mapea puntos de 0 a 1 -> Matriz Extrínseca v1)
    with open(STEREO_01, "r") as f:
        st_01 = json.load(f)
    w2c_cam1 = construir_matriz_4x4(st_01["R"], st_01["T"])
    c2w_cam1 = np.linalg.inv(w2c_cam1) # Invertimos para tener Camera-to-World

    # Cargar Stereo Cam 0 -> Cam 2 (Mapea puntos de 0 a 2 -> Matriz Extrínseca v2)
    with open(STEREO_02, "r") as f:
        st_02 = json.load(f)
    w2c_cam2 = construir_matriz_4x4(st_02["R"], st_02["T"])
    c2w_cam2 = np.linalg.inv(w2c_cam2) # Invertimos para tener Camera-to-World

    # Lista ordenada de nuestras matrices c2w crudas
    raw_c2ws = [c2w_cam0, c2w_cam1, c2w_cam2]

    # 3. Estructurar el JSON final formato Blender
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": []
    }

    for idx, raw_c2w in enumerate(raw_c2ws):
        # Aplicamos la inversión de coordenadas obligatoria para Inria 3DGS
        final_c2w = opencv_to_opengl(raw_c2w)
        
        frame_entry = {
            "file_path": f"./train/{Path(FILE_NAMES[idx]).stem}", # Formato Blender sin extensión
            "rotation": 0.0,
            "transform_matrix": final_c2w.tolist()
        }
        transforms["frames"].append(frame_entry)
        print(f"✅ Cámara {idx} convertida con éxito.")

    # 4. Guardar archivo transforms_train.json
    with open(OUTPUT_JSON, "w") as f:
        json.dump(transforms, f, indent=4)
        
    print(f"\n🎉 ¡Proceso completado! Archivo guardado en: {OUTPUT_JSON.resolve()}")
    print("👉 Asegúrate de que tus 3 imágenes estén dentro de una carpeta llamada 'train' al lado de este JSON.")

if __name__ == "__main__":
    main()