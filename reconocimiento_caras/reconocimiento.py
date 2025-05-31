import cv2
from deepface import DeepFace
import numpy as np

# -----------------------
# CONFIGURACIÓN
# -----------------------
ruta = "../imagenes/barcelona.jpg"
detector_backend = "retinaface"

# -----------------------
# Cargar imagen
# -----------------------
try:
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"Error al cargar la imagen: {e}")
    exit()

print(f"Imagen original - shape: {img.shape}, min: {img.min()}, max: {img.max()}")

# -----------------------
# Aumentar resolución si es pequeña
# -----------------------
if img.shape[0] < 800 or img.shape[1] < 800:
    img_rgb = cv2.resize(img_rgb, (0, 0), fx=1.5, fy=1.5)
    print(f"Imagen redimensionada - shape: {img_rgb.shape}")

# -----------------------
# Detectar rostros con DeepFace
# -----------------------
try:
    faces = DeepFace.extract_faces(img_path=img_rgb, detector_backend=detector_backend, enforce_detection=False)
except Exception as e:
    print(f"Error al detectar rostros: {e}")
    exit()

if not faces:
    print("No se detectaron caras.")
    exit()

print(f"Caras detectadas: {len(faces)}")

# -----------------------
# Mostrar imagen con rostros detectados
# -----------------------
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

for i, face_info in enumerate(faces):
    facial_area = face_info["facial_area"]
    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
    print(f"Rostro #{i+1}: Área facial: {facial_area}")

    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img_bgr, f"Rostro {i+1}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Imagen con rostros detectados", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Mostrar cada rostro recortado
# -----------------------
for i, face_info in enumerate(faces):
    face = face_info["face"]
    
    face_uint8 = (face * 255).astype(np.uint8)
    rostro_bgr = cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR)

    print(f"Rostro recortado #{i+1} - shape: {face.shape}")
    cv2.imshow(f"Rostro #{i+1}", rostro_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
