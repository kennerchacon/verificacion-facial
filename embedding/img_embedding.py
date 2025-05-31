import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import json
import os

# Se carga el modelo de clasificación:
model = MobileNetV2(weights="imagenet", include_top=False, pooling='avg')

# Función para procesar la imágen:
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Importamos la imágen:
img_path = "../imagenes/man.jpg"

# Procesamos la imágen:
input_image = preprocess_image(img_path)

# Obtenemos el embedding (Representación Vectorial)
embedding = model.predict(input_image)[0]

# Asegurarse de que la carpeta 'embedding' exista
os.makedirs("embedding", exist_ok=True)

# Generamos un JSON
with open("embedding/imagen_embedding_tf.json", "w") as f:
    json.dump(embedding.tolist(), f, indent=2)
