import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

MODEL_PATH = "modelo_mnist.h5"

# Verificar si el modelo ya existe para cargarlo o entrenarlo
if os.path.exists(MODEL_PATH):
    print("⚡ Cargando modelo guardado...")
    model = load_model(MODEL_PATH)
else:
    print("🚀 Entrenando modelo...")
    # Carga y procesa el modelo
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Definir modelo CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train_cat, batch_size=128, epochs=5, verbose=1, validation_split=0.1)
    print("🔍 Precisión en test:", model.evaluate(x_test, y_test_cat, verbose=0)[1])

    # Guardar modelo entrenado
    model.save(MODEL_PATH)
    print(f"💾 Modelo guardado en '{MODEL_PATH}'")

# ======================
# Procesamiento desde lienzo
# ======================

lienzo = np.zeros((280, 280), dtype=np.uint8)
dibujando = False

def dibujar(event, x, y, flags, param):
    global dibujando
    if event == cv2.EVENT_LBUTTONDOWN:
        dibujando = True
    elif event == cv2.EVENT_MOUSEMOVE and dibujando:
        cv2.circle(lienzo, (x, y), 8, (255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        dibujando = False

cv2.namedWindow('Dibuja un número del 0 al 9')
cv2.setMouseCallback('Dibuja un número del 0 al 9', dibujar)

def procesar_lienzo(lienzo):
    coords = cv2.findNonZero(lienzo)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    recorte = lienzo[y:y+h, x:x+w]

    _, binarizada = cv2.threshold(recorte, 127, 255, cv2.THRESH_BINARY)
    suavizada = cv2.GaussianBlur(binarizada, (3, 3), 0)

    h_, w_ = suavizada.shape
    if h_ > w_:
        new_h = 20
        new_w = int(w_ * 20 / h_)
    else:
        new_w = 20
        new_h = int(h_ * 20 / w_)
    redim = cv2.resize(suavizada, (new_w, new_h), interpolation=cv2.INTER_AREA)

    imagen_centrada = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    imagen_centrada[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = redim

    return imagen_centrada

# ======================
# Loop principal
# ======================

print("\n🎨 Instrucciones:")
print(" - Dibuja con el mouse")
print(" - Presiona 'p' para predecir")
print(" - Presiona 'c' para limpiar")
print(" - Presiona 'q' para salir")

while True:
    cv2.imshow('Dibuja un número del 0 al 9', lienzo)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        lienzo[:] = 0
        print("🧽 Lienzo limpiado.")
    elif key == ord('q'):
        break
    elif key == ord('p'):
        imagen = procesar_lienzo(lienzo)
        entrada = imagen.reshape(1, 28, 28, 1).astype("float32") / 255.0
        pred = model.predict(entrada)
        clase = np.argmax(pred)

        print(f"🧠 Predicción: {clase} (confianza: {pred[0][clase]*100:.2f}%)")

        fig, ax = plt.subplots(1, 1)
        ax.imshow(imagen, cmap="gray")
        ax.set_title(f"Tu número: {clase}")
        ax.axis("off")
        plt.show()

cv2.destroyAllWindows()
