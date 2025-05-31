import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

MODEL_PATH = "modelo_mnist.h5"

# Cargar o entrenar modelo
if os.path.exists(MODEL_PATH):
    print("⚡ Cargando modelo guardado...")
    model = load_model(MODEL_PATH)
else:
    print("🚀 Entrenando modelo...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

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
    model.save(MODEL_PATH)
    print(f"💾 Modelo guardado en '{MODEL_PATH}'")

# Procesamiento del lienzo (donde se dibuja)
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

def normalizar_imagen(im):
    return im.astype("float32") / 255.0

def mostrar_convolucion_paso_a_paso(imagen_28x28, kernel, paso_delay=60):
    """
    imagen_28x28: numpy array shape (28,28) float32, normalizado
    kernel: numpy array shape (3,3)
    paso_delay: ms de delay para ver el paso
    """

    H, W = imagen_28x28.shape
    kH, kW = kernel.shape
    salida_H = H - kH + 1
    salida_W = W - kW + 1

    # Imagen para dibujar el kernel encima (marcar región)
    imagen_rgb = cv2.cvtColor((imagen_28x28*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Salida inicializada en cero
    salida = np.zeros((salida_H, salida_W), dtype=np.float32)

    # Normalizar kernel para mostrarlo visualmente
    kernel_vis = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel) + 1e-9)
    kernel_vis = (kernel_vis * 255).astype(np.uint8)
    kernel_vis = cv2.resize(kernel_vis, (100, 100), interpolation=cv2.INTER_NEAREST)
    kernel_vis_color = cv2.applyColorMap(kernel_vis, cv2.COLORMAP_JET)

    ventana = "Visualización paso a paso de la convolución"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, 900, 450)

    for y in range(salida_H):
        for x in range(salida_W):
            # Extraer región
            region = imagen_28x28[y:y+kH, x:x+kW]

            # Aplicar kernel
            valor = np.sum(region * kernel)
            salida[y, x] = valor

            # Crear canvas
            canvas = np.zeros((280, 900, 3), dtype=np.uint8)

            img_marcada = imagen_rgb.copy()
            top_left = (x, y)
            bottom_right = (x + kW - 1, y + kH - 1)
            cv2.rectangle(img_marcada, top_left, bottom_right, (0, 0, 255), 1)

            # Redimensionar para que encaje en canvas grande
            img_marcada_resized = cv2.resize(img_marcada, (280, 280), interpolation=cv2.INTER_NEAREST)
            canvas[:280, :280] = img_marcada_resized

            # Mostrar kernel
            canvas[:100, 320:420] = kernel_vis_color

            # Mostrar región de la imagen para kernel
            region_vis = (region * 255).astype(np.uint8)
            region_vis = cv2.resize(region_vis, (100, 100), interpolation=cv2.INTER_NEAREST)
            region_vis_color = cv2.cvtColor(region_vis, cv2.COLOR_GRAY2BGR)
            canvas[120:220, 320:420] = region_vis_color

            # Salida parcial
            salida_norm = salida.copy()
            salida_norm = salida_norm - salida_norm.min()
            if salida_norm.max() > 0:
                salida_norm = salida_norm / salida_norm.max()
            salida_vis = (salida_norm * 255).astype(np.uint8)
            salida_vis = cv2.resize(salida_vis, (280, 280), interpolation=cv2.INTER_NEAREST)
            salida_vis = cv2.applyColorMap(salida_vis, cv2.COLORMAP_VIRIDIS)
            canvas[:280, 620:900] = salida_vis

            # Añadimos el texto:
            cv2.putText(canvas, "Imagen con region de kernel", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(canvas, "Kernel (filtro)", (320, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(canvas, "Region imagen (3x3)", (320, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(canvas, "Salida parcial de convolucion", (620, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow(ventana, canvas)
            key = cv2.waitKey(paso_delay)
            if key == 27: #Telca ESC
                cv2.destroyWindow(ventana)
                return

    cv2.waitKey(0)
    cv2.destroyWindow(ventana)


print("\n🎨 Instrucciones:")
print(" - Dibuja con el mouse")
print(" - Presiona 'p' para predecir y ver convolución paso a paso")
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

        # Mostrar la convolución paso a paso solo con el primer filtro de la primera Conv2D
        pesos = model.layers[0].get_weights()[0]  # pesos shape (3,3,1,32)
        primer_kernel = pesos[:, :, 0, 0]  # kernel 3x3 para el primer filtro

        print("🔎 Mostrando convolución paso a paso...")
        mostrar_convolucion_paso_a_paso(entrada[0, :, :, 0], primer_kernel)

        # Predicción Final
        predicciones = model.predict(entrada)
        clase_predicha = np.argmax(predicciones)
        probabilidad = np.max(predicciones)
        print(f"🧠 Predicción final: {clase_predicha} con probabilidad {probabilidad:.4f}")


cv2.destroyAllWindows()
