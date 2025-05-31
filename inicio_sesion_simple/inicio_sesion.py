import cv2
from deepface import DeepFace

# ----------------------------
# Función para capturar una imagen desde la cámara
# ----------------------------
def capturar_rostro(nombre_archivo="captura.jpg"):
    cap = cv2.VideoCapture(0)  # Abre la cámara (índice 0)
    print("Presiona 's' para capturar imágen de inicio de sesión")
    
    while True:
        ret, frame = cap.read()  # Lee un frame de la cámara
        cv2.imshow("Login - Presiona 's' para capturar", frame)  # Muestra el frame en una ventana

        # Espera a que el usuario presione 's' para capturar la imagen
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(nombre_archivo, frame)  # Guarda la imagen capturada
            print(f"📸 Imagen capturada como {nombre_archivo}")
            break

    # Libera los recursos de la cámara y cierra la ventana
    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# Función para verificar la identidad facial
# ----------------------------
def login_facial(imagen_referencia="usuario.jpg", imagen_login="captura.jpg"):
    try:
        # Carga las imágenes de referencia (usuario registrado) y login (imagen recién capturada)
        img1 = cv2.imread(imagen_referencia)
        img2 = cv2.imread(imagen_login)

        # Validar si ambas imágenes se cargaron correctamente
        if img1 is None or img2 is None:
            print("❗ Error al leer una o ambas imágenes.")
            return

        # Compara los rostros utilizando DeepFace
        resultado = DeepFace.verify(
            img1_path=imagen_referencia,      # Imagen del usuario registrado
            img2_path=imagen_login,           # Imagen capturada para login
            model_name='Facenet512',          # Modelo de reconocimiento facial
            detector_backend='retinaface',    # Detector de rostros a usar
            distance_metric='cosine'          # Métrica de distancia para comparar embeddings
        )

        # Imprime el resultado completo
        print("Resultado:", resultado)

        # Verifica si el rostro fue reconocido exitosamente
        if resultado["verified"]:
            print("✅ Acceso concedido")
        else:
            print("❌ Acceso denegado")

    except Exception as e:
        # Captura y muestra cualquier error durante la verificación
        print("🚫 Error: ", e)

# ----------------------------
# Ejecución del flujo: capturar y verificar
# ----------------------------
capturar_rostro()       # Captura imagen del usuario en tiempo real
login_facial()          # Compara con imagen de referencia y decide acceso
