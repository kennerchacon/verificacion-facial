import traceback
from flask import Flask, render_template, request, redirect, session, url_for
import os
import base64
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'imagenes_usuarios'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.secret_key = 'a2f74cb2e9834b5f9809426b2dc57926cf6f23d4e4f5ad7b6e2139db12e32b2c'  # Secret key para validar sesión

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/registro')
def registro():
    return render_template('registro.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/bienvenido')
def bienvenido():
    if not session.get('usuario_autenticado'):
        return redirect(url_for('login'))
    return render_template('bienvenido.html', nombre=session.get('nombre_usuario'))

@app.route('/guardar', methods=['POST'])
def guardar():
    data = request.form['image']
    nombre = request.form['nombre'].strip()

    if not nombre:
        return "❌ Nombre vacío", 400

    content = data.split(',')[1]
    img_data = base64.b64decode(content)
    filename = f"{nombre}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, 'wb') as f:
        f.write(img_data)

    return f"✅ Imagen guardada como {filename}"

@app.route('/verificar', methods=['POST'])
def verificar():
    data = request.form['image']
    nombre = request.form['nombre'].strip()

    if not nombre:
        return "❌ Nombre vacío", 400

    try:
        content = data.split(',')[1]
        img_data = base64.b64decode(content)
    except Exception:
        return "❌ Imagen no válida", 400

    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return "❌ No se pudo decodificar la imagen", 400

    print("📷 Verificando imagen del usuario:", nombre)

    img_capturada_path = "captura_debug.jpg"
    cv2.imwrite(img_capturada_path, img)

    img_guardada_path = os.path.join(UPLOAD_FOLDER, f"{nombre}.jpg")
    if not os.path.exists(img_guardada_path):
        print("🚫 Imagen no encontrada:", img_guardada_path)
        return "❌ Usuario no registrado", 404

    try:
        resultado = DeepFace.verify(
            img1_path=img_capturada_path,
            img2_path=img_guardada_path,
            model_name='Facenet512',
            detector_backend='retinaface',
            distance_metric='cosine'
        )
        print("✅ Resultado DeepFace:", resultado)

        if resultado["verified"]:
            session['usuario_autenticado'] = True
            session['nombre_usuario'] = nombre
            return "✅ Usuario verificado correctamente"
        else:
            session.clear()
            return "❌ Usuario no coincide"
    except Exception as e:
        print("❌ Error al verificar con DeepFace:")
        print(e)
        traceback.print_exc()
        return f"❌ Error en la verificación: {str(e)}", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
