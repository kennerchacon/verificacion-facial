import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises
img_gray = cv2.imread("../imagenes/barcelona.jpg", cv2.IMREAD_GRAYSCALE)

# También cargar en color para el sobel colorizado
img_color = cv2.imread("../imagenes/barcelona.jpg")
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Lista de kernels
kernels = {
    "Original": np.array([[0]]),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Box Blur": np.ones((3, 3), np.float32) / 9.0,
    "Gaussian Blur": cv2.getGaussianKernel(3, 0) @ cv2.getGaussianKernel(3, 0).T,
    "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Prewitt Y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
}

# Aplicar cada filtro en escala de grises
results = {"Original": img_gray}
for name, kernel in kernels.items():
    if name != "Original":
        results[name] = cv2.filter2D(img_gray, -1, kernel)

# Agregar el filtro Sobel colorizado
from matplotlib.colors import hsv_to_rgb

gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(gx**2 + gy**2)
angle = np.arctan2(gy, gx)
angle_normalized = (angle + np.pi) / (2 * np.pi)

# Crear imagen HSV con mayor intensidad
hue = angle_normalized
saturation = np.ones_like(hue)
value = np.clip((magnitude / magnitude.max()) * 2.5, 0, 1)

hsv_image = np.stack((hue, saturation, value), axis=-1)
rgb_image = (hsv_to_rgb(hsv_image) * 255).astype(np.uint8)
results["Sobel Color"] = rgb_image


# Mostrar los resultados
cols = 4
rows = (len(results) + cols - 1) // cols
plt.figure(figsize=(16, 4 * rows))

for i, (name, result) in enumerate(results.items()):
    plt.subplot(rows, cols, i + 1)
    if name == "Sobel Color":
        plt.imshow(result)
    else:
        plt.imshow(result, cmap='gray')
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()
