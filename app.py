# Author: vlarobbyk
# Versión: 1.3 - Limpieza y Organización
# Fecha: 2025-05-05

from flask import Flask, render_template, Response, request
from io import BytesIO
import time
import cv2
import numpy as np
import requests

app = Flask(__name__)
STREAM_URL = 'http://192.168.18.57:81/stream'


# ==== [ UTILS ] ====
def calculate_fps(prev):
    now = time.time()
    fps = 1.0 / (now - prev["time"])
    prev["time"] = now
    return fps

def add_fps(img, fps):
    return cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# ==== [ MOTION DETECTION ] ====
def create_motion_detector():
    return cv2.createBackgroundSubtractorMOG2()

def detect_motion(mog2, frame):
    mask = mog2.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# ==== [ ILUMINATION FILTERS ] ====
def apply_lighting_filters(gray):
    eq_hist = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gamma = cv2.LUT(gray, np.array([(i / 255.0) ** (1.0 / 1.5) * 255 for i in range(256)]).astype("uint8"))

    return [cv2.putText(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            for img, label in zip([eq_hist, clahe, gamma], ["Hist. Equal.", "CLAHE", "Gamma Corr."])]


# ==== [ RUIDO ] ====
def add_gaussian_noise(image, mean=0, std=25):
    gauss = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
    return cv2.putText(noisy, f"Ruido Gaussiano µ={mean} σ={std}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def add_speckle_noise(image, var=0.04):
    noise = np.random.randn(*image.shape) * var
    noisy = np.clip(image + image * noise, 0, 255).astype(np.uint8)
    return cv2.putText(noisy, f"Ruido Speckle var={var:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ==== [ SMOOTHING FILTERS ] ====
def apply_smoothing(image, k=5):
    # Aseguramos que k sea impar para los filtros
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 3)

    # Definimos los filtros que vamos a aplicar
    blur = cv2.blur(image, (k, k))
    gauss = cv2.GaussianBlur(image, (k, k), 0)
    median = cv2.medianBlur(image, k)

    # Verificar si la imagen es en escala de grises
    if len(image.shape) == 2 or image.shape[2] == 1:
        blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        gauss = cv2.cvtColor(gauss, cv2.COLOR_GRAY2BGR)
        median = cv2.cvtColor(median, cv2.COLOR_GRAY2BGR)

    # Etiquetas para cada filtro
    cv2.putText(blur, f"Blur {k}x{k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(gauss, f"Gaussian {k}x{k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(median, f"Median {k}x{k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return blur, gauss, median

# ==== [ EDGE DETECTION ] ====
def apply_edges(image, smooth=False, k=5):
    if smooth:
        image = cv2.GaussianBlur(image, (k, k), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 50, 150)
    sobel = np.uint8(np.clip(cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)), 0, 255))

    labels = ["Canny", "Sobel"]
    images = [canny, sobel]

    return [cv2.putText(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                        f"{label} {'+Smooth' if smooth else 'Raw'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            for img, label in zip(images, labels)]

# ==== [ STREAM FUNCTION ] ====
def video_capture(mean=0, std=25, var=0.04):
    res = requests.get(STREAM_URL, stream=True)
    mog2 = create_motion_detector()
    prev = {"time": time.time()}

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) <= 100:
            continue
        try:
            frame = cv2.imdecode(np.frombuffer(BytesIO(chunk).read(), np.uint8), 1)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_mask = detect_motion(mog2, frame)

            # Mostrar FPS en frame principal
            fps = calculate_fps(prev)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # --- Agregar Ruido ---
            noisy_gauss = add_gaussian_noise(frame, mean, std)
            noisy_speckle = add_speckle_noise(frame, var)

            # --- Aplicar Filtros a las Imágenes con Ruido ---
            blur, gauss, median = [add_fps(img, fps) for img in apply_smoothing(noisy_gauss)]  # Aquí aplicamos los filtros al frame con ruido gaussiano
            noisy_gauss_blur, noisy_gauss_gauss, noisy_gauss_median = [add_fps(img, fps) for img in apply_smoothing(noisy_gauss)]  # Aplicar los filtros de suavizado al ruido

            # --- Otros Filtros ---
            motion_disp = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            motion_disp = add_fps(motion_disp, fps)
            eq_hist, eq_clahe, eq_gamma = [add_fps(img, fps) for img in apply_lighting_filters(gray)]
            canny_raw, sobel_raw = [add_fps(img, fps) for img in apply_edges(frame, False)]
            canny_smooth, sobel_smooth = [add_fps(img, fps) for img in apply_edges(frame, True)]

            # --- Agrupar las Imágenes ---
            rows = [
                np.hstack((frame, motion_disp, cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR))),
                np.hstack((eq_hist, eq_clahe, eq_gamma)),
                np.hstack((noisy_gauss, noisy_speckle)),
                np.hstack((noisy_gauss_blur, noisy_gauss_gauss, noisy_gauss_median)),
                np.hstack((canny_raw, sobel_raw)),
                np.hstack((canny_smooth, sobel_smooth))
            ]
            max_w = max(row.shape[1] for row in rows)
            stacked = np.vstack([cv2.resize(r, (max_w, r.shape[0])) for r in rows])

            ret, buffer = cv2.imencode('.jpg', stacked)
            if not ret:
                continue

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"[ERROR] {e}")
            continue

# ==== [ FLASK ROUTES ] ====
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def stream():
    mean = int(request.args.get("mean", 0))
    std = int(request.args.get("std", 25))
    var = float(request.args.get("var", 4)) / 100.0
    return Response(video_capture(mean, std, var), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_page")
def stream_page():
    mean = request.args.get("mean", "0")
    std = request.args.get("std", "25")
    var = request.args.get("var", "4")
    t = int(time.time())
    return f"""<html><body style='margin:0'><img src="/video_stream?mean={mean}&std={std}&var={var}&t={t}" style="width:100%;" /></body></html>"""

@app.route("/comparar_ruido")
def comparar_ruido():
    # Obtener los valores de los sliders desde la URL
    k = int(request.args.get("k", 3))  # Obtener el tamaño de la máscara (k) desde la URL, con valor por defecto 5
    mean = int(request.args.get("mean", 0))  # Obtener el valor de la media Gaussiana
    std = int(request.args.get("std", 25))  # Obtener el valor de la desviación estándar Gaussiana
    var = float(request.args.get("var", 4)) / 100.0  # Obtener la varianza del ruido Speckle

    # Obtener el flujo de la cámara
    res = requests.get(STREAM_URL, stream=True)
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            frame = cv2.imdecode(np.frombuffer(BytesIO(chunk).read(), np.uint8), 1)
            if frame is None:
                continue

            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Verificar si los valores de los sliders están en 0 para no aplicar ruido
            if mean == 0 and std == 0 and var == 0:
                noisy = frame.copy()  # Usar la imagen original sin ruido
            else:
                # Aplicar ruido si los valores de los sliders son mayores a 0
                noisy = add_gaussian_noise(frame.copy(), mean=mean, std=std)  # Aquí agregamos el ruido

            # Usar copyTo() para copiar la imagen con ruido
            noisy_copy = noisy.copy()  # Copiar la imagen con el ruido

            # Aplica los filtros de suavizado (con y sin ruido) sobre la imagen con ruido
            blur, gauss, median = apply_smoothing(noisy_copy, k)

            # Usamos copyTo() para copiar las imágenes filtradas a nuevas variables
            blur_bgr = blur.copy() if len(blur.shape) == 2 else blur  # Copia de la imagen filtrada
            gauss_bgr = gauss.copy() if len(gauss.shape) == 2 else gauss
            median_bgr = median.copy() if len(median.shape) == 2 else median

            # Etiquetas para cada filtro
            cv2.putText(blur_bgr, f"Blur {k}x{k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(gauss_bgr, f"Gaussian {k}x{k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(median_bgr, f"Median {k}x{k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Apila horizontalmente los resultados filtrados
            resultado_filtros = np.hstack((blur_bgr, gauss_bgr, median_bgr))

            # Detección de bordes sin suavizado
            canny_raw, sobel_raw = apply_edges(frame, smooth=False, k=k)

            # Detección de bordes con suavizado
            canny_smooth, sobel_smooth = apply_edges(frame, smooth=True, k=k)
            # Etiquetas claras para mostrar la comparación
            cv2.putText(canny_raw, "Canny Raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(sobel_raw, "Sobel Raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canny_smooth, "Canny + Smooth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(sobel_smooth, "Sobel + Smooth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Apila los resultados de la detección de bordes
            resultado_bordes = np.hstack((canny_raw, sobel_raw, canny_smooth, sobel_smooth))

            # Redimensionar todas las imágenes a un tamaño común
            height = min(resultado_filtros.shape[0], resultado_bordes.shape[0], frame.shape[0], noisy_copy.shape[0])
            width = min(resultado_filtros.shape[1], resultado_bordes.shape[1], frame.shape[1], noisy_copy.shape[1])

            # Redimensionar las imágenes para que todas tengan el mismo tamaño
            frame_resized = cv2.resize(frame, (width, height))
            noisy_resized = cv2.resize(noisy_copy, (width, height))  # Usar la imagen con el ruido
            resultado_filtros_resized = cv2.resize(resultado_filtros, (width, height))
            resultado_bordes_resized = cv2.resize(resultado_bordes, (width, height))

            # Asegúrate de que todas las imágenes tienen las mismas dimensiones
            print(
                f"Dimensiones de las imágenes: {frame_resized.shape}, {noisy_resized.shape}, {resultado_filtros_resized.shape}, {resultado_bordes_resized.shape}")

            # Apilar las imágenes para comparación
            resultado_completo = np.vstack(
                (frame_resized, noisy_resized, resultado_filtros_resized, resultado_bordes_resized))

            # Guardar las imágenes de comparación en un solo archivo
            cv2.imwrite("static/comparacion_ruido_bordes.jpg", resultado_completo)

            break

    return """
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Comparación de Filtros y Detección de Bordes</title>
            <style>
                body {
                    font-family: 'Segoe UI', sans-serif;
                    background-color: #f0f0f0;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }

                h2 {
                    color: #2d3e50;
                    font-size: 2rem;
                    margin-top: 20px;
                }

                .image-container {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    max-width: 1200px;
                    width: 100%;
                    padding: 20px;
                    margin-top: 40px;
                }

                .image-container img {
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                    border: 2px solid #ddd;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }

                .image-container .image-title {
                    text-align: center;
                    font-size: 1.1rem;
                    font-weight: bold;
                    margin-top: 10px;
                    color: #2d3e50;
                }

                .back-link {
                    text-decoration: none;
                    color: #2d3e50;
                    font-size: 1.1rem;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 6px;
                    transition: background-color 0.3s ease;
                }

                .back-link:hover {
                    background-color: #45a049;
                }

                @media (max-width: 768px) {
                    .image-container {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <h2>Comparación de Filtros y Detección de Bordes</h2>
            <div class="image-container">
                <div class="image">
                    <img src="/static/comparacion_ruido_bordes.jpg" alt="Comparación de Filtros y Bordes">
                    <div class="image-title">Comparación de Filtros y Detección de Bordes</div>
                </div>
            </div>
            <a href="/" class="back-link">Volver al inicio</a>
        </body>
        </html>
    """

from flask import Response

@app.route('/bitwise')
def video_bitwise_operations_stream():
    url = 'http://192.168.18.57:81/stream'

    try:
        res = requests.get(url, stream=True)
    except Exception as e:
        print(f"[ERROR] No se pudo conectar al stream: {e}")
        return "Stream error", 500

    fgbg = cv2.createBackgroundSubtractorMOG2()

    def generate():
        for chunk in res.iter_content(chunk_size=100000):
            if len(chunk) <= 100:
                continue

            try:
                frame = cv2.imdecode(np.frombuffer(BytesIO(chunk).read(), np.uint8), 1)
                if frame is None:
                    continue

                # Detección de movimiento
                fgmask = fgbg.apply(frame)
                _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

                # Máscara circular
                mask = np.zeros_like(fgmask)
                cv2.circle(mask, (frame.shape[1]//2, frame.shape[0]//2), 100, 255, -1)

                # Operaciones bitwise
                and_result = cv2.bitwise_and(fgmask, mask)
                or_result = cv2.bitwise_or(fgmask, mask)
                xor_result = cv2.bitwise_xor(fgmask, mask)

                # Stack horizontal (puedes usar vertical también con vstack)
                combined = np.hstack((
                    cv2.cvtColor(and_result, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(or_result, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(xor_result, cv2.COLOR_GRAY2BGR)
                ))

                ret, jpeg = cv2.imencode('.jpg', combined)
                if not ret:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            except Exception as e:
                print(f"[WARNING] Error procesando el frame: {e}")
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ==== [ MAIN ] ====
if __name__ == "__main__":
    app.run(debug=False)
