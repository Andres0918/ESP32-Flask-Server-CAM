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


# ==== [ MOTION DETECTION ] ====
def create_motion_detector():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

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
def apply_smoothing(gray, k=5):
    k = k + 1 if k % 2 == 0 else k
    k = max(k, 3)

    methods = {
        "Blur": cv2.blur(gray, (k, k)),
        "Gaussian": cv2.GaussianBlur(gray, (k, k), 0),
        "Median": cv2.medianBlur(gray, k)
    }

    return [cv2.putText(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), f"{name} {k}x{k}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            for name, img in methods.items()]


# ==== [ EDGE DETECTORS ] ====
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
            if frame is None: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_mask = detect_motion(mog2, frame)
            motion_disp = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            motion_only = cv2.bitwise_and(frame, frame, mask=motion_mask)

            fps = calculate_fps(prev)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(motion_only, "Movimiento (AND)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            eq_hist, eq_clahe, eq_gamma = apply_lighting_filters(gray)
            noisy_gauss = add_gaussian_noise(frame, mean, std)
            noisy_speckle = add_speckle_noise(frame, var)
            blur, gauss, median = apply_smoothing(gray)
            canny_raw, sobel_raw = apply_edges(frame, False)
            canny_smooth, sobel_smooth = apply_edges(frame, True)

            rows = [
                np.hstack((frame, motion_disp, motion_only)),
                np.hstack((eq_hist, eq_clahe, eq_gamma)),
                np.hstack((noisy_gauss, noisy_speckle)),
                np.hstack((blur, gauss, median)),
                np.hstack((canny_raw, sobel_raw)),
                np.hstack((canny_smooth, sobel_smooth))
            ]
            max_w = max(row.shape[1] for row in rows)
            stacked = np.vstack([cv2.resize(r, (max_w, r.shape[0])) for r in rows])

            ret, buffer = cv2.imencode('.jpg', stacked)
            if not ret: continue

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
    res = requests.get(STREAM_URL, stream=True)
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            frame = cv2.imdecode(np.frombuffer(BytesIO(chunk).read(), np.uint8), 1)
            if frame is None:
                continue

            # Agrega ruido Gaussiano visible
            noisy = add_gaussian_noise(frame.copy(), mean=0, std=50)
            gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)

            # Aplica filtros
            blur = cv2.blur(gray, (5, 5))
            gauss = cv2.GaussianBlur(gray, (5, 5), 0)
            median = cv2.medianBlur(gray, 5)

            # Pasa a color para apilar
            blur_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
            gauss_bgr = cv2.cvtColor(gauss, cv2.COLOR_GRAY2BGR)
            median_bgr = cv2.cvtColor(median, cv2.COLOR_GRAY2BGR)

            # Etiquetas
            cv2.putText(blur_bgr, "Blur 5x5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(gauss_bgr, "Gaussian 5x5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(median_bgr, "Median 5x5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Apila horizontalmente
            resultado = np.hstack((blur_bgr, gauss_bgr, median_bgr))
            cv2.imwrite("static/comparacion_ruido.jpg", resultado)
            break
    return """
        <html><body style='text-align:center;'>
        <h2>Comparación de Filtros con Ruido Gaussiano</h2>
        <img src='/static/comparacion_ruido.jpg' width='90%'><br>
        <a href='/'>Volver al inicio</a>
        </body></html>
    """


# ==== [ MAIN ] ====
if __name__ == "__main__":
    app.run(debug=False)
