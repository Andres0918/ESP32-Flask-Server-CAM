
# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, Response, stream_with_context, Request
from io import BytesIO
import time
import cv2
import numpy as np
import requests

app = Flask(__name__)
# IP Address
_URL = 'http://192.168.18.57'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])


## -----------Metodos logicos-----------------
def create_motion_detector():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def detect_motion_adaptive(mog2, frame):
    mask = mog2.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def calculate_fps(prev_time_container):
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time_container["time"])
    prev_time_container["time"] = current_time
    return fps

def apply_hist_equalization(gray):
    return cv2.equalizeHist(gray)

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_gamma_correction(gray, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    return cv2.LUT(gray, table)


def video_capture():
    res = requests.get(stream_url, stream=True)
    mog2 = create_motion_detector()
    prev_time_container = {"time": time.time()}

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                # Decodificar el chunk como imagen
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                if cv_img is None:
                    continue

                # === Calcular FPS ===
                fps = calculate_fps(prev_time_container)
                cv2.putText(cv_img, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # === Movimiento (MOG2 adaptativo) ===
                motion_mask = detect_motion_adaptive(mog2, cv_img)
                motion_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

                # === Ruido sal y pimienta ===
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                N = 537
                noise = np.zeros((height, width), dtype=np.uint8)
                rand_y = np.random.randint(0, height, N)
                rand_x = np.random.randint(0, width, N)
                noise[rand_y, rand_x] = 255
                noise_image = cv2.bitwise_or(gray, noise)
                noise_display = cv2.cvtColor(noise_image, cv2.COLOR_GRAY2BGR)

                # === Combinar: original + movimiento + ruido ===
                combined = np.hstack((cv_img, motion_display, noise_display))

                flag, encodedImage = cv2.imencode(".jpg", combined)
                if not flag:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(f"Error procesando chunk: {e}")
                continue


def video_capture_native():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara")

    mog2 = create_motion_detector()
    prev_time_container = {"time": time.time()}

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            continue

        try:
            # === Calcular FPS ===
            fps = calculate_fps(prev_time_container)
            cv2.putText(curr_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # === Convertir a escala de grises ===
            gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            # === Detección de movimiento (MOG2) ===
            motion_mask = detect_motion_adaptive(mog2, curr_frame)
            motion_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

            # === Filtros para mejora de iluminación ===
            eq_hist = apply_hist_equalization(gray)
            eq_clahe = apply_clahe(gray)
            eq_gamma = apply_gamma_correction(gray, gamma=1.5)

            eq_hist_color = cv2.cvtColor(eq_hist, cv2.COLOR_GRAY2BGR)
            eq_clahe_color = cv2.cvtColor(eq_clahe, cv2.COLOR_GRAY2BGR)
            eq_gamma_color = cv2.cvtColor(eq_gamma, cv2.COLOR_GRAY2BGR)
            cv2.putText(eq_hist_color, "Hist. Equal.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.putText(eq_clahe_color, "CLAHE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.putText(eq_gamma_color, "Gamma Corr.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Concatenar original + movimiento
            row1 = np.hstack((curr_frame, motion_display))

            # Concatenar filtros
            row2 = np.hstack((eq_hist_color, eq_clahe_color, eq_gamma_color))

            # Asegurar que ambas filas tengan mismo ancho antes de hacer vstack
            if row1.shape[1] != row2.shape[1]:
                target_width = max(row1.shape[1], row2.shape[1])
                row1 = cv2.resize(row1, (target_width, row1.shape[0]))
                row2 = cv2.resize(row2, (target_width, row2.shape[0]))

            combined = np.vstack((row1, row2))

            # === Codificar como JPEG ===
            flag, encodedImage = cv2.imencode(".jpg", combined)
            if not flag:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) +
                   b'\r\n')

        except Exception as e:
            print(f"Error procesando frame: {e}")
            continue



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture_native(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)

