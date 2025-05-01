
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
_URL = 'http://10.0.0.3'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])


def video_capture():
    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):

        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                N = 537
                height, width = gray.shape
                noise = np.full((height, width), 0, dtype=np.uint8)
                random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))

                noise[random_positions[0], random_positions[1]] = 255

                noise_image = cv2.bitwise_or(gray, noise)

                total_image = np.zeros((height, width * 2), dtype=np.uint8)
                total_image[:, :width] = gray
                total_image[:, width:] = noise_image

                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue

def video_capture_native():
    cap = cv2.VideoCapture(0)  # Usa webcam local
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara")

    prev_gray = None  # frame anterior en escala de grises
    prev_time = time.time()
    mog2 = create_motion_detector()
    prev_time_container = {"time": time.time()}

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            continue

        try:

            fps = calculate_fps(prev_time_container)

            cv2.putText(curr_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Convertir a escala de grises
            gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            motion_mask = detect_motion_adaptive(mog2, curr_frame)
            motion_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

            # Generar ruido sal y pimienta
            N = 537
            noise = np.zeros((height, width), dtype=np.uint8)
            rand_y = np.random.randint(0, height, N)
            rand_x = np.random.randint(0, width, N)
            noise[rand_y, rand_x] = 255
            noise_image = cv2.bitwise_or(gray, noise)
            noise_display = cv2.cvtColor(noise_image, cv2.COLOR_GRAY2BGR)

            # Concatenar: original + movimiento + con ruido
            combined = np.hstack((curr_frame, motion_display, noise_display))

            # Codificar como JPEG
            flag, encodedImage = cv2.imencode(".jpg", combined)
            if not flag:
                continue

            # Enviar por stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) +
                   b'\r\n')

        except Exception as e:
            print(f"Error procesando frame: {e}")
            continue


@app.route("/")
def index():
    return render_template("test.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture_native(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


## -----------Metodos logicos-----------------
def create_motion_detector():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def detect_motion_adaptive(mog2_subtractor, curr_frame):
    # Aplica substracción adaptativa
    mask = mog2_subtractor.apply(curr_frame)

    # Limpieza con operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

import time

# Guarda el tiempo anterior en una variable mutable (como un diccionario)
def calculate_fps(prev_time_container):
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time_container["time"])
    prev_time_container["time"] = current_time
    return fps





if __name__ == "__main__":
    app.run(debug=False)

