
# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, request, Response, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'imagenes_medicas'
app.config['PROCESSED_FOLDER'] = 'static/processed'

# Asegurarse de que existen las carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def apply_morph_operations(img):
    # Usa kernels más grandes
    kernel_sizes = [15, 25, 37]  # Ajusta según necesidad
    results = {}
    
    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))  # Cambia a MORPH_RECT
        
        # Operaciones
        erosion = cv2.erode(img, kernel, iterations=1)
        dilation = cv2.dilate(img, kernel, iterations=1)
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        combined = cv2.add(img, cv2.subtract(tophat, blackhat))
        
        results[f'{size}x{size}'] = {
            'original': img,
            'erosion': erosion,
            'dilation': dilation,
            'tophat': tophat,
            'blackhat': blackhat,
            'combined': combined
        }
    
    return results

@app.route("/")
def index():
    """Página principal con formulario de carga"""
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    """Procesa las imágenes cargadas y prepara datos para el template"""
    files = request.files.getlist("image")
    processed_results = []
    
    for file in files:
        if file.filename != '':
            # Leer y procesar imagen
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            
            # Aplicar operaciones morfológicas
            results = apply_morph_operations(img)
            
            # Guardar resultados y preparar datos para la vista
            for kernel_size, operations in results.items():
                # Guardar todas las versiones procesadas
                image_data = {
                    'original_name': file.filename,
                    'kernel_size': kernel_size
                }
                
                for op_name, processed_img in operations.items():
                    filename = f"{file.filename}_{kernel_size}_{op_name}.jpg"
                    path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
                    cv2.imwrite(path, processed_img)
                    
                    # Asociar cada operación a su ruta
                    image_data[f'{op_name}_path'] = filename
                
                processed_results.append(image_data)
    
    return render_template("resultado_imgs.html", results=processed_results)


@app.route("/processed/<filename>")
def processed_file(filename):
    """Sirve archivos procesados"""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)

