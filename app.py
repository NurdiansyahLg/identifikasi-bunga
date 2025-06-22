from flask import Flask, render_template, request, redirect, url_for
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Cek apakah folder static/uploads sudah ada
upload_folder = 'static/uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Load model CNN dan label
model = load_model('model_bunga.h5')
with open('label_map.json') as f:
    class_indices = json.load(f)
labels = list(class_indices.keys())  # daftar label bunga

# Format gambar yang diperbolehkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Halaman utama
@app.route('/')
def index():
    return render_template('dashboard.html')

# Prediksi bunga dari gambar upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if not allowed_file(file.filename):
        return redirect(url_for('index'))

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    # Proses gambar
    img = image.load_img(filepath, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Prediksi
    preds = model.predict(x)[0]
    pred_idx = np.argmax(preds)
    result = labels[pred_idx]
    confidence = round(float(np.max(preds)) * 100, 2)

    return render_template('result.html',
                           filename=file.filename,
                           jenis_bunga=result,
                           confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
