import os
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

base_dir   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base_dir, 'EfficientNetB0_KD_Hasil_Eksperimen.keras')

model = load_model(MODEL_PATH)
print("Model berhasil dimuat!")

CLASS_NAMES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

DISEASE_INFO = {
    'Bacterial spot'                      : 'Penyakit bercak bakteri yang menyebabkan bercak coklat pada daun.',
    'Early blight'                        : 'Penyakit hawar awal yang disebabkan jamur Alternaria solani.',
    'Late blight'                         : 'Penyakit hawar daun yang disebabkan Phytophthora infestans.',
    'Leaf Mold'                           : 'Jamur kapang daun yang tumbuh di kondisi lembab.',
    'Septoria leaf spot'                  : 'Bercak daun Septoria yang menyebabkan bercak kecil berbatas gelap.',
    'Spider mites Two-spotted spider mite': 'Tungau laba-laba yang merusak jaringan daun.',
    'Target Spot'                         : 'Bercak melingkar seperti target pada permukaan daun.',
    'Tomato Yellow Leaf Curl Virus'       : 'Virus yang menyebabkan daun mengkerut dan menguning.',
    'Tomato mosaic virus'                 : 'Virus mosaik yang menyebabkan pola belang pada daun.',
    'healthy'                             : 'Daun tomat dalam kondisi sehat, tidak ditemukan penyakit.'
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Threshold minimum confidence agar prediksi dianggap valid
# Jika model tidak yakin (semua kelas rendah), gambar dianggap bukan daun tomat
CONFIDENCE_THRESHOLD = 75.0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img       = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Model KD output adalah logits (activation=None), konversi ke probabilitas
    logits      = model.predict(img_array)
    predictions = tf.nn.softmax(logits[0]).numpy()

    predicted_idx  = int(np.argmax(predictions))
    confidence     = float(np.max(predictions)) * 100
    predicted_name = CLASS_NAMES[predicted_idx]
    clean_name     = predicted_name.replace('Tomato___', '').replace('_', ' ')

    # ── Deteksi Non-Tomat ──────────────────────────────────────────
    # Jika confidence tertinggi di bawah threshold, model tidak cukup
    # yakin bahwa gambar ini adalah daun tomat → tolak prediksi
    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence, [], None, False, True   # is_non_tomato=True

    # Top-3 prediksi
    top3_idx = np.argsort(predictions)[::-1][:3]
    top3 = [
        {
            'label'     : CLASS_NAMES[i].replace('Tomato___', '').replace('_', ' '),
            'confidence': round(float(predictions[i]) * 100, 2)
        }
        for i in top3_idx
    ]

    description = DISEASE_INFO.get(clean_name, '-')
    is_healthy  = clean_name.lower() == 'healthy'

    return clean_name, round(confidence, 2), top3, description, is_healthy, False

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Tidak ada file yang diunggah.")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="Pilih gambar terlebih dahulu.")

        if not allowed_file(file.filename):
            return render_template('index.html', error="Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction, confidence, top3, description, is_healthy, is_non_tomato = predict_image(filepath)
            image_url = url_for('static', filename=f'uploads/{filename}')

            # Gambar bukan daun tomat → tampilkan pesan penolakan
            if is_non_tomato:
                return render_template(
                    'index.html',
                    error     = f"Maaf, gambar yang Anda kirim bukan daun tomat (keyakinan model hanya {round(confidence, 2)}%). Silakan unggah foto daun tomat yang jelas.",
                    image_url = image_url
                )

            return render_template(
                'index.html',
                prediction  = prediction,
                confidence  = confidence,
                top3        = top3,
                description = description,
                is_healthy  = is_healthy,
                image_url   = image_url
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)