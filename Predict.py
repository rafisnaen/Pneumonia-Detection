from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')
print("Model loaded from model.h5")

def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)[0][0]  # Extract single prediction probability
    if prediction > 0.5:
        confidence = float(prediction * 100)
        return "PNEUMONIA", confidence
    else:
        confidence = float((1 - prediction) * 100)
        return "NORMAL", confidence

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Invalid file format"}), 400

    
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        result, confidence = predict_pneumonia(file_path)
        return jsonify({"prediction": result, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)