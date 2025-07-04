from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from werkzeug.utils import secure_filename
from aiVShuman.featureExtraction import extract_features

app = Flask(__name__)
CORS(app)  # Allows frontend to talk to backend

# Load model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limit file size to 5 MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        features = extract_features(filepath)
    except Exception as e:
        return jsonify({'error': f'Error extracting features: {str(e)}'}), 500

    features_normalized = scaler.transform([features])
    prediction = model.predict(features_normalized)[0]

    result = "Real" if prediction == 0 else "AI-generated"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
