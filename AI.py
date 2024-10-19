from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import requests

# Initialize Flask app
app = Flask(__name__)

# URL of the model hosted on Google Drive (or another service)
MODEL_URL = "https://drive.google.com/file/d/18SbkGUH90q0JBmiftyvZB_v4U_DxdEML/view?usp=drive_link"

# Download model if not already present
def download_model():
    model_path = 'best_model.keras'
    if not os.path.exists(model_path):
        print("Downloading the model...")
        r = requests.get(MODEL_URL, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
        print("Download complete.")

# Preprocessing function
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image)
    image = tf.image.grayscale_to_rgb(image)
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model exists locally, if not download it
    download_model()

    # Load the model
    model = load_model('best_model.keras')

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    image = Image.open(file.stream).convert('L')  # Convert to grayscale
    preprocessed_image = preprocess_image(image)
    
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
