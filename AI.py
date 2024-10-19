from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import requests
import io

# Initialize Flask app
app = Flask(__name__)

# Google Drive ID of the model (use file ID, not the full link)
MODEL_ID = "18SbkGUH90q0JBmiftyvZB_v4U_DxdEML"
MODEL_PATH = "best_model.keras"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading the model...")
        url = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print(f"Error downloading model: {response.status_code}")

# Preprocessing function
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize to 32x32
    image = np.array(image)
    image = np.stack([image]*3, axis=-1)  # Convert grayscale to RGB by stacking
    image = image.astype('float32') / 255  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model exists locally, if not download it
    download_model()

    # Load the model
    model = load_model(MODEL_PATH)

    # Check if file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    try:
        image = Image.open(file.stream).convert('L')  # Convert to grayscale
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return jsonify({'prediction': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
