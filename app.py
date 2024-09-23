from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your pre-trained model (make sure the path is correct)
model = load_model('models/character_recognition_model.h5')

# Ensure 'uploads' directory exists for storing uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocessing function: resize, grayscale, normalize
def preprocess_image(image_path, target_size=(48, 48)):
    image = cv2.imread(image_path)  # Read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, target_size)  # Resize to target size
    image = image / 255.0  # Normalize to range [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the file to the 'uploads' folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Preprocess the image before prediction
        image = preprocess_image(file_path)
        
        # Make prediction using the loaded model
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)  # Get the predicted class index
        
        # Assuming you have a label encoder to convert index to actual label
        # If not, replace with your own label mapping logic
        labels = ['0', '1', '2', '3', '4','5','6', '7', '8','9','A', 'B', 'C', 'D', 'E','F','G', 'H', 'I', 'J', 'K','L','M', 'N', 'O', 'P', 'Q','R','S', 'T', 'U', 'V', 'W','X','Y','Z']  
        predicted_character = labels[predicted_class]
        
        return render_template('result.html', character=predicted_character)

if __name__ == '__main__':
    app.run(debug=True)
