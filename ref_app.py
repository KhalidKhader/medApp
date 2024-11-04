from flask import Flask, render_template, request, current_app, redirect, jsonify
import firebase_admin
from firebase_admin import credentials, db
import pytesseract
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from gradientai import Gradient
from flask_cors import CORS
import os

os.environ['GRADIENT_ACCESS_TOKEN'] = "SULXgrNptduvU37FKJK72YFGjlfRjP6S"
os.environ['GRADIENT_WORKSPACE_ID'] = "afb4054d-10b4-4ae6-a404-3f65361bd408_workspace"


app = Flask(__name__)
CORS(app)

# Initialize Firebase and Gradient
try:
    cred = credentials.Certificate("./service.json")
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://braintumors-69c21-default-rtdb.firebaseio.com/'})
    ref = db.reference('results')
except Exception as e:
    app.logger.error(f"Failed to initialize Firebase: {e}")
    raise

gradient = Gradient()

# Load machine learning models securely
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        app.logger.error(f"Failed to load model from {model_path}: {e}")
        raise

model1 = load_model('./brain_tumor_classifier_my_model.h5')
model2 = load_model('./covid.h5')
model3 = load_model('./zhim.h5')

# Helper function to preprocess images
def preprocess_image(file_path, target_size=(224, 224), normalize=True):
    try:
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        if normalize == True:
            img_array /= 255.0  
        return img_array
    except Exception as e:
        app.logger.error(f"Failed to process image: {e}")
        raise

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"Error in {func.__name__}: {e}")
            return render_template('error.html', message=str(e)), 500
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/brain')
def brain():
    return render_template('brain.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/Alzeh')
def Alzeh():
    return render_template('Alzeh.html')   

@app.route('/brain_check')
def brain_check():
    return render_template('checklist_form.html')    

@app.route('/')
@handle_errors
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@handle_errors
def predict():
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Invalid query"}), 400

    with gradient as g:
        base_model = g.get_base_model(base_model_slug="nous-hermes2")
        new_model_adapter = base_model.create_model_adapter(name="test model 3")
        completion = new_model_adapter.complete(query=query, max_generated_token_count=100).generated_output

    return jsonify({"result": completion})

# Simplified prediction handler
def make_prediction(model, img_array, class_names):
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence_percent = round((np.max(predictions[0]) * 100), 3)
        predicted_label = class_names[predicted_class]
        return predicted_label, confidence_percent
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        raise

@app.route('/predict_brain_tumor', methods=['POST'])
@handle_errors
def predict_brain_tumor():
    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('error.html', message='No selected file')

    file_path = save_file(file)
    img_array = preprocess_image(file_path, target_size=(224, 224))
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

    predicted_label, confidence_percent = make_prediction(model1, img_array, class_names)
    
    class1 = 'injured' if predicted_label != 'No Tumor' else 'not-injured'
    checklist_entry = {
        'confidence_percent': confidence_percent,
        'predicted_label': predicted_label,
        'for': 'brain'
    }
    
    # Render the results
    return render_template('result.html', result=predicted_label, confidence_percent=confidence_percent, class1=class1, checklists=checklist_entry)

@app.route('/predict_covid', methods=['POST'])
@handle_errors
def predict_covid():
    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('error.html', message='No selected file')

    file_path = save_file(file)
    img_array = preprocess_image(file_path, target_size=(96, 96), normalize=False)
    class_names = ['covid', 'normal', 'pneumonia']

    predicted_label, confidence_percent = make_prediction(model2, img_array, class_names)
    class1 = 'injured' if predicted_label != 'normal' else 'not-injured'
    
    return render_template('result.html', result=predicted_label, confidence_percent=confidence_percent, class1=class1)

@app.route('/predict_Alzeh', methods=['POST'])
@handle_errors
def predict_Alzeh():
    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('error.html', message='No selected file')

    file_path = save_file(file)
    img_array = preprocess_image(file_path, target_size=(128, 128), normalize=False)
    class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

    predicted_label, confidence_percent = make_prediction(model3, img_array, class_names)
    class1 = 'injured' if predicted_label != 'Non Demented' else 'not-injured'
    
    return render_template('result.html', result=predicted_label, confidence_percent=confidence_percent, class1=class1)

# Function to save uploaded file securely
def save_file(file):
    try:
        file_directory = os.path.join(current_app.root_path, 'uploads')
        os.makedirs(file_directory, exist_ok=True)
        file_path = os.path.join(file_directory, file.filename)
        file.save(file_path)
        return file_path
    except Exception as e:
        app.logger.error(f"Failed to save file: {e}")
        raise

# Firebase Integration - Storing Results
@app.route('/predict_checklist', methods=['POST'])
@handle_errors
def predict_checklist():
    checklist_data = {
        'MRI': request.form.get('mri'),
        'expected_tumor': request.form.get('expected_tumor'),
        'name': request.form.get('name'),
        'notes': request.form.get('notes')
    }

    try:
        ref.push({'checklists': checklist_data})
        return redirect('/')
    except Exception as e:
        app.logger.error(f"Failed to save checklist to Firebase: {e}")
        return render_template('error.html', message="Failed to save data")

@app.route('/display_results')
@handle_errors
def display_results():
    try:
        results = ref.get()
        return render_template('all_brain_results.html', results=results)
    except Exception as e:
        app.logger.error(f"Failed to retrieve results: {e}")
        return render_template('error.html', message="Failed to retrieve results")

if __name__ == '__main__':
    app.run(debug=True)
