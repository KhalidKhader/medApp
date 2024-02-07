from flask import Flask, render_template, request, current_app, redirect
import firebase_admin
from firebase_admin import credentials, db
import pytesseract
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Initialize Firebase app
cred = credentials.Certificate("./service.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://braintumors-69c21-default-rtdb.firebaseio.com/'
})
ref = db.reference('results')

# Load the trained model
model1 = tf.keras.models.load_model('./brain_tumor_classifier_my_model.h5')
model2 = tf.keras.models.load_model('./covid.h5')
model3 = tf.keras.models.load_model('./zhim.h5')

@app.route('/')
def index():
    return render_template('index.html')

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

# Checklists for different tests
checklists = {}




@app.route('/predict_brain_tumor', methods=['POST'])
def predict_brain_tumor():
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    try:
        # Use the current app's root path to dynamically determine the directory
        file_directory = os.path.join(current_app.root_path, 'uploads')
        os.makedirs(file_directory, exist_ok=True)  # Ensure the directory exists
        # Use the full file path to save the file
        file_path = os.path.join(file_directory, file.filename)
        file.save(file_path)
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        # Perform inference
        predictions = model1.predict(img_array)
        predicted_class = np.argmax(predictions[0])    
        confidence_percent = round((np.max(predictions[0]) * 100 ),3)   
        # Interpret the predictions
        class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        predicted_label = class_names[predicted_class]
        print('Predicted Label:', predicted_label)
        print('Confidence Percent:', confidence_percent)

        if predicted_label in ['Glioma Tumor', 'Meningioma Tumor','Pituitary Tumor']:
            class1='injured'
        else:
            class1='not-injured'
        
        checklists['confidence_percent'] = confidence_percent
        checklists['predicted_label'] = predicted_label   
        checklists['for'] = 'brain'     
        return render_template('result.html', result=predicted_label, confidence_percent=confidence_percent,class1=class1, img = img, checklists=checklists)
        
    except Exception as e:
        return 'Error uploading file: ' + str(e)


@app.route('/predict_checklist', methods=['POST'])
def predict_checklist(): 
    # Process checklist items
    checklists['MRI'] = request.form.get('mri')
    checklists['expected_tumor'] = request.form.get('expected_tumor')
    checklists['name'] = request.form.get('name')
    checklists['notes'] = request.form.get('notes')
    # Store checklist results in Firebase
    ref.push({
        'checklists': checklists,
    })    
    return redirect('/')

@app.route('/display_results')
def display_results():
    # Retrieve data from Firebase
    try:
        results = ref.get()
        # Pass the data to the template for rendering
        return render_template('all_brain_results.html', results=results)
    except Exception as e:
        return 'Error ' + str(e)    


@app.route('/predict_covid', methods=['POST'])
def predict_covid():
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    try:
        # Use the current app's root path to dynamically determine the directory
        file_directory = os.path.join(current_app.root_path, 'uploads')
        os.makedirs(file_directory, exist_ok=True)  # Ensure the directory exists
        # Use the full file path to save the file
        file_path = os.path.join(file_directory, file.filename)
        file.save(file_path)
        class_names = ['covid', 'normal', 'pneumonia']

   
        img = image.load_img(file_path, target_size=(96, 96))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        # Perform inference
        predictions = model2.predict(img_array)
        predicted_class = np.argmax(predictions[0])    
        confidence_percent = round((np.max(predictions[0]) * 100 ),3)   
        # Interpret the predictions
        
        predicted_label = class_names[predicted_class]
        print('Predicted Label:', predicted_label)
        print('Confidence Percent:', confidence_percent)
        checklists['confidence_percent'] = confidence_percent
        checklists['predicted_label'] = predicted_label   
        checklists['for'] = 'covid'   
        if predicted_label in ['covid', 'pneumonia']:
            class1='injured'
        else:
            class1='not-injured'
                
        return render_template('result.html', result=predicted_label, confidence_percent=confidence_percent,class1=class1, img = img)
        
    except Exception as e:
        return 'Error uploading file: ' + str(e)

@app.route('/predict_Alzeh', methods=['POST'])
def predict_Alzeh():
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    try:
        # Use the current app's root path to dynamically determine the directory
        file_directory = os.path.join(current_app.root_path, 'uploads')
        os.makedirs(file_directory, exist_ok=True)  # Ensure the directory exists
        # Use the full file path to save the file
        file_path = os.path.join(file_directory, file.filename)
        file.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Perform inference
        predictions = model3.predict(img_array)
        predicted_class = np.argmax(predictions[0])    
        confidence_percent = round((np.max(predictions[0]) * 100 ),3)   
        # Interpret the predictions
        class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
        predicted_label = class_names[predicted_class]
        print('Predicted Label:', predicted_label)
        print('Confidence Percent:', confidence_percent)
        if predicted_label in ['Mild Demented', 'Moderate Demented','Very Mild Demented']:
            class1='injured'
        elif predicted_label == 'Non Demented':
            class1='not-injured'
        checklists['confidence_percent'] = confidence_percent
        checklists['predicted_label'] = predicted_label   
        checklists['for'] = 'Alzeh'       
                
        return render_template('result.html', result=predicted_label, confidence_percent=confidence_percent,class1=class1, img = img)
        
    except Exception as e:
        return 'Error uploading file: ' + str(e)


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)




