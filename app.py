from absl import app, logging
import numpy as np
import pickle

from flask import Flask, request, jsonify, abort
import os

# Initialize Flask application
app = Flask(__name__)

# Load the model from a pickle file
model_file_path = 'diabetes_model.sav'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)


# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def get_detections():
    try:
        data = request.json
        print(data)
        gender = data['gender']
        age = data['age']
        hypertension = data['hypertension']
        heart_disease =data['heart_disease']
        smoking_history = data['smoking_history']
        bmi = data['bmi']
        HbA1c_level = data['HbA1c_level']
        blood_glucose_level = data['blood_glucose_level'] 
        print(f"data received: gender= {data['gender']}, age= {age}, hypertension= {hypertension}, heart_disease= {heart_disease}, smoking_history= {data['smoking_history']}, bmi= {bmi}, HbA1c_level= {HbA1c_level}, blood_glucose_level= {blood_glucose_level}")
        #  'gender' and 'smoking_history' are categorical and need to be encoded
       
        #encoded_smoking_history = encoder.transform([smoking_history])[0]
        print(encoded_smoking_history)
        if(gender=="Male"):
            encoded_gender=1
        elif (gender=="Female"):
            encoded_gender=0
        else:
            encoded_gender=2
        
        if(smoking_history=="never"):
            encoded_gender=0
        elif (gender=="ever"):
            encoded_gender=1
        elif (gender=="current"):
            encoded_gender=2
       elif (gender=="not current"):
            encoded_gender=3
        else:
            encoded_gender=4

        # Log received data
        print(f"data received: gender= {data['gender']}, age= {age}, hypertension= {hypertension}, heart_disease={heart_disease}, smoking_history= {data['smoking_history']}, bmi= {bmi}, HbA1c_level= {HbA1c_level}, blood_glucose_level= {blood_glucose_level}")

        # Prepare the model input by replacing 'gender' and 'smoking_history' with their encoded forms
        model_input = np.array([[encoded_gender, age, hypertension, heart_disease, encoded_smoking_history, bmi, HbA1c_level, blood_glucose_level]])

        # Make prediction
        prediction = model.predict(model_input)

        # Log and return the prediction
        print(f"prediction: {prediction}")
        return jsonify({"prediction": int(prediction[0])})

    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
