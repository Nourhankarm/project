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
    data = request.get_json(force=True)
    print(data["gender"])
    features = [data['gender'], data['age'], data['hypertension'], data['heart_disease'],
                data['smoking_history'], data['bmi'], data['HbA1c_level'], data['blood_glucose_level']]
    prediction = model.predict([features])
    print(f"prediction: {prediction}")
    return jsonify({'diabetes_prediction': int(prediction[0])})
  



if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)


