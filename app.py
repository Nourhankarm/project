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
    prediction = rf.predict([features])
    return jsonify({'diabetes_prediction': int(prediction[0])})
        

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


