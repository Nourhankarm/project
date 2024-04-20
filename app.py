import pickle
with open('rf.pkl', 'wb') as file:
    pickle.dump(rf, file)
from werkzeug.wrappers import Request, Response
from flask import Flask
#import tensorflow as tf
from flask import request
from flask import jsonify
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model (make sure the path is accessible from your Jupyter Notebook)
model = pickle.load(open('rf.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data["gender"])
    features = [data['gender'], data['age'], data['hypertension'], data['heart_disease'],
                data['smoking_history'], data['bmi'], data['HbA1c_level'], data['blood_glucose_level']]
    prediction = rf.predict([features])
    return jsonify({'diabetes_prediction': int(prediction[0])})

from werkzeug.serving import run_simple
run_simple('localhost', 8000, app)
