from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import pickle

app = Flask(__name__)
CORS(app)

def preprocessing(data):
    # Load the pickle model from file
    with open('C:/Users/10-Me22/Downloads/ExtraTreesClassifier_model90.pkl', 'rb') as file:
        model = pickle.load(file)

    # Preprocess the input data
    df = pd.DataFrame(data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Make predictions
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        predict = model.predict(scaled_data)
        result = "Diabetic" if predict[0] == 1 else "Non-Diabetic"
        return result
    except Exception as e:
        print("Error during prediction:", str(e))
        raise e


@app.route('/')
def index():
    return "Welcome to the Diabetes Prediction API"

@app.route('/predictApi', methods=["POST"])
def api():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'Error': 'Please try again. The data is missing.'})
        
        result = preprocessing(data['df'])
       
        return jsonify({'prediction': result})
    except FileNotFoundError:
        return jsonify({'Error': 'Model file not found.'})
    except pickle.UnpicklingError:
        return jsonify({'Error': 'Error occurred while unpickling the model.'})
    except Exception as e:
        return jsonify({'Error': f'Error occurred during prediction: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'Error': 'Please try again. The data is missing.'})
        
        result = preprocessing(data['df'])
        return jsonify({'prediction': result})
    except FileNotFoundError:
        return jsonify({'Error': 'Model file not found.'})
    except pickle.UnpicklingError:
        return jsonify({'Error': 'Error occurred while unpickling the model.'})
    except Exception as e:
        return jsonify({'Error': f'Error occurred during prediction: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=False)