from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

app = Flask(__name__)
CORS(app)

def preprossing(image):
    img_array = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    plt.imshow(RGBImg)
    image_array = np.array(RGBImg) / 255.0
    new_model = tf.keras.models.load_model("C:/Users/10-Me22/Downloads/eye_disease_model (5).h5")
    predict = new_model.predict(np.array([image_array]))
    per = np.argmax(predict, axis=1)
    if per == 1:
        return 'No DR'
    else:
        return 'DR'


@app.route('/')
def index():
    return "Welcome to the Eye Disease Prediction API"

@app.route('/predictApi', methods=["POST"])
def api():
    try:
        if 'fileup' not in request.files:
            return jsonify({'Error': 'Please try again. The Image does not exist.'})
        image = request.files.get('fileup')
        
        result = preprossing(image)
       
        return jsonify({'prediction': result})
    except:
        return jsonify({'Error': 'Error occurred during prediction.'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files.get('fileup')
        
        result = preprossing(image)
        return jsonify({'prediction': result})
    except:
        return jsonify({'Error': 'Error occurred during prediction.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)