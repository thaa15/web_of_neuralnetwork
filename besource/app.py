# from flask_sqlalchemy import SQLAlchemy
from pydoc import classname
from datetime import datetime
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
import base64
import sys
import os
import cv2
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)
CORS(app)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
# db = SQLAlchemy(app)

# Load the model
# model = pickle.load(open('model.pkl','rb'))


def convertImage(imgData1):
    new_data = imgData1.replace('data:image/png;base64,', '')
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(new_data))

class_names = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A",
            11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K",
            21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
            31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a", 37: "b", 38: "c", 39: "d", 40: "e",
            41: "f", 42: "g", 43: "h", 44: "i", 45: "j", 46: "k", 47: "l", 48: "m", 49: "n", 50: "o",
            51: "p", 52: "q", 53: "r", 54: "s", 55: "t", 56: "u", 57: "v", 58: "w", 59: "x", 60: "y",
            61: "z"}

@app.route('/api', methods=['POST'])
@cross_origin()
def predict():
    imgData = request.get_json(force=True)
    convertImage(imgData['img'])
    img = Image.open('output.png')
    img_array = image.img_to_array(img)
    im_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    dim = (96, 96)
    resized = cv2.resize(im_rgb, dim, interpolation=cv2.INTER_AREA)
    resized = np.expand_dims(resized,axis=0)
    resized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))
    try:
        graph = tf.compat.v1.get_default_graph()
        with graph.as_default():
            json_file = open('model.json','r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            loaded_model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(0.001),metrics=['accuracy'])
            out = loaded_model.predict(resized)
            response = class_names[np.argmax(out)]
            return str(response)
    except:
        print("Error")
        return "Nakal ya..."


if __name__ == '__main__':
    app.run(debug=True)
