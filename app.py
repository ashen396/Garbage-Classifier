from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)
model = tf.keras.models.load_model('model/model.h5')
CLASSES = ["plastic", "organic", "metal"]
IMG_SIZE = 256
DATA_DIR = "./data" 

def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = preprocess_image(file)
            pred = model.predict(img)
            class_index = np.argmax(pred, axis=1)[0]
            prediction = CLASSES[class_index]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
