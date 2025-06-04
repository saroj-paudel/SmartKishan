from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
# model = load_model('models/tomatoes.h5')


classes=['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato__Target_Spot',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

MODEL = tf.keras.models.load_model("models/tomatoes.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']
    img_path = os.path.join('static', img.filename)
    img.save(img_path)
    img_loaded = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_loaded) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = MODEL.predict(img_array)
    class_idx = np.argmax(prediction)
    detected_class=classes[class_idx]
    return f"Disease class predicted: {detected_class}"


if __name__ == '__main__':
    app.run(debug=True)
