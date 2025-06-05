from flask import Flask, render_template, request
import tensorflow as tf
# from tensorflow.keras.preprocessing import image
import numpy as np
import os
from io import BytesIO
from PIL import Image

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


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['file']
    img_bytes = img.read()
    img_loaded = read_file_as_image(img_bytes)
    img_resized = Image.fromarray(img_loaded).resize((224, 224))  # Adjust size as per your model input
    img_array = np.array(img_resized) / 255.0  # Normalize if model expects normalized input
    img_array = np.expand_dims(img_array, axis=0)
    prediction = MODEL.predict(img_array)
    class_idx = np.argmax(prediction)
    detected_class = classes[class_idx]
    return f"Disease class predicted: {detected_class}, Accuracy:{np.max(prediction)}"


if __name__ == '__main__':
    app.run(debug=True)
