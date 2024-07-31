from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('../Output_files/mnist_cnn_model.h5')

def prepare_image(image):
    # Convert the image to grayscale, resize to 28x28, and normalize
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/')
def upload_image():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        image = Image.open(file)
        processed_image = prepare_image(image)

        # Get the model's prediction
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)

        return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
