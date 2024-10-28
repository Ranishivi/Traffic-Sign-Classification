from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
from PIL import Image
import pandas as pd
import io

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load labels from signname.csv
labels_df = pd.read_csv('signname.csv')
labels = labels_df.set_index('ClassId')['SignName'].to_dict()

app = Flask(__name__)

def prepare_image(image):
    # Resize and normalize the image for the model input
    image = np.array(image).astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/classify", methods=["POST"])
def classify():
    file = request.files['image']
    if file:
        # Open the image and prepare it for prediction
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image)

        # Predict the class
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = labels.get(predicted_class, "Unknown Class")

        # Render the result on the same page
        return render_template('home.html', result=result)
    return render_template('home.html', result="No image uploaded")

if __name__ == "__main__":
    app.run(debug=True)
