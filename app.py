import os
from flask import Flask, request, render_template, url_for, flash, redirect
import numpy as np
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
from skimage.measure import regionprops
import cv2

# Load the pre-trained model for fetal head circumference prediction
model = tf.keras.models.load_model(r"C:\Users\vcsma\Downloads\DAG-V-Net-master\DAG_VNet_model.keras")  

INPUT_SHAPE = (256, 256, 1)  # Adjust input shape for grayscale image

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Directory to save uploaded images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Landing page route

@app.route('/', methods=['GET', 'POST'])
def landing_page():
    circumference = None
    image_url = None
    error_message = None

    if request.method == 'POST':
        file = request.files['image']  # Use 'image' as per form input field name
        if file:
            # Save the uploaded image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            image = Image.open(file)
            img_batch = preprocess_image(image)

            # Make prediction (head circumference)
            circumference = predict_head_circumference(img_batch)

            # Generate the image URL
            image_url = url_for('static', filename=f'uploads/{filename}')
            # Pass the image URL, prediction, and confidence to the template
            return render_template("landing.html", image_url=image_url, circumference=circumference)

    return render_template("landing.html", circumference=None)



# Image preprocessing function
def preprocess_image(image):
    try:
        if image.mode == 'RGBA':
            image = image.convert('RGB')  # Convert RGBA to RGB if needed
        image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))  # Resize to the expected input shape
        image = np.array(image.convert('L'))  # Convert to grayscale
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = image / 255.0  # Normalize
        return np.expand_dims(image, axis=0)
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {e}")


# Head circumference prediction function
def predict_head_circumference(img_batch):
    try:
        # Make prediction with the model
        pred_mask = model.predict(img_batch)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)[0, :, :, 0]  # Apply binary thresholding

        # Calculate properties using regionprops (find perimeter for circumference)
        regions = regionprops(pred_mask)
        if regions:
            return regions[0].perimeter  # Return the perimeter as the predicted circumference
        else:
            raise ValueError("No region detected in the image")
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")


# Main function to run the app
if __name__ == "__main__":
    app.run(debug=True)
