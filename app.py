import os
from flask import Flask, request, render_template, url_for
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image
from werkzeug.utils import secure_filename
from skimage.measure import regionprops
import cv2

# Path to the SavedModel folder
MODEL_PATH = os.path.join("Trained_models", "True_DAG_VNet_savedmodel")

print(f"Loading model from {MODEL_PATH}...")
try:
    # Load SavedModel as inference-only using TFSMLayer
    model = tf.keras.Sequential([
        TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
    ])
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

INPUT_SHAPE = (256, 256, 1)  # Adjust input shape for grayscale image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Landing page route
@app.route('/', methods=['GET', 'POST'])
def landing_page():
    circumference = None
    image_url = None
    error_message = None

    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename:
            try:
                # Save the uploaded image
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Preprocess the image
                image = Image.open(file_path)
                img_batch = preprocess_image(image)

                # Make prediction (head circumference)
                circumference = predict_head_circumference(img_batch)

                # Generate the image URL
                image_url = url_for('static', filename=f'uploads/{filename}')
                
                print(f"Prediction result: {circumference}")
                
            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                print(f"Error: {error_message}")
        else:
            error_message = "No image file provided"

    return render_template("landing.html", 
                         image_url=image_url, 
                         circumference=circumference,
                         error_message=error_message)

# Image preprocessing function
def preprocess_image(image):
    try:
        print(f"Original image mode: {image.mode}")
        print(f"Original image size: {image.size}")
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Resize to expected input shape
        image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
        
        # Convert to grayscale
        image = np.array(image.convert('L'))
        
        print(f"Preprocessed image shape before expand: {image.shape}")
        print(f"Image value range: {image.min()} - {image.max()}")
        
        # Add channel dimension
        image = np.expand_dims(image, axis=-1)
        
        # Normalize
        image = image / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        print(f"Final batch shape: {image_batch.shape}")
        print(f"Final value range: {image_batch.min()} - {image_batch.max()}")
        
        return image_batch
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise ValueError(f"Error during image preprocessing: {e}")

# Head circumference prediction function
def predict_head_circumference(img_batch):
    if model is None:
        return "Model not loaded"
    
    try:
        print("Starting prediction...")
        
        # Run inference
        outputs = model(img_batch)
        
        print(f"Model output type: {type(outputs)}")
        
        # Debug: Print model outputs
        if isinstance(outputs, dict):
            print(f"Output keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                print(f"{key} shape: {value.shape}")
                print(f"{key} value range: {value.numpy().min()} - {value.numpy().max()}")
            
            # Try to get main output first, fallback to first output
            if 'main_output' in outputs:
                pred_mask = outputs['main_output'].numpy()
                print("Using main_output")
            else:
                pred_mask = list(outputs.values())[0].numpy()
                print(f"Using first output: {list(outputs.keys())[0]}")
        else:
            print(f"Output shape: {outputs.shape}")
            pred_mask = outputs.numpy()
        
        print(f"Prediction mask shape: {pred_mask.shape}")
        print(f"Prediction mask range: {pred_mask.min()} - {pred_mask.max()}")
        
        # Apply binary thresholding
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
        
        # Get single mask from batch
        if len(pred_mask_binary.shape) == 4:  # (batch, height, width, channels)
            single_mask = pred_mask_binary[0, :, :, 0]
        elif len(pred_mask_binary.shape) == 3:  # (batch, height, width)
            single_mask = pred_mask_binary[0, :, :]
        else:
            single_mask = pred_mask_binary
        
        print(f"Binary mask shape: {single_mask.shape}")
        print(f"Binary mask unique values: {np.unique(single_mask)}")
        print(f"Number of white pixels: {np.sum(single_mask)}")
        
        # Calculate properties using regionprops
        regions = regionprops(single_mask)
        
        print(f"Number of regions found: {len(regions)}")
        
        if regions:
            perimeter = regions[0].perimeter
            area = regions[0].area
            print(f"Region area: {area}")
            print(f"Region perimeter: {perimeter}")
            
            return f"{perimeter:.2f}"
        else:
            return "No region detected"
            
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# Main function to run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)