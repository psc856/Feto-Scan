# 🍼 Feto-Scan: AI-Powered Fetal Head Circumference Analysis

<div align="center">

![Feto-Scan Banner](https://img.shields.io/badge/Feto--Scan-AI%20Medical%20Analysis-blue?style=for-the-badge&logo=medical&logoColor=white)

**Automatic fetal head segmentation from ultrasound images with precise head circumference measurement**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

</div>

## 🌟 Overview

**Feto-Scan** is a cutting-edge deep learning solution for **automatic fetal head segmentation** from ultrasound images and precise estimation of **head circumference (HC)** in millimeters. The project leverages a state-of-the-art DAG-V-Net architecture for semantic segmentation, followed by advanced post-processing algorithms to calculate accurate head circumference measurements.

### ✨ Key Features

- 🧠 **Advanced AI Model**: DAG-V-Net-based architecture for precise segmentation
- 📏 **Accurate Measurements**: Automatic head circumference calculation in millimeters
- 🌐 **Web Interface**: User-friendly Flask web application
- 📊 **Data Augmentation**: Robust training with augmented dataset
- 🔧 **Easy Integration**: Modular design for easy deployment

## 📂 Project Structure

```
Feto-Scan/
├── 📄 app.py                                    # Flask web application
├── 📄 Create_Mask_Image_For_DataSet.py          # Create segmentation masks
├── 📄 creating_csv_for_augmentedpic.py          # Generate augmented data CSVs
├── 📄 creating_csv_for_dataset.py               # Generate training data CSVs
├── 📄 validation.py                             # Model validation scripts
├── 📄 requirements.txt                          # Python dependencies
├── 📄 README.md                                 # Project documentation
├── 📄 Feto-Scan Report.docx                     # Project report
├── 📄 train_Mask_Augment.csv                    # Augmented mask paths
├── 📄 train_Source_Augment.csv                  # Augmented image paths
├── 📄 Train_X.csv                               # Training images paths
├── 📄 Train_Y.csv                               # Training masks paths
├── 📁 dataset/
│   ├── 📄 test_set_pixel_size.csv               # Test pixel size data
│   ├── 📄 training_set_pixel_size_and_HC.csv    # Training pixel size & HC data
│   ├── 📁 Augmentedmask/                        # Augmented mask images
│   │   ├── 0_1.png
│   │   ├── 0_2.png
│   │   ├── ...
│   │   └── 100_30.png
│   ├── 📁 Augmentedsrc/                         # Augmented source images
│   ├── 📁 test_set/                             # Test ultrasound images
│   ├── 📁 training_set/                         # Original training images
│   └── 📁 training_set_label/                   # Ground-truth masks
├── 📁 datdaprocess/                             # Data processing modules
│   ├── 📄 __init__.py                           # Package initializer
│   ├── 📄 Create_Augmented_Pics.py              # Image augmentation script
│   ├── 📄 utils.py                              # Utility functions
│   └── 📁 Augmentation/                         # Augmentation algorithms
├── 📁 model/
│   └── 📄 DAG_MODEL.ipynb                       # Model training notebook
├── 📁 myenv/                                    # Python virtual environment
├── 📁 static/
│   └── 📁 uploads/                              # Uploaded images storage
└── 📁 templates/
    └── 📄 landing.html                          # Web interface template
```

## 🚀 Quick Start Guide

### 1️⃣ Data Preparation

#### Create Segmentation Masks
```bash
python Create_Mask_Image_For_DataSet.py
```
This script processes the original dataset and creates binary segmentation masks for training.

#### Generate CSV Files for Training
```bash
python creating_csv_for_dataset.py
```
Creates `Train_X.csv` and `Train_Y.csv` with paths to training images and corresponding masks.

### 2️⃣ Data Augmentation

#### Generate Augmented Images
```bash
python datdaprocess/Create_Augmented_Pics.py
```
Applies various augmentation techniques (rotation, scaling, flipping) to increase dataset diversity and stores them in `dataset/Augmentedsrc/` and `dataset/Augmentedmask/`.

#### Create Augmentation CSV Files
```bash
python creating_csv_for_augmentedpic.py
```
Generates CSV files containing paths to augmented images and masks:
- `train_Source_Augment.csv`
- `train_Mask_Augment.csv`

### 3️⃣ Model Training

1. **Open Training Notebook**: Launch `model/DAG_MODEL.ipynb` in Jupyter or Google Colab
2. **Load Data**: Use the generated CSV files (`Train_X.csv`, `Train_Y.csv`, `train_Source_Augment.csv`, `train_Mask_Augment.csv`)
3. **Train the DAG-V-Net model**: Follow the training pipeline in the notebook
4. **Save the trained model**: Export the model for deployment
5. **Validate Results**: Use `validation.py` to test model performance

### 4️⃣ Web Application Deployment

#### Install Dependencies
```bash
# Create virtual environment (optional but recommended)
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Run the Flask App
```bash
python app.py
```

Visit `http://localhost:5000` to access the web interface.

## 🔧 Core Components

### 📱 Web Application (`app.py`)

The Flask web application provides an intuitive interface for fetal head circumference analysis:

```python
# Key Features:
- Image upload and preprocessing
- Real-time prediction using trained model
- Head circumference calculation
- Results visualization through templates/landing.html
```

#### Main Functions:

- **`preprocess_image()`**: Handles image resizing, grayscale conversion, and normalization
- **`predict_head_circumference()`**: Runs inference and calculates circumference from segmentation mask
- **Landing page route**: Manages file uploads and displays results

### 📊 Data Processing Pipeline (`datdaprocess/`)

- **`utils.py`**: Common utility functions for data processing
- **`Create_Augmented_Pics.py`**: Advanced image augmentation algorithms
- **`Augmentation/`**: Specialized augmentation techniques

### 🔍 Model Validation (`validation.py`)

Comprehensive validation suite for model performance evaluation:
- Segmentation accuracy metrics
- Circumference measurement precision
- Cross-validation results

### 🧠 Model Architecture

- **Base Model**: DAG-V-Net (Directed Acyclic Graph V-Net)
- **Input Shape**: 256×256×1 (grayscale images)
- **Output**: Binary segmentation mask
- **Post-processing**: Region properties analysis for circumference calculation

### 📊 Data Processing Pipeline

1. **Image Preprocessing**
   - Resize to 256×256 pixels
   - Convert to grayscale
   - Normalize pixel values (0-1)

2. **Segmentation**
   - Binary thresholding (threshold > 0.5)
   - Connected component analysis

3. **Measurement**
   - Extract region properties
   - Calculate perimeter as head circumference

## 📈 Dataset Information

### Training Data
- **Images**: High-quality ultrasound scans
- **Labels**: Binary segmentation masks
- **Metadata**: Pixel size (mm/pixel) and ground-truth HC measurements

### Augmentation Strategies
- ↻ **Rotation**: ±15 degrees
- 📏 **Scaling**: 0.9-1.1x
- 🔄 **Horizontal flipping**
- 🌈 **Brightness adjustment**
- 📐 **Elastic deformation**

## 🎯 Performance Metrics

- **Segmentation Accuracy**: Dice coefficient, IoU
- **Measurement Precision**: Mean Absolute Error (MAE) in mm
- **Clinical Relevance**: Correlation with ground-truth measurements

## 🛠️ Technical Requirements

### Software Dependencies
```
Python >= 3.8
TensorFlow >= 2.8
Flask >= 2.0
OpenCV >= 4.5
Pillow >= 8.0
scikit-image >= 0.18
numpy >= 1.21
```

### Hardware Requirements
- **GPU**: Recommended for training (CUDA compatible)
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: 2GB for dataset and models

## 🚀 Deployment Options

### Local Development
```bash
git clone https://github.com/your-repo/feto-scan
cd feto-scan
pip install -r requirements.txt
python app.py
```

### Docker Deployment
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

### Cloud Deployment
- **Heroku**: Web app deployment
- **AWS SageMaker**: Model hosting
- **Google Cloud Platform**: Scalable inference

## 📚 Usage Examples

### Web Interface Usage
1. Open browser and navigate to `http://localhost:5000`
2. Upload an ultrasound image (JPG, PNG formats supported)
3. Click "Analyze" to process the image
4. View the predicted head circumference measurement

### Programmatic Usage
```python
from app import preprocess_image, predict_head_circumference
from PIL import Image

# Load and process image
image = Image.open('ultrasound_scan.jpg')
img_batch = preprocess_image(image)

# Get prediction
circumference = predict_head_circumference(img_batch)
print(f"Predicted HC: {circumference:.2f} pixels")
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Medical imaging datasets and research community
- TensorFlow and Keras development teams
- Flask web framework contributors
- Open-source medical imaging libraries

## 📞 Support & Contact

- 📧 **Email**: support@feto-scan.com
- 💬 **Issues**: [GitHub Issues](https://github.com/your-repo/feto-scan/issues)
- 📖 **Documentation**: [Wiki](https://github.com/your-repo/feto-scan/wiki)

---

<div align="center">

**Made with ❤️ for advancing prenatal care through AI**

⭐ **Star this repository if you find it helpful!** ⭐

</div>