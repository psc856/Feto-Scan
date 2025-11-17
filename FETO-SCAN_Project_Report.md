# FETO-SCAN: AUTOMATED FETAL HEAD SEGMENTATION SYSTEM
### Deep Learning-Based Ultrasound Image Analysis for Circumference Measurement

---

**Project Team:**
Prashant Chauhan – Enrollment No: [Add Number]  
Balaji – Enrollment No: [Add Number]  
Pratayaksh Tyagi – Enrollment No: [Add Number]

**Submission Date:** November 18, 2025

---

## ABSTRACT
This project presents FETO-SCAN, an automated system for fetal head segmentation and circumference measurement in 2D ultrasound images using a custom DAG-VNet architecture. Achieving a Dice coefficient of 0.7706 and pixel accuracy of 87.27% on 200 test images, the system offers clinically acceptable performance. The framework leverages deep supervision, residual connections, and multi-scale attention fusion to handle ultrasound-specific challenges like speckle noise and boundary ambiguity. Deployment is realized through a Flask-based web application suitable for clinical screening.

---

## 1. INTRODUCTION
### 1.1 Background & Motivation
Head circumference (HC) is vital in obstetric care for monitoring fetal growth, estimating gestational age, and detecting abnormalities. Traditional manual measurement is time-consuming and suffers from inter-observer variability, making automation crucial. The goal is to minimize errors while streamlining clinical workflow.

### 1.2 Objectives
- **Primary:** Accurate segmentation (Dice > 0.75), real-time circumference measurement, web-based deployment.
- **Secondary:** Address ultrasound imaging challenges, achieve inference <2 seconds, and provide comprehensive metrics.

---

## 2. METHODOLOGY
### 2.1 Dataset & Augmentation
- **Dataset:** 999 augmented grayscale ultrasound images, size 256×256 pixels.
- **Train/Test Split:** 799/200 images (80/20).
- **Augmentation:** Rotation, flipping, brightness, contrast, elastic deformation.

### 2.2 DAG-VNet Architecture
![DAG-VNet Architecture](generated_image:6)

- Residual blocks for deep feature learning.
- Multi-scale feature fusion via DAG connections & attention.
- Deep supervision with three output heads for robust training.

**Parameters:** ~10-15 million, Input: (256, 256, 1), Output: 3×(256, 256, 1)

---

## 3. TRAINING & DEPLOYMENT
- **Training:** 25 epochs, batch size 8, Adam optimizer, weighted BCE loss (main: 1.0, aux1: 0.5, aux2: 0.25).
- **Framework:** TensorFlow 2.16.2, Keras on CPU.
- **Deployment:** Flask web server with REST API, supports real-time inference (<1 sec/image).

---

## 4. RESULTS
### 4.1 Quantitative Metrics
**Segmentation Performance (200 Test Images):**
| Metric      | Mean    | Std     | Min    | Max    | Median | Clinical Threshold |
|------------|---------|---------|--------|--------|--------|--------------------|
| Dice       | 0.7706  | 0.1243  | 0.2923 | 0.9596 | 0.7983 | >0.75              |
| IoU        | 0.6418  | 0.1490  | 0.1711 | 0.9223 | 0.6643 | >0.60              |
| Accuracy   | 0.8727  | 0.0540  | 0.7515 | 0.9777 | 0.8782 | >0.85              |
| Precision  | 0.8282  | 0.2140  | 0.1711 | 1.0000 | 0.9191 | >0.80              |
| Recall     | 0.7808  | 0.1341  | 0.5329 | 1.0000 | 0.7858 | >0.75              |
| F1         | 0.7706  | 0.1243  | 0.2923 | 0.9596 | 0.7983 | -                  |

Circumference prediction:
- Mean Absolute Error: 93.05 pixels (22.79%)
- Std Dev: 82.74 pixels
- Max Error: 385.57 pixels

---

### 4.2 Performance Distribution & Visual Results
#### Metrics Distribution
![Segmentation Metrics Distribution](attached_image:3)

#### Circumference Analysis
![Circumference Error Analysis](attached_image:1)

#### Qualitative Results
![Sample Predictions and Overlays](attached_image:2)

---

## 5. DISCUSSION
- **Strengths:** Robust segmentation with clinical grade Dice (>0.75), high precision, handles both synthetic/real ultrasound, fast deployment.
- **Limitations:** Circumference error (>22%) needs reduction for clinical diagnostics, dataset size limited, some outlier cases (<0.5 Dice).

---

## 6. FUTURE WORK
- Advanced data augmentation, longer training, new loss functions (Focal/Tversky), post-processing refinements, pixel-to-mm calibration, ellipse fitting.
- Expand dataset, uncertainty quantification, multi-task learning, clinical trials, and mobile/3D/streaming extensions.

---

## 7. CONCLUSION
FETO-SCAN delivers fast, reproducible fetal head segmentation and measurement with clinically acceptable segmentation accuracy, reducing manual effort and variability. Improvements in measurement accuracy and robust validation will drive its path toward clinical adoption.

---

## APPENDIX
### A. Software Specs
- tensorflow==2.16.2, opencv-python==4.11, numpy==1.24.3, pandas==2.3.3, matplotlib==3.10.7, seaborn==0.13.2, scikit-learn==1.6.1, scikit-image==0.24, Flask==3.0.3

### B. Directory Structure
```
Feto-Scan/
├── app.py
├── model/
│   ├── DAG_MODEL_updated.ipynb
│   └── evaluate_model.py
├── Trained_models/
│   └── True_DAG_VNet_savedmodel/
├── dataset/
│   ├── Augmentedsrc/
│   └── Augmentedmask/
├── Model_Evaluation_Results/
├── static/uploads/
├── templates/landing.html
├── Train_X.csv
├── Train_Y.csv
└── requirements.txt
```

---

Version: 1.0 · Last Updated: Nov 18, 2025 · Pages: 8 · Classification: Academic Project
