"""
Model Evaluation Script
Run this directly with: python evaluate_model.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import cv2
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Model Evaluation Script")
print("=" * 60)
print("\nImporting TensorFlow (this may take a moment)...")

import tensorflow as tf
from tensorflow import keras

print(f"✅ TensorFlow {tf.__version__} loaded successfully")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Import visualization libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from skimage.measure import regionprops, label
from sklearn.model_selection import train_test_split

print("✅ All libraries imported successfully\n")

# Configuration
class Config:
    MODEL_PATH = r"C:\Users\Asus\Documents\Feto-Scan\Trained_models\True_DAG_VNet_savedmodel"
    IMAGE_CSV = r"C:\Users\Asus\Documents\Feto-Scan\Train_X.csv"
    MASK_CSV = r"C:\Users\Asus\Documents\Feto-Scan\Train_Y.csv"
    OUTPUT_DIR = r"C:\Users\Asus\Documents\Feto-Scan\Model_Evaluation_Results"
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_SAMPLES_TO_VISUALIZE = 10
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

config = Config()

# Create output directory
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("Configuration:")
print(f"  • Model: {config.MODEL_PATH}")
print(f"  • Output: {config.OUTPUT_DIR}")
print(f"  • Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}\n")

# Load model
print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

try:
    # Try loading with TensorFlow directly (for SavedModel format)
    print(f"Loading SavedModel from: {config.MODEL_PATH}")
    model = tf.saved_model.load(config.MODEL_PATH)
    
    # Get the inference function
    infer = model.signatures['serving_default']
    
    print("✅ Model loaded successfully!")
    print(f"  • Model type: TensorFlow SavedModel")
    print(f"  • Signature: {list(model.signatures.keys())}")
    
    # Get input/output info
    input_name = list(infer.structured_input_signature[1].keys())[0]
    output_names = list(infer.structured_outputs.keys())
    
    print(f"  • Input tensor: {input_name}")
    print(f"  • Output tensors: {output_names}")
    print(f"  • Input shape: {infer.structured_input_signature[1][input_name].shape}\n")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nTrying alternative loading method...")
    try:
        # Try with tf_keras (Keras 2 compatibility)
        import tf_keras
        model = tf_keras.models.load_model(config.MODEL_PATH)
        infer = None  # Use model.predict directly
        input_name = None
        output_names = None
        print("✅ Model loaded with tf_keras (Keras 2 compatibility)")
        print(f"  • Total parameters: {model.count_params():,}")
        print(f"  • Input shape: {model.input_shape}")
        print(f"  • Output shape: {model.output_shape}\n")
    except Exception as e2:
        print(f"❌ Alternative loading also failed: {e2}")
        sys.exit(1)

# Load dataset
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

try:
    image_paths = pd.read_csv(config.IMAGE_CSV, header=None)[0].values
    mask_paths = pd.read_csv(config.MASK_CSV, header=None)[0].values
    print(f"✅ Total samples: {len(image_paths)}")
    
    # Split data
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        image_paths, mask_paths, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    print(f"📊 Test set size: {len(X_test)} samples\n")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)

# Preprocessing functions
def preprocess_image(img_path):
    """Preprocess image for model input"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def preprocess_mask(mask_path):
    """Preprocess mask for evaluation"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    mask = cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    mask = (mask > 0.5).astype(np.float32)
    return mask

# Metrics functions
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice

def iou_score(y_true, y_pred, smooth=1e-6):
    """IoU Score"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def pixel_accuracy(y_true, y_pred):
    """Pixel-wise accuracy"""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    correct = np.sum(y_true_f == y_pred_f)
    total = len(y_true_f)
    return correct / total

def precision_recall_f1(y_true, y_pred):
    """Calculate precision, recall, F1"""
    y_true_f = y_true.flatten().astype(int)
    y_pred_f = y_pred.flatten().astype(int)
    
    tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
    fp = np.sum((y_true_f == 0) & (y_pred_f == 1))
    fn = np.sum((y_true_f == 1) & (y_pred_f == 0))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision, recall, f1

def calculate_circumference(pred_mask):
    """Calculate circumference"""
    labeled_mask = label(pred_mask)
    regions = regionprops(labeled_mask)
    
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        return largest_region.perimeter
    else:
        return 0

print("✅ Evaluation functions defined\n")

# Evaluate model
print("=" * 60)
print("EVALUATING MODEL ON TEST SET")
print("=" * 60)

all_metrics = {
    'dice': [],
    'iou': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'circumference_pred': [],
    'circumference_true': []
}

predictions_data = []

print(f"\nProcessing {len(X_test)} test images...")
print("This may take a few minutes...\n")

for idx, (img_path, mask_path) in enumerate(zip(X_test, y_test)):
    try:
        # Load and preprocess
        image = preprocess_image(img_path)
        true_mask = preprocess_mask(mask_path)
        
        # Predict
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)
        
        # Use appropriate prediction method
        if infer is not None:
            # SavedModel inference
            pred_result = infer(tf.constant(image_batch))
            # Get the main output (first output if multiple)
            pred_mask_raw = list(pred_result.values())[0].numpy()
        else:
            # Keras model inference
            pred_mask_raw = model.predict(image_batch, verbose=0)
        
        # Handle multiple outputs
        if isinstance(pred_mask_raw, list):
            pred_mask_raw = pred_mask_raw[0]
        
        pred_mask_raw = pred_mask_raw[0, :, :, 0]
        pred_mask = (pred_mask_raw > 0.5).astype(np.float32)
        
        # Calculate metrics
        dice = dice_coefficient(true_mask[:, :, 0], pred_mask)
        iou = iou_score(true_mask[:, :, 0], pred_mask)
        accuracy = pixel_accuracy(true_mask[:, :, 0], pred_mask)
        precision, recall, f1 = precision_recall_f1(true_mask[:, :, 0], pred_mask)
        
        circ_pred = calculate_circumference(pred_mask)
        circ_true = calculate_circumference(true_mask[:, :, 0])
        
        # Store metrics
        all_metrics['dice'].append(dice)
        all_metrics['iou'].append(iou)
        all_metrics['accuracy'].append(accuracy)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['f1'].append(f1)
        all_metrics['circumference_pred'].append(circ_pred)
        all_metrics['circumference_true'].append(circ_true)
        
        # Store for visualization
        if idx < config.NUM_SAMPLES_TO_VISUALIZE:
            predictions_data.append({
                'image': image,
                'true_mask': true_mask[:, :, 0],
                'pred_mask': pred_mask,
                'pred_mask_raw': pred_mask_raw,
                'dice': dice,
                'iou': iou,
                'image_path': img_path
            })
        
        # Progress update
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(X_test)} images...")
            
    except Exception as e:
        print(f"  ⚠️ Error processing {img_path}: {e}")
        continue

print(f"\n✅ Evaluation complete! Processed {len(all_metrics['dice'])} images successfully.\n")

# Calculate statistics
print("=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

metrics_stats = {}
for metric_name, values in all_metrics.items():
    if len(values) > 0:
        metrics_stats[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }

# Display results
print("\n🎯 Segmentation Metrics:")
print("-" * 60)
for metric in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']:
    if metric in metrics_stats:
        stats = metrics_stats[metric]
        print(f"{metric.upper():12s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"{'':12s}  (min: {stats['min']:.4f}, max: {stats['max']:.4f}, median: {stats['median']:.4f})")

print("\n📏 Circumference Analysis:")
print("-" * 60)
circ_pred = np.array(all_metrics['circumference_pred'])
circ_true = np.array(all_metrics['circumference_true'])
circ_error = np.abs(circ_pred - circ_true)
circ_error_percent = (circ_error / (circ_true + 1e-6)) * 100

print(f"Mean Absolute Error: {np.mean(circ_error):.2f} pixels")
print(f"Mean Error Percentage: {np.mean(circ_error_percent):.2f}%")
print(f"Std Dev: {np.std(circ_error):.2f} pixels")
print(f"Max Error: {np.max(circ_error):.2f} pixels\n")

# Save metrics to JSON
metrics_summary = {
    'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'model_path': config.MODEL_PATH,
    'test_samples': len(all_metrics['dice']),
    'metrics': metrics_stats,
    'circumference': {
        'mae': float(np.mean(circ_error)),
        'mae_percent': float(np.mean(circ_error_percent)),
        'std': float(np.std(circ_error)),
        'max_error': float(np.max(circ_error))
    }
}

json_path = os.path.join(config.OUTPUT_DIR, 'evaluation_metrics.json')
with open(json_path, 'w') as f:
    json.dump(metrics_summary, f, indent=4)

print(f"✅ Metrics saved to: {json_path}\n")

# Generate visualizations
print("=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)
print("\nCreating plots...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Metrics distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Performance Metrics Distribution', fontsize=16, fontweight='bold')

metrics_to_plot = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    values = all_metrics[metric]
    
    ax.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
    ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.3f}')
    
    ax.set_xlabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{metric.upper()} Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, 'metrics_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Metrics distribution plot saved")

# 2. Circumference analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(all_metrics['circumference_true'], all_metrics['circumference_pred'], alpha=0.6, s=30)
axes[0].plot([0, max(all_metrics['circumference_true'])], 
             [0, max(all_metrics['circumference_true'])], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('True Circumference (pixels)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Predicted Circumference (pixels)', fontweight='bold', fontsize=12)
axes[0].set_title('Circumference: Predicted vs True', fontweight='bold', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

circ_errors = circ_pred - circ_true
axes[1].hist(circ_errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1].axvline(np.mean(circ_errors), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(circ_errors):.2f}')
axes[1].set_xlabel('Prediction Error (pixels)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[1].set_title('Circumference Error Distribution', fontweight='bold', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

abs_errors = np.abs(circ_errors)
axes[2].hist(abs_errors, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[2].axvline(np.mean(abs_errors), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(abs_errors):.2f}')
axes[2].set_xlabel('Absolute Error (pixels)', fontweight='bold', fontsize=12)
axes[2].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[2].set_title('Absolute Error Distribution', fontweight='bold', fontsize=13)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, 'circumference_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Circumference analysis plot saved")

# 3. Sample predictions
num_samples = min(config.NUM_SAMPLES_TO_VISUALIZE, len(predictions_data))
fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
if num_samples == 1:
    axes = axes.reshape(1, -1)

fig.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold', y=0.995)

for i, pred_data in enumerate(predictions_data[:num_samples]):
    axes[i, 0].imshow(pred_data['image'][:, :, 0], cmap='gray')
    axes[i, 0].set_title('Input Image', fontweight='bold')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(pred_data['true_mask'], cmap='gray')
    axes[i, 1].set_title('Ground Truth', fontweight='bold')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(pred_data['pred_mask_raw'], cmap='hot', vmin=0, vmax=1)
    axes[i, 2].set_title('Prediction (Probability)', fontweight='bold')
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(pred_data['pred_mask'], cmap='gray')
    axes[i, 3].set_title(f"Binary Prediction\nDice: {pred_data['dice']:.3f}", fontweight='bold')
    axes[i, 3].axis('off')
    
    overlay = pred_data['image'][:, :, 0].copy()
    overlay_color = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    overlay_color[:, :, 0] = overlay
    overlay_color[:, :, 1] = overlay
    overlay_color[:, :, 2] = overlay
    
    overlay_color[:, :, 1] += pred_data['pred_mask'] * 0.5
    overlay_color[:, :, 0] += pred_data['true_mask'] * 0.3
    overlay_color = np.clip(overlay_color, 0, 1)
    
    axes[i, 4].imshow(overlay_color)
    axes[i, 4].set_title(f"Overlay\nIoU: {pred_data['iou']:.3f}", fontweight='bold')
    axes[i, 4].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, 'predictions_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Predictions visualization saved")

# Generate text report
report_path = os.path.join(config.OUTPUT_DIR, 'evaluation_report.txt')

with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {config.MODEL_PATH}\n")
    f.write(f"Test Samples: {len(all_metrics['dice'])}\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("SEGMENTATION METRICS\n")
    f.write("-" * 80 + "\n\n")
    
    for metric in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']:
        if metric in metrics_stats:
            stats = metrics_stats[metric]
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Mean:   {stats['mean']:.4f}\n")
            f.write(f"  Std:    {stats['std']:.4f}\n")
            f.write(f"  Min:    {stats['min']:.4f}\n")
            f.write(f"  Max:    {stats['max']:.4f}\n")
            f.write(f"  Median: {stats['median']:.4f}\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("CIRCUMFERENCE PREDICTION\n")
    f.write("-" * 80 + "\n\n")
    
    f.write(f"Mean Absolute Error: {np.mean(circ_error):.2f} pixels\n")
    f.write(f"Mean Error Percentage: {np.mean(circ_error_percent):.2f}%\n")
    f.write(f"Std Dev: {np.std(circ_error):.2f} pixels\n")
    f.write(f"Max Error: {np.max(circ_error):.2f} pixels\n\n")
    
    f.write("=" * 80 + "\n")

print("  ✅ Comprehensive report saved\n")

print("=" * 60)
print("EVALUATION COMPLETE!")
print("=" * 60)
print(f"\n📂 All results saved to: {config.OUTPUT_DIR}/")
print("\nGenerated files:")
print("  1. evaluation_metrics.json - Detailed metrics")
print("  2. metrics_distribution.png - Metric distributions")
print("  3. circumference_analysis.png - Circumference predictions")
print("  4. predictions_visualization.png - Sample predictions")
print("  5. evaluation_report.txt - Text report")
print("\n" + "=" * 60)
