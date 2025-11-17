
import sys
import os

# Add the directory containing the datdaprocess module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datdaprocess.Augmentation.ImageAugmentation import DataAug

aug = DataAug(rotation=20, width_shift=0.01, height_shift=0.01, rescale=1.1)
aug.DataAugmentation(r"C:\Users\Asus\Documents\Feto-Scan\Train_X.csv", r"C:\Users\Asus\Documents\Feto-Scan\Train_Y.csv", 30,
                     path=r"C:\Users\Asus\Documents\Feto-Scan\dataset\Augmented")