import os
import pandas as pd

# Define directories
src_dir = r"C:\Users\Asus\Documents\Feto-Scan\dataset\Augmentedsrc"
mask_dir = r"C:\Users\Asus\Documents\Feto-Scan\dataset\Augmentedmask"

# Collect file paths
src_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".png")]
mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]

# Ensure both lists are sorted to maintain consistent pairing
src_paths.sort()
mask_paths.sort()

# Verify same number of source and mask images
assert len(src_paths) == len(mask_paths), "Mismatch between source and mask images!"

# Create DataFrames
src_df = pd.DataFrame(src_paths, columns=["image_path"])
mask_df = pd.DataFrame(mask_paths, columns=["mask_path"])

# Save to CSV
src_df.to_csv("train_Source_Augment.csv", index=False)
mask_df.to_csv("train_Mask_Augment.csv", index=False)
