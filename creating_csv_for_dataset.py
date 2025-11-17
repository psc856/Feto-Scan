import os
import csv

def create_csv(dataset_root, out_x, out_y):
    img_dir = os.path.join(dataset_root, "training_set")
    mask_dir = os.path.join(dataset_root, "training_set_label")

    images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    if len(images) != len(masks):
        print("Warning: Number of images and masks do not match!")

    with open(out_x, "w", newline="") as fx, open(out_y, "w", newline="") as fy:
        writer_x = csv.writer(fx)
        writer_y = csv.writer(fy)

        for img, mask in zip(images, masks):
            img_path = os.path.abspath(os.path.join(img_dir, img))
            mask_path = os.path.abspath(os.path.join(mask_dir, mask))

            writer_x.writerow([img_path])
            writer_y.writerow([mask_path])

    print(f"CSV created:\n - {out_x}\n - {out_y}")


if __name__ == "__main__":
    dataset_root = r"C:\Users\Asus\Documents\Feto-Scan\dataset"
    create_csv(dataset_root,
               r"C:\Users\Asus\Documents\Feto-Scan\Train_X.csv",
               r"C:\Users\Asus\Documents\Feto-Scan\Train_Y.csv")
