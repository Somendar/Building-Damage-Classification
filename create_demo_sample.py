import os
import cv2
import pandas as pd

# ==========================
# PATHS
# ==========================

CSV_PATH = "balanced_dataset.csv"
IMAGE_FOLDER = "train/images"
OUTPUT_FOLDER = "sample_data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(CSV_PATH)

classes = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed"
]

padding = 30
samples_per_class = 10

# ==========================
# CREATE DEMO SAMPLES
# ==========================

for label in classes:

    print(f"\nCreating samples for {label}...")

    class_df = df[df["label"] == label].sample(
        n=samples_per_class,
        random_state=42
    ).reset_index(drop=True)

    for i, row in class_df.iterrows():

        pre_path = os.path.join(
            IMAGE_FOLDER,
            row["pre_image"]
        )

        post_path = os.path.join(
            IMAGE_FOLDER,
            row["post_image"]
        )

        pre = cv2.imread(pre_path)
        post = cv2.imread(post_path)

        if pre is None or post is None:
            continue

        xmin = max(0, int(row["xmin"]) - padding)
        ymin = max(0, int(row["ymin"]) - padding)

        xmax = min(pre.shape[1], int(row["xmax"]) + padding)
        ymax = min(pre.shape[0], int(row["ymax"]) + padding)

        pre_crop = pre[ymin:ymax, xmin:xmax]
        post_crop = post[ymin:ymax, xmin:xmax]

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER,
                f"{label}_{i+1}_pre.png"
            ),
            pre_crop
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_FOLDER,
                f"{label}_{i+1}_post.png"
            ),
            post_crop
        )

print("\n===================================")
print("Demo samples created successfully!")
print("===================================")