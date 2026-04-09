import os
import shutil
import pandas as pd

csv_file = "final_top100.csv"
source_folder = "test"
output_folder = "top100_images"

df = pd.read_csv(csv_file)

os.makedirs(output_folder, exist_ok=True)

copied = 0
missing = []

for image_name in df["image"]:
    src = os.path.join(source_folder, image_name)
    dst = os.path.join(output_folder, image_name)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing.append(image_name)

print(f"Copied {copied} images to '{output_folder}'")

if missing:
    print("\nMissing images:")
    for m in missing:
        print(m)