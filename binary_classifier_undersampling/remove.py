import os
import random

no_waste_dir = r"val/no_waste"
waste_dir = r"val/waste"

# get files
no_waste_files = [f for f in os.listdir(no_waste_dir) if f.endswith(".jpg")]
waste_files = [f for f in os.listdir(waste_dir) if f.endswith(".jpg")]

# target size
target_count = len(waste_files)

# how many to remove
remove_count = len(no_waste_files) - target_count

if remove_count <= 0:
    print("No need to remove files")
else:
    # randomly select files to delete
    files_to_remove = random.sample(no_waste_files, remove_count)

    for f in files_to_remove:
        os.remove(os.path.join(no_waste_dir, f))

    print(f"Removed {remove_count} files from no_waste")