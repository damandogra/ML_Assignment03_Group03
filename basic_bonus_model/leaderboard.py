from pathlib import Path

project_root = Path.cwd()
top100_dir = project_root / "top100_output" / "original"
labels_dir = project_root / "test" / "labels"

count_waste = 0
total = 0

for img_path in top100_dir.glob("*.*"):
    label_file = labels_dir / (img_path.stem + ".txt")
    total += 1

    if label_file.exists() and label_file.read_text().strip() != "":
        count_waste += 1

p_at_100 = count_waste / total

print(f"Total images checked: {total}")
print(f"Images with waste: {count_waste}")
print(f"P@100: {p_at_100:.3f}")