from ultralytics import YOLO
from pathlib import Path
import shutil
import cv2

project_root = Path.cwd()

# paths
model_path = project_root / "runs" / "detect" / "runs" / "detect" / "yolo26_5classes" / "weights" / "best.pt"
test_images_dir = project_root / "test" / "images"

# output folders
output_root = project_root / "top100_output"
original_dir = output_root / "original"
visualized_dir = output_root / "visualized"

original_dir.mkdir(parents=True, exist_ok=True)
visualized_dir.mkdir(parents=True, exist_ok=True)

# load model
model = YOLO(str(model_path))

# ===== PASS 1: rank images fairly using low threshold =====
ranking_results = model.predict(
    source=str(test_images_dir),
    imgsz=512,
    batch=6,
    conf=0.001,   # low threshold only for ranking
    device=0,
    save=False,
    verbose=True
)

# store best confidence per image
image_scores = []

for r in ranking_results:
    image_path = Path(r.path)

    if r.boxes is None or len(r.boxes) == 0:
        continue

    best_conf = max(float(box.conf.item()) for box in r.boxes)
    image_scores.append((image_path, best_conf))

# sort descending by best confidence
image_scores.sort(key=lambda x: x[1], reverse=True)

# select top 100 images
top100 = image_scores[:100]

# copy original images first
for image_path, best_conf in top100:
    shutil.copy2(image_path, original_dir / image_path.name)

# ===== PASS 2: clean visualization only for selected top 100 =====
for image_path, best_conf in top100:
    clean_result = model.predict(
        source=str(image_path),
        imgsz=512,
        conf=0.40,   # higher threshold for clean boxes
        iou=0.50,
        device=0,
        save=False,
        verbose=False
    )[0]

    plotted = clean_result.plot()
    cv2.imwrite(str(visualized_dir / image_path.name), plotted)

print(f"\nImages with at least one detection: {len(image_scores)}")
print(f"Top images selected: {len(top100)}")
print(f"\nOriginal images saved to: {original_dir}")
print(f"Visualized images saved to: {visualized_dir}")

print("\nTop 20 preview:")
for image_path, best_conf in top100[:20]:
    print(f"{image_path.name} | {best_conf:.4f}")