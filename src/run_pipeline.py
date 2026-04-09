from pathlib import Path
import torch
from ultralytics import YOLO


# ===== CHECK IF PATH EXISTS =====
def check_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# ===== COUNT FILES IN FOLDER =====
def count_files(folder: Path, suffixes: set[str]) -> int:
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in suffixes)


# ===== FORCE GPU USAGE =====
def enforce_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU required.")

    gpu_index = 0
    print("\n===== GPU INFO =====")
    print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
    return gpu_index


def main():
    # ===== DEFINE PATHS =====
    project_root = Path.cwd()
    dataset_root = project_root
    yaml_path = project_root / "data.yaml"

    # ===== CHECK GPU =====
    gpu_device = enforce_gpu()

    # ===== CHECK DATASET STRUCTURE =====
    required = {
        "data.yaml": yaml_path,
        "train images": dataset_root / "train" / "images",
        "train labels": dataset_root / "train" / "labels",
        "val images": dataset_root / "val" / "images",
        "val labels": dataset_root / "val" / "labels",
        "test images": dataset_root / "test" / "images",
        "test labels": dataset_root / "test" / "labels",
    }

    for label, path in required.items():
        check_exists(path, label)

    # ===== PRINT DATASET SIZE =====
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    print("\n===== DATASET CHECK =====")
    print(f"Train images: {count_files(required['train images'], image_exts)}")
    print(f"Train labels: {count_files(required['train labels'], {'.txt'})}")
    print(f"Val images:   {count_files(required['val images'], image_exts)}")
    print(f"Val labels:   {count_files(required['val labels'], {'.txt'})}")
    print(f"Test images:  {count_files(required['test images'], image_exts)}")
    print(f"Test labels:  {count_files(required['test labels'], {'.txt'})}")

    # # ===== LOAD YOLO26 MODEL =====
    # model = YOLO("yolo26n.pt")

    # # ===== TRAIN MODEL =====
    # print("\n===== TRAINING =====")
    # model.train(
    #     data=str(yaml_path),
    #     epochs=100,
    #     patience=20,
    #     imgsz=512,
    #     batch=8,
    #     workers=0,
    #     device=gpu_device,
    #     cos_lr=True,
    #     project="runs/detect",
    #     name="yolo26_5classes",
    #     exist_ok=True,
    #     pretrained=True,
    #     plots=True
    # )

    # ===== LOAD BEST MODEL =====
    best_model_path = (
    project_root
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "yolo26_5classes"
    / "weights"
    / "best.pt"
    )
    check_exists(best_model_path, "best model")
    print(f"\nUsing trained model: {best_model_path}")

    best_model = YOLO(str(best_model_path))

    # ===== VALIDATION (VAL SET) =====
    print("\n===== VALIDATION =====")
    val_metrics = best_model.val(
        data=str(yaml_path),
        split="val",
        imgsz=512,
        batch=8,
        device=gpu_device,
        plots=True
    )

    # ===== TEST EVALUATION =====
    print("\n===== TEST EVALUATION =====")
    test_metrics = best_model.val(
        data=str(yaml_path),
        split="test",
        imgsz=512,
        batch=8,
        device=gpu_device,
        plots=True
    )

    # ===== PRINT RESULTS =====
    print("\n===== RESULTS =====")
    try:
        print(f"VAL mAP50:  {val_metrics.box.map50:.4f}")
        print(f"VAL mAP50-95: {val_metrics.box.map:.4f}")
        print(f"TEST mAP50: {test_metrics.box.map50:.4f}")
        print(f"TEST mAP50-95: {test_metrics.box.map:.4f}")
    except Exception:
        print("Metrics printed in logs.")

    print("\nOutputs saved in: runs/detect/yolo26_5classes/")


if __name__ == "__main__":
    main()