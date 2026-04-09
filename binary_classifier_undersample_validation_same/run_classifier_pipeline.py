from ultralytics import YOLO
import os

# =========================
# CONFIG
# =========================
DATA_ROOT = r"D:\MSc-Geomatics\Q3\Machine Learning\assignment3\GEO5017-Project-UrbanWaste\final dataset\BINARY_CLASSIFIER_UNDERSAMPLE_VALIDATION_SAME"
MODEL_WEIGHTS = "yolo26n-cls.pt"   # or your trained model
IMG_SIZE = 224
EPOCHS = 100
BATCH = 64
DEVICE = 0

# =========================
# 1. TRAIN
# =========================
def train_model():
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=DATA_ROOT,   # IMPORTANT: folder, not yaml
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        patience=10,
        workers=4,
        name="cls_final"
    )

# =========================
# 2. VALIDATION (val/)
# =========================
def validate_model():
    model = YOLO("runs/classify/cls_final/weights/best.pt")

    results = model.val(
        data=os.path.join(DATA_ROOT, "val"),
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE
    )

    print("\nValidation Results:")
    print(results)

# =========================
# 3. TEST EVALUATION (test/)
# =========================
def test_model():
    model = YOLO("runs/classify/cls_final/weights/best.pt")

    results = model.val(
        data=os.path.join(DATA_ROOT, "test"),
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        name="test_eval"
    )

    print("\nTest Results:")
    print(results)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("1. Training...")
    train_model()

    print("\n2. Validating...")
    validate_model()

    print("\n3. Testing on unseen data...")
    test_model()