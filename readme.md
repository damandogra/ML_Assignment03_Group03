# Waste Detection in the Built Environment

This project implements a YOLO-based pipeline to detect waste in street-view images and evaluate performance using Precision@100 (P@100).

---

## Project Structure

```
FINAL DATASET/
│
├── train/ | val/ | test/
│   ├── images/        # NOT included
│   └── labels/        # NOT included
│
├── data.yaml
├── run_pipeline.py
├── predict_top100.py
├── leaderboard.py
├── requirements.txt
├── readme.md
└── .gitignore
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

GPU (CUDA) is required.

---

## Dataset (Not Included)

Due to size constraints, images and labels are not included in this repository.

Expected structure:

```
train/images/
train/labels/
val/images/
val/labels/
test/images/
test/labels/
```

---

## Pipeline

### 1. Evaluate Model

```bash
python run_pipeline.py
```

- Checks dataset structure
- Runs validation and test evaluation
- Outputs mAP metrics

---

### 2. Generate Top 100 Predictions

```bash
python predict_top100.py
```

- Ranks test images using confidence (conf=0.001)
- Selects Top 100 images
- Saves outputs to:
  - `top100_output/original/`
  - `top100_output/visualized/` (conf=0.40)

---

### 3. Compute P@100

```bash
python leaderboard.py
```

- Compares Top 100 with test labels
- Computes:

```
P@100 = correct waste images / 100
```

---

## Model

- Framework: Ultralytics YOLO
- Image size: 512
- Best model expected at:

```
runs/detect/runs/detect/yolo26_5classes/weights/best.pt
```

---

## Notes

- Dataset is highly imbalanced (many clean images)
- Primary evaluation metric: P@100
- Ensure folder structure matches `data.yaml`

---

## Authors

- Daman Dogra  
- Simon Deuten  
- Wieger van Teeffelen