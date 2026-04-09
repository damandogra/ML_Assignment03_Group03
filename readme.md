# Waste Detection in the Built Environment

This project implements a full machine learning pipeline for detecting waste in street-view images using YOLO (Ultralytics).

It includes:
- Model evaluation (validation + test)
- Top-100 image ranking (based on confidence)
- Precision@100 (P@100) calculation

---

## Project Structure

```
FINAL DATASET/
│
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
│
├── runs/                # YOLO training outputs
├── top100_output/       # generated results
│   ├── original/
│   └── visualized/
│
├── data.yaml
├── run_pipeline.py
├── predict_top100.py
├── leaderboard.py
├── yolo26n.pt
├── readme.md
└── .gitignore
```

---

## Requirements

Install dependencies:

```bash
pip install ultralytics torch opencv-python
```

GPU (CUDA) is required.

---

## Dataset Format

YOLO format:

- Images: `.jpg/.png`
- Labels: `.txt`
- Each label file contains:

```
class_id x_center y_center width height
```

Classes (from `data.yaml`):
- 0: bulky_waste
- 1: garbage_bag
- 2: cardboard
- 3: litter
- 4: other

---

## Pipeline Overview

### 1. Model Evaluation

Runs validation and test evaluation using trained model.

```bash
python run_pipeline.py
```

Outputs:
- mAP@50
- mAP@50-95
- Plots saved in `runs/`

---

### 2. Generate Top 100 Predictions

Ranks all test images based on highest detection confidence.

```bash
python predict_top100.py
```

Process:
- Uses low confidence (0.001) for ranking
- Selects top 100 images
- Re-runs with higher confidence (0.40) for visualization

Outputs:

```
top100_output/
├── original/
└── visualized/
```

---

### 3. Compute Precision@100

Evaluates how many of the Top 100 images actually contain waste.

```bash
python leaderboard.py
```

Output example:

```
Total images checked: 100
Images with waste: 90
P@100: 0.900
```

---

## Model

- Framework: Ultralytics YOLO
- Model: YOLO26n
- Image size: 512 × 512
- Training: 100 epochs, early stopping (patience=20)

Best model expected at:

```
runs/detect/runs/detect/yolo26_5classes/weights/best.pt
```

---

## Reproducibility Steps

1. Place dataset in correct structure
2. Ensure trained model (`best.pt`) exists
3. Run:

```bash
python run_pipeline.py
python predict_top100.py
python leaderboard.py
```

---

## Output Summary

| Step        | Output              |
|------------|--------------------|
| Evaluation | mAP scores         |
| Prediction | Top 100 images     |
| Ranking    | Confidence sorting |
| Metric     | P@100              |

---

## Authors

- Daman Dogra  
- Simon Deuten  
- Wieger van Teeffelen