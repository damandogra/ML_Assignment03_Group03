# Waste Detection in the Built Environment

This repository contains a YOLO-based pipeline for waste detection in street-view images, including model evaluation, Top-100 image ranking, and Precision@100 (P@100) scoring.

---

## Repository Structure

```text
ML_Assignment03_Group03/
├── src/
│   ├── run_pipeline.py
│   ├── predict_top100.py
│   ├── leaderboard.py
│   └── data.yaml
├── requirements.txt
├── readme.md
└── .gitignore
```

---

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is required for running the pipeline.

---

## Dataset

The dataset is not included in this repository.

Expected folder structure:

```text
train/images/
train/labels/
val/images/
val/labels/
test/images/
test/labels/
```

Make sure the dataset structure matches the paths defined in `src/data.yaml`.

---

## Pipeline

Run all commands from the repository root.

### 1. Evaluate model

```bash
python src/run_pipeline.py
```

This script:
- checks dataset structure
- runs validation and test evaluation
- reports mAP metrics

### 2. Generate Top 100 predictions

```bash
python src/predict_top100.py
```

This script:
- predicts on the test set
- ranks images using the highest detection confidence
- selects the Top 100 images
- saves original and visualized outputs

### 3. Compute P@100

```bash
python src/leaderboard.py
```

This script computes:

```text
P@100 = correct waste images / 100
```

---

## Model

- Framework: Ultralytics YOLO
- Image size: 512
- Best model expected at:

```text
runs/detect/runs/detect/yolo26_5classes/weights/best.pt
```

---

## Notes

- The dataset is highly imbalanced, with many clean images.
- The primary evaluation metric is P@100.
- Ensure the dataset paths and model path are correct before running the scripts.

---

## Authors

- Daman Dogra
- Simon Deuten
- Wieger van Teeffelen