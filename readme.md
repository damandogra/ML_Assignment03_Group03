# Waste Detection in the Built Environment

This project implements a YOLO-based pipeline to detect waste in street-view images and evaluate results using Precision@100 (P@100).

---

## Project Structure

```
FINAL DATASET/
│
├── train/ | val/ | test/
│   ├── images/
│   └── labels/
│
├── runs/                # YOLO outputs
├── top100_output/       # results
│   ├── original/
│   └── visualized/
│
├── data.yaml
├── run_pipeline.py
├── predict_top100.py
├── leaderboard.py
├── yolo26n.pt
├── requirements.txt
└── readme.md
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

GPU (CUDA) is required.

---

## Pipeline

### 1. Evaluate Model

```bash
python run_pipeline.py
```

- Checks dataset
- Runs validation + test evaluation
- Outputs mAP metrics

---

### 2. Generate Top 100

```bash
python predict_top100.py
```

- Ranks test images using confidence (conf=0.001)
- Selects Top 100 images
- Saves:
  - `top100_output/original/`
  - `top100_output/visualized/` (conf=0.40)

---

### 3. Compute P@100

```bash
python leaderboard.py
```

- Compares Top 100 with test labels
- Outputs:

```
P@100 = correct waste images / 100
```

---

## Model

- Framework: Ultralytics YOLO
- Image size: 512
- Best model path:

```
runs/detect/runs/detect/yolo26_5classes/weights/best.pt
```

---

## Notes

- Dataset is highly imbalanced (many clean images)
- Ranking (P@100) is primary evaluation metric
- Ensure folder structure matches `data.yaml`

---

## Authors

- Daman Dogra  
- Simon Deuten  
- Wieger van Teeffelen