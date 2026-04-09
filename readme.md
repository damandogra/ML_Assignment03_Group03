# Waste Detection in the Built Environment

## Overview
This project implements a YOLO-based object detection pipeline to detect and rank waste in street-view images. The objective is to identify images containing waste and rank them using **Precision@100 (P@100)**.

---

## Dataset
- Total images: 10,000  
- Split:
  - Train: 2844  
  - Validation: 3064  
  - Test: 4092  
- Classes:
  - bulky_waste
  - garbage_bag
  - cardboard
  - litter
  - other  

⚠️ Note: Dataset is not included due to size.

---

## Methodology
- Model: **YOLO26n (Ultralytics)**
- Input size: 512 × 512
- Epochs: 100 (early stopping: patience = 20)
- Batch size: 8  
- Device: GPU (RTX 3050 Ti)

### Ranking Strategy
- Each image assigned score = **highest confidence detection**
- Ranking uses **low threshold (0.001)** to capture all detections
- Visualization uses **higher threshold (0.40)** for clarity

---

## Results
- mAP@50 (test): **0.102**
- mAP@50-95: **0.046**
- P@100: **0.90**

👉 Despite low detection accuracy, ranking performance is strong.

---

## Repository Structure