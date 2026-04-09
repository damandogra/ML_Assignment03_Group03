from ultralytics import YOLO
import os
import pandas as pd

model = YOLO("runs/classify/cls26_full/weights/best.pt")

results = model.predict(source="test", device=0, verbose=False)

rows = []

for r in results:
    image_name = os.path.basename(r.path)

    probs = r.probs.data.tolist()
    names = r.names

    # find class indices
    waste_idx = [k for k, v in names.items() if v == "waste"][0]
    no_waste_idx = [k for k, v in names.items() if v == "no_waste"][0]

    prob_waste = probs[waste_idx]
    prob_no_waste = probs[no_waste_idx]

    rows.append([image_name, prob_waste, prob_no_waste])

df = pd.DataFrame(rows, columns=["image", "prob_waste", "prob_no_waste"])
df.to_csv("predictions.csv", index=False)

print("Saved predictions.csv")