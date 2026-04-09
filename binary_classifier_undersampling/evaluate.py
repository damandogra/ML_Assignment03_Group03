import pandas as pd

gt = pd.read_csv("test_subset/labels.csv")
pred = pd.read_csv("predictions.csv")

# fix naming
gt["label"] = gt["label"].replace({"clean": "no_waste"})

df = pd.merge(gt, pred, on="image")

# sort by waste probability
df = df.sort_values("prob_waste", ascending=False)

# Top-100 precision
top100 = df.head(100)
p100 = (top100["label"] == "waste").mean()

print(f"Top-100 Precision: {p100:.3f}")