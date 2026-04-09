import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ground truth from annotated subset
gt = pd.read_csv("test_subset/labels.csv")

# model predictions for all test images
pred = pd.read_csv("predictions.csv")

# normalize labels
gt["label"] = gt["label"].replace({"clean": "no_waste"})

# predicted class from probabilities
pred["pred_label"] = pred.apply(
    lambda row: "waste" if row["prob_waste"] >= row["prob_no_waste"] else "no_waste",
    axis=1
)

# keep only annotated images
df = pd.merge(gt, pred, on="image", how="inner")

print(f"Matched annotated images: {len(df)}")

y_true = df["label"]
y_pred = df["pred_label"]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label="waste")
rec = recall_score(y_true, y_pred, pos_label="waste")
f1 = f1_score(y_true, y_pred, pos_label="waste")

print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

cm = confusion_matrix(y_true, y_pred, labels=["no_waste", "waste"])
print("\nConfusion Matrix [rows=true, cols=pred]:")
print(pd.DataFrame(cm, index=["true_no_waste", "true_waste"], columns=["pred_no_waste", "pred_waste"]))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=["no_waste", "waste"]))

# Top-100 precision on annotated subset
top100 = df.sort_values("prob_waste", ascending=False).head(100)
p_at_100 = (top100["label"] == "waste").mean()

print(f"P@100 on annotated subset: {p_at_100:.4f}")

# save detailed outputs
df.to_csv("eval_all_annotated.csv", index=False)
top100.to_csv("eval_top100_annotated.csv", index=False)