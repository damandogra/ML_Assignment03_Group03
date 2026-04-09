import pandas as pd

pred = pd.read_csv("predictions.csv")

top100 = pred.sort_values("prob_waste", ascending=False).head(100)

top100[["image"]].to_csv("final_top100.csv", index=False)

print("Saved final_top100.csv")