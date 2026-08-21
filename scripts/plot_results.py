import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------
# Load predictions
# -----------------------------
pred = pd.read_csv("results/ml_predictions.csv")

# -----------------------------
# Confusion Matrix
# -----------------------------
true_labels = pred["drtype"]
pred_labels = pred["predicted_drtype"]

labels = sorted(
    set(true_labels) | set(pred_labels)
)

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    true_labels,
    pred_labels,
    labels=labels,
    ax=ax
)

plt.title("GenoTB-Care ML Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Created: results/confusion_matrix.png")


# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.read_csv(
    "results/feature_importance.csv"
)

top = importance.head(15).sort_values(
    "importance"
)

plt.figure(figsize=(9, 7))

plt.barh(
    top["feature"],
    top["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Mutation Features")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300
)

plt.close()

print("Created: results/feature_importance.png")

print("\nVisualization completed.")
