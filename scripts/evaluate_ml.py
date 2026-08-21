import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 1. Load feature table
# -----------------------------
df = pd.read_csv("results/feature_table.csv")

# -----------------------------
# 2. Create mutation features
# -----------------------------
all_mutations = set()

for value in df["mutations"].fillna(""):
    if value:
        for mutation in value.split(";"):
            mutation = mutation.strip()
            if mutation:
                all_mutations.add(mutation)

X = df[[
    "total_dr_variants",
    "mutation_count"
]].copy()

for mutation in sorted(all_mutations):
    X["MUT_" + mutation] = df["mutations"].fillna("").apply(
        lambda x: int(mutation in x.split(";"))
    )

# -----------------------------
# 3. Encode target
# -----------------------------
y = df["drtype"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\n==============================")
print("GENOTB-CARE ML EVALUATION")
print("==============================")

print("Samples:", len(df))
print("Features:", X.shape[1])
print("Classes:", list(encoder.classes_))

# -----------------------------
# 4. Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# -----------------------------
# 5. Leave-One-Out CV
# -----------------------------
loo = LeaveOneOut()

predicted = cross_val_predict(
    model,
    X,
    y_encoded,
    cv=loo
)

# -----------------------------
# 6. Accuracy
# -----------------------------
accuracy = accuracy_score(
    y_encoded,
    predicted
)

print("\n==============================")
print("LOOCV RESULT")
print("==============================")

print("Accuracy:", round(accuracy, 3))

# -----------------------------
# 7. Classification report
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        y_encoded,
        predicted,
        target_names=encoder.classes_,
        zero_division=0
    )
)

# -----------------------------
# 8. Confusion matrix
# -----------------------------
cm = confusion_matrix(
    y_encoded,
    predicted
)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# 9. Train final model
# -----------------------------
model.fit(X, y_encoded)

# -----------------------------
# 10. Feature importance
# -----------------------------
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n==============================")
print("TOP FEATURES")
print("==============================")

print(
    importance.head(15).to_string(index=False)
)

# -----------------------------
# 11. Save results
# -----------------------------
importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

predictions = df[[
    "sample_id",
    "drtype"
]].copy()

predictions["predicted_drtype"] = encoder.inverse_transform(
    predicted
)

predictions.to_csv(
    "results/ml_predictions.csv",
    index=False
)

print("\nFiles created:")
print("results/feature_importance.csv")
print("results/ml_predictions.csv")

print("\nEvaluation completed.")
