import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------
# Load feature table
# ---------------------------------------

df = pd.read_csv("results/feature_table.csv")

print("Dataset shape:", df.shape)
print("\nSamples:")
print(df["sample_id"].tolist())

# ---------------------------------------
# Create binary target
# ---------------------------------------
# For prototype:
# resistant = any non-susceptible TB-Profiler drug-resistance category

df["target"] = (
    df["drtype"]
    .fillna("")
    .str.lower()
    .ne("susceptible")
    .astype(int)
)

print("\nTarget distribution:")
print(df["target"].value_counts())

# ---------------------------------------
# Select mutation features only
# ---------------------------------------

feature_columns = [
    c for c in df.columns
    if c.startswith("mutation_")
]

print("\nMutation features:")
print(feature_columns)

if len(feature_columns) == 0:
    raise ValueError("No mutation features found.")

X = df[feature_columns].fillna(0)
y = df["target"]

# ---------------------------------------
# Check dataset size
# ---------------------------------------

if len(df) < 10:
    raise ValueError(
        f"Only {len(df)} samples available. "
        "Collect at least 10 samples before training the prototype."
    )

if y.nunique() < 2:
    raise ValueError(
        "Only one target class is present. "
        "Need both resistant and non-resistant samples."
    )

# ---------------------------------------
# Train / test split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------
# Random Forest
# ---------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------------------------------
# Prediction
# ---------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------
# Evaluation
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1-score :", round(f1, 3))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------
# Feature importance
# ---------------------------------------

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop features:")
print(importance.head(15).to_string(index=False))

# ---------------------------------------
# Save model
# ---------------------------------------

joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    "model/genotb_random_forest.joblib"
)

importance.to_csv(
    "model/feature_importance.csv",
    index=False
)

print("\nModel saved:")
print("model/genotb_random_forest.joblib")
print("model/feature_importance.csv")
