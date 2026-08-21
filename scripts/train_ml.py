import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load feature table
df = pd.read_csv("results/feature_table.csv")

# Target
y = df["drtype"]

# Select numerical features
X = df[[
    "total_dr_variants",
    "mutation_count"
]].copy()

# Add mutation indicators
all_mutations = set()

for m in df["mutations"].fillna(""):
    if m:
        for mutation in m.split(";"):
            all_mutations.add(mutation.strip())

for mutation in sorted(all_mutations):
    X["MUT_" + mutation] = df["mutations"].fillna("").apply(
        lambda x: int(mutation in x.split(";"))
    )

# Encode target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\n==============================")
print("ML DATASET")
print("==============================")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", list(encoder.classes_))

print("\nTarget labels:")
for original, encoded in zip(encoder.classes_, range(len(encoder.classes_))):
    print(encoded, "=", original)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y_encoded)

print("\n==============================")
print("MODEL TRAINED")
print("==============================")
print("Training accuracy:",
      round(model.score(X, y_encoded), 3))

# Feature importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop features:")
print(importance.head(15).to_string(index=False))

# Save model dataset
X_out = X.copy()
X_out["drtype"] = y.values
X_out.to_csv("results/ml_dataset.csv", index=False)

print("\nSaved:")
print("results/ml_dataset.csv")
   
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load feature table
df = pd.read_csv("results/feature_table.csv")

# Target
y = df["drtype"]

# Select numerical features
X = df[[
    "total_dr_variants",
    "mutation_count"
]].copy()

# Add mutation indicators
all_mutations = set()

for m in df["mutations"].fillna(""):
    if m:
        for mutation in m.split(";"):
            all_mutations.add(mutation.strip())

for mutation in sorted(all_mutations):
    X["MUT_" + mutation] = df["mutations"].fillna("").apply(
        lambda x: int(mutation in x.split(";"))
    )

# Encode target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\n==============================")
print("ML DATASET")
print("==============================")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", list(encoder.classes_))

print("\nTarget labels:")
for original, encoded in zip(encoder.classes_, range(len(encoder.classes_))):
    print(encoded, "=", original)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y_encoded)

print("\n==============================")
print("MODEL TRAINED")
print("==============================")
print("Training accuracy:",
      round(model.score(X, y_encoded), 3))

# Feature importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop features:")
print(importance.head(15).to_string(index=False))

# Save model dataset
X_out = X.copy()
X_out["drtype"] = y.values
X_out.to_csv("results/ml_dataset.csv", index=False)

print("\nSaved:")
print("results/ml_dataset.csv")
