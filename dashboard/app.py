import streamlit as st
import pandas as pd
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="GenoTB-Care",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🧬 GenoTB-Care")
st.subheader("TB Genomic Drug-Resistance Prediction")

st.info(
    "Prototype: WGS → TB-Profiler → Feature Engineering → ML"
)

# -----------------------------
# Load data
# -----------------------------
feature_file = "results/feature_table.csv"
prediction_file = "results/ml_predictions.csv"
importance_file = "results/feature_importance.csv"

if not os.path.exists(feature_file):
    st.error("Feature table not found.")
    st.stop()

df = pd.read_csv(feature_file)

# Load predictions if available
if os.path.exists(prediction_file):
    predictions = pd.read_csv(prediction_file)
    df = df.merge(
        predictions[["sample_id", "predicted_drtype"]],
        on="sample_id",
        how="left"
    )

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Sample Selection")

sample = st.sidebar.selectbox(
    "Select TB sample",
    df["sample_id"].tolist()
)

row = df[df["sample_id"] == sample].iloc[0]

# -----------------------------
# Sample overview
# -----------------------------
st.header("Sample Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Sample ID",
        row["sample_id"]
    )

with col2:
    st.metric(
        "Lineage",
        row["main_lineage"]
    )

with col3:
    st.metric(
        "DR Variants",
        int(row["total_dr_variants"])
    )

with col4:
    st.metric(
        "Mutation Count",
        int(row["mutation_count"])
    )

# -----------------------------
# Resistance
# -----------------------------
st.header("Drug-Resistance Profile")

col1, col2 = st.columns(2)

with col1:

    st.write(
        "**TB-Profiler Resistance Classification**"
    )

    st.success(
        str(row["drtype"])
    )

with col2:

    if "predicted_drtype" in row.index:

        st.write(
            "**ML Predicted Classification**"
        )

        st.success(
            str(row["predicted_drtype"])
        )

# -----------------------------
# Mutations
# -----------------------------
st.header("Detected Resistance Mutations")

mutations = str(row.get("mutations", ""))

if mutations == "" or mutations == "nan":

    st.warning(
        "No resistance-associated mutations detected."
    )

else:

    mutation_list = mutations.split(";")

    for mutation in mutation_list:

        mutation = mutation.strip()

        if mutation:
            st.code(
                mutation,
                language="text"
            )

# -----------------------------
# Feature importance
# -----------------------------
st.header("ML Feature Importance")

if os.path.exists(importance_file):

    importance = pd.read_csv(
        importance_file
    )

    top = importance.head(10)

    st.bar_chart(
        top.set_index("feature")["importance"]
    )

else:

    st.warning(
        "Feature importance file not found."
    )

# -----------------------------
# Confusion matrix
# -----------------------------
st.header("ML Evaluation")

confusion_file = (
    "results/confusion_matrix.png"
)

if os.path.exists(confusion_file):

    st.image(
        confusion_file,
        caption="Leave-One-Out Cross-Validation Confusion Matrix"
    )

else:

    st.warning(
        "Confusion matrix not found."
    )

# -----------------------------
# Full feature table
# -----------------------------
st.header("Feature Dataset")

st.dataframe(
    df,
    use_container_width=True
)

# -----------------------------
# Disclaimer
# -----------------------------
st.divider()

st.caption(
    "Research prototype only. "
    "The current model is trained/evaluated on a very small illustrative "
    "dataset and is not clinically validated."
)
