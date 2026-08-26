import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt


# ============================================================
# GenoTB-Care
# TB WGS + TB-Profiler + ML Dashboard
# ============================================================

st.set_page_config(
    page_title="GenoTB-Care",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


# TB-Profiler datasets
SAMPLE_SUMMARY = os.path.join(
    RESULTS_DIR,
    "sample_summary.csv"
)

COMBINED_RESULTS = os.path.join(
    RESULTS_DIR,
    "combined_results.csv"
)


# Existing ML datasets
FEATURE_TABLE = os.path.join(
    RESULTS_DIR,
    "feature_table.csv"
)

ML_DATASET = os.path.join(
    RESULTS_DIR,
    "ml_dataset.csv"
)

ML_PREDICTIONS = os.path.join(
    RESULTS_DIR,
    "ml_predictions.csv"
)

FEATURE_IMPORTANCE = os.path.join(
    RESULTS_DIR,
    "feature_importance.csv"
)

FEATURE_IMPORTANCE_PNG = os.path.join(
    RESULTS_DIR,
    "feature_importance.png"
)

CONFUSION_MATRIX_PNG = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #888888;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data
def load_csv(path):

    if os.path.exists(path):

        try:
            return pd.read_csv(path)

        except Exception as e:

            st.error(
                f"Could not read {path}: {e}"
            )

            return None

    return None


# TB-Profiler data
sample_df = load_csv(SAMPLE_SUMMARY)
results_df = load_csv(COMBINED_RESULTS)


# Existing ML data
feature_df = load_csv(FEATURE_TABLE)
ml_df = load_csv(ML_DATASET)
prediction_df = load_csv(ML_PREDICTIONS)
importance_df = load_csv(FEATURE_IMPORTANCE)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧬 GenoTB-Care")

st.sidebar.markdown(
    """
    ### TB Genomic Analysis

    Whole-genome sequencing based:

    - TB-Profiler
    - Mutation analysis
    - Drug resistance profiling
    - Lineage analysis
    - Quality control
    - Machine learning
    """
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🧬 Sample Analysis",
        "💊 Drug Resistance",
        "🔬 Mutations",
        "🧬 Lineage Analysis",
        "🧪 Quality Control",
        "🤖 ML Prediction",
        "📊 Feature Importance",
        "🎯 Model Performance",
        "📋 Feature Table"
    ]
)


st.sidebar.divider()

st.sidebar.caption(
    "GenoTB-Care | TB WGS + TB-Profiler + ML"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧬 GenoTB-Care</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Whole-Genome Sequencing Based Tuberculosis Drug Resistance Analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("Project Overview")

    st.markdown(
        """
        **GenoTB-Care** is a computational prototype for
        *Mycobacterium tuberculosis* whole-genome sequencing
        analysis.

        **Workflow**

        WGS → TB-Profiler → Variant Detection →
        Drug Resistance Analysis → Lineage Analysis →
        QC → Machine Learning
        """
    )

    st.divider()

    if sample_df is not None:

        total_samples = len(sample_df)

        qc_pass = (
            sample_df["Status"] == "PASS"
        ).sum()

        qc_fail = (
            sample_df["Status"] == "QC FAIL"
        ).sum()

        resistance_samples = (
            sample_df["Resistance_Drug_Count"] > 0
        ).sum()

        total_mutations = (
            sample_df["Resistance_Mutation_Count"].sum()
        )

    else:

        total_samples = 0
        qc_pass = 0
        qc_fail = 0
        resistance_samples = 0
        total_mutations = 0


    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "TB Samples",
        total_samples
    )

    col2.metric(
        "QC PASS",
        qc_pass
    )

    col3.metric(
        "QC FAIL",
        qc_fail
    )

    col4.metric(
        "Resistance Samples",
        resistance_samples
    )


    st.divider()

    # --------------------------------------------------------
    # SAMPLE SUMMARY
    # --------------------------------------------------------

    st.subheader("Sample Summary")

    if sample_df is not None:

        display_columns = [
            "Sample_ID",
            "Status",
            "Lineage",
            "Family",
            "Resistance_Drug_Count",
            "Resistance_Mutation_Count",
            "Percent_Reads_Mapped",
            "Target_Median_Depth"
        ]

        display_columns = [
            c for c in display_columns
            if c in sample_df.columns
        ]

        st.dataframe(
            sample_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "sample_summary.csv not found."
        )


# ============================================================
# SAMPLE ANALYSIS
# ============================================================

elif page == "🧬 Sample Analysis":

    st.header("🧬 Sample Analysis")

    if sample_df is None:

        st.error(
            "sample_summary.csv not found."
        )

    else:

        sample_list = sorted(
            sample_df["Sample_ID"].dropna().unique()
        )

        selected_sample = st.selectbox(
            "Select Sample",
            sample_list
        )

        sample_info = sample_df[
            sample_df["Sample_ID"] == selected_sample
        ]

        if len(sample_info) > 0:

            row = sample_info.iloc[0]

            st.subheader(
                f"Sample: {selected_sample}"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "QC Status",
                row.get("Status", "N/A")
            )

            col2.metric(
                "Lineage",
                row.get("Lineage", "N/A")
            )

            col3.metric(
                "Mapped Reads",
                row.get("Reads_Mapped", "N/A")
            )

            col4.metric(
                "Mapping %",
                row.get("Percent_Reads_Mapped", "N/A")
            )


            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Lineage")

                st.write(
                    "Lineage:",
                    row.get("Lineage", "Not determined")
                )

                st.write(
                    "Family:",
                    row.get("Family", "Not determined")
                )

                st.write(
                    "Lineage Fraction:",
                    row.get("Lineage_Fraction", "N/A")
                )

                st.write(
                    "Spoligotype:",
                    row.get("Spoligotype", "N/A")
                )


            with col2:

                st.subheader("QC")

                st.write(
                    "Status:",
                    row.get("Status", "N/A")
                )

                st.write(
                    "Reason:",
                    row.get("QC_Reason", "")
                )

                st.write(
                    "Target Median Depth:",
                    row.get("Target_Median_Depth", "N/A")
                )

                st.write(
                    "Genome Median Depth:",
                    row.get("Genome_Median_Depth", "N/A")
                )


            st.divider()

            st.subheader(
                "Genotypic Resistance-Associated Drugs"
            )

            drugs = row.get(
                "Resistance_Drugs",
                ""
            )

            if pd.isna(drugs) or drugs == "":

                st.info(
                    "No resistance-associated drug findings."
                )

            else:

                st.write(
                    str(drugs).replace(
                        ";",
                        " • "
                    )
                )


            # ------------------------------------------------
            # MUTATIONS FOR SELECTED SAMPLE
            # ------------------------------------------------

            if results_df is not None:

                sample_results = results_df[
                    results_df["Sample_ID"] == selected_sample
                ]

                st.divider()

                st.subheader(
                    "Resistance-Associated Mutations"
                )

                mutation_columns = [
                    "Gene",
                    "Position",
                    "Reference",
                    "Alternative",
                    "Frequency",
                    "Protein_Change",
                    "Nucleotide_Change",
                    "Drug",
                    "Resistance_Confidence",
                    "Resistance_Source"
                ]

                mutation_columns = [
                    c for c in mutation_columns
                    if c in sample_results.columns
                ]

                if len(sample_results) > 0:

                    st.dataframe(
                        sample_results[
                            mutation_columns
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No resistance-associated variants reported."
                    )


# ============================================================
# DRUG RESISTANCE
# ============================================================

elif page == "💊 Drug Resistance":

    st.header("💊 Drug Resistance Analysis")

    if results_df is None:

        st.error(
            "combined_results.csv not found."
        )

    else:

        resistance_df = results_df[
            results_df["Drug"].notna()
        ].copy()

        resistance_df = resistance_df[
            resistance_df["Drug"] != ""
        ]

        st.metric(
            "Resistance-Associated Records",
            len(resistance_df)
        )

        st.divider()

        # Drug counts

        drug_counts = (
            resistance_df
            .groupby("Drug")["Sample_ID"]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        st.subheader(
            "Samples by Drug"
        )

        st.bar_chart(
            drug_counts
        )

        st.divider()

        # Sample × Drug matrix

        st.subheader(
            "Sample × Drug Matrix"
        )

        matrix = pd.crosstab(
            resistance_df["Sample_ID"],
            resistance_df["Drug"]
        )

        matrix = (
            matrix > 0
        ).astype(int)

        st.dataframe(
            matrix,
            use_container_width=True
        )


# ============================================================
# MUTATIONS
# ============================================================

elif page == "🔬 Mutations":

    st.header("🔬 Mutation Analysis")

    if results_df is None:

        st.error(
            "combined_results.csv not found."
        )

    else:

        mutation_columns = [
            "Sample_ID",
            "Gene",
            "Position",
            "Reference",
            "Alternative",
            "Frequency",
            "Protein_Change",
            "Nucleotide_Change",
            "Drug",
            "Resistance_Confidence",
            "Resistance_Source",
            "Resistance_Comment"
        ]

        mutation_columns = [
            c for c in mutation_columns
            if c in results_df.columns
        ]

        mutation_df = results_df[
            mutation_columns
        ].copy()

        st.metric(
            "Resistance-Associated Mutation Records",
            len(mutation_df)
        )

        st.dataframe(
            mutation_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# LINEAGE
# ============================================================

elif page == "🧬 Lineage Analysis":

    st.header("🧬 Lineage Analysis")

    if sample_df is None:

        st.error(
            "sample_summary.csv not found."
        )

    else:

        lineage_df = sample_df.copy()

        lineage_df["Lineage"] = (
            lineage_df["Lineage"]
            .fillna("Not determined")
            .replace("", "Not determined")
        )

        lineage_counts = (
            lineage_df["Lineage"]
            .value_counts()
        )

        st.subheader(
            "Lineage Distribution"
        )

        st.bar_chart(
            lineage_counts
        )

        st.divider()

        st.dataframe(
            lineage_df[
                [
                    "Sample_ID",
                    "Lineage",
                    "Family",
                    "Lineage_Fraction"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# QUALITY CONTROL
# ============================================================

elif page == "🧪 Quality Control":

    st.header("🧪 Sequencing Quality Control")

    if sample_df is None:

        st.error(
            "sample_summary.csv not found."
        )

    else:

        qc_pass = (
            sample_df["Status"] == "PASS"
        ).sum()

        qc_fail = (
            sample_df["Status"] == "QC FAIL"
        ).sum()

        col1, col2 = st.columns(2)

        col1.metric(
            "QC PASS",
            qc_pass
        )

        col2.metric(
            "QC FAIL",
            qc_fail
        )

        st.divider()

        qc_columns = [
            "Sample_ID",
            "Status",
            "QC_Reason",
            "Percent_Reads_Mapped",
            "Reads_Mapped",
            "Target_Median_Depth",
            "Genome_Median_Depth"
        ]

        qc_columns = [
            c for c in qc_columns
            if c in sample_df.columns
        ]

        st.dataframe(
            sample_df[qc_columns],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ML PREDICTION
# ============================================================

elif page == "🤖 ML Prediction":

    st.header("🤖 Machine Learning Prediction")

    if prediction_df is not None:

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "ML predictions are not available yet."
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "📊 Feature Importance":

    st.header("📊 Feature Importance")

    if importance_df is not None:

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

    elif os.path.exists(
        FEATURE_IMPORTANCE_PNG
    ):

        st.image(
            FEATURE_IMPORTANCE_PNG,
            use_container_width=True
        )

    else:

        st.info(
            "Feature importance results are not available."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🎯 Model Performance":

    st.header("🎯 Model Performance")

    if os.path.exists(
        CONFUSION_MATRIX_PNG
    ):

        st.image(
            CONFUSION_MATRIX_PNG,
            use_container_width=True
        )

    else:

        st.info(
            "Model performance results are not available."
        )


# ============================================================
# FEATURE TABLE
# ============================================================

elif page == "📋 Feature Table":

    st.header("📋 Feature Table")

    if feature_df is not None:

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Feature table is not available."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GenoTB-Care | TB WGS + TB-Profiler + Genotypic Resistance Analysis"
)
