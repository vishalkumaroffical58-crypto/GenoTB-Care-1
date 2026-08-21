import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ============================================================
# GenoTB-Care
# TB Drug Resistance Prediction Dashboard
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

FEATURE_TABLE = os.path.join(
    RESULTS_DIR, "feature_table.csv"
)

ML_DATASET = os.path.join(
    RESULTS_DIR, "ml_dataset.csv"
)

ML_PREDICTIONS = os.path.join(
    RESULTS_DIR, "ml_predictions.csv"
)

FEATURE_IMPORTANCE = os.path.join(
    RESULTS_DIR, "feature_importance.csv"
)

FEATURE_IMPORTANCE_PNG = os.path.join(
    RESULTS_DIR, "feature_importance.png"
)

CONFUSION_MATRIX_PNG = os.path.join(
    RESULTS_DIR, "confusion_matrix.png"
)


# ============================================================
# CUSTOM CSS
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

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333333;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
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

    - Mutation analysis
    - Drug resistance profiling
    - Feature engineering
    - Machine learning
    - Resistance prediction
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
        "🤖 ML Prediction",
        "📊 Feature Importance",
        "🎯 Model Performance",
        "📋 Feature Table"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "GenoTB-Care | TB WGS + Machine Learning"
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
# DASHBOARD HOME
# ============================================================

if page == "🏠 Dashboard":

    st.header("Project Overview")

    st.markdown(
        """
        **GenoTB-Care** is a computational prototype for
        analyzing *Mycobacterium tuberculosis* whole-genome
        sequencing data and predicting drug resistance.

        The workflow integrates:

        **WGS → TB-Profiler → Mutation Detection → Feature Engineering
        → Machine Learning → Resistance Prediction**
        """
    )

    st.divider()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    if feature_df is not None:

        total_samples = len(feature_df)

        if "mutation_count" in feature_df.columns:
            total_mutations = feature_df[
                "mutation_count"
            ].sum()
        else:
            total_mutations = 0

        if "total_dr_variants" in feature_df.columns:
            total_dr = feature_df[
                "total_dr_variants"
            ].sum()
        else:
            total_dr = 0

        total_features = len(feature_df.columns)

    else:

        total_samples = 0
        total_mutations = 0
        total_dr = 0
        total_features = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "TB Samples",
        total_samples
    )

    col2.metric(
        "Total Mutations",
        int(total_mutations)
    )

    col3.metric(
        "Drug Resistance Variants",
        int(total_dr)
    )

    col4.metric(
        "Features",
        total_features
    )

    st.divider()

    # --------------------------------------------------------
    # SAMPLE SUMMARY
    # --------------------------------------------------------

    st.subheader("Sample Summary")

    if feature_df is not None:

        display_columns = [
            "sample_id",
            "main_lineage",
            "sub_lineage",
            "drtype",
            "total_dr_variants",
            "mutation_count"
        ]

        display_columns = [
            c for c in display_columns
            if c in feature_df.columns
        ]

        st.dataframe(
            feature_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "feature_table.csv was not found."
        )


# ============================================================
# SAMPLE ANALYSIS
# ============================================================

elif page == "🧬 Sample Analysis":

    st.header("🧬 Individual TB Sample Analysis")

    if feature_df is None:

        st.error(
            "results/feature_table.csv not found."
        )

    else:

        if "sample_id" not in feature_df.columns:

            st.error(
                "sample_id column not found."
            )

        else:

            sample_list = feature_df[
                "sample_id"
            ].astype(str).tolist()

            selected_sample = st.selectbox(
                "Select TB Sample",
                sample_list
            )

            sample = feature_df[
                feature_df["sample_id"].astype(str)
                == selected_sample
            ]

            if len(sample) > 0:

                sample = sample.iloc[0]

                st.subheader(
                    f"Sample: {selected_sample}"
                )

                col1, col2, col3, col4 = st.columns(4)

                if "main_lineage" in sample:
                    col1.metric(
                        "Lineage",
                        sample["main_lineage"]
                    )

                if "sub_lineage" in sample:
                    col2.metric(
                        "Sub-lineage",
                        sample["sub_lineage"]
                    )

                if "drtype" in sample:
                    col3.metric(
                        "Drug Resistance Type",
                        sample["drtype"]
                    )

                if "mutation_count" in sample:
                    col4.metric(
                        "Mutation Count",
                        sample["mutation_count"]
                    )

                st.divider()

                st.subheader(
                    "Sample Genomic Features"
                )

                sample_table = pd.DataFrame(
                    {
                        "Feature": sample.index,
                        "Value": sample.values
                    }
                )

                st.dataframe(
                    sample_table,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# DRUG RESISTANCE
# ============================================================

elif page == "💊 Drug Resistance":

    st.header("💊 Drug Resistance Analysis")

    if feature_df is None:

        st.error(
            "Feature table not found."
        )

    else:

        if "drtype" in feature_df.columns:

            st.subheader(
                "Resistance Classification"
            )

            resistance_counts = (
                feature_df["drtype"]
                .fillna("Unknown")
                .value_counts()
            )

            st.bar_chart(
                resistance_counts
            )

            st.dataframe(
                resistance_counts
                .reset_index()
                .rename(
                    columns={
                        "index": "Resistance Type",
                        "drtype": "Samples"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

        if "total_dr_variants" in feature_df.columns:

            st.subheader(
                "Drug Resistance Variants per Sample"
            )

            chart_df = feature_df[
                ["sample_id", "total_dr_variants"]
            ].copy()

            chart_df = chart_df.set_index(
                "sample_id"
            )

            st.bar_chart(
                chart_df
            )

        # ----------------------------------------------------
        # DRUG FEATURE COLUMNS
        # ----------------------------------------------------

        drug_columns = [
            c for c in feature_df.columns
            if c.startswith("drug_")
        ]

        if drug_columns:

            st.subheader(
                "Drug Resistance Feature Matrix"
            )

            drug_df = feature_df[
                ["sample_id"] + drug_columns
            ].copy()

            st.dataframe(
                drug_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "Drug Resistance Heatmap"
            )

            heatmap_df = drug_df.set_index(
                "sample_id"
            )

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            ax.imshow(
                heatmap_df.values,
                aspect="auto"
            )

            ax.set_xticks(
                range(len(heatmap_df.columns))
            )

            ax.set_xticklabels(
                heatmap_df.columns,
                rotation=90
            )

            ax.set_yticks(
                range(len(heatmap_df.index))
            )

            ax.set_yticklabels(
                heatmap_df.index
            )

            ax.set_title(
                "Drug Resistance Feature Matrix"
            )

            plt.tight_layout()

            st.pyplot(fig)


# ============================================================
# MUTATIONS
# ============================================================

elif page == "🔬 Mutations":

    st.header("🔬 TB Mutation Analysis")

    if feature_df is None:

        st.error(
            "Feature table not found."
        )

    else:

        # ----------------------------------------------------
        # MUTATION COUNT
        # ----------------------------------------------------

        if "mutation_count" in feature_df.columns:

            st.subheader(
                "Mutation Count by Sample"
            )

            mutation_df = feature_df[
                ["sample_id", "mutation_count"]
            ].copy()

            mutation_df = mutation_df.set_index(
                "sample_id"
            )

            st.bar_chart(
                mutation_df
            )

        # ----------------------------------------------------
        # MUTATION STRINGS
        # ----------------------------------------------------

        if "mutations" in feature_df.columns:

            st.subheader(
                "Detected Mutations"
            )

            mutation_table = feature_df[
                [
                    "sample_id",
                    "mutations"
                ]
            ].copy()

            st.dataframe(
                mutation_table,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # DR VARIANTS
        # ----------------------------------------------------

        if "total_dr_variants" in feature_df.columns:

            st.subheader(
                "Drug-Resistance Variant Count"
            )

            dr_df = feature_df[
                [
                    "sample_id",
                    "total_dr_variants"
                ]
            ].copy()

            st.dataframe(
                dr_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# ML PREDICTION
# ============================================================

elif page == "🤖 ML Prediction":

    st.header(
        "🤖 Machine Learning Resistance Prediction"
    )

    st.markdown(
        """
        This section displays the machine-learning
        results generated from the GenoTB-Care feature
        engineering pipeline.
        """
    )

    if prediction_df is None:

        st.warning(
            "results/ml_predictions.csv was not found."
        )

        st.info(
            "Run the ML pipeline first."
        )

    else:

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DETECT PREDICTION COLUMN
        # ----------------------------------------------------

        possible_prediction_columns = [
            "prediction",
            "predicted",
            "predicted_class",
            "prediction_label",
            "y_pred"
        ]

        prediction_column = None

        for col in possible_prediction_columns:

            if col in prediction_df.columns:

                prediction_column = col
                break

        if prediction_column:

            st.subheader(
                "Prediction Distribution"
            )

            counts = (
                prediction_df[
                    prediction_column
                ]
                .astype(str)
                .value_counts()
            )

            st.bar_chart(
                counts
            )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability_columns = [
            c for c in prediction_df.columns
            if "prob" in c.lower()
        ]

        if probability_columns:

            st.subheader(
                "Prediction Probabilities"
            )

            st.dataframe(
                prediction_df[
                    probability_columns
                ],
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "📊 Feature Importance":

    st.header(
        "📊 ML Feature Importance"
    )

    if importance_df is not None:

        st.subheader(
            "Feature Importance Table"
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        # Automatically detect feature and importance columns

        numeric_columns = (
            importance_df
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        if len(numeric_columns) > 0:

            importance_column = numeric_columns[-1]

            st.subheader(
                "Feature Importance Plot"
            )

            plot_df = importance_df.copy()

            feature_column = None

            for c in [
                "feature",
                "Feature",
                "features",
                "Feature_Name"
            ]:

                if c in plot_df.columns:

                    feature_column = c
                    break

            if feature_column:

                plot_df = plot_df.sort_values(
                    importance_column,
                    ascending=True
                )

                fig, ax = plt.subplots(
                    figsize=(10, 7)
                )

                ax.barh(
                    plot_df[
                        feature_column
                    ].astype(str),
                    plot_df[
                        importance_column
                    ]
                )

                ax.set_xlabel(
                    "Importance"
                )

                ax.set_ylabel(
                    "Feature"
                )

                ax.set_title(
                    "Random Forest Feature Importance"
                )

                plt.tight_layout()

                st.pyplot(fig)

    else:

        st.warning(
            "feature_importance.csv was not found."
        )

    # Existing PNG

    if os.path.exists(
        FEATURE_IMPORTANCE_PNG
    ):

        st.subheader(
            "Saved Feature Importance Figure"
        )

        st.image(
            FEATURE_IMPORTANCE_PNG,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🎯 Model Performance":

    st.header(
        "🎯 Machine Learning Model Performance"
    )

    if os.path.exists(
        CONFUSION_MATRIX_PNG
    ):

        st.subheader(
            "Confusion Matrix"
        )

        st.image(
            CONFUSION_MATRIX_PNG,
            use_container_width=True
        )

    else:

        st.warning(
            "confusion_matrix.png was not found."
        )

    st.divider()

    st.subheader(
        "Prediction Dataset"
    )

    if prediction_df is not None:

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "ML prediction file is not available."
        )


# ============================================================
# COMPLETE FEATURE TABLE
# ============================================================

elif page == "📋 Feature Table":

    st.header(
        "📋 Complete GenoTB-Care Feature Table"
    )

    if feature_df is None:

        st.error(
            "results/feature_table.csv was not found."
        )

    else:

        st.success(
            f"{len(feature_df)} samples × "
            f"{len(feature_df.columns)} features loaded"
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv_data = feature_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Feature Table",
            data=csv_data,
            file_name="genotb_feature_table.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GenoTB-Care | WGS-based TB drug resistance analysis "
    "and machine learning prototype"
)
