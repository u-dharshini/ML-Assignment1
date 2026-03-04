import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from preprocessing import load_data, encode_data
from models import supervised_models, unsupervised_models

st.title("Student Performance ML Project")

# Load data
df = load_data()
df_encoded, label_encoders = encode_data(df)

menu = st.sidebar.selectbox(
    "Select Option",
    ["Supervised Learning", "Unsupervised Learning", "Prediction"]
)

# ==============================
# SUPERVISED
# ==============================
if menu == "Supervised Learning":

    log_model, tree_model, log_metrics, tree_metrics = supervised_models(df_encoded)

    st.subheader("Logistic Regression Results")
    st.write(log_metrics)

    fig, ax = plt.subplots()
    sns.heatmap(log_metrics["Confusion Matrix"], annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    st.subheader("Decision Tree Results")
    st.write(tree_metrics)

    fig2, ax2 = plt.subplots()
    sns.heatmap(tree_metrics["Confusion Matrix"], annot=True, fmt="d", ax=ax2)
    st.pyplot(fig2)
    
        # ==============================
    # MODEL COMPARISON GRAPH
    # ==============================
    st.subheader("Model Comparison")

    import pandas as pd

    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Logistic Regression": [
            log_metrics["Accuracy"],
            log_metrics["Precision"],
            log_metrics["Recall"],
            log_metrics["F1 Score"]
        ],
        "Decision Tree": [
            tree_metrics["Accuracy"],
            tree_metrics["Precision"],
            tree_metrics["Recall"],
            tree_metrics["F1 Score"]
        ]
    })

    comparison_df.set_index("Metric", inplace=True)

    fig3, ax3 = plt.subplots()
    comparison_df.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Score")
    ax3.set_title("Model Performance Comparison")
    ax3.set_ylim(0, 1)
    st.pyplot(fig3)

# ==============================
# UNSUPERVISED
# ==============================
elif menu == "Unsupervised Learning":

    (
        K_range,
        silhouette_scores,
        best_k,
        k_clusters,
        k_sil,
        h_clusters,
        h_sil
    ) = unsupervised_models(df_encoded)

    st.subheader("Silhouette Score vs K")

    fig, ax = plt.subplots()
    ax.plot(K_range, silhouette_scores, marker='o')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    st.pyplot(fig)

    st.success(f"Best K selected automatically: {best_k}")

    df_plot = df.copy()
    df_plot["KMeans Cluster"] = k_clusters
    df_plot["Hierarchical Cluster"] = h_clusters

    st.subheader("KMeans Clustering Graph")

    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        x="math score",
        y="reading score",
        hue="KMeans Cluster",
        data=df_plot,
        ax=ax1
    )
    st.pyplot(fig1)

    st.subheader("Hierarchical Clustering Graph")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x="math score",
        y="reading score",
        hue="Hierarchical Cluster",
        data=df_plot,
        ax=ax2
    )
    st.pyplot(fig2)

    st.write("KMeans Silhouette Score:", k_sil)
    st.write("Hierarchical Silhouette Score:", h_sil)

# ==============================
# PREDICTION
# ==============================
elif menu == "Prediction":

    log_model, tree_model, _, _ = supervised_models(df_encoded)

    st.subheader("Enter Student Details")

    gender = st.selectbox("Gender", df["gender"].unique())
    race = st.selectbox("Race", df["race/ethnicity"].unique())
    parent = st.selectbox("Parental Education", df["parental level of education"].unique())
    lunch = st.selectbox("Lunch", df["lunch"].unique())
    prep = st.selectbox("Test Preparation", df["test preparation course"].unique())
    math = st.number_input("Math Score", 0, 100, step=1)
    read = st.number_input("Reading Score", 0, 100, step=1)
    write = st.number_input("Writing Score", 0, 100, step=1)

    if st.button("Predict Performance"):

        # Create clean input dataframe
        input_data = pd.DataFrame({
            "gender": [gender],
            "race/ethnicity": [race],
            "parental level of education": [parent],
            "lunch": [lunch],
            "test preparation course": [prep],
            "math score": [math],
            "reading score": [read],
            "writing score": [write],
        })

        # Encode categorical columns
        for col in label_encoders:
            if col in input_data.columns:
                input_data[col] = label_encoders[col].transform(input_data[col])

       

        # Ensure correct column order
        # Match training features (exclude Performance and AverageScore)
        X_columns = df_encoded.drop(["Performance", "AverageScore"], axis=1).columns
        input_data = input_data[X_columns]

        # Predict
        log_pred = log_model.predict(input_data)[0]
        tree_pred = tree_model.predict(input_data)[0]

        # Decode prediction
        performance_encoder = label_encoders["Performance"]

        log_label = performance_encoder.inverse_transform([log_pred])[0]
        tree_label = performance_encoder.inverse_transform([tree_pred])[0]

        st.success(f"Logistic Regression Prediction: {log_label}")
        st.success(f"Decision Tree Prediction: {tree_label}")