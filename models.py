from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score
)
import numpy as np


# ==============================
# SUPERVISED MODELS
# ==============================
def supervised_models(df):

    # Features and Target
    X = df.drop(["Performance", "AverageScore"], axis=1)
    y = df["Performance"]

    # Stratified split (VERY IMPORTANT for classification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------
    # Logistic Regression
    # --------------------------
    log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
    )
    log_model.fit(X_train, y_train)
    log_pred = log_model.predict(X_test)

    # --------------------------
    # Decision Tree
    # --------------------------
    tree_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)

    # --------------------------
    # Evaluation Function
    # --------------------------
    def evaluate(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Confusion Matrix": confusion_matrix(y_true, y_pred)
        }

    log_metrics = evaluate(y_test, log_pred)
    tree_metrics = evaluate(y_test, tree_pred)

    return log_model, tree_model, log_metrics, tree_metrics


# ==============================
# UNSUPERVISED MODELS
# ==============================
def unsupervised_models(df):

    data = df[["math score", "reading score", "writing score"]]

    silhouette_scores = []
    K_range = range(2, 7)

    # Compute silhouette scores for K = 2 to 6
    for k in K_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10   # prevents sklearn warning
        )
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)

    # Choose best K
    best_k = list(K_range)[np.argmax(silhouette_scores)]

    # Final KMeans
    final_kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )
    k_clusters = final_kmeans.fit_predict(data)
    k_silhouette = silhouette_score(data, k_clusters)

    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=best_k)
    h_clusters = hierarchical.fit_predict(data)
    h_silhouette = silhouette_score(data, h_clusters)

    return (
        K_range,
        silhouette_scores,
        best_k,
        k_clusters,
        k_silhouette,
        h_clusters,
        h_silhouette
    )