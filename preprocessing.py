import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ==============================
# LOAD DATA
# ==============================
def load_data():
    df = pd.read_csv("data/exams.csv")
    return df


# ==============================
# ENCODE DATA
# ==============================
def encode_data(df):

    # Make copy to avoid modifying original dataframe
    df = df.copy()

    # Create Average Score
    df["AverageScore"] = (
        df["math score"] +
        df["reading score"] +
        df["writing score"]
    ) / 3

    # Create Performance Category
    def categorize(score):
        if score >= 80:
            return "High"
        elif score >= 50:
            return "Medium"
        else:
            return "Low"

    df["Performance"] = df["AverageScore"].apply(categorize)

    label_encoders = {}

    # Encode ALL object columns (including Performance)
    for column in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    return df, label_encoders