
import os
import requests
import pandas as pd
from io import StringIO

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

print("Downloading Heart Disease dataset...")

response = requests.get(url, timeout=30)
response.raise_for_status()

df = pd.read_csv(
    StringIO(response.text),
    names=columns,
    na_values="?",
    skipinitialspace=True
)

for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["target"])
df["target"] = (df["target"] > 0).astype(int)

os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/heart.csv", index=False)

print("\nDataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df["target"].value_counts())
print("\nSaved at: dataset/heart.csv")
