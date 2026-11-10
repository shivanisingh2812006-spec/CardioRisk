
import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

print("Loading trained model...")

df = pd.read_csv("dataset/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = joblib.load("models/best_model.joblib")

# Apply the same preprocessing used during training
preprocessor = pipeline[:-1]
classifier = pipeline.named_steps["model"]

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Creating SHAP explanations...")

explainer = shap.Explainer(classifier, X_train_processed)
shap_values = explainer(X_test_processed)

os.makedirs("models", exist_ok=True)

plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("models/shap_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("SHAP explanation completed!")
print("Saved: models/shap_importance.png")
