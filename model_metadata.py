
import json
import os
from datetime import datetime

import pandas as pd

MODEL_DIR = "models"
COMPARISON_FILE = os.path.join(MODEL_DIR, "model_comparison.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "best_model.joblib")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

# Read model comparison results
results = pd.read_csv(COMPARISON_FILE)

# Find the model with the highest cross-validation ROC-AUC
best_model = results.loc[results["CV ROC-AUC"].idxmax()]

metadata = {
    "project_name": "CardioRisk",
    "model_version": "1.0.0",
    "algorithm": str(best_model["Model"]),
    "selection_metric": "CV ROC-AUC",
    "cv_roc_auc": float(best_model["CV ROC-AUC"]),
    "accuracy": float(best_model["Accuracy"]),
    "precision": float(best_model["Precision"]),
    "recall": float(best_model["Recall"]),
    "f1_score": float(best_model["F1 Score"]),
    "test_roc_auc": float(best_model["ROC-AUC"]),
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "UCI Cleveland Heart Disease Dataset",
    "dataset_rows": 303,
    "number_of_features": 13,
    "model_file": MODEL_FILE
}

# Save metadata as JSON
with open(METADATA_FILE, "w") as file:
    json.dump(metadata, file, indent=4)

print("Model metadata created successfully!")
print(json.dumps(metadata, indent=4))
print(f"\nSaved at: {METADATA_FILE}")
