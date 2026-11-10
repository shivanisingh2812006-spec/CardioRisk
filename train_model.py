
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Define five algorithms
algorithms = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
trained_models = {}

print("Training and comparing five ML models...\n")

for name, algorithm in algorithms.items():

    # Scale data for distance-based and linear models
    if name in ["Logistic Regression", "SVM", "KNN"]:
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", algorithm)
        ]
    else:
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", algorithm)
        ]

    pipeline = Pipeline(steps)

    # Five-fold cross-validation on training data
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring="roc_auc"
    )

    # Train and evaluate on the held-out test data
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "CV ROC-AUC": round(cv_scores.mean(), 4),
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "F1 Score": round(
            f1_score(y_test, predictions, zero_division=0), 4
        ),
        "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4)
    })

    trained_models[name] = pipeline

    print(name, "completed.")

# Compare models
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="CV ROC-AUC", ascending=False
)

print("\nMODEL COMPARISON:")
print(comparison.to_string(index=False))

# Select model using training cross-validation score
best_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

# Save selected model and comparison results
joblib.dump(best_model, "models/best_model.joblib")
comparison.to_csv("models/model_comparison.csv", index=False)

print("\nBest model:", best_name)
print("Saved model: models/best_model.joblib")
print("Saved comparison: models/model_comparison.csv")
print("\nModel training completed successfully!")
