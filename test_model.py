
import joblib
import pandas as pd

print("Starting CardioRisk model tests...\n")

# Test 1: Check whether the trained model loads
model = joblib.load("models/best_model.joblib")
print("TEST 1 PASSED: Model loaded successfully.")

# Test 2: Prepare sample input with all 13 features
features = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

sample = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 4,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 3
}], columns=features)

assert sample.shape == (1, 13)
print("TEST 2 PASSED: All 13 input features are available.")

# Test 3: Check model prediction
prediction = model.predict(sample)[0]

assert prediction in [0, 1]
print("TEST 3 PASSED: Model returned a valid prediction.")

# Test 4: Check prediction probability
probability = model.predict_proba(sample)[0][1]

assert 0 <= probability <= 1
print("TEST 4 PASSED: Prediction probability is valid.")

print("\nAll model tests passed successfully!")
