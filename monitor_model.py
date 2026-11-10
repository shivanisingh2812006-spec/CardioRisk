
import os
import json
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

DATABASE_PATH = "instance/cardiorisk.db"

print("Starting CardioRisk Model Monitoring...\n")

if not os.path.exists(DATABASE_PATH):
    print("Database file not found.")
    print("First, run the Flask application and generate a prediction.")
    raise SystemExit

engine = create_engine(f"sqlite:///{DATABASE_PATH}")

try:
    records = pd.read_sql(
        "SELECT * FROM prediction",
        engine
    )

    if records.empty:
        print("No prediction records available yet.")
        raise SystemExit

    records["input_data"] = records["input_data"].apply(json.loads)

    print("MODEL MONITORING REPORT")
    print("-----------------------")

    print("Total predictions:", len(records))

    print(
        "Average model-estimated probability:",
        round(records["probability"].mean() * 100, 2), "%"
    )

    print("\nPrediction categories:")
    print(records["category"].value_counts())

    print(
        "\nLatest prediction date:",
        records["created_at"].max()
    )

    print("\nMonitoring completed successfully!")

except Exception as error:
    print("Monitoring error:", error)

finally:
    engine.dispose()
