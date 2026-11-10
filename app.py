
import os
import json
from io import BytesIO

import joblib
import pandas as pd
import shap

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, send_file
)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "local-only-cardiorisk-key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///cardiorisk.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

model = joblib.load("models/best_model.joblib")


FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]


FIELD_LABELS = {
    "age": "Age",
    "sex": "Sex (1 = Male, 0 = Female)",
    "cp": "Chest pain type (1-4)",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar (1 = True, 0 = False)",
    "restecg": "Resting ECG (0-2)",
    "thalach": "Maximum heart rate",
    "exang": "Exercise-induced angina (1 = Yes, 0 = No)",
    "oldpeak": "ST depression",
    "slope": "ST slope (0-2)",
    "ca": "Major vessels (0-3)",
    "thal": "Thalassemia category (3, 6, or 7)"
}


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    input_data = db.Column(db.Text, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("Please fill in all fields.")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must contain at least 8 characters.")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("This username is already taken.")
            return redirect(url_for("register"))

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password)
        )

        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please log in.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session.clear()
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    records = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Prediction.created_at.desc()).all()

    return render_template("dashboard.html", records=records)


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        try:
            values = {}

            for feature in FEATURES:
                values[feature] = float(request.form[feature])

            if not 1 <= values["age"] <= 120:
                raise ValueError("Age must be between 1 and 120.")

            if values["sex"] not in (0, 1):
                raise ValueError("Sex must be 0 or 1.")

            if values["cp"] not in (1, 2, 3, 4):
                raise ValueError("Chest pain type must be between 1 and 4.")

            if not 50 <= values["trestbps"] <= 300:
                raise ValueError("Invalid resting blood pressure.")

            if not 50 <= values["chol"] <= 700:
                raise ValueError("Invalid cholesterol value.")

            if values["fbs"] not in (0, 1):
                raise ValueError("Fasting blood sugar must be 0 or 1.")

            if values["restecg"] not in (0, 1, 2):
                raise ValueError("Invalid resting ECG category.")

            if not 40 <= values["thalach"] <= 250:
                raise ValueError("Invalid maximum heart rate.")

            if values["exang"] not in (0, 1):
                raise ValueError("Exercise angina must be 0 or 1.")

            if not 0 <= values["oldpeak"] <= 10:
                raise ValueError("Invalid ST depression value.")

            if values["slope"] not in (0, 1, 2):
                raise ValueError("Invalid ST slope category.")

            if values["ca"] not in (0, 1, 2, 3):
                raise ValueError("Major vessels must be between 0 and 3.")

            if values["thal"] not in (3, 6, 7):
                raise ValueError("Thal category must be 3, 6, or 7.")

            input_df = pd.DataFrame([values], columns=FEATURES)

            prediction = int(model.predict(input_df)[0])

            probability = float(
                model.predict_proba(input_df)[0][1]
            )

            # Individual SHAP explanation

            background_df = pd.read_csv("dataset/heart.csv")

            background_df = background_df[FEATURES].sample(
                n=min(30, len(background_df)),
                random_state=42
            )

            explainer = shap.Explainer(
                model.predict_proba,
                background_df,
                algorithm="permutation"
            )

            shap_explanation = explainer(
                input_df,
                max_evals=(2 * len(FEATURES) * 10 + 1)
            )

            shap_values = shap_explanation.values

            if len(shap_values.shape) == 3:
                feature_contributions = shap_values[0, :, 1]
            else:
                feature_contributions = shap_values[0]

            explanation_rows = []

            for feature, contribution in zip(
                FEATURES, feature_contributions
            ):

                explanation_rows.append({
                    "feature": FIELD_LABELS[feature],
                    "value": values[feature],
                    "contribution": float(contribution),
                    "direction": (
                        "Pushes toward positive class"
                        if contribution > 0
                        else "Pushes toward negative class"
                    )
                })

            explanation_rows.sort(
                key=lambda item: abs(item["contribution"]),
                reverse=True
            )

            if probability < 0.35:
                category = "Lower model-estimated likelihood"

            elif probability < 0.65:
                category = "Moderate model-estimated likelihood"

            else:
                category = "Higher model-estimated likelihood"

            record = Prediction(
                user_id=session["user_id"],
                input_data=json.dumps(values),
                probability=probability,
                category=category
            )

            db.session.add(record)
            db.session.commit()

            return render_template(
                "result.html",
                probability=probability,
                category=category,
                prediction=prediction,
                record_id=record.id,
                labels=FIELD_LABELS,
                values=values,
                explanations=explanation_rows
            )

        except (ValueError, KeyError) as error:
            flash(str(error))
            return redirect(url_for("predict"))

    return render_template("predict.html")


@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect(url_for("login"))

    records = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(Prediction.created_at.desc()).all()

    return render_template("history.html", records=records)


@app.route("/report/<int:record_id>")
def report(record_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    record = Prediction.query.filter_by(
        id=record_id,
        user_id=session["user_id"]
    ).first_or_404()

    values = json.loads(record.input_data)

    buffer = BytesIO()

    pdf = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()

    content = [
        Paragraph("CardioRisk - Prediction Summary", styles["Title"]),
        Spacer(1, 15),

        Paragraph(
            "Educational Machine Learning Prediction Report",
            styles["Normal"]
        ),

        Spacer(1, 15),

        Paragraph(
            f"Model-estimated likelihood: {record.probability * 100:.2f}%",
            styles["Heading2"]
        ),

        Paragraph(
            f"Category: {record.category}",
            styles["Normal"]
        ),

        Spacer(1, 15),

        Paragraph("Patient Input Details", styles["Heading2"])
    ]

    table_data = [["Feature", "Entered Value"]]

    for feature, value in values.items():
        table_data.append([FIELD_LABELS[feature], str(value)])

    content.append(Table(table_data, repeatRows=1))

    content.append(Spacer(1, 20))

    content.append(Paragraph(
        "Disclaimer: This report is generated by an educational "
        "machine-learning project using historical data. It is not a "
        "medical diagnosis or clinically validated risk assessment. "
        "Consult a qualified healthcare professional for medical advice.",
        styles["Normal"]
    ))

    pdf.build(content)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="CardioRisk_Report.pdf",
        mimetype="application/pdf"
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        if not data:
            return {
                "error": "Please provide input data in JSON format."
            }, 400

        values = {}

        for feature in FEATURES:
            if feature not in data:
                return {
                    "error": f"Missing feature: {feature}"
                }, 400

            values[feature] = float(data[feature])

        if not 1 <= values["age"] <= 120:
            raise ValueError("Age must be between 1 and 120.")
        if values["sex"] not in (0, 1):
            raise ValueError("Sex must be 0 or 1.")
        if values["cp"] not in (1, 2, 3, 4):
            raise ValueError("Chest pain type must be between 1 and 4.")
        if not 50 <= values["trestbps"] <= 300:
            raise ValueError("Invalid resting blood pressure.")
        if not 50 <= values["chol"] <= 700:
            raise ValueError("Invalid cholesterol value.")
        if values["fbs"] not in (0, 1):
            raise ValueError("Fasting blood sugar must be 0 or 1.")
        if values["restecg"] not in (0, 1, 2):
            raise ValueError("Invalid resting ECG category.")
        if not 40 <= values["thalach"] <= 250:
            raise ValueError("Invalid maximum heart rate.")
        if values["exang"] not in (0, 1):
            raise ValueError("Exercise angina must be 0 or 1.")
        if not 0 <= values["oldpeak"] <= 10:
            raise ValueError("Invalid ST depression value.")
        if values["slope"] not in (0, 1, 2):
            raise ValueError("Invalid ST slope category.")
        if values["ca"] not in (0, 1, 2, 3):
            raise ValueError("Major vessels must be between 0 and 3.")
        if values["thal"] not in (3, 6, 7):
            raise ValueError("Thal category must be 3, 6, or 7.")

        input_df = pd.DataFrame([values], columns=FEATURES)

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        if probability < 0.35:
            category = "Lower model-estimated likelihood"
        elif probability < 0.65:
            category = "Moderate model-estimated likelihood"
        else:
            category = "Higher model-estimated likelihood"

        return {
            "prediction": prediction,
            "model_estimated_probability": round(probability * 100, 2),
            "category": category,
            "message": "Educational model output only. Not a medical diagnosis."
        }, 200

    except (ValueError, TypeError) as error:
        return {"error": str(error)}, 400

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)