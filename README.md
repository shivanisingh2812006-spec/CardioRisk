# CardioRisk: A Machine Learning Based Heart Disease Risk Prediction Platform

## Project Overview

CardioRisk is a machine learning-based web application developed to demonstrate how machine learning can be used to analyse clinical attributes and estimate the likelihood of heart disease.

The platform allows users to register, log in, enter health-related details and receive a model-generated prediction. It also provides feature-based explanations, prediction history and downloadable PDF reports.

This project is developed for academic and educational purposes. It is not a medical diagnostic system.

## Project Links

- **GitHub Repository:** https://github.com/shivanisingh2812006-spec/CardioRisk
- **Live Demo:** https://cardiorisk-h35z.onrender.com/

## Key Features

- User registration and secure login.
- Heart disease prediction using 13 clinical input features.
- Comparison of five machine learning algorithms.
- Selection and saving of the best-performing model.
- SHAP-based explanations of model predictions.
- Prediction history stored in an SQLite database.
- REST API for prediction requests.
- Downloadable PDF prediction reports.
- Model metadata and version tracking.
- Basic model monitoring and automated testing.
- Docker-based application deployment.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Pandas, NumPy, Scikit-learn, SHAP
- **Web Framework:** Flask
- **Database:** SQLite, Flask-SQLAlchemy
- **Visualisation:** Matplotlib
- **Model Storage:** Joblib
- **Report Generation:** ReportLab
- **Frontend:** HTML, CSS, Bootstrap
- **Containerisation:** Docker
- **Deployment:** Render
- **Version Control:** Git and GitHub

## Machine Learning Algorithms

The project compares the following five algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbours (KNN)

The models are evaluated using accuracy, precision, recall, F1-score and ROC-AUC. Cross-validation ROC-AUC is used for model selection.

Logistic Regression was selected as the best-performing model based on the evaluation process.

## Dataset

The project uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

- **Dataset size:** 303 records
- **Input features:** 13
- **Target:** Binary classification
- Original target values are converted into two classes:
  - 0: No heart disease
  - 1: Heart disease present

## Project Structure

```text
CardioRisk/
│
├── dataset/
│   └── heart.csv
│
├── models/
│   ├── best_model.joblib
│   ├── model_comparison.csv
│   ├── model_metadata.json
│   └── shap_importance.png
│
├── templates/
│   ├── base.html
│   ├── register.html
│   ├── login.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   └── history.html
│
├── app.py
├── download_dataset.py
├── train_model.py
├── explain_model.py
├── model_metadata.py
├── test_model.py
├── monitor_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shivanisingh2812006-spec/CardioRisk.git
cd CardioRisk
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Download the Dataset

```bash
python download_dataset.py
```

### 6. Train the Machine Learning Model

```bash
python train_model.py
```

### 7. Generate SHAP Explanations

```bash
python explain_model.py
```

### 8. Run Model Tests

```bash
python test_model.py
```

### 9. Generate Model Metadata

```bash
python model_metadata.py
```

### 10. Start the Flask Application

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

### 11. Run Model Monitoring

After starting the application and generating prediction records, open another terminal with the virtual environment activated and run:

```bash
python monitor_model.py
```

## REST API

CardioRisk also provides a REST API endpoint.

**Endpoint:** `POST /api/predict`

The endpoint accepts the 13 clinical input features in JSON format and returns:

- Model prediction
- Model-estimated probability
- Prediction category
- Educational disclaimer message

## MLOps Components

The project demonstrates several MLOps practices:

- Dataset preparation and preprocessing.
- Model training and comparison.
- Model evaluation and selection.
- Model version metadata.
- Automated model testing.
- SHAP-based model explainability.
- REST API deployment interface.
- Basic prediction monitoring.
- Docker-based deployment.

## Deployment

The CardioRisk web application is deployed on Render using Docker.

**Live Application:** https://cardiorisk-h35z.onrender.com/

### Storage Limitation

The application uses SQLite on Render's free hosting plan. Since persistent disks are not available on this plan, database records may be lost when the service restarts or is redeployed.

## Disclaimer

CardioRisk is an academic machine learning project. Its predictions and probability values are generated by a model trained on a limited historical dataset. They are not clinically validated and must not be used for medical decisions, diagnosis or treatment.

## Developed For

**Academic Mini Project – Computer Science and Engineering**

- **Project Name:** CardioRisk
- **University:** Chandigarh University