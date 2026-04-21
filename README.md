# Healthcare ML Prediction Project

## Project Description

This project is a **Machine Learning-powered Healthcare Prediction API** that predicts patient test results as:

- **Normal**
- **Abnormal**
- **Inconclusive**

The system uses trained ML models and exposes predictions via a **FastAPI backend**, with a **simple frontend UI** for user interaction.

The application is deployed on **Render**, allowing real-time predictions through a web interface.

---

## Project Structure
Healthcare_ML_Project/
│
├── Data/
│ └── healthcare_dataset.csv
│
├── Models/
│ ├── best_healthcare_model.pkl
│ ├── healthcare_model.pkl
│ └── label_encoder.pkl
│
├── Source_Codes/
│ ├── api.py
│ ├── Database.py
│ ├── Health_Care.py
│ ├── Pipeline.py
│ ├── Scheduler.py
│ └── Train_Model.py
│
├── frontend/
│ └── index.html
│
├── Deployment/
│ ├── render.yaml
│ ├── requirements.txt
│ └── pyproject.toml
│
├── README.md
└── .gitignore

## Setup Instructions

### 1. Clone the repository

```PowerShell

git clone https://github.com/Loi2008/Healthcare_ML_Project.git
cd Healthcare_ML_Project

pip install -r Deployment/requirements.tx

How to Run the API

Run the FastAPI server:

```PowerShell
python -m uvicorn Source_Codes.api:app --reload

Open in browser:

http://127.0.0.1:8000/

Available routes:
/ → Frontend UI
/docs → Swagger API docs
/predict → Prediction endpoint

Live Deployment

https://healthcare-ml-project-1-tklx.onrender.com

API Endpoint
POST /predict

Predict patient test results.

Example Request
{
  "age": 45,
  "gender": "Female",
  "blood_type": "A+",
  "medical_condition": "Diabetes",
  "billing_amount": 5000,
  "insurance_provider": "Aetna",
  "admission_type": "Emergency",
  "medication": "Aspirin",
  "length_of_stay": 7
}
Example Response
{
  "predicted_test_result": "Abnormal",
  "confidence": 0.3504,
  "probabilities": {
    "Abnormal": 0.3504,
    "Inconclusive": 0.3099,
    "Normal": 0.3397
  },
  "model_version": "1776762239"
}
Model Details
Models used:
XGBoost
Logistic Regression 
Decision Tree 

Evaluation metrics:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Model saved using pickle

Future Improvements
Prediction logging (database)
Model monitoring
Docker deployment
CI/CD pipeline
Enhanced frontend UI

👨Author
Developed as part of a Machine Learning internship project.