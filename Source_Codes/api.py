import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "best_healthcare_model.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "Models", "label_encoder.pkl")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_FILE = os.path.join(FRONTEND_DIR, "index.html")


# =========================
# Check required files
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file not found: {ENCODER_PATH}")


# =========================
# Load model assets
# =========================
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
model_version = str(int(os.path.getmtime(MODEL_PATH)))


# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Healthcare Prediction API",
    description="Predict patient test results as Normal, Abnormal, or Inconclusive",
    version="1.0.0"
)


# Serve static frontend files if the folder exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# =========================
# Input schema
# =========================
class PatientData(BaseModel):
    age: float = Field(..., ge=0, le=120)
    gender: Literal["Male", "Female"]
    blood_type: Literal["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    medical_condition: str
    billing_amount: float = Field(..., ge=0)
    insurance_provider: str
    admission_type: Literal["Emergency", "Urgent", "Elective"]
    medication: str
    length_of_stay: int = Field(..., ge=0)


# =========================
# Routes
# =========================
@app.get("/")
def home():
    if os.path.exists(FRONTEND_FILE):
        return FileResponse(FRONTEND_FILE)

    return {
        "message": "Healthcare Prediction API is running",
        "docs": "/docs",
        "endpoint": "POST /predict"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": model_version
    }


@app.post("/predict")
def predict(data: PatientData):
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])

        prediction_encoded = model.predict(input_df)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        probabilities = {}
        confidence = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            class_indices = list(range(len(proba)))
            class_labels = label_encoder.inverse_transform(class_indices)

            probabilities = {
                label: round(float(prob), 4)
                for label, prob in zip(class_labels, proba)
            }

            confidence = round(float(max(proba)), 4)

        return {
            "predicted_test_result": prediction_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": model_version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))