import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from Source_Codes.Pipeline import run_pipeline
from Source_Codes.Train_Model import train_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "healthcare_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_healthcare_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


app = FastAPI(
    title="Healthcare Test Results Prediction API",
    description="Predicts patient test results as Normal, Abnormal, or Inconclusive",
    version="1.0.0"
)


def ensure_model_assets():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        print(" Existing model assets found. Loading...")
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        return model, label_encoder

    print(" Model file not found. Training a new model from dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    raw_data = pd.read_csv(DATA_PATH)
    cleaned_data = run_pipeline(raw_data)

    model, label_encoder = train_model(cleaned_data, save_dir=MODELS_DIR)
    return model, label_encoder


model, label_encoder = ensure_model_assets()


class PatientData(BaseModel):
    age: float = Field(..., ge=0, le=120)
    gender: Literal["Male", "Female"]
    blood_type: Literal["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    medical_condition: str
    billing_amount: float = Field(..., ge=0)
    admission_type: Literal["Emergency", "Urgent", "Elective"]
    medication: str
    length_of_stay: int = Field(..., ge=0)


@app.get("/")
def root():
    return {
        "message": "Healthcare Prediction API is running",
        "endpoint": "POST /predict"
    }


@app.post("/predict")
def predict(data: PatientData):
    input_data = pd.DataFrame([data.model_dump()])

    prediction_encoded = model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    return {"prediction": prediction_label}