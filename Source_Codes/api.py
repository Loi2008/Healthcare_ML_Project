from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Healthcare Test Results Prediction API",
    description="Predicts patient test results as Normal, Abnormal, or Inconclusive",
    version="1.0.0"
)

# Load trained model and label encoder once when the API starts
model = joblib.load("best_healthcare_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# Input schema for prediction
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
    # Convert incoming JSON to DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Predict encoded class
    prediction_encoded = model.predict(input_data)[0]

    # Convert encoded class back to original label
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    return {
        "prediction": prediction_label
    }