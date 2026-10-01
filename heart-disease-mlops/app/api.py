"""API REST para predecir el riesgo de enfermedad cardíaca (Etapa 3)."""
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Rutas relativas a este archivo, para que la API funcione sin importar desde dónde se ejecute
APP_DIR = Path(__file__).resolve().parent
model = joblib.load(APP_DIR / "model.joblib")
metadata = json.loads((APP_DIR / "model_metadata.json").read_text(encoding="utf-8"))

COLUMNAS = metadata["columnas_entrada"]      # orden en que el Pipeline espera las variables
UMBRAL = metadata["umbral"]

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Predice la probabilidad de enfermedad cardíaca (heartdisease = 1) a partir de 11 variables clínicas.",
)

EJEMPLO = {
    "Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 289,
    "FastingBS": 1, "RestingECG": "ST", "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 2.0, "ST_Slope": "Flat",
}


class Input(BaseModel):
    """Datos de un paciente con los mismos nombres de columna del dataset."""

    Age: int = Field(..., ge=1, le=120, description="Edad (años)")
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: float = Field(..., ge=0, le=300, description="Presión arterial en reposo (mm Hg). 0 = no registrada")
    Cholesterol: float = Field(..., ge=0, le=1000, description="Colesterol sérico (mg/dl). 0 = no registrado")
    FastingBS: Literal[0, 1] = Field(..., description="1 si la glucosa en ayunas es > 120 mg/dl")
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: float = Field(..., ge=40, le=250, description="Frecuencia cardíaca máxima")
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(..., ge=-5, le=10, description="Depresión del segmento ST")
    ST_Slope: Literal["Up", "Flat", "Down"]

    model_config = {"json_schema_extra": {"examples": [EJEMPLO]}}


@app.get("/")
def root():
    return {"mensaje": "Heart Disease Prediction API. Documentación interactiva en /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {k: metadata[k] for k in ("modelo", "hiperparametros", "columnas_entrada", "umbral",
                                     "auc_cv", "metricas_test", "sklearn_version", "fecha_entrenamiento")}


@app.post("/predict")
def predict(data: Input):
    # El Pipeline recibe los datos crudos: la imputación de los ceros y el escalado van dentro del modelo
    X = pd.DataFrame([data.model_dump()])[COLUMNAS]
    proba = float(model.predict_proba(X)[0][1])
    return {"heart_disease_probability": proba, "prediction": int(proba > UMBRAL)}
