"""Pruebas de la API (Etapa 5). Usan el TestClient de FastAPI, así que no hace falta levantar el servidor."""
import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

ALTO_RIESGO = {
    "Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 289,
    "FastingBS": 1, "RestingECG": "ST", "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 2.0, "ST_Slope": "Flat",
}
BAJO_RIESGO = {
    "Age": 40, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 120, "Cholesterol": 200,
    "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 172, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert len(r.json()["columnas_entrada"]) == 11


def test_predict_formato_de_respuesta():
    r = client.post("/predict", json=ALTO_RIESGO)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"heart_disease_probability", "prediction"}
    assert 0.0 <= cuerpo["heart_disease_probability"] <= 1.0
    assert cuerpo["prediction"] == int(cuerpo["heart_disease_probability"] > 0.5)


def test_predict_distingue_alto_y_bajo_riesgo():
    alto = client.post("/predict", json=ALTO_RIESGO).json()
    bajo = client.post("/predict", json=BAJO_RIESGO).json()
    assert alto["prediction"] == 1
    assert bajo["prediction"] == 0
    assert alto["heart_disease_probability"] > bajo["heart_disease_probability"]


def test_colesterol_en_cero_se_trata_como_faltante():
    # un 0 es un dato no registrado; el Pipeline lo imputa y la API no debe fallar
    r = client.post("/predict", json={**ALTO_RIESGO, "Cholesterol": 0, "RestingBP": 0})
    assert r.status_code == 200


@pytest.mark.parametrize("campo, valor", [
    ("Sex", "X"),                 # categoría que no existe
    ("ChestPainType", "otro"),
    ("Age", 300),                 # fuera de rango
    ("Cholesterol", -10),
    ("MaxHR", "alto"),            # tipo incorrecto
])
def test_valores_invalidos_devuelven_422(campo, valor):
    r = client.post("/predict", json={**ALTO_RIESGO, campo: valor})
    assert r.status_code == 422


def test_campo_faltante_devuelve_422():
    incompleto = {k: v for k, v in ALTO_RIESGO.items() if k != "ST_Slope"}
    assert client.post("/predict", json=incompleto).status_code == 422
