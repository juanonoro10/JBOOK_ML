"""Pruebas del modelo exportado (Etapa 5): que cargue, que sea coherente con sus metadatos y que no pierda calidad."""
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELO = joblib.load(ROOT / "app" / "model.joblib")
METADATOS = json.loads((ROOT / "app" / "model_metadata.json").read_text(encoding="utf-8"))
TEST_CSV = ROOT / "data" / "test.csv"


def test_es_un_pipeline_con_preprocesamiento():
    assert list(MODELO.named_steps) == ["prep", "clf"]
    assert hasattr(MODELO, "predict_proba")


def test_columnas_coinciden_con_metadatos():
    assert list(MODELO.feature_names_in_) == METADATOS["columnas_entrada"]


def test_version_de_sklearn_coincide_con_la_del_entrenamiento():
    # un joblib solo carga de forma confiable con la misma versión de scikit-learn
    assert sklearn.__version__ == METADATOS["sklearn_version"]


@pytest.mark.skipif(not TEST_CSV.exists(), reason="data/test.csv no está disponible")
def test_probabilidades_validas_y_auc_minimo():
    test = pd.read_csv(TEST_CSV)
    X, y = test.drop(columns="HeartDisease"), test["HeartDisease"]
    proba = MODELO.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    # si un reentrenamiento baja el AUC de 0.90, el CI falla y avisa
    assert roc_auc_score(y, proba) >= 0.90
