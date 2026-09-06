"""Tests del training pipeline (Tarea 3 - Arquitectura FTI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model.train import COLUMNAS_FEATURES, construir_pipeline_modelo
from src.pipelines.training_pipeline.training_pipeline import (
    dividir_train_test,
    ejecutar_training_pipeline,
    evaluar_modelo,
)

N_FILAS = 100
PROPORCION_TEST_ESPERADA = 0.2
TOLERANCIA_PROPORCION = 0.05
METRICA_VALOR_MINIMO = 0.0
METRICA_VALOR_MAXIMO = 1.0


@pytest.fixture
def df_features() -> pd.DataFrame:
    """DataFrame sintetico similar a la salida del feature pipeline: con
    nulos en un par de columnas, como los datos reales validados en la
    Tarea 2.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 15, N_FILAS),
            "Glucose": rng.uniform(70, 180, N_FILAS),
            "BloodPressure": rng.uniform(50, 100, N_FILAS),
            "SkinThickness": rng.uniform(10, 50, N_FILAS),
            "Insulin": rng.uniform(20, 300, N_FILAS),
            "BMI": rng.uniform(18, 45, N_FILAS),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 1.5, N_FILAS),
            "Age": rng.integers(21, 70, N_FILAS),
            "Outcome": rng.choice([True, False], N_FILAS, p=[0.35, 0.65]),
        }
    )
    df.loc[0:4, "Insulin"] = np.nan
    df.loc[10:11, "SkinThickness"] = np.nan
    return df


def test_construir_pipeline_modelo_estructura() -> None:
    """El pipeline debe tener los 3 pasos esperados, en orden."""
    pipeline = construir_pipeline_modelo()

    assert isinstance(pipeline, Pipeline)
    assert [nombre for nombre, _ in pipeline.steps] == [
        "preprocesador",
        "scaler",
        "model",
    ]


def test_pipeline_entrena_y_predice_con_nulos(df_features: pd.DataFrame) -> None:
    """El pipeline debe poder entrenarse y predecir sobre datos con nulos."""
    X = df_features[COLUMNAS_FEATURES]
    y = df_features["Outcome"]

    pipeline = construir_pipeline_modelo()
    pipeline.fit(X, y)
    predicciones = pipeline.predict(X)

    assert len(predicciones) == len(X)


def test_dividir_train_test_proporciones(df_features: pd.DataFrame) -> None:
    """El split debe respetar la proporcion 80/20 y mantener estratificacion."""
    _X_train, X_test, y_train, y_test = dividir_train_test(df_features)

    proporcion_test_real = len(X_test) / len(df_features)
    assert proporcion_test_real == pytest.approx(
        PROPORCION_TEST_ESPERADA, abs=TOLERANCIA_PROPORCION
    )

    proporcion_original = df_features["Outcome"].mean()
    assert y_train.mean() == pytest.approx(proporcion_original, abs=TOLERANCIA_PROPORCION)
    assert y_test.mean() == pytest.approx(proporcion_original, abs=TOLERANCIA_PROPORCION)


def test_evaluar_modelo_retorna_metricas_esperadas(df_features: pd.DataFrame) -> None:
    """evaluar_modelo debe retornar las 5 metricas esperadas, todas en [0,1]."""
    X = df_features[COLUMNAS_FEATURES]
    y = df_features["Outcome"]

    pipeline = construir_pipeline_modelo()
    pipeline.fit(X, y)
    metricas = evaluar_modelo(pipeline, X, y)

    assert set(metricas.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for valor in metricas.values():
        assert METRICA_VALOR_MINIMO <= valor <= METRICA_VALOR_MAXIMO


def test_ejecutar_training_pipeline_end_to_end(tmp_path: Path, df_features: pd.DataFrame) -> None:
    """Prueba de integracion: corre el pipeline completo sobre un parquet temporal."""
    features_path = tmp_path / "diabetes_clean.parquet"
    df_features.to_parquet(features_path, index=False)
    model_path = tmp_path / "modelo.joblib"
    metrics_path = tmp_path / "metricas_test.csv"

    _modelo, metricas = ejecutar_training_pipeline(
        features_path=features_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert set(metricas.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
