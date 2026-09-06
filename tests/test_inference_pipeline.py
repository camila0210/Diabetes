"""Tests del inference pipeline (Tarea 6 - Arquitectura FTI)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from sklearn.pipeline import Pipeline

from src.data.validation import validar_esquema_inferencia
from src.inference.predict import predecir
from src.model.train import construir_pipeline_modelo
from src.pipelines.inference_pipeline.inference_pipeline import (
    ejecutar_inference_pipeline,
)

N_FILAS_ENTRENAMIENTO = 30
GLUCOSE_VALOR_INVALIDO = 500


@pytest.fixture
def modelo_entrenado() -> Pipeline:
    """El mismo pipeline de produccion, entrenado con datos sinteticos
    minimos, solo para probar la mecanica del inference pipeline.
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, N_FILAS_ENTRENAMIENTO),
            "Glucose": rng.uniform(80, 160, N_FILAS_ENTRENAMIENTO),
            "BloodPressure": rng.uniform(60, 90, N_FILAS_ENTRENAMIENTO),
            "SkinThickness": rng.uniform(15, 40, N_FILAS_ENTRENAMIENTO),
            "Insulin": rng.uniform(50, 200, N_FILAS_ENTRENAMIENTO),
            "BMI": rng.uniform(20, 40, N_FILAS_ENTRENAMIENTO),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 1.0, N_FILAS_ENTRENAMIENTO),
            "Age": rng.integers(21, 60, N_FILAS_ENTRENAMIENTO),
        }
    )
    y = rng.choice([True, False], N_FILAS_ENTRENAMIENTO)

    modelo = construir_pipeline_modelo()
    modelo.fit(X, y)
    return modelo


@pytest.fixture
def df_nuevos() -> pd.DataFrame:
    """Datos nuevos validos para predecir (sin columna Outcome)."""
    return pd.DataFrame(
        {
            "Pregnancies": [2, 5],
            "Glucose": [110.0, 140.0],
            "BloodPressure": [70.0, 80.0],
            "SkinThickness": [25.0, 30.0],
            "Insulin": [100.0, 150.0],
            "BMI": [28.0, 33.0],
            "DiabetesPedigreeFunction": [0.4, 0.6],
            "Age": [30, 45],
        }
    )


def test_validar_esquema_inferencia_acepta_datos_sin_outcome(
    df_nuevos: pd.DataFrame,
) -> None:
    """El esquema de inferencia no debe exigir la columna Outcome."""
    df_validado = validar_esquema_inferencia(df_nuevos)

    assert len(df_validado) == len(df_nuevos)


def test_validar_esquema_inferencia_rechaza_fuera_de_rango(
    df_nuevos: pd.DataFrame,
) -> None:
    """Un valor de Glucose fuera de rango debe fallar la validacion."""
    df_invalido = df_nuevos.copy()
    df_invalido.loc[0, "Glucose"] = GLUCOSE_VALOR_INVALIDO

    with pytest.raises(pa.errors.SchemaErrors):
        validar_esquema_inferencia(df_invalido)


def test_predecir_retorna_columnas_esperadas(
    modelo_entrenado: Pipeline, df_nuevos: pd.DataFrame
) -> None:
    """predecir debe agregar columnas 'prediccion' y 'probabilidad'."""
    resultado = predecir(modelo_entrenado, df_nuevos)

    assert "prediccion" in resultado.columns
    assert "probabilidad" in resultado.columns
    assert len(resultado) == len(df_nuevos)
    assert resultado["probabilidad"].between(0, 1).all()


def test_ejecutar_inference_pipeline_end_to_end(
    tmp_path: Path, modelo_entrenado: Pipeline, df_nuevos: pd.DataFrame
) -> None:
    """Prueba de integracion: carga modelo y datos desde archivos, predice y guarda."""
    model_path = tmp_path / "modelo.joblib"
    joblib.dump(modelo_entrenado, model_path)

    input_path = tmp_path / "nuevos.csv"
    df_nuevos.to_csv(input_path, index=False)

    output_path = tmp_path / "predicciones.csv"

    resultado = ejecutar_inference_pipeline(
        model_path=model_path, input_path=input_path, output_path=output_path
    )

    assert output_path.exists()
    assert "prediccion" in resultado.columns
    assert "probabilidad" in resultado.columns
