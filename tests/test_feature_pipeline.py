"""Pruebas unitarias para src/data/cleaning.py y el feature pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import (
    convertir_tipos,
    corregir_escala_dpf,
    eliminar_filas_invalidas,
    unificar_valores_faltantes,
)
from src.pipelines.feature_pipeline.feature_pipeline import ejecutar_feature_pipeline

GLUCOSE_VALOR_VALIDO = 120
FILAS_ESPERADAS_TRAS_LIMPIEZA = 2


@pytest.fixture
def df_crudo() -> pd.DataFrame:
    """DataFrame sintetico que cubre los casos que el pipeline debe manejar."""
    return pd.DataFrame(
        {
            "Pregnancies": [1, 2, 2, 0],
            "Glucose": [120, 0, 0, 130],
            "BloodPressure": [70, 80, 80, 0],
            "SkinThickness": [20, 0, 0, 25],
            "Insulin": [85, 0, 0, 100],
            "BMI": [28.5, 0.0, 0.0, 32.0],
            "DiabetesPedigreeFunction": [0.627, 627.0, 627.0, 1.5],
            "Age": [45, 30, 30, 22],
            "Outcome": [1, 0, 0, np.nan],
        }
    )


def test_unificar_valores_faltantes_convierte_ceros_a_nan(
    df_crudo: pd.DataFrame,
) -> None:
    resultado = unificar_valores_faltantes(df_crudo)

    assert pd.isna(resultado.loc[1, "Glucose"])
    assert pd.isna(resultado.loc[3, "BloodPressure"])
    assert resultado.loc[0, "Glucose"] == GLUCOSE_VALOR_VALIDO


def test_corregir_escala_dpf_divide_valores_mal_formateados(
    df_crudo: pd.DataFrame,
) -> None:
    resultado = corregir_escala_dpf(df_crudo)

    assert resultado.loc[1, "DiabetesPedigreeFunction"] == pytest.approx(0.627)
    assert resultado.loc[2, "DiabetesPedigreeFunction"] == pytest.approx(0.627)
    assert resultado.loc[0, "DiabetesPedigreeFunction"] == pytest.approx(0.627)
    assert resultado.loc[3, "DiabetesPedigreeFunction"] == pytest.approx(1.5)


def test_convertir_tipos_asigna_tipos_nullable(df_crudo: pd.DataFrame) -> None:
    df_sin_ceros = unificar_valores_faltantes(df_crudo)
    resultado = convertir_tipos(df_sin_ceros)

    assert str(resultado["Pregnancies"].dtype) == "Int8"
    assert str(resultado["Age"].dtype) == "Int8"
    assert resultado["Glucose"].dtype == "float64"
    assert str(resultado["Outcome"].dtype) == "boolean"


def test_eliminar_filas_invalidas_quita_outcome_nulo_y_duplicados(
    df_crudo: pd.DataFrame,
) -> None:
    resultado = eliminar_filas_invalidas(df_crudo)

    assert not resultado["Outcome"].isna().any()
    assert len(resultado) == FILAS_ESPERADAS_TRAS_LIMPIEZA


def test_ejecutar_feature_pipeline_end_to_end(tmp_path: Path, df_crudo: pd.DataFrame) -> None:
    """Prueba de integracion: corre el pipeline completo sobre un CSV temporal."""
    input_csv = tmp_path / "diabetes_crudo.csv"
    df_crudo.to_csv(input_csv, index=False)
    output_parquet = tmp_path / "diabetes_clean.parquet"

    resultado = ejecutar_feature_pipeline(input_path=input_csv, output_path=output_parquet)

    assert output_parquet.exists()
    assert len(resultado) == FILAS_ESPERADAS_TRAS_LIMPIEZA
    assert not resultado["Outcome"].isna().any()
