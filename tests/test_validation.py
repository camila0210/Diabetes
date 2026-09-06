"""Tests de la validacion de esquema con Pandera (Tarea 2 - Arquitectura FTI)."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.validation import validar_esquema

N_FILAS = 20
GLUCOSE_VALOR_INVALIDO = 500
PORCENTAJE_NULOS_EXCESIVO = 0.30


@pytest.fixture
def df_valido() -> pd.DataFrame:
    """DataFrame de features valido: sin nulos, sin duplicados, en rango."""
    df = pd.DataFrame(
        {
            "Pregnancies": list(range(N_FILAS)),
            "Glucose": [100.0 + i for i in range(N_FILAS)],
            "BloodPressure": [70.0 + i for i in range(N_FILAS)],
            "SkinThickness": [20.0 + i for i in range(N_FILAS)],
            "Insulin": [100.0 + i * 5 for i in range(N_FILAS)],
            "BMI": [25.0 + i * 0.5 for i in range(N_FILAS)],
            "DiabetesPedigreeFunction": [0.3 + i * 0.01 for i in range(N_FILAS)],
            "Age": [25 + i for i in range(N_FILAS)],
            "Outcome": [i % 2 == 0 for i in range(N_FILAS)],
        }
    )
    df["Pregnancies"] = df["Pregnancies"].astype("Int8")
    df["Age"] = df["Age"].astype("Int8")
    df["Outcome"] = df["Outcome"].astype("boolean")
    return df


def test_esquema_acepta_dataframe_valido(df_valido: pd.DataFrame) -> None:
    """Un DataFrame que cumple todas las reglas debe pasar sin errores."""
    df_validado = validar_esquema(df_valido)

    assert len(df_validado) == N_FILAS


def test_esquema_rechaza_valor_fuera_de_rango(df_valido: pd.DataFrame) -> None:
    """Un valor de Glucose fuera de rango debe hacer fallar la validacion."""
    df_invalido = df_valido.copy()
    df_invalido.loc[0, "Glucose"] = GLUCOSE_VALOR_INVALIDO

    with pytest.raises(pa.errors.SchemaErrors):
        validar_esquema(df_invalido)


def test_esquema_rechaza_outcome_nulo(df_valido: pd.DataFrame) -> None:
    """Outcome no puede ser nulo: esas filas no sirven para entrenar/evaluar."""
    df_invalido = df_valido.copy()
    df_invalido.loc[0, "Outcome"] = pd.NA

    with pytest.raises(pa.errors.SchemaErrors):
        validar_esquema(df_invalido)


def test_esquema_rechaza_duplicados(df_valido: pd.DataFrame) -> None:
    """Filas duplicadas deben hacer fallar la validacion."""
    df_invalido = pd.concat([df_valido, df_valido.iloc[[0]]], ignore_index=True)

    with pytest.raises(pa.errors.SchemaErrors):
        validar_esquema(df_invalido)


def test_esquema_rechaza_exceso_nulos(df_valido: pd.DataFrame) -> None:
    """Un porcentaje de nulos por encima del umbral debe fallar."""
    df_invalido = df_valido.copy()
    n_nulos = int(N_FILAS * PORCENTAJE_NULOS_EXCESIVO)
    df_invalido.loc[: n_nulos - 1, "Glucose"] = None

    with pytest.raises(pa.errors.SchemaErrors):
        validar_esquema(df_invalido)
