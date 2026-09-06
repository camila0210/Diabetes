"""Validación de esquema del DataFrame de features con Pandera.

Valida el DataFrame que produce el feature pipeline (Tarea 1) antes de
guardarlo: tipos de datos, rangos clínicamente razonables y porcentaje
máximo de nulos esperado por columna.

La imputación de los nulos NO se hace aquí: se hace en el training
pipeline, después del split train/test, para evitar data leakage.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

# Porcentaje máximo de nulos tolerado por columna.
# Fijado con margen sobre el % real observado en los datos limpios
# (ver notas de la Tarea 2). Insulin y SkinThickness tienen muchos nulos
# porque una gran parte de los registros originales tenían ceros
# clínicamente inválidos en esas columnas (unificados a NaN en cleaning.py).
PORCENTAJE_MAXIMO_NULOS = {
    "Pregnancies": 0.05,
    "Glucose": 0.05,
    "BloodPressure": 0.10,
    "SkinThickness": 0.40,
    "Insulin": 0.60,
    "BMI": 0.05,
    "DiabetesPedigreeFunction": 0.05,
    "Age": 0.05,
}


def _check_porcentaje_nulos(columna: str) -> Check:
    """Crea un Check que valida que el % de nulos no supere el umbral."""
    umbral = PORCENTAJE_MAXIMO_NULOS[columna]
    return Check(
        lambda s: s.isna().mean() <= umbral,
        ignore_na=False,
        error=f"Porcentaje de nulos en '{columna}' supera el umbral de {umbral:.0%}.",
    )


ESQUEMA_FEATURES = DataFrameSchema(
    {
        "Pregnancies": Column(
            "Int8",
            [Check.in_range(0, 20), _check_porcentaje_nulos("Pregnancies")],
            nullable=True,
        ),
        "Glucose": Column(
            float,
            [Check.in_range(40, 250), _check_porcentaje_nulos("Glucose")],
            nullable=True,
        ),
        "BloodPressure": Column(
            float,
            [Check.in_range(20, 140), _check_porcentaje_nulos("BloodPressure")],
            nullable=True,
        ),
        "SkinThickness": Column(
            float,
            [Check.in_range(5, 100), _check_porcentaje_nulos("SkinThickness")],
            nullable=True,
        ),
        "Insulin": Column(
            float,
            [Check.in_range(10, 900), _check_porcentaje_nulos("Insulin")],
            nullable=True,
        ),
        "BMI": Column(
            float,
            [Check.in_range(15, 70), _check_porcentaje_nulos("BMI")],
            nullable=True,
        ),
        "DiabetesPedigreeFunction": Column(
            float,
            [
                Check.in_range(0.05, 3.0),
                _check_porcentaje_nulos("DiabetesPedigreeFunction"),
            ],
            nullable=True,
        ),
        "Age": Column(
            "Int8",
            [Check.in_range(18, 100), _check_porcentaje_nulos("Age")],
            nullable=True,
        ),
        "Outcome": Column("boolean", nullable=False),
    },
    checks=[
        Check(
            lambda d: not d.duplicated().any(),
            error="El DataFrame contiene filas duplicadas.",
        ),
    ],
)


def validar_esquema(df: pd.DataFrame, lazy: bool = True) -> pd.DataFrame:
    """Valida el DataFrame de features contra ESQUEMA_FEATURES.

    Args:
        df: DataFrame a validar (salida del feature pipeline, antes de
            guardarlo).
        lazy: si True, acumula todos los errores de validación antes de
            lanzar la excepción (recomendado para tener un reporte
            completo en vez de detenerse en el primer error).

    Returns:
        El mismo DataFrame si la validación es exitosa.

    Raises:
        pandera.errors.SchemaErrors: si el DataFrame no cumple el
            esquema (con lazy=True, incluye todos los errores
            encontrados, no solo el primero).
    """
    logger.info("Validando esquema del DataFrame de features (%d filas)...", len(df))
    df_validado = ESQUEMA_FEATURES.validate(df, lazy=lazy)
    logger.info("Validación de esquema exitosa.")
    return df_validado


ESQUEMA_INFERENCIA = DataFrameSchema(
    {
        "Pregnancies": Column("Int8", Check.in_range(0, 20), nullable=True),
        "Glucose": Column(float, Check.in_range(40, 250), nullable=True),
        "BloodPressure": Column(float, Check.in_range(20, 140), nullable=True),
        "SkinThickness": Column(float, Check.in_range(5, 100), nullable=True),
        "Insulin": Column(float, Check.in_range(10, 900), nullable=True),
        "BMI": Column(float, Check.in_range(15, 70), nullable=True),
        "DiabetesPedigreeFunction": Column(float, Check.in_range(0.05, 3.0), nullable=True),
        "Age": Column("Int8", Check.in_range(18, 100), nullable=True),
    },
    coerce=True,
)


def validar_esquema_inferencia(df: pd.DataFrame, lazy: bool = True) -> pd.DataFrame:
    """Valida los datos de entrada para inferencia (sin la columna Outcome).

    A diferencia de `validar_esquema` (Tarea 2):
    - No exige la columna Outcome (no existe en datos nuevos a predecir).
    - No aplica umbral de % de nulos (el lote puede ser de una sola fila,
      y el modelo entrenado ya sabe imputar nulos internamente).
    - No exige ausencia de filas duplicadas.

    Raises:
        pandera.errors.SchemaErrors: si los tipos o rangos son invalidos.
    """
    logger.info("Validando esquema de inferencia (%d filas)...", len(df))
    df_validado = ESQUEMA_INFERENCIA.validate(df, lazy=lazy)
    logger.info("Validacion de esquema de inferencia exitosa.")
    return df_validado
