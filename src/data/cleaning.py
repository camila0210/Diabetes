"""Funciones de limpieza y transformacion de datos para el dataset de diabetes.

Logica trasladada desde los notebooks 2 (Exploracion inicial) y 4 (Feature
Engineering) del POC. Son funciones puras (no leen ni escriben archivos) para
que sean facilmente testeables y reutilizables desde cualquier pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMNAS_SIN_CERO_VALIDO = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]
COLUMNAS_ENTERAS = ["Pregnancies", "Age"]
COLUMNAS_CONTINUAS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
]
TARGET = "Outcome"
UMBRAL_PUNTO_DECIMAL_PERDIDO = 3


def unificar_valores_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica ceros clinicamente invalidos a NaN.

    En Glucose, BloodPressure, SkinThickness, Insulin y BMI, un valor de 0
    es fisiologicamente imposible y en realidad representa un dato faltante.
    """
    df = df.copy()
    df[COLUMNAS_SIN_CERO_VALIDO] = df[COLUMNAS_SIN_CERO_VALIDO].replace(0, np.nan)
    return df


def corregir_escala_dpf(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige el bug de formato en DiabetesPedigreeFunction.

    La mayoria de los valores perdieron el punto decimal en el archivo
    original (ej. 627 en vez de 0.627). Se identifican los valores por
    encima del umbral clinico esperado (> 3) y se dividen entre 1000.
    """
    df = df.copy()
    mascara_dpf_mal_formateado = df["DiabetesPedigreeFunction"] > UMBRAL_PUNTO_DECIMAL_PERDIDO
    df.loc[mascara_dpf_mal_formateado, "DiabetesPedigreeFunction"] /= 1000
    return df


def convertir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los tipos de datos usando tipos nullable de pandas.

    Se usan Int8/float64/boolean (nullable) para conservar los NaN
    existentes sin perder la semantica de entero/booleano.
    """
    df = df.copy()
    df[COLUMNAS_ENTERAS] = df[COLUMNAS_ENTERAS].astype("Int8")
    df[COLUMNAS_CONTINUAS] = df[COLUMNAS_CONTINUAS].astype("float64")
    df[TARGET] = df[TARGET].astype("boolean")
    return df


def eliminar_filas_invalidas(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas sin Outcome (no sirven para entrenar/evaluar) y duplicados exactos."""
    n_inicial = len(df)
    df = df[df[TARGET].notna()].copy()
    n_sin_target = n_inicial - len(df)

    n_antes_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_duplicados = n_antes_dedup - len(df)

    logger.info("Filas eliminadas por Outcome faltante: %d", n_sin_target)
    logger.info("Filas eliminadas por duplicados: %d", n_duplicados)
    logger.info("Filas finales: %d", len(df))
    return df
