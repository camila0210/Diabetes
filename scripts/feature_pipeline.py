"""Feature pipeline: lee el dataset crudo, lo limpia y guarda un dataset de
features listo para el split train/test.

Traslada a un script reproducible la logica ya validada en los notebooks
2 (Exploracion inicial) y 4 (Feature Engineering) del POC:
- Unifica los ceros clinicamente invalidos (Glucose, BloodPressure,
  SkinThickness, Insulin, BMI) a NaN.
- Corrige el bug de formato de DiabetesPedigreeFunction (valores sin punto
  decimal, ej. 627 en vez de 0.627): se dividen entre 1000 los valores > 3.
- Convierte los tipos de datos usando tipos nullable de pandas para
  conservar los NaN existentes.
- Elimina filas sin Outcome (no sirven para entrenar ni evaluar) y
  duplicados exactos.

No incluye imputacion de nulos ni el split train/test: la imputacion debe
ajustarse solo con datos de entrenamiento para evitar data leakage, y el
split ocurre en train_pipeline.py. Este pipeline solo aplica
transformaciones deterministas que no requieren "fit".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "01_raw" / "diabetes.csv"
OUTPUT_PATH = DATA_DIR / "03_primary" / "diabetes_clean.parquet"

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


def cargar_datos_crudos(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lee el dataset crudo desde CSV."""
    logger.info("Cargando datos crudos desde %s", path)
    return pd.read_csv(path)


def unificar_valores_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica ceros clinicamente invalidos y celdas vacias a NaN.

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


def ejecutar_feature_pipeline(
    input_path: Path = RAW_DATA_PATH, output_path: Path = OUTPUT_PATH
) -> pd.DataFrame:
    """Ejecuta el feature pipeline completo: carga, limpia, transforma y guarda."""
    df = cargar_datos_crudos(input_path)
    df = unificar_valores_faltantes(df)
    df = corregir_escala_dpf(df)
    df = convertir_tipos(df)
    df = eliminar_filas_invalidas(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.Table.from_pandas(df).schema
    df.to_parquet(output_path, index=False, schema=schema)
    logger.info("Features guardadas en %s", output_path)
    return df


if __name__ == "__main__":
    ejecutar_feature_pipeline()
