"""Feature pipeline: lee el dataset crudo, lo limpia y guarda un dataset de
features listo para el split train/test (Tarea 1 - Arquitectura FTI).

No incluye imputacion de nulos ni el split train/test: la imputacion debe
ajustarse solo con datos de entrenamiento para evitar data leakage, y el
split ocurre en el training pipeline. Este pipeline solo aplica
transformaciones deterministas que no requieren "fit".
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.cleaning import (
    convertir_tipos,
    corregir_escala_dpf,
    eliminar_filas_invalidas,
    unificar_valores_faltantes,
)
from src.data.validation import validar_esquema

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = REPO_ROOT / "data" / "01_raw" / "diabetes.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "03_primary" / "diabetes_clean.parquet"


def cargar_datos_crudos(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lee el dataset crudo desde CSV."""
    logger.info("Cargando datos crudos desde %s", path)
    return pd.read_csv(path)


def ejecutar_feature_pipeline(
    input_path: Path = RAW_DATA_PATH, output_path: Path = OUTPUT_PATH
) -> pd.DataFrame:
    """Ejecuta el feature pipeline completo: carga, limpia, transforma y guarda."""
    df = cargar_datos_crudos(input_path)
    df = unificar_valores_faltantes(df)
    df = corregir_escala_dpf(df)
    df = convertir_tipos(df)
    df = eliminar_filas_invalidas(df)
    df = validar_esquema(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Features guardadas en %s", output_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ejecutar_feature_pipeline()
