"""Inference pipeline: carga el modelo entrenado, valida datos nuevos y
genera predicciones (Tarea 6 - Arquitectura FTI).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.validation import validar_esquema_inferencia
from src.inference.predict import predecir

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "data" / "06_models" / "pipeline_produccion_random_forest.joblib"
INPUT_PATH = REPO_ROOT / "data" / "01_raw" / "diabetes_nuevos.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "07_model_output" / "predicciones.csv"


def cargar_modelo(model_path: Path = MODEL_PATH) -> Pipeline:
    """Carga el pipeline entrenado desde disco."""
    logger.info("Cargando modelo desde %s", model_path)
    return joblib.load(model_path)


def cargar_datos_nuevos(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Carga los datos nuevos a predecir."""
    logger.info("Cargando datos nuevos desde %s", input_path)
    return pd.read_csv(input_path)


def ejecutar_inference_pipeline(
    model_path: Path = MODEL_PATH,
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Ejecuta el inference pipeline completo: carga, valida, predice y guarda."""
    modelo = cargar_modelo(model_path)
    df_nuevos = cargar_datos_nuevos(input_path)

    df_validado = validar_esquema_inferencia(df_nuevos)
    resultado = predecir(modelo, df_validado)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(output_path, index=False)
    logger.info("Predicciones guardadas en %s", output_path)

    return resultado


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ejecutar_inference_pipeline()
