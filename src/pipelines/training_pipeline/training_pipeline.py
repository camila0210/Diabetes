"""Training pipeline: divide train/test, entrena y evalua el modelo final
(Random Forest afinado) sobre las features validadas por la Tarea 2, y
guarda el modelo entrenado junto con sus metricas de evaluacion en test
(Tarea 3 - Arquitectura FTI).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.split_validation import validar_split_train_test
from src.model.train import COLUMNAS_FEATURES, TARGET, construir_pipeline_modelo
from src.model.validate import (
    comparar_train_cv_test,
    diagnosticar_ajuste,
    validar_modelo_cv,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_PATH = REPO_ROOT / "data" / "03_primary" / "diabetes_clean.parquet"
MODEL_PATH = REPO_ROOT / "data" / "06_models" / "pipeline_produccion_random_forest.joblib"
METRICS_PATH = REPO_ROOT / "data" / "07_model_output" / "metricas_test.csv"
CV_REPORT_PATH = REPO_ROOT / "data" / "07_model_output" / "comparacion_train_cv_test.csv"

PROPORCION_TEST = 0.2
SEMILLA = 42


def cargar_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Carga las features validadas (salida del feature pipeline, Tareas 1-2)."""
    logger.info("Cargando features desde %s", path)
    return pd.read_parquet(path)


def dividir_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide en train/test de forma estratificada (80/20, semilla fija)."""
    X = df[COLUMNAS_FEATURES]
    y = df[TARGET].astype(bool)
    resultado: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(
        X, y, test_size=PROPORCION_TEST, stratify=y, random_state=SEMILLA
    )
    return resultado


def evaluar_modelo(modelo: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evalua el modelo entrenado sobre test con las metricas estandar."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def ejecutar_training_pipeline(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    cv_report_path: Path = CV_REPORT_PATH,
) -> tuple[Pipeline, dict[str, float]]:
    """Ejecuta el training pipeline completo: carga, split, entrena, evalua y guarda."""
    df = cargar_features(features_path)
    X_train, X_test, y_train, y_test = dividir_train_test(df)
    validar_split_train_test(X_train, X_test)

    modelo = construir_pipeline_modelo()
    logger.info(
        "Entrenando modelo (Random Forest afinado) con %d filas de train...",
        len(X_train),
    )
    modelo.fit(X_train, y_train)

    metricas_train = evaluar_modelo(modelo, X_train, y_train)
    metricas_cv = validar_modelo_cv(modelo, X_train, y_train)
    metricas = evaluar_modelo(modelo, X_test, y_test)
    logger.info("Metricas en test: %s", metricas)

    tabla_comparacion = comparar_train_cv_test(metricas_train, metricas_cv, metricas)
    diagnostico = diagnosticar_ajuste(tabla_comparacion)
    logger.info("Diagnostico de ajuste (train vs CV vs test): %s", diagnostico)
    logger.info("Tabla comparativa:\n%s", tabla_comparacion)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metricas]).to_csv(metrics_path, index=False)

    cv_report_path.parent.mkdir(parents=True, exist_ok=True)
    tabla_comparacion.to_csv(cv_report_path)

    logger.info("Modelo guardado en %s", model_path)
    return modelo, metricas


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ejecutar_training_pipeline()
