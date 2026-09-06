"""Prediccion sobre datos nuevos usando el modelo entrenado.

El pipeline entrenado (Tarea 3) ya incluye imputacion, transformacion
log1p y escalado internamente, asi que basta con pasarle las features
crudas (ya validadas) directamente a `.predict()`/`.predict_proba()`.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

COLUMNAS_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def predecir(modelo: Pipeline, X_nuevas: pd.DataFrame) -> pd.DataFrame:
    """Genera predicciones (clase y probabilidad) sobre datos nuevos.

    Returns:
        DataFrame con las mismas filas de `X_nuevas` mas dos columnas:
        `prediccion` (bool) y `probabilidad` (float, probabilidad de
        Outcome=True).
    """
    X = X_nuevas[COLUMNAS_FEATURES]
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)[:, 1]

    logger.info("Predicciones generadas para %d filas.", len(X_nuevas))

    return X_nuevas.assign(prediccion=predicciones, probabilidad=probabilidades)
