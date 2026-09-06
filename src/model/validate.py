"""Validacion del modelo via cross-validation (Tarea 5 - Arquitectura FTI).

Compara el desempeño del modelo en train completo, en cross-validation
(StratifiedKFold) y en test, para detectar overfitting (train mucho mejor
que CV) o underfitting (CV y test ambos con desempeño bajo).
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

N_FOLDS = 5
SEMILLA = 42
METRICAS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
UMBRAL_DIFERENCIA_OVERFITTING = 0.15
UMBRAL_F1_BAJO = 0.5


def validar_modelo_cv(
    modelo: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = N_FOLDS,
) -> dict[str, float]:
    """Corre StratifiedKFold CV sobre una copia del modelo (sin entrenar) y
    retorna las metricas promedio entre folds.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)
    resultados = cross_validate(clone(modelo), X_train, y_train, cv=cv, scoring=METRICAS)
    metricas_cv = {metrica: resultados[f"test_{metrica}"].mean() for metrica in METRICAS}
    logger.info("Metricas promedio en CV (%d folds): %s", n_folds, metricas_cv)
    return metricas_cv


def comparar_train_cv_test(
    metricas_train: dict[str, float],
    metricas_cv: dict[str, float],
    metricas_test: dict[str, float],
) -> pd.DataFrame:
    """Arma una tabla comparativa train/CV/test para diagnosticar over/underfitting."""
    return pd.DataFrame({"train": metricas_train, "cv": metricas_cv, "test": metricas_test})


def diagnosticar_ajuste(tabla_comparacion: pd.DataFrame) -> str:
    """Diagnostica overfitting/underfitting comparando F1 en train, CV y test.

    Heuristica simple:
    - Si F1(train) - F1(CV) supera el umbral, se sospecha overfitting.
    - Si F1(CV) y F1(test) son ambos bajos, se sospecha underfitting.
    - En otro caso, el ajuste se considera razonable.
    """
    f1_train = tabla_comparacion.loc["f1", "train"]
    f1_cv = tabla_comparacion.loc["f1", "cv"]
    f1_test = tabla_comparacion.loc["f1", "test"]

    if (f1_train - f1_cv) > UMBRAL_DIFERENCIA_OVERFITTING:
        return "posible overfitting"
    if f1_cv < UMBRAL_F1_BAJO and f1_test < UMBRAL_F1_BAJO:
        return "posible underfitting"
    return "ajuste razonable"
