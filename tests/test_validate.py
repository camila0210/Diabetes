"""Tests de la validacion del modelo via cross-validation (Tarea 5 - Arquitectura FTI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.validate import (
    comparar_train_cv_test,
    diagnosticar_ajuste,
    validar_modelo_cv,
)

N_FILAS = 100


@pytest.fixture
def datos_clasificacion() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=N_FILAS),
            "b": rng.normal(size=N_FILAS),
        }
    )
    y = pd.Series(rng.choice([True, False], N_FILAS, p=[0.4, 0.6]))
    return X, y


@pytest.fixture
def modelo_simple() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42)),
        ]
    )


def test_validar_modelo_cv_retorna_metricas_esperadas(
    datos_clasificacion: tuple[pd.DataFrame, pd.Series], modelo_simple: Pipeline
) -> None:
    X, y = datos_clasificacion

    metricas_cv = validar_modelo_cv(modelo_simple, X, y, n_folds=5)

    assert set(metricas_cv.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }


def test_comparar_train_cv_test_estructura() -> None:
    metricas_train = {"f1": 0.9, "accuracy": 0.9}
    metricas_cv = {"f1": 0.6, "accuracy": 0.7}
    metricas_test = {"f1": 0.58, "accuracy": 0.68}

    tabla = comparar_train_cv_test(metricas_train, metricas_cv, metricas_test)

    assert list(tabla.columns) == ["train", "cv", "test"]
    assert tabla.loc["f1", "train"] == pytest.approx(0.9)


def test_diagnostico_detecta_overfitting() -> None:
    tabla = pd.DataFrame({"train": {"f1": 0.95}, "cv": {"f1": 0.60}, "test": {"f1": 0.58}})

    assert diagnosticar_ajuste(tabla) == "posible overfitting"


def test_diagnostico_detecta_underfitting() -> None:
    tabla = pd.DataFrame({"train": {"f1": 0.45}, "cv": {"f1": 0.40}, "test": {"f1": 0.38}})

    assert diagnosticar_ajuste(tabla) == "posible underfitting"


def test_diagnostico_ajuste_razonable() -> None:
    tabla = pd.DataFrame({"train": {"f1": 0.70}, "cv": {"f1": 0.66}, "test": {"f1": 0.65}})

    assert diagnosticar_ajuste(tabla) == "ajuste razonable"
