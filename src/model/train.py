"""Construccion del pipeline de modelado (preprocesamiento + Random Forest).

Replica el preprocesamiento validado en el notebook de Feature Engineering
(Tarea 4 del POC) y el modelo seleccionado en el notebook de Seleccion del
Mejor Modelo (Tarea 6 del POC), unificados en un solo `sklearn.Pipeline`
para evitar training/serving skew: el mismo objeto entrenado sabe imputar,
transformar y predecir sobre datos crudos (ya validados por el esquema de
la Tarea 2), sin pasos manuales adicionales en inferencia.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

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
COLUMNAS_SESGADAS = ["Insulin", "DiabetesPedigreeFunction"]
COLUMNAS_NO_SESGADAS = [c for c in COLUMNAS_FEATURES if c not in COLUMNAS_SESGADAS]
TARGET = "Outcome"

SEMILLA = 42

# Hiperparametros afinados via GridSearchCV(cv=5, scoring="f1") en el
# notebook 06.Seleccion-mejor-modelo del POC.
MEJORES_HIPERPARAMETROS_RF = {
    "n_estimators": 200,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": SEMILLA,
}


def construir_pipeline_modelo() -> Pipeline:
    """Construye el pipeline completo: imputacion + log1p + escalado + modelo.

    La imputacion (mediana) se ajusta unicamente con los datos que se pasen
    a `.fit()` (se espera que sea solo train); dividir train/test antes de
    llamar a `.fit()` es responsabilidad de quien use este pipeline (ver
    `training_pipeline.py`). La transformacion log1p es deterministica y no
    requiere ajuste.
    """
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "imputer_log_sesgadas",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "log1p",
                            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                        ),
                    ]
                ),
                COLUMNAS_SESGADAS,
            ),
            (
                "imputer_resto",
                SimpleImputer(strategy="median"),
                COLUMNAS_NO_SESGADAS,
            ),
        ],
        verbose_feature_names_out=False,
    )
    preprocesador.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(**MEJORES_HIPERPARAMETROS_RF)),
        ]
    )
