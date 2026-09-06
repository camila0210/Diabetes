"""Validacion de integridad del split train/test contra leakage.

Detecta si hay filas identicas entre train y test (comparando por los
valores de las features), una forma sutil de data leakage que no captura
la validacion de esquema de la Tarea 2 (esa valida cada conjunto por
separado, no la relacion entre ambos).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class LeakageError(RuntimeError):
    """Se lanza cuando se detecta contaminacion entre train y test."""


def validar_split_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Valida que no haya filas identicas entre train y test.

    Compara las filas de ambos conjuntos por sus valores de features,
    tratando los nulos como iguales entre si (igual que
    `pandas.DataFrame.duplicated`). Si una fila de test es identica a una
    de train, el modelo pudo haber "visto" esa fila durante el
    entrenamiento, invalidando la evaluacion en test.

    Raises:
        LeakageError: si se encuentra al menos una fila de test duplicada
            en train.
    """
    combinado = pd.concat(
        [X_train.assign(_origen="train"), X_test.assign(_origen="test")],
        ignore_index=True,
    )
    columnas_features = [c for c in combinado.columns if c != "_origen"]
    duplicados = combinado[combinado.duplicated(subset=columnas_features, keep=False)]
    filas_test_duplicadas = duplicados[duplicados["_origen"] == "test"]

    n_duplicados = len(filas_test_duplicadas)
    logger.info(
        "Filas de test identicas a filas de train: %d de %d",
        n_duplicados,
        len(X_test),
    )

    if n_duplicados > 0:
        porcentaje = n_duplicados / len(X_test)
        raise LeakageError(
            f"Se encontraron {n_duplicados} filas de test ({porcentaje:.2%}) "
            "identicas a filas de train. Posible data leakage en el split."
        )
