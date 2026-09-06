"""Tests de la validacion de leakage entre train/test (Tarea 4 - Arquitectura FTI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split_validation import LeakageError, validar_split_train_test


@pytest.fixture
def X_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Glucose": [100.0, 110.0, 120.0, 130.0],
            "Age": [25, 30, 35, 40],
        }
    )


def test_valida_split_sin_leakage(X_train: pd.DataFrame) -> None:
    """Un split sin filas compartidas no debe lanzar ninguna excepcion."""
    X_test = pd.DataFrame({"Glucose": [140.0, 150.0], "Age": [45, 50]})

    validar_split_train_test(X_train, X_test)


def test_detecta_leakage_fila_duplicada(X_train: pd.DataFrame) -> None:
    """Una fila de test identica a una de train debe lanzar LeakageError."""
    X_test = pd.DataFrame({"Glucose": [100.0, 150.0], "Age": [25, 50]})

    with pytest.raises(LeakageError):
        validar_split_train_test(X_train, X_test)


def test_detecta_leakage_con_nulos() -> None:
    """La deteccion debe funcionar tambien cuando la fila duplicada tiene nulos."""
    X_train = pd.DataFrame({"Glucose": [100.0, np.nan], "Age": [25, 30]})
    X_test = pd.DataFrame({"Glucose": [np.nan, 150.0], "Age": [30, 50]})

    with pytest.raises(LeakageError):
        validar_split_train_test(X_train, X_test)


def test_duplicados_solo_dentro_de_train_no_son_leakage() -> None:
    """Duplicados internos de train (sin correspondencia en test) no cuentan como leakage."""
    X_train = pd.DataFrame({"Glucose": [100.0, 100.0], "Age": [25, 25]})
    X_test = pd.DataFrame({"Glucose": [140.0], "Age": [45]})

    validar_split_train_test(X_train, X_test)
