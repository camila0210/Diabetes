"""CLI wrapper para ejecutar el training pipeline (Tarea 3).

La logica principal vive en src/pipelines/training_pipeline/training_pipeline.py
(medida por coverage); este script solo lo invoca desde la linea de comandos.
"""

from __future__ import annotations

import logging

from src.pipelines.training_pipeline.training_pipeline import (
    ejecutar_training_pipeline,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ejecutar_training_pipeline()
