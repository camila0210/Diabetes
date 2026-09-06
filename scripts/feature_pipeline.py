"""CLI wrapper para ejecutar el feature pipeline (Tarea 1).

La logica principal vive en src/pipelines/feature_pipeline/feature_pipeline.py
(medida por coverage); este script solo la invoca desde la linea de comandos.
"""

from __future__ import annotations

import logging

from src.pipelines.feature_pipeline.feature_pipeline import ejecutar_feature_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ejecutar_feature_pipeline()
