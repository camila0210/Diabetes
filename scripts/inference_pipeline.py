"""CLI wrapper para ejecutar el inference pipeline (Tarea 6).

La logica principal vive en src/pipelines/inference_pipeline/inference_pipeline.py
(medida por coverage); este script solo lo invoca desde la linea de comandos.
"""

from __future__ import annotations

import logging

from src.pipelines.inference_pipeline.inference_pipeline import (
    ejecutar_inference_pipeline,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ejecutar_inference_pipeline()
