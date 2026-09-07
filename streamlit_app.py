"""App de Streamlit: demo funcional del modelo de prediccion de diabetes.

Categoria 3: despliegue funcional con Streamlit.
- Tarea 1: demo online (prediccion individual).
- Tarea 2: procesamiento batch (prediccion por lote via CSV).

Ambas pestanas usan el pipeline de produccion entrenado en la Categoria 2
(Arquitectura FTI): valida los datos con Pandera (src/data/validation.py) y
genera la prediccion con la logica de inferencia ya testeada
(src/inference/predict.py).

Es un wrapper delgado sobre codigo ya testeado en la Categoria 2 (mismo
criterio que los wrappers de scripts/), por eso no tiene tests propios.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.data.validation import validar_esquema_inferencia
from src.inference.predict import predecir

REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "data" / "06_models" / "pipeline_produccion_random_forest.joblib"
EJEMPLO_ENTRADA_PATH = REPO_ROOT / "data" / "01_raw" / "ejemplo_lote_entrada.csv"

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


@st.cache_resource
def cargar_modelo() -> Pipeline:
    """Carga el pipeline de produccion entrenado (cacheado entre sesiones)."""
    return joblib.load(MODEL_PATH)


def predecir_una_fila(datos_usuario: dict[str, float]) -> tuple[bool, float]:
    """Valida (Pandera) y predice sobre un unico registro del formulario."""
    fila = pd.DataFrame([datos_usuario])[COLUMNAS_FEATURES]
    fila_validada = validar_esquema_inferencia(fila)
    modelo = cargar_modelo()
    resultado = predecir(modelo, fila_validada)
    return bool(resultado.loc[0, "prediccion"]), float(resultado.loc[0, "probabilidad"])


def predecir_lote(df_nuevo: pd.DataFrame) -> pd.DataFrame:
    """Valida (Pandera) y predice sobre un lote de registros nuevos."""
    df_validado = validar_esquema_inferencia(df_nuevo)
    modelo = cargar_modelo()
    return predecir(modelo, df_validado)


st.set_page_config(page_title="Prediccion de Diabetes", page_icon="🩺")
st.title("Prediccion de Diabetes")
st.write("Demo funcional del pipeline de produccion (Categoria 2 - Arquitectura FTI).")

tab_individual, tab_lote = st.tabs(["Prediccion individual", "Prediccion por lote"])

with tab_individual:
    st.write("Ingresa los datos clinicos de una persona para estimar su riesgo de diabetes.")

    with st.form("formulario_prediccion"):
        columna1, columna2 = st.columns(2)
        with columna1:
            pregnancies = st.number_input(
                "Embarazos (Pregnancies)", min_value=0, max_value=20, value=1, step=1
            )
            glucose = st.number_input(
                "Glucosa (mg/dL)", min_value=40, max_value=250, value=110, step=1
            )
            blood_pressure = st.number_input(
                "Presion arterial diastolica (mm Hg)",
                min_value=20,
                max_value=140,
                value=70,
                step=1,
            )
            skin_thickness = st.number_input(
                "Grosor del pliegue cutaneo (mm)",
                min_value=5,
                max_value=100,
                value=25,
                step=1,
            )
        with columna2:
            insulin = st.number_input(
                "Insulina serica (mu U/ml)",
                min_value=10,
                max_value=900,
                value=100,
                step=1,
            )
            bmi = st.number_input(
                "Indice de masa corporal (BMI)",
                min_value=15.0,
                max_value=70.0,
                value=28.0,
                step=0.1,
            )
            dpf = st.number_input(
                "Funcion de pedigri de diabetes (DPF)",
                min_value=0.05,
                max_value=3.0,
                value=0.4,
                step=0.01,
            )
            age = st.number_input("Edad", min_value=18, max_value=100, value=33, step=1)

        enviado = st.form_submit_button("Predecir")

    if enviado:
        datos_usuario = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }
        try:
            prediccion, probabilidad = predecir_una_fila(datos_usuario)
        except Exception as error:  # esquema invalido u otro error de validacion
            st.error(f"No se pudo generar la prediccion: {error}")
        else:
            st.subheader("Resultado")
            if prediccion:
                st.error(f"Riesgo de diabetes: **Alto** (probabilidad = {probabilidad:.1%})")
            else:
                st.success(f"Riesgo de diabetes: **Bajo** (probabilidad = {probabilidad:.1%})")
            st.progress(min(probabilidad, 1.0))

with tab_lote:
    st.write(
        "Sube un archivo CSV con una o mas filas de datos clinicos "
        f"(columnas: {', '.join(COLUMNAS_FEATURES)}) para obtener "
        "una prediccion por cada fila."
    )

    if EJEMPLO_ENTRADA_PATH.exists():
        st.download_button(
            "Descargar CSV de ejemplo",
            data=EJEMPLO_ENTRADA_PATH.read_bytes(),
            file_name="ejemplo_lote_entrada.csv",
            mime="text/csv",
        )

    archivo = st.file_uploader("Archivo CSV", type="csv")

    if archivo is not None:
        try:
            df_nuevo = pd.read_csv(archivo)
            resultado_lote = predecir_lote(df_nuevo)
        except Exception as error:  # esquema invalido u otro error de validacion
            st.error(f"No se pudo procesar el archivo: {error}")
        else:
            st.subheader("Resultados")
            st.dataframe(resultado_lote)
            csv_resultado = resultado_lote.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar predicciones (CSV)",
                data=csv_resultado,
                file_name="predicciones.csv",
                mime="text/csv",
            )
