"""Demo funcional: prediccion de diabetes con el modelo Random Forest (Tarea 6/7)."""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

MODEL_PATH = "data/06_models/modelo_final_random_forest.joblib"
UMBRAL_OPTIMO = 0.436
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
COLUMNAS_LOG = ["Insulin", "DiabetesPedigreeFunction"]


@st.cache_resource
def cargar_modelo() -> Pipeline:
    return joblib.load(MODEL_PATH)


def predecir(datos_usuario: dict) -> tuple[bool, float]:
    fila = pd.DataFrame([datos_usuario])[COLUMNAS_FEATURES]
    for columna in COLUMNAS_LOG:
        fila[columna] = np.log1p(fila[columna])
    modelo = cargar_modelo()
    probabilidad = modelo.predict_proba(fila)[0, 1]
    return probabilidad >= UMBRAL_OPTIMO, probabilidad


st.title("Prediccion de Diabetes")
st.write(
    "Ingresa los datos clinicos del paciente para estimar el riesgo de "
    "diabetes con el modelo Random Forest entrenado en la Tarea 6."
)

with st.form("formulario_prediccion"):
    columna1, columna2 = st.columns(2)
    with columna1:
        pregnancies = st.number_input(
            "Embarazos (Pregnancies)", min_value=0, max_value=20, value=1, step=1
        )
        glucose = st.number_input("Glucosa (mg/dL)", min_value=40, max_value=250, value=110, step=1)
        blood_pressure = st.number_input(
            "Presion arterial diastolica (mm Hg)",
            min_value=30,
            max_value=140,
            value=70,
            step=1,
        )
        skin_thickness = st.number_input(
            "Grosor del pliegue cutaneo (mm)", min_value=1, max_value=100, value=25, step=1
        )
    with columna2:
        insulin = st.number_input(
            "Insulina serica (mu U/ml)", min_value=1, max_value=900, value=100, step=1
        )
        bmi = st.number_input(
            "Indice de masa corporal (BMI)",
            min_value=10.0,
            max_value=70.0,
            value=28.0,
            step=0.1,
        )
        dpf = st.number_input(
            "Funcion de pedigri de diabetes (DPF)",
            min_value=0.01,
            max_value=3.0,
            value=0.4,
            step=0.01,
        )
        age = st.number_input("Edad", min_value=1, max_value=120, value=33, step=1)

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
    prediccion, probabilidad = predecir(datos_usuario)

    st.subheader("Resultado")
    if prediccion:
        st.error(f"Riesgo de diabetes: **Alto** (probabilidad = {probabilidad:.1%})")
    else:
        st.success(f"Riesgo de diabetes: **Bajo** (probabilidad = {probabilidad:.1%})")

    st.caption(
        f"Umbral de decision usado: {UMBRAL_OPTIMO:.3f} (optimizado para F1 en la "
        "Tarea 6, prioriza detectar casos reales de diabetes sobre minimizar "
        "falsas alarmas)."
    )
    st.progress(min(probabilidad, 1.0))
