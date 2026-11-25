from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="PAVEMENT AI | SENATI", page_icon="🛣️", layout="wide")
st.title("🛣️ PAVEMENT AI")
st.caption("SENATI · Inteligencia Artificial · Proyecto educativo de pavimento rígido")

DATA_PATH = Path(__file__).parent / "datos" / "pavimento_simulado.csv"
PATHOLOGIES = [
    "Grietas longitudinales", "Grietas transversales", "Grietas de esquina",
    "Losa dividida", "Despostillamiento", "Escalonamiento", "Bombeo", "Otro",
]
SEVERITIES = ["Baja", "Media", "Alta"]
REQUIRED = ["tramo", "fecha", "antiguedad", "transito", "patologia", "severidad", "pci"]


def make_sample() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    ages = np.tile(np.arange(1, 11), 6)
    rng.shuffle(ages)
    pci = np.clip(107 - 8.0 * ages + rng.normal(0, 5, len(ages)), 10, 100).round(1)
    pathologies = rng.choice(PATHOLOGIES[:-1], size=len(ages))
    severity = rng.choice(SEVERITIES, size=len(ages), p=[0.38, 0.4, 0.22])
    start = pd.Timestamp("2024-01-01")
    return pd.DataFrame({
        "tramo": [f"TR-{i+1:03}" for i in range(len(ages))],
        "fecha": [(start + pd.Timedelta(days=int(d))).date().isoformat() for d in rng.integers(0, 900, len(ages))],
        "antiguedad": ages.astype(int),
        "transito": rng.integers(150, 4200, len(ages)),
        "patologia": pathologies,
        "severidad": severity,
        "pci": pci,
        "origen": "Simulado",
    })


def load_sample() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return make_sample()


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED if col not in data.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))
    result = data.copy()
    result["antiguedad"] = pd.to_numeric(result["antiguedad"], errors="coerce")
    result["pci"] = pd.to_numeric(result["pci"], errors="coerce")
    result["transito"] = pd.to_numeric(result["transito"], errors="coerce")
    result = result.dropna(subset=["antiguedad", "pci"])
    result = result[result["antiguedad"].between(0, 100) & result["pci"].between(0, 100)]
    if result.empty:
        raise ValueError("No hay filas válidas. Revisa antigüedad y PCI (rango 0–100).")
    result["origen"] = result.get("origen", "Cargado").fillna("Cargado") if isinstance(result.get("origen", "Cargado"), pd.Series) else "Cargado"
    return result


def train_model(data: pd.DataFrame):
    if len(data) < 4:
        raise ValueError("Se necesitan al menos 4 registros válidos para entrenar y evaluar.")
    x_train, x_test, y_train, y_test = train_test_split(
        data[["antiguedad"]], data["pci"], test_size=0.25, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "R²": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    return model, metrics


def pci_condition(value: float) -> str:
    if value >= 85: return "Bueno"
    if value >= 70: return "Satisfactorio"
    if value >= 55: return "Regular"
    if value >= 40: return "Pobre"
    if value >= 25: return "Muy pobre"
    if value >= 10: return "Grave"
    return "Fallado"


sample = load_sample()
if "records" not in st.session_state:
    st.session_state.records = sample.copy()
data = clean_data(st.session_state.records)
model, metrics = train_model(data)

tab_dashboard, tab_simulator, tab_register, tab_data = st.tabs(
    ["Dashboard", "Simulador PCI", "Registrar inspección", "Datos y ayuda"]
)

with tab_dashboard:
    st.subheader("Resumen de inspecciones")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros válidos", len(data))
    c2.metric("PCI promedio", f"{data.pci.mean():.1f}")
    c3.metric("Antigüedad promedio", f"{data.antiguedad.mean():.1f} años")
    c4.metric("Origen", "Simulado" if (data.origen == "Simulado").all() else "Mixto / cargado")
    left, right = st.columns(2)
    with left:
        st.markdown("**PCI observado por antigüedad**")
        fig = px.scatter(data, x="antiguedad", y="pci", color="severidad", hover_name="tramo", trendline=None,
                         labels={"antiguedad": "Antigüedad (años)", "pci": "PCI observado", "severidad": "Severidad"})
        ages_line = np.linspace(data.antiguedad.min(), data.antiguedad.max(), 100)
        fig.add_scatter(x=ages_line, y=model.predict(pd.DataFrame({"antiguedad": ages_line})), mode="lines", name="Regresión lineal")
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Métricas en conjunto de prueba**")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metrics['R²']:.3f}")
        m2.metric("MAE", f"{metrics['MAE']:.2f}")
        m3.metric("RMSE", f"{metrics['RMSE']:.2f}")
        st.caption("Evaluación con partición aleatoria 75/25 y semilla fija. El modelo usa solo antigüedad.")
        st.markdown("**Regresión ajustada**")
        st.code(f"PCI = {model.intercept_:.2f} + ({model.coef_[0]:.2f} × antigüedad)")
        st.markdown("**PCI promedio por patología**")
        by_path = data.groupby("patologia", as_index=False).pci.mean().sort_values("pci")
        st.plotly_chart(px.bar(by_path, x="pci", y="patologia", orientation="h", labels={"pci": "PCI promedio", "patologia": "Patología"}), use_container_width=True)
    st.dataframe(data, use_container_width=True, hide_index=True)

with tab_simulator:
    st.subheader("Simula el PCI futuro")
    st.write("El modelo estima PCI a partir de antigüedad. Los valores son orientativos y dependen de los datos disponibles.")
    max_age = max(1, int(np.ceil(data.antiguedad.max())))
    a, b = st.columns(2)
    with a:
        age = st.number_input("Antigüedad actual (años)", min_value=0.0, max_value=100.0, value=float(min(max_age, 5)), step=1.0)
    with b:
        horizon = st.number_input("Horizonte (años)", min_value=1, max_value=30, value=3, step=1)
    future_age = age + horizon
    current_pci = float(np.clip(model.predict(pd.DataFrame({"antiguedad": [age]}))[0], 0, 100))
    future_pci = float(np.clip(model.predict(pd.DataFrame({"antiguedad": [future_age]}))[0], 0, 100))
    s1, s2, s3 = st.columns(3)
    s1.metric("PCI actual estimado", f"{current_pci:.1f}")
    s2.metric("PCI futuro estimado", f"{future_pci:.1f}", delta=f"{future_pci-current_pci:.1f}")
    s3.metric("Condición futura", pci_condition(future_pci))
    if future_age > data.antiguedad.max():
        st.warning("La predicción futura está fuera del rango de antigüedad observado; es una extrapolación y tiene mayor incertidumbre.")
    st.caption("El PCI se limita a 0–100 para su presentación. Esta limitación no mejora la precisión del modelo.")
    st.markdown("**Tres escenarios**")
    scenarios = pd.DataFrame({"Escenario": ["1 año", "3 años", "5 años"], "Antigüedad": [age+1, age+3, age+5]})
    scenarios["PCI estimado"] = np.clip(model.predict(scenarios[["Antigüedad"]].rename(columns={"Antigüedad": "antiguedad"})), 0, 100).round(1)
    scenarios["Condición"] = scenarios["PCI estimado"].map(pci_condition)
    st.dataframe(scenarios, hide_index=True, use_container_width=True)
    st.download_button("Descargar escenarios CSV", scenarios.to_csv(index=False).encode("utf-8-sig"), "escenarios_pci.csv", "text/csv")
    chart_ages = np.arange(0, max(float(data.antiguedad.max()), future_age, age) + 2)
    chart = px.scatter(data, x="antiguedad", y="pci", color="severidad", labels={"antiguedad": "Antigüedad (años)", "pci": "PCI"})
    chart.add_scatter(x=chart_ages, y=np.clip(model.predict(pd.DataFrame({"antiguedad": chart_ages})), 0, 100), mode="lines", name="Modelo")
    chart.add_scatter(x=[age, future_age], y=[current_pci, future_pci], mode="markers+lines", name="Tu escenario", marker={"size": 12})
    chart.update_yaxes(range=[0, 100])
    st.plotly_chart(chart, use_container_width=True)

with tab_register:
    st.subheader("Registrar inspección de un tramo")
    with st.form("inspection_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        tramo = c1.text_input("Identificación del tramo", placeholder="Ej. HU-001")
        inspection_date = c2.date_input("Fecha de inspección", value=date.today())
        c3, c4 = st.columns(2)
        new_age = c3.number_input("Antigüedad (años)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        traffic = c4.number_input("Tránsito diario estimado", min_value=0, value=1000, step=100)
        c5, c6, c7 = st.columns(3)
        pathology = c5.selectbox("Patología", PATHOLOGIES)
        severity = c6.selectbox("Severidad", SEVERITIES)
        observed_pci = c7.number_input("PCI observado (evaluación técnica)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        notes = st.text_area("Observaciones", placeholder="Ubicación, extensión u otras notas")
        photos = st.file_uploader("Fotografías opcionales", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        submitted = st.form_submit_button("Guardar inspección")
    if submitted:
        if not tramo.strip():
            st.error("Ingresa la identificación del tramo.")
        else:
            row = {"tramo": tramo.strip(), "fecha": inspection_date.isoformat(), "antiguedad": new_age,
                   "transito": traffic, "patologia": pathology, "severidad": severity,
                   "pci": observed_pci, "origen": "Registrado", "observaciones": notes,
                   "fotografias": ", ".join(photo.name for photo in photos) if photos else ""}
            st.session_state.records = pd.concat([st.session_state.records, pd.DataFrame([row])], ignore_index=True)
            st.session_state.inspection_saved = True
            if photos:
                st.session_state.inspection_photo_notice = True
            st.rerun()
    if st.session_state.pop("inspection_saved", False):
        st.success("Inspección guardada en esta sesión.")
    if st.session_state.pop("inspection_photo_notice", False):
        st.info("Los nombres de las fotografías se incluyen en la exportación; los archivos no se almacenan al cerrar la sesión.")
    st.caption("El PCI ingresado debe proceder de una inspección técnica; no se calcula a partir de la severidad.")
    st.download_button("Exportar registros CSV", data.to_csv(index=False).encode("utf-8-sig"), "inspecciones_pavimento.csv", "text/csv")

with tab_data:
    st.subheader("Datos de entrenamiento")
    st.info("Los datos que vienen con el proyecto son simulados. No representan mediciones reales ni deben presentarse como inspecciones de Huánuco.")
    st.write("Para cargar datos propios, usa un CSV con estas columnas: `tramo`, `fecha`, `antiguedad`, `transito`, `patologia`, `severidad`, `pci`.")
    uploaded = st.file_uploader("Cargar CSV de inspecciones", type=["csv"], key="csv_upload")
    if uploaded is not None:
        try:
            incoming = clean_data(pd.read_csv(uploaded))
            incoming["origen"] = "Cargado"
            if st.button("Usar este CSV en el dashboard", key="use_csv"):
                st.session_state.records = incoming
                st.rerun()
            st.success(f"CSV válido: {len(incoming)} registros utilizables.")
        except Exception as exc:
            st.error(str(exc))
    st.download_button("Descargar CSV simulado (60 registros)", sample.to_csv(index=False).encode("utf-8-sig"), "pavimento_simulado.csv", "text/csv")
    st.markdown("### Guía rápida de interpretación")
    st.markdown("- **R²:** proporción de variación explicada por el modelo en la prueba; puede ser negativa si predice peor que usar el promedio.\n- **MAE:** error absoluto promedio, en puntos PCI.\n- **RMSE:** error que penaliza más los errores grandes, en puntos PCI.\n- Una pendiente negativa indica que, en estos datos, el PCI tiende a bajar al aumentar la antigüedad; no demuestra causalidad.\n- Antigüedad por sí sola no representa tránsito, drenaje, construcción, mantenimiento ni severidad de daños.")
    st.warning("La aplicación es un prototipo educativo. No reemplaza la metodología PCI, una inspección de campo ni una evaluación estructural.")
