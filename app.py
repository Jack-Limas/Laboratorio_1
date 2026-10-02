import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Laboratorio 1 | Minería de Datos & BI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# CSS estilizado moderno para interfaz profesional
st.markdown("""
<style>
    /* Estilo del contenedor principal */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #86efac;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .prediction-title {
        color: #166534;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .prediction-result {
        color: #14532d;
        font-size: 2.4rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo(nombre_archivo):
    ruta = os.path.join(MODELS_DIR, nombre_archivo)
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None


@st.cache_data
def cargar_datos(nombre_archivo):
    ruta = os.path.join(DATA_DIR, nombre_archivo)
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    return None


# Barra Lateral (Sidebar)
st.sidebar.image("https://img.icons8.com/clouds/200/combo-chart.png", width=120)
st.sidebar.title("Minería de Datos")
st.sidebar.markdown("**Laboratorio 1 | CRISP-DM**")
st.sidebar.markdown("---")

opcion_escenario = st.sidebar.radio(
    "Seleccione el Escenario:",
    [
        "💵 1. Precio del Dólar",
        "🩸 2. Niveles de Glucosa",
        "⚡ 3. Consumo de Energía",
        "📋 4. Comparativa Global & CRISP-DM"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Fases de Metodología CRISP-DM:**
1. Comprensión del Negocio
2. Comprensión de los Datos
3. Preparación de los Datos
4. Modelado
5. Evaluación
6. Despliegue
""")
st.sidebar.info("Universidad Cooperativa de Colombia\nInteligencia de Negocios y Minería de Datos")


# =============================================================================
# VISTA 1: PRECIO DEL DÓLAR
# =============================================================================
if "1. Precio del Dólar" in opcion_escenario:
    st.markdown("""
    <div class="main-header">
        <h1>💵 Predicción del Precio del Dólar</h1>
        <p>Regresión Lineal Múltiple aplicada a variables macroeconómicas y temporales.</p>
    </div>
    """, unsafe_allow_html=True)

    modelo_info = cargar_modelo('modelo_dolar.joblib')
    df_dolar = cargar_datos('dolar_data.csv')

    if modelo_info is None:
        st.error("⚠️ El modelo `modelo_dolar.joblib` no ha sido encontrado. Ejecute `run_all.py` o `main_analysis.py`.")
    else:
        metrics = modelo_info['metrics']
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("R² (Bondad de Ajuste)", f"{metrics['r2']:.4f}", f"{metrics['r2']*100:.1f}%")
        with col_m2:
            st.metric("MSE (Error Cuadrático Medio)", f"{metrics['mse']:.2f}")
        with col_m3:
            st.metric("RMSE (Raíz de MSE)", f"{metrics['rmse']:.2f} COP")
        with col_m4:
            st.metric("Intercepto (b₀)", f"{modelo_info['intercept']:.2f}")

        st.markdown("---")

        tab_pred, tab_model, tab_plots, tab_data = st.tabs([
            "🎯 Inferencia & Predicción",
            "📐 Ecuación y Coeficientes",
            "📈 Visualizaciones & Gráficos",
            "📊 Exploración del Dataset"
        ])

        with tab_pred:
            st.subheader("Calcular Predicción en Tiempo Real")
            st.write("Ingrese los valores de las variables independientes para estimar el precio del dólar:")

            stats = modelo_info.get('stats', {})
            c1, c2, c3 = st.columns(3)

            with c1:
                dia_default = int(stats.get('mean', {}).get('Dia', 250))
                val_dia = st.number_input("Día de la Serie / Período (1 a 1000)", min_value=1, max_value=1000, value=dia_default, step=1)
            with c2:
                inf_default = float(stats.get('mean', {}).get('Inflacion', 0.020))
                val_inflacion = st.number_input("Inflación mensual (Ej: 0.02 = 2%)", min_value=-0.1, max_value=1.0, value=inf_default, format="%.5f", step=0.001)
            with c3:
                tasa_default = float(stats.get('mean', {}).get('Tasa_interes', 5.0))
                val_tasa = st.number_input("Tasa de interés (%) (Ej: 5.0%)", min_value=0.0, max_value=50.0, value=tasa_default, format="%.3f", step=0.1)

            if st.button("🚀 Calcular Predicción del Dólar", key="btn_dolar", use_container_width=True):
                x_input = np.array([[val_dia, val_inflacion, val_tasa]])
                prediccion = modelo_info['model'].predict(x_input)[0]

                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-title">PRECIO ESTIMADO DEL DÓLAR (COP)</div>
                    <div class="prediction-result">${prediccion:,.2f} COP</div>
                    <p style="margin-top: 8px; color: #15803d; font-size: 0.95rem;">
                        Intervalo de error esperado aproximado (± 1 RMSE): <b>${prediccion - metrics['rmse']:,.2f}</b> a <b>${prediccion + metrics['rmse']:,.2f}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with tab_model:
            st.subheader("Ecuación del Modelo de Regresión")
            b0 = modelo_info['intercept']
            b_dia = modelo_info['coefficients']['Dia']
            b_inf = modelo_info['coefficients']['Inflacion']
            b_tas = modelo_info['coefficients']['Tasa_interes']

            latex_eq = rf"\text{{Precio\_Dolar}} = {b0:.4f} + ({b_dia:.4f} \cdot \text{{Dia}}) + ({b_inf:.4f} \cdot \text{{Inflacion}}) + ({b_tas:.4f} \cdot \text{{Tasa\_interes}})"
            st.latex(latex_eq)

            st.write("#### Tabla de Coeficientes e Impacto")
            coef_data = []
            for var, coef in modelo_info['coefficients'].items():
                coef_data.append({
                    "Variable": var,
                    "Coeficiente (Pendiente)": f"{coef:.4f}",
                    "Dirección del Efecto": "📈 Incrementa el precio" if coef > 0 else "📉 Disminuye el precio",
                    "Interpretación": f"Por cada incremento unitario en {var}, el dólar varía en {coef:.4f} COP."
                })
            st.table(pd.DataFrame(coef_data))

        with tab_plots:
            st.subheader("Gráficos de Dispersión y Líneas de Tendencia")
            plot_path = os.path.join(PLOTS_DIR, 'dolar_plots.png')
            if os.path.exists(plot_path):
                img = Image.open(plot_path)
                st.image(img, caption="Regresión Lineal: Variables vs Precio del Dólar", use_container_width=True)
            else:
                st.warning("El gráfico `dolar_plots.png` no fue encontrado.")

        with tab_data:
            if df_dolar is not None:
                st.subheader("Muestra de Datos (dolar_data.csv)")
                st.dataframe(df_dolar.head(50), use_container_width=True)
                st.write("**Estadísticas Descriptivas:**")
                st.dataframe(df_dolar.describe(), use_container_width=True)


# =============================================================================
# VISTA 2: NIVELES DE GLUCOSA
# =============================================================================
elif "2. Niveles de Glucosa" in opcion_escenario:
    st.markdown("""
    <div class="main-header">
        <h1>🩸 Predicción de Niveles de Glucosa</h1>
        <p>Análisis biomédico sobre el impacto de la Edad, el Índice de Masa Corporal (IMC) y la Actividad Física.</p>
    </div>
    """, unsafe_allow_html=True)

    modelo_info = cargar_modelo('modelo_glucosa.joblib')
    df_glucosa = cargar_datos('glucosa_data.csv')

    if modelo_info is None:
        st.error("⚠️ El modelo `modelo_glucosa.joblib` no ha sido encontrado. Ejecute `run_all.py` o `main_analysis.py`.")
    else:
        metrics = modelo_info['metrics']
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("R² (Bondad de Ajuste)", f"{metrics['r2']:.4f}", f"{metrics['r2']*100:.1f}%")
        with col_m2:
            st.metric("MSE", f"{metrics['mse']:.2f}")
        with col_m3:
            st.metric("RMSE", f"{metrics['rmse']:.2f} mg/dL")
        with col_m4:
            st.metric("Intercepto (b₀)", f"{modelo_info['intercept']:.2f}")

        st.markdown("---")

        tab_pred, tab_model, tab_plots, tab_data = st.tabs([
            "🎯 Inferencia & Predicción",
            "📐 Ecuación y Análisis Clínico",
            "📈 Visualizaciones & Gráficos",
            "📊 Exploración del Dataset"
        ])

        with tab_pred:
            st.subheader("Calcular Predicción de Glucosa en Sangre")
            stats = modelo_info.get('stats', {})
            c1, c2, c3 = st.columns(3)

            with c1:
                edad_default = int(stats.get('mean', {}).get('Edad', 45))
                val_edad = st.number_input("Edad del paciente (Años)", min_value=1, max_value=120, value=edad_default, step=1)
            with c2:
                imc_default = float(stats.get('mean', {}).get('IMC', 25.0))
                val_imc = st.number_input("Índice de Masa Corporal - IMC (kg/m²)", min_value=10.0, max_value=60.0, value=imc_default, format="%.2f", step=0.1)
            with c3:
                act_default = int(stats.get('mean', {}).get('Actividad_Fisica', 3))
                val_act = st.number_input("Horas de Actividad Física Semanal", min_value=0, max_value=40, value=act_default, step=1)

            if st.button("🚀 Calcular Nivel de Glucosa", key="btn_glucosa", use_container_width=True):
                x_input = np.array([[val_edad, val_imc, val_act]])
                prediccion = modelo_info['model'].predict(x_input)[0]

                if prediccion < 100:
                    categoria = "Normal (< 100 mg/dL)"
                    badge_color = "#15803d"
                elif 100 <= prediccion <= 125:
                    categoria = "Prediabetes (100 - 125 mg/dL)"
                    badge_color = "#b45309"
                else:
                    categoria = "Diabetes / Nivel Alto (≥ 126 mg/dL)"
                    badge_color = "#b91c1c"

                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-title">GLUCOSA EN AYUNAS ESTIMADA</div>
                    <div class="prediction-result">{prediccion:.2f} mg/dL</div>
                    <p style="margin-top: 10px; font-weight: 600; color: {badge_color};">
                        Diagnóstico Orientativo: {categoria}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with tab_model:
            st.subheader("Ecuación del Modelo de Glucosa")
            b0 = modelo_info['intercept']
            b_edad = modelo_info['coefficients']['Edad']
            b_imc = modelo_info['coefficients']['IMC']
            b_act = modelo_info['coefficients']['Actividad_Fisica']

            latex_eq = rf"\text{{Nivel\_Glucosa}} = {b0:.4f} + ({b_edad:.4f} \cdot \text{{Edad}}) + ({b_imc:.4f} \cdot \text{{IMC}}) + ({b_act:.4f} \cdot \text{{Actividad\_Fisica}})"
            st.latex(latex_eq)

            st.write("#### Factores Protectores vs Factores de Riesgo")
            st.success(f"💪 **Factor Protector (Impacto Negativo en Glucosa):** `Actividad_Fisica` ({b_act:.4f}). Realizar más ejercicio disminuye los niveles de glucosa.")
            st.warning(f"⚠️ **Factores de Riesgo (Impacto Positivo en Glucosa):** `IMC` ({b_imc:.4f}) y `Edad` ({b_edad:.4f}). Mayor peso y edad incrementan la glucosa basal.")

        with tab_plots:
            st.subheader("Comportamiento Gráfico de Variables Clínicas")
            plot_path = os.path.join(PLOTS_DIR, 'glucosa_plots.png')
            if os.path.exists(plot_path):
                img = Image.open(plot_path)
                st.image(img, caption="Regresión Lineal: Factores de Estilo de Vida y Biológicos vs Glucosa", use_container_width=True)
            else:
                st.warning("El gráfico `glucosa_plots.png` no fue encontrado.")

        with tab_data:
            if df_glucosa is not None:
                st.subheader("Muestra de Datos (glucosa_data.csv)")
                st.dataframe(df_glucosa.head(50), use_container_width=True)
                st.write("**Estadísticas Descriptivas:**")
                st.dataframe(df_glucosa.describe(), use_container_width=True)


# =============================================================================
# VISTA 3: CONSUMO DE ENERGÍA
# =============================================================================
elif "3. Consumo de Energía" in opcion_escenario:
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Predicción de Consumo de Energía Eléctrica</h1>
        <p>Optimización energética y modelado basado en Temperatura Ambiental, Hora y Día de la Semana.</p>
    </div>
    """, unsafe_allow_html=True)

    modelo_info = cargar_modelo('modelo_energia.joblib')
    df_energia = cargar_datos('energia_data.csv')

    if modelo_info is None:
        st.error("⚠️ El modelo `modelo_energia.joblib` no ha sido encontrado. Ejecute `run_all.py` o `main_analysis.py`.")
    else:
        metrics = modelo_info['metrics']
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("R² (Bondad de Ajuste)", f"{metrics['r2']:.4f}", f"{metrics['r2']*100:.1f}%")
        with col_m2:
            st.metric("RMSE (Error Cuadrático Raíz)", f"{metrics['rmse']:.3f} kWh")
        with col_m3:
            st.metric("MSE", f"{metrics['mse']:.3f}")
        with col_m4:
            st.metric("Intercepto (b₀)", f"{modelo_info['intercept']:.2f}")

        st.markdown("---")

        tab_pred, tab_model, tab_plots, tab_data = st.tabs([
            "🎯 Inferencia & Predicción",
            "📐 Ecuación y Matriz de Correlación",
            "📈 Heatmap & Dispersión",
            "📊 Exploración del Dataset"
        ])

        with tab_pred:
            st.subheader("Calcular Consumo Energético Previsto")
            stats = modelo_info.get('stats', {})
            c1, c2, c3 = st.columns(3)

            with c1:
                temp_default = float(stats.get('mean', {}).get('Temperatura', 22.0))
                val_temp = st.number_input("Temperatura (°C)", min_value=-15.0, max_value=55.0, value=temp_default, format="%.2f", step=0.5)
            with c2:
                hora_default = int(stats.get('mean', {}).get('Hora', 12))
                val_hora = st.slider("Hora del Día (1 a 24 hrs)", min_value=1, max_value=24, value=hora_default)
            with c3:
                dia_default = int(stats.get('mean', {}).get('Dia_Semana', 3))
                val_diasem = st.selectbox(
                    "Día de la Semana",
                    options=[1, 2, 3, 4, 5, 6, 7],
                    format_func=lambda x: {1: "1 - Lunes", 2: "2 - Martes", 3: "3 - Miércoles", 4: "4 - Jueves", 5: "5 - Viernes", 6: "6 - Sábado", 7: "7 - Domingo"}[x],
                    index=dia_default-1 if 1 <= dia_default <= 7 else 0
                )

            if st.button("🚀 Calcular Consumo Energético", key="btn_energia", use_container_width=True):
                x_input = np.array([[val_temp, val_hora, val_diasem]])
                prediccion = modelo_info['model'].predict(x_input)[0]

                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-title">CONSUMO ENERGÉTICO ESTIMADO</div>
                    <div class="prediction-result">{prediccion:,.2f} kWh</div>
                    <p style="margin-top: 8px; color: #166534; font-size: 0.95rem;">
                        Margen de incertidumbre estadística: <b>± {metrics['rmse']:.2f} kWh</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with tab_model:
            st.subheader("Ecuación del Modelo Energético")
            b0 = modelo_info['intercept']
            b_temp = modelo_info['coefficients']['Temperatura']
            b_hora = modelo_info['coefficients']['Hora']
            b_dia = modelo_info['coefficients']['Dia_Semana']

            latex_eq = rf"\text{{Consumo\_Energia}} = {b0:.4f} + ({b_temp:.4f} \cdot \text{{Temperatura}}) + ({b_hora:.4f} \cdot \text{{Hora}}) + ({b_dia:.4f} \cdot \text{{Dia\_Semana}})"
            st.latex(latex_eq)

            st.write("#### Impacto por Variable:")
            st.markdown(f"- **Temperatura:** Coeficiente `{b_temp:.4f}`. Un aumento de 1°C varía el consumo en {b_temp:.4f} kWh (impacto directo de climatización).")
            st.markdown(f"- **Hora:** Coeficiente `{b_hora:.4f}`. Refleja los ciclos operativos a lo largo del día.")
            st.markdown(f"- **Día de la Semana:** Coeficiente `{b_dia:.4f}`. Modula el consumo entre jornadas laborales y fines de semana.")

        with tab_plots:
            st.subheader("Mapa de Calor (Heatmap) y Gráficos de Regresión")
            plot_path = os.path.join(PLOTS_DIR, 'energia_plots.png')
            if os.path.exists(plot_path):
                img = Image.open(plot_path)
                st.image(img, caption="Matriz de Correlación Pearson y Regresiones de Energía", use_container_width=True)
            else:
                st.warning("El gráfico `energia_plots.png` no fue encontrado.")

        with tab_data:
            if df_energia is not None:
                st.subheader("Muestra de Datos (energia_data.csv)")
                st.dataframe(df_energia.head(50), use_container_width=True)
                st.write("**Estadísticas Descriptivas:**")
                st.dataframe(df_energia.describe(), use_container_width=True)


# =============================================================================
# VISTA 4: COMPARATIVA GLOBAL & METODOLOGÍA CRISP-DM
# =============================================================================
else:
    st.markdown("""
    <div class="main-header">
        <h1>📋 Panel Comparativo & Metodología CRISP-DM</h1>
        <p>Visión integral de los tres modelos de minería de datos, métricas y fases del estándar industrial.</p>
    </div>
    """, unsafe_allow_html=True)

    m_dolar = cargar_modelo('modelo_dolar.joblib')
    m_glucosa = cargar_modelo('modelo_glucosa.joblib')
    m_energia = cargar_modelo('modelo_energia.joblib')

    st.subheader("Resumen Comparativo de Modelos Entrenados")

    tabla_comparativa = []
    if m_dolar:
        tabla_comparativa.append({
            "Proyecto": "1. Precio del Dólar",
            "Variable Objetivo (Y)": "Precio_Dolar",
            "Variables Predictoras (X)": "Dia, Inflacion, Tasa_interes",
            "R²": f"{m_dolar['metrics']['r2']:.4f}",
            "MSE": f"{m_dolar['metrics']['mse']:.2f}",
            "RMSE": f"{m_dolar['metrics']['rmse']:.2f}"
        })
    if m_glucosa:
        tabla_comparativa.append({
            "Proyecto": "2. Niveles de Glucosa",
            "Variable Objetivo (Y)": "Nivel_Glucosa",
            "Variables Predictoras (X)": "Edad, IMC, Actividad_Fisica",
            "R²": f"{m_glucosa['metrics']['r2']:.4f}",
            "MSE": f"{m_glucosa['metrics']['mse']:.2f}",
            "RMSE": f"{m_glucosa['metrics']['rmse']:.2f}"
        })
    if m_energia:
        tabla_comparativa.append({
            "Proyecto": "3. Consumo de Energía",
            "Variable Objetivo (Y)": "Consumo_Energia",
            "Variables Predictoras (X)": "Temperatura, Hora, Dia_Semana",
            "R²": f"{m_energia['metrics']['r2']:.4f}",
            "MSE": f"{m_energia['metrics']['mse']:.2f}",
            "RMSE": f"{m_energia['metrics']['rmse']:.2f}"
        })

    if tabla_comparativa:
        st.table(pd.DataFrame(tabla_comparativa))
    else:
        st.info("Ejecute el entrenamiento para visualizar las métricas consolidadas.")

    st.markdown("---")
    st.subheader("Aplicación del Marco de Trabajo CRISP-DM en este Laboratorio")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. Comprensión del Negocio (Business Understanding):**
        - Identificar problemas de toma de decisiones financieras, de salud preventiva y eficiencia energética.
        - Definición de los objetivos analíticos y formulación cuantitativa.

        **2. Comprensión de los Datos (Data Understanding):**
        - Recolección y exploración de los tres conjuntos de datos (`dolar_data.csv`, `glucosa_data.csv`, `energia_data.csv`).
        - Identificación de distribuciones, tipos de datos y dispersiones.

        **3. Preparación de los Datos (Data Preparation):**
        - Limpieza, estructuración de matrices de características $X$ y vectores objetivo $y$.
        - Verificación de consistencia y tipos numéricos continuos.
        """)

    with c2:
        st.markdown("""
        **4. Modelado (Modeling):**
        - Entrenamiento de modelos de Regresión Lineal Múltiple con Scikit-Learn.
        - Estimación analítica por Mínimos Cuadrados Ordinarios (OLS).

        **5. Evaluación (Evaluation):**
        - Cálculo riguroso de métricas: Coeficiente de Determinación $R^2$, Error Cuadrático Medio (MSE) y RMSE.
        - Análisis de la magnitud y significancia física de cada coeficiente.

        **6. Despliegue (Deployment):**
        - Persistencia de modelos serializados con `joblib`.
        - Creación de una aplicación web interactiva en tiempo real con **Streamlit**.
        - Script orquestador `run_all.py` para ejecución integral y reproducible.
        """)
