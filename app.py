
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import math

st.set_page_config(
    page_title="Planta Virtual",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Planta Virtual de Molienda y Tamizaje")
st.write("Simulador educativo de manejo de sólidos")

st.subheader("1. Alimentación y operación")

c1, c2, c3 = st.columns(3)

with c1:
    material = st.selectbox(
        "Material",
        ["Caliza", "Arena", "Granos"]
    )
    masa = st.number_input(
        "Alimentación (kg)",
        min_value=1.0,
        value=500.0
    )

with c2:
    molino = st.selectbox(
        "Tipo de molino",
        ["Martillos", "Bolas", "Rodillos"]
    )
    f80 = st.number_input(
        "Tamaño inicial F80 (mm)",
        min_value=0.1,
        value=10.0
    )

with c3:
    intensidad = st.slider(
        "Intensidad de molienda",
        1, 10, 5
    )
    abertura = st.select_slider(
        "Abertura del tamiz (mm)",
        options=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        value=2.0
    )
    eficiencia = st.slider(
        "Eficiencia del tamiz (%)",
        50, 100, 90
    )

# Modelo simplificado de molienda:
# el tamaño característico disminuye con la intensidad.
d50 = f80 / (1 + 0.45 * intensidad)

# Distribución granulométrica lognormal ilustrativa.
# La mediana es d50 y la dispersión es constante.
sigma = 0.65

def fraccion_pasante(d):
    z = np.log(d / d50) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

fraccion_fina = fraccion_pasante(abertura)

# Separación imperfecta en el tamiz.
# Una parte de los finos puede quedar en el rechazo.
fino_recuperado = fraccion_fina * eficiencia / 100

masa_fina = masa * fino_recuperado
masa_gruesa = masa - masa_fina

# Energía específica ilustrativa (kWh/t).
energia_especifica = 8 + 1.5 * intensidad
energia_total = (masa / 1000) * energia_especifica

st.divider()
st.subheader("2. Diagrama del proceso")

a, b, c = st.columns([1, 0.3, 1])

with a:
    st.info(f"📦 Alimentación\n\n{masa:.1f} kg")

with b:
    st.markdown("### ➜")

with c:
    st.info(f"⚙️ Molino: {molino}\n\nF80: {f80:.2f} mm")

st.markdown("### ↓ Tamiz vibratorio")

r1, r2, r3, r4 = st.columns(4)

r1.metric("Producto fino", f"{masa_fina:.2f} kg")
r2.metric("Rechazo grueso", f"{masa_gruesa:.2f} kg")
r3.metric("Energía estimada", f"{energia_total:.2f} kWh")
r4.metric(
    "Recuperación de finos",
    f"{100 * masa_fina / masa:.1f}%"
)

st.subheader("3. Distribución granulométrica")

diametros = np.geomspace(
    max(f80 / 100, 0.01),
    max(f80 * 2, 0.1),
    100
)

pasantes = np.array([
    fraccion_pasante(d) * 100
    for d in diametros
])

datos = pd.DataFrame({
    "Tamaño de partícula (mm)": diametros,
    "Pasante acumulado (%)": pasantes
})

fig = px.line(
    datos,
    x="Tamaño de partícula (mm)",
    y="Pasante acumulado (%)",
    log_x=True,
    title="Distribución granulométrica estimada"
)

fig.update_yaxes(range=[0, 100])
st.plotly_chart(fig, use_container_width=True)

st.subheader("4. Balance de masa")

tabla = pd.DataFrame({
    "Corriente": ["Alimentación", "Producto fino", "Rechazo grueso"],
    "Masa (kg)": [masa, masa_fina, masa_gruesa]
})

st.dataframe(tabla, hide_index=True, use_container_width=True)

st.success(
    f"Balance: {masa:.2f} kg de entrada = "
    f"{masa_fina + masa_gruesa:.2f} kg de salida."
)

st.caption(
    "Modelo educativo simplificado. Los parámetros de molienda, "
    "la distribución de tamaños y la separación deben calibrarse "
    "con datos experimentales antes de representar una planta real."
)