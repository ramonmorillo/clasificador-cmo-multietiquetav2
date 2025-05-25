
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import numpy as np

# Cargar modelo y binarizador
modelo = joblib.load("modelo_multietiqueta_cmo_v2.pkl")
binarizador = joblib.load("binarizador_multietiqueta_cmo_v2.pkl")

UMBRAL = 0.3  # Umbral ajustado

st.title("Clasificador CMO multietiqueta")
st.write("Identifica múltiples intervenciones farmacéuticas CMO desde texto clínico libre. Incluye opciones de estratificación, comentario y exportación.")

# Registro de intervenciones
if "registro" not in st.session_state:
    st.session_state.registro = []

# Entradas de usuario
col1, col2 = st.columns(2)
with col1:
    usuario = st.text_input("👤 Identificador del usuario")
with col2:
    nivel = st.selectbox("📊 Nivel de estratificación del paciente", [
        "Nivel 1",
        "Nivel 2",
        "Nivel 3"
    ])

texto_input = st.text_area("📄 Texto clínico libre", "")
comentario_general = st.text_area("📝 Comentario general (opcional)", "")

# Opcional: mostrar tabla de probabilidades
ver_detalles = st.checkbox("Mostrar detalles de la predicción (probabilidades)", value=False)

# Clasificar
if st.button("📌 Clasificar y registrar intervención"):
    if texto_input.strip() == "" or usuario.strip() == "":
        st.warning("⚠️ Introduce un texto clínico y un identificador de usuario.")
    else:
        probas = modelo.predict_proba([texto_input])[0]
        etiquetas_activas = [etiqueta for etiqueta, prob in zip(binarizador.classes_, probas) if prob >= UMBRAL]

        if etiquetas_activas:
            st.success(f"✅ Intervenciones detectadas: {', '.join(etiquetas_activas)}")
            fila = {
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Usuario": usuario,
                "Nivel estratificación": nivel,
                "Texto": texto_input,
                "Intervenciones CMO": ", ".join(etiquetas_activas),
                "Comentario": comentario_general
            }
            st.session_state.registro.append(fila)
        else:
            st.info("🔍 No se detectaron intervenciones con suficiente confianza.")

        if ver_detalles:
            st.subheader("🔬 Probabilidades por subcategoría")
            df_probas = pd.DataFrame({
                "Código": binarizador.classes_,
                "Probabilidad": np.round(probas, 3)
            }).sort_values(by="Probabilidad", ascending=False)
            st.dataframe(df_probas)

# Botón para reiniciar campos
if st.button("➕ Registrar otra intervención"):
    st.experimental_rerun()

# Historial
if st.session_state.registro:
    df_hist = pd.DataFrame(st.session_state.registro)
    st.subheader("📚 Historial de intervenciones registradas")
    st.dataframe(df_hist)

    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar historial en CSV", csv, "historial_intervenciones_cmo.csv", "text/csv")
else:
    st.info("ℹ️ No hay intervenciones registradas aún.")
