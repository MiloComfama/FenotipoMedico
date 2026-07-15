"""Punto de entrada de la aplicación Streamlit.

Ejecutar con:  streamlit run app/main.py
"""
from __future__ import annotations

import streamlit as st

from app.config import BRAND
from app.db.database import init_db
from app.ui import branding, doctor_view, patient_view

st.set_page_config(
    page_title="Fenotipo Médico · Comfama",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    init_db()
    branding.inject_theme()
    branding.header(
        "Segmentación Inteligente de Pacientes",
        "Programa de Medicina Funcional · Comfama",
    )

    with st.sidebar:
        branding.sidebar_brand()
        st.markdown("#### Modo de ingreso")
        view = st.radio(
            "Selecciona tu perfil",
            ["👤 Paciente", "🩺 Médico"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            "Prototipo de segmentación por fenotipo clínico "
            "(Cardiometabólico · Digestivo · Mixto)."
        )
        st.caption(
            "Uso sujeto a autorización del paciente y al marco de habeas data "
            "y políticas internas de datos clínicos de Comfama."
        )

    if view.startswith("👤"):
        patient_view.render()
    else:
        doctor_view.render()


if __name__ == "__main__":
    main()
