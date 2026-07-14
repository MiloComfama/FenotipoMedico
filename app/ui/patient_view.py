"""Vista del paciente: ingreso, cuestionario conversacional y tablero."""
from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from app.config import FOLLOW_UP_DAYS
from app.db import repository as repo
from app.domain.questionnaire import load_questionnaire
from app.domain.recommendations import get_protocol
from app.services import intake as intake_service
from app.services.chat import create_intake, using_llm
from app.ui import charts
from app.ui.branding import focus_chat_input, phenotype_badge

DOC_TYPES = ["CC", "TI", "CE", "PA", "RC"]


# --- Utilidades -------------------------------------------------------------
def _reset_patient_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("pt_"):
            del st.session_state[key]


def _consultation_rows(patient) -> list[dict]:
    rows = []
    for c in sorted(patient.consultations, key=lambda x: x.consultation_number):
        m = c.measurement
        rows.append({
            "n": c.consultation_number,
            "phenotype": c.phenotype_final or c.phenotype_model,
            "date": c.consultation_date,
            "bmi": m.bmi if m else None,
            "waist_cm": m.waist_cm if m else None,
            "bp_systolic": m.bp_systolic if m else None,
            "scores": json.loads(c.classification_score) if c.classification_score else {},
        })
    return rows


# --- Pantalla de ingreso ----------------------------------------------------
def _login_screen() -> None:
    st.markdown("### Bienvenido/a a tu espacio de salud")
    st.caption(
        "Ingresa con tu documento para ver tu clasificación, tus tableros y "
        "recomendaciones, o para completar tu cuestionario."
    )
    with st.form("pt_login"):
        col1, col2 = st.columns([1, 2])
        doc_type = col1.selectbox("Tipo de documento", DOC_TYPES)
        doc_number = col2.text_input("Número de documento", placeholder="Ej: 1035********")
        submitted = st.form_submit_button("Ingresar")
    if submitted:
        if not doc_number.strip().isdigit():
            st.error("Ingresa un número de documento válido (solo dígitos).")
            return
        st.session_state["pt_doc_type"] = doc_type
        st.session_state["pt_doc_number"] = doc_number.strip()
        st.rerun()


# --- Cuestionario conversacional --------------------------------------------
def _start_survey(is_first: bool) -> None:
    questionnaire = load_questionnaire()
    intake = create_intake(questionnaire, is_first)
    opening = intake.start()
    st.session_state["pt_intake"] = intake
    st.session_state["pt_chat"] = [{"role": "assistant", "content": opening}]
    st.session_state["pt_survey_done"] = False
    st.session_state["pt_stage"] = "survey"


def _survey_screen(is_first: bool) -> None:
    st.markdown("### Cuestionario de Medicina Funcional")
    mode = "Asistente IA (Opus 4.8)" if using_llm() else "Asistente guiado"
    st.caption(f"Modo: **{mode}**. Responde con naturalidad; puedes preguntar dudas.")

    intake = st.session_state["pt_intake"]
    done, total = intake.progress()
    if total:
        st.progress(min(done / total, 1.0), text=f"Avance: {done}/{total}")

    for msg in st.session_state["pt_chat"]:
        avatar = "🧑" if msg["role"] == "user" else "💬"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if st.session_state.get("pt_survey_done"):
        st.success("Cuestionario completado. Generando tu clasificación…")
        if st.button("Ver mis resultados", type="primary"):
            _finish_survey()
        return

    user_text = st.chat_input("Escribe tu respuesta…")
    focus_chat_input()
    if user_text:
        st.session_state["pt_chat"].append({"role": "user", "content": user_text})
        turn = intake.handle(user_text)
        st.session_state["pt_chat"].append({"role": "assistant", "content": turn.assistant_text})
        if turn.done:
            st.session_state["pt_survey_done"] = True
        st.rerun()


def _finish_survey() -> None:
    intake = st.session_state["pt_intake"]
    result = intake_service.save_new_consultation(
        doc_type=st.session_state["pt_doc_type"],
        doc_number=st.session_state["pt_doc_number"],
        patient_fields={},
        answers=intake.answers,
    )
    st.session_state["pt_last_result"] = result
    for k in ("pt_intake", "pt_chat", "pt_survey_done"):
        st.session_state.pop(k, None)
    st.session_state["pt_stage"] = "dashboard"
    st.rerun()


# --- Tablero ----------------------------------------------------------------
def _dashboard_screen(patient) -> None:
    rows = _consultation_rows(patient)
    latest = rows[-1]
    phenotype = latest["phenotype"]
    scores = latest["scores"]

    fu = repo.check_follow_up(patient)

    top = st.columns([2, 1])
    with top[0]:
        st.markdown(f"#### Hola{', ' + patient.full_name if patient.full_name else ''} 👋")
        st.markdown(
            f"Tu clasificación actual es &nbsp; {phenotype_badge(phenotype)}",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Consulta #{latest['n']} · "
            f"{latest['date'].strftime('%Y-%m-%d') if latest['date'] else ''}"
        )
    with top[1]:
        if not fu.allowed:
            st.info(
                f"🗓️ Tu próxima cita de seguimiento estará disponible el "
                f"**{fu.next_allowed_date.strftime('%Y-%m-%d')}** "
                f"(faltan {fu.days_remaining} días)."
            )
        else:
            if st.button("Actualizar mi información 🔄", type="primary"):
                _start_survey(is_first=False)
                st.rerun()

    st.divider()

    # Recomendaciones + confianza
    cols = st.columns([1, 1])
    with cols[0]:
        if scores:
            st.markdown("**Afinidad por eje clínico**")
            st.plotly_chart(charts.score_bars(scores), use_container_width=True)
            conf = max(scores.values()) if scores else 0.5
            st.plotly_chart(charts.phenotype_gauge(phenotype, conf), use_container_width=True)
    with cols[1]:
        protocol = get_protocol(phenotype)
        if protocol:
            st.markdown(f"**Plan orientado a: {phenotype}**")
            st.write(protocol.summary)
            st.markdown("**Recomendaciones de estilo de vida**")
            for item in protocol.lifestyle:
                st.markdown(f"- {item}")
            st.markdown("**Qué monitorearemos**")
            for item in protocol.monitoring:
                st.markdown(f"- {item}")
            st.caption("ℹ️ " + protocol.disclaimer)

    # Evolución (si hay más de una consulta)
    if len(rows) > 1:
        st.divider()
        st.markdown("#### Tu evolución")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.phenotype_history(rows), use_container_width=True)
        with c2:
            if any(r["bmi"] for r in rows):
                st.plotly_chart(
                    charts.measurement_trend(rows, "bmi", "IMC"), use_container_width=True
                )


# --- Router de la vista de paciente -----------------------------------------
def render() -> None:
    if "pt_doc_number" not in st.session_state:
        _login_screen()
        return

    doc_type = st.session_state["pt_doc_type"]
    doc_number = st.session_state["pt_doc_number"]

    header_cols = st.columns([4, 1])
    header_cols[0].caption(f"Sesión: {doc_type} {doc_number}")
    if header_cols[1].button("Cerrar sesión"):
        _reset_patient_state()
        st.rerun()

    # Si estamos en medio de un cuestionario, seguimos ahí.
    if st.session_state.get("pt_stage") == "survey":
        _survey_screen(is_first="pt_last_result" not in st.session_state)
        return

    patient = repo.load_patient_full(doc_type, doc_number)

    if patient is None or not patient.consultations:
        st.info("No encontramos información previa. Completemos tu primer cuestionario. 📝")
        if st.button("Comenzar cuestionario", type="primary"):
            _start_survey(is_first=True)
            st.rerun()
        return

    _dashboard_screen(patient)
