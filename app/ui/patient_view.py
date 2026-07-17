"""Vista del paciente: ingreso, cuestionario conversacional y tablero."""
from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from app.config import ASSETS_DIR, FOLLOW_UP_DAYS
from app.db import repository as repo
from app.domain.portfolio import get_portfolio_recommendation
from app.domain.questionnaire import load_questionnaire
from app.domain.recommendations import get_protocol
from app.services import intake as intake_service
from app.services import stt, tts
from app.services.chat import create_intake, using_llm
from app.ui import charts
from app.ui.branding import focus_chat_input, phenotype_badge

DOC_TYPES = ["CC", "TI", "CE", "PA", "RC"]
FENIX_AVATAR = str(ASSETS_DIR / "BotIcon.png")


# --- Utilidades -------------------------------------------------------------
def _reset_patient_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("pt_"):
            del st.session_state[key]


def _decode_classification(raw: str | None) -> tuple[dict, dict]:
    """Lee el score guardado; soporta el formato legado (solo puntajes planos)
    y el formato actual ``{"scores": ..., "rationale": ...}``."""
    if not raw:
        return {}, {}
    data = json.loads(raw)
    if isinstance(data, dict) and "scores" in data:
        return data.get("scores") or {}, data.get("rationale") or {}
    return data, {}


def _consultation_rows(patient) -> list[dict]:
    rows = []
    for c in sorted(patient.consultations, key=lambda x: x.consultation_number):
        m = c.measurement
        scores, rationale = _decode_classification(c.classification_score)
        rows.append({
            "n": c.consultation_number,
            "phenotype": c.phenotype_final or c.phenotype_model,
            "date": c.consultation_date,
            "bmi": m.bmi if m else None,
            "waist_cm": m.waist_cm if m else None,
            "bp_systolic": m.bp_systolic if m else None,
            "scores": scores,
            "rationale": rationale,
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


def _submit_answer(intake, text: str) -> None:
    st.session_state["pt_chat"].append({"role": "user", "content": text})
    turn = intake.handle(text)
    st.session_state["pt_chat"].append({"role": "assistant", "content": turn.assistant_text})
    if turn.done:
        st.session_state["pt_survey_done"] = True
    st.rerun()


def _survey_screen(is_first: bool) -> None:
    st.markdown("### Cuestionario de Medicina Funcional")
    mode = "IA · Opus 4.8" if using_llm() else "modo guiado"
    st.caption(f"Hablas con **Fénix** ({mode}). Responde con naturalidad; puedes preguntar dudas.")

    intake = st.session_state["pt_intake"]
    done, total = intake.progress()
    if total:
        st.progress(min(done / total, 1.0), text=f"Avance: {done}/{total}")

    just_generated_idx = st.session_state.pop("pt_tts_just_generated", None)
    for i, msg in enumerate(st.session_state["pt_chat"]):
        avatar = "🧑" if msg["role"] == "user" else FENIX_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and tts.is_configured():
                audio_key = f"pt_tts_audio_{i}"
                if st.button("🔊 Escuchar", key=f"pt_tts_btn_{i}"):
                    with st.spinner("Generando audio…"):
                        try:
                            st.session_state[audio_key] = tts.synthesize(msg["content"])
                            just_generated_idx = i
                        except Exception as e:
                            st.error(f"No se pudo generar el audio: {e}")
                audio_bytes = st.session_state.get(audio_key)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=(i == just_generated_idx))

    if st.session_state.get("pt_survey_done"):
        st.success("Cuestionario completado. Generando tu clasificación…")
        if st.button("Ver mis resultados", type="primary"):
            _finish_survey()
        return

    col_input, col_mic = st.columns([6, 1])
    with col_input:
        user_text = st.chat_input("Escribe tu respuesta…")

    voice_text = None
    voice_error = None
    with col_mic:
        if stt.is_configured():
            with st.popover("🎤"):
                audio_value = st.audio_input(
                    "Grábate y vuelve a dar clic para detener", key="pt_voice_input"
                )
                if audio_value is not None:
                    audio_bytes = audio_value.getvalue()
                    audio_hash = hash(audio_bytes)
                    if audio_hash != st.session_state.get("pt_last_voice_hash"):
                        st.session_state["pt_last_voice_hash"] = audio_hash
                        with st.spinner("Transcribiendo tu respuesta…"):
                            try:
                                voice_text = stt.transcribe(audio_bytes)
                                if not voice_text:
                                    voice_error = (
                                        "No detecté voz en la grabación. Intenta de nuevo, "
                                        "hablando cerca del micrófono."
                                    )
                            except Exception as e:
                                voice_error = f"No se pudo transcribir el audio: {e}"

    if voice_error:
        st.error(voice_error)

    focus_chat_input()
    if user_text:
        _submit_answer(intake, user_text)
    elif voice_text:
        _submit_answer(intake, voice_text)


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
    rationale = latest["rationale"]

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
            st.markdown("**Perfil de Afinidad Clínica**")
            st.caption(
                "Qué tan relacionadas están tus respuestas de la encuesta con cada perfil "
                "de salud. Pasa el cursor sobre cada barra para ver el detalle."
            )
            st.plotly_chart(charts.score_bars(scores, rationale), use_container_width=True)
            with st.expander("¿Por qué obtuve este resultado?"):
                for axis, reasons in rationale.items():
                    st.markdown(f"**{axis}**")
                    for r in reasons:
                        st.markdown(f"- {r}")
            st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
            st.markdown("**Categoría del fenotipo**")
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

            portfolio = get_portfolio_recommendation(phenotype)
            if portfolio:
                st.markdown("**Servicios del portafolio Comfama para ti**")
                for name, desc in portfolio.programs:
                    st.markdown(f"- **{name}**" + (f" — {desc}" if desc else ""))

            st.caption("ℹ️ " + protocol.disclaimer)
            st.markdown(
                "🔗 [Conoce e ingresa a los servicios de Comfama](https://www.comfama.com)"
            )

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
