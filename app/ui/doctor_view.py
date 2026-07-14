"""Vista del médico (administrador): consulta, datos clínicos y reclasificación."""
from __future__ import annotations

import json

import streamlit as st

from app.config import PHENOTYPES
from app.db import repository as repo
from app.db.database import get_session
from app.db.models import Consultation
from app.domain.recommendations import get_protocol
from app.services import intake as intake_service
from app.ui import charts
from app.ui.branding import phenotype_badge
from app.ui.patient_view import DOC_TYPES, _consultation_rows

# PIN de demostración para el acceso profesional (solo prototipo).
DOCTOR_PIN = "comfama"

# Laboratorios editables por el médico (EAV). Ampliable en el futuro.
LAB_FIELDS = [
    ("colesterol_total", "Colesterol total", "mg/dL"),
    ("hdl", "HDL", "mg/dL"),
    ("ldl", "LDL", "mg/dL"),
    ("trigliceridos", "Triglicéridos", "mg/dL"),
    ("glicemia", "Glicemia en ayunas", "mg/dL"),
]


# --- Persistencia -----------------------------------------------------------
def _save_measurement(consultation_id: int, data: dict) -> None:
    with get_session() as session:
        c = session.get(Consultation, consultation_id)
        repo.upsert_measurement(session, c, **data)


def _save_labs(consultation_id: int, labs: dict[str, float | None]) -> None:
    with get_session() as session:
        c = session.get(Consultation, consultation_id)
        for key, label, unit in LAB_FIELDS:
            repo.set_lab_result(session, c, key, label, labs.get(key), unit)


def _apply_reclassify(consultation_id: int, phenotype: str, doctor: str, notes: str) -> None:
    with get_session() as session:
        c = session.get(Consultation, consultation_id)
        repo.reclassify(session, c, phenotype, doctor, notes)


# --- Gate de acceso ---------------------------------------------------------
def _gate() -> bool:
    if st.session_state.get("dr_auth"):
        return True
    st.markdown("### Acceso profesional")
    st.caption("Consola para el equipo asistencial del programa.")
    with st.form("dr_gate"):
        pin = st.text_input("PIN de acceso", type="password")
        ok = st.form_submit_button("Ingresar")
    if ok:
        if pin == DOCTOR_PIN:
            st.session_state["dr_auth"] = True
            st.rerun()
        else:
            st.error("PIN incorrecto.")
    return False


# --- Pantalla principal -----------------------------------------------------
def render() -> None:
    if not _gate():
        return

    st.markdown("### Consola médica")
    with st.form("dr_search"):
        c1, c2, c3 = st.columns([1, 2, 1])
        doc_type = c1.selectbox("Tipo de documento", DOC_TYPES)
        doc_number = c2.text_input("Número de documento")
        search = c3.form_submit_button("Buscar paciente")
    if search:
        st.session_state["dr_doc"] = (doc_type, doc_number.strip())

    if "dr_doc" not in st.session_state:
        return

    doc_type, doc_number = st.session_state["dr_doc"]
    patient = repo.load_patient_full(doc_type, doc_number)
    if patient is None or not patient.consultations:
        st.warning("No se encontró un paciente con consultas para ese documento.")
        return

    rows = _consultation_rows(patient)
    latest = rows[-1]

    # --- Resumen del paciente ---
    st.divider()
    info = st.columns([2, 1, 1])
    info[0].markdown(
        f"**Paciente:** {patient.full_name or '—'}  \n"
        f"**Documento:** {doc_type} {doc_number}  \n"
        f"**Sexo:** {patient.sex or '—'} · **Nacimiento:** {patient.birthdate or '—'}"
    )
    info[1].metric("Consultas", len(rows))
    info[2].markdown("**Clasificación actual**")
    info[2].markdown(phenotype_badge(latest["phenotype"]), unsafe_allow_html=True)

    # --- Selección de consulta a editar ---
    numbers = [r["n"] for r in rows]
    sel_n = st.selectbox(
        "Consulta a gestionar", numbers, index=len(numbers) - 1,
        format_func=lambda n: f"Consulta #{n}",
    )
    consultation_id = _consultation_id_for(patient, sel_n)

    tab_data, tab_class, tab_charts = st.tabs(
        ["🩺 Datos clínicos", "🧭 Clasificación", "📊 Gráficas"]
    )

    with tab_data:
        _clinical_data_form(patient, sel_n, consultation_id)

    with tab_class:
        _classification_panel(consultation_id, latest["phenotype"], rows, sel_n)

    with tab_charts:
        _charts_panel(rows)


def _consultation_id_for(patient, n: int) -> int:
    return next(c.id for c in patient.consultations if c.consultation_number == n)


def _current_measurement(patient, n: int):
    c = next(c for c in patient.consultations if c.consultation_number == n)
    return c.measurement, {l.test_key: l.value for l in c.labs}


def _clinical_data_form(patient, n: int, consultation_id: int) -> None:
    m, labs = _current_measurement(patient, n)
    st.markdown("**Datos básicos** (el IMC se calcula automáticamente)")
    with st.form(f"dr_meas_{consultation_id}"):
        c1, c2, c3 = st.columns(3)
        weight = c1.number_input("Peso (kg)", 0.0, 400.0, float(m.weight_kg or 0) if m else 0.0, 0.1)
        height = c2.number_input("Estatura (m)", 0.0, 2.5, float(m.height_m or 0) if m else 0.0, 0.01)
        waist = c3.number_input("Perímetro abdominal (cm)", 0.0, 250.0, float(m.waist_cm or 0) if m else 0.0, 0.5)
        c4, c5 = st.columns(2)
        sbp = c4.number_input("Presión sistólica (mmHg)", 0, 300, int(m.bp_systolic or 0) if m else 0)
        dbp = c5.number_input("Presión diastólica (mmHg)", 0, 200, int(m.bp_diastolic or 0) if m else 0)

        bmi = round(weight / (height ** 2), 1) if weight and height else None
        st.metric("IMC (autocalculado)", bmi if bmi else "—")

        st.markdown("**Laboratorios**")
        lab_cols = st.columns(len(LAB_FIELDS))
        lab_values = {}
        for col, (key, label, unit) in zip(lab_cols, LAB_FIELDS):
            lab_values[key] = col.number_input(
                f"{label} ({unit})", 0.0, 1000.0,
                float(labs.get(key) or 0), 1.0, key=f"lab_{consultation_id}_{key}",
            ) or None

        saved = st.form_submit_button("Guardar datos clínicos", type="primary")

    if saved:
        _save_measurement(consultation_id, {
            "weight_kg": weight or None,
            "height_m": height or None,
            "bmi": bmi,
            "waist_cm": waist or None,
            "bp_systolic": sbp or None,
            "bp_diastolic": dbp or None,
        })
        _save_labs(consultation_id, lab_values)
        result = intake_service.recompute_classification(consultation_id)
        st.success(
            f"Datos guardados. Sugerencia del modelo actualizada: **{result.phenotype}**."
        )
        st.rerun()


def _classification_panel(consultation_id: int, current_phenotype: str, rows, n: int) -> None:
    row = next(r for r in rows if r["n"] == n)
    scores = row["scores"]
    st.markdown("**Sugerencia del modelo**")
    if scores:
        st.plotly_chart(charts.score_bars(scores), use_container_width=True)

    st.markdown("**Reclasificación por criterio médico**")
    with st.form(f"dr_reclass_{consultation_id}"):
        idx = PHENOTYPES.index(row["phenotype"]) if row["phenotype"] in PHENOTYPES else 0
        phenotype = st.selectbox("Fenotipo definitivo", PHENOTYPES, index=idx)
        doctor = st.text_input("Profesional que reclasifica")
        notes = st.text_area("Notas clínicas (opcional)")
        ok = st.form_submit_button("Guardar reclasificación", type="primary")
    if ok:
        if not doctor.strip():
            st.error("Indica el nombre del profesional.")
        else:
            _apply_reclassify(consultation_id, phenotype, doctor.strip(), notes.strip())
            st.success(f"Paciente reclasificado como **{phenotype}** por {doctor}.")
            st.rerun()

    protocol = get_protocol(row["phenotype"])
    if protocol:
        with st.expander("Protocolo sugerido para este fenotipo"):
            st.write(protocol.summary)
            for item in protocol.focus_areas:
                st.markdown(f"- {item}")


def _charts_panel(rows) -> None:
    if len(rows) > 1:
        st.plotly_chart(charts.phenotype_history(rows), use_container_width=True)
        c1, c2 = st.columns(2)
        if any(r["bmi"] for r in rows):
            c1.plotly_chart(charts.measurement_trend(rows, "bmi", "IMC"), use_container_width=True)
        if any(r["waist_cm"] for r in rows):
            c2.plotly_chart(
                charts.measurement_trend(rows, "waist_cm", "Perímetro abdominal (cm)"),
                use_container_width=True,
            )
    else:
        st.info("Se necesitan al menos dos consultas para mostrar tendencias.")
