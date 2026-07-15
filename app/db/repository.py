"""Operaciones de acceso a datos (repositorio)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import FOLLOW_UP_DAYS
from app.db.database import get_session
from app.db.models import (
    Consultation,
    LabResult,
    Measurement,
    Patient,
    SurveyAnswer,
)


# --- Pacientes --------------------------------------------------------------
def get_patient(session: Session, doc_type: str, doc_number: str) -> Patient | None:
    stmt = (
        select(Patient)
        .where(Patient.doc_type == doc_type, Patient.doc_number == str(doc_number))
        .options(selectinload(Patient.consultations))
    )
    return session.scalars(stmt).first()


def get_or_create_patient(
    session: Session, doc_type: str, doc_number: str, **fields: Any
) -> Patient:
    patient = get_patient(session, doc_type, doc_number)
    if patient is None:
        patient = Patient(doc_type=doc_type, doc_number=str(doc_number), **fields)
        session.add(patient)
        session.flush()
    else:
        for k, v in fields.items():
            if v:
                setattr(patient, k, v)
    return patient


def load_patient_full(doc_type: str, doc_number: str) -> Patient | None:
    """Carga un paciente con todas sus consultas y relaciones (para lectura)."""
    with get_session() as session:
        stmt = (
            select(Patient)
            .where(Patient.doc_type == doc_type, Patient.doc_number == str(doc_number))
            .options(
                selectinload(Patient.consultations).selectinload(Consultation.answers),
                selectinload(Patient.consultations).selectinload(Consultation.measurement),
                selectinload(Patient.consultations).selectinload(Consultation.labs),
            )
        )
        patient = session.scalars(stmt).first()
        if patient:
            session.expunge_all()
        return patient


# --- Regla de cita de seguimiento (1 mes) -----------------------------------
@dataclass
class FollowUpStatus:
    allowed: bool
    last_date: datetime | None
    next_allowed_date: datetime | None
    days_remaining: int
    is_first: bool


def check_follow_up(patient: Patient | None, today: datetime | None = None) -> FollowUpStatus:
    """Determina si el paciente puede registrar una nueva consulta."""
    today = today or datetime.now()
    if patient is None or not patient.consultations:
        return FollowUpStatus(True, None, None, 0, True)

    last = max(patient.consultations, key=lambda c: c.consultation_date)
    next_allowed = last.consultation_date + timedelta(days=FOLLOW_UP_DAYS)
    days_remaining = (next_allowed.date() - today.date()).days
    return FollowUpStatus(
        allowed=today >= next_allowed,
        last_date=last.consultation_date,
        next_allowed_date=next_allowed,
        days_remaining=max(days_remaining, 0),
        is_first=False,
    )


# --- Consultas --------------------------------------------------------------
def next_consultation_number(patient: Patient) -> int:
    if not patient.consultations:
        return 1
    return max(c.consultation_number for c in patient.consultations) + 1


def _encode_classification(
    scores: dict[str, float] | None, rationale: dict[str, list[str]] | None
) -> str | None:
    if not scores:
        return None
    return json.dumps({"scores": scores, "rationale": rationale or {}}, ensure_ascii=False)


def create_consultation(
    session: Session,
    patient: Patient,
    answers: dict[str, Any],
    questionnaire_texts: dict[str, str],
    phenotype_model: str | None,
    scores: dict[str, float] | None,
    rationale: dict[str, list[str]] | None = None,
) -> Consultation:
    """Crea una consulta con sus respuestas de cuestionario."""
    number = next_consultation_number(patient)
    consultation = Consultation(
        patient_id=patient.id,
        consultation_number=number,
        consultation_type="Ingreso" if number == 1 else "Control",
        phenotype_model=phenotype_model,
        phenotype_final=phenotype_model,
        classification_source="modelo",
        classification_score=_encode_classification(scores, rationale),
    )
    session.add(consultation)
    session.flush()

    for key, value in answers.items():
        if key in ("sexo", "fecha_nacimiento"):
            continue  # se guardan en el paciente, no como respuesta
        consultation.answers.append(
            SurveyAnswer(
                question_key=key,
                question_text=questionnaire_texts.get(key, key),
                answer_value=_serialize(value),
            )
        )
    session.flush()
    return consultation


def update_classification_score(
    session: Session,
    consultation: Consultation,
    scores: dict[str, float] | None,
    rationale: dict[str, list[str]] | None,
) -> None:
    consultation.classification_score = _encode_classification(scores, rationale)


def _serialize(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    if isinstance(value, bool):
        return "Sí" if value else "No"
    return str(value)


# --- Datos del médico -------------------------------------------------------
def upsert_measurement(
    session: Session, consultation: Consultation, **fields: Any
) -> Measurement:
    m = consultation.measurement or Measurement(consultation_id=consultation.id)
    for k, v in fields.items():
        setattr(m, k, v)
    if consultation.measurement is None:
        session.add(m)
        consultation.measurement = m
    return m


def upsert_survey_answer(
    session: Session, consultation: Consultation, key: str, text: str, value: Any,
) -> None:
    existing = next((a for a in consultation.answers if a.question_key == key), None)
    serialized = _serialize(value)
    if existing:
        existing.answer_value = serialized
    else:
        consultation.answers.append(
            SurveyAnswer(question_key=key, question_text=text, answer_value=serialized)
        )


def set_lab_result(
    session: Session, consultation: Consultation, test_key: str, test_label: str,
    value: float | None, unit: str | None,
) -> None:
    existing = next((l for l in consultation.labs if l.test_key == test_key), None)
    if existing:
        existing.value = value
        existing.unit = unit
    else:
        consultation.labs.append(
            LabResult(test_key=test_key, test_label=test_label, value=value, unit=unit)
        )


def reclassify(
    session: Session, consultation: Consultation, phenotype: str, doctor: str,
    notes: str | None = None,
) -> None:
    consultation.phenotype_final = phenotype
    consultation.classification_source = "medico"
    consultation.reclassified_by = doctor
    if notes:
        consultation.notes = notes
