"""Orquestación: construir features, clasificar y persistir consultas."""
from __future__ import annotations

from typing import Any

from app.db import repository as repo
from app.db.database import get_session
from app.db.models import Consultation, Measurement, Patient
from app.domain.classifier import ClassificationResult, get_classifier
from app.domain.questionnaire import load_questionnaire


def build_features(
    patient: Patient,
    answers: dict[str, Any],
    measurement: Measurement | None = None,
    labs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Combina respuestas del cuestionario + datos del médico para el clasificador."""
    features: dict[str, Any] = dict(answers)
    features.setdefault("sexo", patient.sex)
    if measurement:
        features.update({
            "bmi": measurement.bmi,
            "waist_cm": measurement.waist_cm,
            "bp_systolic": measurement.bp_systolic,
            "bp_diastolic": measurement.bp_diastolic,
        })
    if labs:
        features.update(labs)
    return features


def classify(features: dict[str, Any]) -> ClassificationResult:
    return get_classifier().predict(features)


def save_new_consultation(
    doc_type: str,
    doc_number: str,
    patient_fields: dict[str, Any],
    answers: dict[str, Any],
) -> dict[str, Any]:
    """Crea (o reutiliza) el paciente y registra una nueva consulta clasificada."""
    questionnaire = load_questionnaire()
    texts = {k: q.text for k, q in questionnaire.all_questions().items()}

    with get_session() as session:
        patient = repo.get_or_create_patient(
            session, doc_type, doc_number,
            full_name=patient_fields.get("full_name"),
            sex=answers.get("sexo") or patient_fields.get("sex"),
            birthdate=answers.get("fecha_nacimiento") or patient_fields.get("birthdate"),
        )

        features = build_features(patient, answers)
        result = classify(features)

        consultation = repo.create_consultation(
            session, patient, answers, texts,
            phenotype_model=result.phenotype,
            scores=result.scores,
        )
        return {
            "consultation_number": consultation.consultation_number,
            "phenotype": result.phenotype,
            "scores": result.scores,
            "rationale": result.rationale,
        }


def recompute_classification(consultation_id: int) -> ClassificationResult:
    """Reclasifica con el modelo usando respuestas + medidas + labs actuales."""
    with get_session() as session:
        consultation = session.get(Consultation, consultation_id)
        answers = {a.question_key: _deserialize(a.answer_value) for a in consultation.answers}
        labs = {l.test_key: l.value for l in consultation.labs if l.value is not None}
        features = build_features(
            consultation.patient, answers, consultation.measurement, labs
        )
        result = classify(features)
        consultation.phenotype_model = result.phenotype
        # Solo actualiza el definitivo si aún no lo tocó un médico.
        if consultation.classification_source != "medico":
            consultation.phenotype_final = result.phenotype
        return result


def _deserialize(value: str | None) -> Any:
    if value is None:
        return None
    if ";" in value:
        return [v for v in value.split(";") if v]
    if value in ("Sí", "No"):
        return value == "Sí"
    return value
