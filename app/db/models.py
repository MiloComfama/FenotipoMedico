"""Modelo de datos (SQLAlchemy ORM).

Diseño pensado para migrar a SQL Server / Fabric más adelante:

  patients          1 --- N  consultations
  consultations     1 --- N  survey_answers   (modelo llave-valor / EAV)
  consultations     1 --- 1  measurements     (datos que ingresa el médico)
  consultations     1 --- N  lab_results      (EAV: colesterol total, etc.)

La clasificación del paciente se guarda por consulta: 'phenotype_model'
(sugerido por el modelo de IA) y 'phenotype_final' (definitivo, que el médico
puede sobrescribir). Así queda trazabilidad modelo vs. criterio médico.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"
    __table_args__ = (UniqueConstraint("doc_type", "doc_number", name="uq_patient_doc"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_type: Mapped[str] = mapped_column(String(8), default="CC")
    doc_number: Mapped[str] = mapped_column(String(32), index=True)
    full_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    sex: Mapped[str | None] = mapped_column(String(24), nullable=True)
    birthdate: Mapped[str | None] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    consultations: Mapped[list["Consultation"]] = relationship(
        back_populates="patient",
        order_by="Consultation.consultation_number",
        cascade="all, delete-orphan",
    )


class Consultation(Base):
    __tablename__ = "consultations"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    consultation_number: Mapped[int] = mapped_column(Integer)  # #1, #2, #3 ...
    consultation_type: Mapped[str] = mapped_column(String(16), default="Ingreso")  # Ingreso / Control
    consultation_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    phenotype_model: Mapped[str | None] = mapped_column(String(32), nullable=True)
    phenotype_final: Mapped[str | None] = mapped_column(String(32), nullable=True)
    classification_source: Mapped[str] = mapped_column(String(16), default="modelo")  # modelo / medico
    classification_score: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON con puntajes
    reclassified_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="consultations")
    answers: Mapped[list["SurveyAnswer"]] = relationship(
        back_populates="consultation", cascade="all, delete-orphan"
    )
    measurement: Mapped["Measurement | None"] = relationship(
        back_populates="consultation", uselist=False, cascade="all, delete-orphan"
    )
    labs: Mapped[list["LabResult"]] = relationship(
        back_populates="consultation", cascade="all, delete-orphan"
    )


class SurveyAnswer(Base):
    """Respuesta del cuestionario (modelo llave-valor)."""

    __tablename__ = "survey_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    consultation_id: Mapped[int] = mapped_column(ForeignKey("consultations.id"), index=True)
    question_key: Mapped[str] = mapped_column(String(64), index=True)
    question_text: Mapped[str] = mapped_column(Text)
    answer_value: Mapped[str | None] = mapped_column(Text, nullable=True)

    consultation: Mapped[Consultation] = relationship(back_populates="answers")


class Measurement(Base):
    """Datos antropométricos y signos vitales que ingresa el médico."""

    __tablename__ = "measurements"

    id: Mapped[int] = mapped_column(primary_key=True)
    consultation_id: Mapped[int] = mapped_column(ForeignKey("consultations.id"), unique=True)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    height_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    bmi: Mapped[float | None] = mapped_column(Float, nullable=True)  # autocalculado
    waist_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    bp_systolic: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bp_diastolic: Mapped[int | None] = mapped_column(Integer, nullable=True)

    consultation: Mapped[Consultation] = relationship(back_populates="measurement")


class LabResult(Base):
    """Resultados de laboratorio (EAV): colesterol total, etc."""

    __tablename__ = "lab_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    consultation_id: Mapped[int] = mapped_column(ForeignKey("consultations.id"), index=True)
    test_key: Mapped[str] = mapped_column(String(64), index=True)
    test_label: Mapped[str] = mapped_column(String(120))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(24), nullable=True)

    consultation: Mapped[Consultation] = relationship(back_populates="labs")
