"""Protocolos y recomendaciones por fenotipo clínico.

Estas recomendaciones son de referencia para el prototipo y deben validarse
con el equipo asistencial de Comfama antes de un uso clínico real.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Protocol:
    phenotype: str
    summary: str
    focus_areas: list[str]
    lifestyle: list[str]
    monitoring: list[str]
    disclaimer: str = (
        "Recomendaciones informativas del programa de medicina funcional. "
        "No reemplazan la valoración médica presencial."
    )


PROTOCOLS: dict[str, Protocol] = {
    "Cardiometabólico": Protocol(
        phenotype="Cardiometabólico",
        summary=(
            "Tu perfil se orienta al eje cardiometabólico: factores como peso, "
            "perímetro abdominal, presión arterial o lípidos guían tu plan."
        ),
        focus_areas=[
            "Control de peso y perímetro abdominal",
            "Salud cardiovascular y presión arterial",
            "Metabolismo de azúcar y lípidos",
        ],
        lifestyle=[
            "Prioriza vegetales y proteína de calidad; reduce azúcares y repostería.",
            "Ajusta porciones de carbohidratos según tu actividad diaria.",
            "Actividad física aeróbica 150 min/semana + fuerza 2 veces/semana.",
            "Técnicas de manejo del estrés (respiración, meditación) de forma regular.",
        ],
        monitoring=[
            "Perímetro abdominal e IMC en cada control.",
            "Presión arterial.",
            "Perfil lipídico y glicemia según indicación médica.",
        ],
    ),
    "Digestivo": Protocol(
        phenotype="Digestivo",
        summary=(
            "Tu perfil se orienta al eje digestivo: los síntomas gastrointestinales "
            "y hábitos relacionados guían tu plan."
        ),
        focus_areas=[
            "Salud gastrointestinal y microbiota",
            "Manejo de reflujo y síntomas dispépticos",
            "Relación entre estrés/ansiedad y digestión",
        ],
        lifestyle=[
            "Comidas en horarios regulares, masticación lenta y porciones moderadas.",
            "Identifica y modera alimentos que disparan tus síntomas.",
            "Hidratación adecuada y fibra progresiva según tolerancia.",
            "Manejo del estrés y del sueño para el eje intestino-cerebro.",
        ],
        monitoring=[
            "Frecuencia e intensidad de síntomas gastrointestinales.",
            "Uso de IBP / antiácidos.",
            "Hábito intestinal.",
        ],
    ),
    "Mixto": Protocol(
        phenotype="Mixto",
        summary=(
            "Tu perfil combina señales cardiometabólicas y digestivas: tu plan "
            "integra ambos ejes de forma coordinada."
        ),
        focus_areas=[
            "Abordaje cardiometabólico + digestivo simultáneo",
            "Alimentación antiinflamatoria",
            "Estrés como factor común a ambos ejes",
        ],
        lifestyle=[
            "Patrón de alimentación que cuide metabolismo y digestión (vegetales, proteína, menos ultraprocesados).",
            "Actividad física regular adaptada a tu tolerancia digestiva.",
            "Rutina consistente de manejo del estrés y del sueño.",
            "Registro de síntomas para ajustar el plan en cada control.",
        ],
        monitoring=[
            "Antropometría y presión arterial.",
            "Síntomas gastrointestinales.",
            "Laboratorios según indicación (lípidos, glicemia).",
        ],
    ),
}


def get_protocol(phenotype: str | None) -> Protocol | None:
    if not phenotype:
        return None
    return PROTOCOLS.get(phenotype)
