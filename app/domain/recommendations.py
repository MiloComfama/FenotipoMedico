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
    "Obesidad": Protocol(
        phenotype="Obesidad",
        summary=(
            "Tu perfil se orienta al manejo del peso: tu IMC y tu perímetro "
            "abdominal guían tu plan."
        ),
        focus_areas=[
            "Control de peso y composición corporal",
            "Perímetro abdominal y presión arterial",
            "Actividad física progresiva",
        ],
        lifestyle=[
            "Prioriza vegetales y proteína de calidad; reduce azúcares y repostería.",
            "Ajusta porciones de carbohidratos según tu actividad diaria.",
            "Actividad física aeróbica 150 min/semana + fuerza 2 veces/semana.",
            "Registra tu progreso de peso y perímetro abdominal en cada control.",
        ],
        monitoring=[
            "Peso, IMC y perímetro abdominal en cada control.",
            "Presión arterial.",
            "Perfil lipídico y glicemia según indicación médica.",
        ],
    ),
    "Dislipidemia": Protocol(
        phenotype="Dislipidemia",
        summary=(
            "Tu perfil se orienta al colesterol: tus lípidos en sangre guían tu "
            "plan de alimentación y seguimiento."
        ),
        focus_areas=[
            "Colesterol total y HDL",
            "Calidad de las grasas de la dieta",
            "Adherencia al tratamiento hipolipemiante (si aplica)",
        ],
        lifestyle=[
            "Prefiere grasas saludables (aceite de oliva, aguacate, frutos secos) sobre grasas saturadas.",
            "Aumenta el consumo de fibra (avena, leguminosas, vegetales).",
            "Reduce ultraprocesados, frituras y embutidos.",
            "Actividad física aeróbica regular; ayuda a mejorar el HDL.",
        ],
        monitoring=[
            "Perfil lipídico (colesterol total, HDL) según indicación médica.",
            "Adherencia a hipolipemiantes si te fueron prescritos.",
            "Peso y perímetro abdominal.",
        ],
    ),
    "Glicemia": Protocol(
        phenotype="Glicemia",
        summary=(
            "Tu perfil se orienta al control de la glicemia: tu hemoglobina "
            "glicosilada guía tu plan de alimentación y seguimiento."
        ),
        focus_areas=[
            "Control de glicemia y hemoglobina glicosilada (HbA1c)",
            "Manejo de carbohidratos en la dieta",
            "Adherencia al tratamiento hipoglicemiante (si aplica)",
        ],
        lifestyle=[
            "Ajusta las porciones y el tipo de carbohidrato (prefiere integrales y de bajo índice glicémico).",
            "Distribuye las comidas en horarios regulares; evita ayunos prolongados.",
            "Actividad física regular; ayuda al control de la glicemia.",
            "Registra síntomas de hipo/hiperglicemia si los presentas.",
        ],
        monitoring=[
            "Hemoglobina glicosilada (HbA1c) según indicación médica.",
            "Adherencia a hipoglicemiantes si te fueron prescritos.",
            "Peso e IMC.",
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
    "Bajo riesgo": Protocol(
        phenotype="Bajo riesgo",
        summary=(
            "Tus mediciones y respuestas no muestran marcadores clínicos elevados: "
            "tu plan se enfoca en mantener tus buenos hábitos y prevenir riesgos futuros."
        ),
        focus_areas=[
            "Mantenimiento de hábitos saludables",
            "Prevención cardiometabólica y digestiva",
            "Seguimiento periódico",
        ],
        lifestyle=[
            "Mantén una alimentación variada, rica en vegetales y proteína de calidad.",
            "Actividad física regular (150 min/semana de cardio + fuerza 2 veces/semana).",
            "Técnicas de manejo del estrés y buen descanso.",
            "No abandones tus controles periódicos aunque te sientas bien.",
        ],
        monitoring=[
            "Peso, IMC y perímetro abdominal en cada control.",
            "Presión arterial y perfil lipídico/glicemia según periodicidad indicada.",
            "Aparición de nuevos síntomas gastrointestinales o cardiometabólicos.",
        ],
    ),
}


def get_protocol(phenotype: str | None) -> Protocol | None:
    if not phenotype:
        return None
    return PROTOCOLS.get(phenotype)
