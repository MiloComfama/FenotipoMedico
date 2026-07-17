"""Clasificador de fenotipo clínico.

PROTOTIPO: implementación basada en reglas transparentes. Está detrás de la
interfaz ``Classifier`` para que el modelo entrenado con el histórico real
(``app.domain.ml_model.TrainedClassifier``) pueda reemplazarla sin tocar la
interfaz de usuario ni la base de datos — se usa solo como respaldo si
``data/model.pkl`` no existe o no se puede cargar (ver ``get_classifier``).

Salida: fenotipo en {"Obesidad", "Dislipidemia", "Glicemia", "Digestivo",
"Bajo riesgo"} + puntajes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.config import MODEL_PATH


@dataclass
class ClassificationResult:
    phenotype: str
    scores: dict[str, float]  # puntaje 0-1 por eje
    rationale: dict[str, list[str]]  # explicación legible, en lenguaje sencillo, por eje


class Classifier(Protocol):
    def predict(self, features: dict[str, Any]) -> ClassificationResult: ...


# --- Utilidades de extracción de señales ------------------------------------
def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [p.strip() for p in str(value).split(";") if p.strip()]


def _is_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"sí", "si", "true", "yes", "1"}


class RuleBasedClassifier:
    """Clasificador de referencia basado en señales clínicas conocidas."""

    LOW_RISK_THRESHOLD = 0.2  # por debajo de esto en las 4 señales, se asigna "Bajo riesgo"

    def predict(self, features: dict[str, Any]) -> ClassificationResult:
        obesidad, dislipidemia, glicemia, digest = 0.0, 0.0, 0.0, 0.0
        rationale: dict[str, list[str]] = {
            "Obesidad": [], "Dislipidemia": [], "Glicemia": [], "Digestivo": [], "Bajo riesgo": [],
        }

        # --- Digestivo: síntomas gastrointestinales -------------------------
        gi = [s for s in _as_list(features.get("sintomas_digestivos")) if s != "Ninguno"]
        if gi:
            digest += min(len(gi) / 6.0, 1.0) * 0.6
            palabra = "síntoma" if len(gi) == 1 else "síntomas"
            rationale["Digestivo"].append(
                f"Contaste {len(gi)} {palabra} digestivo(s) en tu encuesta (ej. hinchazón, "
                "acidez, dolor abdominal)."
            )

        meds = _as_list(features.get("medicamentos")) + _as_list(features.get("medicamentos_actuales"))
        if any("bomba de protones" in m.lower() or "antiácid" in m.lower() for m in meds):
            digest += 0.25
            rationale["Digestivo"].append(
                "Nos contaste que usas antiácidos o medicamentos para la acidez."
            )

        if _is_yes(features.get("persona_ansiosa")) or _is_yes(features.get("persona_nerviosa")):
            digest += 0.15
            rationale["Digestivo"].append(
                "Dijiste que te sientes ansioso/a o nervioso/a con frecuencia; esto puede "
                "influir en tu digestión."
            )

        # --- Obesidad: antropometría y presión arterial ----------------------
        bmi = _to_float(features.get("bmi"))
        if bmi is not None and bmi >= 30:
            obesidad += 0.5
            rationale["Obesidad"].append(
                f"Tu peso en relación con tu estatura (IMC {bmi:.1f}) está en un rango que "
                "conviene vigilar."
            )
        elif bmi is not None and bmi >= 25:
            obesidad += 0.25
            rationale["Obesidad"].append(
                f"Tu peso en relación con tu estatura (IMC {bmi:.1f}) está un poco por encima "
                "de lo recomendado."
            )

        waist = _to_float(features.get("waist_cm"))
        sex = str(features.get("sexo", "")).lower()
        waist_limit = 80 if sex.startswith("f") else 90
        if waist is not None and waist >= waist_limit:
            obesidad += 0.3
            rationale["Obesidad"].append(
                f"Tu perímetro abdominal ({waist:.0f} cm) está por encima del rango recomendado."
            )

        sbp = _to_float(features.get("bp_systolic"))
        dbp = _to_float(features.get("bp_diastolic"))
        if (sbp and sbp >= 130) or (dbp and dbp >= 85):
            obesidad += 0.2
            rationale["Obesidad"].append(
                "Tu presión arterial registrada está por encima del rango recomendado."
            )

        # --- Dislipidemia: colesterol e hipolipemiantes -----------------------
        chol = _to_float(features.get("colesterol_total"))
        if chol is not None and chol >= 200:
            dislipidemia += 0.5
            rationale["Dislipidemia"].append(
                f"Tu colesterol total ({chol:.0f} mg/dL) está por encima del rango recomendado."
            )
        if any("hipolipemiante" in m.lower() for m in meds):
            dislipidemia += 0.5
            rationale["Dislipidemia"].append("Nos contaste que usas medicamentos para el colesterol.")

        # --- Glicemia: HbA1c e hipoglicemiantes -------------------------------
        hba1c = _to_float(features.get("hba1c"))
        if hba1c is not None and hba1c >= 6.5:
            glicemia += 0.5
            rationale["Glicemia"].append(
                f"Tu hemoglobina glicosilada ({hba1c:.1f}%) está en rango compatible con diabetes."
            )
        elif hba1c is not None and hba1c >= 5.7:
            glicemia += 0.25
            rationale["Glicemia"].append(
                f"Tu hemoglobina glicosilada ({hba1c:.1f}%) está en rango de prediabetes."
            )
        if any("hipoglicemiante" in m.lower() for m in meds):
            glicemia += 0.5
            rationale["Glicemia"].append("Nos contaste que usas medicamentos para el azúcar en la sangre.")

        if _is_yes(features.get("altas_cargas_estres")) and not _is_yes(
            features.get("tecnica_gestion_estres")
        ):
            digest += 0.1
            rationale["Digestivo"].append(
                "El estrés que reportaste, sin una técnica de manejo, también puede afectar "
                "tu digestión."
            )

        obesidad = min(obesidad, 1.0)
        dislipidemia = min(dislipidemia, 1.0)
        glicemia = min(glicemia, 1.0)
        digest = min(digest, 1.0)

        phenotype = self._decide(obesidad, dislipidemia, glicemia, digest)
        scores = {
            "Obesidad": round(obesidad, 3),
            "Dislipidemia": round(dislipidemia, 3),
            "Glicemia": round(glicemia, 3),
            "Digestivo": round(digest, 3),
            "Bajo riesgo": round(max(0.0, 1 - max(obesidad, dislipidemia, glicemia, digest)), 3),
        }
        for axis, reasons in rationale.items():
            if not reasons and axis != "Bajo riesgo":
                reasons.append(
                    "Todavía no tenemos suficientes respuestas de tu encuesta que apunten a "
                    "este fenotipo."
                )
        if phenotype == "Bajo riesgo":
            rationale["Bajo riesgo"].append(
                "Tus respuestas y mediciones no muestran marcadores elevados en ningún eje clínico."
            )
        return ClassificationResult(phenotype=phenotype, scores=scores, rationale=rationale)

    def _decide(self, obesidad: float, dislipidemia: float, glicemia: float, digest: float) -> str:
        candidates = {
            "Obesidad": obesidad,
            "Dislipidemia": dislipidemia,
            "Glicemia": glicemia,
            "Digestivo": digest,
        }
        best = max(candidates, key=candidates.get)
        if candidates[best] < self.LOW_RISK_THRESHOLD:
            return "Bajo riesgo"
        return best


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_classifier() -> Classifier:
    """Devuelve el clasificador activo.

    Si existe un modelo entrenado en ``MODEL_PATH`` se cargará aquí (pendiente);
    por ahora se usa el clasificador basado en reglas.
    """
    if MODEL_PATH.exists():
        try:
            from app.domain.ml_model import TrainedClassifier  # pragma: no cover

            return TrainedClassifier(MODEL_PATH)
        except Exception:
            pass
    return RuleBasedClassifier()
