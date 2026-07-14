"""Clasificador de fenotipo clínico.

PROTOTIPO: implementación basada en reglas transparentes. Está detrás de la
interfaz ``Classifier`` para que el modelo de IA entrenado (sklearn / Fabric)
pueda reemplazarla sin tocar la interfaz de usuario ni la base de datos.

Salida: fenotipo en {"Cardiometabólico", "Digestivo", "Mixto"} + puntajes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.config import MODEL_PATH


@dataclass
class ClassificationResult:
    phenotype: str
    scores: dict[str, float]  # puntaje 0-1 por eje
    rationale: list[str]      # explicación legible


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

    CARDIO_SYMPTOMS_THRESHOLD = 0.45
    DIGEST_SYMPTOMS_THRESHOLD = 0.45

    def predict(self, features: dict[str, Any]) -> ClassificationResult:
        cardio, digest = 0.0, 0.0
        rationale: list[str] = []

        # --- Eje digestivo: síntomas gastrointestinales ---------------------
        gi = [s for s in _as_list(features.get("sintomas_digestivos")) if s != "Ninguno"]
        if gi:
            digest += min(len(gi) / 6.0, 1.0) * 0.6
            rationale.append(f"{len(gi)} síntoma(s) gastrointestinal(es) reportado(s).")

        meds = _as_list(features.get("medicamentos"))
        if any("bomba de protones" in m.lower() or "antiácid" in m.lower() for m in meds):
            digest += 0.25
            rationale.append("Uso de IBP / antiácidos.")

        if _is_yes(features.get("persona_ansiosa")) or _is_yes(features.get("persona_nerviosa")):
            digest += 0.15
            rationale.append("Rasgos de ansiedad / nerviosismo (asociados al eje digestivo).")

        # --- Eje cardiometabólico: antropometría, signos y medicamentos -----
        if any("hipoglicemiante" in m.lower() or "hipolipemiante" in m.lower() for m in meds):
            cardio += 0.3
            rationale.append("Uso de hipoglicemiantes / hipolipemiantes.")

        bmi = _to_float(features.get("bmi"))
        if bmi is not None and bmi >= 30:
            cardio += 0.25
            rationale.append(f"IMC en rango de obesidad ({bmi:.1f}).")
        elif bmi is not None and bmi >= 25:
            cardio += 0.12
            rationale.append(f"IMC en sobrepeso ({bmi:.1f}).")

        waist = _to_float(features.get("waist_cm"))
        sex = str(features.get("sexo", "")).lower()
        waist_limit = 80 if sex.startswith("f") else 90
        if waist is not None and waist >= waist_limit:
            cardio += 0.2
            rationale.append(f"Perímetro abdominal elevado ({waist:.0f} cm).")

        sbp = _to_float(features.get("bp_systolic"))
        dbp = _to_float(features.get("bp_diastolic"))
        if (sbp and sbp >= 130) or (dbp and dbp >= 85):
            cardio += 0.2
            rationale.append("Presión arterial elevada.")

        chol = _to_float(features.get("colesterol_total"))
        if chol is not None and chol >= 200:
            cardio += 0.2
            rationale.append(f"Colesterol total elevado ({chol:.0f} mg/dL).")

        if _is_yes(features.get("altas_cargas_estres")) and not _is_yes(
            features.get("tecnica_gestion_estres")
        ):
            cardio += 0.1
            digest += 0.1
            rationale.append("Altas cargas de estrés sin técnica de manejo.")

        cardio = min(cardio, 1.0)
        digest = min(digest, 1.0)

        phenotype = self._decide(cardio, digest)
        scores = {"Cardiometabólico": round(cardio, 3), "Digestivo": round(digest, 3)}
        if not rationale:
            rationale.append("Información insuficiente; clasificación preliminar.")
        return ClassificationResult(phenotype=phenotype, scores=scores, rationale=rationale)

    def _decide(self, cardio: float, digest: float) -> str:
        c_hi = cardio >= self.CARDIO_SYMPTOMS_THRESHOLD
        d_hi = digest >= self.DIGEST_SYMPTOMS_THRESHOLD
        if c_hi and d_hi:
            return "Mixto"
        if abs(cardio - digest) < 0.15 and (cardio > 0.25 and digest > 0.25):
            return "Mixto"
        return "Cardiometabólico" if cardio >= digest else "Digestivo"


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
