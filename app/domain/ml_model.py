"""Clasificador entrenado con el histórico real (``data/model.pkl``).

Ver ``scripts/train_model.py`` para el proceso de entrenamiento completo y
``databricks/`` para el EDA y el perfilamiento de los clusters. Implementa
la misma interfaz ``Classifier`` que ``RuleBasedClassifier``, por lo que
``get_classifier()`` puede intercambiarlos sin tocar el resto de la app.

Resumen del enfoque (ver docstring de ``scripts/train_model.py`` para más
detalle): dos K-Means (k=2) — uno con features cardiometabólicas, otro con
features digestivas — definen un centroide "de riesgo" por eje. La afinidad
de un paciente a cada eje es su percentil de cercanía a ese centroide
respecto al histórico de entrenamiento. Si ambas afinidades caen en el
cuartil superior, el fenotipo es "Mixto"; si no, gana el eje con mayor
percentil.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.domain.classifier import ClassificationResult
from app.domain.ml_features import build_feature_row


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class TrainedClassifier:
    def __init__(self, model_path: Path):
        with open(model_path, "rb") as fh:
            self._bundle = dict(pickle.load(fh))
        # Se guardan en escala RMS (distancia media al cuadrado por dimensión)
        # para poder comparar con vectores parciales cuando falten mediciones
        # clínicas (ver `_axis_affinity`).
        n_cardio = len(self._bundle["cardio_features"])
        n_digest = len(self._bundle["digest_features"])
        self._sorted_dist_cardio = np.sort(self._bundle["dist_cardio_train"]) / np.sqrt(n_cardio)
        self._sorted_dist_digest = np.sort(self._bundle["dist_digest_train"]) / np.sqrt(n_digest)

    def predict(self, features: dict[str, Any]) -> ClassificationResult:
        raw_row = build_feature_row(features)
        medians = self._bundle["medians"]
        feature_names = self._bundle["feature_names"]

        imputed = {k: (raw_row.get(k) if raw_row.get(k) is not None else medians[k]) for k in feature_names}
        vector = pd.DataFrame([imputed], columns=feature_names)
        scaled = self._bundle["scaler"].transform(vector)[0]
        scaled_by_name = dict(zip(feature_names, scaled))

        cardio_features = self._bundle["cardio_features"]
        digest_features = self._bundle["digest_features"]
        # Solo las 7 mediciones/laboratorios continuos pueden faltar; los
        # conteos y banderas binarias siempre tienen un valor (0 si no aplica).
        cardio_available = [f for f in cardio_features if raw_row.get(f) is not None]
        digest_available = [f for f in digest_features if raw_row.get(f) is not None]

        km_cardio = self._bundle["km_cardio"]
        km_digest = self._bundle["km_digest"]

        pct_cardio = self._axis_affinity(
            scaled_by_name, cardio_features, cardio_available,
            km_cardio, self._bundle["hi_cardio_cluster"], self._sorted_dist_cardio,
        )
        pct_digest = self._axis_affinity(
            scaled_by_name, digest_features, digest_available,
            km_digest, self._bundle["hi_digest_cluster"], self._sorted_dist_digest,
        )

        threshold = self._bundle["mixto_percentile"]
        if pct_cardio >= threshold and pct_digest >= threshold:
            phenotype = "Mixto"
        elif pct_cardio >= pct_digest:
            phenotype = "Cardiometabólico"
        else:
            phenotype = "Digestivo"

        scores = {
            "Cardiometabólico": round(pct_cardio, 3),
            "Digestivo": round(pct_digest, 3),
        }
        rationale = self._build_rationale(features, raw_row)
        return ClassificationResult(phenotype=phenotype, scores=scores, rationale=rationale)

    @staticmethod
    def _axis_affinity(
        scaled_by_name: dict[str, float],
        axis_features: list[str],
        available_features: list[str],
        km,
        hi_cluster: int,
        sorted_train_distances: np.ndarray,
    ) -> float:
        """Afinidad (0-1) al centroide "de riesgo" de un eje, calculada solo
        con las dimensiones realmente observadas (distancia RMS). Si no hay
        NINGÚN dato disponible para el eje, devuelve un puntaje neutral
        (0.5) en vez de apoyarse en valores imputados, que distorsionarían
        el resultado."""
        if not available_features:
            return 0.5
        idx = [axis_features.index(f) for f in available_features]
        vec = np.array([scaled_by_name[f] for f in available_features])
        centroid = km.cluster_centers_[hi_cluster][idx]
        rms = float(np.sqrt(np.mean((vec - centroid) ** 2)))
        return TrainedClassifier._affinity_percentile(rms, sorted_train_distances)

    @staticmethod
    def _affinity_percentile(distance: float, sorted_train_distances: np.ndarray) -> float:
        """Fracción del histórico de entrenamiento MÁS LEJOS del centroide de
        riesgo que este paciente (más lejos => este paciente tiene mayor
        afinidad relativa al eje)."""
        n = len(sorted_train_distances)
        count_closer_or_equal = int(np.searchsorted(sorted_train_distances, distance, side="right"))
        return round(1 - count_closer_or_equal / n, 4)

    def _build_rationale(
        self, features: dict[str, Any], raw_row: dict[str, float | None]
    ) -> dict[str, list[str]]:
        rationale: dict[str, list[str]] = {"Cardiometabólico": [], "Digestivo": []}
        sex = str(features.get("sexo", "")).lower()

        imc = raw_row.get("imc")
        if imc is not None and imc >= 30:
            rationale["Cardiometabólico"].append(f"Tu IMC ({imc:.1f}) está en un rango que conviene vigilar.")
        elif imc is not None and imc >= 25:
            rationale["Cardiometabólico"].append(f"Tu IMC ({imc:.1f}) está un poco por encima de lo recomendado.")

        waist = raw_row.get("perimetro_abdominal")
        if waist is not None:
            waist_limit = 80 if sex.startswith("f") else 90
            if waist >= waist_limit:
                rationale["Cardiometabólico"].append(
                    f"Tu perímetro abdominal ({waist:.0f} cm) está por encima del rango recomendado."
                )

        sbp, dbp = raw_row.get("pa_sistolica"), raw_row.get("pa_diastolica")
        if (sbp is not None and sbp >= 130) or (dbp is not None and dbp >= 85):
            rationale["Cardiometabólico"].append("Tu presión arterial registrada está por encima del rango recomendado.")

        chol = raw_row.get("colesterol_total")
        if chol is not None and chol >= 200:
            rationale["Cardiometabólico"].append(f"Tu colesterol total ({chol:.0f} mg/dL) está por encima del rango recomendado.")

        hba1c = raw_row.get("hba1c")
        if hba1c is not None and hba1c >= 6.5:
            rationale["Cardiometabólico"].append(f"Tu hemoglobina glicosilada ({hba1c:.1f}%) está en rango compatible con diabetes.")
        elif hba1c is not None and hba1c >= 5.7:
            rationale["Cardiometabólico"].append(f"Tu hemoglobina glicosilada ({hba1c:.1f}%) está en rango de prediabetes.")

        if raw_row.get("usa_hipoglicemiante") or raw_row.get("usa_hipolipemiante"):
            rationale["Cardiometabólico"].append("Nos contaste que usas medicamentos para el colesterol o el azúcar en la sangre.")

        n_gi = raw_row.get("n_sintomas_gi") or 0
        if n_gi > 0:
            palabra = "síntoma" if n_gi == 1 else "síntomas"
            rationale["Digestivo"].append(f"Contaste {int(n_gi)} {palabra} digestivo(s) en tu encuesta.")

        if raw_row.get("usa_antiacido_ibp"):
            rationale["Digestivo"].append("Nos contaste que usas antiácidos o medicamentos para la acidez.")

        if raw_row.get("estres_alto_bin") and not raw_row.get("tecnica_estres_bin"):
            rationale["Cardiometabólico"].append("Tienes altas cargas de estrés y no usas una técnica para manejarlo.")
            rationale["Digestivo"].append("El estrés sin manejo también puede afectar tu digestión.")

        for axis, reasons in rationale.items():
            if not reasons:
                reasons.append("Todavía no tenemos suficientes datos que apunten claramente a este eje.")
        return rationale
