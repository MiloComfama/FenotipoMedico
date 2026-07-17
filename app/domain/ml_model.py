"""Clasificador entrenado con el histórico real (``data/model.pkl``).

Ver ``scripts/train_model.py`` para el proceso de entrenamiento completo y
``databricks/`` para el EDA y el perfilamiento de los clusters. Implementa
la misma interfaz ``Classifier`` que ``RuleBasedClassifier``, por lo que
``get_classifier()`` puede intercambiarlos sin tocar el resto de la app.

Resumen del enfoque (ver docstring de ``scripts/train_model.py`` para más
detalle): un único K-Means (k=5) sobre las 11 features clínicas + síntomas
digestivos asigna cada consulta al fenotipo de su centroide más cercano
("Obesidad", "Dislipidemia", "Glicemia", "Digestivo" o "Bajo riesgo"). La
afinidad de un paciente a cada fenotipo es su percentil de cercanía al
centroide de ese cluster, respecto al histórico de entrenamiento — el
fenotipo final es el de mayor afinidad.
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
        # Distancia RMS (media al cuadrado por dimensión) para poder comparar
        # con vectores parciales cuando falten mediciones clínicas.
        n_features = len(self._bundle["feature_names"])
        self._sorted_dist = {
            phenotype: np.sort(dist) / np.sqrt(n_features)
            for phenotype, dist in self._bundle["dist_train"].items()
        }

    def predict(self, features: dict[str, Any]) -> ClassificationResult:
        raw_row = build_feature_row(features)
        medians = self._bundle["medians"]
        feature_names = self._bundle["feature_names"]

        imputed = {k: (raw_row.get(k) if raw_row.get(k) is not None else medians[k]) for k in feature_names}
        vector = pd.DataFrame([imputed], columns=feature_names)
        scaled = self._bundle["scaler"].transform(vector)[0]
        scaled_by_name = dict(zip(feature_names, scaled))

        available = [f for f in feature_names if raw_row.get(f) is not None]
        km = self._bundle["kmeans"]
        cluster_to_phenotype = self._bundle["cluster_to_phenotype"]

        scores = {}
        for cluster_idx, phenotype in cluster_to_phenotype.items():
            scores[phenotype] = self._axis_affinity(
                scaled_by_name, feature_names, available, km, cluster_idx,
                self._sorted_dist[phenotype],
            )
        phenotype = max(scores, key=scores.get)

        rationale = self._build_rationale(raw_row)
        return ClassificationResult(phenotype=phenotype, scores=scores, rationale=rationale)

    @staticmethod
    def _axis_affinity(
        scaled_by_name: dict[str, float],
        all_features: list[str],
        available_features: list[str],
        km,
        cluster_idx: int,
        sorted_train_distances: np.ndarray,
    ) -> float:
        """Afinidad (0-1) al centroide de un fenotipo, calculada solo con las
        dimensiones realmente observadas (distancia RMS). Si no hay NINGÚN
        dato disponible, devuelve un puntaje neutral (0.5) en vez de apoyarse
        en valores imputados, que distorsionarían el resultado."""
        if not available_features:
            return 0.5
        idx = [all_features.index(f) for f in available_features]
        vec = np.array([scaled_by_name[f] for f in available_features])
        centroid = km.cluster_centers_[cluster_idx][idx]
        rms = float(np.sqrt(np.mean((vec - centroid) ** 2)))
        return TrainedClassifier._affinity_percentile(rms, sorted_train_distances)

    @staticmethod
    def _affinity_percentile(distance: float, sorted_train_distances: np.ndarray) -> float:
        """Fracción del histórico de entrenamiento MÁS LEJOS del centroide de
        este fenotipo que este paciente (más lejos => este paciente tiene
        mayor afinidad relativa a ese fenotipo)."""
        n = len(sorted_train_distances)
        count_closer_or_equal = int(np.searchsorted(sorted_train_distances, distance, side="right"))
        return round(1 - count_closer_or_equal / n, 4)

    def _build_rationale(self, raw_row: dict[str, float | None]) -> dict[str, list[str]]:
        rationale: dict[str, list[str]] = {
            p: [] for p in self._bundle["cluster_to_phenotype"].values()
        }

        imc = raw_row.get("imc")
        if imc is not None and imc >= 30:
            rationale["Obesidad"].append(f"Tu IMC ({imc:.1f}) está en un rango que conviene vigilar.")
        elif imc is not None and imc >= 25:
            rationale["Obesidad"].append(f"Tu IMC ({imc:.1f}) está un poco por encima de lo recomendado.")

        waist = raw_row.get("perimetro_abdominal")
        if waist is not None and waist >= 90:
            rationale["Obesidad"].append(
                f"Tu perímetro abdominal ({waist:.0f} cm) está por encima del rango recomendado."
            )

        sbp, dbp = raw_row.get("pa_sistolica"), raw_row.get("pa_diastolica")
        if (sbp is not None and sbp >= 130) or (dbp is not None and dbp >= 85):
            rationale["Obesidad"].append("Tu presión arterial registrada está por encima del rango recomendado.")

        chol = raw_row.get("colesterol_total")
        if chol is not None and chol >= 200:
            rationale["Dislipidemia"].append(f"Tu colesterol total ({chol:.0f} mg/dL) está por encima del rango recomendado.")
        hdl = raw_row.get("colesterol_hdl")
        if hdl is not None and hdl < 40:
            rationale["Dislipidemia"].append(f"Tu colesterol HDL ({hdl:.0f} mg/dL) está por debajo del rango recomendado.")
        if raw_row.get("usa_hipolipemiante"):
            rationale["Dislipidemia"].append("Nos contaste que usas medicamentos para el colesterol.")

        hba1c = raw_row.get("hba1c")
        if hba1c is not None and hba1c >= 6.5:
            rationale["Glicemia"].append(f"Tu hemoglobina glicosilada ({hba1c:.1f}%) está en rango compatible con diabetes.")
        elif hba1c is not None and hba1c >= 5.7:
            rationale["Glicemia"].append(f"Tu hemoglobina glicosilada ({hba1c:.1f}%) está en rango de prediabetes.")
        if raw_row.get("usa_hipoglicemiante"):
            rationale["Glicemia"].append("Nos contaste que usas medicamentos para el azúcar en la sangre.")

        n_gi = raw_row.get("n_sintomas_gi") or 0
        if n_gi > 0:
            palabra = "síntoma" if n_gi == 1 else "síntomas"
            rationale["Digestivo"].append(f"Contaste {int(n_gi)} {palabra} digestivo(s) en tu encuesta.")
        if raw_row.get("usa_antiacido_ibp"):
            rationale["Digestivo"].append("Nos contaste que usas antiácidos o medicamentos para la acidez.")

        if raw_row.get("estres_alto_bin") and not raw_row.get("tecnica_estres_bin"):
            rationale["Digestivo"].append("Tienes altas cargas de estrés y no usas una técnica para manejarlo; esto también puede afectar tu digestión.")

        for phenotype, reasons in rationale.items():
            if not reasons and phenotype != "Bajo riesgo":
                reasons.append("Todavía no tenemos suficientes datos que apunten claramente a este fenotipo.")
        if not any(rationale[p] for p in rationale if p != "Bajo riesgo"):
            rationale["Bajo riesgo"].append(
                "Tus mediciones y respuestas no muestran marcadores elevados en ningún eje clínico."
            )
        else:
            rationale["Bajo riesgo"].append(
                "Este fenotipo agrupa a quienes no presentan marcadores elevados en ningún eje clínico."
            )
        return rationale
