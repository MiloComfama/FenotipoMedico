"""Ingeniería de features compartida entre el entrenamiento del modelo
(``scripts/train_model.py``) y la inferencia en producción
(``app/domain/ml_model.py``).

Mantener esta lista y las funciones de mapeo en un solo lugar evita que el
entrenamiento y el servicio en vivo diverjan (train/serve skew).
"""
from __future__ import annotations

from typing import Any

FEATURE_NAMES = [
    "imc", "perimetro_abdominal", "pa_sistolica", "pa_diastolica",
    "colesterol_total", "colesterol_hdl", "hba1c",
    "usa_hipoglicemiante", "usa_hipolipemiante",
    "n_sintomas_gi", "usa_antiacido_ibp",
    "estres_alto_bin", "ansioso_bin", "nervioso_bin", "tecnica_estres_bin",
    "hace_actividad", "actividad_fuerza", "frecuencia_actividad_num",
    "n_habitos_saludables",
]

# Subconjuntos usados para calcular la afinidad a cada eje clínico.
CARDIO_FEATURES = [
    "imc", "perimetro_abdominal", "pa_sistolica", "pa_diastolica",
    "colesterol_total", "colesterol_hdl", "hba1c",
    "usa_hipoglicemiante", "usa_hipolipemiante",
]
# Nota: 'estres_alto_bin'/'tecnica_estres_bin' se probaron aquí también (están
# correlacionadas con síntomas digestivos a nivel poblacional, ver EDA en
# databricks/03_entrenamiento_clustering.ipynb), pero al ser binarias con baja
# prevalencia dominaban la distancia RMS de solo 4 dimensiones y producían
# falsos "Digestivo" en pacientes estresados sin ningún síntoma gastrointestinal.
# Se dejan fuera del eje de puntaje (sí se usan en el clustering conjunto).
DIGEST_FEATURES = ["n_sintomas_gi", "usa_antiacido_ibp"]

_FREQ_MAP_APP = {
    "1 a 2 días": 1.5,
    "3 a 4 días": 3.5,
    "5 o más días": 5.0,
}


def _is_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"sí", "si", "true", "yes", "1"}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [p.strip() for p in str(value).split(";") if p.strip()]


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)


def build_feature_row(features: dict[str, Any]) -> dict[str, float | None]:
    """Convierte el diccionario de features de la app (respuestas del
    cuestionario + mediciones + laboratorios) al vector canónico que espera
    el modelo entrenado. Los valores ausentes quedan como ``None`` — el
    modelo los imputa con las medianas de entrenamiento."""
    habitos = _as_list(features.get("habitos_alimentacion"))
    n_habitos = len([h for h in habitos if h and h.lower() != "ninguno"])

    hace_actividad = 1 if _is_yes(features.get("realiza_actividad")) else 0
    tipo_actividad = str(features.get("tipo_actividad") or "")
    actividad_fuerza = 1 if _has_any(tipo_actividad.lower(), ["fuerza", "pesa", "gimnasio"]) else 0

    frecuencia_txt = str(features.get("frecuencia_actividad") or "").strip()
    frecuencia_num = _FREQ_MAP_APP.get(frecuencia_txt, 0.0) if hace_actividad else 0.0

    # 'medicamentos' (cuestionario del paciente) + 'medicamentos_actuales'
    # (ingresados por el médico) — se combinan porque cualquiera de los dos
    # confirma el uso del medicamento.
    meds_text = " ".join(
        _as_list(features.get("medicamentos")) + _as_list(features.get("medicamentos_actuales"))
    ).lower()
    usa_hipoglicemiante = 1 if "hipoglicemiante" in meds_text else 0
    usa_hipolipemiante = 1 if "hipolipemiante" in meds_text else 0
    usa_antiacido_ibp = 1 if _has_any(meds_text, ["antiácid", "antiacid", "bomba de protones"]) else 0

    sintomas = _as_list(features.get("sintomas_digestivos"))
    n_sintomas_gi = len([s for s in sintomas if s and s.lower() != "ninguno"])

    return {
        "imc": _to_float(features.get("bmi")),
        "perimetro_abdominal": _to_float(features.get("waist_cm")),
        "pa_sistolica": _to_float(features.get("bp_systolic")),
        "pa_diastolica": _to_float(features.get("bp_diastolic")),
        "colesterol_total": _to_float(features.get("colesterol_total")),
        "colesterol_hdl": _to_float(features.get("hdl")),
        "hba1c": _to_float(features.get("hba1c")),
        "usa_hipoglicemiante": usa_hipoglicemiante,
        "usa_hipolipemiante": usa_hipolipemiante,
        "n_sintomas_gi": n_sintomas_gi,
        "usa_antiacido_ibp": usa_antiacido_ibp,
        "estres_alto_bin": 1 if _is_yes(features.get("altas_cargas_estres")) else 0,
        "ansioso_bin": 1 if _is_yes(features.get("persona_ansiosa")) else 0,
        "nervioso_bin": 1 if _is_yes(features.get("persona_nerviosa")) else 0,
        "tecnica_estres_bin": 1 if _is_yes(features.get("tecnica_gestion_estres")) else 0,
        "hace_actividad": hace_actividad,
        "actividad_fuerza": actividad_fuerza,
        "frecuencia_actividad_num": frecuencia_num,
        "n_habitos_saludables": n_habitos,
    }
