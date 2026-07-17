"""Crea pacientes de demostración para probar el prototipo.

Uso:  py scripts/seed_demo.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import repository as repo  # noqa: E402
from app.db.database import get_session, init_db  # noqa: E402
from app.services import intake as intake_service  # noqa: E402


DEMOS = [
    {
        "doc": ("CC", "1001"),
        "name": "María Digestiva (demo)",
        "sex": "Femenino",
        "answers": {
            "sexo": "Femenino",
            "sintomas_digestivos": ["Reflujo gastroesofágico", "Gases", "Sensación de agriera", "Distensión abdominal"],
            "medicamentos": ["Inhibidores de bomba de protones / antiácidos (omeprazol, etc.)"],
            "persona_ansiosa": True,
            "altas_cargas_estres": True,
            "tecnica_gestion_estres": False,
            "realiza_actividad": False,
            "habitos_alimentacion": ["Incorporar más vegetales"],
        },
    },
    {
        "doc": ("CC", "1002"),
        "name": "Carlos Glicemia (demo)",
        "sex": "Masculino",
        "answers": {
            "sexo": "Masculino",
            "sintomas_digestivos": ["Ninguno"],
            "medicamentos": ["Hipolipemiantes (para el colesterol)", "Hipoglicemiantes (para el azúcar)"],
            "altas_cargas_estres": True,
            "tecnica_gestion_estres": False,
            "realiza_actividad": True,
            "tipo_actividad": "Caminata",
            "duracion_actividad": "30 a 60 min",
            "frecuencia_actividad": "3 a 4 días",
            "habitos_alimentacion": ["Ajustar porciones de carbohidratos"],
        },
        "measurement": {"weight_kg": 96, "height_m": 1.72, "bmi": 32.4, "waist_cm": 104, "bp_systolic": 138, "bp_diastolic": 88},
        "labs": {"colesterol_total": 240.0, "ldl": 160.0, "trigliceridos": 210.0},
    },
]


def run() -> None:
    init_db()
    for demo in DEMOS:
        doc_type, doc_number = demo["doc"]
        res = intake_service.save_new_consultation(
            doc_type, doc_number,
            {"full_name": demo["name"], "sex": demo["sex"]},
            demo["answers"],
        )
        print(f"[{doc_type} {doc_number}] {demo['name']} -> {res['phenotype']} (consulta #{res['consultation_number']})")

        # Datos clínicos y labs (opcional) + reclasificación con el modelo.
        if demo.get("measurement") or demo.get("labs"):
            patient = repo.load_patient_full(doc_type, doc_number)
            cid = patient.consultations[-1].id
            with get_session() as s:
                from app.db.models import Consultation
                c = s.get(Consultation, cid)
                if demo.get("measurement"):
                    repo.upsert_measurement(s, c, **demo["measurement"])
                for k, v in (demo.get("labs") or {}).items():
                    repo.set_lab_result(s, c, k, k, v, "mg/dL")
            intake_service.recompute_classification(cid)

    print("\nListo. Pacientes de demostración creados.")
    print("PIN de la consola médica: 'comfama'")


if __name__ == "__main__":
    run()
