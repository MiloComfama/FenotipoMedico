"""Configuración central del prototipo: rutas, paleta de marca y ajustes."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Rutas ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "app" / "assets"

DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "fenotipo.db"
QUESTIONNAIRE_PATH = DATA_DIR / "questionnaire.yaml"
MODEL_PATH = DATA_DIR / "model.pkl"  # ranura para el futuro modelo entrenado

# --- Variables de entorno ---------------------------------------------------
load_dotenv(BASE_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8").strip()

# --- Reglas de negocio ------------------------------------------------------
# Tiempo mínimo entre consultas (cita de seguimiento).
FOLLOW_UP_DAYS = 30

# --- Marca Comfama ----------------------------------------------------------
# Paleta basada en la identidad de marca de Comfama (magenta como color primario)
# más colores de apoyo neutros y por fenotipo clínico.
BRAND = {
    "primary": "#FF277E",       # Magenta Comfama
    "primary_dark": "#C71760",
    "ink": "#2A2A33",           # Texto principal
    "muted": "#6B6B76",         # Texto secundario
    "bg": "#FFFFFF",
    "surface": "#F7F5F8",       # Fondo de tarjetas
    "surface_alt": "#FBEAF2",   # Fondo tenue magenta
    "border": "#E6E1EA",
    "success": "#1FA97B",
    "warning": "#E9A23B",
    "danger": "#E0334C",
}

# Colores por fenotipo clínico (para gráficas y badges).
PHENOTYPE_COLORS = {
    "Cardiometabólico": "#E0334C",
    "Digestivo": "#1FA97B",
    "Mixto": "#7B4F9E",
}

PHENOTYPES = list(PHENOTYPE_COLORS.keys())
