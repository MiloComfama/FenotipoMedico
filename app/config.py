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
# Nombre del recurso de Azure AI Foundry (solo si el modelo se consume vía Azure
# en lugar de la API directa de Anthropic). Ej: "mi-recurso-foundry".
ANTHROPIC_FOUNDRY_RESOURCE = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE", "").strip()

# --- Microsoft Fabric: exámenes médicos (ayudas diagnósticas) ---------------
# Servidor y base de datos no son secretos (son parte de la infraestructura del
# workspace). La autenticación es Azure AD interactiva (ver app/db/fabric.py):
# la cuenta del workspace tiene MFA/acceso condicional, que bloquea el flujo
# usuario+contraseña directo, así que no hay contraseña que guardar aquí.
FABRIC_SQL_SERVER = "jjwf2sltqteerjqzaniw7wlnr4-sv3xtungb5sedk5j5ptldf3o5m.datawarehouse.fabric.microsoft.com"
FABRIC_SQL_DATABASE = "LH_FabricData"
FABRIC_SQL_SCHEMA = "Hackaton2026"
FABRIC_SQL_TABLE = "ResultadosAyudasDiagnosticas"
FABRIC_SQL_USER = os.getenv("FABRIC_SQL_USER", "").strip()

# --- ElevenLabs: texto a voz (accesibilidad) --------------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()

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
# 5 fenotipos derivados del clustering sobre el histórico real (ver
# databricks/03_entrenamiento_clustering.ipynb): el antiguo cluster único
# "Cardiometabólico" se separa en tres perfiles con vías de intervención
# distintas (Obesidad, Dislipidemia, Glicemia); "Digestivo" y "Bajo riesgo"
# se mantienen como clusters propios.
PHENOTYPE_COLORS = {
    "Obesidad": "#E0334C",
    "Dislipidemia": "#E9A23B",
    "Glicemia": "#7B4F9E",
    "Digestivo": "#1FA97B",
    "Bajo riesgo": "#3B6FB0",
}

PHENOTYPES = list(PHENOTYPE_COLORS.keys())
