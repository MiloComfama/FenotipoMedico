"""Conexión de solo lectura al Data Warehouse de Microsoft Fabric para
consultar exámenes médicos (ayudas diagnósticas) desde la consola médica.

Autenticación: Azure AD interactiva (``ActiveDirectoryInteractive``) vía el
driver ODBC 18 para SQL Server — la cuenta de este workspace tiene MFA /
acceso condicional, que bloquea el flujo usuario+contraseña directo
(``ActiveDirectoryPassword``); la interactiva abre una ventana de login del
navegador la primera vez y reutiliza el token mientras siga vigente.

Si ``FABRIC_SQL_USER`` no está configurado, ``is_configured()`` devuelve
``False`` y la consola médica sigue funcionando con ingreso manual de
laboratorios.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.config import (
    FABRIC_SQL_DATABASE,
    FABRIC_SQL_SCHEMA,
    FABRIC_SQL_SERVER,
    FABRIC_SQL_TABLE,
    FABRIC_SQL_USER,
)

_QUALIFIED_TABLE = f"[{FABRIC_SQL_SCHEMA}].[{FABRIC_SQL_TABLE}]"

# Mapea cada laboratorio de la app (ver app/ui/doctor_view.LAB_FIELDS) a los
# patrones (prefijo, mayúsculas) que identifican ese examen en
# `descripcion_item_prueba`. Se probó primero el listado real de valores
# distintos de esa columna (hay muchas variantes de "glicemia" para pruebas
# de tolerancia oral/postprandial que NO son la glicemia en ayunas que usa
# la app, y variantes con/sin tilde para "triglicéridos").
LAB_TEST_PATTERNS: dict[str, list[str]] = {
    "colesterol_total": ["COLESTEROL TOTAL"],
    "hdl": ["COLESTEROL HDL"],
    "ldl": ["COLESTEROL LDL"],
    "trigliceridos": ["TRIGLICERIDOS", "TRIGLICÉRIDOS"],
    "glicemia": ["GLICEMIA EN AYUNAS"],
    "hba1c": ["HEMOGLOBINA GLICOSILADA (HBA1C)", "HEMOGLOBINA GLICADA (HBA1C)"],
}


def is_configured() -> bool:
    return bool(FABRIC_SQL_USER)


def _connection_string() -> str:
    return (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server={FABRIC_SQL_SERVER},1433;"
        f"Database={FABRIC_SQL_DATABASE};"
        f"UID={FABRIC_SQL_USER};"
        "Authentication=ActiveDirectoryInteractive;"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;"
    )


def get_connection():
    """Abre una conexión nueva. El llamador es responsable de cerrarla
    (usar como context manager: ``with get_connection() as conn:``)."""
    import pyodbc  # import perezoso: solo se necesita si se usa esta integración

    return pyodbc.connect(_connection_string())


def _rows_to_dicts(cursor) -> list[dict[str, Any]]:
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def list_columns() -> list[str]:
    """Lista las columnas reales de la tabla (para descubrir el esquema)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT TOP 0 * FROM {_QUALIFIED_TABLE}")
        return [c[0] for c in cursor.description]


def fetch_sample(n: int = 5) -> list[dict[str, Any]]:
    """Trae las primeras ``n`` filas tal cual, para inspección del esquema."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT TOP {int(n)} * FROM {_QUALIFIED_TABLE}")
        return _rows_to_dicts(cursor)


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _classify(test_name: str) -> str | None:
    name = (test_name or "").strip().upper()
    for lab_key, patterns in LAB_TEST_PATTERNS.items():
        if any(name.startswith(p) for p in patterns):
            return lab_key
    return None


def fetch_lab_results(doc_type: str, doc_number: str) -> dict[str, dict[str, Any]]:
    """Trae, para el paciente indicado, el resultado MÁS RECIENTE de cada
    laboratorio que usa la app (ver ``LAB_TEST_PATTERNS``).

    Devuelve ``{lab_key: {"value": float, "date": datetime, "raw_name": str,
    "interpretation": str}}`` — solo incluye los laboratorios encontrados.
    """
    like_clauses = " OR ".join(
        "descripcion_item_prueba LIKE ?" for patterns in LAB_TEST_PATTERNS.values() for _ in patterns
    )
    like_params = [
        f"{p}%" for patterns in LAB_TEST_PATTERNS.values() for p in patterns
    ]
    query = (
        "SELECT descripcion_item_prueba, valor_resultado, interpretacion_resultado, fecha_validacion "
        f"FROM {_QUALIFIED_TABLE} "
        "WHERE tipo_id_paciente = ? AND numero_id_paciente = ? "
        f"AND ({like_clauses}) "
        "ORDER BY fecha_validacion DESC"
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, doc_type, doc_number, *like_params)
        rows = _rows_to_dicts(cursor)

    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        lab_key = _classify(row["descripcion_item_prueba"])
        if lab_key is None or lab_key in results:
            continue  # ya se tiene el más reciente de este laboratorio (rows viene ordenado DESC)
        value = _to_float(row["valor_resultado"])
        if value is None:
            continue
        results[lab_key] = {
            "value": value,
            "date": row["fecha_validacion"],
            "raw_name": row["descripcion_item_prueba"],
            "interpretation": row["interpretacion_resultado"],
        }
    return results
