"""Diagnóstico puntual: verifica la conexión a Microsoft Fabric y muestra las
columnas y una muestra de filas de la tabla de exámenes médicos.

Uso:  py scripts/inspect_fabric_table.py

Requiere FABRIC_SQL_USER en .env (ver .env.example); pedirá login
interactivo de Azure AD la primera vez. No se usa en producción — es solo
para descubrir el esquema real de la tabla una vez.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import fabric  # noqa: E402


def main() -> None:
    if not fabric.is_configured():
        print("FABRIC_SQL_USER / FABRIC_SQL_PASSWORD no están configurados en .env.")
        return

    print("Conectando a Microsoft Fabric...")
    columns = fabric.list_columns()
    print(f"\nColumnas ({len(columns)}):")
    for c in columns:
        print(f"  - {c}")

    print("\nMuestra de 5 filas:")
    for row in fabric.fetch_sample(5):
        print(row)


if __name__ == "__main__":
    main()
