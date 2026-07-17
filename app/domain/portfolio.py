"""Recomendaciones de servicios del portafolio Comfama por fenotipo.

Se cargan desde ``data/recomendacion_clusters_portafolio.xlsx`` (curado por
el equipo de negocio a partir de los 5 clusters del modelo — ver
``databricks/`` y ``app/config.PHENOTYPES``). Complementan, sin reemplazar,
las recomendaciones clínicas de ``app/domain/recommendations.py``.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import pandas as pd

from app.config import DATA_DIR

PORTFOLIO_PATH = DATA_DIR / "recomendacion_clusters_portafolio.xlsx"

_ITEM_SPLIT = re.compile(r"\(\d+\)\s*")


@dataclass
class PortfolioRecommendation:
    phenotype: str
    intro: str
    programs: list[tuple[str, str]]  # (nombre_programa, descripción)


def _parse(raw: str) -> tuple[str, list[tuple[str, str]]]:
    intro, _, tail = raw.partition("Se recomienda:")
    intro = intro.strip()
    programs: list[tuple[str, str]] = []
    for part in _ITEM_SPLIT.split(tail):
        part = part.strip().rstrip(";.").strip()
        if not part:
            continue
        name, _, desc = part.partition("—")
        programs.append((name.strip(), desc.strip()))
    return intro, programs


@functools.lru_cache(maxsize=1)
def _load() -> dict[str, PortfolioRecommendation]:
    if not PORTFOLIO_PATH.exists():
        return {}
    df = pd.read_excel(PORTFOLIO_PATH, sheet_name="Recomendaciones")
    result: dict[str, PortfolioRecommendation] = {}
    for _, row in df.iterrows():
        phenotype = str(row["Cluster"]).strip()
        intro, programs = _parse(str(row["Programas recomendados (portafolio Comfama)"]))
        result[phenotype] = PortfolioRecommendation(phenotype=phenotype, intro=intro, programs=programs)
    return result


def get_portfolio_recommendation(phenotype: str | None) -> PortfolioRecommendation | None:
    if not phenotype:
        return None
    return _load().get(phenotype)
