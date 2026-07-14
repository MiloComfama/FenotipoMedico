"""Carga y utilidades del cuestionario definido en data/questionnaire.yaml."""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import yaml

from app.config import QUESTIONNAIRE_PATH


@dataclass
class Question:
    key: str
    text: str
    type: str
    options: list[str] = field(default_factory=list)
    feature: bool = False
    depends_on: dict[str, Any] | None = None
    unit: str | None = None
    help: str | None = None
    min: float | None = None
    max: float | None = None
    format: str | None = None

    def is_active(self, answers: dict[str, Any]) -> bool:
        """Devuelve True si la pregunta debe mostrarse según respuestas previas."""
        if not self.depends_on:
            return True
        dep_key = self.depends_on.get("key")
        expected = self.depends_on.get("equals")
        return answers.get(dep_key) == expected


@dataclass
class Section:
    id: str
    title: str
    questions: list[Question]
    only_first_consultation: bool = False


@dataclass
class Questionnaire:
    version: int
    title: str
    intro: str
    sections: list[Section]

    def questions_for(self, is_first_consultation: bool) -> list[Question]:
        """Lista plana de preguntas aplicables a la consulta."""
        result: list[Question] = []
        for section in self.sections:
            if section.only_first_consultation and not is_first_consultation:
                continue
            result.extend(section.questions)
        return result

    def all_questions(self) -> dict[str, Question]:
        return {q.key: q for s in self.sections for q in s.questions}


@functools.lru_cache(maxsize=1)
def load_questionnaire() -> Questionnaire:
    with open(QUESTIONNAIRE_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    sections = []
    for s in raw["sections"]:
        questions = [
            Question(
                key=q["key"],
                text=q["text"],
                type=q["type"],
                options=q.get("options", []),
                feature=q.get("feature", False),
                depends_on=q.get("depends_on"),
                unit=q.get("unit"),
                help=q.get("help"),
                min=q.get("min"),
                max=q.get("max"),
                format=q.get("format"),
            )
            for q in s["questions"]
        ]
        sections.append(
            Section(
                id=s["id"],
                title=s["title"],
                questions=questions,
                only_first_consultation=s.get("only_first_consultation", False),
            )
        )

    return Questionnaire(
        version=raw.get("version", 1),
        title=raw.get("title", "Cuestionario"),
        intro=raw.get("intro", ""),
        sections=sections,
    )
