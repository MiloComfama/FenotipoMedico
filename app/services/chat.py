"""Asistente conversacional de ingreso del cuestionario.

Dos implementaciones intercambiables tras la misma interfaz ``BaseIntake``:

* ``LLMIntake``      -> usa el modelo Opus 4.8 (requiere ANTHROPIC_API_KEY).
                        Guía la conversación, resuelve dudas del paciente y
                        captura respuestas estructuradas mediante *tool use*.
* ``ScriptedIntake`` -> flujo guiado determinista de respaldo (sin IA), para
                        que el prototipo funcione aunque no haya clave de API.

El estado (historial + respuestas) vive dentro del objeto, por lo que puede
guardarse en ``st.session_state`` y sobrevivir a los reruns de Streamlit.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.config import ANTHROPIC_API_KEY, ANTHROPIC_FOUNDRY_RESOURCE, ANTHROPIC_MODEL
from app.domain.questionnaire import Question, Questionnaire


@dataclass
class ChatTurn:
    assistant_text: str
    done: bool = False


# --- Lógica común de progreso -----------------------------------------------
def active_questions(q: Questionnaire, is_first: bool, answers: dict[str, Any]) -> list[Question]:
    return [qq for qq in q.questions_for(is_first) if qq.is_active(answers)]


def next_unanswered(q: Questionnaire, is_first: bool, answers: dict[str, Any]) -> Question | None:
    for question in active_questions(q, is_first, answers):
        val = answers.get(question.key)
        if val is None or val == "" or val == []:
            return question
    return None


class BaseIntake:
    def __init__(self, questionnaire: Questionnaire, is_first: bool):
        self.q = questionnaire
        self.is_first = is_first
        self.answers: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def progress(self) -> tuple[int, int]:
        total = len(active_questions(self.q, self.is_first, self.answers))
        done = sum(
            1 for k in self.answers
            if self.answers[k] not in (None, "", [])
        )
        return min(done, total), total

    def start(self) -> str:  # pragma: no cover - implementado por subclases
        raise NotImplementedError

    def handle(self, user_text: str) -> ChatTurn:  # pragma: no cover
        raise NotImplementedError


# --- Formato de preguntas para el flujo guiado ------------------------------
def _format_question(question: Question) -> str:
    lines = [question.text]
    if question.type == "bool":
        lines.append("_(Responde: Sí / No)_")
    elif question.type in ("single", "multi"):
        for i, opt in enumerate(question.options, 1):
            lines.append(f"  **{i}.** {opt}")
        if question.type == "multi":
            lines.append("_(Puedes elegir varias, separadas por coma. Ej: 1,3)_")
        else:
            lines.append("_(Responde con el número de la opción)_")
    return "\n".join(lines)


_SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _normalize_year(year: int) -> int:
    """Interpreta años de 2 dígitos como parte del siglo XX (fechas de
    nacimiento razonables), y deja los de 4 dígitos intactos."""
    return 1900 + year if year < 100 else year


def _parse_natural_date(text: str) -> str | None:
    """Interpreta una fecha en lenguaje natural o en formatos comunes y la
    normaliza a AAAA-MM-DD. Devuelve ``None`` si no logra interpretarla."""
    text = text.strip().lower()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            datetime.strptime(text, "%Y-%m-%d")
            return text
        except ValueError:
            return None

    # "26 de julio de 1991" / "26 de julio del 91" / "26 julio 1991"
    match = re.search(
        r"(\d{1,2})\s*(?:de)?\s*([a-záéíóúñ]+)\s*(?:de|del)?\s*(\d{2,4})", text
    )
    if match:
        day, month_name, year = match.groups()
        month = _SPANISH_MONTHS.get(month_name.strip())
        if month:
            try:
                return datetime(
                    _normalize_year(int(year)), month, int(day)
                ).strftime("%Y-%m-%d")
            except ValueError:
                return None

    # Formatos numéricos: DD/MM/AAAA, DD-MM-AA, AAAA/MM/DD, etc.
    match = re.fullmatch(r"(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})", text)
    if match:
        a, b, c = match.groups()
        try:
            if len(a) == 4:
                year, month, day = int(a), int(b), int(c)
            else:
                day, month, year = int(a), int(b), _normalize_year(int(c))
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def _parse_answer(question: Question, text: str) -> tuple[Any, str | None]:
    """Convierte el texto libre del usuario a un valor válido para la pregunta.

    Devuelve ``(valor, None)`` si la respuesta es válida, o ``(None, mensaje)``
    con una explicación clara de qué formato se esperaba.
    """
    text = text.strip()
    if not text:
        return None, "No recibí ninguna respuesta."

    if question.type == "bool":
        low = text.lower()
        if low in {"sí", "si", "s", "yes", "y", "1", "true", "verdadero"}:
            return True, None
        if low in {"no", "n", "0", "false", "falso"}:
            return False, None
        return None, "Por favor responde **Sí** o **No**."

    if question.type == "single":
        idx = _first_int(text)
        if idx is not None:
            if 1 <= idx <= len(question.options):
                return question.options[idx - 1], None
            return None, f"Ese número no existe. Elige un valor entre 1 y {len(question.options)}."
        for opt in question.options:
            if opt.lower() in text.lower():
                return opt, None
        return None, (
            "No reconocí esa opción. Responde con el **número** de una de las "
            "opciones listadas."
        )

    if question.type == "multi":
        chosen: list[str] = []
        for token in text.replace(" ", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                i = int(token)
                if 1 <= i <= len(question.options) and question.options[i - 1] not in chosen:
                    chosen.append(question.options[i - 1])
            else:
                match = next((opt for opt in question.options if opt.lower() in token.lower()), None)
                if match and match not in chosen:
                    chosen.append(match)
        if not chosen:
            return None, (
                "No logré identificar ninguna opción válida. Responde con los "
                "**números** de las opciones separadas por coma (ej: 1,3)."
            )
        return chosen, None

    if question.type == "number":
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            unit = f" (en {question.unit})" if question.unit else ""
            return None, f"Por favor ingresa solo un valor numérico{unit}."
        if question.min is not None and value < question.min:
            return None, f"El valor debe ser mayor o igual a {question.min}."
        if question.max is not None and value > question.max:
            return None, f"El valor debe ser menor o igual a {question.max}."
        return (int(value) if value.is_integer() else value), None

    if question.format == "date":
        normalized = _parse_natural_date(text)
        if normalized is None:
            return None, (
                "No logré identificar una fecha válida. Puedes escribirla como "
                "**AAAA-MM-DD** o en lenguaje natural (ej. \"26 de julio de 1991\")."
            )
        return normalized, None

    return text, None  # texto libre sin formato específico


def _first_int(text: str) -> int | None:
    for part in text.replace(",", " ").split():
        if part.isdigit():
            return int(part)
    return None


_STRAY_TAG_PATTERN = re.compile(r"</?[a-zA-Z_][\w:-]*\s*/?>")


def _strip_stray_tags(text: str) -> str:
    """Elimina etiquetas tipo HTML/XML sueltas (p. ej. ``<user>``) que a veces
    aparecen en la respuesta cruda del modelo. Sin esto, el navegador oculta
    la etiqueta pero deja visible el texto interior pegado al resto del
    mensaje (ej. "...responde **Sí** o **No**. 😊\\n\\nusersi"), que es
    confuso para quien lee el chat."""
    return _STRAY_TAG_PATTERN.sub("", text)


_DOUBT_PATTERN = re.compile(
    r"\?|no entiendo|no comprendo|no s[eé] q|qu[eé] (es|significa)|"
    r"que (es|significa)|c[oó]mo (respondo|lleno|contesto)|ay[uú]dame|ayuda|"
    r"explica|duda|no me qued[oó] claro",
    re.IGNORECASE,
)


def _looks_like_doubt(text: str) -> bool:
    """Heurística simple para detectar si el usuario está preguntando algo
    en lugar de responder (flujo guiado sin IA)."""
    return bool(_DOUBT_PATTERN.search(text.strip()))


def _explain_question(question: Question) -> str:
    if question.help:
        return question.help.strip()
    return "Elige la opción que mejor describa tu situación; no hay respuestas correctas o incorrectas."


# ===========================================================================
# Flujo guiado determinista (respaldo sin IA)
# ===========================================================================
class ScriptedIntake(BaseIntake):
    def start(self) -> str:
        intro = (
            f"¡Hola! 👋 Soy **Fénix**, tu asistente del **Programa de Medicina Funcional "
            f"de Comfama**. {self.q.intro.strip()}\n\nComencemos."
        )
        question = next_unanswered(self.q, self.is_first, self.answers)
        return f"{intro}\n\n{_format_question(question)}" if question else intro

    def handle(self, user_text: str) -> ChatTurn:
        current = next_unanswered(self.q, self.is_first, self.answers)
        if current is None:
            return ChatTurn("¡Gracias! Ya tenemos toda tu información.", done=True)

        if _looks_like_doubt(user_text):
            return ChatTurn(
                f"{_explain_question(current)}\n\n" + _format_question(current)
            )

        parsed, error = _parse_answer(current, user_text)
        if error:
            return ChatTurn(f"{error} 🙏\n\n" + _format_question(current))

        self.answers[current.key] = parsed
        nxt = next_unanswered(self.q, self.is_first, self.answers)
        if nxt is None:
            return ChatTurn(
                "¡Perfecto! 🎉 Has completado el cuestionario. "
                "Ahora generaré tu clasificación y recomendaciones.",
                done=True,
            )
        return ChatTurn("Anotado ✅\n\n" + _format_question(nxt))


# ===========================================================================
# Flujo con Opus 4.8
# ===========================================================================
SYSTEM_PROMPT = """\
Te llamas Fénix, el asistente conversacional del Programa de Medicina Funcional
de Comfama. Tu único objetivo es acompañar a la persona a completar un
cuestionario de salud de forma cálida, clara y respetuosa, en español (Colombia).

REGLAS:
- Preséntate por tu nombre (Fénix) en tu primer mensaje de la conversación.
- Haz UNA pregunta a la vez, en lenguaje sencillo. No abrumes con listas largas.
- Cuando una pregunta sea de tipo 'single' o 'multi', presenta SIEMPRE las
  opciones como una lista numerada (1., 2., 3., ...) e invita a responder con
  el número (o números separados por coma para 'multi'). Esto facilita que la
  persona conteste sin tener que escribir el texto completo de la opción.
- Solo resuelve dudas sobre la PREGUNTA ACTUAL pendiente (qué significa, cómo
  responderla, ejemplos). Si la persona pregunta algo que NO tiene relación con
  la pregunta actual (otro tema del programa, salud general, u otro asunto
  ajeno), NO lo respondas: redirige amablemente indicando que puedes ayudarle
  con eso más adelante o con el equipo médico, y vuelve a formular la pregunta
  pendiente.
- NO te desvíes del tema. Si te piden algo ajeno al programa (política, tareas,
  código, etc.), redirige amablemente al cuestionario.
- NO entregues diagnósticos ni prescripciones. Aclara que la clasificación final
  y el tratamiento los define el equipo médico.

VALIDACIÓN DE FORMATO (muy importante):
- Antes de guardar una respuesta, verifica que tenga el formato exacto esperado
  según el 'tipo' de la pregunta:
  · single  -> debe corresponder EXACTAMENTE a una de las 'opciones' listadas.
  · multi   -> cada elemento debe ser una de las 'opciones' listadas (arreglo).
  · bool    -> interprétalo como verdadero/falso (sí/no, afirmaciones claras).
  · number  -> debe ser un valor numérico; si hay min/max, debe estar en rango.
  · text    -> si el 'formato' pide un patrón concreto (p. ej. una fecha
    AAAA-MM-DD), interpreta la INTENCIÓN de la persona aunque la escriba en
    lenguaje natural o en otro formato (ej. "26 de julio del 91", "5/05/1990",
    "nací en 1994") y conviértela TÚ MISMO al formato exacto requerido antes de
    guardarla con `guardar_respuesta`. No le pidas que reescriba la fecha en un
    formato específico si su intención ya es clara.
- Si la respuesta NO cumple el formato esperado Y además es ambigua o
  incompleta (p. ej. falta el año, o no queda claro a qué valor se refiere),
  NO llames a `guardar_respuesta`. En su lugar, explica en una frase breve y
  amable qué información falta (con un ejemplo si ayuda) y vuelve a formular
  la MISMA pregunta pendiente.
- Cuando la persona responda correctamente, registra su respuesta con la
  herramienta `guardar_respuesta` usando el 'key' exacto.
- Solo pregunta por los ítems aplicables; respeta las dependencias (depends_on).
- Cuando ya tengas TODAS las respuestas aplicables, llama a `finalizar` con un
  mensaje de cierre amable.

REGLA CRÍTICA — DUDAS SOBRE LA PREGUNTA ACTUAL (léela dos veces):
Si el mensaje de la persona NO es una respuesta a la pregunta pendiente sino
una duda/pregunta sobre ella (qué significa, cómo responderla, ejemplos,
parece confundida), tu turno completo debe consistir ÚNICAMENTE en: (1) una
aclaración breve, y (2) repetir la MISMA pregunta pendiente (con sus opciones
numeradas si aplica). En ese turno tienes PROHIBIDO:
  · Llamar a `guardar_respuesta` (la pregunta actual sigue SIN responder).
  · Llamar a `finalizar`.
  · Mencionar, adelantar o formular cualquier pregunta distinta a la actual.
Nunca avances a la siguiente pregunta hasta que la persona conteste
efectivamente la pregunta pendiente — aclarar una duda no cuenta como
respuesta.

Ejemplo de lo que NO debes hacer:
  Pregunta pendiente: "¿Realiza actividad física?"
  Persona: "¿eso incluye caminar al trabajo?"
  ❌ Incorrecto: "Sí, cuenta. Por cierto, ya registré tu actividad. Ahora,
     ¿duerme bien en la noche?" (avanzó sin que la persona respondiera)
  ✅ Correcto: "Sí, caminar cuenta como actividad física. ¿Realiza actividad
     física? (Sí/No)"

CUESTIONARIO (formato JSON):
{schema}
"""

TOOLS = [
    {
        "name": "guardar_respuesta",
        "description": "Registra la respuesta de la persona a una pregunta del cuestionario.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Identificador exacto de la pregunta."},
                "value": {
                    "description": "Respuesta. Para 'multi' usa un arreglo de textos; "
                    "para 'bool' usa true/false; para 'single' el texto de la opción.",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "finalizar",
        "description": "Indica que el cuestionario está completo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mensaje_cierre": {"type": "string"},
            },
            "required": ["mensaje_cierre"],
        },
    },
]


class LLMIntake(BaseIntake):
    def __init__(self, questionnaire: Questionnaire, is_first: bool):
        super().__init__(questionnaire, is_first)

        if ANTHROPIC_FOUNDRY_RESOURCE:
            from anthropic import AnthropicFoundry  # import perezoso

            self._client = AnthropicFoundry(
                api_key=ANTHROPIC_API_KEY, resource=ANTHROPIC_FOUNDRY_RESOURCE
            )
        else:
            from anthropic import Anthropic  # import perezoso

            self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self._system = SYSTEM_PROMPT.format(schema=self._schema_json())
        self._done = False

    def _schema_json(self) -> str:
        items = []
        for q in self.q.questions_for(self.is_first):
            items.append({
                "key": q.key,
                "pregunta": q.text,
                "tipo": q.type,
                "opciones": q.options or None,
                "depends_on": q.depends_on,
                "unidad": q.unit,
                "min": q.min,
                "max": q.max,
                "formato": q.format,
                "ayuda": q.help,
            })
        return json.dumps(items, ensure_ascii=False, indent=2)

    def start(self) -> str:
        # Mensaje inicial: pedimos al modelo que salude y haga la primera pregunta.
        self.history.append({
            "role": "user",
            "content": "Hola, quiero completar el cuestionario del programa.",
        })
        return self._run_turn()

    def handle(self, user_text: str) -> ChatTurn:
        self.history.append({"role": "user", "content": user_text})
        text = self._run_turn()
        return ChatTurn(text, done=self._done)

    def _run_turn(self, max_iterations: int = 6) -> str:
        assistant_text_parts: list[str] = []
        for _ in range(max_iterations):
            response = self._client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1024,
                system=self._system,
                tools=TOOLS,
                messages=self.history,
            )
            # Guardamos el turno del asistente en el historial.
            self.history.append({"role": "assistant", "content": response.content})

            tool_results = []
            has_text = False
            for block in response.content:
                if block.type == "text":
                    if block.text.strip():
                        has_text = True
                    assistant_text_parts.append(block.text)
                elif block.type == "tool_use":
                    result = self._apply_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if tool_results:
                self.history.append({"role": "user", "content": tool_results})
                if self._done or has_text:
                    # Ya le mostramos texto real a la persona (p. ej. la
                    # siguiente pregunta): paramos aquí y esperamos su próxima
                    # respuesta real, en vez de dejar que el modelo siga
                    # encadenando turnos por su cuenta.
                    break
                continue  # el modelo aún no ha formulado la siguiente pregunta; démosle un turno más
            break  # sin herramientas: el modelo ya respondió con texto

        joined = "\n\n".join(p for p in assistant_text_parts if p.strip()).strip()
        return _strip_stray_tags(joined) or "…"

    def _apply_tool(self, name: str, payload: dict[str, Any]) -> str:
        if name == "guardar_respuesta":
            key = payload.get("key")
            value = payload.get("value")
            self.answers[key] = value
            return f"Guardado: {key} = {value}"
        if name == "finalizar":
            self._done = True
            return "Cuestionario finalizado."
        return "Herramienta desconocida."


# --- Fábrica ----------------------------------------------------------------
def create_intake(questionnaire: Questionnaire, is_first: bool) -> BaseIntake:
    """Devuelve el asistente adecuado según la disponibilidad de la clave de API."""
    if ANTHROPIC_API_KEY:
        try:
            return LLMIntake(questionnaire, is_first)
        except Exception:
            pass
    return ScriptedIntake(questionnaire, is_first)


def using_llm() -> bool:
    return bool(ANTHROPIC_API_KEY)
