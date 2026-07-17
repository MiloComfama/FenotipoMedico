"""Texto a voz (ElevenLabs) para accesibilidad: lee en voz alta las
preguntas del cuestionario, para personas con baja visión o con
dificultades de lectura.

Si ``ELEVENLABS_API_KEY`` no está configurada, ``is_configured()`` devuelve
``False`` y el botón de "escuchar" no se muestra — el cuestionario sigue
funcionando normalmente por texto.
"""
from __future__ import annotations

import re

import requests

from app.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# Limpieza para que no se lea en voz alta la sintaxis de markdown ni emojis.
_MARKDOWN_NOISE = re.compile(r"[*_#>`]|^\s*[-•]\s+", re.MULTILINE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+"
)


def is_configured() -> bool:
    return bool(ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID)


def _clean_text(text: str) -> str:
    text = _EMOJI.sub("", text)
    text = _MARKDOWN_NOISE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize(text: str) -> bytes:
    """Devuelve el audio (mp3) del texto leído en voz alta.

    Lanza ``requests.HTTPError``/``requests.RequestException`` si la llamada
    al API falla; el llamador debe manejarlo (ver ``app/ui/patient_view.py``).
    """
    clean = _clean_text(text)
    response = requests.post(
        _API_URL.format(voice_id=ELEVENLABS_VOICE_ID),
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": clean,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.content
