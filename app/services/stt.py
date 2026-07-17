"""Voz a texto (Whisper local, vía ``faster-whisper``) para responder por
micrófono en el cuestionario conversacional.

Corre completamente en la máquina local — sin API ni costo por uso. El
modelo se descarga una sola vez (la primera transcripción) y se cachea en
memoria para las siguientes.

Si ``faster-whisper`` no está instalado, ``is_configured()`` devuelve
``False`` y el botón de micrófono no se muestra — el cuestionario sigue
funcionando normalmente por texto.
"""
from __future__ import annotations

import functools
import os
import tempfile

from app.config import WHISPER_MODEL_SIZE


def is_configured() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel

    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe un audio (voz del paciente) a texto en español."""
    model = _get_model()
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        # vad_filter=False: en grabaciones cortas del navegador, el filtro de
        # detección de voz a veces descarta la respuesta completa (falso
        # negativo) y devuelve texto vacío sin ningún error.
        segments, _info = model.transcribe(path, language="es", vad_filter=False)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        os.remove(path)
