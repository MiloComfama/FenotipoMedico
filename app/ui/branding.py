"""Elementos de marca Comfama para Streamlit."""
from __future__ import annotations

import base64
import functools

import streamlit as st
import streamlit.components.v1 as components

from app.config import ASSETS_DIR, BRAND, PHENOTYPE_COLORS


@functools.lru_cache(maxsize=1)
def _css() -> str:
    return (ASSETS_DIR / "styles.css").read_text(encoding="utf-8")


@functools.lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    png = (ASSETS_DIR / "Logo_Comfama.png").read_bytes()
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def inject_theme() -> None:
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="brand-header">
            <img class="brand-logo" src="{_logo_data_uri()}" alt="Comfama"/>
            <div>
                <p class="brand-title">{title}</p>
                <p class="brand-sub">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Isotipo de FENIX: un ave/llama estilizada (guiño al nombre "Fénix") en
# degradado magenta de marca, con una pulsación (signo vital) integrada para
# anclarlo al sector salud.
_FENIX_ICON_SVG = """
<svg width="38" height="38" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="FENIX">
  <defs>
    <linearGradient id="fenixGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#FF277E"/>
      <stop offset="1" stop-color="#C71760"/>
    </linearGradient>
  </defs>
  <circle cx="20" cy="20" r="20" fill="url(#fenixGrad)"/>
  <path d="M20 7c4 4.5 9 7.5 9 14.5C29 28 25 33 20 33s-9-5-9-11.5C11 15 16 11.5 20 7Z"
        fill="#FFFFFF" opacity="0.95"/>
  <path d="M14 22h4l2.4-4.4L23 24h3" fill="none" stroke="url(#fenixGrad)"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
""".strip()


def sidebar_brand() -> None:
    """Isotipo + nombre de la aplicación (FENIX) para el tope de la barra
    lateral, sobre la paleta de marca Comfama."""
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-row">
                {_FENIX_ICON_SVG}
                <p class="sidebar-brand-name">FENIX</p>
            </div>
            <p class="sidebar-brand-tag">
                Fenotipos Inteligentes para la eXploración Clínica
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def focus_chat_input() -> None:
    """Mantiene el foco en el input de chat para que el usuario pueda seguir
    escribiendo sin necesidad de hacer clic después de cada turno."""
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            function focusInput() {
                const el = doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (el && doc.activeElement !== el) {
                    el.focus();
                }
            }
            focusInput();
            const observer = new MutationObserver(focusInput);
            observer.observe(doc.body, {childList: true, subtree: true});
            setTimeout(() => observer.disconnect(), 3000);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def phenotype_badge(phenotype: str | None) -> str:
    if not phenotype:
        return '<span class="badge" style="background:#9AA">Sin clasificar</span>'
    color = PHENOTYPE_COLORS.get(phenotype, BRAND["primary"])
    return f'<span class="badge" style="background:{color}">{phenotype}</span>'
