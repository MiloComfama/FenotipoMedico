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
    svg = (ASSETS_DIR / "logo.svg").read_text(encoding="utf-8")
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


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
