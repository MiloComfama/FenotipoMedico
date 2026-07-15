"""Gráficas Plotly para las vistas de paciente y médico."""
from __future__ import annotations

import plotly.graph_objects as go

from app.config import BRAND, PHENOTYPE_COLORS


def _layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, Segoe UI, Arial", color=BRAND["ink"]),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0, min(255, c))):02X}" for c in rgb)


def _semaphore_color(value: float) -> str:
    """Degradado rojo → ámbar → verde según la magnitud del porcentaje (no por
    rangos fijos), para que el color varíe con cada valor y la gráfica no se
    vea plana."""
    value = max(0.0, min(100.0, value))
    stops = [(0, BRAND["danger"]), (50, BRAND["warning"]), (100, BRAND["success"])]
    for (v0, c0), (v1, c1) in zip(stops, stops[1:]):
        if v0 <= value <= v1:
            t = (value - v0) / (v1 - v0)
            rgb0, rgb1 = _hex_to_rgb(c0), _hex_to_rgb(c1)
            blended = tuple(a + (b - a) * t for a, b in zip(rgb0, rgb1))
            return _rgb_to_hex(blended)
    return stops[-1][1]


def score_bars(
    scores: dict[str, float], rationale: dict[str, list[str]] | None = None
) -> go.Figure:
    """Barras horizontales con la afinidad a cada eje clínico.

    Al pasar sobre cada barra se muestra, en lenguaje sencillo, qué respuestas
    reales de la encuesta explican ese porcentaje.
    """
    labels = list(scores.keys())
    values = [round(scores[k] * 100) for k in labels]
    colors = [_semaphore_color(v) for v in values]
    hover = [
        "<br>".join(f"• {r}" for r in (rationale or {}).get(label, []))
        or "Aún no hay suficientes respuestas para explicar este eje."
        for label in labels
    ]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v}%" for v in values], textposition="outside",
            hovertext=hover, hoverinfo="text",
        )
    )
    fig.update_xaxes(range=[0, 100], title="Afinidad (%)")
    return _layout(fig, height=240)


def phenotype_gauge(phenotype: str, confidence: float) -> go.Figure:
    value = round(confidence * 100)
    color = _semaphore_color(value)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": f"Categoría · {phenotype}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": BRAND["surface"],
                "steps": [
                    {"range": [0, 40], "color": "#FBEAEC"},
                    {"range": [40, 70], "color": "#FCF1DD"},
                    {"range": [70, 100], "color": "#E5F5EE"},
                ],
            },
        )
    )
    fig = _layout(fig, height=200)
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=10))
    return fig


def measurement_trend(consultations: list[dict], field: str, label: str) -> go.Figure:
    xs = [f"#{c['n']}" for c in consultations]
    ys = [c.get(field) for c in consultations]
    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=BRAND["primary"], width=3),
            marker=dict(size=9, color=BRAND["primary"]),
        )
    )
    fig.update_yaxes(title=label)
    fig.update_xaxes(title="Consulta")
    return _layout(fig, height=280)


def phenotype_history(consultations: list[dict]) -> go.Figure:
    """Línea temporal categórica del fenotipo por consulta."""
    order = list(PHENOTYPE_COLORS.keys())
    xs = [f"#{c['n']}" for c in consultations]
    ys = [c.get("phenotype") for c in consultations]
    colors = [PHENOTYPE_COLORS.get(p, "#999") for p in ys]
    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys, mode="markers+lines",
            line=dict(color=BRAND["muted"], width=1, dash="dot"),
            marker=dict(size=16, color=colors),
        )
    )
    fig.update_yaxes(categoryorder="array", categoryarray=order)
    fig.update_xaxes(title="Consulta")
    return _layout(fig, height=260)
