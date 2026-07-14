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


def score_bars(scores: dict[str, float]) -> go.Figure:
    """Barras horizontales con la afinidad a cada eje clínico."""
    labels = list(scores.keys())
    values = [round(scores[k] * 100) for k in labels]
    colors = [PHENOTYPE_COLORS.get(k, BRAND["primary"]) for k in labels]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v}%" for v in values], textposition="outside",
        )
    )
    fig.update_xaxes(range=[0, 100], title="Afinidad (%)")
    return _layout(fig, height=240)


def phenotype_gauge(phenotype: str, confidence: float) -> go.Figure:
    color = PHENOTYPE_COLORS.get(phenotype, BRAND["primary"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100),
            number={"suffix": "%"},
            title={"text": f"Confianza · {phenotype}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": BRAND["surface"],
            },
        )
    )
    return _layout(fig, height=260)


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
