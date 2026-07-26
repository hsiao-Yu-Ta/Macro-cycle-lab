from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PHASE_COLORS = {
    "擴張加速": "rgba(46, 204, 113, 0.14)",
    "擴張減速": "rgba(241, 196, 15, 0.14)",
    "復甦": "rgba(52, 152, 219, 0.14)",
    "放緩": "rgba(243, 156, 18, 0.14)",
    "谷底改善": "rgba(155, 89, 182, 0.14)",
    "衰退惡化": "rgba(231, 76, 60, 0.14)",
}


def add_phase_bands(fig, phase: pd.Series, row: int | None = None):
    phase = phase.dropna()
    if phase.empty:
        return fig

    start = phase.index[0]
    current = phase.iloc[0]
    previous_date = phase.index[0]

    for date, value in phase.iloc[1:].items():
        if value != current:
            fig.add_vrect(
                x0=start,
                x1=previous_date,
                fillcolor=PHASE_COLORS.get(current, "rgba(150,150,150,0.08)"),
                opacity=1,
                layer="below",
                line_width=0,
                row=row,
                col=1 if row else None,
            )
            start = date
            current = value
        previous_date = date

    fig.add_vrect(
        x0=start,
        x1=previous_date,
        fillcolor=PHASE_COLORS.get(current, "rgba(150,150,150,0.08)"),
        opacity=1,
        layer="below",
        line_width=0,
        row=row,
        col=1 if row else None,
    )
    return fig


def cycle_chart(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{}]],
    )

    if "PRICE" in frame:
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["PRICE"], name="價格", line=dict(width=2)),
            row=1, col=1, secondary_y=False
        )

    fig.add_trace(
        go.Scatter(x=frame.index, y=frame["SCORE"], name="綜合分數", line=dict(width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=frame["MOMENTUM_3M"], name="3月動能", line=dict(width=1.5)),
        row=2, col=1
    )

    add_phase_bands(fig, frame["PHASE"], row=1)
    add_phase_bands(fig, frame["PHASE"], row=2)
    fig.add_hline(y=0.35, line_dash="dot", row=2, col=1)
    fig.add_hline(y=-0.35, line_dash="dot", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", row=2, col=1)

    fig.update_layout(
        title=title,
        height=720,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(type="log", title_text="價格（對數）", row=1, col=1)
    fig.update_yaxes(title_text="模型分數", row=2, col=1)
    return fig


def comparison_chart(revised: pd.Series, original: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=revised.index, y=revised, name="修正版"))
    fig.add_trace(go.Scatter(x=original.index, y=original, name="原模型"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="原模型與修正版分數比較",
        height=430,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def equity_chart(backtests: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for name, df in backtests.items():
        col = "STRATEGY_EQUITY" if "STRATEGY_EQUITY" in df else "BUY_HOLD_EQUITY"
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name))
    fig.update_layout(
        title="資產曲線",
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(type="log", title="資產淨值（對數）")
    return fig
