"""Plotly figures. Red is reserved for the max-Sharpe portfolio everywhere,
so it always means "the optimal portfolio"."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .optimizer import OptimizationResult, risk_contributions

INK = "#1D2B36"
MUTED = "#5B6B77"
RULE = "#CBD4D8"
STEEL = "#2E5E7E"      # frontier, primary interface colour
TARGET = "#C8372D"     # the optimal (max-Sharpe) portfolio, and only that
TEAL = "#3F8A7A"       # minimum-variance portfolio
SLATE = "#8C979F"      # equal-weight portfolio
CLOUD = "#A9B7BF"
OCHRE = "#B8862F"

SERIES_PALETTE = ["#2E5E7E", "#3F8A7A", "#B8862F", "#6B5B95", "#4F7CAC",
                  "#7A9E4F", "#A0673C", "#5F6F7A", "#9A7FB0", "#2F8FA8"]

FONT = "Barlow, 'Helvetica Neue', Arial, sans-serif"


def _base_layout(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height,
        font=dict(family=FONT, size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=12, b=12),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=RULE, font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=MUTED)),
        **kwargs,
    )
    axis = dict(gridcolor=RULE, gridwidth=0.6, zeroline=False, linecolor=RULE, automargin=True,
                tickfont=dict(color=MUTED, size=12), title_font=dict(color=MUTED, size=13))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _label_positions(x: np.ndarray, y: np.ndarray) -> list[str]:
    """Put a label on the left when another asset sits just to its right."""
    xs = (x - x.min()) / max(np.ptp(x), 1e-12)
    ys = (y - y.min()) / max(np.ptp(y), 1e-12)
    positions = []
    for i in range(len(x)):
        crowded = any(0 <= xs[j] - xs[i] < 0.12 and abs(ys[j] - ys[i]) < 0.06
                      for j in range(len(x)) if j != i)
        near_edge = xs[i] > 0.88
        positions.append("middle left" if crowded or near_edge else "middle right")
    return positions


def frontier_figure(result: OptimizationResult, cloud: pd.DataFrame | None = None,
                    names: dict[str, str] | None = None) -> go.Figure:
    """Efficient frontier, capital market line and the tangency 'shot'."""
    names = names or {}
    rf = result.risk_free_rate
    tan, gmv, eq = result.max_sharpe, result.min_variance, result.equal_weight
    asset_vol = np.sqrt(np.diag(result.cov.to_numpy()))
    fig = go.Figure()

    if cloud is not None and not cloud.empty:
        fig.add_trace(go.Scatter(
            x=cloud["volatility"], y=cloud["return"], mode="markers", name="Random long-only portfolios",
            marker=dict(size=4, color=CLOUD, opacity=0.35),
            hovertemplate="Return %{y:.1%}<br>Volatility %{x:.1%}<extra>Random portfolio</extra>",
        ))

    f = result.frontier
    fig.add_trace(go.Scatter(
        x=f["volatility"], y=f["return"], mode="lines", name="Efficient frontier",
        line=dict(color=STEEL, width=3),
        customdata=f["sharpe"],
        hovertemplate="Return %{y:.1%}<br>Volatility %{x:.1%}<br>Sharpe %{customdata:.2f}<extra>Frontier</extra>",
    ))

    # Capital market line: from the risk-free rate through the tangency portfolio.
    x_max = max(float(f["volatility"].max()), float(asset_vol.max()), tan.volatility) * 1.08
    fig.add_trace(go.Scatter(
        x=[0, x_max], y=[rf, rf + tan.sharpe * x_max], mode="lines", name="Capital market line",
        line=dict(color=TARGET, width=1.5, dash="dash"),
        hovertemplate="Slope = Sharpe ratio %{customdata:.2f}<extra>Capital market line</extra>",
        customdata=[tan.sharpe, tan.sharpe],
    ))

    labels = list(result.mu.index)
    fig.add_trace(go.Scatter(
        x=asset_vol, y=result.mu, mode="markers+text", name="Individual assets",
        text=labels, textposition=_label_positions(asset_vol, result.mu.to_numpy()),
        textfont=dict(size=12, color=INK),
        marker=dict(size=8, color=INK, symbol="diamond"),
        customdata=[[names.get(t, t), (m - rf) / v] for t, m, v in zip(labels, result.mu, asset_vol)],
        hovertemplate="<b>%{text}</b> %{customdata[0]}<br>Return %{y:.1%}<br>Volatility %{x:.1%}"
                      "<br>Sharpe %{customdata[1]:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[0], y=[rf], mode="markers", name="Risk-free rate", showlegend=False, cliponaxis=False,
        marker=dict(size=9, color="#FFFFFF", line=dict(color=TARGET, width=2)),
        hovertemplate=f"Risk-free rate {rf:.2%}<extra></extra>",
    ))
    for p, colour, symbol in ((gmv, TEAL, "square"), (eq, SLATE, "circle")):
        fig.add_trace(go.Scatter(
            x=[p.volatility], y=[p.expected_return], mode="markers", name=p.name,
            marker=dict(size=11, color=colour, symbol=symbol, line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate=f"<b>{p.name}</b><br>Return %{{y:.1%}}<br>Volatility %{{x:.1%}}"
                          f"<br>Sharpe {p.sharpe:.2f}<extra></extra>",
        ))

    # The reticle: an open ring with a thin cross through the optimal portfolio.
    fig.add_trace(go.Scatter(
        x=[tan.volatility], y=[tan.expected_return], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=34, symbol="cross-thin", line=dict(color=TARGET, width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=[tan.volatility], y=[tan.expected_return], mode="markers", name="Max Sharpe",
        marker=dict(size=20, symbol="circle-open", color=TARGET, line=dict(width=2.5)),
        hovertemplate=f"<b>Max Sharpe</b><br>Return %{{y:.1%}}<br>Volatility %{{x:.1%}}"
                      f"<br>Sharpe {tan.sharpe:.2f}<extra></extra>",
    ))

    y_vals = np.concatenate([f["return"], result.mu.to_numpy(), [rf, tan.expected_return]])
    pad = (y_vals.max() - min(y_vals.min(), 0)) * 0.08
    _base_layout(fig, height=540)
    # Legend below the plot so it never hides an asset.
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0,
                                  font=dict(size=12, color=INK)))
    fig.update_xaxes(title="Volatility (annualized standard deviation)", tickformat=".0%",
                     range=[0, x_max], rangemode="tozero")
    fig.update_yaxes(title="Expected return (annualized)", tickformat=".0%",
                     range=[min(y_vals.min(), 0) - pad, y_vals.max() + pad])
    return fig


def weights_figure(result: OptimizationResult) -> go.Figure:
    """Optimal weights as bars; the two reference portfolios as tick marks."""
    table = result.weights_table().sort_values("Max Sharpe")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=table.index, x=table["Max Sharpe"], orientation="h", name="Max Sharpe",
        marker=dict(color=TARGET), width=0.55,
        text=[f"{w:.1%}" for w in table["Max Sharpe"]], textposition="outside",
        textfont=dict(color=INK, size=12), cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra>Max Sharpe</extra>",
    ))
    for col, colour, symbol in (("Min variance", TEAL, "line-ns"), ("Equal weight", SLATE, "line-ns")):
        fig.add_trace(go.Scatter(
            y=table.index, x=table[col], mode="markers", name=col,
            marker=dict(symbol=symbol, size=18, line=dict(color=colour, width=3)),
            hovertemplate=f"%{{y}}: %{{x:.2%}}<extra>{col}</extra>",
        ))
    lo = min(0.0, float(table.min().min()))
    hi = float(table.max().max())
    _base_layout(fig, height=max(260, 46 * len(table) + 80))
    fig.update_xaxes(tickformat=".0%", range=[lo * 1.15 - 0.02, hi * 1.18 + 0.02],
                     zeroline=True, zerolinecolor=MUTED, zerolinewidth=1)
    fig.update_yaxes(showgrid=False)
    return fig


def correlation_figure(corr: pd.DataFrame) -> go.Figure:
    colorscale = [[0.0, OCHRE], [0.5, "#F7F8F6"], [1.0, STEEL]]
    z = corr.to_numpy()
    text = [[f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=corr.columns, y=corr.index, zmin=-1, zmax=1, zmid=0, colorscale=colorscale,
        text=text, texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=MUTED)),
        xgap=2, ygap=2,
    ))
    _base_layout(fig, height=max(320, 44 * len(corr) + 80))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return fig


def risk_contribution_figure(result: OptimizationResult) -> go.Figure:
    w = result.max_sharpe.weights
    rc = risk_contributions(w, result.cov)
    order = w.sort_values().index
    fig = go.Figure()
    fig.add_trace(go.Bar(y=order, x=w[order], orientation="h", name="Share of capital",
                         marker=dict(color=TARGET, opacity=0.35),
                         hovertemplate="%{y}: %{x:.1%} of capital<extra></extra>"))
    fig.add_trace(go.Bar(y=order, x=rc[order], orientation="h", name="Share of risk",
                         marker=dict(color=STEEL),
                         hovertemplate="%{y}: %{x:.1%} of variance<extra></extra>"))
    _base_layout(fig, height=max(260, 46 * len(order) + 80), barmode="group", bargap=0.3)
    fig.update_xaxes(tickformat=".0%", zeroline=True, zerolinecolor=MUTED)
    fig.update_yaxes(showgrid=False)
    return fig


def growth_figure(growth: pd.DataFrame) -> go.Figure:
    colours = {"Max Sharpe": TARGET, "Min variance": TEAL, "Equal weight": SLATE}
    fig = go.Figure()
    for col in growth.columns:
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth[col], mode="lines", name=col,
            line=dict(color=colours.get(col, STEEL), width=2.4 if col == "Max Sharpe" else 1.6),
            hovertemplate=f"%{{x|%d %b %Y}}: %{{y:.2f}}<extra>{col}</extra>",
        ))
    _base_layout(fig, height=400)
    fig.update_yaxes(title="Value of 1 invested", tickformat=".2f")
    return fig


def prices_figure(prices: pd.DataFrame) -> go.Figure:
    rebased = prices / prices.iloc[0]
    fig = go.Figure()
    for i, col in enumerate(rebased.columns):
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased[col], mode="lines", name=col,
            line=dict(color=SERIES_PALETTE[i % len(SERIES_PALETTE)], width=1.5),
            hovertemplate=f"%{{x|%d %b %Y}}: %{{y:.2f}}<extra>{col}</extra>",
        ))
    _base_layout(fig, height=400)
    fig.update_yaxes(title="Price rebased to 1")
    return fig
