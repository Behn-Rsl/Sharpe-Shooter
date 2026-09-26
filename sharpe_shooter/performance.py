"""In-sample performance of fixed-weight portfolios (rebalanced every period)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def growth_of_one(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Value of 1 invested in each weight column, rebalanced every period."""
    port = returns[weights.index] @ weights
    growth = (1 + port).cumprod()
    start = returns.index[0] - (returns.index[1] - returns.index[0])
    growth.loc[start] = 1.0
    return growth.sort_index()


def max_drawdown(values: pd.Series) -> float:
    return float((values / values.cummax() - 1).min())


def performance_table(growth: pd.DataFrame, periods_per_year: int, rf: float) -> pd.DataFrame:
    rets = growth.pct_change().dropna()
    years = len(rets) / periods_per_year
    rows = {}
    for col in growth.columns:
        r = rets[col]
        vol = r.std() * np.sqrt(periods_per_year)
        rows[col] = {
            "Total return": growth[col].iloc[-1] - 1,
            "Annual growth (CAGR)": growth[col].iloc[-1] ** (1 / years) - 1,
            "Volatility": vol,
            "Realized Sharpe": (r.mean() * periods_per_year - rf) / vol,
            "Max drawdown": max_drawdown(growth[col]),
        }
    return pd.DataFrame(rows).T


def asset_table(mu: pd.Series, cov: pd.DataFrame, rf: float, weights: pd.Series,
                names: dict[str, str] | None = None) -> pd.DataFrame:
    vol = pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=cov.index)
    frame = pd.DataFrame({
        "Expected return": mu,
        "Volatility": vol,
        "Sharpe ratio": (mu - rf) / vol,
        "Optimal weight": weights,
    })
    if names:
        frame.insert(0, "Name", [names.get(t, "") for t in frame.index])
    return frame
