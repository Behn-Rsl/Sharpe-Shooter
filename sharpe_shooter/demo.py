"""Three demo portfolios that work offline.

The prices are simulated, not downloaded. Each asset is given a long-run
return, a volatility and loadings on a market factor and a sector factor;
daily returns are drawn from a fat-tailed (Student-t) multivariate
distribution with that structure. The tickers are real so the demos read
naturally, but the numbers are illustrative. Tick "Use live market data"
in the app to run the same tickers on real Yahoo Finance prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoAsset:
    ticker: str
    name: str
    annual_return: float
    annual_vol: float
    market: float   # loading on the common market factor
    group: str      # assets in the same group share a second factor
    group_loading: float


@dataclass(frozen=True)
class DemoPortfolio:
    key: str
    name: str
    description: str
    assets: tuple[DemoAsset, ...]
    seed: int

    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]


DEMOS: dict[str, DemoPortfolio] = {
    "big_tech": DemoPortfolio(
        key="big_tech",
        name="Big Tech",
        description="Six mega-cap technology stocks. High returns, high volatility and strong "
                    "co-movement, so diversification is limited and the optimizer has to pick favorites.",
        seed=11,
        assets=(
            DemoAsset("AAPL", "Apple", 0.22, 0.28, 0.62, "platform", 0.42),
            DemoAsset("MSFT", "Microsoft", 0.21, 0.26, 0.64, "platform", 0.45),
            DemoAsset("GOOGL", "Alphabet", 0.17, 0.30, 0.60, "platform", 0.44),
            DemoAsset("AMZN", "Amazon", 0.18, 0.34, 0.58, "platform", 0.38),
            DemoAsset("NVDA", "NVIDIA", 0.42, 0.52, 0.55, "semis", 0.30),
            DemoAsset("META", "Meta Platforms", 0.20, 0.40, 0.55, "platform", 0.36),
        ),
    ),
    "blue_chips": DemoPortfolio(
        key="blue_chips",
        name="Diversified blue chips",
        description="Seven large companies across healthcare, banking, energy, consumer staples and "
                    "industrials. Defensive and cyclical names pull in different directions, which "
                    "gives the covariance matrix something to work with.",
        seed=23,
        assets=(
            DemoAsset("JNJ", "Johnson & Johnson", 0.07, 0.18, 0.45, "defensive", 0.40),
            DemoAsset("PG", "Procter & Gamble", 0.09, 0.17, 0.45, "defensive", 0.45),
            DemoAsset("KO", "Coca-Cola", 0.08, 0.16, 0.44, "defensive", 0.45),
            DemoAsset("WMT", "Walmart", 0.12, 0.20, 0.42, "defensive", 0.30),
            DemoAsset("JPM", "JPMorgan Chase", 0.14, 0.26, 0.68, "cyclical", 0.35),
            DemoAsset("XOM", "Exxon Mobil", 0.11, 0.30, 0.45, "energy", 0.55),
            DemoAsset("CAT", "Caterpillar", 0.15, 0.29, 0.66, "cyclical", 0.35),
        ),
    ),
    "global_etfs": DemoPortfolio(
        key="global_etfs",
        name="Global multi-asset ETFs",
        description="US, developed and emerging-market equities alongside long Treasuries, gold and "
                    "real estate. Low and negative correlations make this the clearest illustration "
                    "of why the covariance matrix matters more than any single asset's return.",
        seed=37,
        assets=(
            DemoAsset("SPY", "S&P 500", 0.11, 0.17, 0.95, "equity", 0.20),
            DemoAsset("EFA", "Developed ex-US equities", 0.07, 0.17, 0.82, "equity", 0.30),
            DemoAsset("EEM", "Emerging-market equities", 0.06, 0.21, 0.72, "equity", 0.30),
            DemoAsset("TLT", "20+ year US Treasuries", 0.035, 0.15, -0.25, "rates", 0.60),
            DemoAsset("GLD", "Gold", 0.075, 0.15, 0.08, "real", 0.45),
            DemoAsset("VNQ", "US real estate", 0.075, 0.22, 0.74, "rates", 0.25),
        ),
    ),
}

TRADING_DAYS = 252


def correlation_matrix(demo: DemoPortfolio) -> np.ndarray:
    """Two-factor correlation structure, guaranteed positive definite."""
    groups = sorted({a.group for a in demo.assets})
    b = np.zeros((len(demo.assets), 1 + len(groups)))
    for i, a in enumerate(demo.assets):
        b[i, 0] = a.market
        b[i, 1 + groups.index(a.group)] = a.group_loading
    corr = b @ b.T
    np.fill_diagonal(corr, 1.0)  # idiosyncratic variance tops the diagonal up to 1
    return corr


def simulate_prices(demo: DemoPortfolio, years: int = 5, end: str = "2025-12-31", dof: int = 5) -> pd.DataFrame:
    """Simulated daily closing prices, reproducible for a given demo."""
    rng = np.random.default_rng(demo.seed)
    dates = pd.bdate_range(end=end, periods=years * TRADING_DAYS + 1)
    n_days = len(dates) - 1

    vols = np.array([a.annual_vol for a in demo.assets]) / np.sqrt(TRADING_DAYS)
    means = np.array([a.annual_return for a in demo.assets]) / TRADING_DAYS
    cov = np.outer(vols, vols) * correlation_matrix(demo)
    chol = np.linalg.cholesky(cov)

    # Multivariate Student-t shocks scaled to unit variance: fatter tails than a normal.
    z = rng.standard_normal((n_days, len(vols)))
    scale = np.sqrt(rng.chisquare(dof, size=(n_days, 1)) / dof) * np.sqrt(dof / (dof - 2))
    shocks = (z / scale) @ chol.T

    # Pin the sample mean to the target so the demos behave as described.
    returns = shocks - shocks.mean(axis=0) + means
    returns = np.clip(returns, -0.5, None)

    prices = 100 * np.vstack([np.ones(len(vols)), np.cumprod(1 + returns, axis=0)])
    return pd.DataFrame(prices, index=dates, columns=demo.tickers).round(4)


def asset_names(demo: DemoPortfolio) -> dict[str, str]:
    return {a.ticker: a.name for a in demo.assets}
