"""Turning prices into the two inputs the optimizer needs: a vector of
expected returns (mu) and a covariance matrix (Sigma), both annualized."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Periods per year used to annualize statistics at each sampling frequency.
FREQUENCIES: dict[str, int] = {"Daily": 252, "Weekly": 52, "Monthly": 12}

_RESAMPLE_RULES = {"Weekly": "W-FRI", "Monthly": "ME"}

COVARIANCE_METHODS = {
    "sample": "Sample covariance",
    "ledoit_wolf": "Ledoit–Wolf shrinkage",
}


@dataclass(frozen=True)
class CovarianceEstimate:
    matrix: pd.DataFrame
    method: str
    shrinkage: float | None = None  # only set for Ledoit–Wolf


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Guess the sampling frequency of a date index from its median spacing."""
    if len(index) < 3:
        return 252
    gaps = pd.Series(index).diff().dt.days.dropna()
    median_gap = float(gaps.median())
    if median_gap < 4:
        return 252
    if median_gap < 15:
        return 52
    if median_gap < 45:
        return 12
    return 4


_LABELS = {252: "daily", 52: "weekly", 12: "monthly", 4: "quarterly"}


def resample_prices(prices: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, int, str, list[str]]:
    """Resample prices to the requested frequency.

    Returns (prices, periods_per_year, label, notes). Data can only be made
    coarser, so if the source is weekly and daily is requested we keep it
    weekly and say so.
    """
    notes: list[str] = []
    native = infer_periods_per_year(prices.index)
    requested = FREQUENCIES[frequency]
    if requested > native:
        label = _LABELS[native]
        notes.append(
            f"The source data is {label}, so returns are computed {label} "
            f"rather than {frequency.lower()}."
        )
        return prices, native, label, notes
    if requested == native:
        return prices, requested, _LABELS[requested], notes
    resampled = prices.resample(_RESAMPLE_RULES[frequency]).last().dropna(how="all")
    return resampled, requested, _LABELS[requested], notes


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Align price series on common dates and drop anything unusable."""
    notes: list[str] = []
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.where(prices > 0)

    empty = [c for c in prices.columns if prices[c].notna().sum() < 3]
    if empty:
        notes.append(f"Dropped {', '.join(map(str, empty))}: not enough price history.")
        prices = prices.drop(columns=empty)
    if prices.shape[1] == 0:
        return prices, notes

    # Warn when one short history truncates everyone else's.
    first_valid = prices.apply(pd.Series.first_valid_index)
    latest_start = first_valid.max()
    if first_valid.min() < latest_start - pd.Timedelta(days=90):
        laggard = first_valid.idxmax()
        notes.append(
            f"{laggard} only has data from {latest_start:%d %b %Y}, so every asset is "
            f"analyzed from that date to keep the covariance matrix consistent."
        )
    prices = prices.dropna(how="any")
    return prices, notes


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (arithmetic) period returns: r_t = P_t / P_{t-1} - 1."""
    return prices.pct_change().dropna(how="any")


def expected_returns(returns: pd.DataFrame, periods_per_year: int) -> pd.Series:
    """Annualized arithmetic mean return of each asset."""
    return returns.mean() * periods_per_year


def sample_covariance(returns: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    """Annualized unbiased sample covariance matrix."""
    return returns.cov() * periods_per_year


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> float:
    """Optimal shrinkage intensity toward a scaled identity matrix.

    Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for
    large-dimensional covariance matrices." Journal of Multivariate Analysis.
    """
    x = returns.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    n, p = x.shape
    s = x.T @ x / n                                   # MLE sample covariance
    m = np.trace(s) / p                               # target: m * I
    d2 = (np.sum(s**2) - p * m**2) / p               # ||S - mI||^2 / p
    row_norms_sq = np.sum(x**2, axis=1)
    b2_bar = (np.sum(row_norms_sq**2) / n - np.sum(s**2)) / (n * p)
    b2 = min(b2_bar, d2)
    return 0.0 if d2 == 0 else float(b2 / d2)


def ledoit_wolf_covariance(returns: pd.DataFrame, periods_per_year: int) -> tuple[pd.DataFrame, float]:
    """Annualized Ledoit–Wolf shrunk covariance and the shrinkage used."""
    x = returns.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    n, p = x.shape
    s = x.T @ x / n
    m = np.trace(s) / p
    delta = ledoit_wolf_shrinkage(returns)
    shrunk = (1 - delta) * s + delta * m * np.eye(p)
    cov = pd.DataFrame(shrunk * periods_per_year, index=returns.columns, columns=returns.columns)
    return cov, delta


def estimate_covariance(returns: pd.DataFrame, periods_per_year: int, method: str = "sample") -> CovarianceEstimate:
    if method == "sample":
        return CovarianceEstimate(sample_covariance(returns, periods_per_year), method)
    if method == "ledoit_wolf":
        cov, delta = ledoit_wolf_covariance(returns, periods_per_year)
        return CovarianceEstimate(cov, method, delta)
    raise ValueError(f"Unknown covariance method '{method}'. Use one of {list(COVARIANCE_METHODS)}.")


def correlation_from_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    sd = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(sd, sd)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
