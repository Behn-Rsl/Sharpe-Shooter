"""Sharpe Shooter: maximum-Sharpe portfolio optimization."""

from .estimators import (
    FREQUENCIES,
    clean_prices,
    estimate_covariance,
    expected_returns,
    prices_to_returns,
    resample_prices,
)
from .optimizer import (
    Constraints,
    OptimizationError,
    OptimizationResult,
    Portfolio,
    optimize,
    risk_contributions,
)

__all__ = [
    "FREQUENCIES",
    "Constraints",
    "OptimizationError",
    "OptimizationResult",
    "Portfolio",
    "clean_prices",
    "estimate_covariance",
    "expected_returns",
    "optimize",
    "prices_to_returns",
    "resample_prices",
    "risk_contributions",
]

__version__ = "1.0.0"
