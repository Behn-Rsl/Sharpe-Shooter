"""Mean–variance optimization: the maximum-Sharpe (tangency) portfolio,
the minimum-variance portfolio and the efficient frontier.

Notation (all quantities annualized):
    mu     expected return vector, shape (n,)
    Sigma  covariance matrix, shape (n, n)
    rf     risk-free rate
    w      portfolio weights, summing to 1

    portfolio return      R(w) = w' mu
    portfolio volatility  s(w) = sqrt(w' Sigma w)
    Sharpe ratio          S(w) = (w' mu - rf) / s(w)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .estimators import estimate_covariance, expected_returns

_TOL = 1e-7


class OptimizationError(RuntimeError):
    """Raised when no sensible optimum exists for the inputs given."""


@dataclass(frozen=True)
class Constraints:
    """Per-asset weight bounds. Weights always sum to 1 (fully invested)."""

    lower: float = 0.0
    upper: float = 1.0

    @classmethod
    def long_only(cls, cap: float = 1.0) -> "Constraints":
        return cls(0.0, cap)

    @classmethod
    def long_short(cls, cap: float = 1.0) -> "Constraints":
        return cls(-cap, cap)

    @classmethod
    def unconstrained(cls) -> "Constraints":
        return cls(-np.inf, np.inf)

    @property
    def is_unconstrained(self) -> bool:
        return np.isinf(self.lower) and np.isinf(self.upper)

    @property
    def allows_shorts(self) -> bool:
        return self.lower < 0

    def bounds(self, n: int) -> list[tuple[float | None, float | None]]:
        lo = None if np.isinf(self.lower) else self.lower
        hi = None if np.isinf(self.upper) else self.upper
        return [(lo, hi)] * n

    def check_feasible(self, n: int) -> None:
        if self.upper * n < 1 - 1e-9:
            raise OptimizationError(
                f"A {self.upper:.0%} cap on {n} assets can hold at most {self.upper * n:.0%} "
                f"of the portfolio. Raise the cap to at least {1 / n:.0%}."
            )
        if self.lower * n > 1 + 1e-9:
            raise OptimizationError("The minimum weight is too high for the weights to sum to 100%.")

    def satisfied_by(self, w: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(
            abs(w.sum() - 1) < tol
            and np.all(w >= self.lower - tol)
            and np.all(w <= self.upper + tol)
        )

    def describe(self) -> str:
        if self.is_unconstrained:
            return "Unconstrained (short selling allowed, no caps)"
        if self.allows_shorts:
            return f"Long/short, each weight between −{self.upper:.0%} and {self.upper:.0%}"
        return f"Long-only, each weight at most {self.upper:.0%}"


@dataclass
class Portfolio:
    name: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


@dataclass
class OptimizationResult:
    mu: pd.Series
    cov: pd.DataFrame
    risk_free_rate: float
    constraints: Constraints
    covariance_method: str
    shrinkage: float | None
    max_sharpe: Portfolio
    min_variance: Portfolio
    equal_weight: Portfolio
    frontier: pd.DataFrame
    method: str
    warnings: list[str] = field(default_factory=list)

    @property
    def assets(self) -> list[str]:
        return list(self.mu.index)

    def weights_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Max Sharpe": self.max_sharpe.weights,
                "Min variance": self.min_variance.weights,
                "Equal weight": self.equal_weight.weights,
            }
        )


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def portfolio_volatility(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(max(w @ sigma @ w, 0.0)))


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    vol = portfolio_volatility(w, sigma)
    return (portfolio_return(w, mu) - rf) / vol if vol > 0 else np.nan


def make_portfolio(name: str, w: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> Portfolio:
    m, s = mu.to_numpy(), cov.to_numpy()
    return Portfolio(
        name=name,
        weights=pd.Series(w, index=mu.index, name=name),
        expected_return=portfolio_return(w, m),
        volatility=portfolio_volatility(w, s),
        sharpe=sharpe_ratio(w, m, s, rf),
    )


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Share of portfolio variance contributed by each asset.

    RC_i = w_i (Sigma w)_i / (w' Sigma w). These sum to 1 (i.e. 100%).
    """
    w = weights.to_numpy()
    sigma_w = cov.to_numpy() @ w
    total = float(w @ sigma_w)
    return pd.Series(w * sigma_w / total, index=weights.index, name="Risk contribution")


# ---------------------------------------------------------------------------
# Closed-form solutions (no bounds on individual weights)
# ---------------------------------------------------------------------------

def tangency_closed_form(mu: np.ndarray, sigma: np.ndarray, rf: float) -> np.ndarray:
    """w* = Sigma^-1 (mu - rf 1) / 1' Sigma^-1 (mu - rf 1)."""
    z = np.linalg.solve(sigma, mu - rf)
    denom = z.sum()
    if denom <= 0:
        gmv_ret = float(mu @ gmv_closed_form(sigma))
        raise OptimizationError(
            f"The risk-free rate ({rf:.2%}) is at or above the return of the minimum-variance "
            f"portfolio ({gmv_ret:.2%}), so with unlimited short selling no tangency portfolio "
            f"exists: the Sharpe ratio keeps rising as leverage grows. Lower the risk-free rate "
            f"or cap the weights."
        )
    return z / denom


def gmv_closed_form(sigma: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio: Sigma^-1 1 / 1' Sigma^-1 1."""
    z = np.linalg.solve(sigma, np.ones(len(sigma)))
    return z / z.sum()


# ---------------------------------------------------------------------------
# Numerical solutions (bounded weights)
# ---------------------------------------------------------------------------

def _max_feasible_excess(excess: np.ndarray, cons: Constraints) -> tuple[float, np.ndarray]:
    """Largest achievable w'(mu - rf) under the constraints (a small LP)."""
    n = len(excess)
    res = linprog(-excess, A_eq=np.ones((1, n)), b_eq=[1.0], bounds=cons.bounds(n), method="highs")
    if not res.success:
        raise OptimizationError(f"Could not find a feasible portfolio: {res.message}")
    return float(-res.fun), res.x


def _max_sharpe_convex(excess: np.ndarray, sigma: np.ndarray, cons: Constraints, w_start: np.ndarray) -> np.ndarray | None:
    """Solve max Sharpe as a convex QP via the substitution y = w / (w' excess).

        minimize   y' Sigma y
        subject to excess' y = 1
                   sum(y) = k >= 0
                   lower * k <= y_i <= upper * k

    then w = y / sum(y). Because it is convex, the local optimum SLSQP finds
    is the global one.
    """
    n = len(excess)
    rows = [np.ones(n)]                                  # k >= 0
    if np.isfinite(cons.lower):
        rows.extend(np.eye(n) - cons.lower * np.ones((n, n)))   # y_i - lower*k >= 0
    if np.isfinite(cons.upper):
        rows.extend(cons.upper * np.ones((n, n)) - np.eye(n))   # upper*k - y_i >= 0
    a = np.vstack(rows)

    y0 = w_start / float(w_start @ excess)
    res = minimize(
        lambda y: y @ sigma @ y,
        y0,
        jac=lambda y: 2 * sigma @ y,
        method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda y: excess @ y - 1.0, "jac": lambda y: excess},
            {"type": "ineq", "fun": lambda y: a @ y, "jac": lambda y: a},
        ],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    k = res.x.sum()
    if not res.success or k <= 1e-12:
        return None
    return res.x / k


def _max_sharpe_direct(mu: np.ndarray, sigma: np.ndarray, rf: float, cons: Constraints, w_start: np.ndarray) -> np.ndarray | None:
    """Fallback: maximize the Sharpe ratio directly with SLSQP."""

    def neg_sharpe(w):
        return -sharpe_ratio(w, mu, sigma, rf)

    def neg_sharpe_grad(w):
        vol = portfolio_volatility(w, sigma)
        ex = w @ mu - rf
        return -(mu * vol - ex * (sigma @ w) / vol) / vol**2

    res = minimize(
        neg_sharpe,
        w_start,
        jac=neg_sharpe_grad,
        method="SLSQP",
        bounds=cons.bounds(len(mu)),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x if res.success else None


def _tidy(w: np.ndarray, cons: Constraints) -> np.ndarray:
    """Remove numerical dust (e.g. 1e-17 weights) and renormalize."""
    w = np.where(np.abs(w) < 1e-9, 0.0, w)
    if not cons.is_unconstrained:
        w = np.clip(w, cons.lower, cons.upper)
    return w / w.sum()


def max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame, rf: float, cons: Constraints) -> tuple[np.ndarray, str]:
    """Weights of the maximum-Sharpe portfolio and a note on how they were found."""
    m, s = mu.to_numpy(float), cov.to_numpy(float)
    n = len(m)
    if cons.is_unconstrained:
        return tangency_closed_form(m, s, rf), "closed form"

    cons.check_feasible(n)
    excess = m - rf
    best_excess, w_lp = _max_feasible_excess(excess, cons)
    if best_excess <= _TOL:
        raise OptimizationError(
            f"No allowed portfolio is expected to beat the risk-free rate of {rf:.2%} "
            f"(the best achievable is {best_excess + rf:.2%}), so the Sharpe ratio cannot be "
            f"positive. Lower the risk-free rate or add assets with higher returns."
        )

    # A start point that is feasible and has positive excess return:
    # blend equal weight with the LP solution until excess return is positive.
    w_eq = np.full(n, 1.0 / n)
    w_start = w_eq if cons.satisfied_by(w_eq) and w_eq @ excess > 0 else 0.5 * w_eq + 0.5 * w_lp
    if not (cons.satisfied_by(w_start) and w_start @ excess > 0):
        w_start = w_lp

    candidates: list[tuple[np.ndarray, str]] = []
    w_qp = _max_sharpe_convex(excess, s, cons, w_start)
    if w_qp is not None:
        candidates.append((w_qp, "convex quadratic program (SLSQP)"))
    for start in (w_start, w_lp):
        w_d = _max_sharpe_direct(m, s, rf, cons, start)
        if w_d is not None:
            candidates.append((w_d, "direct Sharpe maximization (SLSQP)"))

    feasible = [(w, how) for w, how in candidates if cons.satisfied_by(w, tol=1e-5)]
    if not feasible:
        raise OptimizationError("The optimizer did not converge. Try fewer assets or looser weight caps.")
    scores = [sharpe_ratio(w, m, s, rf) for w, _ in feasible]
    best = max(scores)
    # Prefer the convex solution unless a cross-check found something genuinely better.
    w_best, how = next(c for c, sc in zip(feasible, scores) if sc >= best - 1e-9)
    return _tidy(w_best, cons), how


def min_variance_weights(cov: pd.DataFrame, cons: Constraints, mu: pd.Series | None = None, target: float | None = None,
                         w_start: np.ndarray | None = None) -> np.ndarray | None:
    """Minimum-variance weights, optionally for a target expected return."""
    s = cov.to_numpy(float)
    n = len(s)
    if cons.is_unconstrained and target is None:
        return gmv_closed_form(s)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}]
    if target is not None:
        m = mu.to_numpy(float)
        constraints.append({"type": "eq", "fun": lambda w: w @ m - target, "jac": lambda w: m})
    res = minimize(
        lambda w: w @ s @ w,
        np.full(n, 1.0 / n) if w_start is None else w_start,
        jac=lambda w: 2 * s @ w,
        method="SLSQP",
        bounds=cons.bounds(n),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    if not res.success or not cons.satisfied_by(res.x, tol=1e-5):
        return None
    return _tidy(res.x, cons)


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, cons: Constraints, rf: float,
                       w_gmv: np.ndarray, w_tan: np.ndarray, points: int = 50) -> pd.DataFrame:
    """Upper (efficient) branch of the minimum-variance frontier."""
    m, s = mu.to_numpy(float), cov.to_numpy(float)
    r_low = float(w_gmv @ m)

    if cons.is_unconstrained:
        # Analytic hyperbola: var(r) = (A r^2 - 2 B r + C) / (A C - B^2)
        inv = np.linalg.inv(s)
        one = np.ones(len(m))
        a, b, c = one @ inv @ one, one @ inv @ m, m @ inv @ m
        r_high = max(float(m.max()), float(w_tan @ m)) * 1.25
        targets = np.linspace(r_low, r_high, points)
        var = (a * targets**2 - 2 * b * targets + c) / (a * c - b**2)
        vols = np.sqrt(np.maximum(var, 0))
    else:
        r_high = -linprog(-m, A_eq=np.ones((1, len(m))), b_eq=[1.0], bounds=cons.bounds(len(m)), method="highs").fun
        targets = np.linspace(r_low, r_high - 1e-6 * max(1.0, abs(r_high)), points)
        vols, kept = [], []
        w_prev = w_gmv
        for t in targets:
            w = min_variance_weights(cov, cons, mu, target=float(t), w_start=w_prev)
            if w is None:
                continue
            w_prev = w
            kept.append(t)
            vols.append(portfolio_volatility(w, s))
        targets, vols = np.array(kept), np.array(vols)

    frame = pd.DataFrame({"return": targets, "volatility": vols})
    frame["sharpe"] = (frame["return"] - rf) / frame["volatility"]
    return frame


def random_portfolios(mu: pd.Series, cov: pd.DataFrame, rf: float, n: int = 2500, seed: int = 7) -> pd.DataFrame:
    """Random long-only portfolios (uniform on the simplex) for context."""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(mu)), size=n)
    m, s = mu.to_numpy(float), cov.to_numpy(float)
    rets = w @ m
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, s, w))
    return pd.DataFrame({"return": rets, "volatility": vols, "sharpe": (rets - rf) / vols})


# ---------------------------------------------------------------------------
# One call that does everything the app needs
# ---------------------------------------------------------------------------

def optimize(
    returns: pd.DataFrame,
    periods_per_year: int,
    risk_free_rate: float = 0.04,
    covariance_method: str = "sample",
    constraints: Constraints = Constraints.long_only(),
    frontier_points: int = 50,
) -> OptimizationResult:
    """Estimate mu and Sigma from periodic returns and find the optimal portfolios."""
    if returns.shape[1] < 2:
        raise OptimizationError("Add at least two assets: with one asset there is nothing to optimize.")
    if len(returns) < returns.shape[1] + 2:
        raise OptimizationError(
            f"Only {len(returns)} return observations for {returns.shape[1]} assets. "
            f"You need more observations than assets for a usable covariance matrix."
        )

    warnings: list[str] = []
    mu = expected_returns(returns, periods_per_year)
    estimate = estimate_covariance(returns, periods_per_year, covariance_method)
    cov = estimate.matrix

    cond = np.linalg.cond(cov.to_numpy())
    if cond > 1e8:
        warnings.append(
            "The covariance matrix is nearly singular (some assets move almost identically). "
            "Weights may be unstable; Ledoit–Wolf shrinkage usually helps."
        )
    if len(returns) < 10 * returns.shape[1]:
        warnings.append(
            f"{len(returns)} observations for {returns.shape[1]} assets is thin. "
            "Estimates are noisy, so consider a longer history or Ledoit–Wolf shrinkage."
        )

    w_tan, method = max_sharpe_weights(mu, cov, risk_free_rate, constraints)
    w_gmv = min_variance_weights(cov, constraints)
    if w_gmv is None:
        raise OptimizationError("Could not compute the minimum-variance portfolio.")
    n = len(mu)
    w_eq = np.full(n, 1.0 / n)

    frontier = efficient_frontier(mu, cov, constraints, risk_free_rate, w_gmv, w_tan, frontier_points)

    return OptimizationResult(
        mu=mu,
        cov=cov,
        risk_free_rate=risk_free_rate,
        constraints=constraints,
        covariance_method=covariance_method,
        shrinkage=estimate.shrinkage,
        max_sharpe=make_portfolio("Max Sharpe", w_tan, mu, cov, risk_free_rate),
        min_variance=make_portfolio("Min variance", w_gmv, mu, cov, risk_free_rate),
        equal_weight=make_portfolio("Equal weight", w_eq, mu, cov, risk_free_rate),
        frontier=frontier,
        method=method,
        warnings=warnings,
    )
