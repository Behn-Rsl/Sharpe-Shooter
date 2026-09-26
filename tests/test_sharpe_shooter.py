"""Run with:  pytest -q"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sharpe_shooter import providers
from sharpe_shooter.demo import DEMOS, simulate_prices
from sharpe_shooter.estimators import (
    clean_prices,
    infer_periods_per_year,
    ledoit_wolf_covariance,
    prices_to_returns,
    resample_prices,
    sample_covariance,
    expected_returns,
)
from sharpe_shooter.optimizer import (
    Constraints,
    OptimizationError,
    max_sharpe_weights,
    optimize,
    risk_contributions,
    sharpe_ratio,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(simulate_prices(DEMOS["global_etfs"]))


@pytest.fixture(scope="module")
def inputs(returns):
    return expected_returns(returns, 252), sample_covariance(returns, 252)


def brute_force_best_sharpe(mu, cov, rf, cap=1.0, samples=200_000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.full(len(mu), 0.4), size=samples)
    w = w[(w <= cap).all(axis=1)]
    rets = w @ mu.to_numpy()
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov.to_numpy(), w))
    return float(((rets - rf) / vols).max())


def test_closed_form_matches_numerical_when_caps_do_not_bind(inputs):
    mu, cov = inputs
    w_closed, _ = max_sharpe_weights(mu, cov, 0.02, Constraints.unconstrained())
    w_numeric, _ = max_sharpe_weights(mu, cov, 0.02, Constraints.long_short(50))
    np.testing.assert_allclose(w_closed, w_numeric, atol=1e-4)


def test_long_only_beats_every_random_portfolio(inputs):
    mu, cov = inputs
    w, _ = max_sharpe_weights(mu, cov, 0.03, Constraints.long_only())
    assert np.all(w >= -1e-9) and abs(w.sum() - 1) < 1e-9
    best_random = brute_force_best_sharpe(mu, cov, 0.03)
    assert sharpe_ratio(w, mu.to_numpy(), cov.to_numpy(), 0.03) >= best_random - 1e-6


def test_cap_is_respected_and_optimal(inputs):
    mu, cov = inputs
    w, _ = max_sharpe_weights(mu, cov, 0.03, Constraints.long_only(0.3))
    assert w.max() <= 0.3 + 1e-6
    best_random = brute_force_best_sharpe(mu, cov, 0.03, cap=0.3)
    assert sharpe_ratio(w, mu.to_numpy(), cov.to_numpy(), 0.03) >= best_random - 1e-6


def test_tangency_is_the_steepest_point_on_the_frontier(returns):
    result = optimize(returns, 252, 0.03, constraints=Constraints.long_only())
    assert result.max_sharpe.sharpe >= result.frontier["sharpe"].max() - 1e-4
    assert result.max_sharpe.sharpe >= result.equal_weight.sharpe
    assert result.min_variance.volatility <= result.frontier["volatility"].min() + 1e-6


def test_risk_contributions_sum_to_one(returns):
    result = optimize(returns, 252, 0.03)
    assert risk_contributions(result.max_sharpe.weights, result.cov).sum() == pytest.approx(1.0)


def test_rf_above_every_return_is_reported(inputs):
    mu, cov = inputs
    with pytest.raises(OptimizationError, match="beat the risk-free rate"):
        max_sharpe_weights(mu, cov, 0.50, Constraints.long_only())


def test_rf_above_gmv_return_is_reported_when_unconstrained(inputs):
    mu, cov = inputs
    with pytest.raises(OptimizationError, match="no tangency portfolio"):
        max_sharpe_weights(mu, cov, 0.30, Constraints.unconstrained())


def test_infeasible_cap_is_reported(inputs):
    mu, cov = inputs
    with pytest.raises(OptimizationError, match="Raise the cap"):
        max_sharpe_weights(mu, cov, 0.03, Constraints.long_only(0.1))


def test_ledoit_wolf_matches_scikit_learn(returns):
    sklearn = pytest.importorskip("sklearn.covariance")
    expected, expected_shrinkage = sklearn.ledoit_wolf(returns.to_numpy())
    cov, shrinkage = ledoit_wolf_covariance(returns, 1)
    assert shrinkage == pytest.approx(expected_shrinkage)
    np.testing.assert_allclose(cov.to_numpy(), expected, rtol=1e-10, atol=1e-14)


def test_frequency_inference_and_resampling():
    prices = simulate_prices(DEMOS["big_tech"], years=2)
    assert infer_periods_per_year(prices.index) == 252
    weekly, periods, label, _ = resample_prices(prices, "Weekly")
    assert periods == 52 and label == "weekly" and infer_periods_per_year(weekly.index) == 52
    # Asking for daily from weekly data keeps it weekly and says so.
    same, periods, label, notes = resample_prices(weekly, "Daily")
    assert periods == 52 and notes


def test_clean_prices_aligns_short_histories():
    idx = pd.bdate_range("2020-01-01", periods=400)
    prices = pd.DataFrame({"A": np.linspace(100, 120, 400), "B": np.linspace(50, 60, 400)}, index=idx)
    prices.iloc[:200, 1] = np.nan
    cleaned, notes = clean_prices(prices)
    assert len(cleaned) == 200 and any("only has data from" in n for n in notes)


def test_parse_tickers():
    assert providers.parse_tickers("aapl, msft  VOD.L;aapl") == ["AAPL", "MSFT", "VOD.L"]


def test_load_csv_prices_and_returns(tmp_path):
    frame = pd.DataFrame({"Date": ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"],
                          "AAA": [0.01, 0.02, -0.01, 0.03], "BBB": [0.00, 0.01, 0.02, -0.02]})
    path = tmp_path / "r.csv"
    frame.to_csv(path, index=False)
    loaded = providers.load_csv(path.read_bytes(), "returns")
    assert list(loaded.prices.columns) == ["AAA", "BBB"]
    rebuilt = prices_to_returns(loaded.prices)
    np.testing.assert_allclose(rebuilt["AAA"].to_numpy(), frame["AAA"].to_numpy())


# -- providers with the network mocked out -----------------------------------------------

class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise providers.requests.HTTPError(str(self.status_code))


def test_alpha_vantage_parses_adjusted_closes(monkeypatch):
    def fake_get(url, params, timeout):
        return FakeResponse({"Weekly Adjusted Time Series": {
            "2024-01-12": {"5. adjusted close": "101.0"},
            "2024-01-05": {"5. adjusted close": "100.0"},
        }})
    monkeypatch.setattr(providers.requests, "get", fake_get)
    out = providers.AlphaVantageProvider().fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 2, 1), "key")
    assert out.prices.shape == (2, 2) and out.prices["AAA"].iloc[-1] == 101.0


def test_alpha_vantage_rate_limit_stops_the_fetch(monkeypatch):
    monkeypatch.setattr(providers.requests, "get",
                        lambda *a, **k: FakeResponse({"Information": "25 requests per day"}))
    with pytest.raises(providers.DataProviderError, match="refused"):
        providers.AlphaVantageProvider().fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 2, 1), "key")


def test_tiingo_parses_and_skips_unknown_symbols(monkeypatch):
    def fake_get(url, params, headers, timeout):
        if "zzzz" in url:
            return FakeResponse({"detail": "Not found"}, status=404)
        return FakeResponse([{"date": "2024-01-02T00:00:00.000Z", "adjClose": 10.0},
                             {"date": "2024-01-03T00:00:00.000Z", "adjClose": 10.5}])
    monkeypatch.setattr(providers.requests, "get", fake_get)
    out = providers.TiingoProvider().fetch(["AAA", "BBB", "ZZZZ"], date(2024, 1, 1), date(2024, 1, 5), "key")
    assert list(out.prices.columns) == ["AAA", "BBB"] and any("ZZZZ" in w for w in out.warnings)


def test_yahoo_handles_multiindex_columns(monkeypatch):
    yf = pytest.importorskip("yfinance")
    idx = pd.bdate_range("2024-01-01", periods=5)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAA", "BBB"]], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4) + 1, index=idx, columns=cols)
    monkeypatch.setattr(yf, "download", lambda **kwargs: raw)
    out = providers.YahooFinanceProvider().fetch(["AAA", "BBB", "CCC"], date(2024, 1, 1), date(2024, 1, 8))
    assert list(out.prices.columns) == ["AAA", "BBB"] and any("CCC" in w for w in out.warnings)
