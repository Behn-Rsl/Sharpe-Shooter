"""Where prices come from when you build your own portfolio.

Every provider turns a list of tickers and a date range into one DataFrame
of adjusted closing prices: a DatetimeIndex and one column per ticker.
To add another data source, subclass PriceProvider, implement
fetch_one(), and add an instance to PROVIDERS at the bottom of this file.
"""

from __future__ import annotations

import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import requests


class DataProviderError(RuntimeError):
    """A readable explanation of why prices could not be fetched."""


@dataclass
class FetchResult:
    prices: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def parse_tickers(text: str) -> list[str]:
    """Split 'aapl, msft  VOD.L' into ['AAPL', 'MSFT', 'VOD.L'], dropping repeats."""
    seen: list[str] = []
    for token in re.split(r"[\s,;]+", text.strip()):
        t = token.strip().upper()
        if t and t not in seen:
            seen.append(t)
    return seen


class PriceProvider(ABC):
    """Base class. Subclasses fetch one ticker at a time unless they override fetch()."""

    name: str = ""
    needs_api_key: bool = False
    key_env_var: str | None = None
    signup_url: str | None = None
    notes: str = ""
    options: dict[str, list[str]] = {}

    def fetch(self, tickers: list[str], start: date, end: date, api_key: str | None = None,
              **options) -> FetchResult:
        if self.needs_api_key and not api_key:
            raise DataProviderError(f"{self.name} needs an API key. Get a free one at {self.signup_url}.")
        series, failures = {}, []
        for ticker in tickers:
            try:
                s = self.fetch_one(ticker, start, end, api_key, **options)
                if s is None or s.dropna().empty:
                    failures.append(f"{ticker}: no prices in that date range")
                else:
                    series[ticker] = s
            except DataProviderError as exc:
                if getattr(exc, "fatal", False):
                    raise
                failures.append(f"{ticker}: {exc}")
        return _assemble(series, failures, self.name)

    @abstractmethod
    def fetch_one(self, ticker: str, start: date, end: date, api_key: str | None, **options) -> pd.Series:
        ...


def _fatal(message: str) -> DataProviderError:
    """An error that should stop the whole fetch (bad key, rate limit)."""
    exc = DataProviderError(message)
    exc.fatal = True  # type: ignore[attr-defined]
    return exc


def _assemble(series: dict[str, pd.Series], failures: list[str], source: str) -> FetchResult:
    if not series:
        detail = "; ".join(failures) if failures else "no data returned"
        raise DataProviderError(f"{source} returned no usable prices ({detail}).")
    prices = pd.concat(series, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)
    warnings = [f"Skipped {f}." for f in failures]
    return FetchResult(prices=prices, warnings=warnings)


# ---------------------------------------------------------------------------
# Yahoo Finance via yfinance (no key needed)
# ---------------------------------------------------------------------------

class YahooFinanceProvider(PriceProvider):
    name = "Yahoo Finance"
    needs_api_key = False
    notes = ("Free, no key. Uses split- and dividend-adjusted closes. Use Yahoo's symbols: "
             "BRK-B rather than BRK.B, and exchange suffixes such as VOD.L (London) or SAP.DE (Xetra).")

    def fetch(self, tickers, start, end, api_key=None, **options) -> FetchResult:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise DataProviderError("Install yfinance to use Yahoo Finance: pip install yfinance") from exc

        try:
            raw = yf.download(
                tickers=list(tickers),
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # yfinance's end date is exclusive
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:  # yfinance raises a variety of network errors
            raise DataProviderError(f"Could not reach Yahoo Finance: {exc}") from exc

        if raw is None or raw.empty:
            raise DataProviderError(
                "Yahoo Finance returned no data. Check the symbols and your internet connection."
            )

        if isinstance(raw.columns, pd.MultiIndex):
            fields = raw.columns.get_level_values(0)
            close = raw["Close"] if "Close" in fields else raw["Adj Close"]
        else:  # older yfinance with a single ticker
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        close.columns = [str(c).upper() for c in close.columns]

        series = {t: close[t].dropna() for t in tickers if t in close.columns and close[t].notna().any()}
        failures = [f"{t}: Yahoo Finance has no prices for this symbol" for t in tickers if t not in series]
        return _assemble(series, failures, self.name)

    def fetch_one(self, ticker, start, end, api_key, **options):  # pragma: no cover - fetch() is overridden
        return self.fetch([ticker], start, end).prices[ticker]


# ---------------------------------------------------------------------------
# Alpha Vantage (free key; the free tier includes weekly and monthly adjusted series)
# ---------------------------------------------------------------------------

class AlphaVantageProvider(PriceProvider):
    name = "Alpha Vantage"
    needs_api_key = True
    key_env_var = "ALPHAVANTAGE_API_KEY"
    signup_url = "https://www.alphavantage.co/support/#api-key"
    notes = ("Free key. The free tier allows about 25 requests a day (one per ticker), and its "
             "adjusted series are weekly or monthly, so daily returns are not available here.")
    options = {"interval": ["Weekly", "Monthly"]}

    _SERIES = {
        "Weekly": ("TIME_SERIES_WEEKLY_ADJUSTED", "Weekly Adjusted Time Series"),
        "Monthly": ("TIME_SERIES_MONTHLY_ADJUSTED", "Monthly Adjusted Time Series"),
    }
    URL = "https://www.alphavantage.co/query"

    def fetch_one(self, ticker, start, end, api_key, interval: str = "Weekly", **options):
        function, key = self._SERIES[interval]
        try:
            resp = requests.get(self.URL, params={"function": function, "symbol": ticker, "apikey": api_key},
                                timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataProviderError(f"request failed ({exc})") from exc

        if "Error Message" in payload:
            raise DataProviderError("Alpha Vantage does not recognise this symbol")
        for limit_key in ("Note", "Information"):
            if limit_key in payload:
                raise _fatal(f"Alpha Vantage refused the request: {payload[limit_key]}")
        data = payload.get(key)
        if not data:
            raise DataProviderError("unexpected response from Alpha Vantage")

        s = pd.Series({pd.Timestamp(d): float(v["5. adjusted close"]) for d, v in data.items()}).sort_index()
        return s.loc[pd.Timestamp(start):pd.Timestamp(end)]


# ---------------------------------------------------------------------------
# Tiingo (free key; daily adjusted end-of-day prices)
# ---------------------------------------------------------------------------

class TiingoProvider(PriceProvider):
    name = "Tiingo"
    needs_api_key = True
    key_env_var = "TIINGO_API_KEY"
    signup_url = "https://www.tiingo.com/account/api/token"
    notes = "Free key. Daily split- and dividend-adjusted prices for US stocks, ETFs and mutual funds."

    URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

    def fetch_one(self, ticker, start, end, api_key, **options):
        try:
            resp = requests.get(
                self.URL.format(ticker=ticker.lower()),
                params={"startDate": start.isoformat(), "endDate": end.isoformat(), "token": api_key},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise DataProviderError(f"request failed ({exc})") from exc

        if resp.status_code in (401, 403):
            raise _fatal("Tiingo rejected the API key. Check it on your Tiingo account page.")
        if resp.status_code == 429:
            raise _fatal("Tiingo's rate limit was reached. Wait an hour or reduce the number of tickers.")
        if resp.status_code == 404:
            raise DataProviderError("Tiingo does not recognise this symbol")
        try:
            resp.raise_for_status()
            rows = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataProviderError(f"unexpected response ({exc})") from exc
        if isinstance(rows, dict):  # Tiingo reports some errors as {"detail": "..."}
            raise DataProviderError(rows.get("detail", "unexpected response from Tiingo"))
        if not rows:
            return pd.Series(dtype=float)

        frame = pd.DataFrame(rows)
        idx = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        return pd.Series(frame["adjClose"].to_numpy(dtype=float), index=idx).sort_index()


# ---------------------------------------------------------------------------
# CSV upload
# ---------------------------------------------------------------------------

def load_csv(file_or_buffer, contents: str = "prices") -> FetchResult:
    """Read a wide CSV: first column dates, one column per ticker.

    contents="prices"  -> values are prices (any currency, adjusted closes ideally)
    contents="returns" -> values are periodic returns as decimals (0.01 = 1%)
    """
    if isinstance(file_or_buffer, (bytes, bytearray)):
        file_or_buffer = io.BytesIO(file_or_buffer)
    try:
        frame = pd.read_csv(file_or_buffer)
    except Exception as exc:
        raise DataProviderError(f"Could not read the CSV: {exc}") from exc
    if frame.shape[1] < 3:
        raise DataProviderError("The CSV needs a date column followed by at least two ticker columns.")

    date_col = frame.columns[0]
    dates = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    if dates.isna().mean() > 0.5:
        raise DataProviderError(f"The first column ('{date_col}') should contain dates, e.g. 2024-01-31.")
    values = frame.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    values.index = dates
    values = values[values.index.notna()].sort_index()
    values.columns = [str(c).strip().upper() for c in values.columns]

    warnings: list[str] = []
    if contents == "returns":
        values = values.dropna(how="all")
        if values.abs().median().median() > 0.5:
            values = values / 100
            warnings.append("The returns looked like percentages, so they were divided by 100.")
        # Rebuild a price index so the rest of the pipeline is identical.
        first = values.index[0] - (values.index[1] - values.index[0])
        prices = (1 + values.fillna(0)).cumprod() * 100
        prices.loc[first] = 100.0
        values = prices.sort_index()

    return FetchResult(prices=values, warnings=warnings)


PROVIDERS: dict[str, PriceProvider] = {
    p.name: p for p in (YahooFinanceProvider(), AlphaVantageProvider(), TiingoProvider())
}
