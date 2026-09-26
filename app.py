"""Sharpe Shooter: find the portfolio weights that maximize the Sharpe ratio.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import math
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from sharpe_shooter import charts
from sharpe_shooter.demo import DEMOS, asset_names, simulate_prices
from sharpe_shooter.estimators import (
    COVARIANCE_METHODS,
    FREQUENCIES,
    clean_prices,
    correlation_from_covariance,
    prices_to_returns,
    resample_prices,
)
from sharpe_shooter.optimizer import Constraints, OptimizationError, optimize, random_portfolios
from sharpe_shooter.performance import asset_table, growth_of_one, performance_table
from sharpe_shooter.providers import PROVIDERS, DataProviderError, load_csv, parse_tickers

ROOT = Path(__file__).parent
SAMPLE_CSV = ROOT / "examples" / "sample_prices.csv"
CSV_OPTION = "Upload a CSV"
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

st.set_page_config(page_title="Sharpe Shooter", page_icon="🎯", layout="wide")

st.html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap');

[data-testid="stMainBlockContainer"], .block-container { padding-top: 3.6rem; max-width: 1400px; }

.ss-mark { display: flex; align-items: center; gap: .55rem; margin: 0 0 .15rem; }
.ss-mark span { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700;
                font-size: 1.85rem; line-height: 1; color: #1D2B36; letter-spacing: .005em; }
.ss-tagline { color: #5B6B77; font-size: .92rem; margin: 0 0 .4rem; }

.ss-head { display: flex; align-items: baseline; flex-wrap: wrap; gap: .4rem 1rem; }
.ss-head h1 { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 600;
              font-size: 2.6rem; line-height: 1.05; margin: 0; padding: 0; color: #1D2B36; }
.ss-badge { font-size: .82rem; color: #1D2B36; background: #FFFFFF; border: 1px solid #CBD4D8;
            border-radius: 999px; padding: .12rem .6rem; white-space: nowrap; }
.ss-badge.sim { border-style: dashed; }
.ss-lede { color: #33434F; font-size: 1.02rem; max-width: 72ch; margin: .45rem 0 .15rem; line-height: 1.5; }
.ss-meta { color: #5B6B77; font-size: .9rem; margin: 0; font-variant-numeric: tabular-nums; }

.ss-readout { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; margin-top: .4rem; }
.ss-readout th { font-weight: 500; color: #5B6B77; font-size: .85rem; text-align: right; padding: 0 0 .5rem .6rem;
                 vertical-align: bottom; line-height: 1.2; }
.ss-readout th:first-child { text-align: left; padding-left: 0; }
.ss-readout th i { display: block; width: .7rem; height: .7rem; border-radius: 50%; margin: 0 0 .3rem auto; }
.ss-readout td { text-align: right; padding: .5rem 0 .5rem .6rem; border-top: 1px solid #CBD4D8; font-size: 1rem;
                 color: #33434F; }
.ss-readout td:first-child { text-align: left; padding-left: 0; color: #1D2B36; white-space: nowrap; }
.ss-readout td.opt { color: #C8372D; font-weight: 600; }
.ss-readout tr.sharpe td { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; 
                           font-size: 1.35rem; font-weight: 600; padding-top: .35rem; padding-bottom: .35rem; }
.ss-readout tr.sharpe td:first-child { font-family: 'Barlow', sans-serif; font-size: 1rem; font-weight: 400; }
.ss-readout tr.sharpe td.opt { font-size: 2.1rem; }

.ss-positions { margin: 1.1rem 0 0; padding: .8rem 0 0; border-top: 1px solid #CBD4D8; }
.ss-positions h4 { font-family: 'Barlow', sans-serif; font-size: .9rem; font-weight: 500; color: #5B6B77; margin: 0 0 .35rem; padding: 0; }
.ss-positions ol { margin: 0; padding: 0; list-style: none; }
.ss-positions li { display: flex; justify-content: space-between; padding: .18rem 0; font-variant-numeric: tabular-nums; }
.ss-positions li b { font-weight: 600; }
.ss-settings { color: #5B6B77; font-size: .86rem; line-height: 1.55; margin: 1rem 0 0; }

.ss-empty { background: #FFFFFF; border: 1px solid #CBD4D8; border-radius: .25rem; padding: 1.4rem 1.6rem; max-width: 62ch; }
.ss-empty h2 { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.7rem; margin: 0 0 .5rem; padding: 0; }
.ss-empty p, .ss-empty li { color: #33434F; line-height: 1.55; }

@media (max-width: 640px) { .ss-head h1 { font-size: 2rem; } .ss-readout tr.sharpe td.opt { font-size: 1.9rem; } }
</style>
"""
)

WORDMARK = """
<div class="ss-mark">
  <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true">
    <circle cx="15" cy="15" r="9.5" fill="none" stroke="#C8372D" stroke-width="2"/>
    <path d="M15 1.5v8M15 20.5v8M1.5 15h8M20.5 15h8" stroke="#C8372D" stroke-width="1.6"/>
    <circle cx="15" cy="15" r="1.7" fill="#C8372D"/>
  </svg>
  <span>Sharpe Shooter</span>
</div>
<p class="ss-tagline">Find the weights that give your portfolio the highest Sharpe ratio.</p>
"""


# ---------------------------------------------------------------------------
# Cached data and computation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def demo_prices(key: str) -> pd.DataFrame:
    return simulate_prices(DEMOS[key])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(provider: str, tickers: tuple[str, ...], start: date, end: date,
                 api_key: str | None, options: tuple[tuple[str, str], ...]):
    return PROVIDERS[provider].fetch(list(tickers), start, end, api_key, **dict(options))


@st.cache_data(show_spinner=False)
def run_optimization(returns: pd.DataFrame, periods: int, rf: float, cov_method: str, lower: float, upper: float):
    return optimize(returns, periods, rf, cov_method, Constraints(lower, upper))


@st.cache_data(show_spinner=False)
def cloud_for(mu: pd.Series, cov: pd.DataFrame, rf: float) -> pd.DataFrame:
    return random_portfolios(mu, cov, rf)


def stored_api_key(provider) -> str:
    """API key from Streamlit secrets or an environment variable, if either is set."""
    if not provider.key_env_var:
        return ""
    try:
        if provider.key_env_var in st.secrets:
            return str(st.secrets[provider.key_env_var])
    except Exception:  # no secrets.toml present
        pass
    return os.getenv(provider.key_env_var, "")


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}".replace("-", "−")


# ---------------------------------------------------------------------------
# Sidebar: choose a portfolio
# ---------------------------------------------------------------------------

portfolio: dict | None = None  # {"prices", "title", "badge", "simulated", "description", "names", "notes"}

with st.sidebar:
    st.html(WORDMARK)
    st.subheader("Portfolio")
    source = st.radio("Portfolio source", ["Demo portfolio", "Build your own"], horizontal=True,
                      label_visibility="collapsed")

    if source == "Demo portfolio":
        demo_key = st.selectbox("Demo", list(DEMOS), format_func=lambda k: DEMOS[k].name)
        demo = DEMOS[demo_key]
        live = st.toggle(
            "Use live market data",
            help="Download real prices for these tickers from Yahoo Finance instead of using the "
                 "bundled simulated prices. Needs an internet connection.",
        )
        portfolio = {
            "prices": demo_prices(demo_key),
            "title": demo.name,
            "badge": "Simulated prices",
            "simulated": True,
            "description": demo.description,
            "names": asset_names(demo),
            "notes": [],
        }
        if live:
            years = st.slider("Years of history", 1, 20, 5)
            end = date.today()
            start = end - timedelta(days=int(365.25 * years))
            try:
                with st.spinner("Downloading prices from Yahoo Finance"):
                    fetched = fetch_prices("Yahoo Finance", tuple(demo.tickers), start, end, None, ())
                portfolio.update(prices=fetched.prices, badge="Yahoo Finance", simulated=False,
                                 notes=list(fetched.warnings))
            except DataProviderError as exc:
                st.error(f"{exc} Showing the simulated prices instead.")

    else:
        provider_name = st.selectbox("Data source", [*PROVIDERS, CSV_OPTION])

        if provider_name == CSV_OPTION:
            upload = st.file_uploader("CSV file", type=["csv"],
                                      help="First column: dates. Then one column per ticker.")
            contents = st.radio("The numbers in the file are", ["prices", "returns"], horizontal=True)
            if SAMPLE_CSV.exists():
                st.download_button("Download an example CSV", SAMPLE_CSV.read_bytes(), "sample_prices.csv",
                                   "text/csv", type="tertiary")
            if upload is not None:
                try:
                    loaded = load_csv(upload.getvalue(), contents)
                    st.session_state["custom"] = {
                        "prices": loaded.prices, "title": Path(upload.name).stem.replace("_", " ").title(),
                        "badge": "Uploaded CSV", "notes": loaded.warnings,
                    }
                except DataProviderError as exc:
                    st.error(str(exc))
        else:
            provider = PROVIDERS[provider_name]
            with st.form("fetch"):
                title = st.text_input("Portfolio name", value="My portfolio")
                tickers_text = st.text_area("Tickers", value="AAPL, MSFT, JNJ, JPM, XOM, GLD, TLT",
                                            help="Separate symbols with commas or spaces.")
                c1, c2 = st.columns(2)
                start = c1.date_input("From", value=date.today() - timedelta(days=5 * 365), format="DD/MM/YYYY")
                end = c2.date_input("To", value=date.today(), format="DD/MM/YYYY")
                api_key = None
                if provider.needs_api_key:
                    api_key = st.text_input(f"{provider.name} API key", value=stored_api_key(provider),
                                            type="password",
                                            help=f"Or set the {provider.key_env_var} environment variable.")
                chosen_options = {}
                for option, choices in provider.options.items():
                    chosen_options[option] = st.selectbox(option.capitalize(), choices)
                submitted = st.form_submit_button("Fetch prices", type="primary", width="stretch")
            st.caption(provider.notes)
            if provider.signup_url:
                st.caption(f"[Get a free {provider.name} key]({provider.signup_url})")

            if submitted:
                tickers = parse_tickers(tickers_text)
                if len(tickers) < 2:
                    st.error("Enter at least two tickers.")
                elif start >= end:
                    st.error("The start date must be before the end date.")
                else:
                    try:
                        with st.spinner(f"Downloading prices from {provider.name}"):
                            fetched = fetch_prices(provider_name, tuple(tickers), start, end,
                                                   api_key or None, tuple(sorted(chosen_options.items())))
                        st.session_state["custom"] = {
                            "prices": fetched.prices, "title": title.strip() or "My portfolio",
                            "badge": provider.name, "notes": list(fetched.warnings),
                        }
                    except DataProviderError as exc:
                        st.error(str(exc))

        custom = st.session_state.get("custom")
        if custom is not None:
            portfolio = {**custom, "simulated": False, "description": "", "names": {}}

    # -- optimization settings -------------------------------------------------
    st.subheader("Optimization")
    rf_pct = st.number_input(
        "Risk-free rate (% a year)", min_value=-2.0, max_value=25.0, value=4.0, step=0.25, format="%.2f",
        help="The return on a riskless asset over the same horizon, such as a 3-month Treasury bill "
             "or gilt yield. The Sharpe ratio measures return in excess of this.",
    )
    frequency = st.selectbox(
        "Return frequency", list(FREQUENCIES),
        help="Prices are sampled at this frequency before computing returns. Weekly or monthly returns "
             "are less affected by assets that trade at different times of day.",
    )
    cov_method = st.radio(
        "Covariance estimate", list(COVARIANCE_METHODS), format_func=COVARIANCE_METHODS.get,
        help="Ledoit–Wolf blends the sample covariance with a simple structured target, which "
             "reduces estimation noise and usually gives more stable weights.",
    )
    limits = st.selectbox(
        "Position limits", ["Long-only", "Long and short", "Unconstrained"],
        help="Long-only: no negative weights. Long and short: weights between −cap and +cap. "
             "Unconstrained: any weights, solved analytically.",
    )
    n_assets = portfolio["prices"].shape[1] if portfolio else 2
    if limits == "Unconstrained":
        st.caption("Weights can take any size or sign. The solution is the closed-form tangency portfolio.")
        lower, upper = -math.inf, math.inf
    else:
        floor = min(100, math.ceil(100 / max(n_assets, 1)))
        cap = st.slider("Largest position per asset (%)", min_value=floor, max_value=100, value=100, step=1,
                        help=f"With {n_assets} assets the cap must be at least {floor}% so the weights can reach 100%.")
        lower, upper = (0.0, cap / 100) if limits == "Long-only" else (-cap / 100, cap / 100)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if portfolio is None:
    st.html(
        """
<div class="ss-empty">
  <h2>Build a portfolio to optimize</h2>
  <p>Choose a data source in the sidebar, enter two or more tickers and select <b>Fetch prices</b>.
  Yahoo Finance needs no key; Alpha Vantage and Tiingo need a free one.</p>
  <p>To use your own data, pick <b>Upload a CSV</b>: dates in the first column and one column
  of prices per ticker.</p>
</div>
"""
    )
    st.stop()

prices, notes = clean_prices(portfolio["prices"])
notes = list(portfolio.get("notes", [])) + notes
if prices.shape[1] < 2:
    st.error("At least two assets with overlapping price history are needed. Add tickers or widen the date range.")
    st.stop()

prices, periods, freq_label, freq_notes = resample_prices(prices, frequency)
notes += freq_notes
returns = prices_to_returns(prices)
rf = rf_pct / 100
names = portfolio.get("names", {})

try:
    result = run_optimization(returns, periods, rf, cov_method, lower, upper)
except OptimizationError as exc:
    st.error(str(exc))
    st.stop()

notes += result.warnings
tan, gmv, eq = result.max_sharpe, result.min_variance, result.equal_weight

# -- header -------------------------------------------------------------------
badge_class = "ss-badge sim" if portfolio["simulated"] else "ss-badge"
lede = f'<p class="ss-lede">{html.escape(portfolio["description"])}</p>' if portfolio["description"] else ""
st.html(
    f"""
<div class="ss-head"><h1>{html.escape(portfolio["title"])}</h1>
<span class="{badge_class}">{html.escape(portfolio["badge"])}</span></div>
{lede}
<p class="ss-meta">{len(returns):,} {freq_label} returns for {returns.shape[1]} assets,
{returns.index[0]:%d %b %Y} to {returns.index[-1]:%d %b %Y}.</p>
"""
)
if notes:
    st.info("\n".join(f"- {n}" for n in notes))

# -- hero: frontier + readout ---------------------------------------------------
chart_col, readout_col = st.columns([2.1, 1], gap="large")
with chart_col:
    st.plotly_chart(charts.frontier_figure(result, cloud_for(result.mu, result.cov, rf), names),
                    theme=None, config=PLOT_CONFIG)

with readout_col:
    top = tan.weights.reindex(tan.weights.abs().sort_values(ascending=False).index)
    top = top[top.abs() > 5e-4].head(5)
    positions = "".join(f"<li><span>{html.escape(t)}</span><b>{pct(w)}</b></li>" for t, w in top.items())
    cov_note = COVARIANCE_METHODS[result.covariance_method]
    if result.shrinkage is not None:
        cov_note += f" ({result.shrinkage:.0%} shrinkage)"
    st.html(
        f"""
<table class="ss-readout">
  <thead><tr><th></th>
    <th><i style="background:{charts.TARGET}"></i>Max<br>Sharpe</th>
    <th><i style="background:{charts.TEAL}"></i>Min<br>variance</th>
    <th><i style="background:{charts.SLATE}"></i>Equal<br>weight</th></tr></thead>
  <tbody>
    <tr class="sharpe"><td>Sharpe ratio</td><td class="opt">{tan.sharpe:.2f}</td>
        <td>{gmv.sharpe:.2f}</td><td>{eq.sharpe:.2f}</td></tr>
    <tr><td>Expected return</td><td class="opt">{pct(tan.expected_return)}</td>
        <td>{pct(gmv.expected_return)}</td><td>{pct(eq.expected_return)}</td></tr>
    <tr><td>Volatility</td><td class="opt">{pct(tan.volatility)}</td>
        <td>{pct(gmv.volatility)}</td><td>{pct(eq.volatility)}</td></tr>
  </tbody>
</table>
<div class="ss-positions"><h4>Largest positions in the max-Sharpe portfolio</h4><ol>{positions}</ol></div>
<p class="ss-settings">Risk-free rate {pct(rf, 2)}.<br>{html.escape(cov_note)}.<br>
{html.escape(result.constraints.describe())}.<br>Solved by {html.escape(result.method)}.</p>
"""
    )

# -- detail tabs ------------------------------------------------------------------
tab_weights, tab_risk, tab_growth, tab_data, tab_maths = st.tabs(
    ["Weights", "Risk and correlation", "In-sample growth", "Price data", "How it works"]
)

with tab_weights:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.plotly_chart(charts.weights_figure(result), theme=None, config=PLOT_CONFIG)
    with right:
        table = result.weights_table()
        weight_cols = ["Max Sharpe", "Min variance", "Equal weight"]
        st.dataframe(table.sort_values("Max Sharpe", ascending=False).style.format(pct, subset=weight_cols))
        st.download_button("Download weights as CSV", result.weights_table().to_csv().encode(),
                           f"{portfolio['title'].lower().replace(' ', '-')}-weights.csv", "text/csv")
    st.caption("Bars show the max-Sharpe weights. Tick marks show the minimum-variance and equal-weight "
               "portfolios for comparison.")

with tab_risk:
    st.markdown("#### Asset statistics")
    stats = asset_table(result.mu, result.cov, rf, tan.weights, names)
    pct_cols = ("Expected return", "Volatility", "Optimal weight")
    st.dataframe(stats.style.format(pct, subset=list(pct_cols)).format("{:.2f}", subset=["Sharpe ratio"]))
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Correlation matrix")
        st.plotly_chart(charts.correlation_figure(correlation_from_covariance(result.cov)),
                        theme=None, config=PLOT_CONFIG)
    with right:
        st.markdown("#### Share of capital vs share of risk")
        st.plotly_chart(charts.risk_contribution_figure(result), theme=None, config=PLOT_CONFIG)
        st.caption("Each asset's share of portfolio variance is its weight times its covariance with the "
                   "whole portfolio. Volatile or highly correlated assets take more risk than capital.")
    with st.expander("Annualized covariance matrix used by the optimizer"):
        st.dataframe(result.cov.style.format("{:.5f}"))

with tab_growth:
    growth = growth_of_one(returns, result.weights_table())
    st.plotly_chart(charts.growth_figure(growth), theme=None, config=PLOT_CONFIG)
    perf = performance_table(growth, periods, rf)
    perf_pct = ["Total return", "Annual growth (CAGR)", "Volatility", "Max drawdown"]
    st.dataframe(perf.style.format(pct, subset=perf_pct).format("{:.2f}", subset=["Realized Sharpe"]))
    st.caption("These weights were chosen using this same history, so the max-Sharpe line benefits from "
               "hindsight. Treat this chart as a description of the past, not a forecast. Portfolios are "
               "rebalanced back to their target weights every period, with no trading costs.")

with tab_data:
    st.plotly_chart(charts.prices_figure(prices), theme=None, config=PLOT_CONFIG)
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.dataframe(prices.sort_index(ascending=False).style.format("{:,.2f}"), height=320)
    with right:
        st.markdown(f"**{len(prices):,}** {freq_label} prices  \n**{returns.shape[1]}** assets  \n"
                    f"Annualized with **{periods}** periods a year")
        st.download_button("Download returns as CSV", returns.to_csv().encode(), "returns.csv", "text/csv")
        st.download_button("Download prices as CSV", prices.to_csv().encode(), "prices.csv", "text/csv")

with tab_maths:
    st.markdown(
        r"""
#### What is being maximized

For weights $w$ that sum to 1, annualized expected returns $\mu$ and covariance matrix $\Sigma$,
the portfolio's expected return is $w^\top\mu$ and its volatility is $\sqrt{w^\top \Sigma w}$.
The Sharpe ratio is the excess return per unit of volatility:
"""
    )
    st.latex(r"S(w) = \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}")
    st.markdown(
        r"""
#### How it is solved

With no limits on individual weights, the maximum has a closed form, the **tangency portfolio**:
"""
    )
    st.latex(r"w^\ast = \frac{\Sigma^{-1}(\mu - r_f\mathbf{1})}{\mathbf{1}^\top\Sigma^{-1}(\mu - r_f\mathbf{1})}")
    st.markdown(
        r"""
With limits (long-only or capped weights), the Sharpe ratio is rescaled into a convex quadratic program
by substituting $y = w / (w^\top\mu - r_f)$:
"""
    )
    st.latex(r"\min_y\; y^\top \Sigma y \quad\text{s.t.}\quad (\mu - r_f\mathbf 1)^\top y = 1,\;\; "
             r"\ell\,\mathbf 1^\top y \le y_i \le u\,\mathbf 1^\top y")
    st.markdown(
        r"""
and the weights are recovered as $w = y / \mathbf 1^\top y$. Because the problem is convex, the solution
is the global maximum. The direct problem is also solved from two starting points as a cross-check, and
the best feasible answer is kept.

On the chart, the dashed line from the risk-free rate through the max-Sharpe portfolio is the
**capital market line**. Its slope is the Sharpe ratio, and no portfolio on or below the efficient
frontier can have a steeper line, which is why the optimum is the point where the line just touches
the frontier.

The README walks through the derivation, the covariance estimators and the limitations in more detail.
"""
    )
