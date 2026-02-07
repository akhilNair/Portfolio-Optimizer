"""yfinance wrappers for historical and real-time market data."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

from src.models.schemas import HistoricalReturnsData, RealTimeMarketSnapshot


class MarketDataFetcher:
    """Fetch market data from Yahoo Finance."""

    def get_historical_returns(
        self,
        tickers: list[str],
        lookback_days: int = 252,
    ) -> HistoricalReturnsData:
        """Fetch ~1 year of daily data, compute annualized mean returns and covariance."""
        end = datetime.now()
        start = end - timedelta(days=int(lookback_days * 1.5))

        data = yf.download(
            tickers=" ".join(tickers),
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            threads=True,
        )

        if len(tickers) == 1:
            close = data[["Close"]].dropna()
            close.columns = tickers
        else:
            close = data["Close"][tickers].dropna()

        daily_returns = np.log(close / close.shift(1)).dropna()

        mean_returns = (daily_returns.mean() * 252).tolist()
        cov_matrix = (daily_returns.cov() * 252).values.tolist()

        return HistoricalReturnsData(
            tickers=tickers,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            lookback_days=lookback_days,
        )

    def get_realtime_snapshot(
        self,
        tickers: list[str],
        window_days: int = 30,
    ) -> RealTimeMarketSnapshot:
        """Fetch latest window_days of daily data plus current prices."""
        end = datetime.now()
        start = end - timedelta(days=int(window_days * 2))

        data = yf.download(
            tickers=" ".join(tickers),
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            threads=True,
        )

        if len(tickers) == 1:
            close = data[["Close"]].dropna()
            close.columns = tickers
        else:
            close = data["Close"][tickers].dropna()

        daily_returns = np.log(close / close.shift(1)).dropna().tail(window_days)
        current_prices = {t: float(close[t].iloc[-1]) for t in tickers}

        return RealTimeMarketSnapshot(
            tickers=tickers,
            current_prices=current_prices,
            daily_returns_30d=daily_returns.values.tolist(),
            cov_matrix_30d=daily_returns.cov().values.tolist(),
            mean_returns_30d=daily_returns.mean().tolist(),
            fetch_timestamp=datetime.now(),
        )
