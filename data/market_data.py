"""
Market Data Fetcher - Historical and Real-time Market Data
==========================================================
Fetches market data from various sources with caching support.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

from config.settings import MARKET_INDICES


@dataclass
class TickerInfo:
    """Information about a stock ticker."""
    symbol: str
    name: str
    sector: Optional[str]
    industry: Optional[str]
    currency: str
    exchange: str
    current_price: float
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]


class MarketDataFetcher:
    """
    Fetches and caches market data from various sources.

    Primary data source is Yahoo Finance via yfinance library.
    """

    CACHE_DIR = Path(__file__).parent / "cache"

    def __init__(self, cache_enabled: bool = True, cache_expiry_hours: int = 1):
        """
        Initialize the market data fetcher.

        Args:
            cache_enabled: Whether to use caching.
            cache_expiry_hours: Hours before cache expires.
        """
        if not YF_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        self.cache_enabled = cache_enabled
        self.cache_expiry = timedelta(hours=cache_expiry_hours)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, ticker: str, data_type: str) -> Path:
        """Get cache file path for a ticker."""
        safe_ticker = ticker.replace('.', '_').replace('^', '_')
        return self.CACHE_DIR / f"{safe_ticker}_{data_type}.parquet"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < self.cache_expiry

    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path) -> None:
        """Save data to cache."""
        if self.cache_enabled:
            data.to_parquet(cache_path)

    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        if self.cache_enabled and self._is_cache_valid(cache_path):
            return pd.read_parquet(cache_path)
        return None

    def get_historical_data(self, ticker: str, period: str = "1y",
                           interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol.
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max).
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo).

        Returns:
            DataFrame with OHLCV data.
        """
        cache_path = self._get_cache_path(ticker, f"history_{period}_{interval}")

        # Try cache first
        cached = self._load_from_cache(cache_path)
        if cached is not None:
            return cached

        # Fetch from yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)

        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        # Ensure proper column names
        data.columns = [c.title() for c in data.columns]

        # Save to cache
        self._save_to_cache(data, cache_path)

        return data

    def get_historical_range(self, ticker: str, start_date: datetime,
                            end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """
        Get historical data for a specific date range.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start of date range.
            end_date: End of date range.
            interval: Data interval.

        Returns:
            DataFrame with OHLCV data.
        """
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)

        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        data.columns = [c.title() for c in data.columns]
        return data

    def get_ticker_info(self, ticker: str) -> TickerInfo:
        """
        Get information about a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            TickerInfo with ticker details.
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        return TickerInfo(
            symbol=ticker,
            name=info.get('longName', info.get('shortName', ticker)),
            sector=info.get('sector'),
            industry=info.get('industry'),
            currency=info.get('currency', 'USD'),
            exchange=info.get('exchange', 'Unknown'),
            current_price=info.get('currentPrice', info.get('regularMarketPrice', 0)),
            market_cap=info.get('marketCap'),
            pe_ratio=info.get('trailingPE'),
            dividend_yield=info.get('dividendYield')
        )

    def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Current price.
        """
        stock = yf.Ticker(ticker)
        return stock.info.get('currentPrice', stock.info.get('regularMarketPrice', 0))

    def get_multiple_tickers(self, tickers: List[str], period: str = "1y",
                            interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple tickers.

        Args:
            tickers: List of ticker symbols.
            period: Data period.
            interval: Data interval.

        Returns:
            Dictionary mapping tickers to their data.
        """
        results = {}

        # Use yfinance batch download for efficiency
        data = yf.download(tickers, period=period, interval=interval, group_by='ticker')

        for ticker in tickers:
            try:
                if len(tickers) > 1:
                    ticker_data = data[ticker].copy()
                else:
                    ticker_data = data.copy()

                ticker_data.columns = [c.title() for c in ticker_data.columns]
                results[ticker] = ticker_data.dropna()
            except Exception:
                continue

        return results

    def get_index_data(self, index_name: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical data for a market index.

        Args:
            index_name: Name of index (e.g., 'S&P 500').
            period: Data period.

        Returns:
            DataFrame with index data.
        """
        ticker = MARKET_INDICES.get(index_name)
        if not ticker:
            raise ValueError(f"Unknown index: {index_name}. "
                           f"Available: {list(MARKET_INDICES.keys())}")

        return self.get_historical_data(ticker, period)

    def calculate_returns(self, data: pd.DataFrame,
                         column: str = 'Close') -> pd.DataFrame:
        """
        Calculate various return metrics.

        Args:
            data: DataFrame with price data.
            column: Column to calculate returns for.

        Returns:
            DataFrame with return metrics.
        """
        returns = pd.DataFrame(index=data.index)

        returns['daily_return'] = data[column].pct_change()
        returns['cumulative_return'] = (1 + returns['daily_return']).cumprod() - 1
        returns['rolling_7d'] = data[column].pct_change(7)
        returns['rolling_30d'] = data[column].pct_change(30)
        returns['volatility_20d'] = returns['daily_return'].rolling(20).std() * np.sqrt(252)

        return returns

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame with technical indicators.
        """
        indicators = pd.DataFrame(index=data.index)

        # Moving averages
        indicators['SMA_20'] = data['Close'].rolling(window=20).mean()
        indicators['SMA_50'] = data['Close'].rolling(window=50).mean()
        indicators['SMA_200'] = data['Close'].rolling(window=200).mean()
        indicators['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        indicators['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        indicators['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = indicators['BB_Middle'] + (bb_std * 2)
        indicators['BB_Lower'] = indicators['BB_Middle'] - (bb_std * 2)

        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['ATR'] = tr.rolling(window=14).mean()

        # Volume indicators
        indicators['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
        indicators['Volume_Ratio'] = data['Volume'] / indicators['Volume_SMA_20']

        return indicators

    def prepare_ml_dataset(self, ticker: str, period: str = "2y",
                          include_sentiment: bool = False) -> pd.DataFrame:
        """
        Prepare a complete dataset for ML training.

        Args:
            ticker: Stock ticker symbol.
            period: Data period.
            include_sentiment: Whether to include sentiment scores.

        Returns:
            DataFrame ready for ML training.
        """
        # Get price data
        data = self.get_historical_data(ticker, period)

        # Calculate returns
        returns = self.calculate_returns(data)

        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)

        # Merge all data
        dataset = data.join(returns, rsuffix='_ret')
        dataset = dataset.join(indicators, rsuffix='_ind')

        # Add sentiment if requested
        if include_sentiment:
            from core.sentiment_scorer import MarketSentimentScorer
            scorer = MarketSentimentScorer()

            sentiments = []
            for date in dataset.index:
                try:
                    sentiment = scorer.calculate_sentiment(date.to_pydatetime())
                    sentiments.append(sentiment.overall_score)
                except Exception:
                    sentiments.append(50.0)  # Default neutral

            dataset['sentiment_score'] = sentiments

        # Drop NaN rows
        dataset = dataset.dropna()

        return dataset

    def get_sector_performance(self, period: str = "1mo") -> Dict[str, float]:
        """
        Get performance of major sectors.

        Args:
            period: Period for performance calculation.

        Returns:
            Dictionary mapping sectors to their returns.
        """
        # Sector ETFs as proxies
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Finance': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Communications': 'XLC',
        }

        performance = {}
        data = self.get_multiple_tickers(list(sector_etfs.values()), period)

        for sector, etf in sector_etfs.items():
            if etf in data and len(data[etf]) > 0:
                etf_data = data[etf]
                returns = (etf_data['Close'].iloc[-1] / etf_data['Close'].iloc[0]) - 1
                performance[sector] = round(returns * 100, 2)

        return performance

    def clear_cache(self) -> None:
        """Clear all cached data."""
        for cache_file in self.CACHE_DIR.glob("*.parquet"):
            cache_file.unlink()

    def validate_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker is valid.

        Args:
            ticker: Ticker symbol to validate.

        Returns:
            True if ticker is valid, False otherwise.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return 'symbol' in info or 'shortName' in info
        except Exception:
            return False
