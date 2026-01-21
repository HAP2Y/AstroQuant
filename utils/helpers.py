"""
Utility Functions
=================
Helper functions for formatting, validation, and common operations.
"""

from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional


def format_currency(value: float, currency: str = "USD") -> str:
    """
    Format a value as currency.

    Args:
        value: Numeric value.
        currency: Currency code.

    Returns:
        Formatted currency string.
    """
    symbols = {
        "USD": "$",
        "INR": "Rs.",
        "EUR": "EUR",
        "GBP": "GBP"
    }

    symbol = symbols.get(currency, currency)

    if abs(value) >= 1e9:
        return f"{symbol}{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{symbol}{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{symbol}{value/1e3:.2f}K"
    else:
        return f"{symbol}{value:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as percentage.

    Args:
        value: Numeric value (0-1 or 0-100).
        decimals: Decimal places.

    Returns:
        Formatted percentage string.
    """
    # Assume value > 1 means it's already a percentage
    if abs(value) > 1:
        return f"{value:.{decimals}f}%"
    else:
        return f"{value*100:.{decimals}f}%"


def get_date_range(start: datetime, end: datetime,
                   step_days: int = 1) -> List[datetime]:
    """
    Generate a list of dates between start and end.

    Args:
        start: Start date.
        end: End date.
        step_days: Days between each date.

    Returns:
        List of datetime objects.
    """
    dates = []
    current = start

    while current <= end:
        dates.append(current)
        current += timedelta(days=step_days)

    return dates


def validate_date(date_str: str, format: str = "%Y-%m-%d") -> Tuple[bool, Optional[datetime]]:
    """
    Validate a date string.

    Args:
        date_str: Date string to validate.
        format: Expected date format.

    Returns:
        Tuple of (is_valid, parsed_date).
    """
    try:
        parsed = datetime.strptime(date_str, format)
        return True, parsed
    except ValueError:
        return False, None


def calculate_trading_days(start: datetime, end: datetime) -> int:
    """
    Calculate number of trading days between dates.

    Args:
        start: Start date.
        end: End date.

    Returns:
        Number of trading days (excludes weekends).
    """
    days = 0
    current = start

    while current <= end:
        if current.weekday() < 5:  # Monday=0, Friday=4
            days += 1
        current += timedelta(days=1)

    return days


def get_fiscal_quarter(dt: datetime) -> Tuple[int, int]:
    """
    Get fiscal quarter for a date.

    Args:
        dt: Datetime to check.

    Returns:
        Tuple of (year, quarter).
    """
    quarter = (dt.month - 1) // 3 + 1
    return dt.year, quarter


def normalize_ticker(ticker: str) -> str:
    """
    Normalize a ticker symbol.

    Args:
        ticker: Raw ticker input.

    Returns:
        Normalized ticker symbol.
    """
    ticker = ticker.upper().strip()

    # Handle common formats
    if ticker.startswith("$"):
        ticker = ticker[1:]

    # Remove any whitespace
    ticker = ticker.replace(" ", "")

    return ticker


def is_market_open(dt: datetime = None, market: str = "US") -> bool:
    """
    Check if market is open at given time.

    Args:
        dt: Datetime to check (default: now).
        market: Market code (US, IN).

    Returns:
        True if market is open.
    """
    if dt is None:
        dt = datetime.now()

    # Check if weekend
    if dt.weekday() >= 5:
        return False

    # Market hours (simplified)
    market_hours = {
        "US": (9, 30, 16, 0),  # 9:30 AM - 4:00 PM
        "IN": (9, 15, 15, 30),  # 9:15 AM - 3:30 PM
    }

    if market in market_hours:
        open_h, open_m, close_h, close_m = market_hours[market]
        open_time = dt.replace(hour=open_h, minute=open_m, second=0)
        close_time = dt.replace(hour=close_h, minute=close_m, second=0)
        return open_time <= dt <= close_time

    return True  # Unknown market, assume open


def get_next_trading_day(dt: datetime = None) -> datetime:
    """
    Get the next trading day.

    Args:
        dt: Starting datetime (default: now).

    Returns:
        Next trading day datetime.
    """
    if dt is None:
        dt = datetime.now()

    next_day = dt + timedelta(days=1)

    while next_day.weekday() >= 5:  # Skip weekends
        next_day += timedelta(days=1)

    return next_day.replace(hour=9, minute=30, second=0, microsecond=0)
