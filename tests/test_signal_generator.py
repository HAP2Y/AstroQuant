"""
Tests for the Signal Generator module.
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.signal_generator import (
    SignalGenerator, SignalType, TradingSignal, SignalCalendar,
    MarketIndexSignalGenerator
)


class TestSignalGenerator:
    """Test suite for SignalGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a SignalGenerator instance for testing."""
        return SignalGenerator()

    @pytest.fixture
    def test_date(self):
        """Standard test date."""
        return datetime(2024, 6, 15, 12, 0, 0)

    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly."""
        assert generator is not None
        assert generator.sentiment_scorer is not None
        assert generator.sector_mapper is not None

    def test_generate_signal_returns_valid_signal(self, generator, test_date):
        """Test that signal generation returns valid signal."""
        signal = generator.generate_signal(
            ticker="AAPL",
            date=test_date,
            current_price=150.0,
            ml_prediction=0.02,  # 2% predicted increase
            ml_confidence=0.7
        )

        assert isinstance(signal, TradingSignal)
        assert signal.date == test_date
        assert isinstance(signal.signal_type, SignalType)

    def test_signal_has_confidence(self, generator, test_date):
        """Test that signal includes confidence level."""
        signal = generator.generate_signal(
            ticker="AAPL",
            date=test_date,
            current_price=150.0,
            ml_prediction=0.02,
            ml_confidence=0.7
        )

        assert 0 <= signal.confidence <= 1

    def test_signal_has_sentiment_score(self, generator, test_date):
        """Test that signal includes sentiment score."""
        signal = generator.generate_signal(
            ticker="AAPL",
            date=test_date,
            current_price=150.0,
            ml_prediction=0.02,
            ml_confidence=0.7
        )

        assert 0 <= signal.sentiment_score <= 100

    def test_signal_has_price_targets(self, generator, test_date):
        """Test that non-hold signals have price targets."""
        signal = generator.generate_signal(
            ticker="AAPL",
            date=test_date,
            current_price=150.0,
            ml_prediction=0.05,  # Strong prediction
            ml_confidence=0.8
        )

        if signal.signal_type != SignalType.HOLD:
            assert signal.price_target is not None
            assert signal.stop_loss is not None

    def test_signal_has_key_factors(self, generator, test_date):
        """Test that signal includes key factors."""
        signal = generator.generate_signal(
            ticker="AAPL",
            date=test_date,
            current_price=150.0,
            ml_prediction=0.02,
            ml_confidence=0.7
        )

        assert signal.key_factors is not None
        assert isinstance(signal.key_factors, list)

    def test_signal_risk_level_is_valid(self, generator, test_date):
        """Test that risk level is valid."""
        signal = generator.generate_signal(
            ticker="AAPL",
            date=test_date,
            current_price=150.0,
            ml_prediction=0.02,
            ml_confidence=0.7
        )

        assert signal.risk_level in ['low', 'medium', 'high']

    def test_signal_time_horizon_is_valid(self, generator, test_date):
        """Test that time horizon is valid."""
        signal = generator.generate_signal(
            ticker="AAPL",
            date=test_date,
            current_price=150.0,
            ml_prediction=0.02,
            ml_confidence=0.7
        )

        assert signal.time_horizon in ['short', 'medium', 'long']


class TestSignalCalendar:
    """Test signal calendar generation."""

    @pytest.fixture
    def generator(self):
        return SignalGenerator()

    @pytest.fixture
    def test_date(self):
        return datetime(2024, 6, 1)

    def test_generate_signal_calendar(self, generator, test_date):
        """Test generating a signal calendar."""
        calendar = generator.generate_signal_calendar(
            ticker="AAPL",
            start_date=test_date,
            days=30,
            current_price=150.0
        )

        assert isinstance(calendar, SignalCalendar)
        assert calendar.ticker == "AAPL"
        assert len(calendar.signals) == 30

    def test_calendar_has_summary(self, generator, test_date):
        """Test that calendar includes summary statistics."""
        calendar = generator.generate_signal_calendar(
            ticker="AAPL",
            start_date=test_date,
            days=30,
            current_price=150.0
        )

        assert calendar.summary is not None
        assert 'total_signals' in calendar.summary
        assert 'average_confidence' in calendar.summary
        assert 'bullish_bias' in calendar.summary

    def test_calendar_has_best_dates(self, generator, test_date):
        """Test that calendar identifies best dates."""
        calendar = generator.generate_signal_calendar(
            ticker="AAPL",
            start_date=test_date,
            days=30,
            current_price=150.0
        )

        assert isinstance(calendar.best_buy_dates, list)
        assert isinstance(calendar.best_sell_dates, list)


class TestPivotPoints:
    """Test pivot point detection."""

    @pytest.fixture
    def generator(self):
        return SignalGenerator()

    def test_find_pivot_points(self, generator):
        """Test finding pivot points."""
        start = datetime(2024, 6, 1)

        pivots = generator.find_pivot_points(
            ticker="AAPL",
            start_date=start,
            days=90
        )

        assert isinstance(pivots, list)
        for pivot in pivots:
            assert 'date' in pivot
            assert 'type' in pivot
            assert 'confidence' in pivot

    def test_pivot_types_are_valid(self, generator):
        """Test that pivot types are valid."""
        start = datetime(2024, 6, 1)

        pivots = generator.find_pivot_points(
            ticker="AAPL",
            start_date=start,
            days=90
        )

        valid_types = ['potential_bottom', 'potential_top',
                       'bullish_crossover', 'bearish_crossover']

        for pivot in pivots:
            assert pivot['type'] in valid_types


class TestMarketIndexSignalGenerator:
    """Test market index signal generation."""

    @pytest.fixture
    def generator(self):
        return MarketIndexSignalGenerator()

    def test_generate_market_signals(self, generator):
        """Test generating signals for market index."""
        start = datetime(2024, 6, 1)

        calendar = generator.generate_market_signals(
            index_name="S&P 500",
            start_date=start,
            days=30
        )

        assert isinstance(calendar, SignalCalendar)
        assert calendar.ticker == "S&P 500"
        assert len(calendar.signals) == 30
