"""
Tests for the Market Sentiment Scorer module.
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sentiment_scorer import MarketSentimentScorer, MarketSentiment, SentimentComponents


class TestMarketSentimentScorer:
    """Test suite for MarketSentimentScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a MarketSentimentScorer instance for testing."""
        return MarketSentimentScorer()

    @pytest.fixture
    def test_date(self):
        """Standard test date."""
        return datetime(2024, 6, 15, 12, 0, 0)

    def test_scorer_initialization(self, scorer):
        """Test that scorer initializes correctly."""
        assert scorer is not None
        assert scorer.weights is not None
        assert scorer.engine is not None
        assert scorer.vedic is not None

    def test_calculate_sentiment_returns_valid_score(self, scorer, test_date):
        """Test that sentiment calculation returns valid score."""
        sentiment = scorer.calculate_sentiment(test_date)

        assert isinstance(sentiment, MarketSentiment)
        assert 0 <= sentiment.overall_score <= 100

    def test_sentiment_has_all_components(self, scorer, test_date):
        """Test that sentiment includes all component scores."""
        sentiment = scorer.calculate_sentiment(test_date)

        assert isinstance(sentiment.components, SentimentComponents)
        assert 0 <= sentiment.components.planetary_dignity_score <= 1
        assert 0 <= sentiment.components.nakshatra_influence_score <= 1
        assert 0 <= sentiment.components.aspect_harmony_score <= 1
        assert 0 <= sentiment.components.transit_strength_score <= 1
        assert 0 <= sentiment.components.lunar_phase_score <= 1
        assert 0 <= sentiment.components.retrograde_impact_score <= 1

    def test_sentiment_signal_is_valid(self, scorer, test_date):
        """Test that signal is one of valid options."""
        sentiment = scorer.calculate_sentiment(test_date)

        valid_signals = ['strong_buy', 'buy', 'neutral', 'sell', 'strong_sell']
        assert sentiment.signal in valid_signals

    def test_sentiment_volatility_is_valid(self, scorer, test_date):
        """Test that volatility forecast is valid."""
        sentiment = scorer.calculate_sentiment(test_date)

        valid_volatility = ['low', 'moderate', 'high', 'extreme']
        assert sentiment.volatility_forecast in valid_volatility

    def test_sentiment_has_interpretation(self, scorer, test_date):
        """Test that sentiment includes interpretation text."""
        sentiment = scorer.calculate_sentiment(test_date)

        assert sentiment.interpretation is not None
        assert len(sentiment.interpretation) > 0

    def test_sentiment_has_key_factors(self, scorer, test_date):
        """Test that sentiment includes key factors."""
        sentiment = scorer.calculate_sentiment(test_date)

        assert sentiment.key_factors is not None
        assert isinstance(sentiment.key_factors, list)

    def test_get_sentiment_range(self, scorer):
        """Test getting sentiment over a date range."""
        start = datetime(2024, 6, 1)
        end = datetime(2024, 6, 7)

        sentiments = scorer.get_sentiment_range(start, end, interval_hours=24)

        assert len(sentiments) == 7
        for sentiment in sentiments:
            assert 0 <= sentiment.overall_score <= 100

    def test_sentiment_consistency(self, scorer, test_date):
        """Test that same date produces consistent results."""
        sentiment1 = scorer.calculate_sentiment(test_date)
        sentiment2 = scorer.calculate_sentiment(test_date)

        # Scores should be identical for same datetime
        assert sentiment1.overall_score == sentiment2.overall_score

    def test_sentiment_varies_over_time(self, scorer):
        """Test that sentiment changes over time."""
        scores = []
        for i in range(30):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            sentiment = scorer.calculate_sentiment(date)
            scores.append(sentiment.overall_score)

        # Scores should vary (not all identical)
        assert len(set(scores)) > 1

    def test_find_optimal_dates_buy(self, scorer):
        """Test finding optimal buy dates."""
        start = datetime(2024, 6, 1)
        end = datetime(2024, 6, 30)

        optimal = scorer.find_optimal_dates(start, end, signal_type='buy', min_score=55.0)

        # Should return list of tuples
        assert isinstance(optimal, list)
        for item in optimal:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert item[1] >= 55.0

    def test_find_optimal_dates_sell(self, scorer):
        """Test finding optimal sell dates."""
        start = datetime(2024, 6, 1)
        end = datetime(2024, 6, 30)

        optimal = scorer.find_optimal_dates(start, end, signal_type='sell', min_score=55.0)

        # Should return list of tuples
        assert isinstance(optimal, list)
        for item in optimal:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert item[1] <= 45.0  # 100 - 55


class TestSentimentScoreRanges:
    """Test that sentiment scores produce correct signals."""

    @pytest.fixture
    def scorer(self):
        return MarketSentimentScorer()

    def test_high_score_produces_bullish_signal(self, scorer):
        """Test that very high scores produce strong_buy signal."""
        # Find a date with high score
        for i in range(365):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            sentiment = scorer.calculate_sentiment(date)

            if sentiment.overall_score >= 75:
                assert sentiment.signal == 'strong_buy'
                break

    def test_low_score_produces_bearish_signal(self, scorer):
        """Test that very low scores produce strong_sell signal."""
        # Find a date with low score
        for i in range(365):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            sentiment = scorer.calculate_sentiment(date)

            if sentiment.overall_score <= 25:
                assert sentiment.signal == 'strong_sell'
                break
