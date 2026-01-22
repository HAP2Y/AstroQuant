"""
Test module imports to ensure all dependencies are properly configured.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModuleImports:
    """Test that all modules can be imported successfully."""

    def test_config_imports(self):
        """Test config module imports."""
        from config.settings import MARKET_INDICES, SECTOR_PLANETARY_MAP, Planet
        assert MARKET_INDICES is not None
        assert SECTOR_PLANETARY_MAP is not None
        assert Planet is not None

    def test_core_planetary_engine(self):
        """Test planetary engine import."""
        from core.planetary_engine import PlanetaryEngine
        assert PlanetaryEngine is not None

    def test_core_vedic_calculator(self):
        """Test vedic calculator import."""
        from core.vedic_calculator import VedicCalculator
        assert VedicCalculator is not None

    def test_core_sentiment_scorer(self):
        """Test sentiment scorer import."""
        from core.sentiment_scorer import MarketSentimentScorer
        assert MarketSentimentScorer is not None

    def test_core_sector_mapper(self):
        """Test sector mapper import."""
        from core.sector_mapper import SectorMapper, SectorPhase
        assert SectorMapper is not None
        assert SectorPhase is not None

    def test_models_signal_generator(self):
        """Test signal generator import."""
        from models.signal_generator import SignalGenerator, MarketIndexSignalGenerator
        assert SignalGenerator is not None
        assert MarketIndexSignalGenerator is not None

    def test_data_market_data(self):
        """Test market data import."""
        from data.market_data import MarketDataFetcher, TickerInfo
        assert MarketDataFetcher is not None
        assert TickerInfo is not None

    def test_ui_components(self):
        """Test UI components import."""
        from ui.components import (
            render_sentiment_gauge, render_signal_card, render_sector_card
        )
        assert render_sentiment_gauge is not None
        assert render_signal_card is not None
        assert render_sector_card is not None

    def test_ui_visualizations(self):
        """Test UI visualizations import."""
        from ui.visualizations import (
            create_sentiment_timeline, create_sector_heatmap
        )
        assert create_sentiment_timeline is not None
        assert create_sector_heatmap is not None


class TestBasicFunctionality:
    """Test basic functionality of core modules."""

    def test_planetary_engine_initialization(self):
        """Test PlanetaryEngine can be initialized."""
        from core.planetary_engine import PlanetaryEngine
        engine = PlanetaryEngine()
        assert engine is not None

    def test_sentiment_scorer_initialization(self):
        """Test MarketSentimentScorer can be initialized."""
        from core.sentiment_scorer import MarketSentimentScorer
        scorer = MarketSentimentScorer()
        assert scorer is not None

    def test_sector_mapper_initialization(self):
        """Test SectorMapper can be initialized."""
        from core.sector_mapper import SectorMapper
        mapper = SectorMapper()
        assert mapper is not None

    def test_signal_generator_initialization(self):
        """Test SignalGenerator can be initialized."""
        from models.signal_generator import SignalGenerator
        generator = SignalGenerator()
        assert generator is not None

    def test_calculate_sentiment(self):
        """Test sentiment calculation works."""
        from datetime import datetime
        from core.sentiment_scorer import MarketSentimentScorer

        scorer = MarketSentimentScorer()
        sentiment = scorer.calculate_sentiment(datetime.now())

        assert sentiment is not None
        assert hasattr(sentiment, 'overall_score')
        assert 0 <= sentiment.overall_score <= 100
