"""
Tests for the Sector Mapper module.
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import SECTOR_PLANETARY_MAP
from core.sector_mapper import SectorMapper, SectorAnalysis, SectorPhase, SectorForecast


class TestSectorMapper:
    """Test suite for SectorMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create a SectorMapper instance for testing."""
        return SectorMapper()

    @pytest.fixture
    def test_date(self):
        """Standard test date."""
        return datetime(2024, 6, 15, 12, 0, 0)

    def test_mapper_initialization(self, mapper):
        """Test that mapper initializes correctly."""
        assert mapper is not None
        assert mapper.engine is not None
        assert mapper.vedic is not None
        assert mapper.sector_map is not None

    def test_analyze_sector_returns_valid_analysis(self, mapper, test_date):
        """Test that sector analysis returns valid result."""
        analysis = mapper.analyze_sector("Technology", test_date)

        assert isinstance(analysis, SectorAnalysis)
        assert analysis.sector_name == "Technology"
        assert isinstance(analysis.phase, SectorPhase)
        assert 0 <= analysis.score <= 100

    def test_analyze_all_sectors(self, mapper, test_date):
        """Test analyzing all sectors at once."""
        analyses = mapper.analyze_all_sectors(test_date)

        assert len(analyses) == len(SECTOR_PLANETARY_MAP)
        for sector_name, analysis in analyses.items():
            assert analysis.sector_name == sector_name
            assert isinstance(analysis.phase, SectorPhase)

    def test_sector_analysis_has_confidence(self, mapper, test_date):
        """Test that analysis includes confidence level."""
        analysis = mapper.analyze_sector("Technology", test_date)

        assert 0 <= analysis.confidence <= 1

    def test_sector_analysis_has_recommendation(self, mapper, test_date):
        """Test that analysis includes recommended action."""
        analysis = mapper.analyze_sector("Technology", test_date)

        assert analysis.recommended_action is not None
        assert len(analysis.recommended_action) > 0

    def test_sector_analysis_has_primary_planet_status(self, mapper, test_date):
        """Test that analysis includes primary planet status."""
        analysis = mapper.analyze_sector("Technology", test_date)

        assert analysis.primary_planet_status is not None
        assert "Mercury" in analysis.primary_planet_status  # Tech ruled by Mercury

    def test_get_sector_rankings(self, mapper, test_date):
        """Test getting sector rankings."""
        rankings = mapper.get_sector_rankings(test_date)

        assert len(rankings) == len(SECTOR_PLANETARY_MAP)
        # Should be sorted by score (descending)
        scores = [r[2] for r in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_all_phases_are_valid(self, mapper, test_date):
        """Test that all returned phases are valid enum values."""
        analyses = mapper.analyze_all_sectors(test_date)

        valid_phases = [SectorPhase.GOLDEN, SectorPhase.FAVORABLE,
                       SectorPhase.NEUTRAL, SectorPhase.STRESS, SectorPhase.CRITICAL]

        for analysis in analyses.values():
            assert analysis.phase in valid_phases


class TestSectorForecast:
    """Test sector forecasting."""

    @pytest.fixture
    def mapper(self):
        return SectorMapper()

    @pytest.fixture
    def test_date(self):
        return datetime(2024, 6, 1)

    def test_forecast_sector(self, mapper, test_date):
        """Test generating sector forecast."""
        forecast = mapper.forecast_sector("Technology", test_date, days=30)

        assert isinstance(forecast, SectorForecast)
        assert forecast.sector_name == "Technology"
        assert forecast.current_score is not None

    def test_forecast_has_phase_shifts(self, mapper, test_date):
        """Test that forecast tracks phase shifts."""
        forecast = mapper.forecast_sector("Technology", test_date, days=90)

        assert isinstance(forecast.upcoming_shifts, list)
        for shift in forecast.upcoming_shifts:
            assert len(shift) == 3  # (date, new_phase, reason)

    def test_forecast_has_golden_periods(self, mapper, test_date):
        """Test that forecast identifies golden periods."""
        forecast = mapper.forecast_sector("Technology", test_date, days=90)

        assert isinstance(forecast.golden_periods, list)

    def test_forecast_has_stress_periods(self, mapper, test_date):
        """Test that forecast identifies stress periods."""
        forecast = mapper.forecast_sector("Technology", test_date, days=90)

        assert isinstance(forecast.stress_periods, list)

    def test_forecast_has_best_entry_dates(self, mapper, test_date):
        """Test that forecast identifies best entry dates."""
        forecast = mapper.forecast_sector("Technology", test_date, days=90)

        assert isinstance(forecast.best_entry_dates, list)

    def test_get_all_sectors_forecast(self, mapper, test_date):
        """Test forecasting all sectors."""
        forecasts = mapper.get_all_sectors_forecast(test_date, days=30)

        assert len(forecasts) == len(SECTOR_PLANETARY_MAP)


class TestSectorTickerMapping:
    """Test ticker-to-sector mapping."""

    @pytest.fixture
    def mapper(self):
        return SectorMapper()

    def test_get_sector_for_known_ticker(self, mapper):
        """Test finding sector for known ticker."""
        sector = mapper.get_sector_for_ticker("AAPL")

        assert sector == "Technology"

    def test_get_sector_for_unknown_ticker(self, mapper):
        """Test handling unknown ticker."""
        sector = mapper.get_sector_for_ticker("UNKNOWN123")

        assert sector is None

    def test_get_sector_tickers(self, mapper):
        """Test getting tickers for a sector."""
        tickers = mapper.get_sector_tickers("Technology")

        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert "AAPL" in tickers or "MSFT" in tickers

    def test_get_tickers_for_invalid_sector(self, mapper):
        """Test handling invalid sector."""
        tickers = mapper.get_sector_tickers("InvalidSector")

        assert tickers == []


class TestSectorScoreConsistency:
    """Test sector score consistency and range."""

    @pytest.fixture
    def mapper(self):
        return SectorMapper()

    def test_scores_vary_by_sector(self, mapper):
        """Test that different sectors have different scores."""
        test_date = datetime(2024, 6, 15)
        analyses = mapper.analyze_all_sectors(test_date)

        scores = [a.score for a in analyses.values()]

        # Scores should vary (not all identical)
        assert len(set(scores)) > 1

    def test_scores_change_over_time(self, mapper):
        """Test that sector scores change over time."""
        scores = []
        for i in range(30):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            analysis = mapper.analyze_sector("Technology", date)
            scores.append(analysis.score)

        # Scores should vary over time
        assert len(set(scores)) > 1

    def test_invalid_sector_raises_error(self, mapper):
        """Test that invalid sector raises ValueError."""
        test_date = datetime(2024, 6, 15)

        with pytest.raises(ValueError):
            mapper.analyze_sector("InvalidSector", test_date)
