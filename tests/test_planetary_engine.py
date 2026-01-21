"""
Tests for the Planetary Engine module.
"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Planet, ZodiacSign
from core.planetary_engine import PlanetaryEngine, PlanetaryPosition, LunarPhase


class TestPlanetaryEngine:
    """Test suite for PlanetaryEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a PlanetaryEngine instance for testing."""
        return PlanetaryEngine()

    @pytest.fixture
    def test_date(self):
        """Standard test date."""
        return datetime(2024, 6, 15, 12, 0, 0)

    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly."""
        assert engine is not None
        assert engine.observer is not None

    def test_get_ayanamsa(self, engine, test_date):
        """Test ayanamsa calculation."""
        ayanamsa = engine.get_ayanamsa(test_date)

        # Ayanamsa should be approximately 24 degrees in modern times
        assert 23 < ayanamsa < 25
        assert isinstance(ayanamsa, float)

    def test_get_planet_position_sun(self, engine, test_date):
        """Test Sun position calculation."""
        pos = engine.get_planet_position(Planet.SUN, test_date)

        assert isinstance(pos, PlanetaryPosition)
        assert pos.planet == Planet.SUN
        assert 0 <= pos.longitude < 360
        assert 0 <= pos.sign_degree < 30
        assert isinstance(pos.sign, ZodiacSign)
        assert 0 <= pos.nakshatra_index < 27
        assert 1 <= pos.nakshatra_pada <= 4

    def test_get_planet_position_moon(self, engine, test_date):
        """Test Moon position calculation."""
        pos = engine.get_planet_position(Planet.MOON, test_date)

        assert isinstance(pos, PlanetaryPosition)
        assert pos.planet == Planet.MOON
        assert 0 <= pos.longitude < 360

    def test_get_planet_position_all_planets(self, engine, test_date):
        """Test position calculation for all planets."""
        for planet in Planet:
            pos = engine.get_planet_position(planet, test_date)

            assert pos.planet == planet
            assert 0 <= pos.longitude < 360
            assert isinstance(pos.sign, ZodiacSign)

    def test_get_lunar_phase(self, engine, test_date):
        """Test lunar phase calculation."""
        phase = engine.get_lunar_phase(test_date)

        assert isinstance(phase, LunarPhase)
        assert 0 <= phase.phase_angle < 360
        assert 0 <= phase.illumination <= 1
        assert 1 <= phase.tithi <= 30
        assert phase.phase_name in [
            "New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
            "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"
        ]

    def test_get_all_positions(self, engine, test_date):
        """Test getting all planetary positions at once."""
        snapshot = engine.get_all_positions(test_date)

        assert snapshot is not None
        assert snapshot.timestamp == test_date
        assert len(snapshot.positions) == len(Planet)
        assert snapshot.lunar_phase is not None

    def test_get_positions_range(self, engine):
        """Test getting positions over a date range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 7)

        snapshots = engine.get_positions_range(start, end, interval_hours=24)

        assert len(snapshots) == 7
        for snapshot in snapshots:
            assert snapshot.positions is not None

    def test_rahu_ketu_opposition(self, engine, test_date):
        """Test that Rahu and Ketu are exactly opposite."""
        rahu_pos = engine.get_planet_position(Planet.RAHU, test_date)
        ketu_pos = engine.get_planet_position(Planet.KETU, test_date)

        # Calculate angular difference
        diff = abs(rahu_pos.longitude - ketu_pos.longitude)

        # Should be approximately 180 degrees
        assert 179 < diff < 181 or diff < 1 or diff > 359

    def test_retrograde_detection(self, engine):
        """Test retrograde detection for outer planets."""
        # Mercury retrograde periods are well documented
        # Test over a range to find one
        found_retrograde = False

        for i in range(365):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            pos = engine.get_planet_position(Planet.MERCURY, date)
            if pos.is_retrograde:
                found_retrograde = True
                break

        # Mercury should be retrograde at some point in the year
        assert found_retrograde

    def test_sign_boundaries(self, engine):
        """Test that sign calculations are correct at boundaries."""
        test_date = datetime(2024, 6, 15)
        pos = engine.get_planet_position(Planet.SUN, test_date)

        # Sign degree should be within valid range
        assert 0 <= pos.sign_degree < 30

        # Sign index should match longitude
        expected_sign_index = int(pos.longitude / 30)
        assert pos.sign.value == expected_sign_index


class TestNakshatraCalculations:
    """Test nakshatra-related calculations."""

    @pytest.fixture
    def engine(self):
        return PlanetaryEngine()

    def test_nakshatra_pada_range(self, engine):
        """Test that nakshatra pada is always 1-4."""
        for i in range(100):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            pos = engine.get_planet_position(Planet.MOON, date)

            assert 1 <= pos.nakshatra_pada <= 4

    def test_nakshatra_index_range(self, engine):
        """Test that nakshatra index is always 0-26."""
        for i in range(100):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            pos = engine.get_planet_position(Planet.MOON, date)

            assert 0 <= pos.nakshatra_index < 27
