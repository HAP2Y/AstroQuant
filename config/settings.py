"""
AstroQuant Configuration Settings
=================================
Central configuration for planetary mappings, scoring weights, and system constants.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum

# =============================================================================
# PLANETARY CONSTANTS
# =============================================================================

class Planet(Enum):
    SUN = "Sun"
    MOON = "Moon"
    MARS = "Mars"
    MERCURY = "Mercury"
    JUPITER = "Jupiter"
    VENUS = "Venus"
    SATURN = "Saturn"
    RAHU = "Rahu"  # North Node
    KETU = "Ketu"  # South Node
    URANUS = "Uranus"
    NEPTUNE = "Neptune"
    PLUTO = "Pluto"


class ZodiacSign(Enum):
    ARIES = 0
    TAURUS = 1
    GEMINI = 2
    CANCER = 3
    LEO = 4
    VIRGO = 5
    LIBRA = 6
    SCORPIO = 7
    SAGITTARIUS = 8
    CAPRICORN = 9
    AQUARIUS = 10
    PISCES = 11


# =============================================================================
# NAKSHATRA DEFINITIONS (27 Lunar Mansions)
# =============================================================================

NAKSHATRAS: List[Dict] = [
    {"name": "Ashwini", "ruler": Planet.KETU, "start": 0.0, "nature": "swift", "market_energy": 0.7},
    {"name": "Bharani", "ruler": Planet.VENUS, "start": 13.333, "nature": "fierce", "market_energy": 0.4},
    {"name": "Krittika", "ruler": Planet.SUN, "start": 26.667, "nature": "mixed", "market_energy": 0.5},
    {"name": "Rohini", "ruler": Planet.MOON, "start": 40.0, "nature": "fixed", "market_energy": 0.8},
    {"name": "Mrigashira", "ruler": Planet.MARS, "start": 53.333, "nature": "soft", "market_energy": 0.6},
    {"name": "Ardra", "ruler": Planet.RAHU, "start": 66.667, "nature": "sharp", "market_energy": 0.3},
    {"name": "Punarvasu", "ruler": Planet.JUPITER, "start": 80.0, "nature": "movable", "market_energy": 0.75},
    {"name": "Pushya", "ruler": Planet.SATURN, "start": 93.333, "nature": "light", "market_energy": 0.85},
    {"name": "Ashlesha", "ruler": Planet.MERCURY, "start": 106.667, "nature": "sharp", "market_energy": 0.35},
    {"name": "Magha", "ruler": Planet.KETU, "start": 120.0, "nature": "fierce", "market_energy": 0.5},
    {"name": "Purva Phalguni", "ruler": Planet.VENUS, "start": 133.333, "nature": "fierce", "market_energy": 0.65},
    {"name": "Uttara Phalguni", "ruler": Planet.SUN, "start": 146.667, "nature": "fixed", "market_energy": 0.7},
    {"name": "Hasta", "ruler": Planet.MOON, "start": 160.0, "nature": "light", "market_energy": 0.75},
    {"name": "Chitra", "ruler": Planet.MARS, "start": 173.333, "nature": "soft", "market_energy": 0.6},
    {"name": "Swati", "ruler": Planet.RAHU, "start": 186.667, "nature": "movable", "market_energy": 0.55},
    {"name": "Vishakha", "ruler": Planet.JUPITER, "start": 200.0, "nature": "mixed", "market_energy": 0.65},
    {"name": "Anuradha", "ruler": Planet.SATURN, "start": 213.333, "nature": "soft", "market_energy": 0.7},
    {"name": "Jyeshtha", "ruler": Planet.MERCURY, "start": 226.667, "nature": "sharp", "market_energy": 0.4},
    {"name": "Mula", "ruler": Planet.KETU, "start": 240.0, "nature": "sharp", "market_energy": 0.3},
    {"name": "Purva Ashadha", "ruler": Planet.VENUS, "start": 253.333, "nature": "fierce", "market_energy": 0.55},
    {"name": "Uttara Ashadha", "ruler": Planet.SUN, "start": 266.667, "nature": "fixed", "market_energy": 0.7},
    {"name": "Shravana", "ruler": Planet.MOON, "start": 280.0, "nature": "movable", "market_energy": 0.8},
    {"name": "Dhanishta", "ruler": Planet.MARS, "start": 293.333, "nature": "movable", "market_energy": 0.6},
    {"name": "Shatabhisha", "ruler": Planet.RAHU, "start": 306.667, "nature": "movable", "market_energy": 0.45},
    {"name": "Purva Bhadrapada", "ruler": Planet.JUPITER, "start": 320.0, "nature": "fierce", "market_energy": 0.5},
    {"name": "Uttara Bhadrapada", "ruler": Planet.SATURN, "start": 333.333, "nature": "fixed", "market_energy": 0.65},
    {"name": "Revati", "ruler": Planet.MERCURY, "start": 346.667, "nature": "soft", "market_energy": 0.75},
]


# =============================================================================
# PLANETARY DIGNITIES (Vedic)
# =============================================================================

# Exaltation signs for each planet
EXALTATION: Dict[Planet, ZodiacSign] = {
    Planet.SUN: ZodiacSign.ARIES,
    Planet.MOON: ZodiacSign.TAURUS,
    Planet.MARS: ZodiacSign.CAPRICORN,
    Planet.MERCURY: ZodiacSign.VIRGO,
    Planet.JUPITER: ZodiacSign.CANCER,
    Planet.VENUS: ZodiacSign.PISCES,
    Planet.SATURN: ZodiacSign.LIBRA,
    Planet.RAHU: ZodiacSign.TAURUS,
    Planet.KETU: ZodiacSign.SCORPIO,
}

# Debilitation signs for each planet
DEBILITATION: Dict[Planet, ZodiacSign] = {
    Planet.SUN: ZodiacSign.LIBRA,
    Planet.MOON: ZodiacSign.SCORPIO,
    Planet.MARS: ZodiacSign.CANCER,
    Planet.MERCURY: ZodiacSign.PISCES,
    Planet.JUPITER: ZodiacSign.CAPRICORN,
    Planet.VENUS: ZodiacSign.VIRGO,
    Planet.SATURN: ZodiacSign.ARIES,
    Planet.RAHU: ZodiacSign.SCORPIO,
    Planet.KETU: ZodiacSign.TAURUS,
}

# Own signs (Rulership)
OWN_SIGNS: Dict[Planet, List[ZodiacSign]] = {
    Planet.SUN: [ZodiacSign.LEO],
    Planet.MOON: [ZodiacSign.CANCER],
    Planet.MARS: [ZodiacSign.ARIES, ZodiacSign.SCORPIO],
    Planet.MERCURY: [ZodiacSign.GEMINI, ZodiacSign.VIRGO],
    Planet.JUPITER: [ZodiacSign.SAGITTARIUS, ZodiacSign.PISCES],
    Planet.VENUS: [ZodiacSign.TAURUS, ZodiacSign.LIBRA],
    Planet.SATURN: [ZodiacSign.CAPRICORN, ZodiacSign.AQUARIUS],
    Planet.RAHU: [ZodiacSign.AQUARIUS],
    Planet.KETU: [ZodiacSign.SCORPIO],
}

# Friendly signs for planets
FRIENDLY_SIGNS: Dict[Planet, List[ZodiacSign]] = {
    Planet.SUN: [ZodiacSign.ARIES, ZodiacSign.SAGITTARIUS, ZodiacSign.SCORPIO],
    Planet.MOON: [ZodiacSign.TAURUS, ZodiacSign.PISCES],
    Planet.MARS: [ZodiacSign.LEO, ZodiacSign.SAGITTARIUS, ZodiacSign.PISCES],
    Planet.MERCURY: [ZodiacSign.TAURUS, ZodiacSign.LEO, ZodiacSign.LIBRA],
    Planet.JUPITER: [ZodiacSign.LEO, ZodiacSign.ARIES, ZodiacSign.SCORPIO],
    Planet.VENUS: [ZodiacSign.GEMINI, ZodiacSign.CAPRICORN, ZodiacSign.AQUARIUS],
    Planet.SATURN: [ZodiacSign.TAURUS, ZodiacSign.GEMINI, ZodiacSign.VIRGO, ZodiacSign.LIBRA],
}


# =============================================================================
# SECTOR-PLANETARY MAPPING
# =============================================================================

SECTOR_PLANETARY_MAP: Dict[str, Dict] = {
    "Technology": {
        "primary_planet": Planet.MERCURY,
        "secondary_planets": [Planet.URANUS, Planet.RAHU],
        "favorable_signs": [ZodiacSign.GEMINI, ZodiacSign.VIRGO, ZodiacSign.AQUARIUS],
        "weight": 1.0,
        "tickers": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TCS.NS", "INFY.NS"],
    },
    "Finance": {
        "primary_planet": Planet.JUPITER,
        "secondary_planets": [Planet.VENUS, Planet.MERCURY],
        "favorable_signs": [ZodiacSign.SAGITTARIUS, ZodiacSign.PISCES, ZodiacSign.TAURUS],
        "weight": 1.0,
        "tickers": ["JPM", "BAC", "GS", "HDFCBANK.NS", "ICICIBANK.NS"],
    },
    "Healthcare": {
        "primary_planet": Planet.MOON,
        "secondary_planets": [Planet.JUPITER, Planet.NEPTUNE],
        "favorable_signs": [ZodiacSign.CANCER, ZodiacSign.VIRGO, ZodiacSign.PISCES],
        "weight": 0.9,
        "tickers": ["JNJ", "PFE", "UNH", "SUNPHARMA.NS", "DRREDDY.NS"],
    },
    "Energy": {
        "primary_planet": Planet.SUN,
        "secondary_planets": [Planet.MARS, Planet.PLUTO],
        "favorable_signs": [ZodiacSign.LEO, ZodiacSign.ARIES, ZodiacSign.SCORPIO],
        "weight": 1.1,
        "tickers": ["XOM", "CVX", "COP", "RELIANCE.NS", "ONGC.NS"],
    },
    "Defense": {
        "primary_planet": Planet.MARS,
        "secondary_planets": [Planet.SATURN, Planet.PLUTO],
        "favorable_signs": [ZodiacSign.ARIES, ZodiacSign.SCORPIO, ZodiacSign.CAPRICORN],
        "weight": 1.1,
        "tickers": ["LMT", "RTX", "NOC", "HAL.NS", "BEL.NS"],
    },
    "Real Estate": {
        "primary_planet": Planet.SATURN,
        "secondary_planets": [Planet.MOON, Planet.VENUS],
        "favorable_signs": [ZodiacSign.CANCER, ZodiacSign.CAPRICORN, ZodiacSign.TAURUS],
        "weight": 0.9,
        "tickers": ["AMT", "PLD", "SPG", "DLF.NS", "GODREJPROP.NS"],
    },
    "Consumer Goods": {
        "primary_planet": Planet.VENUS,
        "secondary_planets": [Planet.MOON, Planet.MERCURY],
        "favorable_signs": [ZodiacSign.TAURUS, ZodiacSign.LIBRA, ZodiacSign.CANCER],
        "weight": 0.85,
        "tickers": ["PG", "KO", "PEP", "HINDUNILVR.NS", "ITC.NS"],
    },
    "Communications": {
        "primary_planet": Planet.MERCURY,
        "secondary_planets": [Planet.URANUS, Planet.VENUS],
        "favorable_signs": [ZodiacSign.GEMINI, ZodiacSign.AQUARIUS, ZodiacSign.LIBRA],
        "weight": 0.9,
        "tickers": ["VZ", "T", "TMUS", "BHARTIARTL.NS", "IDEA.NS"],
    },
    "Mining & Metals": {
        "primary_planet": Planet.MARS,
        "secondary_planets": [Planet.SATURN, Planet.SUN],
        "favorable_signs": [ZodiacSign.ARIES, ZodiacSign.CAPRICORN, ZodiacSign.LEO],
        "weight": 1.0,
        "tickers": ["RIO", "BHP", "NEM", "TATASTEEL.NS", "HINDALCO.NS"],
    },
    "Transportation": {
        "primary_planet": Planet.MERCURY,
        "secondary_planets": [Planet.MARS, Planet.SATURN],
        "favorable_signs": [ZodiacSign.GEMINI, ZodiacSign.SAGITTARIUS, ZodiacSign.ARIES],
        "weight": 0.85,
        "tickers": ["UPS", "FDX", "DAL", "INDIGO.NS", "CONCOR.NS"],
    },
    "Agriculture": {
        "primary_planet": Planet.MOON,
        "secondary_planets": [Planet.VENUS, Planet.SATURN],
        "favorable_signs": [ZodiacSign.CANCER, ZodiacSign.TAURUS, ZodiacSign.VIRGO],
        "weight": 0.8,
        "tickers": ["ADM", "BG", "DE", "UPL.NS", "PIIND.NS"],
    },
    "Luxury & Entertainment": {
        "primary_planet": Planet.VENUS,
        "secondary_planets": [Planet.JUPITER, Planet.SUN],
        "favorable_signs": [ZodiacSign.LIBRA, ZodiacSign.TAURUS, ZodiacSign.LEO],
        "weight": 0.9,
        "tickers": ["LVMUY", "DIS", "NKE", "TITAN.NS", "PVRINOX.NS"],
    },
}


# =============================================================================
# ASPECT CONFIGURATIONS (Vedic)
# =============================================================================

# Vedic aspects: planet -> degrees of aspect (from itself)
VEDIC_ASPECTS: Dict[Planet, List[int]] = {
    Planet.SUN: [180],
    Planet.MOON: [180],
    Planet.MERCURY: [180],
    Planet.VENUS: [180],
    Planet.MARS: [90, 180, 210],  # 4th, 7th, 8th aspects
    Planet.JUPITER: [120, 180, 240],  # 5th, 7th, 9th aspects
    Planet.SATURN: [60, 180, 270],  # 3rd, 7th, 10th aspects
    Planet.RAHU: [120, 180, 240],
    Planet.KETU: [120, 180, 240],
}

# Aspect quality multipliers
ASPECT_QUALITY: Dict[str, float] = {
    "conjunction": 1.0,  # 0 degrees
    "opposition": -0.8,  # 180 degrees
    "trine": 0.7,  # 120 degrees
    "square": -0.6,  # 90 degrees
    "sextile": 0.5,  # 60 degrees
}


# =============================================================================
# SCORING WEIGHTS
# =============================================================================

@dataclass
class ScoringWeights:
    """Weights for Market Sentiment Score calculation."""
    planetary_dignity: float = 0.20
    nakshatra_influence: float = 0.15
    aspect_harmony: float = 0.20
    transit_strength: float = 0.25
    lunar_phase: float = 0.10
    retrograde_impact: float = 0.10


@dataclass
class PredictionConfig:
    """Configuration for prediction models."""
    lookback_days: int = 365
    forecast_days: int = 90
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    arima_order: Tuple[int, int, int] = (5, 1, 0)
    confidence_threshold: float = 0.65
    signal_strength_threshold: float = 0.7


# =============================================================================
# MARKET INDICES
# =============================================================================

MARKET_INDICES: Dict[str, str] = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Nikkei 225": "^N225",
}


# =============================================================================
# PLANETARY WEIGHTS FOR MARKET IMPACT
# =============================================================================

PLANETARY_MARKET_WEIGHTS: Dict[Planet, float] = {
    Planet.SUN: 0.12,
    Planet.MOON: 0.15,
    Planet.MARS: 0.10,
    Planet.MERCURY: 0.12,
    Planet.JUPITER: 0.15,
    Planet.VENUS: 0.10,
    Planet.SATURN: 0.12,
    Planet.RAHU: 0.07,
    Planet.KETU: 0.07,
}


# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================

DEFAULT_SCORING_WEIGHTS = ScoringWeights()
DEFAULT_PREDICTION_CONFIG = PredictionConfig()
