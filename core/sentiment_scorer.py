"""
Market Sentiment Scorer - Cosmic Weather Algorithm
===================================================
Synthesizes all Vedic astrological factors into a nuanced
Market Sentiment Score (0-100) reflecting the "cosmic weather".
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from config.settings import (
    Planet, ZodiacSign, NAKSHATRAS, ScoringWeights,
    PLANETARY_MARKET_WEIGHTS, DEFAULT_SCORING_WEIGHTS
)
from .planetary_engine import PlanetaryEngine, PlanetarySnapshot, LunarPhase
from .vedic_calculator import VedicCalculator, VedicAnalysis, AspectInfo


@dataclass
class SentimentComponents:
    """Breakdown of sentiment score components."""
    planetary_dignity_score: float
    nakshatra_influence_score: float
    aspect_harmony_score: float
    transit_strength_score: float
    lunar_phase_score: float
    retrograde_impact_score: float


@dataclass
class MarketSentiment:
    """Complete market sentiment analysis."""
    timestamp: datetime
    overall_score: float  # 0-100
    components: SentimentComponents
    interpretation: str
    signal: str  # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
    volatility_forecast: str  # 'low', 'moderate', 'high', 'extreme'
    key_factors: List[str]


class MarketSentimentScorer:
    """
    Calculates the Market Sentiment Score by synthesizing multiple
    Vedic astrological factors into a single 0-100 score.
    """

    # Lunar phase market correlations
    LUNAR_PHASE_SCORES = {
        "New Moon": 0.4,  # New beginnings, uncertainty
        "Waxing Crescent": 0.55,  # Building momentum
        "First Quarter": 0.6,  # Decision points
        "Waxing Gibbous": 0.7,  # Growing optimism
        "Full Moon": 0.65,  # Peak energy, reversals possible
        "Waning Gibbous": 0.55,  # Distribution phase
        "Last Quarter": 0.45,  # Correction period
        "Waning Crescent": 0.35,  # Exhaustion, bottoming
    }

    # Tithi-based market tendencies (simplified)
    TITHI_MARKET_SCORES = {
        # Shukla Paksha (bright half) - generally more bullish
        1: 0.5, 2: 0.55, 3: 0.6, 4: 0.55, 5: 0.65,  # Panchami is auspicious
        6: 0.6, 7: 0.65, 8: 0.5,  # Ashtami is challenging
        9: 0.7, 10: 0.65, 11: 0.75,  # Ekadashi is very auspicious
        12: 0.6, 13: 0.65, 14: 0.5,
        15: 0.6,  # Purnima (Full Moon)
        # Krishna Paksha (dark half) - generally more bearish
        16: 0.5, 17: 0.45, 18: 0.5, 19: 0.45, 20: 0.55,
        21: 0.5, 22: 0.55, 23: 0.4,  # Ashtami
        24: 0.5, 25: 0.45, 26: 0.55,  # Ekadashi
        27: 0.4, 28: 0.45, 29: 0.35,
        30: 0.3,  # Amavasya (New Moon)
    }

    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize the sentiment scorer.

        Args:
            weights: Optional custom scoring weights.
        """
        self.weights = weights or DEFAULT_SCORING_WEIGHTS
        self.engine = PlanetaryEngine()
        self.vedic = VedicCalculator(self.engine)

    def _calculate_dignity_score(self, analysis: VedicAnalysis) -> float:
        """
        Calculate score based on planetary dignities.

        Args:
            analysis: VedicAnalysis object.

        Returns:
            Score from 0.0 to 1.0.
        """
        total_score = 0.0
        total_weight = 0.0

        for planet, dignity in analysis.dignities.items():
            weight = PLANETARY_MARKET_WEIGHTS.get(planet, 0.1)
            total_score += dignity.strength * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _calculate_nakshatra_score(self, analysis: VedicAnalysis) -> float:
        """
        Calculate score based on nakshatra influences.

        The Moon's nakshatra has the highest impact, followed by
        other significant planets.
        """
        score = 0.0

        # Moon's nakshatra is most important (40% weight)
        moon_nakshatra = analysis.nakshatra_positions.get(Planet.MOON)
        if moon_nakshatra:
            score += moon_nakshatra.market_energy * 0.4

        # Sun's nakshatra (15% weight)
        sun_nakshatra = analysis.nakshatra_positions.get(Planet.SUN)
        if sun_nakshatra:
            score += sun_nakshatra.market_energy * 0.15

        # Jupiter and Venus nakshatras (benefics - 15% each)
        jupiter_nakshatra = analysis.nakshatra_positions.get(Planet.JUPITER)
        if jupiter_nakshatra:
            score += jupiter_nakshatra.market_energy * 0.15

        venus_nakshatra = analysis.nakshatra_positions.get(Planet.VENUS)
        if venus_nakshatra:
            score += venus_nakshatra.market_energy * 0.15

        # Consider nakshatra nature
        nature_bonus = 0.0
        for planet in [Planet.MOON, Planet.JUPITER]:
            nak_info = analysis.nakshatra_positions.get(planet)
            if nak_info:
                if nak_info.nature in ['fixed', 'light']:
                    nature_bonus += 0.05
                elif nak_info.nature in ['sharp', 'fierce']:
                    nature_bonus -= 0.05

        return min(1.0, max(0.0, score + nature_bonus))

    def _calculate_aspect_score(self, analysis: VedicAnalysis) -> float:
        """
        Calculate score based on planetary aspects.

        Harmonious aspects (trines, sextiles) boost the score,
        while challenging aspects (squares, oppositions) reduce it.
        """
        if not analysis.aspects:
            return 0.5  # Neutral if no aspects

        total_score = 0.0
        total_weight = 0.0

        for aspect in analysis.aspects:
            # Weight by planet importance
            p1_weight = PLANETARY_MARKET_WEIGHTS.get(aspect.planet1, 0.1)
            p2_weight = PLANETARY_MARKET_WEIGHTS.get(aspect.planet2, 0.1)
            combined_weight = (p1_weight + p2_weight) / 2

            # Aspect strength already includes sign (positive/negative)
            # Convert to 0-1 scale
            aspect_contribution = (aspect.strength + 1) / 2

            # Applying aspects are stronger
            if aspect.is_applying:
                combined_weight *= 1.2

            total_score += aspect_contribution * combined_weight
            total_weight += combined_weight

        base_score = total_score / total_weight if total_weight > 0 else 0.5

        # Bonus for Jupiter aspects (benefic influence)
        jupiter_aspects = [a for a in analysis.aspects
                         if a.planet1 == Planet.JUPITER or a.planet2 == Planet.JUPITER]
        benefic_bonus = sum(0.02 for a in jupiter_aspects if a.strength > 0)

        # Penalty for Mars-Saturn hard aspects
        mars_saturn = [a for a in analysis.aspects
                      if {a.planet1, a.planet2} == {Planet.MARS, Planet.SATURN}
                      and a.aspect_type in ['conjunction', 'square', 'opposition']]
        malefic_penalty = sum(0.05 for _ in mars_saturn)

        return min(1.0, max(0.0, base_score + benefic_bonus - malefic_penalty))

    def _calculate_transit_score(self, snapshot: PlanetarySnapshot) -> float:
        """
        Calculate score based on current planetary positions and transits.

        Considers the overall planetary distribution and key transits.
        """
        positions = snapshot.positions

        score = 0.5  # Start neutral

        # Check key planets in favorable/unfavorable signs
        # Jupiter in own/exalted sign: very positive
        jupiter_pos = positions[Planet.JUPITER]
        if jupiter_pos.sign in [ZodiacSign.SAGITTARIUS, ZodiacSign.PISCES, ZodiacSign.CANCER]:
            score += 0.1

        # Saturn in challenging position: negative
        saturn_pos = positions[Planet.SATURN]
        if saturn_pos.sign == ZodiacSign.ARIES:  # Debilitated
            score -= 0.08

        # Venus well-placed: positive for markets
        venus_pos = positions[Planet.VENUS]
        if venus_pos.sign in [ZodiacSign.TAURUS, ZodiacSign.LIBRA, ZodiacSign.PISCES]:
            score += 0.06

        # Mercury well-placed: positive for communication/tech
        mercury_pos = positions[Planet.MERCURY]
        if mercury_pos.sign in [ZodiacSign.GEMINI, ZodiacSign.VIRGO]:
            score += 0.05

        # Check for planets in Gandanta (junction points)
        # These are sensitive degrees at water-fire sign junctions
        gandanta_penalty = 0.0
        for planet, pos in positions.items():
            # Check if within 3 degrees of sign junction
            if pos.sign_degree < 3 or pos.sign_degree > 27:
                if pos.sign in [ZodiacSign.CANCER, ZodiacSign.SCORPIO, ZodiacSign.PISCES,
                               ZodiacSign.LEO, ZodiacSign.SAGITTARIUS, ZodiacSign.ARIES]:
                    gandanta_penalty += 0.02 * PLANETARY_MARKET_WEIGHTS.get(planet, 0.1)

        score -= gandanta_penalty

        return min(1.0, max(0.0, score))

    def _calculate_lunar_score(self, lunar_phase: LunarPhase) -> float:
        """
        Calculate score based on lunar phase and tithi.
        """
        # Base score from phase
        phase_score = self.LUNAR_PHASE_SCORES.get(lunar_phase.phase_name, 0.5)

        # Tithi adjustment
        tithi_score = self.TITHI_MARKET_SCORES.get(lunar_phase.tithi, 0.5)

        # Combine with more weight to tithi (more precise)
        combined = phase_score * 0.4 + tithi_score * 0.6

        # Illumination factor (markets sometimes correlate with brightness)
        if lunar_phase.is_waxing:
            illumination_boost = lunar_phase.illumination * 0.1
        else:
            illumination_boost = -lunar_phase.illumination * 0.05

        return min(1.0, max(0.0, combined + illumination_boost))

    def _calculate_retrograde_score(self, snapshot: PlanetarySnapshot) -> float:
        """
        Calculate impact of retrograde planets.

        Retrograde planets increase uncertainty and often correlate
        with market reversals or stagnation.
        """
        score = 1.0  # Start at maximum (no retrogrades = best)

        for planet, pos in snapshot.positions.items():
            if pos.is_retrograde:
                # Different impacts for different planets
                if planet == Planet.MERCURY:
                    score -= 0.15  # Mercury Rx: communication issues, tech glitches
                elif planet == Planet.VENUS:
                    score -= 0.12  # Venus Rx: value reassessment
                elif planet == Planet.MARS:
                    score -= 0.10  # Mars Rx: energy blockage
                elif planet == Planet.JUPITER:
                    score -= 0.08  # Jupiter Rx: growth slows
                elif planet == Planet.SATURN:
                    score -= 0.06  # Saturn Rx: delays, restructuring
                elif planet in [Planet.URANUS, Planet.NEPTUNE, Planet.PLUTO]:
                    score -= 0.03  # Outer planets: subtle effects

        return max(0.0, score)

    def calculate_sentiment(self, dt: datetime) -> MarketSentiment:
        """
        Calculate the complete Market Sentiment Score.

        Args:
            dt: Datetime to analyze.

        Returns:
            MarketSentiment with overall score and breakdown.
        """
        # Get planetary data
        snapshot = self.engine.get_all_positions(dt)
        analysis = self.vedic.get_complete_analysis(dt)

        # Calculate component scores
        dignity_score = self._calculate_dignity_score(analysis)
        nakshatra_score = self._calculate_nakshatra_score(analysis)
        aspect_score = self._calculate_aspect_score(analysis)
        transit_score = self._calculate_transit_score(snapshot)
        lunar_score = self._calculate_lunar_score(snapshot.lunar_phase)
        retrograde_score = self._calculate_retrograde_score(snapshot)

        components = SentimentComponents(
            planetary_dignity_score=dignity_score,
            nakshatra_influence_score=nakshatra_score,
            aspect_harmony_score=aspect_score,
            transit_strength_score=transit_score,
            lunar_phase_score=lunar_score,
            retrograde_impact_score=retrograde_score
        )

        # Weighted combination
        weighted_score = (
            dignity_score * self.weights.planetary_dignity +
            nakshatra_score * self.weights.nakshatra_influence +
            aspect_score * self.weights.aspect_harmony +
            transit_score * self.weights.transit_strength +
            lunar_score * self.weights.lunar_phase +
            retrograde_score * self.weights.retrograde_impact
        )

        # Convert to 0-100 scale
        overall_score = weighted_score * 100

        # Determine signal
        if overall_score >= 75:
            signal = 'strong_buy'
        elif overall_score >= 60:
            signal = 'buy'
        elif overall_score >= 40:
            signal = 'neutral'
        elif overall_score >= 25:
            signal = 'sell'
        else:
            signal = 'strong_sell'

        # Determine volatility forecast
        volatility = self._assess_volatility(snapshot, analysis)

        # Generate interpretation
        interpretation = self._generate_interpretation(
            overall_score, components, analysis, snapshot
        )

        # Identify key factors
        key_factors = self._identify_key_factors(analysis, snapshot)

        return MarketSentiment(
            timestamp=dt,
            overall_score=round(overall_score, 2),
            components=components,
            interpretation=interpretation,
            signal=signal,
            volatility_forecast=volatility,
            key_factors=key_factors
        )

    def _assess_volatility(self, snapshot: PlanetarySnapshot,
                          analysis: VedicAnalysis) -> str:
        """Assess expected market volatility."""
        volatility_score = 0

        # Count retrogrades
        retrogrades = sum(1 for p in snapshot.positions.values() if p.is_retrograde)
        volatility_score += retrogrades * 0.1

        # Check for challenging aspects
        hard_aspects = [a for a in analysis.aspects
                       if a.aspect_type in ['square', 'opposition']
                       and a.orb < 3]
        volatility_score += len(hard_aspects) * 0.15

        # Mars aspects increase volatility
        mars_aspects = [a for a in analysis.aspects
                       if a.planet1 == Planet.MARS or a.planet2 == Planet.MARS]
        volatility_score += len(mars_aspects) * 0.1

        # Rahu/Ketu involvement
        node_aspects = [a for a in analysis.aspects
                       if a.planet1 in [Planet.RAHU, Planet.KETU]
                       or a.planet2 in [Planet.RAHU, Planet.KETU]]
        volatility_score += len(node_aspects) * 0.12

        # Near full/new moon
        if snapshot.lunar_phase.phase_name in ["New Moon", "Full Moon"]:
            volatility_score += 0.2

        if volatility_score >= 0.7:
            return 'extreme'
        elif volatility_score >= 0.5:
            return 'high'
        elif volatility_score >= 0.3:
            return 'moderate'
        else:
            return 'low'

    def _generate_interpretation(self, score: float, components: SentimentComponents,
                                analysis: VedicAnalysis, snapshot: PlanetarySnapshot) -> str:
        """Generate a human-readable interpretation."""
        parts = []

        # Overall assessment
        if score >= 70:
            parts.append("Strong bullish cosmic alignment.")
        elif score >= 55:
            parts.append("Mildly favorable cosmic conditions.")
        elif score >= 45:
            parts.append("Mixed cosmic signals suggest caution.")
        elif score >= 30:
            parts.append("Challenging cosmic environment.")
        else:
            parts.append("Significant cosmic headwinds present.")

        # Add specific insights
        if components.planetary_dignity_score > 0.7:
            parts.append("Planets are well-positioned for growth.")
        elif components.planetary_dignity_score < 0.4:
            parts.append("Planetary weaknesses may limit upside.")

        if components.retrograde_impact_score < 0.6:
            retro_planets = [p.value for p, pos in snapshot.positions.items()
                           if pos.is_retrograde and p not in [Planet.RAHU, Planet.KETU]]
            if retro_planets:
                parts.append(f"Retrograde {', '.join(retro_planets)} suggests review/reversal.")

        # Yogas
        if analysis.active_yogas:
            if any('wealth' in y.lower() or 'strong' in y.lower() for y in analysis.active_yogas):
                parts.append("Positive wealth yogas active.")
            if any('obstacle' in y.lower() or 'challenge' in y.lower() for y in analysis.active_yogas):
                parts.append("Some challenging combinations present.")

        # Lunar phase
        parts.append(f"Lunar phase: {snapshot.lunar_phase.phase_name} ({snapshot.lunar_phase.tithi_name}).")

        return " ".join(parts)

    def _identify_key_factors(self, analysis: VedicAnalysis,
                             snapshot: PlanetarySnapshot) -> List[str]:
        """Identify the most significant factors affecting the score."""
        factors = []

        # Check dignities
        for planet, dignity in analysis.dignities.items():
            if dignity.state == 'exalted':
                factors.append(f"{planet.value} exalted (very positive)")
            elif dignity.state == 'debilitated':
                factors.append(f"{planet.value} debilitated (challenging)")

        # Check significant aspects
        for aspect in analysis.aspects:
            if aspect.orb < 2:  # Tight aspects
                if aspect.strength > 0.5:
                    factors.append(f"{aspect.planet1.value}-{aspect.planet2.value} {aspect.aspect_type} (harmonious)")
                elif aspect.strength < -0.5:
                    factors.append(f"{aspect.planet1.value}-{aspect.planet2.value} {aspect.aspect_type} (tense)")

        # Retrogrades
        for planet, pos in snapshot.positions.items():
            if pos.is_retrograde and planet not in [Planet.RAHU, Planet.KETU]:
                factors.append(f"{planet.value} retrograde")

        # Active yogas (limit to top 3)
        factors.extend(analysis.active_yogas[:3])

        return factors[:8]  # Limit to top 8 factors

    def get_sentiment_range(self, start_dt: datetime, end_dt: datetime,
                           interval_hours: int = 24) -> List[MarketSentiment]:
        """
        Calculate sentiment scores over a date range.

        Args:
            start_dt: Start datetime.
            end_dt: End datetime.
            interval_hours: Hours between calculations.

        Returns:
            List of MarketSentiment objects.
        """
        sentiments = []
        current = start_dt

        while current <= end_dt:
            sentiments.append(self.calculate_sentiment(current))
            current += timedelta(hours=interval_hours)

        return sentiments

    def find_optimal_dates(self, start_dt: datetime, end_dt: datetime,
                          signal_type: str = 'buy',
                          min_score: float = 60.0) -> List[Tuple[datetime, float]]:
        """
        Find dates with optimal sentiment for a given signal type.

        Args:
            start_dt: Start of search range.
            end_dt: End of search range.
            signal_type: 'buy' or 'sell'.
            min_score: Minimum score threshold.

        Returns:
            List of (datetime, score) tuples.
        """
        sentiments = self.get_sentiment_range(start_dt, end_dt)

        optimal_dates = []
        for sentiment in sentiments:
            if signal_type == 'buy' and sentiment.overall_score >= min_score:
                optimal_dates.append((sentiment.timestamp, sentiment.overall_score))
            elif signal_type == 'sell' and sentiment.overall_score <= (100 - min_score):
                optimal_dates.append((sentiment.timestamp, sentiment.overall_score))

        # Sort by score (descending for buy, ascending for sell)
        reverse = signal_type == 'buy'
        optimal_dates.sort(key=lambda x: x[1], reverse=reverse)

        return optimal_dates
