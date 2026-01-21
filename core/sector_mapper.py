"""
Sector Mapper - Planetary-Industry Sector Mapping
=================================================
Maps planetary energies to modern industry sectors and identifies
Golden Phase (Buy) and Stress Phase (Sell) periods.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.settings import (
    Planet, ZodiacSign, SECTOR_PLANETARY_MAP, PLANETARY_MARKET_WEIGHTS
)
from .planetary_engine import PlanetaryEngine, PlanetarySnapshot
from .vedic_calculator import VedicCalculator


class SectorPhase(Enum):
    GOLDEN = "Golden Phase"  # Strong buy signal
    FAVORABLE = "Favorable"  # Mild buy signal
    NEUTRAL = "Neutral"
    STRESS = "Stress Phase"  # Mild sell signal
    CRITICAL = "Critical Phase"  # Strong sell signal


@dataclass
class SectorAnalysis:
    """Analysis result for a single sector."""
    sector_name: str
    phase: SectorPhase
    score: float  # 0-100
    primary_planet_status: str
    secondary_influences: List[str]
    key_transits: List[str]
    recommended_action: str
    confidence: float  # 0-1


@dataclass
class SectorForecast:
    """Forecast for sector performance over time."""
    sector_name: str
    current_phase: SectorPhase
    current_score: float
    upcoming_shifts: List[Tuple[datetime, SectorPhase, str]]  # (date, new_phase, reason)
    golden_periods: List[Tuple[datetime, datetime]]  # (start, end)
    stress_periods: List[Tuple[datetime, datetime]]
    best_entry_dates: List[datetime]
    worst_dates: List[datetime]


class SectorMapper:
    """
    Maps planetary energies to industry sectors and analyzes
    their cosmic favorability.
    """

    def __init__(self):
        self.engine = PlanetaryEngine()
        self.vedic = VedicCalculator(self.engine)
        self.sector_map = SECTOR_PLANETARY_MAP

    def _get_planet_sector_score(self, planet: Planet, snapshot: PlanetarySnapshot,
                                sector_config: Dict) -> float:
        """
        Calculate how well a planet supports its sector.

        Args:
            planet: Planet to evaluate.
            snapshot: Current planetary positions.
            sector_config: Sector configuration from SECTOR_PLANETARY_MAP.

        Returns:
            Score from 0.0 to 1.0.
        """
        pos = snapshot.positions[planet]
        score = 0.5  # Start neutral

        # Check if in favorable sign for this sector
        favorable_signs = sector_config.get('favorable_signs', [])
        if pos.sign in favorable_signs:
            score += 0.2

        # Get dignity
        dignity = self.vedic.get_planetary_dignity(planet, pos.sign)
        score += (dignity.strength - 0.5) * 0.3

        # Retrograde penalty
        if pos.is_retrograde:
            score -= 0.15

        # Sign-specific bonuses
        if dignity.state == 'exalted':
            score += 0.15
        elif dignity.state == 'debilitated':
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _check_sector_aspects(self, sector_config: Dict,
                             snapshot: PlanetarySnapshot) -> Tuple[float, List[str]]:
        """
        Check aspects affecting a sector's ruling planets.

        Returns:
            Tuple of (aspect_score, list of aspect descriptions).
        """
        primary = sector_config['primary_planet']
        secondaries = sector_config.get('secondary_planets', [])
        all_rulers = [primary] + secondaries

        aspect_score = 0.0
        descriptions = []

        for ruler in all_rulers:
            ruler_pos = snapshot.positions[ruler]

            # Check aspects from benefics (Jupiter, Venus)
            for benefic in [Planet.JUPITER, Planet.VENUS]:
                if benefic == ruler:
                    continue
                benefic_pos = snapshot.positions[benefic]
                aspect = self.vedic.calculate_aspect(ruler_pos, benefic_pos)
                if aspect:
                    if aspect.aspect_type in ['conjunction', 'trine', 'sextile']:
                        aspect_score += 0.1
                        descriptions.append(f"{benefic.value} {aspect.aspect_type} {ruler.value} (supportive)")

            # Check aspects from malefics (Mars, Saturn, Rahu)
            for malefic in [Planet.MARS, Planet.SATURN, Planet.RAHU]:
                if malefic == ruler:
                    continue
                malefic_pos = snapshot.positions[malefic]
                aspect = self.vedic.calculate_aspect(ruler_pos, malefic_pos)
                if aspect:
                    if aspect.aspect_type in ['conjunction', 'square', 'opposition']:
                        aspect_score -= 0.1
                        descriptions.append(f"{malefic.value} {aspect.aspect_type} {ruler.value} (challenging)")

        return aspect_score, descriptions

    def analyze_sector(self, sector_name: str, dt: datetime) -> SectorAnalysis:
        """
        Analyze a single sector's cosmic favorability.

        Args:
            sector_name: Name of sector to analyze.
            dt: Datetime for analysis.

        Returns:
            SectorAnalysis with complete sector evaluation.
        """
        if sector_name not in self.sector_map:
            raise ValueError(f"Unknown sector: {sector_name}")

        sector_config = self.sector_map[sector_name]
        snapshot = self.engine.get_all_positions(dt)

        # Calculate primary planet score
        primary_planet = sector_config['primary_planet']
        primary_score = self._get_planet_sector_score(primary_planet, snapshot, sector_config)

        # Calculate secondary planet scores
        secondary_scores = []
        secondary_planets = sector_config.get('secondary_planets', [])
        for planet in secondary_planets:
            score = self._get_planet_sector_score(planet, snapshot, sector_config)
            secondary_scores.append(score)

        # Average secondary influence
        secondary_avg = sum(secondary_scores) / len(secondary_scores) if secondary_scores else 0.5

        # Get aspect influences
        aspect_score, aspect_descriptions = self._check_sector_aspects(sector_config, snapshot)

        # Calculate composite score
        # Primary planet: 50%, Secondary planets: 30%, Aspects: 20%
        composite_score = (
            primary_score * 0.5 +
            secondary_avg * 0.3 +
            (0.5 + aspect_score) * 0.2
        )

        # Apply sector weight
        sector_weight = sector_config.get('weight', 1.0)
        final_score = composite_score * 100 * sector_weight

        # Determine phase
        if final_score >= 70:
            phase = SectorPhase.GOLDEN
        elif final_score >= 55:
            phase = SectorPhase.FAVORABLE
        elif final_score >= 45:
            phase = SectorPhase.NEUTRAL
        elif final_score >= 30:
            phase = SectorPhase.STRESS
        else:
            phase = SectorPhase.CRITICAL

        # Generate primary planet status
        primary_pos = snapshot.positions[primary_planet]
        dignity = self.vedic.get_planetary_dignity(primary_planet, primary_pos.sign)
        retro_status = " (retrograde)" if primary_pos.is_retrograde else ""
        primary_status = f"{primary_planet.value} in {primary_pos.sign.name} - {dignity.state}{retro_status}"

        # Generate secondary influences
        secondary_influences = []
        for planet in secondary_planets:
            pos = snapshot.positions[planet]
            d = self.vedic.get_planetary_dignity(planet, pos.sign)
            retro = " Rx" if pos.is_retrograde else ""
            secondary_influences.append(f"{planet.value}: {pos.sign.name} ({d.state}){retro}")

        # Recommended action
        if phase == SectorPhase.GOLDEN:
            action = "Strong accumulation opportunity. Consider increasing exposure."
        elif phase == SectorPhase.FAVORABLE:
            action = "Favorable conditions. Gradual position building recommended."
        elif phase == SectorPhase.NEUTRAL:
            action = "Hold existing positions. Wait for clearer signals."
        elif phase == SectorPhase.STRESS:
            action = "Caution advised. Consider reducing exposure."
        else:
            action = "High risk period. Defensive positioning recommended."

        # Confidence based on aspect clarity
        confidence = 0.6 + (abs(final_score - 50) / 100) * 0.4

        return SectorAnalysis(
            sector_name=sector_name,
            phase=phase,
            score=round(final_score, 2),
            primary_planet_status=primary_status,
            secondary_influences=secondary_influences,
            key_transits=aspect_descriptions[:5],
            recommended_action=action,
            confidence=round(confidence, 2)
        )

    def analyze_all_sectors(self, dt: datetime) -> Dict[str, SectorAnalysis]:
        """
        Analyze all sectors at once.

        Args:
            dt: Datetime for analysis.

        Returns:
            Dictionary mapping sector names to their analyses.
        """
        results = {}
        for sector_name in self.sector_map.keys():
            results[sector_name] = self.analyze_sector(sector_name, dt)
        return results

    def get_sector_rankings(self, dt: datetime) -> List[Tuple[str, SectorPhase, float]]:
        """
        Get sectors ranked by favorability.

        Args:
            dt: Datetime for analysis.

        Returns:
            List of (sector_name, phase, score) tuples, sorted by score.
        """
        analyses = self.analyze_all_sectors(dt)
        rankings = [(name, a.phase, a.score) for name, a in analyses.items()]
        rankings.sort(key=lambda x: x[2], reverse=True)
        return rankings

    def forecast_sector(self, sector_name: str, start_dt: datetime,
                       days: int = 90) -> SectorForecast:
        """
        Generate a sector forecast over a time period.

        Args:
            sector_name: Sector to forecast.
            start_dt: Start of forecast period.
            days: Number of days to forecast.

        Returns:
            SectorForecast with detailed projections.
        """
        end_dt = start_dt + timedelta(days=days)

        # Get current analysis
        current = self.analyze_sector(sector_name, start_dt)

        # Track phase changes
        phase_history = []
        golden_periods = []
        stress_periods = []
        best_dates = []
        worst_dates = []

        current_phase = current.phase
        phase_start = start_dt

        prev_analysis = current
        daily_scores = [(start_dt, current.score)]

        # Analyze each day
        current_dt = start_dt + timedelta(days=1)
        while current_dt <= end_dt:
            analysis = self.analyze_sector(sector_name, current_dt)
            daily_scores.append((current_dt, analysis.score))

            # Check for phase change
            if analysis.phase != current_phase:
                # Record the shift
                reason = self._determine_phase_change_reason(
                    sector_name, prev_analysis, analysis, current_dt
                )
                phase_history.append((current_dt, analysis.phase, reason))

                # Track golden/stress periods
                if current_phase == SectorPhase.GOLDEN:
                    golden_periods.append((phase_start, current_dt))
                elif current_phase in [SectorPhase.STRESS, SectorPhase.CRITICAL]:
                    stress_periods.append((phase_start, current_dt))

                current_phase = analysis.phase
                phase_start = current_dt

            # Track best/worst dates
            if analysis.score >= 75:
                best_dates.append(current_dt)
            elif analysis.score <= 25:
                worst_dates.append(current_dt)

            prev_analysis = analysis
            current_dt += timedelta(days=1)

        # Close any open periods
        if current_phase == SectorPhase.GOLDEN:
            golden_periods.append((phase_start, end_dt))
        elif current_phase in [SectorPhase.STRESS, SectorPhase.CRITICAL]:
            stress_periods.append((phase_start, end_dt))

        # Find optimal entry dates (local maxima in golden periods)
        optimal_entries = self._find_optimal_entries(daily_scores, golden_periods)

        return SectorForecast(
            sector_name=sector_name,
            current_phase=current.phase,
            current_score=current.score,
            upcoming_shifts=phase_history,
            golden_periods=golden_periods,
            stress_periods=stress_periods,
            best_entry_dates=optimal_entries[:10],
            worst_dates=worst_dates[:10]
        )

    def _determine_phase_change_reason(self, sector_name: str,
                                       prev: SectorAnalysis, curr: SectorAnalysis,
                                       dt: datetime) -> str:
        """Determine the reason for a phase change."""
        config = self.sector_map[sector_name]
        primary = config['primary_planet']

        snapshot = self.engine.get_all_positions(dt)
        prev_snapshot = self.engine.get_all_positions(dt - timedelta(days=1))

        # Check for sign change
        if snapshot.positions[primary].sign != prev_snapshot.positions[primary].sign:
            return f"{primary.value} entered {snapshot.positions[primary].sign.name}"

        # Check for retrograde change
        if snapshot.positions[primary].is_retrograde != prev_snapshot.positions[primary].is_retrograde:
            if snapshot.positions[primary].is_retrograde:
                return f"{primary.value} turned retrograde"
            else:
                return f"{primary.value} turned direct"

        # Check secondary planets
        for planet in config.get('secondary_planets', []):
            if snapshot.positions[planet].sign != prev_snapshot.positions[planet].sign:
                return f"{planet.value} entered {snapshot.positions[planet].sign.name}"

        return "Gradual shift in planetary influences"

    def _find_optimal_entries(self, daily_scores: List[Tuple[datetime, float]],
                             golden_periods: List[Tuple[datetime, datetime]]) -> List[datetime]:
        """Find optimal entry points within golden periods."""
        optimal = []

        for period_start, period_end in golden_periods:
            # Get scores within this period
            period_scores = [(dt, score) for dt, score in daily_scores
                            if period_start <= dt <= period_end]

            if not period_scores:
                continue

            # Find local maximum
            max_score = max(s for _, s in period_scores)
            for dt, score in period_scores:
                if score == max_score:
                    optimal.append(dt)
                    break

        return sorted(optimal)

    def get_sector_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Find which sector a ticker belongs to.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Sector name or None if not found.
        """
        for sector_name, config in self.sector_map.items():
            if ticker.upper() in [t.upper() for t in config.get('tickers', [])]:
                return sector_name
        return None

    def get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get all tickers in a sector."""
        if sector_name not in self.sector_map:
            return []
        return self.sector_map[sector_name].get('tickers', [])

    def get_all_sectors_forecast(self, start_dt: datetime,
                                days: int = 90) -> Dict[str, SectorForecast]:
        """
        Generate forecasts for all sectors.

        Args:
            start_dt: Start of forecast period.
            days: Number of days to forecast.

        Returns:
            Dictionary mapping sector names to forecasts.
        """
        forecasts = {}
        for sector_name in self.sector_map.keys():
            forecasts[sector_name] = self.forecast_sector(sector_name, start_dt, days)
        return forecasts
