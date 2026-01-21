"""
Vedic Calculator - Advanced Vedic Astrology Calculations
========================================================
Handles nakshatras, planetary dignities, aspects, yogas, and transits.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from config.settings import (
    Planet, ZodiacSign, NAKSHATRAS, EXALTATION, DEBILITATION,
    OWN_SIGNS, FRIENDLY_SIGNS, VEDIC_ASPECTS, ASPECT_QUALITY,
    PLANETARY_MARKET_WEIGHTS
)
from .planetary_engine import PlanetaryEngine, PlanetaryPosition, PlanetarySnapshot


@dataclass
class PlanetaryDignity:
    """Represents a planet's dignity state."""
    planet: Planet
    state: str  # 'exalted', 'own', 'friendly', 'neutral', 'enemy', 'debilitated'
    strength: float  # 0.0 to 1.0
    description: str


@dataclass
class AspectInfo:
    """Information about a planetary aspect."""
    planet1: Planet
    planet2: Planet
    aspect_type: str  # 'conjunction', 'opposition', 'trine', 'square', 'sextile'
    exact_degrees: float
    orb: float  # How far from exact
    is_applying: bool  # Whether planets are moving toward exact aspect
    strength: float  # -1.0 to 1.0 (negative for challenging aspects)


@dataclass
class NakshatraInfo:
    """Detailed nakshatra information."""
    index: int
    name: str
    ruler: Planet
    pada: int
    nature: str
    market_energy: float
    degree_in_nakshatra: float


@dataclass
class TransitEvent:
    """A significant transit event."""
    date: datetime
    event_type: str  # 'sign_change', 'nakshatra_change', 'retrograde_start', 'retrograde_end', 'aspect'
    planet: Planet
    description: str
    market_impact: float  # -1.0 to 1.0


@dataclass
class VedicAnalysis:
    """Complete Vedic analysis for a moment in time."""
    timestamp: datetime
    dignities: Dict[Planet, PlanetaryDignity]
    aspects: List[AspectInfo]
    nakshatra_positions: Dict[Planet, NakshatraInfo]
    active_yogas: List[str]
    overall_strength: float


class VedicCalculator:
    """
    Advanced Vedic astrology calculator.

    Computes planetary dignities, aspects, nakshatras, and yogas
    for market analysis purposes.
    """

    # Aspect orbs (degrees of tolerance)
    ASPECT_ORBS = {
        'conjunction': 8.0,
        'opposition': 8.0,
        'trine': 7.0,
        'square': 7.0,
        'sextile': 5.0,
    }

    # Aspect angles
    ASPECT_ANGLES = {
        0: 'conjunction',
        60: 'sextile',
        90: 'square',
        120: 'trine',
        180: 'opposition',
    }

    def __init__(self, engine: Optional[PlanetaryEngine] = None):
        """
        Initialize the Vedic calculator.

        Args:
            engine: Optional PlanetaryEngine instance to use.
        """
        self.engine = engine or PlanetaryEngine()

    def get_planetary_dignity(self, planet: Planet, sign: ZodiacSign) -> PlanetaryDignity:
        """
        Calculate the dignity state of a planet in a sign.

        Args:
            planet: Planet to analyze.
            sign: Zodiac sign the planet is in.

        Returns:
            PlanetaryDignity with state and strength.
        """
        # Check exaltation
        if EXALTATION.get(planet) == sign:
            return PlanetaryDignity(
                planet=planet,
                state='exalted',
                strength=1.0,
                description=f"{planet.value} is exalted in {sign.name}, extremely powerful"
            )

        # Check debilitation
        if DEBILITATION.get(planet) == sign:
            return PlanetaryDignity(
                planet=planet,
                state='debilitated',
                strength=0.2,
                description=f"{planet.value} is debilitated in {sign.name}, very weak"
            )

        # Check own sign
        own_signs = OWN_SIGNS.get(planet, [])
        if sign in own_signs:
            return PlanetaryDignity(
                planet=planet,
                state='own',
                strength=0.85,
                description=f"{planet.value} is in its own sign {sign.name}, strong"
            )

        # Check friendly sign
        friendly = FRIENDLY_SIGNS.get(planet, [])
        if sign in friendly:
            return PlanetaryDignity(
                planet=planet,
                state='friendly',
                strength=0.7,
                description=f"{planet.value} is in friendly sign {sign.name}, favorable"
            )

        # Check if enemy sign (debilitation lord's signs minus own)
        debil_sign = DEBILITATION.get(planet)
        if debil_sign:
            # Find who owns the debilitation sign
            for p, signs in OWN_SIGNS.items():
                if debil_sign in signs and sign in signs:
                    return PlanetaryDignity(
                        planet=planet,
                        state='enemy',
                        strength=0.35,
                        description=f"{planet.value} is in enemy sign {sign.name}, challenged"
                    )

        # Default: neutral
        return PlanetaryDignity(
            planet=planet,
            state='neutral',
            strength=0.5,
            description=f"{planet.value} is neutral in {sign.name}"
        )

    def get_nakshatra_info(self, position: PlanetaryPosition) -> NakshatraInfo:
        """
        Get detailed nakshatra information for a planetary position.

        Args:
            position: PlanetaryPosition to analyze.

        Returns:
            NakshatraInfo with detailed nakshatra data.
        """
        nakshatra = NAKSHATRAS[position.nakshatra_index]
        nakshatra_size = 360 / 27

        degree_in_nakshatra = position.longitude % nakshatra_size

        return NakshatraInfo(
            index=position.nakshatra_index,
            name=nakshatra['name'],
            ruler=nakshatra['ruler'],
            pada=position.nakshatra_pada,
            nature=nakshatra['nature'],
            market_energy=nakshatra['market_energy'],
            degree_in_nakshatra=degree_in_nakshatra
        )

    def calculate_aspect(self, pos1: PlanetaryPosition, pos2: PlanetaryPosition) -> Optional[AspectInfo]:
        """
        Calculate if two planets are in aspect.

        Args:
            pos1: First planet's position.
            pos2: Second planet's position.

        Returns:
            AspectInfo if planets are in aspect, None otherwise.
        """
        # Calculate angular separation
        diff = pos1.longitude - pos2.longitude
        if diff < 0:
            diff += 360
        if diff > 180:
            diff = 360 - diff

        # Check each aspect type
        for angle, aspect_type in self.ASPECT_ANGLES.items():
            orb = self.ASPECT_ORBS[aspect_type]
            deviation = abs(diff - angle)

            if deviation <= orb:
                # Determine if applying or separating
                # Positive combined speed means separating, negative means applying
                relative_speed = pos1.speed - pos2.speed
                is_applying = (relative_speed < 0 and diff < angle) or (relative_speed > 0 and diff > angle)

                # Calculate strength based on orb tightness
                orb_strength = 1.0 - (deviation / orb)

                # Get base aspect quality
                quality = ASPECT_QUALITY.get(aspect_type, 0.0)
                strength = quality * orb_strength

                return AspectInfo(
                    planet1=pos1.planet,
                    planet2=pos2.planet,
                    aspect_type=aspect_type,
                    exact_degrees=angle,
                    orb=deviation,
                    is_applying=is_applying,
                    strength=strength
                )

        return None

    def get_all_aspects(self, snapshot: PlanetarySnapshot) -> List[AspectInfo]:
        """
        Calculate all planetary aspects in a snapshot.

        Args:
            snapshot: PlanetarySnapshot to analyze.

        Returns:
            List of AspectInfo for all active aspects.
        """
        aspects = []
        planets = list(snapshot.positions.keys())

        for i, p1 in enumerate(planets):
            for p2 in planets[i+1:]:
                pos1 = snapshot.positions[p1]
                pos2 = snapshot.positions[p2]
                aspect = self.calculate_aspect(pos1, pos2)
                if aspect:
                    aspects.append(aspect)

        return aspects

    def detect_yogas(self, snapshot: PlanetarySnapshot) -> List[str]:
        """
        Detect active yogas (planetary combinations) for market influence.

        Args:
            snapshot: PlanetarySnapshot to analyze.

        Returns:
            List of active yoga names with descriptions.
        """
        yogas = []
        positions = snapshot.positions

        # Gajakesari Yoga: Jupiter in kendra (1, 4, 7, 10) from Moon
        moon_sign = positions[Planet.MOON].sign.value
        jupiter_sign = positions[Planet.JUPITER].sign.value
        sign_diff = (jupiter_sign - moon_sign) % 12
        if sign_diff in [0, 3, 6, 9]:
            yogas.append("Gajakesari Yoga (Jupiter-Moon): Strong financial prospects")

        # Budhaditya Yoga: Mercury conjunct Sun (within 15 degrees)
        sun_lon = positions[Planet.SUN].longitude
        mercury_lon = positions[Planet.MERCURY].longitude
        diff = abs(sun_lon - mercury_lon)
        if diff > 180:
            diff = 360 - diff
        if diff < 15:
            yogas.append("Budhaditya Yoga (Sun-Mercury): Favorable for tech/communications")

        # Shubha Kartari: Benefics (Jupiter, Venus) flanking a house/planet
        venus_sign = positions[Planet.VENUS].sign.value
        if (jupiter_sign == (moon_sign + 1) % 12 and venus_sign == (moon_sign - 1) % 12) or \
           (venus_sign == (moon_sign + 1) % 12 and jupiter_sign == (moon_sign - 1) % 12):
            yogas.append("Shubha Kartari Yoga: Protected financial environment")

        # Paapa Kartari: Malefics (Mars, Saturn) flanking - negative
        mars_sign = positions[Planet.MARS].sign.value
        saturn_sign = positions[Planet.SATURN].sign.value
        if (mars_sign == (moon_sign + 1) % 12 and saturn_sign == (moon_sign - 1) % 12) or \
           (saturn_sign == (moon_sign + 1) % 12 and mars_sign == (moon_sign - 1) % 12):
            yogas.append("Paapa Kartari Yoga: Constrained market conditions")

        # Vesi Yoga: Planet (not Moon) in 2nd from Sun
        sun_sign = positions[Planet.SUN].sign.value
        for planet in [Planet.MARS, Planet.MERCURY, Planet.JUPITER, Planet.VENUS, Planet.SATURN]:
            if positions[planet].sign.value == (sun_sign + 1) % 12:
                yogas.append(f"Vesi Yoga ({planet.value}): Enhanced market activity")
                break

        # Vosi Yoga: Planet (not Moon) in 12th from Sun
        for planet in [Planet.MARS, Planet.MERCURY, Planet.JUPITER, Planet.VENUS, Planet.SATURN]:
            if positions[planet].sign.value == (sun_sign - 1) % 12:
                yogas.append(f"Vosi Yoga ({planet.value}): Strategic positioning opportunity")
                break

        # Multiple retrograde planets - market uncertainty
        retrogrades = [p for p in positions.values() if p.is_retrograde]
        if len(retrogrades) >= 3:
            yogas.append(f"Multiple Retrograde ({len(retrogrades)} planets): Market volatility likely")

        # Jupiter-Saturn aspect - major economic cycles
        jup_sat_aspect = self.calculate_aspect(positions[Planet.JUPITER], positions[Planet.SATURN])
        if jup_sat_aspect:
            if jup_sat_aspect.aspect_type == 'conjunction':
                yogas.append("Jupiter-Saturn Conjunction: Major economic cycle shift")
            elif jup_sat_aspect.aspect_type == 'opposition':
                yogas.append("Jupiter-Saturn Opposition: Economic tension and rebalancing")

        # Venus-Jupiter conjunction - wealth combination
        ven_jup_aspect = self.calculate_aspect(positions[Planet.VENUS], positions[Planet.JUPITER])
        if ven_jup_aspect and ven_jup_aspect.aspect_type == 'conjunction':
            yogas.append("Venus-Jupiter Conjunction: Wealth accumulation period")

        # Mars-Saturn aspect - frustration and delays
        mars_sat_aspect = self.calculate_aspect(positions[Planet.MARS], positions[Planet.SATURN])
        if mars_sat_aspect and mars_sat_aspect.aspect_type in ['conjunction', 'square', 'opposition']:
            yogas.append("Mars-Saturn Challenging Aspect: Obstacles and market friction")

        return yogas

    def calculate_transit_events(self, start_dt: datetime, end_dt: datetime) -> List[TransitEvent]:
        """
        Calculate significant transit events over a date range.

        Args:
            start_dt: Start of range.
            end_dt: End of range.

        Returns:
            List of TransitEvent objects.
        """
        events = []
        current = start_dt
        prev_snapshot = None

        while current <= end_dt:
            snapshot = self.engine.get_all_positions(current)

            if prev_snapshot:
                # Check for sign changes
                for planet in Planet:
                    curr_pos = snapshot.positions[planet]
                    prev_pos = prev_snapshot.positions[planet]

                    if curr_pos.sign != prev_pos.sign:
                        # Sign change detected
                        impact = self._calculate_sign_change_impact(planet, curr_pos.sign)
                        events.append(TransitEvent(
                            date=current,
                            event_type='sign_change',
                            planet=planet,
                            description=f"{planet.value} enters {curr_pos.sign.name}",
                            market_impact=impact
                        ))

                    if curr_pos.nakshatra_index != prev_pos.nakshatra_index:
                        # Nakshatra change
                        nakshatra = NAKSHATRAS[curr_pos.nakshatra_index]
                        events.append(TransitEvent(
                            date=current,
                            event_type='nakshatra_change',
                            planet=planet,
                            description=f"{planet.value} enters {nakshatra['name']} nakshatra",
                            market_impact=nakshatra['market_energy'] - 0.5
                        ))

                    # Retrograde changes
                    if curr_pos.is_retrograde and not prev_pos.is_retrograde:
                        events.append(TransitEvent(
                            date=current,
                            event_type='retrograde_start',
                            planet=planet,
                            description=f"{planet.value} turns retrograde",
                            market_impact=-0.3
                        ))
                    elif not curr_pos.is_retrograde and prev_pos.is_retrograde:
                        events.append(TransitEvent(
                            date=current,
                            event_type='retrograde_end',
                            planet=planet,
                            description=f"{planet.value} turns direct",
                            market_impact=0.2
                        ))

            prev_snapshot = snapshot
            current += timedelta(days=1)

        return sorted(events, key=lambda x: x.date)

    def _calculate_sign_change_impact(self, planet: Planet, new_sign: ZodiacSign) -> float:
        """Calculate market impact of a planet entering a new sign."""
        dignity = self.get_planetary_dignity(planet, new_sign)
        planet_weight = PLANETARY_MARKET_WEIGHTS.get(planet, 0.1)

        # Impact is based on dignity change weighted by planet importance
        impact = (dignity.strength - 0.5) * planet_weight * 2

        return max(-1.0, min(1.0, impact))

    def get_complete_analysis(self, dt: datetime) -> VedicAnalysis:
        """
        Perform a complete Vedic analysis for a given moment.

        Args:
            dt: Datetime to analyze.

        Returns:
            VedicAnalysis with all computed factors.
        """
        snapshot = self.engine.get_all_positions(dt)

        # Calculate dignities
        dignities = {}
        for planet, pos in snapshot.positions.items():
            dignities[planet] = self.get_planetary_dignity(planet, pos.sign)

        # Get aspects
        aspects = self.get_all_aspects(snapshot)

        # Get nakshatra info
        nakshatra_positions = {}
        for planet, pos in snapshot.positions.items():
            nakshatra_positions[planet] = self.get_nakshatra_info(pos)

        # Detect yogas
        yogas = self.detect_yogas(snapshot)

        # Calculate overall strength
        total_strength = 0.0
        total_weight = 0.0
        for planet, dignity in dignities.items():
            weight = PLANETARY_MARKET_WEIGHTS.get(planet, 0.1)
            total_strength += dignity.strength * weight
            total_weight += weight

        # Add aspect influence
        aspect_sum = sum(a.strength for a in aspects)
        aspect_contribution = aspect_sum / max(len(aspects), 1) * 0.2

        overall_strength = (total_strength / total_weight) + aspect_contribution
        overall_strength = max(0.0, min(1.0, overall_strength))

        return VedicAnalysis(
            timestamp=dt,
            dignities=dignities,
            aspects=aspects,
            nakshatra_positions=nakshatra_positions,
            active_yogas=yogas,
            overall_strength=overall_strength
        )

    def get_analysis_range(self, start_dt: datetime, end_dt: datetime,
                          interval_days: int = 1) -> List[VedicAnalysis]:
        """
        Get Vedic analysis over a date range.

        Args:
            start_dt: Start datetime.
            end_dt: End datetime.
            interval_days: Days between analyses.

        Returns:
            List of VedicAnalysis objects.
        """
        analyses = []
        current = start_dt

        while current <= end_dt:
            analyses.append(self.get_complete_analysis(current))
            current += timedelta(days=interval_days)

        return analyses
