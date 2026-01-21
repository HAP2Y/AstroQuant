"""
Planetary Engine - High-precision planetary position calculations.
================================================================
Calculates planetary positions using astronomical algorithms.
Supports both tropical and sidereal (Vedic) zodiac systems.
"""

import ephem
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import numpy as np

from config.settings import Planet, ZodiacSign


# Ayanamsa constant (Lahiri - most commonly used for Vedic astrology)
# This is the precession offset to convert from tropical to sidereal
LAHIRI_AYANAMSA_2000 = 23.853  # degrees at J2000.0
AYANAMSA_ANNUAL_RATE = 50.3 / 3600  # degrees per year (precession rate)


@dataclass
class PlanetaryPosition:
    """Represents a planet's position in the sky."""
    planet: Planet
    longitude: float  # Sidereal longitude (0-360)
    tropical_longitude: float  # Tropical longitude
    latitude: float
    distance: float  # AU from Earth
    sign: ZodiacSign
    sign_degree: float  # Degree within sign (0-30)
    nakshatra_index: int
    nakshatra_pada: int  # 1-4
    is_retrograde: bool
    speed: float  # degrees per day

    @property
    def nakshatra_degree(self) -> float:
        """Degree within the current nakshatra (0-13.333)."""
        nakshatra_size = 360 / 27
        return self.longitude % nakshatra_size


@dataclass
class LunarPhase:
    """Represents the current lunar phase."""
    phase_angle: float  # 0-360
    phase_name: str
    illumination: float  # 0-1
    is_waxing: bool
    days_since_new: float
    tithi: int  # Vedic lunar day (1-30)
    tithi_name: str


@dataclass
class PlanetarySnapshot:
    """Complete planetary state at a given moment."""
    timestamp: datetime
    positions: Dict[Planet, PlanetaryPosition]
    lunar_phase: LunarPhase
    ayanamsa: float


class PlanetaryEngine:
    """
    High-precision planetary position calculator.

    Uses PyEphem for astronomical calculations and applies
    Lahiri Ayanamsa for Vedic (sidereal) calculations.
    """

    # Tithi names (30 lunar days)
    TITHI_NAMES = [
        "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami",
        "Shashthi", "Saptami", "Ashtami", "Navami", "Dashami",
        "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi", "Purnima",
        "Pratipada", "Dwitiya", "Tritiya", "Chaturthi", "Panchami",
        "Shashthi", "Saptami", "Ashtami", "Navami", "Dashami",
        "Ekadashi", "Dwadashi", "Trayodashi", "Chaturdashi", "Amavasya"
    ]

    # Phase names
    PHASE_NAMES = [
        "New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
        "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"
    ]

    def __init__(self, location: Optional[Tuple[float, float]] = None):
        """
        Initialize the planetary engine.

        Args:
            location: Optional (latitude, longitude) tuple for location-specific calculations.
        """
        self.observer = ephem.Observer()
        if location:
            self.observer.lat = str(location[0])
            self.observer.lon = str(location[1])
        else:
            # Default to Greenwich
            self.observer.lat = '51.4772'
            self.observer.lon = '0.0'

        # Planet mapping to PyEphem objects
        self._planet_objects = {
            Planet.SUN: ephem.Sun,
            Planet.MOON: ephem.Moon,
            Planet.MARS: ephem.Mars,
            Planet.MERCURY: ephem.Mercury,
            Planet.JUPITER: ephem.Jupiter,
            Planet.VENUS: ephem.Venus,
            Planet.SATURN: ephem.Saturn,
            Planet.URANUS: ephem.Uranus,
            Planet.NEPTUNE: ephem.Neptune,
            Planet.PLUTO: ephem.Pluto,
        }

    def get_ayanamsa(self, dt: datetime) -> float:
        """
        Calculate Lahiri Ayanamsa for a given date.

        The ayanamsa is the difference between tropical and sidereal zodiacs.

        Args:
            dt: Datetime to calculate ayanamsa for.

        Returns:
            Ayanamsa in degrees.
        """
        # Years since J2000.0
        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        years_since_j2000 = (dt - j2000).days / 365.25

        ayanamsa = LAHIRI_AYANAMSA_2000 + (years_since_j2000 * AYANAMSA_ANNUAL_RATE)
        return ayanamsa

    def _tropical_to_sidereal(self, tropical_lon: float, ayanamsa: float) -> float:
        """Convert tropical longitude to sidereal."""
        sidereal = tropical_lon - ayanamsa
        if sidereal < 0:
            sidereal += 360
        return sidereal % 360

    def _get_sign_and_degree(self, longitude: float) -> Tuple[ZodiacSign, float]:
        """Get zodiac sign and degree within sign from longitude."""
        sign_index = int(longitude / 30)
        sign_degree = longitude % 30
        return ZodiacSign(sign_index), sign_degree

    def _get_nakshatra_info(self, longitude: float) -> Tuple[int, int]:
        """
        Get nakshatra index and pada from sidereal longitude.

        Args:
            longitude: Sidereal longitude in degrees.

        Returns:
            Tuple of (nakshatra_index, pada).
        """
        nakshatra_size = 360 / 27  # 13.333... degrees
        nakshatra_index = int(longitude / nakshatra_size)

        # Calculate pada (quarter) within nakshatra
        degree_in_nakshatra = longitude % nakshatra_size
        pada = int(degree_in_nakshatra / (nakshatra_size / 4)) + 1

        return nakshatra_index, pada

    def _calculate_rahu_ketu(self, dt: datetime, ayanamsa: float) -> Tuple[PlanetaryPosition, PlanetaryPosition]:
        """
        Calculate positions of Rahu (North Node) and Ketu (South Node).

        The lunar nodes are calculated based on the Moon's orbital parameters.
        """
        self.observer.date = dt

        # Get Moon's position to calculate nodes
        moon = ephem.Moon(self.observer)

        # The ascending node (Rahu) longitude
        # Using mean node calculation (simplified but adequate for market analysis)
        jd = ephem.julian_date(dt)

        # Mean longitude of ascending node (simplified formula)
        T = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000

        # Mean longitude of ascending node
        rahu_tropical = 125.0445479 - 1934.1362891 * T + 0.0020754 * T**2
        rahu_tropical = rahu_tropical % 360

        # Apply ayanamsa for sidereal position
        rahu_lon = self._tropical_to_sidereal(rahu_tropical, ayanamsa)

        # Ketu is exactly opposite to Rahu
        ketu_lon = (rahu_lon + 180) % 360

        # Get sign and nakshatra info
        rahu_sign, rahu_sign_deg = self._get_sign_and_degree(rahu_lon)
        ketu_sign, ketu_sign_deg = self._get_sign_and_degree(ketu_lon)

        rahu_nakshatra, rahu_pada = self._get_nakshatra_info(rahu_lon)
        ketu_nakshatra, ketu_pada = self._get_nakshatra_info(ketu_lon)

        # Nodes are always retrograde in mean motion
        rahu_speed = -0.053  # degrees per day (retrograde)

        rahu_pos = PlanetaryPosition(
            planet=Planet.RAHU,
            longitude=rahu_lon,
            tropical_longitude=rahu_tropical,
            latitude=0.0,
            distance=0.0,  # Nodes have no physical distance
            sign=rahu_sign,
            sign_degree=rahu_sign_deg,
            nakshatra_index=rahu_nakshatra,
            nakshatra_pada=rahu_pada,
            is_retrograde=True,
            speed=rahu_speed
        )

        ketu_pos = PlanetaryPosition(
            planet=Planet.KETU,
            longitude=ketu_lon,
            tropical_longitude=(rahu_tropical + 180) % 360,
            latitude=0.0,
            distance=0.0,
            sign=ketu_sign,
            sign_degree=ketu_sign_deg,
            nakshatra_index=ketu_nakshatra,
            nakshatra_pada=ketu_pada,
            is_retrograde=True,
            speed=rahu_speed
        )

        return rahu_pos, ketu_pos

    def get_planet_position(self, planet: Planet, dt: datetime,
                           ayanamsa: Optional[float] = None) -> PlanetaryPosition:
        """
        Calculate the position of a planet at a given time.

        Args:
            planet: Planet to calculate position for.
            dt: Datetime for the calculation.
            ayanamsa: Optional pre-calculated ayanamsa.

        Returns:
            PlanetaryPosition with all calculated values.
        """
        if ayanamsa is None:
            ayanamsa = self.get_ayanamsa(dt)

        # Handle Rahu and Ketu separately
        if planet == Planet.RAHU:
            rahu_pos, _ = self._calculate_rahu_ketu(dt, ayanamsa)
            return rahu_pos
        elif planet == Planet.KETU:
            _, ketu_pos = self._calculate_rahu_ketu(dt, ayanamsa)
            return ketu_pos

        # Set observer date
        self.observer.date = dt

        # Create and compute planet
        planet_class = self._planet_objects.get(planet)
        if planet_class is None:
            raise ValueError(f"Unknown planet: {planet}")

        body = planet_class(self.observer)

        # Get tropical longitude (in radians from PyEphem)
        tropical_lon = math.degrees(float(body.hlong))
        if tropical_lon < 0:
            tropical_lon += 360

        # Convert to sidereal
        sidereal_lon = self._tropical_to_sidereal(tropical_lon, ayanamsa)

        # Get latitude
        latitude = math.degrees(float(body.hlat))

        # Get distance (in AU)
        distance = float(body.earth_distance)

        # Get sign and nakshatra info
        sign, sign_degree = self._get_sign_and_degree(sidereal_lon)
        nakshatra_index, nakshatra_pada = self._get_nakshatra_info(sidereal_lon)

        # Calculate daily motion for retrograde detection
        # Compare position with next day
        dt_next = dt + timedelta(days=1)
        self.observer.date = dt_next
        body_next = planet_class(self.observer)
        next_lon = math.degrees(float(body_next.hlong))
        if next_lon < 0:
            next_lon += 360

        # Calculate speed (handle 360/0 boundary)
        speed = next_lon - tropical_lon
        if speed > 180:
            speed -= 360
        elif speed < -180:
            speed += 360

        is_retrograde = speed < 0

        return PlanetaryPosition(
            planet=planet,
            longitude=sidereal_lon,
            tropical_longitude=tropical_lon,
            latitude=latitude,
            distance=distance,
            sign=sign,
            sign_degree=sign_degree,
            nakshatra_index=nakshatra_index,
            nakshatra_pada=nakshatra_pada,
            is_retrograde=is_retrograde,
            speed=speed
        )

    def get_lunar_phase(self, dt: datetime) -> LunarPhase:
        """
        Calculate the lunar phase for a given datetime.

        Args:
            dt: Datetime for the calculation.

        Returns:
            LunarPhase with all calculated values.
        """
        self.observer.date = dt

        # Get Sun and Moon positions
        sun = ephem.Sun(self.observer)
        moon = ephem.Moon(self.observer)

        # Phase angle (elongation from Sun)
        sun_lon = math.degrees(float(sun.hlong))
        moon_lon = math.degrees(float(moon.hlong))

        phase_angle = (moon_lon - sun_lon) % 360

        # Calculate illumination
        illumination = (1 - math.cos(math.radians(phase_angle))) / 2

        # Determine phase name
        phase_index = int(phase_angle / 45)
        phase_name = self.PHASE_NAMES[phase_index]

        # Is waxing (phase angle 0-180)
        is_waxing = phase_angle < 180

        # Days since new moon (approximate)
        days_since_new = phase_angle / (360 / 29.53)

        # Calculate Tithi (Vedic lunar day)
        tithi = int(phase_angle / 12) + 1
        if tithi > 30:
            tithi = 30
        tithi_name = self.TITHI_NAMES[tithi - 1]

        return LunarPhase(
            phase_angle=phase_angle,
            phase_name=phase_name,
            illumination=illumination,
            is_waxing=is_waxing,
            days_since_new=days_since_new,
            tithi=tithi,
            tithi_name=tithi_name
        )

    def get_all_positions(self, dt: datetime) -> PlanetarySnapshot:
        """
        Get positions of all planets at a given time.

        Args:
            dt: Datetime for the calculation.

        Returns:
            PlanetarySnapshot containing all planetary positions.
        """
        ayanamsa = self.get_ayanamsa(dt)

        positions = {}
        for planet in Planet:
            positions[planet] = self.get_planet_position(planet, dt, ayanamsa)

        lunar_phase = self.get_lunar_phase(dt)

        return PlanetarySnapshot(
            timestamp=dt,
            positions=positions,
            lunar_phase=lunar_phase,
            ayanamsa=ayanamsa
        )

    def get_positions_range(self, start_dt: datetime, end_dt: datetime,
                           interval_hours: int = 24) -> List[PlanetarySnapshot]:
        """
        Get planetary positions over a date range.

        Args:
            start_dt: Start datetime.
            end_dt: End datetime.
            interval_hours: Hours between each snapshot.

        Returns:
            List of PlanetarySnapshots.
        """
        snapshots = []
        current = start_dt

        while current <= end_dt:
            snapshots.append(self.get_all_positions(current))
            current += timedelta(hours=interval_hours)

        return snapshots

    def find_next_aspect(self, planet1: Planet, planet2: Planet,
                        aspect_degrees: float, start_dt: datetime,
                        orb: float = 1.0) -> Optional[datetime]:
        """
        Find the next exact aspect between two planets.

        Args:
            planet1: First planet.
            planet2: Second planet.
            aspect_degrees: Aspect angle to find (0=conjunction, 180=opposition, etc.)
            start_dt: Start searching from this date.
            orb: Allowed orb in degrees.

        Returns:
            Datetime of exact aspect, or None if not found within 365 days.
        """
        current = start_dt
        max_search = start_dt + timedelta(days=365)

        ayanamsa = self.get_ayanamsa(current)

        while current < max_search:
            pos1 = self.get_planet_position(planet1, current, ayanamsa)
            pos2 = self.get_planet_position(planet2, current, ayanamsa)

            # Calculate angular separation
            diff = abs(pos1.longitude - pos2.longitude)
            if diff > 180:
                diff = 360 - diff

            # Check if within orb of aspect
            aspect_diff = abs(diff - aspect_degrees)
            if aspect_diff <= orb:
                return current

            # Adaptive step size based on how close we are
            if aspect_diff < 5:
                step = timedelta(hours=6)
            elif aspect_diff < 15:
                step = timedelta(days=1)
            else:
                step = timedelta(days=3)

            current += step

        return None

    def get_retrograde_periods(self, planet: Planet, start_dt: datetime,
                               end_dt: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Find retrograde periods for a planet within a date range.

        Args:
            planet: Planet to check.
            start_dt: Start of range.
            end_dt: End of range.

        Returns:
            List of (start, end) tuples for retrograde periods.
        """
        if planet in [Planet.SUN, Planet.MOON, Planet.RAHU, Planet.KETU]:
            return []  # These don't have meaningful retrograde periods

        periods = []
        current = start_dt
        in_retrograde = False
        retro_start = None

        ayanamsa = self.get_ayanamsa(current)

        while current <= end_dt:
            pos = self.get_planet_position(planet, current, ayanamsa)

            if pos.is_retrograde and not in_retrograde:
                # Retrograde started
                in_retrograde = True
                retro_start = current
            elif not pos.is_retrograde and in_retrograde:
                # Retrograde ended
                in_retrograde = False
                if retro_start:
                    periods.append((retro_start, current))
                retro_start = None

            current += timedelta(days=1)

        # Handle case where retrograde continues past end_dt
        if in_retrograde and retro_start:
            periods.append((retro_start, end_dt))

        return periods
