"""Numerical constants adopted in the manuscript."""

from dataclasses import dataclass


BEAMS_AU = (2, 3, 5, 10, 20)
MJUP_PER_MSUN = 1047.348644
MEARTH_PER_MJUP = 317.82838


@dataclass(frozen=True)
class SolarSystemParameters:
    stellar_mass_msun: float = 1.00
    stellar_mass_error_msun: float = 0.01
    age_myr: float = 4570.0
    age_error_myr: float = 50.0
    asteroid_belt_outer_au: float = 4.0
    neptune_semimajor_axis_au: float = 30.07
    neptune_mass_mearth: float = 17.15
    neptune_eccentricity: float = 0.009


SOLAR_SYSTEM = SolarSystemParameters()
