"""Planet-inference calculations used by the benchmark."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .clearance import CLEARANCE_MODELS, pearce_wyatt_2014_exterior
from .constants import MEARTH_PER_MJUP, MJUP_PER_MSUN, SOLAR_SYSTEM


def scattering_minimum_mass_mjup(
    semimajor_axis_au: float,
    disc_inner_edge_au: float,
    age_myr: float = SOLAR_SYSTEM.age_myr,
    stellar_mass_msun: float = SOLAR_SYSTEM.stellar_mass_msun,
) -> float:
    """Pearce et al. (2022) equation 7, originating in Pearce & Wyatt (2014)."""
    return (
        0.331
        * semimajor_axis_au
        * disc_inner_edge_au ** (-0.25)
        * age_myr ** (-0.5)
        * stellar_mass_msun**0.75
    )


def model_intersection(
    clearance_model,
    disc_inner_edge_au: float,
    stellar_mass_msun: float = SOLAR_SYSTEM.stellar_mass_msun,
    age_myr: float = SOLAR_SYSTEM.age_myr,
) -> tuple[float, float]:
    """Return ``(a_p [au], M_p [Earth masses])`` for one clearance law."""

    def residual(mass_mjup: float) -> float:
        mu = mass_mjup / (MJUP_PER_MSUN * stellar_mass_msun)
        semimajor_axis = disc_inner_edge_au / (1.0 + clearance_model(mu))
        return mass_mjup - scattering_minimum_mass_mjup(
            semimajor_axis, disc_inner_edge_au, age_myr, stellar_mass_msun
        )

    mass_mjup = brentq(residual, 1.0e-8, 10.0)
    mu = mass_mjup / (MJUP_PER_MSUN * stellar_mass_msun)
    semimajor_axis = disc_inner_edge_au / (1.0 + clearance_model(mu))
    return semimajor_axis, mass_mjup * MEARTH_PER_MJUP


def all_model_intersections(
    disc_inner_edge_au: float,
    stellar_mass_msun: float = SOLAR_SYSTEM.stellar_mass_msun,
    age_myr: float = SOLAR_SYSTEM.age_myr,
) -> dict[str, tuple[float, float]]:
    return {
        name: model_intersection(model, disc_inner_edge_au, stellar_mass_msun, age_myr)
        for name, model in CLEARANCE_MODELS.items()
    }


def stirring_eccentricity(
    planet_semimajor_axis_au: float,
    planet_mass_mearth: float,
    disc_outer_edge_au: float,
    age_myr: float = SOLAR_SYSTEM.age_myr,
    stellar_mass_msun: float = SOLAR_SYSTEM.stellar_mass_msun,
) -> float:
    """Invert Pearce et al. (2022) equation 22 for planet eccentricity.

    This is the general secular-stirring timescale evaluated at the separately
    inferred planet semimajor axis.  It is not the boundary-placement special
    case in that paper's equation 23.
    """
    planet_mass_mjup = planet_mass_mearth / MEARTH_PER_MJUP
    coefficient = (
        5.07e-5
        * disc_outer_edge_au**4.5
        * planet_semimajor_axis_au**-3.0
        * age_myr**-1.0
        * stellar_mass_msun**0.5
    )
    return brentq(
        lambda eccentricity: (
            coefficient
            * (1.0 - eccentricity**2) ** 1.5
            / eccentricity
            - planet_mass_mjup
        ),
        1.0e-10,
        0.999,
    )


def prediction(
    disc_inner_edge_au: float,
    disc_outer_edge_au: float,
    stellar_mass_msun: float = SOLAR_SYSTEM.stellar_mass_msun,
    age_myr: float = SOLAR_SYSTEM.age_myr,
    model: str = "median",
) -> np.ndarray:
    intersections = all_model_intersections(
        disc_inner_edge_au, stellar_mass_msun, age_myr
    )
    if model == "median":
        semimajor_axis, mass_mearth = np.median(
            np.asarray(list(intersections.values())), axis=0
        )
    elif model == "pearce":
        semimajor_axis, mass_mearth = intersections["Pearce & Wyatt (2014)"]
    else:
        raise ValueError("model must be 'median' or 'pearce'")
    eccentricity = stirring_eccentricity(
        semimajor_axis,
        mass_mearth,
        disc_outer_edge_au,
        age_myr,
        stellar_mass_msun,
    )
    return np.asarray((semimajor_axis, mass_mearth, eccentricity))


def propagated_prediction(edge: dict, model: str = "median") -> tuple[np.ndarray, ...]:
    """Propagate one-sided edge, stellar-mass and age errors in quadrature."""
    nominal_inputs = {
        "disc_inner_edge_au": edge["inner"],
        "disc_outer_edge_au": edge["outer"],
        "stellar_mass_msun": SOLAR_SYSTEM.stellar_mass_msun,
        "age_myr": SOLAR_SYSTEM.age_myr,
    }
    errors = {
        "disc_inner_edge_au": (edge["inner_up"], edge["inner_down"]),
        "disc_outer_edge_au": (edge["outer_up"], edge["outer_down"]),
        "stellar_mass_msun": (
            SOLAR_SYSTEM.stellar_mass_error_msun,
            SOLAR_SYSTEM.stellar_mass_error_msun,
        ),
        "age_myr": (SOLAR_SYSTEM.age_error_myr, SOLAR_SYSTEM.age_error_myr),
    }
    nominal = prediction(**nominal_inputs, model=model)
    deltas = []
    for parameter, (upper_error, lower_error) in errors.items():
        upper_inputs = nominal_inputs.copy()
        lower_inputs = nominal_inputs.copy()
        upper_inputs[parameter] += upper_error
        lower_inputs[parameter] -= lower_error
        deltas.extend(
            (
                prediction(**upper_inputs, model=model) - nominal,
                prediction(**lower_inputs, model=model) - nominal,
            )
        )
    deltas = np.asarray(deltas)
    upper = np.sqrt(np.sum(np.maximum(deltas, 0.0) ** 2, axis=0))
    lower = np.sqrt(np.sum(np.maximum(-deltas, 0.0) ** 2, axis=0))
    return nominal, upper, lower


def shannon_clearing_floor_mearth(
    outer_gap_edge_au: float,
    age_myr: float = SOLAR_SYSTEM.age_myr,
    stellar_mass_msun: float = SOLAR_SYSTEM.stellar_mass_msun,
) -> float:
    """Shannon et al. (2016) equation 4."""
    return 4.0 / age_myr * outer_gap_edge_au**1.5 * stellar_mass_msun**0.5


def shannon_mass_for_count_mearth(
    number: int,
    inner_gap_edge_au: float,
    outer_gap_edge_au: float,
    stellar_mass_msun: float = SOLAR_SYSTEM.stellar_mass_msun,
) -> float:
    """Invert Shannon et al. (2016) equation 5 for a specified planet count.

    This is the equal mass implied by the paper's approximate K=20 spacing
    relation.  It is not the age-based clearing floor from equation 4.
    """
    if number < 2:
        raise ValueError("number must be at least 2")
    ratio = (outer_gap_edge_au / inner_gap_edge_au) ** (1.0 / (number - 1.0))
    delta = (ratio - 1.0) / (ratio + 1.0)
    return (delta / (0.13 * stellar_mass_msun ** (-1.0 / 3.0))) ** 3


def shannon_exact_spacing_mass_mearth(
    number: int,
    inner_gap_edge_au: float,
    outer_gap_edge_au: float,
    stellar_mass_msun: float = SOLAR_SYSTEM.stellar_mass_msun,
    mutual_hill_spacing: float = 20.0,
) -> float:
    """Shannon et al. (2016) equation 1 without the equation-5 rounding."""
    ratio = (outer_gap_edge_au / inner_gap_edge_au) ** (1.0 / (number - 1.0))
    x = (ratio - 1.0) / (ratio + 1.0)
    mu = 12.0 / mutual_hill_spacing**3 * x**3
    msun_per_mearth = MJUP_PER_MSUN * MEARTH_PER_MJUP
    return mu * stellar_mass_msun * msun_per_mearth


def pearce_prediction(
    disc_inner_edge_au: float,
    disc_outer_edge_au: float,
) -> np.ndarray:
    """Convenience wrapper used in figure and regression code."""
    semimajor_axis, mass_mearth = model_intersection(
        pearce_wyatt_2014_exterior, disc_inner_edge_au
    )
    return np.asarray(
        (
            semimajor_axis,
            mass_mearth,
            stirring_eccentricity(
                semimajor_axis, mass_mearth, disc_outer_edge_au
            ),
        )
    )
