import numpy as np

from solar_system_debris_disc import clearance
from solar_system_debris_disc.inference import (
    scattering_minimum_mass_mjup,
    shannon_clearing_floor_mearth,
    shannon_exact_spacing_mass_mearth,
    shannon_mass_for_count_mearth,
    stirring_eccentricity,
)


def test_clearance_prescriptions_match_published_forms():
    mu = 1.0e-4
    assert np.isclose(clearance.wisdom_1980(mu), 1.30 * mu ** (2 / 7))
    assert np.isclose(clearance.duncan_1989(mu), 1.49 * mu ** (2 / 7))
    assert np.isclose(clearance.gladman_1993(mu), 2.10 * mu ** (1 / 3))
    assert np.isclose(clearance.malhotra_1998(mu), 1.40 * mu ** (2 / 7))
    assert np.isclose(clearance.lazzoni_2018_exterior(mu), 1.30 * mu ** (2 / 7))
    assert np.isclose(clearance.morrison_malhotra_2015_exterior(mu), 1.70 * mu**0.31)
    assert np.isclose(
        clearance.pearce_wyatt_2014_exterior(mu),
        5.0 / 3.0 ** (1 / 3) * mu ** (1 / 3),
    )


def test_pearce_constraints_reproduce_two_au_values():
    mass = scattering_minimum_mass_mjup(32.59455240, 37.06110674558824)
    assert np.isclose(mass * 317.82838, 20.55786423, rtol=2e-8)
    eccentricity = stirring_eccentricity(32.59455240, 20.55786423, 47.11958697874581)
    assert np.isclose(eccentricity, 1.67601536e-4, rtol=2e-8)


def test_shannon_floor_and_spacing_are_distinct_quantities():
    outer_edge = 37.06110674558824
    floor = shannon_clearing_floor_mearth(outer_edge)
    approximate_spacing = shannon_mass_for_count_mearth(4, 4.0, outer_edge)
    exact_spacing = shannon_exact_spacing_mass_mearth(4, 4.0, outer_edge)
    assert np.isclose(floor, 0.197479204, rtol=1e-8)
    assert np.isclose(approximate_spacing, 20.34744335, rtol=1e-8)
    assert np.isclose(exact_spacing, 22.32107532, rtol=1e-8)
    assert floor < approximate_spacing < exact_spacing
