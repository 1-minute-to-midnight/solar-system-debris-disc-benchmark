"""Published exterior clearance-width prescriptions.

Each function returns ``(R_in - a_p) / a_p`` for a planet-to-star mass
ratio ``mu``.  The benchmark uses circular planets, so eccentricity defaults
to zero where a published eccentric extension is available.
"""

from collections import OrderedDict


def wisdom_1980(mu: float) -> float:
    """Wisdom (1980), resonance-overlap width."""
    return 1.30 * mu ** (2.0 / 7.0)


def duncan_1989(mu: float) -> float:
    """Duncan, Quinn & Tremaine (1989), numerical calibration."""
    return 1.49 * mu ** (2.0 / 7.0)


def gladman_1993(mu: float) -> float:
    """Gladman (1993), initially circular orbit-crossing criterion."""
    return 2.10 * mu ** (1.0 / 3.0)


def malhotra_1998(mu: float) -> float:
    """Malhotra (1998), resonance-overlap width."""
    return 1.40 * mu ** (2.0 / 7.0)


def pearce_wyatt_2014_exterior(mu: float, eccentricity: float = 0.0) -> float:
    """Pearce & Wyatt (2014), exterior unstable-zone edge."""
    return (
        eccentricity
        + 5.0
        * (1.0 + eccentricity)
        / (3.0 - eccentricity) ** (1.0 / 3.0)
        * mu ** (1.0 / 3.0)
    )


def lazzoni_2018_exterior(mu: float, eccentricity: float = 0.0) -> float:
    """Lazzoni et al. (2018), equation 9 exterior width."""
    return 1.30 * mu ** (2.0 / 7.0) * (1.0 + eccentricity)


def morrison_malhotra_2015_exterior(mu: float) -> float:
    """Morrison & Malhotra (2015), cleared exterior-zone width."""
    return 1.70 * mu**0.31


CLEARANCE_MODELS = OrderedDict(
    (
        ("Wisdom (1980)", wisdom_1980),
        ("Duncan et al. (1989)", duncan_1989),
        ("Gladman (1993)", gladman_1993),
        ("Malhotra (1998)", malhotra_1998),
        ("Pearce & Wyatt (2014)", pearce_wyatt_2014_exterior),
        ("Lazzoni et al. (2018)", lazzoni_2018_exterior),
        ("Morrison & Malhotra (2015)", morrison_malhotra_2015_exterior),
    )
)
