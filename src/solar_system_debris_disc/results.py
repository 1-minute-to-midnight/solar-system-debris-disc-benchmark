"""Assemble and serialise all numerical benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import BEAMS_AU, SOLAR_SYSTEM
from .inference import (
    all_model_intersections,
    propagated_prediction,
    shannon_clearing_floor_mearth,
    shannon_exact_spacing_mass_mearth,
    shannon_mass_for_count_mearth,
)


ROOT = Path(__file__).resolve().parents[2]
EDGE_PATH = ROOT / "results" / "surface_density_edges.json"
PREDICTION_PATH = ROOT / "results" / "planet_predictions.json"


def _asymmetric_variation(function, nominal_inputs: dict, errors: dict):
    nominal = np.asarray(function(**nominal_inputs), dtype=float)
    deltas = []
    for parameter, (upper_error, lower_error) in errors.items():
        upper = nominal_inputs.copy()
        lower = nominal_inputs.copy()
        upper[parameter] += upper_error
        lower[parameter] -= lower_error
        deltas.extend((np.asarray(function(**upper)) - nominal, np.asarray(function(**lower)) - nominal))
    deltas = np.asarray(deltas)
    up = np.sqrt(np.sum(np.maximum(deltas, 0.0) ** 2, axis=0))
    down = np.sqrt(np.sum(np.maximum(-deltas, 0.0) ** 2, axis=0))
    return nominal, up, down


def four_planet_chain(edge: dict) -> dict:
    def chain(inner_edge_au: float, stellar_mass_msun: float):
        mass = shannon_mass_for_count_mearth(
            4,
            SOLAR_SYSTEM.asteroid_belt_outer_au,
            inner_edge_au,
            stellar_mass_msun,
        )
        return np.r_[
            np.geomspace(SOLAR_SYSTEM.asteroid_belt_outer_au, inner_edge_au, 4),
            mass,
        ]

    nominal, up, down = _asymmetric_variation(
        chain,
        {
            "inner_edge_au": edge["inner"],
            "stellar_mass_msun": SOLAR_SYSTEM.stellar_mass_msun,
        },
        {
            "inner_edge_au": (edge["inner_up"], edge["inner_down"]),
            "stellar_mass_msun": (
                SOLAR_SYSTEM.stellar_mass_error_msun,
                SOLAR_SYSTEM.stellar_mass_error_msun,
            ),
        },
    )
    return {
        "semimajor_axes_au": nominal[:4].tolist(),
        "semimajor_axes_up": up[:4].tolist(),
        "semimajor_axes_down": down[:4].tolist(),
        "spacing_implied_mass_mearth": float(nominal[4]),
        "spacing_implied_mass_up": float(up[4]),
        "spacing_implied_mass_down": float(down[4]),
        "exact_equation_1_mass_mearth": shannon_exact_spacing_mass_mearth(
            4,
            SOLAR_SYSTEM.asteroid_belt_outer_au,
            edge["inner"],
        ),
        "age_based_clearing_floor_mearth": shannon_clearing_floor_mearth(edge["inner"]),
    }


def generate_predictions(edge_path: Path = EDGE_PATH, output: Path = PREDICTION_PATH) -> dict:
    edges = json.loads(edge_path.read_text())
    result = {
        "metadata": {
            "stellar_mass_msun": SOLAR_SYSTEM.stellar_mass_msun,
            "stellar_mass_error_msun": SOLAR_SYSTEM.stellar_mass_error_msun,
            "age_myr": SOLAR_SYSTEM.age_myr,
            "age_error_myr": SOLAR_SYSTEM.age_error_myr,
            "asteroid_belt_outer_au": SOLAR_SYSTEM.asteroid_belt_outer_au,
        },
        "beams": {},
    }
    for beam in BEAMS_AU:
        edge = edges[str(beam)]
        median, median_up, median_down = propagated_prediction(edge, model="median")
        pearce, pearce_up, pearce_down = propagated_prediction(edge, model="pearce")
        intersections = all_model_intersections(edge["inner"])
        result["beams"][str(beam)] = {
            "median": {
                "semimajor_axis_au": float(median[0]),
                "semimajor_axis_up": float(median_up[0]),
                "semimajor_axis_down": float(median_down[0]),
                "mass_mearth": float(median[1]),
                "mass_up": float(median_up[1]),
                "mass_down": float(median_down[1]),
                "stirring_eccentricity": float(median[2]),
                "stirring_eccentricity_up": float(median_up[2]),
                "stirring_eccentricity_down": float(median_down[2]),
            },
            "pearce_wyatt": {
                "semimajor_axis_au": float(pearce[0]),
                "semimajor_axis_up": float(pearce_up[0]),
                "semimajor_axis_down": float(pearce_down[0]),
                "mass_mearth": float(pearce[1]),
                "mass_up": float(pearce_up[1]),
                "mass_down": float(pearce_down[1]),
                "stirring_eccentricity": float(pearce[2]),
                "stirring_eccentricity_up": float(pearce_up[2]),
                "stirring_eccentricity_down": float(pearce_down[2]),
            },
            "all_clearance_models": {
                name: {"semimajor_axis_au": values[0], "mass_mearth": values[1]}
                for name, values in intersections.items()
            },
            "four_planet_chain": four_planet_chain(edge),
        }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def print_summary(result: dict) -> None:
    print("beam  model   a_p [au]          M_p [Mearth]       e_stir")
    for beam in BEAMS_AU:
        values = result["beams"][str(beam)]
        for label, key in (("median", "median"), ("Pearce", "pearce_wyatt")):
            row = values[key]
            print(
                f"{beam:>4}  {label:<7} "
                f"{row['semimajor_axis_au']:.8f} "
                f"+{row['semimajor_axis_up']:.8f} -{row['semimajor_axis_down']:.8f}  "
                f"{row['mass_mearth']:.8f} +{row['mass_up']:.8f} -{row['mass_down']:.8f}  "
                f"{row['stirring_eccentricity']:.10e}"
            )
