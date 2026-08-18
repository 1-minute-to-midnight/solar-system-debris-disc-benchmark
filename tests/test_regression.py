import json
from pathlib import Path

import numpy as np

from solar_system_debris_disc.profiles import recover_and_fit
from solar_system_debris_disc.results import EDGE_PATH, PREDICTION_PATH, generate_predictions


def test_checked_in_two_au_edges_match_a_clean_refit():
    expected = json.loads(EDGE_PATH.read_text())["2"]
    fitted = recover_and_fit("alma_sim_2au.fits")
    for name in ("inner", "inner_up", "inner_down", "outer", "outer_up", "outer_down"):
        assert np.isclose(fitted["edges"][name], expected[name], atol=1e-7)
    assert fitted["best_name"] == expected["best_model"] == "Double Gaussian"


def test_predictions_regenerate_without_numerical_drift(tmp_path: Path):
    regenerated = generate_predictions(output=tmp_path / "predictions.json")
    expected = json.loads(PREDICTION_PATH.read_text())
    for beam in ("2", "3", "5", "10", "20"):
        for model in ("median", "pearce_wyatt"):
            for quantity in ("semimajor_axis_au", "mass_mearth", "stirring_eccentricity"):
                assert np.isclose(
                    regenerated["beams"][beam][model][quantity],
                    expected["beams"][beam][model][quantity],
                    rtol=1e-12,
                )
