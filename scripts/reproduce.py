#!/usr/bin/env python3
"""Run the complete deterministic analysis and regenerate every output."""

from solar_system_debris_disc.plotting import make_all_figures
from solar_system_debris_disc.profiles import main as fit_profiles
from solar_system_debris_disc.results import generate_predictions, print_summary


if __name__ == "__main__":
    fit_profiles()
    predictions = generate_predictions()
    print_summary(predictions)
    make_all_figures()
    print("reproduction complete")
