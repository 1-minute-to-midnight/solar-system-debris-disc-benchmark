#!/usr/bin/env python3
"""Calculate and save all planet predictions from the fitted belt edges."""

from solar_system_debris_disc.results import generate_predictions, print_summary


if __name__ == "__main__":
    print_summary(generate_predictions())
