"""Publication-figure generation from the repository inputs and results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from scipy.optimize import brentq

from .clearance import CLEARANCE_MODELS
from .constants import MEARTH_PER_MJUP, MJUP_PER_MSUN, SOLAR_SYSTEM
from .inference import (
    all_model_intersections,
    scattering_minimum_mass_mjup,
    shannon_clearing_floor_mearth,
    shannon_mass_for_count_mearth,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "kuiper_only_images"
FIGURES = ROOT / "figures"
EDGES = ROOT / "results" / "surface_density_edges.json"
PREDICTIONS = ROOT / "results" / "planet_predictions.json"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "figure.dpi": 120,
        }
    )


def make_synthetic_images() -> None:
    _style()
    beams = (2, 5, 10, 20)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.45), sharex=True, sharey=True)
    last = None
    for label, axis, beam in zip("abcd", axes, beams):
        with fits.open(DATA / f"alma_sim_{beam}au.fits") as hdul:
            image = np.maximum(hdul[0].data.astype(float), 0.0)
            pixel = float(hdul[0].header["PIXSCALE"])
        image /= image.max()
        half_x = image.shape[1] * pixel / 2.0
        half_y = image.shape[0] * pixel / 2.0
        last = axis.imshow(
            image,
            origin="lower",
            extent=(-half_x, half_x, -half_y, half_y),
            cmap="inferno",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.add_patch(plt.Circle((-55, -54), beam / 2.0, color="white"))
        axis.plot((35, 55), (-57, -57), color="white", lw=2.5)
        axis.text(45, -53, "20 au", color="white", ha="center", va="bottom", fontsize=9)
        axis.text(-59, 55, f"({label})", color="white", weight="bold", fontsize=12)
        axis.set_title(rf"FWHM $={beam}$ au")
        axis.set(xlim=(-65, 65), ylim=(-65, 65), aspect="equal")
        axis.set_xticks((-50, 0, 50))
        axis.set_yticks((-50, 0, 50))
    axes[0].set_ylabel("Projected distance (au)")
    for axis in axes:
        axis.set_xlabel("Projected distance (au)")
    fig.subplots_adjust(left=0.055, right=0.91, bottom=0.15, top=0.88, wspace=0.035)
    colourbar_axis = fig.add_axes((0.925, 0.18, 0.012, 0.66))
    colourbar = fig.colorbar(last, cax=colourbar_axis)
    colourbar.set_label("Relative surface brightness")
    colourbar.set_ticks((0, 0.5, 1))
    fig.savefig(FIGURES / "synthetic_images.png", dpi=300)
    plt.close(fig)


def _invert_clearance(targets: np.ndarray, function) -> np.ndarray:
    values = np.full_like(targets, np.nan)
    for index, target in enumerate(targets):
        if target > 0:
            try:
                values[index] = brentq(lambda mu: function(mu) - target, 1e-12, 1.0)
            except ValueError:
                pass
    return values


def make_single_planet() -> None:
    _style()
    edges = json.loads(EDGES.read_text())["2"]
    predictions = json.loads(PREDICTIONS.read_text())["beams"]["2"]
    radius_in, radius_out = edges["inner"], edges["outer"]
    semimajor_axes = np.linspace(1.0, radius_in, 2000)
    relative_width = radius_in / semimajor_axes - 1.0
    scattering = np.asarray(
        [scattering_minimum_mass_mjup(axis, radius_in) for axis in semimajor_axes]
    )
    colours = ("red", "orange", "green", "blue", "purple", "cyan", "brown")
    curves = {}
    for (name, function), colour in zip(CLEARANCE_MODELS.items(), colours):
        mass_ratio = _invert_clearance(relative_width, function)
        curves[name] = (mass_ratio * MJUP_PER_MSUN, colour)
    intersections = all_model_intersections(radius_in)

    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.axvspan(0, SOLAR_SYSTEM.asteroid_belt_outer_au, color="brown", alpha=0.25)
    axis.axvspan(radius_in, radius_out, color="gray", alpha=0.35)
    axis.text(2.0, 0.1, "Asteroid Belt", ha="center", va="center", rotation=90,
              color="darkred", weight="bold")
    axis.text((radius_in + radius_out) / 2.0, 0.1, "Kuiper Belt", ha="center",
              va="center", rotation=90, color="0.35", weight="bold")
    for (name, (curve, colour)) in curves.items():
        axis.semilogy(semimajor_axes, curve, color=colour, lw=1.4, label=name)
        location, mass_earth = intersections[name]
        axis.plot(location, mass_earth / MEARTH_PER_MJUP, "o", color=colour,
                  mec="black", mew=0.8, ms=7, zorder=4)
    axis.semilogy(semimajor_axes, scattering, "k--", lw=1.5, label="Scattering constraint")
    pearce = predictions["pearce_wyatt"]
    axis.errorbar(
        pearce["semimajor_axis_au"],
        pearce["mass_mearth"] / MEARTH_PER_MJUP,
        xerr=[[pearce["semimajor_axis_down"]], [pearce["semimajor_axis_up"]]],
        yerr=[
            [pearce["mass_down"] / MEARTH_PER_MJUP],
            [pearce["mass_up"] / MEARTH_PER_MJUP],
        ],
        fmt="none", color="purple", capsize=3, zorder=5,
    )
    axis.plot(
        SOLAR_SYSTEM.neptune_semimajor_axis_au,
        SOLAR_SYSTEM.neptune_mass_mearth / MEARTH_PER_MJUP,
        "*", color="navy", mec="black", mew=0.8, ms=15, label="Neptune", zorder=6,
    )
    axis.set(xlim=(0, 45), ylim=(1e-4, 1e3),
             xlabel=r"Semimajor axis $a_p$ [au]", ylabel=r"Planet mass $M_p$ [$M_{\rm Jup}$]")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=7.2, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "single_planet_prediction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_multi_planet() -> None:
    _style()
    edges = json.loads(EDGES.read_text())["2"]
    inner, outer = SOLAR_SYSTEM.asteroid_belt_outer_au, edges["inner"]
    numbers = np.arange(2, 26)
    masses_earth = np.asarray(
        [shannon_mass_for_count_mearth(number, inner, outer) for number in numbers]
    )
    masses_jup = masses_earth / MEARTH_PER_MJUP
    floor_earth = shannon_clearing_floor_mearth(outer)
    floor_jup = floor_earth / MEARTH_PER_MJUP
    first_too_light = int(numbers[np.flatnonzero(masses_earth < floor_earth)[0]])
    colours = {2: "#d62728", 4: "#ff7f0e", 7: "#1b9e77", 10: "#377eb8", 15: "#984ea3"}

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 7.2), gridspec_kw={"hspace": 0.30})
    top.axvspan(0, inner, color="#d98c3a", alpha=0.16)
    top.axvspan(outer, edges["outer"], color="#8da0cb", alpha=0.18)
    top.axhspan(1e-6, floor_jup, color="#d95f5f", alpha=0.10)
    top.axhline(floor_jup, color="#b22222", lw=1)
    top.text(5.2, floor_jup * 0.42, f"Age-based clearing floor ({floor_earth:.2f} $M_\\oplus$)",
             color="#9c2222", fontsize=8, style="italic")
    for number, colour in colours.items():
        mass = shannon_mass_for_count_mearth(number, inner, outer) / MEARTH_PER_MJUP
        radii = np.geomspace(inner, outer, number)
        top.plot(radii, np.full(number, mass), "o-", color=colour, lw=1.4, ms=4.5)
        top.text(outer + 0.5, mass, rf"$N={number}$", color=colour, va="center", fontsize=8)
    top.text(inner / 2, 0.012, "Asteroid Belt", color="#994c00", ha="center", va="center", rotation=90)
    top.text((outer + edges["outer"]) / 2, 0.012, "Kuiper Belt", color="#4c5c92", ha="center", va="center", rotation=90)
    top.set(xlim=(0, 48), ylim=(5e-5, 2), yscale="log", xlabel="Semimajor axis (au)",
            ylabel=r"Spacing-implied mass ($M_{\rm Jup}$)")
    top.text(0.02, 0.96, "(a)", transform=top.transAxes, va="top", weight="bold")
    top.grid(alpha=0.18)

    bottom.axhspan(1e-6, floor_jup, color="#d95f5f", alpha=0.10)
    bottom.axhspan(floor_jup, 10, color="#66a061", alpha=0.08)
    bottom.axhline(floor_jup, color="#b22222", lw=1)
    bottom.semilogy(numbers, masses_jup, "-", color="0.55", lw=1)
    for number, mass in zip(numbers, masses_jup):
        bottom.plot(number, mass, "o", color=colours.get(int(number), "0.55"), ms=4)
    bottom.axvline(first_too_light - 0.5, color="#b22222", ls="--", lw=1)
    bottom.annotate(rf"$N_{{\rm crit}}={first_too_light}$", xy=(first_too_light - 0.5, floor_jup),
                    xytext=(19, floor_jup * 9), ha="center", color="#9c2222",
                    arrowprops={"arrowstyle": "->", "color": "#9c2222"})
    bottom.text(2.0, floor_jup * 2.5, "Can clear gap", color="#397739", style="italic")
    bottom.text(3.2, floor_jup * 0.35, "Too light to clear gap", color="#9c2222", style="italic")
    bottom.set(xlim=(0, 26), ylim=(5e-5, 2), yscale="log", xlabel="Number of planets $N$",
               ylabel=r"Spacing-implied mass ($M_{\rm Jup}$)")
    bottom.text(0.02, 0.96, "(b)", transform=bottom.transAxes, va="top", weight="bold")
    bottom.grid(alpha=0.18)
    fig.savefig(FIGURES / "multi_planet_chains.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_four_planet() -> None:
    _style()
    edges = json.loads(EDGES.read_text())
    predictions = json.loads(PREDICTIONS.read_text())["beams"]
    real_a = np.asarray((5.20, 9.58, 19.20, 30.07))
    real_m = np.asarray((317.8, 95.0, 14.6, 17.15))
    names = ("Jupiter", "Saturn", "Uranus", "Neptune")
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.0), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.08})
    for axis, beam, colour, title in zip(
        axes, ("2", "20"), ("#7b3294", "#1b7837"),
        ("High resolution (2 au beam)", "Low resolution (20 au beam)"),
    ):
        edge = edges[beam]
        chain = predictions[beam]["four_planet_chain"]
        radii = np.asarray(chain["semimajor_axes_au"])
        mass_earth = chain["spacing_implied_mass_mearth"]
        mass_jup = mass_earth / MEARTH_PER_MJUP
        axis.axvspan(0, SOLAR_SYSTEM.asteroid_belt_outer_au, color="#b22222", alpha=0.18)
        axis.axvspan(edge["inner"], edge["outer"], color="0.5", alpha=0.28)
        axis.text(2, 0.11, "Asteroid Belt", ha="center", va="center", rotation=90,
                  color="#9b2226", weight="bold")
        axis.text((edge["inner"] + edge["outer"]) / 2, 0.11, "Kuiper Belt",
                  ha="center", va="center", rotation=90, color="0.35", weight="bold")
        axis.errorbar(
            radii, np.full(4, mass_jup),
            xerr=np.vstack((chain["semimajor_axes_down"], chain["semimajor_axes_up"])),
            yerr=np.vstack((np.full(4, chain["spacing_implied_mass_down"] / MEARTH_PER_MJUP),
                            np.full(4, chain["spacing_implied_mass_up"] / MEARTH_PER_MJUP))),
            fmt="o-", color=colour, lw=2, ms=6, capsize=3,
            label=r"Equal-mass $N=4$ chain", zorder=3,
        )
        axis.plot(real_a, real_m / MEARTH_PER_MJUP, "*", color="#ffd000", mec="black",
                  mew=0.8, ms=18, label="Solar System giants", zorder=4)
        ratios = real_m / mass_earth
        offsets = ((25, 10), (0, 10), (0, -24), (0, 10))
        for name, x, y, ratio, offset in zip(names, real_a, real_m / MEARTH_PER_MJUP, ratios, offsets):
            axis.annotate(f"{name} ({ratio:.1f}× chain mass)", (x, y), xytext=offset,
                          textcoords="offset points", ha="center",
                          va="bottom" if offset[1] > 0 else "top", fontsize=8.5, weight="bold")
        axis.text(0.03, 0.94, title, transform=axis.transAxes, va="top", weight="bold",
                  bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.6"})
        axis.set(yscale="log", xlim=(0, 51), ylim=(1e-2, 5),
                 ylabel=r"Planet mass [$M_{\rm Jup}$]")
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("Semimajor axis [au]")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.savefig(FIGURES / "four_planet_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _polar_brightness(image: np.ndarray, pixel_scale: float):
    y, x = np.indices(image.shape)
    x = (x - (image.shape[1] - 1) / 2.0) * pixel_scale
    y = (y - (image.shape[0] - 1) / 2.0) * pixel_scale
    radius = np.hypot(x, y)
    angle = np.degrees(np.arctan2(y, x)) % 360.0
    radial_edges = np.arange(28.0, 50.5, 0.5)
    angle_edges = np.arange(0.0, 362.0, 2.0)
    weighted, _, _ = np.histogram2d(radius.ravel(), angle.ravel(), bins=(radial_edges, angle_edges),
                                     weights=image.ravel())
    counts, _, _ = np.histogram2d(radius.ravel(), angle.ravel(), bins=(radial_edges, angle_edges))
    profile = np.divide(weighted, counts, out=np.zeros_like(weighted), where=counts > 0)
    return radial_edges, angle_edges, profile


def make_azimuthal_structure() -> None:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 5.2), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.12})
    neptune_angle = 1.0
    resonances = {
        "3:2": SOLAR_SYSTEM.neptune_semimajor_axis_au * (3 / 2) ** (2 / 3),
        "5:3": SOLAR_SYSTEM.neptune_semimajor_axis_au * (5 / 3) ** (2 / 3),
        "7:4": SOLAR_SYSTEM.neptune_semimajor_axis_au * (7 / 4) ** (2 / 3),
    }
    for axis, beam in zip(axes, (2, 5)):
        with fits.open(DATA / f"alma_sim_{beam}au.fits") as hdul:
            image = np.maximum(hdul[0].data.astype(float), 0.0)
            pixel = float(hdul[0].header["PIXSCALE"])
        image /= image.max()
        radial_edges, angle_edges, profile = _polar_brightness(image, pixel)
        mesh = axis.pcolormesh(angle_edges, radial_edges, profile, cmap="twilight_shifted",
                               shading="flat", vmin=0, vmax=0.28)
        axis.axvline(neptune_angle, color="cyan", ls="--", lw=1.2)
        axis.plot(neptune_angle, SOLAR_SYSTEM.neptune_semimajor_axis_au, "*", color="cyan", ms=10)
        for name, radius in resonances.items():
            axis.axhline(radius, color="black", ls=":", lw=1)
            axis.text(360.4, radius, name, va="center", fontsize=9, weight="bold", clip_on=False)
        axis.text(0.02, 0.92, f"FWHM = {beam} au", transform=axis.transAxes, va="top",
                  weight="bold", fontsize=11,
                  bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.5"})
        axis.set_ylabel("Heliocentric\ndistance (au)")
        colourbar = fig.colorbar(mesh, ax=axis, pad=0.025, fraction=0.025)
        colourbar.set_label("Normalised\nsurface brightness")
    axes[-1].set_xlabel("Azimuthal angle (deg; counter-clockwise from image +x)")
    axes[-1].set_xlim(0, 360)
    axes[-1].set_xticks(np.arange(0, 361, 30))
    axes[-1].set_ylim(28, 50)
    fig.savefig(FIGURES / "azimuthal_structure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_all_figures() -> None:
    make_synthetic_images()
    make_single_planet()
    make_multi_planet()
    make_four_planet()
    make_azimuthal_structure()
