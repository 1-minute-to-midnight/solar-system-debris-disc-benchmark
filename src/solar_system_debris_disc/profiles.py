#!/usr/bin/env python3
"""Fit temperature-corrected radial surface-density profiles.

The synthetic FITS images contain surface brightness proportional to
surface density times r**(-1/2).  This script follows the paper pipeline
(15 Richardson-Lucy iterations), azimuthally averages each deconvolved
image, multiplies the recovered profile by r**(1/2), fits the literature
profile families, selects the simplest model within Delta BIC <= 2, and
propagates the fitted-parameter covariance to the half-maximum edges.

It writes the revised radial-profile figure and a JSON file containing all
edge measurements needed by the planet-inference calculations.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from lmfit import Model
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf
from skimage import restoration


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURE = ROOT / "figures" / "radial_surface_density.png"
RESULTS = ROOT / "results" / "surface_density_edges.json"
FITS_FILES = [f"alma_sim_{beam}au.fits" for beam in (2, 3, 5, 10, 20)]


def gaussian_psf(fwhm_pixels: float) -> np.ndarray:
    sigma = fwhm_pixels / 2.355
    half = int(np.ceil(3 * sigma))
    axis = np.arange(-half, half + 1)
    x_grid, y_grid = np.meshgrid(axis, axis)
    psf = np.exp(-0.5 * ((x_grid / sigma) ** 2 + (y_grid / sigma) ** 2))
    return psf / psf.sum()


def double_power_law(r, r_c, alpha_in, alpha_out, gamma=2.0):
    return (
        (r / r_c) ** (-alpha_in * gamma)
        + (r / r_c) ** (-alpha_out * gamma)
    ) ** (-1.0 / gamma)


def triple_power_law(
    r, r_in, r_out, alpha_in, alpha_mid, alpha_out, gamma_in=2.0, gamma_out=2.0
):
    term_in = (
        (r / r_in) ** (-alpha_in * gamma_in)
        + (r / r_in) ** (-alpha_mid * gamma_in)
    ) ** (-1.0 / gamma_in)
    term_out = (
        (r / r_out) ** (-alpha_mid * gamma_out)
        + (r / r_out) ** (-alpha_out * gamma_out)
    ) ** (-1.0 / gamma_out)
    return (r_in / r_out) ** (-alpha_mid) * term_in * term_out


def power_law_erf(r, r_c, sigma_in, alpha_out):
    argument = (r_c - r) / (np.sqrt(2.0) * sigma_in * r_c)
    return (1.0 - erf(argument)) * (r / r_c) ** (-alpha_out)


def gaussian(r, centre, sigma):
    return np.exp(-0.5 * ((r - centre) / sigma) ** 2)


def asymmetric_gaussian(r, centre, sigma_in, sigma_out):
    sigma = np.where(r < centre, sigma_in, sigma_out)
    return gaussian(r, centre, sigma)


def double_gaussian(r, r_1, r_2, sigma_1, sigma_2, fraction):
    return fraction * gaussian(r, r_1, sigma_1) + (1.0 - fraction) * gaussian(
        r, r_2, sigma_2
    )


def triple_gaussian(
    r, r_1, r_2, r_3, sigma_1, sigma_2, sigma_3, fraction_1, fraction_2
):
    return (
        fraction_1 * gaussian(r, r_1, sigma_1)
        + fraction_2 * gaussian(r, r_2, sigma_2)
        + (1.0 - fraction_1 - fraction_2) * gaussian(r, r_3, sigma_3)
    )


MODELS = {
    "Double power law": Model(
        lambda r, amplitude, r_c, alpha_in, alpha_out: amplitude
        * double_power_law(r, r_c, alpha_in, alpha_out),
        independent_vars=["r"],
    ),
    "Triple power law": Model(
        lambda r, amplitude, r_in, r_out, alpha_in, alpha_mid, alpha_out: amplitude
        * triple_power_law(r, r_in, r_out, alpha_in, alpha_mid, alpha_out),
        independent_vars=["r"],
    ),
    "Power law + erf": Model(
        lambda r, amplitude, r_c, sigma_in, alpha_out: amplitude
        * power_law_erf(r, r_c, sigma_in, alpha_out),
        independent_vars=["r"],
    ),
    "Gaussian": Model(
        lambda r, amplitude, centre, sigma: amplitude * gaussian(r, centre, sigma),
        independent_vars=["r"],
    ),
    "Asymmetric Gaussian": Model(
        lambda r, amplitude, centre, sigma_in, sigma_out: amplitude
        * asymmetric_gaussian(r, centre, sigma_in, sigma_out),
        independent_vars=["r"],
    ),
    "Double Gaussian": Model(
        lambda r, amplitude, r_1, r_2, sigma_1, sigma_2, fraction: amplitude
        * double_gaussian(r, r_1, r_2, sigma_1, sigma_2, fraction),
        independent_vars=["r"],
    ),
    "Triple Gaussian": Model(
        lambda r, amplitude, r_1, r_2, r_3, sigma_1, sigma_2, sigma_3, fraction_1, fraction_2: amplitude
        * triple_gaussian(
            r, r_1, r_2, r_3, sigma_1, sigma_2, sigma_3, fraction_1, fraction_2
        ),
        independent_vars=["r"],
    ),
    "Gaussian + double PL": Model(
        lambda r, amplitude, centre, sigma, fraction, r_c, alpha_in, alpha_out: amplitude
        * (
            fraction * gaussian(r, centre, sigma)
            + (1.0 - fraction) * double_power_law(r, r_c, alpha_in, alpha_out)
        ),
        independent_vars=["r"],
    ),
}


def model_parameters():
    parameters = {}
    parameters["Double power law"] = MODELS["Double power law"].make_params(
        amplitude=1.0, r_c=43.0, alpha_in=3.0, alpha_out=3.0
    )
    parameters["Double power law"]["r_c"].set(min=30, max=60)
    for name in ("amplitude", "alpha_in", "alpha_out"):
        parameters["Double power law"][name].min = 0

    parameters["Triple power law"] = MODELS["Triple power law"].make_params(
        amplitude=1.0,
        r_in=40.0,
        r_out=50.0,
        alpha_in=3.0,
        alpha_mid=0.0,
        alpha_out=3.0,
    )
    parameters["Triple power law"]["r_in"].set(min=30, max=50)
    parameters["Triple power law"]["r_out"].set(min=40, max=60)
    for name in ("amplitude", "alpha_in", "alpha_out"):
        parameters["Triple power law"][name].min = 0

    parameters["Power law + erf"] = MODELS["Power law + erf"].make_params(
        amplitude=1.0, r_c=43.0, sigma_in=0.1, alpha_out=3.0
    )
    parameters["Power law + erf"]["r_c"].set(min=30, max=60)
    parameters["Power law + erf"]["sigma_in"].set(min=0.01, max=1)
    for name in ("amplitude", "alpha_out"):
        parameters["Power law + erf"][name].min = 0

    parameters["Gaussian"] = MODELS["Gaussian"].make_params(
        amplitude=1.0, centre=43.0, sigma=5.0
    )
    parameters["Gaussian"]["centre"].set(min=30, max=60)
    parameters["Gaussian"]["sigma"].min = 0.5
    parameters["Gaussian"]["amplitude"].min = 0

    parameters["Asymmetric Gaussian"] = MODELS["Asymmetric Gaussian"].make_params(
        amplitude=1.0, centre=43.0, sigma_in=3.0, sigma_out=8.0
    )
    parameters["Asymmetric Gaussian"]["centre"].set(min=30, max=60)
    for name in ("sigma_in", "sigma_out"):
        parameters["Asymmetric Gaussian"][name].min = 0.5
    parameters["Asymmetric Gaussian"]["amplitude"].min = 0

    parameters["Double Gaussian"] = MODELS["Double Gaussian"].make_params(
        amplitude=1.0, r_1=40.0, r_2=48.0, sigma_1=3.0, sigma_2=5.0, fraction=0.5
    )
    for name in ("r_1", "r_2"):
        parameters["Double Gaussian"][name].set(min=30, max=60)
    for name in ("sigma_1", "sigma_2"):
        parameters["Double Gaussian"][name].min = 0.5
    parameters["Double Gaussian"]["fraction"].set(min=0, max=1)
    parameters["Double Gaussian"]["amplitude"].min = 0

    parameters["Triple Gaussian"] = MODELS["Triple Gaussian"].make_params(
        amplitude=1.0,
        r_1=39.0,
        r_2=43.0,
        r_3=48.0,
        sigma_1=2.0,
        sigma_2=3.0,
        sigma_3=4.0,
        fraction_1=0.3,
        fraction_2=0.4,
    )
    for name in ("r_1", "r_2", "r_3"):
        parameters["Triple Gaussian"][name].set(min=30, max=60)
    for name in ("sigma_1", "sigma_2", "sigma_3"):
        parameters["Triple Gaussian"][name].min = 0.5
    for name in ("fraction_1", "fraction_2"):
        parameters["Triple Gaussian"][name].set(min=0, max=1)
    parameters["Triple Gaussian"]["amplitude"].min = 0

    parameters["Gaussian + double PL"] = MODELS["Gaussian + double PL"].make_params(
        amplitude=1.0,
        centre=43.0,
        sigma=5.0,
        fraction=0.5,
        r_c=43.0,
        alpha_in=3.0,
        alpha_out=3.0,
    )
    parameters["Gaussian + double PL"]["centre"].set(min=30, max=60)
    parameters["Gaussian + double PL"]["r_c"].set(min=30, max=60)
    parameters["Gaussian + double PL"]["sigma"].min = 0.5
    parameters["Gaussian + double PL"]["fraction"].set(min=0, max=1)
    for name in ("amplitude", "alpha_in", "alpha_out"):
        parameters["Gaussian + double PL"][name].min = 0
    return parameters


def half_max_edges(r, profile):
    half_max = np.max(profile) / 2.0
    peak_index = int(np.argmax(profile))
    crossings = np.where(np.diff(np.sign(profile - half_max)))[0]
    inner_index = crossings[crossings < peak_index][-1]
    outer_index = crossings[crossings >= peak_index][0]
    interpolation = lambda value: np.interp(value, r, profile) - half_max
    inner = brentq(interpolation, r[inner_index], r[inner_index + 1])
    outer = brentq(interpolation, r[outer_index], r[outer_index + 1])
    return inner, outer, half_max


def edge_uncertainties(fit, r, draws=5000):
    best_profile = fit.eval(r=r)
    inner, outer, half_max = half_max_edges(r, best_profile)
    if fit.covar is None:
        raise RuntimeError("selected model has no covariance matrix")
    names = [name for name in fit.params if fit.params[name].vary]
    values = np.array([fit.params[name].value for name in names])
    random = np.random.default_rng(42).multivariate_normal(values, fit.covar, size=draws)
    inner_draws, outer_draws = [], []
    for sample in random:
        parameters = fit.params.copy()
        valid = True
        for name, value in zip(names, sample):
            parameter = parameters[name]
            if value < parameter.min or value > parameter.max:
                valid = False
                break
            parameter.value = value
        if not valid:
            continue
        try:
            sampled_profile = fit.model.eval(params=parameters, r=r)
            sampled_inner, sampled_outer, _ = half_max_edges(r, sampled_profile)
        except (ValueError, IndexError):
            continue
        inner_draws.append(sampled_inner)
        outer_draws.append(sampled_outer)
    if len(inner_draws) < 500:
        raise RuntimeError(f"only {len(inner_draws)} valid covariance draws")
    inner_16, inner_84 = np.percentile(inner_draws, (15.865, 84.135))
    outer_16, outer_84 = np.percentile(outer_draws, (15.865, 84.135))
    return {
        "inner": inner,
        "inner_up": max(0.0, inner_84 - inner),
        "inner_down": max(0.0, inner - inner_16),
        "outer": outer,
        "outer_up": max(0.0, outer_84 - outer),
        "outer_down": max(0.0, outer - outer_16),
        "half_max": half_max,
        "valid_draws": len(inner_draws),
    }


def recover_and_fit(filename):
    with fits.open(DATA / "profile_fit_images" / filename) as hdul:
        image = np.maximum(hdul[0].data.astype(float), 0.0)
        pixel_scale = float(hdul[0].header["PIXSCALE"])
        beam = float(hdul[0].header["BEAMFWHM"])
    deconvolved = restoration.richardson_lucy(
        image, gaussian_psf(beam / pixel_scale), num_iter=15, clip=True
    )
    y_pixels, x_pixels = deconvolved.shape
    x = np.arange(x_pixels) - x_pixels // 2
    y = np.arange(y_pixels) - y_pixels // 2
    x_grid, y_grid = np.meshgrid(x, y)
    radius = np.hypot(x_grid, y_grid) * pixel_scale
    radial_edges = np.arange(30.0, 60.0, 1.0)
    radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    brightness = np.array(
        [
            np.mean(
                deconvolved[
                    (radius >= radial_edges[index]) & (radius < radial_edges[index + 1])
                ]
            )
            for index in range(len(radial_centres))
        ]
    )
    surface_density = brightness * np.sqrt(radial_centres)
    surface_density /= surface_density.max()

    fits_by_model = {}
    parameter_sets = model_parameters()
    for name, model in MODELS.items():
        # Some rejected over-parameterised candidates have an indefinite
        # covariance matrix. lmfit warns while estimating their unused stderr;
        # model selection and the selected model covariance remain valid.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in sqrt", category=RuntimeWarning
            )
            fits_by_model[name] = model.fit(
                surface_density,
                parameter_sets[name],
                r=radial_centres,
                method="least_squares",
                max_nfev=50000,
            )
    ranking = sorted(fits_by_model.items(), key=lambda item: item[1].bic)
    threshold = ranking[0][1].bic + 2.0
    candidates = [item for item in ranking if item[1].bic <= threshold]
    best_name, best_fit = min(candidates, key=lambda item: (item[1].nvarys, item[1].bic))
    smooth_radius = np.linspace(30.0, 60.0, 1000)
    edges = edge_uncertainties(best_fit, smooth_radius)
    return {
        "filename": filename,
        "beam": beam,
        "radial_centres": radial_centres,
        "surface_density": surface_density,
        "best_name": best_name,
        "best_fit": best_fit,
        "smooth_radius": smooth_radius,
        "edges": edges,
        "ranking_bic": {name: float(result.bic) for name, result in ranking},
        "ranking_aic": {
            name: float(result.aic)
            for name, result in sorted(fits_by_model.items(), key=lambda item: item[1].aic)
        },
    }


def make_figure(results):
    truth_centres, truth = np.loadtxt(
        DATA / "catalogue_surface_density.csv", delimiter=",", skiprows=1
    ).T

    chosen = {result["beam"]: result for result in results}
    colours = {2.0: "#d62728", 5.0: "#1f77b4"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
    for axis, beam in zip(axes, (2.0, 5.0)):
        result = chosen[beam]
        colour = colours[beam]
        radius = result["radial_centres"]
        density = result["surface_density"]
        smooth = result["smooth_radius"]
        fitted = result["best_fit"].eval(r=smooth)
        edges = result["edges"]
        axis.plot(truth_centres, truth, "k--", lw=1, alpha=0.5, label="Catalogue surface density")
        axis.plot(radius, density, "-o", color=colour, ms=5, lw=1.5, alpha=0.75,
                  label="Recovered surface density", zorder=5)
        for index, offset in enumerate(np.linspace(-beam / 2, beam / 2, 30)):
            label = "Recovered profile shifted by up to half a beam width" if index == 0 else None
            axis.plot(radius + offset, density, color=colour, alpha=0.06, lw=8,
                      label=label, zorder=1)
        axis.plot(smooth, fitted, color=colour, lw=2.5,
                  label=f"Best fit: {result['best_name']}")
        axis.axvspan(edges["inner"] - edges["inner_down"],
                    edges["inner"] + edges["inner_up"], color="0.5", alpha=0.22)
        axis.axvspan(edges["outer"] - edges["outer_down"],
                    edges["outer"] + edges["outer_up"], color="0.5", alpha=0.22)
        axis.axvline(edges["inner"], color="0.3", ls="--", lw=1.2)
        axis.axvline(edges["outer"], color="0.3", ls="--", lw=1.2)
        axis.text(0.03, 0.95, f"Beam = {beam:.0f} au", transform=axis.transAxes,
                  va="top", fontsize=16, weight="bold",
                  bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.5"})
        axis.set_xlim(30, 60)
        axis.set_ylim(-0.02, 1.12)
        axis.set_ylabel("Normalised surface density", fontsize=18)
        axis.grid(ls=":", alpha=0.2)
        axis.legend(loc="upper right", fontsize=12)
    axes[-1].set_xlabel("Heliocentric distance (au)", fontsize=18)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")


def main():
    recovered = [recover_and_fit(filename) for filename in FITS_FILES]
    serialisable = {}
    for result in recovered:
        beam = str(int(result["beam"]))
        serialisable[beam] = {
            "best_model": result["best_name"],
            **{name: float(value) for name, value in result["edges"].items()},
            "ranking_bic": result["ranking_bic"],
            "ranking_aic": result["ranking_aic"],
        }
        edge = result["edges"]
        print(
            f"{beam:>2} au  {result['best_name']:<18} "
            f"Rin={edge['inner']:.6f} +{edge['inner_up']:.6f} -{edge['inner_down']:.6f}  "
            f"Rout={edge['outer']:.6f} +{edge['outer_up']:.6f} -{edge['outer_down']:.6f}  "
            f"draws={edge['valid_draws']}"
        )
    RESULTS.write_text(json.dumps(serialisable, indent=2) + "\n")
    make_figure(recovered)
    print(f"wrote {RESULTS}")
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    main()
