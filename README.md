# What planets would we infer from an extrasolar Kuiper Belt?

Reproducible code and compact derived data for the MNRAS manuscript
*What planets would we infer from an extrasolar Kuiper Belt? Testing
debris-disc planet predictions in our Solar System* by Rishin Georgy Anil,
Avaneesh Subramanian, and Tim D. Pearce.

The benchmark turns the Kuiper Belt into synthetic face-on ALMA-like images,
recovers its radial surface-density profile at 2--20 au physical resolution,
and asks what standard single- and multi-planet debris-disc prescriptions
would infer. The high-resolution result is Neptune-like: the Pearce--Wyatt
single-planet model gives approximately 32.6 au and 20.6 Earth masses from the
2-au image.

## Reproduce the analysis

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --extra test
uv run python scripts/reproduce.py
uv run python -m pytest
```

The complete run refits the five FITS images, writes machine-readable values
under `results/`, and recreates six figures under `figures/`. Random covariance
sampling uses a fixed seed, so the regression results are deterministic.

## Repository layout

```text
data/                 Compact input data and provenance
SCIENTIFIC_VALIDATION.md  Equation audit and control tests
src/                  Profile fitting and dynamical inference library
scripts/reproduce.py  One-command reproduction entry point
results/              Fitted edges and derived planet predictions
figures/              Recreated publication figures
tests/                 Numerical regression and equation checks
```

The original exploratory notebooks, LaTeX build products, repeated plots,
large REBOUND snapshots, and unused catalogue columns are deliberately absent.

## Analysis choices

The synthetic images contain surface brightness proportional to surface
density times `r**-0.5`. After 15 Richardson--Lucy iterations, the radial
profiles are therefore multiplied by `sqrt(r)` before fitting. This restores
surface density rather than fitting temperature-weighted surface brightness.
The preferred profile is the least-parameterised fit within two BIC units of
the minimum; AIC and BIC values for every candidate are retained in the JSON.
Half-maximum edge uncertainties come from 5,000 fixed-seed covariance draws.

Single-planet locations are intersections between each published exterior
clearance-width law and the scattering-time lower bound. The stirring
eccentricity separately inverts the general secular-stirring timescale at the
inferred semimajor axis and measured outer edge.

For the multi-planet calculation, two masses must not be conflated:

- the age-based clearing floor is Shannon et al. (2016) equation 4
  (about 0.20 Earth masses here);
- the approximately 20.3-Earth-mass four-planet value is obtained by inverting
  their equation 5 and requiring exactly four equal-mass planets at the adopted
  typical spacing to span 4 au to the fitted Kuiper Belt inner edge.

The latter is a spacing-implied mass conditional on `N=4`, not the general
age-based minimum mass. The code also records the unrounded equation-1 value
so the approximation introduced by the `0.13` coefficient is transparent.

## Numerical headline (2-au beam)

| Quantity | Reproduced value |
|---|---:|
| Kuiper Belt inner edge | 37.061 au |
| Kuiper Belt outer edge | 47.120 au |
| Pearce--Wyatt planet semimajor axis | 32.595 au |
| Pearce--Wyatt planet mass | 20.558 Earth masses |
| Required stirring eccentricity | 1.676e-4 |
| Four-planet spacing-implied mass | 20.347 Earth masses |

## Primary references checked against the implementation

- [Pearce et al. (2022), A&A 659, A135](https://doi.org/10.1051/0004-6361/202142720): scattering constraint (equation 7) and general secular-stirring timescale (equation 22).
- [Pearce & Wyatt (2014), MNRAS 443, 2541](https://doi.org/10.1093/mnras/stu1302): eccentric exterior unstable-zone relation and clearing times.
- [Shannon et al. (2016), MNRAS Letters 462, L116](https://doi.org/10.1093/mnrasl/slw143): equal-mass planet spacing and clearing-time constraints (equations 1, 4, and 5).
- [Morrison & Malhotra (2015), ApJ 799, 41](https://doi.org/10.1088/0004-637X/799/1/41): exterior cleared-zone width and summary of the Wisdom, Duncan, and Malhotra coefficients.
- [Lazzoni et al. (2018), A&A 611, A43](https://doi.org/10.1051/0004-6361/201731426): eccentric exterior chaotic-zone width (equation 9).
- [Wyatt (2003), ApJ 598, 1321](https://doi.org/10.1086/379064): resonant-clump interpretation and migration-dependent 3:2 libration centre.

## Scope and limitations

This is a controlled best-case benchmark, not an end-to-end ALMA simulator.
The images use circular Gaussian beams and omit thermal noise, incomplete
Fourier sampling, calibration errors, and realistic visibility modelling. The
JPL catalogue is selection biased, and its simplified azimuthal randomisation
is not a survey debiasing model. The inference prescriptions are intentionally
applied in the same idealised form used in the manuscript.

The distinction between the archived radial-fit maps and the Kuiper-only
display maps is documented in `data/README.md` and quantified in
`SCIENTIFIC_VALIDATION.md`.

## License and citation

Software is released under the MIT License. Citation metadata is provided in
`CITATION.cff`; please also cite the accompanying manuscript when available.
