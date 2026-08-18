# Scientific validation

This audit was performed while extracting the publication analysis from the
exploratory project. It separates exact numerical reproduction from checks of
the physical formulae and input assumptions.

## Equation audit

| Calculation | Implemented relation | Primary-source check | Result |
|---|---|---|---|
| Circular chaotic-zone widths | Wisdom, Duncan, Gladman and Malhotra coefficients listed in the manuscript appendix | Summarised with the original coefficients in Morrison & Malhotra (2015) | Match |
| Eccentric exterior width | Pearce & Wyatt (2014) exterior unstable-zone relation | Pearce & Wyatt (2014), subsequently used by Pearce et al. (2022) | Match |
| Lazzoni exterior width | `1.3 mu**(2/7) (1+e_p)` | Lazzoni et al. (2018), equation 9 | Match |
| Morrison exterior width | `1.7 mu**0.31` | Morrison & Malhotra (2015) | Match |
| Scattering lower bound | `0.331 a_p R_in**(-1/4) t**(-1/2) M_star**(3/4)` in Jupiter masses | Pearce et al. (2022), equation 7 | Match |
| Secular stirring | General timescale evaluated at the independently inferred `a_p` and measured `R_out` | Pearce et al. (2022), equation 22 | Match |
| Multi-planet clearing floor | `4 Myr/t * a_2**(3/2) * M_star**(1/2)` in Earth masses | Shannon et al. (2016), equation 4 | Match |
| Multi-planet count/spacing | Published `0.13` form and its inversion | Shannon et al. (2016), equations 1 and 5 | Match, with rounding documented |

The stirring implementation deliberately uses the general timescale (Pearce
et al. equation 22). The boundary-placement special case in their equation 23
is not appropriate once the planet semimajor axis has already been inferred
independently.

## Numerical reproduction

With `profile_fit_images/`, a clean pinned environment reproduces the
manuscript analysis exactly (differences below `1e-6` in the fitted edges and
quoted planet values). In particular, the 2-au Pearce--Wyatt result is:

- `R_in = 37.061107 au`, `R_out = 47.119587 au`;
- `a_p = 32.594552 au`, `M_p = 20.557864 M_earth`;
- `e_p,stir = 1.676015e-4`;
- four-planet spacing-implied mass `= 20.3474 M_earth`.

## Kuiper-only input control

The original profile-fit FITS files retain an inner catalogue component, even
though only the 30--60 au annulus is fitted. Re-rendering from objects with
`a >= 30 au` gives, at 2-au resolution, `R_in = 36.582 au` and
`R_out = 47.179 au`. Thus the inner-edge change is `0.48 au` (1.3 per cent),
and the Neptune-like headline inference is unchanged. Leakage from the inner
component grows with beam size, so the coarsest-beam edge values should be
treated as resolution-systematics demonstrations rather than precise physical
measurements.

The public repository keeps the archived profile inputs for exact
reproducibility and uses the separately labelled Kuiper-only maps for the
display and azimuthal figures. This makes the distinction visible rather than
silently substituting newly generated data.

## Interpretation checks

- Multiplication by `sqrt(r)` is required because the synthetic images were
  weighted by `r**-0.5`; the fitted quantity is then surface density.
- The approximately 20.3-Earth-mass four-planet result is conditional on
  exactly four equal-mass planets spanning the gap at the adopted spacing. The
  age-based Shannon clearing floor is only about 0.20 Earth masses.
- The 20-au covariance distribution is strongly asymmetric and has fewer
  valid draws. Its one-sided uncertainty is retained in the JSON rather than
  symmetrised.
- The catalogue morphology is selection biased and the synthetic observations
  omit interferometric sampling and noise. These limitations prevent the
  exercise from being interpreted as a realistic detectability forecast.
