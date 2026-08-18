# Input data

`profile_fit_images/` contains the five archived FITS files used for the
reported radial-profile fits. Their headers record the physical pixel scale
and beam FWHM. The fit itself only reads the 30--60 au annulus. These archived
maps retain the inner catalogue component used during the original analysis;
at high resolution it does not leak materially into the fitted annulus, while
at coarse resolution it is part of the benchmark's measured resolution bias.

`kuiper_only_images/` is a controlled re-render from the same catalogue after
selecting objects with semimajor axes `a >= 30 au`. The sparse set of selected
points projected within 20 au is also omitted from these display maps. These
files are used only for the manuscript's displayed Kuiper Belt and azimuthal
figures, not to replace the exact profile-fit inputs after the fact.
`SCIENTIFIC_VALIDATION.md` quantifies the resulting high-resolution control
check.

Both sets originate from JPL Small-Body Database orbital elements downloaded
on 2026-03-05, propagated to MJD 61104.61, azimuthally randomised as described
in the manuscript, and weighted by the adopted blackbody-temperature factor
`r**-0.5`.

`catalogue_surface_density.csv` is the 0.1-au binned, area-normalised radial
profile of that catalogue. It is retained instead of the 78-MB intermediate
object array because the analysis only uses this one-dimensional derived
quantity. The public repository therefore contains every archived input
needed to reproduce the reported profile fits and figures without retaining
redundant per-object columns.

The JPL query interface is at <https://ssd.jpl.nasa.gov/tools/sbdb_query.html>.
The catalogue is observationally selected and is not an intrinsic-population
model; this benchmark uses it for controlled morphology, not population
abundances.
