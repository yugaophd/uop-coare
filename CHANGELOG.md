# Changelog

## 1.5.0

- Pass NaNs through the COARE solver without dropping calculation-grid rows.
- Handle non-finite samples in the Payne (1972) albedo lookup without raising
  an indexing error.
- Preserve full-grid `L3_*` inputs and add `COARE_valid_sample` plus total and
  valid sample counts to NetCDF output.
- Keep the existing conservative radiation requirement: missing shortwave or
  longwave radiation can leave turbulent fluxes missing, including with cool
  skin disabled.
- Fully valid input results and the radiation policy are unchanged.
