# uop-coare

Python package for COARE 3.6 bulk air-sea flux calculations (no warm layer computations).

## Install

Editable install (recommended while developing):
```bash
pip install -e .
```

Standard install:
```bash
pip install .
```

## Update

Editable install (pull latest code, then re-run if needed):
```bash
pip install -e .
```

Standard install:
```bash
pip install -U .
```

## Quick Start

```python
from uop_coare.coare36vn_zrf_et import coare36vn_zrf_et

# Example call (replace with real data/arrays)
A = coare36vn_zrf_et(
    u=5.0,
    zu=10.0,
    t=20.0,
    zt=10.0,
    rh=80.0,
    zq=10.0,
    P=1013.0,
    ts=21.0,
    sw_dn=200.0,
    lw_dn=400.0,
    lat=45.0,
    lon=-30.0,
    jd=100.0,
    zi=600.0,
    rain=0.0,
    Ss=35.0,
    cp=None,
    sigH=None,
    zrf_u=10.0,
    zrf_t=10.0,
    zrf_q=10.0,
    coolskin=True,
    albedo="constant",
    return_albedo=False,
)
```

## NetCDF Wrapper

Use the xarray wrapper to run COARE and write a NetCDF file directly:

```python
import numpy as np
from uop_coare import coare36_to_netcdf

N = 24
coare36_to_netcdf(
    "coare36_fluxes.nc",
    "global_attrs.json",
    u=np.full(N, 5.0),
    zu=np.full(N, 10.0),
    t=np.full(N, 20.0),
    zt=np.full(N, 10.0),
    rh=np.full(N, 80.0),
    zq=np.full(N, 10.0),
    P=np.full(N, 1013.0),
    ts=np.full(N, 21.0),
    sw_dn=np.full(N, 200.0),
    lw_dn=np.full(N, 400.0),
    lat=np.full(N, 45.0),
    lon=np.full(N, -30.0),
    jd=np.linspace(100.0, 101.0, N),
    zi=np.full(N, 600.0),
    rain=np.full(N, 0.0),
    Ss=np.full(N, 35.0),
    albedo="Payne1972",
)
```

`global_attrs.json` must be a JSON object with custom global attributes,
for example `{ "project": "S-MODE", "config_file": "WFIP_config_Site3.json" }`.

The output NetCDF includes COARE variables with units, CF-style metadata,
and coordinate variables including latitude, longitude, and fractional day-of-year.

## Imports

```python
# High-level wrapper (recommended)
from uop_coare import coare36_to_netcdf

# Low-level solver (returns raw numpy array)
from uop_coare.coare36vn_zrf_et import coare36vn_zrf_et

# Variable metadata dictionary
from uop_coare import get_variables_info
```

## Notes

- `coolskin=True` runs the cool-skin model; set `coolskin=False` to disable it.
- `albedo="constant"` uses a constant albedo correction; `albedo="Payne1972"` uses zenith-angle varying albedo from Payne (1972).
- `coare36_to_netcdf` accepts the same core inputs as `coare36vn_zrf_et` and writes a CF-convention NetCDF file with all output variables named and attributed from `get_variables_info`.
- NetCDF reference-height metadata defaults to 10 m for wind and 2 m for air temperature and humidity.
- The air-density output is COARE `rhoa`, computed inside the solver from `t`, `rh`, and pressure adjusted to `zt`; it is not a caller-supplied input and is not recomputed from the reference-height profile outputs.
- `uop_coare` can compute current-relative wind for you: pass measured wind
  in `u` and `wind_direction`, and eastward/northward surface current
  components in `current_east` / `current_north`. The wrapper subtracts the
  current internally and passes the current-relative wind speed and direction
  to the COARE 3.6 core. `L3_wind_speed` and `L3_wind_direction` continue to
  store the measured values (with attrs from `input_vars_used`), while the
  arrays fed to the core are exported as
  `L3_relative_wind_speed` / `L3_relative_wind_direction`. Omit the current
  kwargs to keep the legacy behaviour where `u` is expected to be
  pre-subtracted.
- `current_available=True` in `coare36_to_netcdf` only annotates L3 wind metadata as current-relative. It does not change wind values or perform any wind/current adjustment.
- If surface current is missing, callers should handle that upstream by passing true wind, not by substituting `0 m/s` wind.
- When `source_dataset` is provided to `coare36_to_netcdf`, instrument variables are copied through unchanged and non-instrument L3 variables are carried into the L4 file as `L3_*` variables with their metadata preserved.
- All COARE-derived outputs carry `coverage_content_type: modelResult`; copied L3 inputs carry `physicalMeasurement`.
- Inputs may be scalars or vectors. See the function docstring for full details.
- `jd` can be numeric yearday or datetime-like inputs (UTC); datetime inputs are converted to yearday internally.

## NaN handling (v1.5.0)

The solver and NetCDF wrapper accept full-grid input arrays containing NaNs.
Rows are not dropped, inputs are not filled, and raw `L3_*` values are
preserved. Outputs that depend on a missing input are NaN for that sample;
physically independent outputs may remain finite. With `albedo="Payne1972"`,
non-finite shortwave, yearday, latitude, or longitude produces NaN albedo and
solar diagnostics for that sample.

The wrapper writes `COARE_valid_sample` (1 where `friction_velocity` is finite)
and the global sample-count attributes `coare_total_samples` and
`coare_valid_samples`. Missing shortwave or longwave radiation can still leave
turbulent fluxes missing when `coolskin=False`; this conservative radiation
requirement is unchanged. No default filling or radiation-independent mode is
provided.

## Test Demo

```bash
pip install -e ".[test]"
pytest
```

Plot outputs (saved by default):
- `tests/artifacts/albedo_payne1972.png`
- `tests/artifacts/albedo_payne1972_first48h.png`
- `tests/artifacts/sw_net_compare_first48h.png`
- `tests/artifacts/sw_net_constant_albedo.png`

Disable plot generation:
```bash
PLOT_TESTS=0 pytest
```
