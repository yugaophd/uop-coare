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

## Notes

- `coolskin=True` runs the cool-skin model; set `coolskin=False` to disable it.
- `albedo="constant"` uses a constant albedo correction; use `albedo="Payne1972"` for zenith-angle varying albedo.
- `return_albedo=False` keeps the legacy output shape; set `return_albedo=True` to append `alb`, `T_sw`, `solarmax_sw`, and `psi_sw` to the output array.
- Inputs may be scalars or vectors. See the function docstring for full details.
- `jd` can be numeric yearday or datetime-like inputs (UTC); datetime inputs are converted to yearday internally.

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
