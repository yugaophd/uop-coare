# pycoare

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

## Quick Start

```python
from pycoare.coare36vn_zrf_et import coare36vn_zrf_et

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
)
```

## Notes

- `coolskin=True` runs the cool-skin model; set `coolskin=False` to disable it.
- `albedo="constant"` uses a constant albedo correction; use `albedo="Payne1972"` for zenith-angle varying albedo.
- Inputs may be scalars or vectors. See the function docstring for full details.
