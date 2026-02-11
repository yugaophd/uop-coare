import os
import sys
from datetime import datetime, timezone

import numpy as np
import pytest


# Ensure src/ is on the import path for direct module import without installation
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(THIS_DIR, "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

from uop_coare.coare36vn_zrf_et import calculate_jd, coare36vn_zrf_et


def _build_constant_inputs(N, sw_dn=None, lat=45.0, lon=-30.0, jd=None):
    if sw_dn is None:
        sw_dn = np.array([0.0, 200.0, 600.0, 100.0], dtype=float)[:N]
    if jd is None:
        jd = np.arange(N, dtype=float)

    u = np.full(N, 5.0)
    zu = np.full(N, 10.0)
    t = np.full(N, 20.0)
    zt = np.full(N, 10.0)
    rh = np.full(N, 80.0)
    zq = np.full(N, 10.0)
    P = np.full(N, 1013.0)
    ts = np.full(N, 21.0)
    lw_dn = np.full(N, 400.0)
    zi = np.full(N, 600.0)
    rain = np.full(N, 0.0)
    Ss = np.full(N, 35.0)
    lat_arr = np.full(N, float(lat))
    lon_arr = np.full(N, float(lon))

    return dict(
        u=u,
        zu=zu,
        t=t,
        zt=zt,
        rh=rh,
        zq=zq,
        P=P,
        ts=ts,
        sw_dn=np.asarray(sw_dn, dtype=float),
        lw_dn=lw_dn,
        lat=lat_arr,
        lon=lon_arr,
        jd=np.asarray(jd),
        zi=zi,
        rain=rain,
        Ss=Ss,
    )


def test_calculate_jd_numpy_datetime64():
    times = np.array([
        np.datetime64("2025-01-01T00:00:00"),
        np.datetime64("2025-01-01T12:00:00"),
        np.datetime64("2025-01-02T00:00:00"),
        np.datetime64("2025-01-03T06:00:00"),
    ])
    jd = calculate_jd(times)
    expected = np.array([1.0, 1.5, 2.0, 3.25])
    assert np.allclose(jd, expected)


def test_calculate_jd_python_datetime_utc():
    times = [
        datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 3, 6, 0, tzinfo=timezone.utc),
    ]
    jd = calculate_jd(times)
    expected = np.array([1.0, 1.5, 2.0, 3.25])
    assert np.allclose(jd, expected)


def test_calculate_jd_pandas_index():
    pd = pytest.importorskip("pandas")
    times = pd.to_datetime([
        "2025-01-01T00:00:00Z",
        "2025-01-01T12:00:00Z",
        "2025-01-02T00:00:00Z",
        "2025-01-03T06:00:00Z",
    ])
    jd = calculate_jd(times)
    expected = np.array([1.0, 1.5, 2.0, 3.25])
    assert np.allclose(jd, expected)


def test_coare36vn_accepts_datetime_jd_small():
    # Build a small test with datetime jd inputs to ensure internal conversion
    N = 4
    pd = None
    try:
        import pandas as _pd
        pd = _pd
    except Exception:
        pass

    if pd:
        times = pd.to_datetime([
            "2025-01-01T00:00:00Z",
            "2025-01-01T12:00:00Z",
            "2025-01-02T00:00:00Z",
            "2025-01-03T06:00:00Z",
        ])
    else:
        times = np.array([
            np.datetime64("2025-01-01T00:00:00"),
            np.datetime64("2025-01-01T12:00:00"),
            np.datetime64("2025-01-02T00:00:00"),
            np.datetime64("2025-01-03T06:00:00"),
        ])

    inputs = _build_constant_inputs(N, jd=times)
    A = coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=True)
    assert isinstance(A, np.ndarray)
    assert A.shape[0] == N
    # sw_net column should be finite for daylight values
    sw_net = A[:, 28]
    assert np.any(np.isfinite(sw_net))


def test_albedo_error_raised_on_inconsistent_sw(monkeypatch):
    # Force a daytime scenario (psi_sw > 0) with near-zero albedo to trigger the guard.
    N = 3
    inputs = _build_constant_inputs(N, sw_dn=np.array([50.0, 100.0, 200.0]))

    def fake_albedo_vector(sw_dn, jd, lon, lat, eorw="E"):
        alb = np.zeros_like(sw_dn, dtype=float)  # < 0.001
        T_sw = np.ones_like(sw_dn, dtype=float)
        solarmax = np.ones_like(sw_dn, dtype=float) * 1000.0
        psi = np.ones_like(sw_dn, dtype=float) * 30.0  # daytime
        return alb, T_sw, solarmax, psi

    # Patch the albedo function used by coare36vn_zrf_et
    import uop_coare.coare36vn_zrf_et as mod
    monkeypatch.setattr(mod, "albedo_vector", fake_albedo_vector)

    with pytest.raises(ValueError) as exc:
        coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=False)

    assert "Albedo error" in str(exc.value)
