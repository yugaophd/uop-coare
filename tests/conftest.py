import glob
import os

import numpy as np
import pytest


def _find_dataarray(ds, candidates):
    for name in candidates:
        if name in ds:
            return ds[name]
    return None


def _find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return ds.coords[name]
    return None


def _select_time_series(da):
    # Reduce to a 1D time series if possible by selecting the first index
    # for non-time dimensions.
    if "time" in da.dims:
        for dim in da.dims:
            if dim != "time" and da.sizes.get(dim, 0) > 1:
                da = da.isel({dim: 0})
    return da


def _broadcast_to_target(target, arr, name):
    arr = np.asarray(arr)
    if arr.size == 1:
        return np.full_like(target, float(arr.reshape(-1)[0]))
    if arr.size == target.size:
        return arr.reshape(target.shape)
    try:
        return np.broadcast_to(arr, target.shape)
    except ValueError as exc:
        raise pytest.SkipTest(
            f"Cannot align '{name}' with sw_dn shape {target.shape}; got {arr.shape}."
        ) from exc


def _get_jd_from_time(time_da):
    pandas = pytest.importorskip("pandas")
    times = pandas.to_datetime(time_da.values)
    dayofyear = times.dayofyear
    frac = (times.hour + times.minute / 60 + times.second / 3600) / 24.0
    return np.asarray(dayofyear + frac, dtype=float)


@pytest.fixture(scope="session")
def l4_file_list():
    campaign = os.environ.get("CAMPAIGN", "IOP1")
    version = os.environ.get("VERSION", "V2")
    default_dir = f"/mnt/d/tom_data/S-MODE/Wave_Glider/data/{campaign}/L4/"
    l4_dir = os.environ.get("L4_DIR", default_dir)
    pattern = os.path.join(l4_dir, f"*{version}.nc")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        pytest.skip(
            f"No L4 files found at {pattern}. Set L4_DIR/CAMPAIGN/VERSION to locate data."
        )
    return file_list


@pytest.fixture(scope="session")
def l4_sample_arrays(l4_file_list):
    xr = pytest.importorskip("xarray")

    ds = xr.open_dataset(l4_file_list[0])

    sw_dn_da = _find_dataarray(
        ds,
        [
            "L3_SMP21_shortwave_flux",
            "L3_shortwave_flux",
            "sw_dn",
            "SWDN",
            "shortwave_down",
            "sw_down",
            "SW_down",
            "downwelling_shortwave",
        ],
    )
    if sw_dn_da is None:
        pytest.skip("No shortwave downwelling variable found for sw_dn.")

    sw_dn_da = _select_time_series(sw_dn_da)
    sw_dn = np.asarray(sw_dn_da.values, dtype=float).reshape(-1)

    lat_da = _find_coord(ds, ["lat", "latitude"])
    if lat_da is None:
        lat_da = _find_dataarray(ds, ["lat", "latitude"])

    lon_da = _find_coord(ds, ["lon", "longitude"])
    if lon_da is None:
        lon_da = _find_dataarray(ds, ["lon", "longitude"])

    if lat_da is None or lon_da is None:
        pytest.skip("Latitude/longitude not found in dataset.")

    lat = _select_time_series(lat_da) if hasattr(lat_da, "dims") else lat_da
    lon = _select_time_series(lon_da) if hasattr(lon_da, "dims") else lon_da

    time_da = sw_dn_da.coords.get("time")
    if time_da is None:
        time_da = ds.coords.get("time")

    jd_da = _find_dataarray(ds, ["jd", "yearday", "day_of_year", "doy", "yday"])
    if jd_da is not None:
        jd_da = _select_time_series(jd_da)
        jd = np.asarray(jd_da.values, dtype=float).reshape(-1)
    else:
        if time_da is None:
            pytest.skip("No jd variable or time coordinate to compute yearday.")
        jd = _get_jd_from_time(time_da).reshape(-1)

    lat = _broadcast_to_target(sw_dn, lat, "lat")
    lon = _broadcast_to_target(sw_dn, lon, "lon")
    jd = _broadcast_to_target(sw_dn, jd, "jd")

    return sw_dn, lat, lon, jd, time_da
