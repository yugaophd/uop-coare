#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(repo_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

from uop_coare.coare36vn_zrf_et import albedo_vector, coare36vn_zrf_et


def _find_dataarray(ds, candidates):
    for name in candidates:
        if name in ds:
            return ds[name]
    return None


def _select_time_series(da):
    if "time" in da.dims:
        for dim in da.dims:
            if dim != "time" and da.sizes.get(dim, 0) > 1:
                da = da.isel({dim: 0})
    return da


def _get_jd_from_time(time_da):
    import pandas as pd

    times = pd.to_datetime(time_da.values)
    dayofyear = times.dayofyear
    frac = (times.hour + times.minute / 60 + times.second / 3600) / 24.0
    return np.asarray(dayofyear + frac, dtype=float)


def _to_1d(arr, name):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0:
        raise SystemExit(f"{name} has no values.")
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Compute Payne (1972) albedo time series from a Wave Glider L4 file."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="/mnt/d/tom_data/S-MODE/Wave_Glider/data/IOP1/L4/S_MODE_IOP1_Wavegliders_Carson_L4_20min_V2.nc",
        help="Path to the L4 netCDF file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a quick plot of albedo vs time.",
    )
    args = parser.parse_args()

    import xarray as xr

    if not os.path.exists(args.path):
        raise SystemExit(f"File not found: {args.path}")

    ds = xr.open_dataset(args.path)

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
        raise SystemExit("No shortwave downwelling variable found.")

    sw_dn_da = _select_time_series(sw_dn_da)
    sw_dn = _to_1d(sw_dn_da.values, "sw_dn")

    lat = _select_time_series(ds["latitude"])
    lon = _select_time_series(ds["longitude"])

    if "jd" in ds:
        jd = _to_1d(_select_time_series(ds["jd"]).values, "jd")
    else:
        jd = _to_1d(_get_jd_from_time(sw_dn_da.coords["time"]), "jd")

    lat = np.broadcast_to(_to_1d(lat, "lat"), sw_dn.shape)
    lon = np.broadcast_to(_to_1d(lon, "lon"), sw_dn.shape)
    jd = np.broadcast_to(_to_1d(jd, "jd"), sw_dn.shape)

    # Build simple constant inputs to exercise the albedo pathway.
    inputs = dict(
        u=np.full_like(sw_dn, 5.0),
        zu=np.full_like(sw_dn, 10.0),
        t=np.full_like(sw_dn, 20.0),
        zt=np.full_like(sw_dn, 10.0),
        rh=np.full_like(sw_dn, 80.0),
        zq=np.full_like(sw_dn, 10.0),
        P=np.full_like(sw_dn, 1013.0),
        ts=np.full_like(sw_dn, 21.0),
        sw_dn=sw_dn,
        lw_dn=np.full_like(sw_dn, 400.0),
        lat=lat,
        lon=lon,
        jd=jd,
        zi=np.full_like(sw_dn, 600.0),
        rain=np.full_like(sw_dn, 0.0),
        Ss=np.full_like(sw_dn, 35.0),
    )

    A = np.asarray(coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=True))
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.shape[0] != sw_dn.size:
        raise SystemExit(
            f"Output length mismatch: sw_dn has {sw_dn.size} values but output has {A.shape[0]} rows."
        )
    alb = np.ravel(A[:, -4])

    alb_ref, _, _, _ = albedo_vector(sw_dn, jd, lon, lat, eorw="E")

    print(f"albedo stats: min={np.nanmin(alb):.3f} max={np.nanmax(alb):.3f} mean={np.nanmean(alb):.3f}")
    diff = alb - alb_ref
    print(f"albedo diff vs direct albedo_vector: max_abs={np.nanmax(np.abs(diff)):.3e}")

    if args.plot:
        import matplotlib.pyplot as plt

        times = sw_dn_da.coords["time"].values
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, alb, label="Payne1972 albedo")
        ax.set_ylabel("Albedo")
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
