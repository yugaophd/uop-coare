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


def _find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
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
    frac = (times.hour + times.minute / 60.0 + times.second / 3600.0) / 24.0
    return np.asarray(dayofyear + frac, dtype=float)


def _broadcast_to_target(target, arr, name):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0:
        raise SystemExit(f"{name} has no values.")
    if arr.size == 1:
        return np.full_like(target, arr[0], dtype=float)
    if arr.size == target.size:
        return arr.reshape(target.shape)
    try:
        return np.broadcast_to(arr, target.shape).astype(float, copy=False)
    except ValueError as exc:
        raise SystemExit(
            f"Cannot align '{name}' with sw_dn shape {target.shape}; got {arr.shape}."
        ) from exc


def _build_constant_inputs(sw_dn, lat, lon, jd):
    return dict(
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


def _build_synthetic_contradiction():
    # Build a synthetic daily shortwave cycle centered at 04:48 UTC. At lon=0
    # and lat=45 this sits in nighttime / dawn solar geometry, while lon+180
    # shifts local solar time by 12 hours into daytime / dusk. That makes the
    # shifted case strongly contradictory without using a flat sw_dn line.
    N = 72
    jd = 100.0 + np.arange(N, dtype=float) / N
    utc_hour = np.mod(jd, 1.0) * 24.0
    peak_hour = 4.8
    wrapped_hour_offset = ((utc_hour - peak_hour + 12.0) % 24.0) - 12.0

    # A narrow daily hump: strong enough to exceed 500 W/m^2 near the peak,
    # while keeping the baseline case below the error guard.
    sw_dn = 850.0 * np.exp(-0.5 * (wrapped_hour_offset / 0.6) ** 2)
    sw_dn[sw_dn < 5.0] = 0.0
    lat = np.full(N, 45.0)
    lon = np.zeros(N)
    times = np.arange(N)
    return sw_dn, lat, lon, jd, times


def _load_l4_inputs(path):
    import xarray as xr

    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")

    ds = xr.open_dataset(path)

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
    sw_dn = np.asarray(sw_dn_da.values, dtype=float).reshape(-1)

    lat_da = _find_coord(ds, ["lat", "latitude"])
    if lat_da is None:
        lat_da = _find_dataarray(ds, ["lat", "latitude"])

    lon_da = _find_coord(ds, ["lon", "longitude"])
    if lon_da is None:
        lon_da = _find_dataarray(ds, ["lon", "longitude"])

    if lat_da is None or lon_da is None:
        raise SystemExit("Latitude/longitude not found in dataset.")

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
            raise SystemExit("No jd variable or time coordinate to compute yearday.")
        jd = _get_jd_from_time(time_da).reshape(-1)

    lat = _broadcast_to_target(sw_dn, lat, "lat")
    lon = _broadcast_to_target(sw_dn, lon, "lon")
    jd = _broadcast_to_target(sw_dn, jd, "jd")

    return sw_dn, lat, lon, jd, time_da


def _window_slice(size, start, count):
    if start < 0 or start >= size:
        raise SystemExit(f"--start must be between 0 and {size - 1}.")
    stop = size if count is None else min(size, start + count)
    if stop <= start:
        raise SystemExit("Selected window is empty.")
    return slice(start, stop)


def _run_case(inputs):
    result = {
        "error": None,
        "A": None,
    }
    try:
        result["A"] = np.asarray(
            coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=True)
        )
    except Exception as exc:
        result["error"] = exc
    return result


def _format_error(err):
    if err is None:
        return "none"
    return f"{type(err).__name__}: {err}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Explore how shifting longitude, with or without a matching yearday shift, "
            "affects the Payne1972 albedo path in coare36vn_zrf_et."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="/mnt/d/tom_data/S-MODE/Wave_Glider/data/IOP1/L4/S_MODE_IOP1_Wavegliders_Carson_L4_20min_V2.nc",
        help="Path to an L4 netCDF file.",
    )
    parser.add_argument(
        "--lon-shift-deg",
        type=float,
        default=180.0,
        help="Longitude shift to apply to the perturbed case.",
    )
    parser.add_argument(
        "--jd-shift-days",
        type=float,
        default=0.0,
        help="Explicit yearday shift to apply. Default is 0, which keeps UTC time fixed.",
    )
    parser.add_argument(
        "--preserve-solar-time",
        action="store_true",
        help="Override --jd-shift-days and shift jd by lon_shift_deg / 360 to preserve local solar time.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First sample index to plot.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=288,
        help="Number of samples to include. Use 288 for about 4 days of 20 min data.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output image path. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--synthetic-failing-case",
        action="store_true",
        help="Use a deterministic synthetic dataset that produces a strong longitude-only contradiction.",
    )
    args = parser.parse_args()

    if args.synthetic_failing_case:
        sw_dn, lat, lon, jd, time_da = _build_synthetic_contradiction()
    else:
        sw_dn, lat, lon, jd, time_da = _load_l4_inputs(args.path)
    jd_shift_days = args.jd_shift_days
    if args.preserve_solar_time:
        jd_shift_days = args.lon_shift_deg / 360.0

    window = _window_slice(sw_dn.size, args.start, args.count)
    sw_dn = sw_dn[window]
    lat = lat[window]
    lon = lon[window]
    jd = jd[window]
    if time_da is None:
        times = None
    elif hasattr(time_da, "values"):
        times = time_da.values[window]
    else:
        times = np.asarray(time_da)[window]

    inputs_base = _build_constant_inputs(sw_dn, lat, lon, jd)
    lon_bad = lon + args.lon_shift_deg
    jd_bad = jd + jd_shift_days
    inputs_shift = _build_constant_inputs(sw_dn, lat, lon_bad, jd_bad)

    base = _run_case(inputs_base)
    shifted = _run_case(inputs_shift)

    alb_base, _, _, psi_base = albedo_vector(sw_dn, jd, lon, lat, eorw="E")
    alb_shift, _, _, psi_shift = albedo_vector(sw_dn, jd_bad, lon_bad, lat, eorw="E")

    if args.synthetic_failing_case:
        print("file: synthetic failing case")
    else:
        print(f"file: {args.path}")
    print(f"window: start={args.start} count={sw_dn.size}")
    print(f"baseline error: {_format_error(base['error'])}")
    print(
        "shifted inputs: "
        f"lon += {args.lon_shift_deg:.3f} deg, jd += {jd_shift_days:.3f} days"
    )
    if args.preserve_solar_time:
        print("shift mode: preserve solar time")
    else:
        print("shift mode: fixed UTC time")
    print(f"shifted error: {_format_error(shifted['error'])}")

    bad_high = np.isfinite(sw_dn) & np.isfinite(alb_shift) & (sw_dn > 500.0) & (alb_shift > 0.1)
    bad_low = (
        np.isfinite(sw_dn)
        & np.isfinite(alb_shift)
        & np.isfinite(psi_shift)
        & (psi_shift >= 2.0)
        & (sw_dn > 10.0)
        & (alb_shift < 0.01)
    )
    print(f"shifted high-sw/high-albedo count: {int(np.count_nonzero(bad_high))}")
    print(f"shifted daytime high-sw/near-zero-albedo count: {int(np.count_nonzero(bad_low))}")
    print(f"baseline albedo range: {np.nanmin(alb_base):.3f} to {np.nanmax(alb_base):.3f}")
    print(f"shifted albedo range: {np.nanmin(alb_shift):.3f} to {np.nanmax(alb_shift):.3f}")

    import matplotlib.pyplot as plt

    if times is None:
        x = np.arange(sw_dn.size)
        xlabel = "sample index"
    elif args.synthetic_failing_case:
        x = times
        xlabel = "synthetic sample index"
    else:
        x = times
        xlabel = "time"

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(x, sw_dn, color="tab:orange", label="sw_dn")
    axes[0].set_ylabel("W/m^2")
    axes[0].set_title("Downwelling Shortwave")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, alb_base, color="tab:blue", label="baseline albedo")
    axes[1].plot(x, alb_shift, color="tab:red", label="shifted albedo", alpha=0.8)
    axes[1].axhline(0.1, color="black", linestyle="--", linewidth=1.0, label="albedo = 0.1")
    axes[1].set_ylabel("albedo")
    axes[1].set_title("Payne1972 Albedo")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, psi_base, color="tab:green", label="baseline psi_sw")
    axes[2].plot(x, psi_shift, color="tab:purple", label="shifted psi_sw", alpha=0.8)
    axes[2].axhline(2.0, color="black", linestyle="--", linewidth=1.0, label="psi_sw = 2 deg")
    axes[2].set_ylabel("deg")
    axes[2].set_xlabel(xlabel)
    axes[2].set_title("Solar Elevation Angle")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved plot: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
