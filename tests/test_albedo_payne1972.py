import os
import pytest

import numpy as np

from uop_coare.coare36vn_zrf_et import albedo_vector, coare36vn_zrf_et


def _build_constant_inputs(sw_dn, lat, lon, jd):
    # Build simple constant inputs with the same shape as sw_dn.
    u = np.full_like(sw_dn, 5.0)
    zu = np.full_like(sw_dn, 10.0)
    t = np.full_like(sw_dn, 20.0)
    zt = np.full_like(sw_dn, 10.0)
    rh = np.full_like(sw_dn, 80.0)
    zq = np.full_like(sw_dn, 10.0)
    P = np.full_like(sw_dn, 1013.0)
    ts = np.full_like(sw_dn, 21.0)
    lw_dn = np.full_like(sw_dn, 400.0)
    zi = np.full_like(sw_dn, 600.0)
    rain = np.full_like(sw_dn, 0.0)
    Ss = np.full_like(sw_dn, 35.0)

    return dict(
        u=u,
        zu=zu,
        t=t,
        zt=zt,
        rh=rh,
        zq=zq,
        P=P,
        ts=ts,
        sw_dn=sw_dn,
        lw_dn=lw_dn,
        lat=lat,
        lon=lon,
        jd=jd,
        zi=zi,
        rain=rain,
        Ss=Ss,
    )


def test_return_albedo_appends(l4_sample_arrays):
    sw_dn, lat, lon, jd, _time = l4_sample_arrays
    inputs = _build_constant_inputs(sw_dn, lat, lon, jd)

    base = coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=False)
    with_alb = coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=True)

    assert with_alb.shape[1] == base.shape[1] + 4


def test_payne_albedo_consistency(l4_sample_arrays):
    sw_dn, lat, lon, jd, time_da = l4_sample_arrays
    inputs = _build_constant_inputs(sw_dn, lat, lon, jd)

    A = coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=True)
    alb_ref, T_sw_ref, solarmax_ref, psi_ref = albedo_vector(sw_dn, jd, lon, lat, eorw="E")

    alb_out = A[:, -4]
    T_sw_out = A[:, -3]
    solarmax_out = A[:, -2]
    psi_out = A[:, -1]

    valid = np.isfinite(sw_dn)
    assert np.allclose(alb_out[valid], alb_ref[valid], rtol=1e-6, atol=1e-6)
    assert np.allclose(T_sw_out[valid], T_sw_ref[valid], rtol=1e-6, atol=1e-6)
    assert np.allclose(solarmax_out[valid], solarmax_ref[valid], rtol=1e-6, atol=1e-6)
    assert np.allclose(psi_out[valid], psi_ref[valid], rtol=1e-6, atol=1e-6)

    sw_net_out = A[:, 28]
    assert np.allclose(sw_net_out[valid], (1 - alb_ref[valid]) * sw_dn[valid], rtol=1e-6, atol=1e-6)

    # Sanity check: albedo should be near minimum around local noon.
    if time_da is not None:
        import pandas as pd

        times = pd.to_datetime(time_da.values)
        # Convert UTC -> local solar time using longitude (east positive).
        # Normalize lon to [-180, 180] to handle 0..360 datasets.
        lon_scalar = float(np.nanmedian(lon))
        lon_norm = ((lon_scalar + 180.0) % 360.0) - 180.0
        lon_hours = lon_norm / 15.0
        local_times = times + pd.to_timedelta(lon_hours, unit="h")
        local_dates = local_times.date
        local_hours = local_times.hour + local_times.minute / 60.0 + local_times.second / 3600.0

        daylight = (sw_dn > 50) & np.isfinite(alb_out) & (alb_out > 0)
        checked_days = 0
        for day in np.unique(local_dates):
            idx = (local_dates == day) & daylight
            if np.count_nonzero(idx) < 6:
                continue
            checked_days += 1
            # Use the time of maximum shortwave as a proxy for local noon.
            noon_k = np.argmax(sw_dn[idx])
            alb_at_noon = alb_out[idx][noon_k]
            alb_min = np.min(alb_out[idx])
            # Albedo should be near its daily minimum around local noon.
            assert alb_at_noon <= alb_min + 0.05
        if checked_days == 0:
            # Not enough daylight samples to evaluate local-noon behavior.
            pass

    if os.environ.get("PLOT_TESTS", "1") == "1":
        import matplotlib.pyplot as plt

        os.makedirs("tests/artifacts", exist_ok=True)
        if time_da is not None:
            import pandas as pd

            times = pd.to_datetime(time_da.values)
            lon_scalar = float(np.nanmedian(lon))
            lon_norm = ((lon_scalar + 180.0) % 360.0) - 180.0
            lon_hours = lon_norm / 15.0
            x = times + pd.to_timedelta(lon_hours, unit="h")
            x_label = "local time"
        else:
            x = np.arange(sw_dn.size)
            x_label = "sample index"

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, alb_out, label="alb (from coare36vn_zrf_et)")
        ax.plot(x, alb_ref, label="alb (albedo_vector)", alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel("albedo")
        ax.set_title("Payne1972 Albedo Consistency")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("tests/artifacts/albedo_payne1972.png", dpi=150)
        plt.close(fig)

        if time_da is not None:
            # Short window for diurnal cycle (first 48 hours, local time).
            t0 = x[0]
            t1 = t0 + pd.Timedelta(hours=48)
            mask = (x >= t0) & (x <= t1)
            if np.any(mask):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x[mask], alb_out[mask], label="alb (from coare36vn_zrf_et)")
                ax.plot(x[mask], alb_ref[mask], label="alb (albedo_vector)", alpha=0.7)
                ax.set_xlabel("local time")
                ax.set_ylabel("albedo")
                ax.set_title("Payne1972 Albedo (First 48h, Local Time)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig("tests/artifacts/albedo_payne1972_first48h.png", dpi=150)
                plt.close(fig)

                # Compare L3 input sw_dn vs net solar for Payne1972 and constant albedo.
                A_const = coare36vn_zrf_et(**inputs, albedo="constant", return_albedo=False)
                sw_net_payne = A[:, 28]
                sw_net_const = A_const[:, 28]

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x[mask], sw_dn[mask], label="L3 sw_dn (input)")
                ax.plot(x[mask], sw_net_payne[mask], label="sw_net (Payne1972)")
                ax.plot(x[mask], sw_net_const[mask], label="sw_net (constant albedo)")
                ax.set_xlabel("local time")
                ax.set_ylabel("W/m^2")
                ax.set_title("Net Solar vs Input (First 48h, Local Time)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig("tests/artifacts/sw_net_compare_first48h.png", dpi=150)
                plt.close(fig)


def test_constant_albedo_consistency(l4_sample_arrays):
    sw_dn, lat, lon, jd, _time = l4_sample_arrays
    inputs = _build_constant_inputs(sw_dn, lat, lon, jd)

    A = coare36vn_zrf_et(**inputs, albedo="constant", return_albedo=True)

    alb_out = A[:, -4]
    T_sw_out = A[:, -3]
    solarmax_out = A[:, -2]
    psi_out = A[:, -1]

    valid = np.isfinite(sw_dn)
    assert np.allclose(alb_out[valid], 0.055, rtol=0.0, atol=0.0)
    assert np.all(np.isnan(T_sw_out[valid]))
    assert np.all(np.isnan(solarmax_out[valid]))
    assert np.all(np.isnan(psi_out[valid]))

    sw_net_out = A[:, 28]
    assert np.allclose(sw_net_out[valid], 0.945 * sw_dn[valid], rtol=1e-6, atol=1e-6)

    if os.environ.get("PLOT_TESTS", "1") == "1":
        import matplotlib.pyplot as plt

        os.makedirs("tests/artifacts", exist_ok=True)
        if _time is not None:
            import pandas as pd

            times = pd.to_datetime(_time.values)
            lon_scalar = float(np.nanmedian(lon))
            lon_norm = ((lon_scalar + 180.0) % 360.0) - 180.0
            lon_hours = lon_norm / 15.0
            x = times + pd.to_timedelta(lon_hours, unit="h")
            x_label = "local time"
        else:
            x = np.arange(sw_dn.size)
            x_label = "sample index"

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, sw_dn, label="sw_dn")
        ax.plot(x, sw_net_out, label="sw_net (constant albedo)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("W/m^2")
        ax.set_title("Constant Albedo: sw_dn vs sw_net")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("tests/artifacts/sw_net_constant_albedo.png", dpi=150)
        plt.close(fig)


def test_datetime_jd_input(l4_sample_arrays):
    sw_dn, lat, lon, jd, time_da = l4_sample_arrays
    if time_da is None:
        return

    inputs_num = _build_constant_inputs(sw_dn, lat, lon, jd)
    inputs_dt = dict(inputs_num)
    inputs_dt["jd"] = time_da.values

    A_num = coare36vn_zrf_et(**inputs_num, albedo="Payne1972", return_albedo=True)
    A_dt = coare36vn_zrf_et(**inputs_dt, albedo="Payne1972", return_albedo=True)

    alb_num = A_num[:, -4]
    alb_dt = A_dt[:, -4]

    valid = np.isfinite(sw_dn)
    assert np.allclose(alb_num[valid], alb_dt[valid], rtol=1e-6, atol=1e-6)


def test_shortwave_time_offset_misalignment(l4_sample_arrays):
    sw_dn, lat, lon, jd, time_da = l4_sample_arrays
    if time_da is None:
        pytest = __import__("pytest")
        pytest.skip("Time coordinate required to apply a 6-hour offset.")

    import pandas as pd
    times = pd.to_datetime(time_da.values)
    if times.size < 2:
        pytest = __import__("pytest")
        pytest.skip("Insufficient time samples to compute sampling interval.")

    dt_hours = np.median(np.diff(times.values).astype("timedelta64[s]").astype(float) / 3600.0)
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        pytest = __import__("pytest")
        pytest.skip("Invalid sampling interval; cannot compute 6-hour offset.")

    samples_per_hour = int(round(1.0 / dt_hours))
    offset_n = int(round(6 * samples_per_hour))
    window_n = int(round(60 * samples_per_hour))

    if offset_n >= sw_dn.size:
        pytest = __import__("pytest")
        pytest.skip("Dataset too short to apply 6-hour offset.")

    # Ensure we stay within bounds when slicing with the offset.
    max_len = sw_dn.size - offset_n
    N = min(window_n + 1, max_len)
    if N < 10:
        pytest = __import__("pytest")
        pytest.skip("Not enough samples in 0–60h window after applying offset.")

    # Hour 0–60: use original lat/lon/jd from start, but sw_dn from +6h.
    sw_dn_offset = sw_dn[offset_n : offset_n + N]
    lat_slice = lat[:N]
    lon_slice = lon[:N]
    jd_slice = jd[:N]

    inputs = _build_constant_inputs(sw_dn_offset, lat_slice, lon_slice, jd_slice)

    A = coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=True)

    alb_out = A[:, -4]
    sw_net_out = A[:, 28]

    valid = np.isfinite(sw_dn_offset) & np.isfinite(alb_out)
    assert np.allclose(
        sw_net_out[valid], (1.0 - alb_out[valid]) * sw_dn_offset[valid], rtol=1e-6, atol=1e-6
    )

    # The intentional time offset should produce cases where albedo is substantial
    # while shortwave is very small. Use dataset-relative thresholds for robustness.
    sw_low_thresh = np.nanpercentile(sw_dn_offset[valid], 25)
    alb_high_thresh = np.nanpercentile(alb_out[valid], 75)
    mismatch = valid & (sw_dn_offset <= sw_low_thresh) & (alb_out >= alb_high_thresh)

    # If percentiles degenerate (e.g., constant arrays), fall back to absolute thresholds.
    if np.count_nonzero(mismatch) == 0:
        mismatch = valid & (sw_dn_offset < 100.0) & (alb_out > 0.10)

    assert np.count_nonzero(mismatch) > 0

    if os.environ.get("PLOT_TESTS", "1") == "1":
        import matplotlib.pyplot as plt

        os.makedirs("tests/artifacts", exist_ok=True)

        # Plot against local time for clarity.
        lon_scalar = float(np.nanmedian(lon_slice))
        lon_norm = ((lon_scalar + 180.0) % 360.0) - 180.0
        lon_hours = lon_norm / 15.0
        x = times[:N] + pd.to_timedelta(lon_hours, unit="h")

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(x, alb_out, label="albedo", color="tab:blue")
        ax1.set_ylabel("albedo", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(x, sw_dn_offset, label="sw_dn (offset +6h)", color="tab:red", alpha=0.7)
        ax2.set_ylabel("W/m^2", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax1.set_title("Albedo vs Shortwave (0–60h window, sw offset +6h)")
        ax1.set_xlabel("local time")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("tests/artifacts/albedo_vs_sw_offset_0_60h.png", dpi=150)
        plt.close(fig)


def test_albedo_stop_triggers_with_bad_albedo(monkeypatch, l4_sample_arrays):
    # Verify the sanity/error check raises when daytime has substantial shortwave
    # but albedo is near-zero (contrived via monkeypatch).
    sw_dn, lat, lon, jd, _time = l4_sample_arrays
    N = sw_dn.size

    def fake_albedo_vector(sw_dn_arg, jd_arg, lon_arg, lat_arg, eorw="E"):
        # Positive solar altitude for all samples (daytime)
        psi_sw = np.full(N, 30.0)
        # Force near-zero albedo when shortwave is substantial
        alb = np.where(sw_dn > 100.0, 0.0, 0.05).astype(float)
        # Placeholder arrays for returned extras
        T_sw = np.full(N, np.nan)
        solarmax_sw = np.full(N, np.nan)
        return alb, T_sw, solarmax_sw, psi_sw

    # Patch the function used inside coare36 to produce the bad condition.
    monkeypatch.setattr("uop_coare.coare36vn_zrf_et.albedo_vector", fake_albedo_vector)

    inputs = _build_constant_inputs(sw_dn, lat, lon, jd)

    with pytest.raises(ValueError, match="Albedo error:"):
        coare36vn_zrf_et(**inputs, albedo="Payne1972", return_albedo=True)
