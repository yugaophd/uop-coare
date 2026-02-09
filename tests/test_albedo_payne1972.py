import os

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
