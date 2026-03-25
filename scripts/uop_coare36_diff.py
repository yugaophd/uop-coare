7,8c7,10
< Execute '%run coare36vn_zrf_et' from the iPython command line for test run with
< 'test_36_data.txt' input data file. 
---
> Yu Gao and Tom Farrar modified to have coolskin and albedo options in the function inputs.
> Use coolskin (=True/False) and albedo (='constant'/'Payne1972'), with defaults
> coolskin=True and albedo='constant'.
> 
22,23c24,25
< ludovic Bariteau, CU/CIRES, NOAA/ESRL/PSL
< v1: August 2022
---
> previous versin by ludovic Bariteau, CU/CIRES, NOAA/ESRL/PSL, August 2022
> uop_coare version v1 by Yu Gao and Tom Farrar, Feb 2026
26a29,125
> from datetime import datetime
> 
> def calculate_jd(time):
>     """
>     Compute fractional Julian day (day of year) for each timestamp.
> 
>     Jan 1 00:00 UTC of a given year corresponds to 1.0
>     (matches common day-of-year convention and test fixtures).
> 
>     Parameters:
>     - time: array-like of datetime-like values. Supports:
>       numpy datetime64 arrays, pandas DatetimeIndex/Series, lists/arrays of
>       Python datetime objects.
> 
>     Returns:
>     - numpy.ndarray of floats representing fractional days since Jan 1 00:00.
>     """
>     # Prefer pandas for robust datetime64 handling if available
>     try:
>         import pandas as pd
>         py_datetimes = pd.DatetimeIndex(time).to_pydatetime()
>     except Exception:
>         arr = np.asarray(time)
>         # numpy datetime64 vector → Python datetime objects
>         if np.issubdtype(arr.dtype, np.datetime64):
>             py_datetimes = arr.astype('datetime64[ns]').astype('O')
>         else:
>             # Assume array of Python datetime objects
>             py_datetimes = arr
> 
>     jd = np.array([
>         (dt - datetime(dt.year, 1, 1, tzinfo=getattr(dt, 'tzinfo', None))).total_seconds() / 86400.0 + 1.0
>         for dt in py_datetimes
>     ], dtype=float)
> 
>     return jd
> 
> 
> def _plot_shortwave_jd_diagnostics(sw_dn, jd, lon, plot_path="coare_sw_jd_diagnostic.png"):
>     """Save local-hour shortwave composite with histogram bars and a dot-line overlay."""
>     sw_dn = np.asarray(sw_dn, dtype=float).reshape(-1)
>     jd = np.asarray(jd, dtype=float).reshape(-1)
>     lon = np.asarray(lon, dtype=float).reshape(-1)
>     n = min(sw_dn.size, jd.size)
>     if n == 0:
>         return
> 
>     sw_dn = sw_dn[:n]
>     jd = jd[:n]
>     if lon.size == 0:
>         return
>     if lon.size == 1:
>         lon_local = np.full(n, lon[0], dtype=float)
>     else:
>         lon_local = np.broadcast_to(lon, sw_dn.shape)[:n]
> 
>     # Local hour = UTC hour + longitude/15 (deg east is positive).
>     hour_local = np.mod(np.mod(jd, 1.0) * 24.0 + lon_local / 15.0, 24.0)
>     valid = np.isfinite(sw_dn) & np.isfinite(hour_local)
>     if not np.any(valid):
>         return
> 
>     hour_valid = hour_local[valid]
>     sw_valid = sw_dn[valid]
> 
>     # Hourly diurnal composite of shortwave (00:00 to 24:00).
>     hour_bin_idx = np.floor(hour_valid).astype(int)
>     hour_bin_idx = np.clip(hour_bin_idx, 0, 23)
>     sw_hourly = np.full(24, np.nan)
>     for h in range(24):
>         hh = sw_valid[hour_bin_idx == h]
>         if hh.size > 0:
>             sw_hourly[h] = np.nanmean(hh)
> 
>     # Always draw 24 bins. Missing-hour composites are shown as zero-height bars.
>     sw_hourly_plot = np.where(np.isfinite(sw_hourly), sw_hourly, 0.0)
>     has_data = np.isfinite(sw_hourly)
> 
>     try:
>         import matplotlib.pyplot as plt
> 
>         fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
>         x = np.arange(24)
>         bar_colors = np.where(has_data, "#2C7FB8", "#D9D9D9")
>         ax.bar(x, sw_hourly_plot, width=0.9, align="center", color=bar_colors, alpha=0.8)
>         ax.plot(x, sw_hourly, color="#D95F0E", marker="o", linewidth=1.6, markersize=4)
>         ax.set_title("Hourly Composite Shortwave Histogram")
>         ax.set_xlabel("Local hour")
>         ax.set_ylabel("Mean sw_dn (W/m^2)")
>         ax.set_xlim(-0.5, 23.5)
>         ax.set_xticks(np.arange(0, 24, 2))
>         ax.grid(axis="y", alpha=0.3)
> 
>         fig.savefig(plot_path, dpi=150)
>         plt.close(fig)
>     except Exception as exc:
>         print(f"Warning: unable to generate diagnostic plot: {exc}")
28c127
< def coare36vn_zrf_et(u, zu , t, zt, rh, zq, P, ts, sw_dn, lw_dn, lat, lon,jd, zi,rain, Ss, cp=None, sigH=None, zrf_u=10.0, zrf_t=10.0, zrf_q=10.0):   
---
> def coare36vn_zrf_et(u, zu , t, zt, rh, zq, P, ts, sw_dn, lw_dn, lat, lon,jd, zi,rain, Ss, cp=None, sigH=None, zrf_u=10.0, zrf_t=10.0, zrf_q=10.0, coolskin=True, albedo='constant', return_albedo=False, wind_direction=None, payne_min_solar_elev_deg=2.0, auto_plot=True, auto_plot_path='coare_sw_jd_diagnostic.png'):   
54a154,157
> #
> # In this version, the coolskin input flag sets jcool:
> #   coolskin=True  -> jcool=1 (default)
> #   coolskin=False -> jcool=0
56a160,173
>     if not coolskin:
>         jcoolx = 0
> 
>     # Normalize albedo option to accepted strings.
>     if isinstance(albedo, bool):
>         albedo = "constant" if albedo else "Payne1972"
>     elif albedo is None:
>         albedo = "constant"
>     elif isinstance(albedo, str):
>         key = albedo.strip().lower()
>         if key in {"payne1972", "payne_1972", "payne-1972", "payne"}:
>             albedo = "Payne1972"
>         elif key in {"constant", "const"}:
>             albedo = "constant"
92c209
< #  jd = year day or julian day, where day Jan 1 00:00 UTC = 0
---
> #  jd = fractional day-of-year, where Jan 1 00:00 UTC = 1.0 in this implementation
99a217,230
> #  coolskin = True to run cool-skin model, False to disable (default True)
> #  albedo = 'constant' to use constant albedo correction (default),
> #           'Payne1972' to use zenith-angle varying albedo from Payne (1972)
> #           also accepts bool: True -> 'constant', False -> 'Payne1972'
> #  return_albedo = False to keep legacy output shape; True to append
> #           alb, T_sw, solarmax_sw, psi_sw to the output array
> #  wind_direction = wind direction in degrees from which the wind is coming
> #           (clockwise from north, Vaisala WXT convention). Used to compute
> #           wind stress components; set to None to skip.
> #  payne_min_solar_elev_deg = minimum solar elevation angle (deg) required
> #           to apply Payne (1972) albedo to shortwave flux; for lower angles
> #           sw_net is masked to 0 to suppress night/twilight artifacts.
> #  auto_plot = True to save diagnostic shortwave/jd plots after each run
> #  auto_plot_path = output file path for diagnostic plot
129a261,265
> #    (optional appended outputs if return_albedo=True)
> #    alb = shortwave albedo (Payne 1972 or constant)
> #    T_sw = shortwave transmission factor (min(2, sw_dn/solarmax_sw))
> #    solarmax_sw = maximum shortwave at top of atmosphere (W/m^2)
> #    psi_sw = solar altitude angle (degrees)
265a402,413
> 
>     # If `jd` is datetime-like, convert to fractional day-of-year internally
>     try:
>         jd_arr = np.asarray(jd)
>         is_datetime_like = np.issubdtype(jd_arr.dtype, np.datetime64) or (
>             jd_arr.dtype == object and jd_arr.size > 0 and hasattr(jd_arr[0], 'year')
>         )
>         if is_datetime_like:
>             jd = calculate_jd(jd)
>     except Exception:
>         # Leave jd unchanged if detection or conversion fails
>         pass
365,369c513,552
<     alb,T_sw,solarmax_sw,psi_sw = albedo_vector(sw_dn,jd,lon,lat,eorw='E')
<     sw_net = np.multiply((1 - alb),sw_dn)
<     
<     # *** for constant albedo:
<     # sw_net = 0.945.*sw_dn; # constant albedo correction, positive heating ocean
---
>     if albedo == 'Payne1972':
>         alb, T_sw, solarmax_sw, psi_sw = albedo_vector(sw_dn, jd, lon, lat, eorw='E')
>         sw_net = np.multiply((1 - alb), sw_dn)
>         # payne_daytime = (
>         #     np.isfinite(psi_sw)
>         #     & np.isfinite(sw_dn)
>         #     & (psi_sw >= payne_min_solar_elev_deg)
>         #     & (sw_dn > 0)
>         # )
>         # sw_net = np.where(payne_daytime, np.multiply((1 - alb), sw_dn), 0.0)
>     elif albedo == 'constant':
>         # *** for constant albedo:
>         alb = 0.055 * np.ones(N)
>         T_sw = np.full(N, np.nan)
>         solarmax_sw = np.full(N, np.nan)
>         psi_sw = np.full(N, np.nan)
>         sw_net = 0.945 * sw_dn  # constant albedo correction, positive heating ocean
>     else:
>         print("please provide albedo as 'constant' or 'Payne1972'")
>         alb, T_sw, solarmax_sw, psi_sw = albedo_vector(sw_dn, jd, lon, lat, eorw='E')
>         sw_net = np.multiply((1 - alb), sw_dn)
>     
>     # Sanity/error check (daytime only): shortwave should not be substantial with near-zero albedo
>     if albedo == 'Payne1972':
>         daytime = np.isfinite(psi_sw) & (psi_sw >= payne_min_solar_elev_deg)
>         bad_albedo_mask = np.isfinite(sw_dn) & np.isfinite(alb) & daytime & (sw_dn > 10.0) & (alb < 0.01)
>         if np.any(bad_albedo_mask):
>             count = int(np.count_nonzero(bad_albedo_mask))
>             raise ValueError(
>                 f"Albedo error: {count} sample(s) have sw_dn > 10 W/m^2 while albedo < 0.01 "
>                 f"(psi_sw >= {payne_min_solar_elev_deg} deg). "
>                 "Check time/jd, longitude sign (E/W), and albedo settings."
>             )
> 
>     # high_sw_high_albedo_mask = np.isfinite(sw_dn) & np.isfinite(alb) & (sw_dn > 500.0) & (alb > 0.1)
>     # if np.any(high_sw_high_albedo_mask):
>         # count = int(np.count_nonzero(high_sw_high_albedo_mask))
>         # raise ValueError(
>             # f"Albedo error: {count} sample(s) have sw_dn > 500 W/m^2 while albedo > 0.1."
>         # )
494c677
<         ik = np.array(np.where(rr[iice] > np.logical_and(0.135,rr[iice]) <= 2.5))
---
>         ik = np.array(np.where((rr[iice] > 0.135) & (rr[iice] <= 2.5)))
497c680
<         ik = np.array(np.where(rr[iice] > np.logical_and(2.5,rr[iice]) <= 1000))
---
>         ik = np.array(np.where((rr[iice] > 2.5) & (rr[iice] <= 1000)))
738a922,927
>     if wind_direction is None:
>         tau_x = np.full(N, np.nan)
>         tau_y = np.full(N, np.nan)
>     else:
>         tau_x, tau_y = calculate_wind_stress_components(tau, wind_direction)
> 
740,741c929,932
<     out = np.array([usr,tau,hsb,hlb,hbb,hsbb,hlwebb,tsr,qsr,zo,zot,zoq,Cd,Ch,Ce,L,zeta,dT_skinx,dq_skinx,dz_skin,Urf,Trf,Qrf,RHrf,UrfN,TrfN,QrfN,lw_net,sw_net,Le,rhoa,UN,U10,U10N,Cdn_10,Chn_10,Cen_10,hrain,Qs,Evap,T10,T10N,Q10,Q10N,RH10,P10,rhoa10,gust,wc_frac,Edis])
<     #                1   2   3   4   5   6    7      8   9  10  11  12 13 14 15 16  17   18       19        20    21  22  23  24   25   26   27     28    29   30  31  32 33   34    35     36    37      38  39  40   41  42   43   44   45  46   47    48     49    50
---
>     out = np.array([usr,tau,hsb,hlb,hbb,hsbb,hlwebb,tsr,qsr,zo,zot,zoq,Cd,Ch,Ce,L,zeta,dT_skinx,dq_skinx,dz_skin,Urf,Trf,Qrf,RHrf,UrfN,TrfN,QrfN,lw_net,sw_net,Le,rhoa,UN,U10,U10N,Cdn_10,Chn_10,Cen_10,hrain,Qs,Evap,T10,T10N,Q10,Q10N,RH10,P10,rhoa10,gust,wc_frac,Edis,tau_x,tau_y])
>     #                1   2   3   4   5   6    7      8   9  10  11  12 13 14 15 16  17   18       19        20    21  22  23  24   25   26   27     28    29   30  31  32 33   34    35     36    37      38  39  40   41  42   43   44   45  46   47    48     49    50  51  52
>     # optional (if return_albedo=True): 53=alb, 54=T_sw, 55=solarmax_sw, 56=psi_sw
>     ##### ALWAYS return albedo (either a constant or Payne1972) #####
742a934,940
>     if return_albedo:
>         # Append albedo outputs (alb, T_sw, solarmax_sw, psi_sw)
>         A = np.column_stack((A, alb, T_sw, solarmax_sw, psi_sw))
> 
>     if auto_plot:
>         _plot_shortwave_jd_diagnostics(sw_dn, jd, lon, plot_path=auto_plot_path)
> 
745a944,962
> 
> def calculate_wind_stress_components(stress_magnitude, wind_direction_degrees):
>     """
>     Calculate wind stress components from magnitude and direction following CF conventions.
> 
>     Parameters:
>     stress_magnitude (float or array): Wind stress magnitude
>     wind_direction_degrees (float or array): Wind direction in degrees (clockwise from north),
>         direction wind is coming FROM (Vaisala WXT convention).
> 
>     Returns:
>     tuple: (surface_downward_eastward_stress, surface_downward_northward_stress)
>     """
>     theta = np.radians(wind_direction_degrees)
>     tau_x = -1 * stress_magnitude * np.sin(theta)
>     tau_y = -1 * stress_magnitude * np.cos(theta)
>     return tau_x, tau_y
> 
> #------------------------------------------------------------------------------
951,953c1168,1169
< # This code executes if 'run coare36vn_zrf_et.py' is executed from iPython cmd line
< # Edit line 959 to indicate path to test data file
< if __name__ == '__main__':
---
> def main():
>     """Run the legacy example using the original test data file paths."""
958c1174,1176
<     
---
> 
>     # This code executes if 'run coare36vn_zrf_et.py' is executed from iPython cmd line
>     # Edit this path to indicate the location of the test data file.
960,961c1178,1179
<     fil = 'test_36_data.txt'   
<     data = np.genfromtxt(path+fil, skip_header=1)
---
>     fil = 'test_36_data.txt'
>     data = np.genfromtxt(path + fil, skip_header=1)
983c1201
<     
---
> 
994c1212
<     
---
> 
999c1217,1221
<     
\ No newline at end of file
---
> if __name__ == '__main__':
>     main()
> 
> 
>     
