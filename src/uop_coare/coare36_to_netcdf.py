from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import xarray as xr

from .coare36_variables_config import (
    DEFAULT_REFERENCE_AIR_TEMPERATURE_HEIGHT_M,
    DEFAULT_REFERENCE_HUMIDITY_HEIGHT_M,
    DEFAULT_REFERENCE_WIND_HEIGHT_M,
    get_variables_info,
)
from .coare36vn_zrf_et import coare36vn_zrf_et


try:
    UOP_COARE_VERSION = version("uop-coare")
except PackageNotFoundError:
    UOP_COARE_VERSION = "unknown"

def _as_numpy_1d(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value.values if hasattr(value, "values") else value)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D, got shape {arr.shape}")
    return arr


def _coerce_coord(coord: Optional[Iterable[Any]], n: int, dim_name: str) -> np.ndarray:
    if coord is None:
        return np.arange(n, dtype=int)
    arr = np.asarray(coord)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), n)
    if arr.ndim != 1:
        raise ValueError(f"{dim_name} coordinate must be 1D, got shape {arr.shape}")
    if arr.size != n:
        raise ValueError(f"{dim_name} coordinate has size {arr.size}, expected {n}")
    return arr


def _as_float_attr(value: Any, name: str) -> float:
    arr = np.asarray(value.values if hasattr(value, "values") else value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 0:
        raise ValueError(f"{name} is empty; expected scalar or non-empty array")
    return float(np.nanmean(arr))


def _broadcast_input_to_output(value: Any, n_rows: int, name: str) -> np.ndarray:
    arr = _as_numpy_1d(value, name)
    if arr.size == 1:
        return np.full(n_rows, arr[0])
    if arr.size != n_rows:
        raise ValueError(f"{name} has size {arr.size}, expected 1 or {n_rows}")
    return arr.copy()


def _resolve_instrument_name(
    source_dataset: xr.Dataset,
    instrument_name: str,
    role: str,
) -> str:
    if instrument_name in source_dataset.variables:
        return instrument_name

    matches = sorted(name for name in source_dataset.variables if name.startswith(instrument_name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        match_list = ", ".join(matches)
        raise ValueError(
            f"Ambiguous instrument_config for role '{role}': '{instrument_name}' matches "
            f"multiple source variables ({match_list}). Pass the exact variable name."
        )
    raise ValueError(
        f"Instrument '{instrument_name}' for role '{role}' was not found in source_dataset "
        "and did not match any variable-name prefix."
    )


def _resolve_instrument_config(
    source_dataset: Optional[xr.Dataset],
    instrument_config: Optional[Dict[str, str]],
) -> Dict[str, str]:
    if instrument_config is None:
        return {}
    if source_dataset is None:
        return {role: str(inst_name) for role, inst_name in instrument_config.items()}
    return {
        role: _resolve_instrument_name(source_dataset, str(inst_name), role)
        for role, inst_name in instrument_config.items()
    }


def _default_l3_input_metadata() -> Dict[str, Dict[str, str]]:
    return {
        "L3_wind_speed": {
            "units": "m s-1",
            "long_name": "L3 wind speed used by COARE",
            "description": "Wind speed passed to uop_coare.coare36_to_netcdf",
        },
        "L3_wind_height": {
            "units": "m",
            "long_name": "L3 wind measurement height used by COARE",
            "description": "Wind measurement height passed to uop_coare.coare36_to_netcdf",
        },
        "L3_air_temperature": {
            "units": "degC",
            "long_name": "L3 air temperature used by COARE",
            "description": "Air temperature passed to uop_coare.coare36_to_netcdf",
        },
        "L3_air_temperature_height": {
            "units": "m",
            "long_name": "L3 air temperature measurement height used by COARE",
            "description": "Air temperature measurement height passed to uop_coare.coare36_to_netcdf",
        },
        "L3_relative_humidity": {
            "units": "%",
            "long_name": "L3 relative humidity used by COARE",
            "description": "Relative humidity passed to uop_coare.coare36_to_netcdf",
        },
        "L3_relative_humidity_height": {
            "units": "m",
            "long_name": "L3 humidity measurement height used by COARE",
            "description": "Humidity measurement height passed to uop_coare.coare36_to_netcdf",
        },
        "L3_air_pressure": {
            "units": "hPa",
            "long_name": "L3 atmospheric pressure used by COARE",
            "description": "Atmospheric pressure passed to uop_coare.coare36_to_netcdf",
        },
        "L3_sea_surface_temperature": {
            "units": "degC",
            "long_name": "L3 sea surface temperature used by COARE",
            "description": "Sea surface temperature passed to uop_coare.coare36_to_netcdf",
        },
        "L3_downwelling_shortwave_flux": {
            "units": "W m-2",
            "long_name": "L3 downwelling shortwave flux used by COARE",
            "description": "Downwelling shortwave radiation passed to uop_coare.coare36_to_netcdf",
        },
        "L3_downwelling_longwave_flux": {
            "units": "W m-2",
            "long_name": "L3 downwelling longwave flux used by COARE",
            "description": "Downwelling longwave radiation passed to uop_coare.coare36_to_netcdf",
        },
        "L3_latitude": {
            "units": "degrees_north",
            "long_name": "L3 latitude used by COARE",
            "description": "Latitude passed to uop_coare.coare36_to_netcdf",
        },
        "L3_longitude": {
            "units": "degrees_east",
            "long_name": "L3 longitude used by COARE",
            "description": "Longitude passed to uop_coare.coare36_to_netcdf",
        },
        "L3_julian_day": {
            "units": "day",
            "long_name": "L3 fractional Julian day used by COARE",
            "description": "Fractional Julian day passed to uop_coare.coare36_to_netcdf",
        },
        "L3_boundary_layer_height": {
            "units": "m",
            "long_name": "L3 boundary layer height used by COARE",
            "description": "Boundary layer height passed to uop_coare.coare36_to_netcdf",
        },
        "L3_rain_rate": {
            "units": "mm hr-1",
            "long_name": "L3 rain rate used by COARE",
            "description": "Rain rate passed to uop_coare.coare36_to_netcdf",
        },
        "L3_sea_surface_salinity": {
            "units": "PSU",
            "long_name": "L3 sea surface salinity used by COARE",
            "description": "Sea surface salinity passed to uop_coare.coare36_to_netcdf",
        },
        "L3_wave_phase_speed": {
            "units": "m s-1",
            "long_name": "L3 wave phase speed used by COARE",
            "description": "Wave phase speed passed to uop_coare.coare36_to_netcdf",
        },
        "L3_significant_wave_height": {
            "units": "m",
            "long_name": "L3 significant wave height used by COARE",
            "description": "Significant wave height passed to uop_coare.coare36_to_netcdf",
        },
        "L3_wind_direction": {
            "units": "degree",
            "long_name": "L3 wind direction used by COARE",
            "description": "Wind direction passed to uop_coare.coare36_to_netcdf",
        },
    }


def _l3_input_specs(
    *,
    u: Any,
    zu: Any,
    t: Any,
    zt: Any,
    rh: Any,
    zq: Any,
    P: Any,
    ts: Any,
    sw_dn: Any,
    lw_dn: Any,
    lat: Any,
    lon: Any,
    jd: Any,
    zi: Any,
    rain: Any,
    Ss: Any,
    cp: Any = None,
    sigH: Any = None,
    wind_direction: Any = None,
) -> Dict[str, tuple[Any, str]]:
    specs = {
        "L3_wind_speed": (u, "wind_speed"),
        "L3_wind_height": (zu, "wind_height"),
        "L3_air_temperature": (t, "air_temperature"),
        "L3_air_temperature_height": (zt, "air_temperature_height"),
        "L3_relative_humidity": (rh, "relative_humidity"),
        "L3_relative_humidity_height": (zq, "relative_humidity_height"),
        "L3_air_pressure": (P, "air_pressure"),
        "L3_sea_surface_temperature": (ts, "sea_surface_temperature"),
        "L3_downwelling_shortwave_flux": (sw_dn, "downwelling_shortwave_flux"),
        "L3_downwelling_longwave_flux": (lw_dn, "downwelling_longwave_flux"),
        "L3_latitude": (lat, "latitude"),
        "L3_longitude": (lon, "longitude"),
        "L3_julian_day": (jd, "julian_day"),
        "L3_boundary_layer_height": (zi, "boundary_layer_height"),
        "L3_rain_rate": (rain, "rain_rate"),
        "L3_sea_surface_salinity": (Ss, "sea_surface_salinity"),
    }
    if cp is not None:
        specs["L3_wave_phase_speed"] = (cp, "wave_phase_speed")
    if sigH is not None:
        specs["L3_significant_wave_height"] = (sigH, "significant_wave_height")
    if wind_direction is not None:
        specs["L3_wind_direction"] = (wind_direction, "wind_direction")
    return specs


def _build_l3_input_dataarray(
    values: Any,
    n_rows: int,
    output_dim: str,
    *,
    source_var: Optional[xr.DataArray] = None,
    fallback_attrs: Optional[Dict[str, Any]] = None,
) -> xr.DataArray:
    data = _broadcast_input_to_output(values, n_rows, output_dim)
    attrs = dict(source_var.attrs) if source_var is not None else dict(fallback_attrs or {})
    return xr.DataArray(data, dims=(output_dim,), attrs=_sanitize_netcdf_attrs(attrs))


def _add_instrument_variables(
    ds_out: xr.Dataset,
    source_dataset: Optional[xr.Dataset],
    instrument_prefix: str | Iterable[str],
) -> None:
    if source_dataset is None:
        return
    prefixes = _normalize_prefixes(instrument_prefix)
    for var_name in source_dataset.variables:
        if var_name in source_dataset.dims:
            continue
        if var_name.startswith(prefixes):
            ds_out[var_name] = source_dataset[var_name].copy(deep=True)


def _normalize_prefixes(prefixes: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(prefixes, str):
        return (prefixes,)
    return tuple(prefix for prefix in prefixes if isinstance(prefix, str) and prefix)


def _build_l3_dataarray(
    values: Any,
    n_rows: int,
    output_dim: str,
    *,
    source_var: Optional[xr.DataArray] = None,
) -> Optional[xr.DataArray]:
    data = np.asarray(values.values if hasattr(values, "values") else values)
    if data.ndim == 0:
        data = np.full(n_rows, data.item())
    if data.ndim != 1 or data.size != n_rows:
        return None

    attrs = dict(source_var.attrs) if source_var is not None else {}
    return xr.DataArray(data, dims=(output_dim,), attrs=attrs)


def _l3_output_name(var_name: str) -> str:
    return var_name if var_name.startswith("L3_") else f"L3_{var_name}"


def _copy_prefixed_l3_variables(
    ds_out: xr.Dataset,
    source_dataset: Optional[xr.Dataset],
    n_rows: int,
    output_dim: str,
    instrument_prefix: str | Iterable[str],
    *,
    skip_outputs: Optional[set[str]] = None,
) -> None:
    if source_dataset is None:
        return

    skip_outputs = skip_outputs or set()
    prefixes = _normalize_prefixes(instrument_prefix)
    for var_name, source_var in source_dataset.data_vars.items():
        if var_name.startswith(prefixes):
            continue

        output_name = _l3_output_name(var_name)
        if output_name in skip_outputs:
            continue
        data = np.asarray(source_var.values)
        attrs = dict(source_var.attrs)

        if data.ndim == 0:
            ds_out[output_name] = xr.DataArray(data.item(), attrs=attrs)
            continue
        if data.ndim == 1 and data.size == n_rows:
            ds_out[output_name] = xr.DataArray(data.copy(), dims=(output_dim,), attrs=attrs)


def _sanitize_netcdf_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            clean[key] = int(value)
            continue
        clean[key] = value
    return clean


def _add_coare_height_variables(
    ds_out: xr.Dataset,
    *,
    zu: float,
    zt: float,
    zq: float,
    zrf_u: float,
    zrf_t: float,
    zrf_q: float,
) -> None:
    """Add scalar COARE observation and reference heights as dataset variables."""
    ds_out["coare_wind_observation_height"] = xr.DataArray(
        float(zu),
        attrs={
            "units": "m",
            "long_name": "COARE wind observation height",
            "description": "Wind measurement height passed to COARE (zu)",
        },
    )
    ds_out["coare_air_temperature_observation_height"] = xr.DataArray(
        float(zt),
        attrs={
            "units": "m",
            "long_name": "COARE air temperature observation height",
            "description": "Air temperature measurement height passed to COARE (zt)",
        },
    )
    ds_out["coare_humidity_observation_height"] = xr.DataArray(
        float(zq),
        attrs={
            "units": "m",
            "long_name": "COARE humidity observation height",
            "description": "Humidity measurement height passed to COARE (zq)",
        },
    )
    ds_out["coare_wind_reference_height"] = xr.DataArray(
        float(zrf_u),
        attrs={
            "units": "m",
            "long_name": "COARE wind reference height",
            "description": "Wind reference height used by COARE outputs (zrf_u)",
        },
    )
    ds_out["coare_air_temperature_reference_height"] = xr.DataArray(
        float(zrf_t),
        attrs={
            "units": "m",
            "long_name": "COARE air temperature reference height",
            "description": "Air temperature reference height used by COARE outputs (zrf_t)",
        },
    )
    ds_out["coare_humidity_reference_height"] = xr.DataArray(
        float(zrf_q),
        attrs={
            "units": "m",
            "long_name": "COARE humidity reference height",
            "description": "Humidity reference height used by COARE outputs (zrf_q)",
        },
    )


def _wind_from_meteorological_direction(
    speed: np.ndarray, direction_from_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological (from) direction to eastward/northward components."""
    theta = np.deg2rad(direction_from_deg)
    east = -speed * np.sin(theta)
    north = -speed * np.cos(theta)
    return east, north


def _components_to_meteorological_direction(
    east: np.ndarray, north: np.ndarray
) -> np.ndarray:
    """Eastward/northward components to meteorological (from) direction in degrees."""
    direction = np.rad2deg(np.arctan2(-east, -north))
    return np.mod(direction, 360.0)


def _compute_current_relative_wind(
    u: Any,
    wind_direction: Any,
    current_east: Any,
    current_north: Any,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_rel, wd_rel) magnitudes/directions after subtracting surface current."""
    u_arr = _broadcast_input_to_output(u, n_rows, "u")
    wd_arr = _broadcast_input_to_output(wind_direction, n_rows, "wind_direction")
    ce_arr = _broadcast_input_to_output(current_east, n_rows, "current_east")
    cn_arr = _broadcast_input_to_output(current_north, n_rows, "current_north")
    wind_east, wind_north = _wind_from_meteorological_direction(u_arr, wd_arr)
    rel_east = wind_east - ce_arr
    rel_north = wind_north - cn_arr
    u_rel = np.sqrt(rel_east ** 2 + rel_north ** 2)
    wd_rel = _components_to_meteorological_direction(rel_east, rel_north)
    return u_rel, wd_rel


def _load_global_attrs_json(global_attrs_json: str | Path) -> Dict[str, Any]:
    path = Path(global_attrs_json)
    if not path.exists():
        raise FileNotFoundError(f"Global attributes JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Global attributes JSON must contain a top-level object/dict")
    return data


def coare36_to_netcdf(
    output_path: str | Path,
    global_attrs_json: str | Path,
    *,
    u,
    zu,
    t,
    zt,
    rh,
    zq,
    P,
    ts,
    sw_dn,
    lw_dn,
    lat,
    lon,
    jd,
    zi,
    rain,
    Ss,
    cp=None,
    sigH=None,
    zrf_u: float = DEFAULT_REFERENCE_WIND_HEIGHT_M,
    zrf_t: float = DEFAULT_REFERENCE_AIR_TEMPERATURE_HEIGHT_M,
    zrf_q: float = DEFAULT_REFERENCE_HUMIDITY_HEIGHT_M,
    coolskin: bool = True,
    albedo: str = "constant",
    wind_direction=None,
    payne_min_solar_elev_deg: float = 2.0,
    source_dataset: Optional[xr.Dataset] = None,
    instrument_prefix: str | Iterable[str] = "INST_",
    input_vars_used: Optional[Dict[str, str]] = None,
    sea_water_temperature: Optional[Any] = None,
    sea_water_temperature_attrs: Optional[Dict[str, Any]] = None,
    instrument_config: Optional[Dict[str, str]] = None,
    current_available: bool = False,
    current_east: Optional[Any] = None,
    current_north: Optional[Any] = None,
    output_dim: str = "time",
    output_coord: Optional[Iterable[Any]] = None,
    dataset_attrs: Optional[Dict[str, Any]] = None,
    auto_plot: bool = False,
    to_netcdf_kwargs: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    """
    Run COARE 3.6, assemble an L4 ``xarray.Dataset``, and write it to NetCDF.

    The wrapper calls ``coare36vn_zrf_et`` with the supplied meteorological
    inputs, converts solver columns to named output variables using
    ``get_variables_info()``, optionally copies instrument variables from an
    L3 source dataset, optionally carries L3 variables into the L4 output as
    ``L3_*`` variables, writes the dataset to ``output_path``, and returns the
    in-memory dataset.

    When ``source_dataset`` is provided, non-instrument variables are copied
    into the L4 dataset with an ``L3_`` prefix while preserving their original
    variable attributes. Variables already named ``L3_*`` keep that name. For
    built-in preserved inputs such as ``L3_wind_speed``,
    ``L3_air_temperature``, and the corresponding instrument-height variables,
    the wrapper always writes the actual values passed into this function while
    preserving metadata from the matching source L3 variable when available.

    Parameters
    ----------
    output_path
        Destination path for the NetCDF file. Parent directories are created
        automatically.
    global_attrs_json
        Path to a JSON file containing global attributes to merge into the
        output dataset.
    u
        Wind speed in m/s. Scalar or 1-D array-like.
    zu
        Height of the wind measurement in m. Scalar or 1-D array-like.
    t
        Air temperature in degC. Scalar or 1-D array-like.
    zt
        Height of the air-temperature measurement in m. Scalar or 1-D
        array-like.
    rh
        Relative humidity in percent. Scalar or 1-D array-like.
    zq
        Height of the humidity measurement in m. Scalar or 1-D array-like.
    P
        Surface atmospheric pressure in mb. Scalar or 1-D array-like.
    ts
        Sea-surface temperature used by COARE, in degC. Scalar or 1-D
        array-like.
    sw_dn
        Downwelling shortwave radiation in W/m^2. Scalar or 1-D array-like.
    lw_dn
        Downwelling longwave radiation in W/m^2. Scalar or 1-D array-like.
    lat
        Latitude in degrees north. Scalar or 1-D array-like.
    lon
        Longitude in degrees east. Scalar or 1-D array-like.
    jd
        Fractional Julian day of year. 1-D array-like of length ``N``.
    zi
        Boundary-layer height in m. Scalar or 1-D array-like.
    rain
        Rain rate in mm/hr. Scalar or 1-D array-like.
    Ss
        Sea-surface salinity in PSU. Scalar or 1-D array-like.
    cp
        Optional phase speed of dominant swell in m/s.
    sigH
        Optional significant wave height in m.
    zrf_u
        Reference height for output wind speed in m.
    zrf_t
        Reference height for output air temperature in m.
    zrf_q
        Reference height for output specific humidity in m.
    coolskin
        If ``True``, enable the cool-skin parameterization.
    albedo
        Surface albedo method. Supported values are ``"constant"`` and
        ``"Payne1972"``.
    wind_direction
        Meteorological wind direction in degrees clockwise from north
        (direction the wind is coming from). Required when the caller wants
        current-relative wind computed internally (see ``current_east`` /
        ``current_north``). Passed to the COARE solver so wind-direction
        columns of derived outputs are populated.
    payne_min_solar_elev_deg
        Minimum solar elevation angle used by the Payne 1972 albedo lookup.
    source_dataset
        Optional L3 ``xarray.Dataset``. Variables whose names start with
        ``instrument_prefix`` are copied into the output unchanged. Other
        scalar variables and 1-D variables of length ``N`` are copied into the
        output as ``L3_*`` variables with their metadata preserved.
    instrument_prefix
        Prefix, or iterable of prefixes, used to identify instrument variables
        in ``source_dataset``.
    input_vars_used
        Optional mapping of ``{output_name: source_variable_name}``. This is
        primarily useful for built-in preserved ``L3_*`` outputs when the
        desired source-variable metadata lives under a different L3 variable
        name. It can also add extra 1-D variables from ``source_dataset`` to
        the output.
    sea_water_temperature
        Optional sea-water temperature array to include as
        ``L3_sea_water_temperature``.
    sea_water_temperature_attrs
        Optional attribute overrides for ``L3_sea_water_temperature``.
    instrument_config
        Optional mapping of ``{role: instrument_name}`` written into global
        attributes as ``instrument_<role>``.
    current_available
        If ``True``, annotate the current-relative wind diagnostics
        (``L3_relative_wind_speed`` / ``L3_relative_wind_direction``) as
        derived from a surface current. Automatically set to ``True`` when
        ``current_east`` and ``current_north`` are supplied.
    current_east, current_north
        Optional eastward and northward surface current components in m/s.
        When both are provided together with ``wind_direction``, the wrapper
        computes the current-relative wind speed and direction internally and
        passes those arrays to the COARE 3.6 core. The measured ``u`` and
        ``wind_direction`` remain the values stored on ``L3_wind_speed`` /
        ``L3_wind_direction``; the derived arrays are additionally exported as
        ``L3_relative_wind_speed`` / ``L3_relative_wind_direction``.
    output_dim
        Name of the output dimension.
    output_coord
        Coordinate values for ``output_dim``. Must have length ``N`` when
        provided.
    dataset_attrs
        Additional global attributes merged after the JSON and source-dataset
        attributes.
    auto_plot
        If ``True``, enable COARE diagnostic plotting.
    to_netcdf_kwargs
        Extra keyword arguments forwarded to ``xr.Dataset.to_netcdf``.

    Returns
    -------
    xr.Dataset
        The assembled L4 dataset that was written to disk.
    """
    # Step 1: load required custom global attributes from JSON.
    custom_global_attrs = _load_global_attrs_json(global_attrs_json)
    cp_solver = _as_numpy_1d(cp, "cp") if cp is not None else None
    sigH_solver = _as_numpy_1d(sigH, "sigH") if sigH is not None else None

    # Step 1b: if surface currents are supplied, compute current-relative wind
    # and route it into the COARE core, while keeping the measured u and
    # wind_direction as the source arrays for L3_wind_speed/L3_wind_direction.
    currents_supplied = current_east is not None and current_north is not None
    if currents_supplied and wind_direction is None:
        raise ValueError(
            "current_east/current_north require wind_direction to compute "
            "current-relative wind."
        )
    if currents_supplied:
        current_available = True
        n_wind = _as_numpy_1d(u, "u").size
        u_rel, wd_rel = _compute_current_relative_wind(
            u, wind_direction, current_east, current_north, n_wind
        )
        u_for_solver = u_rel
        wind_direction_for_solver = wd_rel
    else:
        u_for_solver = u
        wind_direction_for_solver = wind_direction

    # Step 2: run COARE solver and force albedo outputs for complete metadata export.
    solver_output = coare36vn_zrf_et(
        u=u_for_solver,
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
        cp=cp_solver,
        sigH=sigH_solver,
        zrf_u=zrf_u,
        zrf_t=zrf_t,
        zrf_q=zrf_q,
        coolskin=coolskin,
        albedo=albedo,
        return_albedo=True,
        wind_direction=wind_direction_for_solver,
        payne_min_solar_elev_deg=payne_min_solar_elev_deg,
        auto_plot=auto_plot,
    )

    resolved_instrument_config = _resolve_instrument_config(source_dataset, instrument_config)

    # Step 3: validate output shape against metadata-driven variable list.
    variables_info = get_variables_info(
        albedo_method=albedo,
        zrf_u=zrf_u,
        zrf_t=zrf_t,
        zrf_q=zrf_q,
    )
    variable_names = list(variables_info.keys())
    n_rows, n_cols = solver_output.shape
    if n_cols < len(variable_names):
        raise ValueError(
            f"Unexpected COARE output column count {n_cols}; expected at least {len(variable_names)}."
        )

    # Step 4: initialize dataset with output coordinate.
    dim_coord = _coerce_coord(output_coord, n_rows, output_dim)
    ds = xr.Dataset(coords={output_dim: dim_coord})

    # Step 5: map COARE columns to named variables with metadata and reverse_sign handling.
    for idx, name in enumerate(variable_names):
        var_meta = variables_info.get(name, {})
        attrs = _sanitize_netcdf_attrs(dict(var_meta))
        data = solver_output[:, idx].copy()
        if bool(var_meta.get("reverse_sign", False)):
            data = -data
        ds[name] = xr.DataArray(data, dims=(output_dim,), attrs=attrs)

    lat_arr = _as_numpy_1d(lat, "lat")
    lon_arr = _as_numpy_1d(lon, "lon")
    if lat_arr.size == 1:
        lat_arr = np.full(n_rows, lat_arr[0], dtype=float)
    if lon_arr.size == 1:
        lon_arr = np.full(n_rows, lon_arr[0], dtype=float)
    if lat_arr.size == n_rows:
        ds["latitude"] = xr.DataArray(
            lat_arr,
            dims=(output_dim,),
            attrs={"units": "degrees_north", "standard_name": "latitude"},
        )
    if lon_arr.size == n_rows:
        ds["longitude"] = xr.DataArray(
            lon_arr,
            dims=(output_dim,),
            attrs={"units": "degrees_east", "standard_name": "longitude"},
        )

    # Step 6: append instrument scalar variables from source dataset (INST_*).
    _add_instrument_variables(ds, source_dataset, instrument_prefix)

    # Step 7: append original wrapper inputs as L3_* variables, preserving source L3 attrs when available.
    preserved_inputs = _l3_input_specs(
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
        lat=lat_arr,
        lon=lon_arr,
        jd=jd,
        zi=zi,
        rain=rain,
        Ss=Ss,
        cp=cp if cp is not None else cp_solver,
        sigH=sigH if sigH is not None else sigH_solver,
        wind_direction=wind_direction,
    )
    default_l3_metadata = _default_l3_input_metadata()
    for output_name, (values, source_name) in preserved_inputs.items():
        resolved_source_name = input_vars_used.get(output_name, source_name) if input_vars_used else source_name
        source_var = source_dataset.get(resolved_source_name) if source_dataset is not None else None
        ds[output_name] = _build_l3_input_dataarray(
            values,
            n_rows,
            output_dim,
            source_var=source_var,
            fallback_attrs=default_l3_metadata.get(output_name),
        )

    # Step 8: copy L3 source variables directly with an L3_ prefix.
    _copy_prefixed_l3_variables(
        ds,
        source_dataset,
        n_rows,
        output_dim,
        instrument_prefix,
        skip_outputs=set(preserved_inputs),
    )

    # Step 9: append additional source variables if requested by mapping.
    if source_dataset is not None and input_vars_used:
        for output_name, input_var in input_vars_used.items():
            if output_name in preserved_inputs:
                continue
            if input_var not in source_dataset:
                continue
            data_array = _build_l3_dataarray(
                source_dataset[input_var],
                n_rows,
                output_dim,
                source_var=source_dataset[input_var],
            )
            if data_array is not None:
                ds[output_name] = data_array

    # Step 10: optionally append sea-water temperature input variable.
    if sea_water_temperature is not None:
        ts_vals = np.asarray(sea_water_temperature, dtype=float)
        if ts_vals.ndim == 0:
            ts_vals = np.full(n_rows, float(ts_vals))
        if ts_vals.ndim == 1 and ts_vals.size == n_rows:
            extra_attrs = sea_water_temperature_attrs or {}
            ds["L3_sea_water_temperature"] = xr.DataArray(
                ts_vals,
                dims=(output_dim,),
                attrs={
                    "units": extra_attrs.get("units", "degC"),
                    "long_name": extra_attrs.get("long_name", "L3 input: sea surface temperature"),
                    "coverage_content_type": extra_attrs.get("coverage_content_type", "physicalMeasurement"),
                    "standard_name": extra_attrs.get("standard_name", ""),
                    "description": extra_attrs.get("description", "Sea surface temperature used as COARE input"),
                    "source_variable": extra_attrs.get("source_variable", "sea_surface_temperature"),
                },
            )

    # Step 11: add optional comments when callers supplied current-relative wind.
    # This is metadata-only and does not alter the wind values passed to COARE.
    # Step 11: when currents are available, emit derived current-relative
    # wind diagnostics and annotate them. L3_wind_speed/L3_wind_direction
    # continue to carry the measured values passed via u/wind_direction.
    if current_available:
        if currents_supplied:
            rel_speed_attrs = {
                "units": "m s-1",
                "long_name": "Wind speed relative to surface current",
                "description": (
                    "Magnitude of measured wind minus surface current vector; "
                    "this is the wind speed actually passed to the COARE 3.6 core."
                ),
                "comment": "Wind speed relative to surface current",
            }
            rel_dir_attrs = {
                "units": "degree",
                "long_name": "Wind direction relative to surface current",
                "description": (
                    "Meteorological direction (clockwise from north) of the "
                    "measured-wind-minus-current vector; passed to the COARE "
                    "3.6 core."
                ),
                "comment": "Wind direction relative to surface current",
            }
            ds["L3_relative_wind_speed"] = xr.DataArray(
                np.asarray(u_for_solver, dtype=float),
                dims=(output_dim,),
                attrs=_sanitize_netcdf_attrs(rel_speed_attrs),
            )
            if wind_direction_for_solver is not None:
                ds["L3_relative_wind_direction"] = xr.DataArray(
                    np.asarray(wind_direction_for_solver, dtype=float),
                    dims=(output_dim,),
                    attrs=_sanitize_netcdf_attrs(rel_dir_attrs),
                )
        else:
            # Legacy path: caller pre-subtracted current before calling; keep
            # the historical annotation on L3_wind_speed / L3_wind_direction.
            if "L3_wind_direction" in ds:
                ds["L3_wind_direction"].attrs["comment"] = (
                    "Wind direction relative to surface current"
                )
            if "L3_wind_speed" in ds:
                ds["L3_wind_speed"].attrs["comment"] = (
                    "Wind speed relative to surface current"
                )

    # Step 12: create scalar metadata used by COARE and used in global attrs.
    zu_scalar = _as_float_attr(zu, "zu")
    zt_scalar = _as_float_attr(zt, "zt")
    zq_scalar = _as_float_attr(zq, "zq")
    zrf_u_scalar = float(zrf_u)
    zrf_t_scalar = float(zrf_t)
    zrf_q_scalar = float(zrf_q)

    _add_coare_height_variables(
        ds,
        zu=zu_scalar,
        zt=zt_scalar,
        zq=zq_scalar,
        zrf_u=zrf_u_scalar,
        zrf_t=zrf_t_scalar,
        zrf_q=zrf_q_scalar,
    )

    # Step 13: write global metadata.
    base_from_source = dict(source_dataset.attrs) if source_dataset is not None else {}
    base_attrs = {
        "title": "COARE 3.6 air-sea flux outputs",
        "summary": "Computed with uop_coare.coare36vn_zrf_et and exported with xarray.",
        "feature_type": "timeSeries",
        "albedo_method": str(albedo),
        "coolskin": str(bool(coolskin)).lower(),
        "source": "uop_coare.coare36vn_zrf_et",
        "uop_coare_version": UOP_COARE_VERSION,
        "processing_level": "L4",
        "input_level": "L3",
        "input_variables_preserved": "True",
        "options": "Cool skin(jcoolx = 1 is default)" if coolskin else "Cool skin disabled",
        "description": "Outputs from COARE 3.6 model calculations.",
        "references": "https://github.com/NOAA-PSL/COARE-algorithm/blob/master/References/coare36_readme_1.pdf",
        "wind_instrument_height_m": zu_scalar,
        "air_temperature_instrument_height_m": zt_scalar,
        "humidity_instrument_height_m": zq_scalar,
        "reference_wind_height_m": zrf_u_scalar,
        "reference_air_temperature_height_m": zrf_t_scalar,
        "reference_humidity_height_m": zrf_q_scalar,
    }
    if resolved_instrument_config:
        for role, inst_name in resolved_instrument_config.items():
            base_attrs[f"instrument_{role}"] = str(inst_name)
    base_attrs = {**base_from_source, **base_attrs, **custom_global_attrs}
    if dataset_attrs:
        base_attrs.update(dataset_attrs)
    ds.attrs.update(base_attrs)

    # Step 14: write NetCDF file and return in-memory dataset.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    netcdf_kwargs = dict(to_netcdf_kwargs or {})
    ds.to_netcdf(output_path, **netcdf_kwargs)
    return ds
