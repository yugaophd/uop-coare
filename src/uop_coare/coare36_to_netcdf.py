from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import xarray as xr

from .coare36_variables_config import get_variables_info
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
    source_var: xr.DataArray,
) -> Optional[xr.DataArray]:
    data = np.asarray(values.values if hasattr(values, "values") else values)
    if data.ndim == 0:
        data = np.full(n_rows, data.item())
    if data.ndim != 1 or data.size != n_rows:
        return None

    return xr.DataArray(data, dims=(output_dim,), attrs=dict(source_var.attrs))


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
    zrf_u: float = 10.0,
    zrf_t: float = 10.0,
    zrf_q: float = 10.0,
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
    output_dim: str = "time",
    output_coord: Optional[Iterable[Any]] = None,
    dataset_attrs: Optional[Dict[str, Any]] = None,
    auto_plot: bool = False,
    to_netcdf_kwargs: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    """
    Run COARE 3.6 and export outputs directly to a NetCDF file.

    Calls ``coare36vn_zrf_et`` with the supplied meteorological inputs,
    maps every output column to a named xarray variable (with CF-compliant
    attributes from ``get_variables_info``), optionally copies instrument
    metadata and original L3 inputs from a source dataset, writes the
    assembled dataset to a NetCDF file at ``output_path``, and returns the
    in-memory ``xr.Dataset``.

    Parameters
    ----------
    - output_path: destination path for the NetCDF file. Parent directories
      are created automatically if they do not exist.
    - global_attrs_json: path to a JSON file whose top-level keys are merged
      into the dataset's global attributes (highest priority, overrides
      base and source attributes).

    Required meteorological inputs (keyword-only):
    - u: wind speed (m/s); 1-D array-like of length N.
    - zu: height of the wind measurement (m); scalar or 1-D array.
    - t: air temperature (degC); 1-D array-like of length N.
    - zt: height of the air-temperature measurement (m); scalar or 1-D array.
    - rh: relative humidity (%); 1-D array-like of length N.
    - zq: height of the humidity measurement (m); scalar or 1-D array.
    - P: surface atmospheric pressure (mb); 1-D array-like of length N.
    - ts: sea surface (skin) temperature (degC); 1-D array-like of length N.
    - sw_dn: downwelling shortwave radiation (W/m²); 1-D array-like of length N.
    - lw_dn: downwelling longwave radiation (W/m²); 1-D array-like of length N.
    - lat: latitude (degrees north); scalar or 1-D array-like.
    - lon: longitude (degrees east); scalar or 1-D array-like.
    - jd: fractional Julian day of year (e.g. output of ``calculate_jd``);
      1-D array-like of length N.
    - zi: boundary layer height (m); scalar or 1-D array-like.
    - rain: rainfall rate (mm/hr); 1-D array-like of length N.
    - Ss: sea surface salinity (PSU); scalar or 1-D array-like.

    Optional COARE solver inputs (keyword-only):
    - cp: phase speed of dominant ocean swell (m/s); pass ``None`` to omit
      wave-state correction. Default ``None``.
    - sigH: significant wave height (m); pass ``None`` to omit. Default ``None``.
    - zrf_u: reference height for output wind speed (m). Default 10.0.
    - zrf_t: reference height for output air temperature (m). Default 10.0.
    - zrf_q: reference height for output specific humidity (m). Default 10.0.
    - coolskin: when ``True`` (default) enable the cool-skin parameterisation
      (jcoolx = 1 in the original MATLAB code).
    - albedo: surface albedo method. ``"constant"`` uses a fixed value;
      ``"Payne1972"`` computes solar-elevation-dependent albedo. Default
      ``"constant"``.
    - wind_direction: meteorological wind direction (degrees clockwise from
      north); only used when ``current_available`` is True. Default ``None``.
    - payne_min_solar_elev_deg: minimum solar elevation angle (degrees) below
      which the Payne 1972 albedo lookup returns a floor value. Only used
      when ``albedo="Payne1972"``. Default 2.0.
    - auto_plot: when ``True``, the solver produces diagnostic plots. Default
      ``False``.

    Dataset construction options (keyword-only):
    - source_dataset: optional xarray Dataset whose variables will be
      inspected for instrument metadata (prefix ``instrument_prefix``) and
      as a metadata source for preserved wrapper inputs. Default ``None``.
    - instrument_prefix: prefix or prefixes used to identify instrument-
      metadata variables in ``source_dataset`` that should be copied to the
      output. Default ``"INST_"``.
    - input_vars_used: optional mapping of ``{output_name: source_variable_name}``
      for adding additional 1-D variables from ``source_dataset`` to the output
      dataset with provenance attributes. Default ``None``.
    - sea_water_temperature: optional array-like of sea-water temperature
      values (length N) to include as the variable ``"L3_sea_water_temperature"``.
      Default ``None``.
    - sea_water_temperature_attrs: additional CF attributes (e.g. ``units``,
      ``long_name``, ``standard_name``) for the ``"L3_sea_water_temperature"``
      variable. Keys not provided fall back to sensible defaults. Default
      ``None``.
    - instrument_config: mapping of ``{role: instrument_name}`` added to the
      global attributes as ``instrument_<role>``. Default ``None``.
    - current_available: when ``True``, annotates ``"L3_wind_direction"`` and
      ``"L3_wind_speed"`` as being relative to the surface current. Default
      ``False``.
    - output_dim: name of the output dimension (e.g. ``"time"``). Default
      ``"time"``.
    - output_coord: coordinate values for ``output_dim``; must have length N.
      When ``None`` an integer range ``0…N-1`` is used. Default ``None``.
    - dataset_attrs: extra global attributes merged last (after
      ``global_attrs_json``). Default ``None``.
    - to_netcdf_kwargs: additional keyword arguments forwarded verbatim to
      ``xr.Dataset.to_netcdf``. Default ``None``.

    Returns
    -------
    - xr.Dataset: the assembled dataset (same object written to disk), with
      CF-compliant variable attributes and global metadata.
    """
    # Step 1: load required custom global attributes from JSON.
    custom_global_attrs = _load_global_attrs_json(global_attrs_json)

    # Step 2: run COARE solver and force albedo outputs for complete metadata export.
    solver_output = coare36vn_zrf_et(
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
        cp=cp,
        sigH=sigH,
        zrf_u=zrf_u,
        zrf_t=zrf_t,
        zrf_q=zrf_q,
        coolskin=coolskin,
        albedo=albedo,
        return_albedo=True,
        wind_direction=wind_direction,
        payne_min_solar_elev_deg=payne_min_solar_elev_deg,
        auto_plot=auto_plot,
    )

    # Step 3: validate output shape against metadata-driven variable list.
    variables_info = get_variables_info(albedo_method=albedo)
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

    # Step 7: append original wrapper inputs as L3_* variables.
    preserved_inputs = {
        "L3_wind_speed": (u, "wind_speed"),
        "L3_air_temperature": (t, "air_temperature"),
        "L3_relative_humidity": (rh, "relative_humidity"),
        "L3_air_pressure": (P, "air_pressure"),
        "L3_sea_surface_skin_temperature": (ts, "sea_surface_skin_temperature"),
        "L3_downwelling_shortwave_flux": (sw_dn, "downwelling_shortwave_flux"),
        "L3_downwelling_longwave_flux": (lw_dn, "downwelling_longwave_flux"),
        "L3_latitude": (lat_arr, "latitude"),
        "L3_longitude": (lon_arr, "longitude"),
        "L3_julian_day": (jd, "julian_day"),
        "L3_boundary_layer_height": (zi, "boundary_layer_height"),
        "L3_rain_rate": (rain, "rain_rate"),
        "L3_sea_surface_salinity": (Ss, "sea_surface_salinity"),
    }
    optional_inputs = {
        "L3_wave_phase_speed": (cp, "wave_phase_speed"),
        "L3_significant_wave_height": (sigH, "significant_wave_height"),
        "L3_wind_direction": (wind_direction, "wind_direction"),
    }
    preserved_inputs.update({name: spec for name, spec in optional_inputs.items() if spec[0] is not None})

    for output_name, (values, source_name) in preserved_inputs.items():
        source_var = source_dataset[source_name]
        data_array = _build_l3_dataarray(
            source_var,
            n_rows,
            output_dim,
            source_var=source_var,
        )
        if data_array is not None:
            ds[output_name] = data_array

    # Step 8: append additional source variables if requested by mapping.
    if source_dataset is not None and input_vars_used:
        for output_name, input_var in input_vars_used.items():
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

    # Step 9: optionally append sea-water temperature input variable.
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

    # Step 10: add optional comments when relative wind/current path was used.
    if current_available:
        if "L3_wind_direction" in ds:
            ds["L3_wind_direction"].attrs["comment"] = "Wind direction relative to surface current"
        if "L3_wind_speed" in ds:
            ds["L3_wind_speed"].attrs["comment"] = "Wind speed relative to surface current"

    # Step 11: create scalar metadata used by COARE and used in global attrs.
    zu_scalar = _as_float_attr(zu, "zu")
    zt_scalar = _as_float_attr(zt, "zt")
    zq_scalar = _as_float_attr(zq, "zq")
    zrf_u_scalar = float(zrf_u)
    zrf_t_scalar = float(zrf_t)
    zrf_q_scalar = float(zrf_q)

    # Step 12: write global metadata.
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
    if instrument_config is not None:
        for role, inst_name in instrument_config.items():
            base_attrs[f"instrument_{role}"] = str(inst_name)
    base_attrs = {**base_from_source, **base_attrs, **custom_global_attrs}
    if dataset_attrs:
        base_attrs.update(dataset_attrs)
    ds.attrs.update(base_attrs)

    # Step 13: write NetCDF file and return in-memory dataset.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    netcdf_kwargs = dict(to_netcdf_kwargs or {})
    ds.to_netcdf(output_path, **netcdf_kwargs)
    return ds
