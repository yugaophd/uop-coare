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
    instrument_prefix: str,
) -> None:
    if source_dataset is None:
        return
    for var_name in source_dataset.data_vars:
        if var_name.startswith(instrument_prefix):
            ds_out[var_name] = source_dataset[var_name].copy(deep=True)


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
    instrument_prefix: str = "INST_",
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

    Returns the in-memory xarray Dataset after writing to disk.
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

    # Step 7: append original input variables if requested by mapping.
    if source_dataset is not None and input_vars_used:
        for output_name, input_var in input_vars_used.items():
            if input_var not in source_dataset:
                continue
            src = source_dataset[input_var]
            src_vals = np.asarray(src.values)
            if src_vals.ndim != 1:
                continue
            if src_vals.size != n_rows:
                continue
            ds[output_name] = xr.DataArray(
                src_vals,
                dims=(output_dim,),
                attrs={
                    "units": src.attrs.get("units", ""),
                    "long_name": f"L3 input: {src.attrs.get('long_name', input_var)}",
                    "coverage_content_type": src.attrs.get("coverage_content_type", "physicalMeasurement"),
                    "standard_name": src.attrs.get("standard_name", ""),
                    "description": f"Original L3 input variable: {input_var}",
                    "source_variable": input_var,
                },
            )

    # Step 8: optionally append sea-water temperature input variable.
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

    # Step 9: add optional comments when relative wind/current path was used.
    if current_available:
        if "L3_wind_direction" in ds:
            ds["L3_wind_direction"].attrs["comment"] = "Wind direction relative to surface current"
        if "L3_wind_speed" in ds:
            ds["L3_wind_speed"].attrs["comment"] = "Wind speed relative to surface current"

    # Step 10: create scalar metadata used by COARE and used in global attrs.
    zu_scalar = _as_float_attr(zu, "zu")
    zt_scalar = _as_float_attr(zt, "zt")
    zq_scalar = _as_float_attr(zq, "zq")
    zrf_u_scalar = float(zrf_u)
    zrf_t_scalar = float(zrf_t)
    zrf_q_scalar = float(zrf_q)

    # Step 11: write global metadata.
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
        "input_variables_preserved": "Original L3 variables included with L3_ prefix" if input_vars_used else "False",
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

    # Step 12: write NetCDF file and return in-memory dataset.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    netcdf_kwargs = dict(to_netcdf_kwargs or {})
    ds.to_netcdf(output_path, **netcdf_kwargs)
    return ds
