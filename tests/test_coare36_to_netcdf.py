import numpy as np
import pytest
import xarray as xr

from uop_coare.coare36_variables_config import get_variables_info
from uop_coare.coare36_to_netcdf import coare36_to_netcdf
from uop_coare.coare36vn_zrf_et import coare36vn_zrf_et


def _build_constant_inputs(N):
    return dict(
        u=np.full(N, 5.0),
        zu=np.full(N, 10.0),
        t=np.full(N, 20.0),
        zt=np.full(N, 10.0),
        rh=np.full(N, 80.0),
        zq=np.full(N, 10.0),
        P=np.full(N, 1013.0),
        ts=np.full(N, 21.0),
        sw_dn=np.full(N, 200.0),
        lw_dn=np.full(N, 400.0),
        lat=np.full(N, 45.0),
        lon=np.full(N, -30.0),
        jd=np.linspace(100.0, 100.5, N),
        zi=np.full(N, 600.0),
        rain=np.full(N, 0.0),
        Ss=np.full(N, 35.0),
    )


def test_coare36_to_netcdf_writes_file_and_metadata(tmp_path):
    N = 4
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_output.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text('{"project": "unit-test", "config_file": "test.json"}', encoding="utf-8")
    src = xr.Dataset(
        {
            "INST_Gill": xr.DataArray(1.0, attrs={"long_name": "wind instrument", "serial_number": "G-123"}),
            "INST_HC2A": xr.DataArray(2.0, attrs={"long_name": "temp/humidity instrument"}),
            "not_instrument": xr.DataArray(3.0),
            "wind_speed": xr.DataArray(
                inputs["u"],
                dims=("time",),
                attrs={
                    "units": "m/s",
                    "long_name": "streamwise wind speed",
                    "standard_name": "wind_speed",
                    "comment": "measured by source L3 pipeline",
                },
            ),
            "air_temperature": xr.DataArray(
                inputs["t"],
                dims=("time",),
                attrs={"units": "degC", "long_name": "air temperature"},
            ),
        }
    )

    ds = coare36_to_netcdf(
        out_path,
        attrs_json,
        **inputs,
        albedo="Payne1972",
        zrf_u=3.0,
        zrf_t=3.5,
        zrf_q=4.0,
        source_dataset=src,
    )
    assert out_path.exists()
    assert "friction_velocity" in ds
    assert "net_solar_radiation" in ds
    assert "albedo" in ds
    assert ds["friction_velocity"].attrs["units"] == "m/s"
    assert ds["net_solar_radiation"].attrs["standard_name"] == "surface_net_downward_shortwave_flux"
    assert ds.attrs["wind_instrument_height_m"] == 10.0
    assert ds.attrs["air_temperature_instrument_height_m"] == 10.0
    assert ds.attrs["humidity_instrument_height_m"] == 10.0
    assert ds.attrs["reference_wind_height_m"] == 3.0
    assert ds.attrs["reference_air_temperature_height_m"] == 3.5
    assert ds.attrs["reference_humidity_height_m"] == 4.0
    assert ds.attrs["project"] == "unit-test"
    assert ds.attrs["config_file"] == "test.json"
    assert "uop_coare_version" in ds.attrs
    assert str(ds.attrs["uop_coare_version"]) != ""
    assert "INST_Gill" in ds
    assert "INST_HC2A" in ds
    assert "not_instrument" not in ds
    assert ds["INST_Gill"].attrs["serial_number"] == "G-123"
    assert "L3_wind_speed" in ds
    assert "L3_air_temperature" in ds
    assert "L3_wind_height" in ds
    assert "L3_air_temperature_height" in ds
    np.testing.assert_allclose(ds["L3_wind_speed"].values, inputs["u"])
    np.testing.assert_allclose(ds["L3_air_temperature"].values, inputs["t"])
    np.testing.assert_allclose(ds["L3_wind_height"].values, inputs["zu"])
    np.testing.assert_allclose(ds["L3_air_temperature_height"].values, inputs["zt"])
    assert ds["L3_wind_speed"].attrs["units"] == "m/s"
    assert ds["L3_wind_speed"].attrs["long_name"] == "streamwise wind speed"
    assert ds["L3_wind_speed"].attrs["standard_name"] == "wind_speed"
    assert ds["L3_wind_speed"].attrs["comment"] == "measured by source L3 pipeline"
    assert ds["L3_wind_height"].attrs["units"] == "m"
    assert "description" in ds["L3_wind_height"].attrs
    assert ds.attrs["input_variables_preserved"] == "True"

    with xr.open_dataset(out_path) as saved:
        assert "friction_velocity" in saved
        assert "albedo" in saved
        assert saved.sizes["time"] == N
        assert saved["albedo"].attrs["coverage_content_type"] == "radiation"
        assert saved.attrs["wind_instrument_height_m"] == 10.0
        assert saved.attrs["reference_humidity_height_m"] == 4.0
        assert "uop_coare_version" in saved.attrs
        assert "INST_Gill" in saved
        assert "INST_HC2A" in saved
        assert saved["INST_Gill"].attrs["serial_number"] == "G-123"
        assert "L3_wind_speed" in saved
        assert "L3_wind_height" in saved
        np.testing.assert_allclose(saved["L3_wind_speed"].values, inputs["u"])
        np.testing.assert_allclose(saved["L3_wind_height"].values, inputs["zu"])
        assert saved["L3_wind_speed"].attrs["long_name"] == "streamwise wind speed"
        assert saved["L3_wind_speed"].attrs["comment"] == "measured by source L3 pipeline"


def test_coare36_to_netcdf_resolves_exact_instrument_names_in_attrs(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_exact_instrument.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text("{}", encoding="utf-8")
    src = xr.Dataset(
        {
            "INST_GILL_BOW": xr.DataArray(1.0),
            "INST_WXT": xr.DataArray(2.0),
        }
    )

    ds = coare36_to_netcdf(
        out_path,
        attrs_json,
        **inputs,
        source_dataset=src,
        instrument_config={"wind": "INST_GILL_BOW", "air_temperature": "INST_WXT"},
    )

    assert ds.attrs["instrument_wind"] == "INST_GILL_BOW"
    assert ds.attrs["instrument_air_temperature"] == "INST_WXT"


def test_coare36_to_netcdf_resolves_unique_instrument_prefix_in_attrs(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_unique_instrument.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text("{}", encoding="utf-8")
    src = xr.Dataset(
        {
            "INST_GILL_BOW": xr.DataArray(1.0),
            "INST_WXT": xr.DataArray(2.0),
        }
    )

    ds = coare36_to_netcdf(
        out_path,
        attrs_json,
        **inputs,
        source_dataset=src,
        instrument_config={"wind": "INST_GILL"},
    )

    assert ds.attrs["instrument_wind"] == "INST_GILL_BOW"


def test_coare36_to_netcdf_raises_for_ambiguous_instrument_prefix(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_ambiguous_instrument.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text("{}", encoding="utf-8")
    src = xr.Dataset(
        {
            "INST_GILL_BOW": xr.DataArray(1.0),
            "INST_GILL_MAST": xr.DataArray(2.0),
        }
    )

    with pytest.raises(ValueError, match="Ambiguous instrument_config.*INST_GILL.*Pass the exact variable name"):
        coare36_to_netcdf(
            out_path,
            attrs_json,
            **inputs,
            source_dataset=src,
            instrument_config={"wind": "INST_GILL"},
        )


def test_coare36_to_netcdf_raises_for_missing_instrument_name(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_missing_instrument.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text("{}", encoding="utf-8")
    src = xr.Dataset({"INST_WXT": xr.DataArray(1.0)})

    with pytest.raises(ValueError, match="Instrument 'INST_GILL'.*role 'wind'.*not found"):
        coare36_to_netcdf(
            out_path,
            attrs_json,
            **inputs,
            source_dataset=src,
            instrument_config={"wind": "INST_GILL"},
        )


def test_coare36_to_netcdf_preserves_full_coare_inputs_as_variables(tmp_path):
    N = 4
    inputs = _build_constant_inputs(N)
    inputs["zu"] = 10.0
    inputs["zt"] = 12.0
    inputs["zq"] = 14.0
    inputs["zi"] = 600.0
    inputs["lat"] = 45.0
    inputs["lon"] = -30.0
    inputs["cp"] = 8.0
    inputs["sigH"] = 2.5
    out_path = tmp_path / "coare36_inputs.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text("{}", encoding="utf-8")

    ds = coare36_to_netcdf(out_path, attrs_json, **inputs)

    expected = {
        "L3_wind_speed": np.full(N, 5.0),
        "L3_wind_height": np.full(N, 10.0),
        "L3_air_temperature": np.full(N, 20.0),
        "L3_air_temperature_height": np.full(N, 12.0),
        "L3_relative_humidity": np.full(N, 80.0),
        "L3_relative_humidity_height": np.full(N, 14.0),
        "L3_air_pressure": np.full(N, 1013.0),
        "L3_sea_surface_temperature": np.full(N, 21.0),
        "L3_downwelling_shortwave_flux": np.full(N, 200.0),
        "L3_downwelling_longwave_flux": np.full(N, 400.0),
        "L3_latitude": np.full(N, 45.0),
        "L3_longitude": np.full(N, -30.0),
        "L3_julian_day": inputs["jd"],
        "L3_boundary_layer_height": np.full(N, 600.0),
        "L3_rain_rate": np.full(N, 0.0),
        "L3_sea_surface_salinity": np.full(N, 35.0),
        "L3_wave_phase_speed": np.full(N, 8.0),
        "L3_significant_wave_height": np.full(N, 2.5),
    }

    for name, values in expected.items():
        assert name in ds
        assert ds[name].dims == ("time",)
        np.testing.assert_allclose(ds[name].values, values)
        assert "long_name" in ds[name].attrs
        assert "units" in ds[name].attrs
        assert "description" in ds[name].attrs

    with xr.open_dataset(out_path) as saved:
        for name, values in expected.items():
            assert name in saved
            np.testing.assert_allclose(saved[name].values, values)


def test_coare36_to_netcdf_omits_optional_wave_inputs_when_not_provided(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_inputs_no_waves.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text("{}", encoding="utf-8")

    ds = coare36_to_netcdf(out_path, attrs_json, **inputs)

    assert "L3_wave_phase_speed" not in ds
    assert "L3_significant_wave_height" not in ds


def test_coare36_to_netcdf_matches_solver_output(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_compare.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text('{"project": "unit-test"}', encoding="utf-8")

    direct = coare36vn_zrf_et(**inputs, albedo="constant", return_albedo=True, auto_plot=False)
    ds = coare36_to_netcdf(out_path, attrs_json, **inputs, albedo="constant")

    names = list(get_variables_info(albedo_method="constant").keys())
    i_friction_velocity = names.index("friction_velocity")
    i_net_solar_radiation = names.index("net_solar_radiation")
    i_albedo = names.index("albedo")

    np.testing.assert_allclose(ds["friction_velocity"].values, direct[:, i_friction_velocity])
    np.testing.assert_allclose(ds["net_solar_radiation"].values, direct[:, i_net_solar_radiation])
    np.testing.assert_allclose(ds["albedo"].values, direct[:, i_albedo])


def test_coare36_to_netcdf_copies_l3_source_variables_with_prefixed_names(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_l3_copy.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text('{"project": "unit-test"}', encoding="utf-8")
    src = xr.Dataset(
        {
            "wind_speed": xr.DataArray(
                inputs["u"],
                dims=("sample",),
                attrs={
                    "units": "m/s",
                    "long_name": "true wind speed",
                    "standard_name": "wind_speed",
                    "instrument": "Gill",
                },
            ),
            "air_temperature": xr.DataArray(
                inputs["t"],
                dims=("sample",),
                attrs={"units": "degC", "long_name": "air temperature from L3"},
            ),
            "INST_Gill": xr.DataArray(1.0, attrs={"serial_number": "G-123"}),
        }
    )

    ds = coare36_to_netcdf(out_path, attrs_json, **inputs, source_dataset=src)

    assert "L3_wind_speed" in ds
    assert "L3_air_temperature" in ds
    np.testing.assert_allclose(ds["L3_wind_speed"].values, inputs["u"])
    np.testing.assert_allclose(ds["L3_air_temperature"].values, inputs["t"])
    assert ds["L3_wind_speed"].attrs == src["wind_speed"].attrs
    assert ds["L3_air_temperature"].attrs == src["air_temperature"].attrs
    assert "source_variable" not in ds["L3_wind_speed"].attrs
    assert "description" not in ds["L3_wind_speed"].attrs
    assert ds["INST_Gill"].attrs == src["INST_Gill"].attrs


def test_coare36_to_netcdf_copies_all_l3_source_variables_with_prefix(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_l3_all_prefixed.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text('{"project": "unit-test"}', encoding="utf-8")
    src = xr.Dataset(
        {
            "wind_speed": xr.DataArray(
                np.full(N, 7.5),
                dims=("sample",),
                attrs={"units": "m/s", "long_name": "source wind speed"},
            ),
            "sea_surface_temperature": xr.DataArray(
                np.full(N, 19.5),
                dims=("sample",),
                attrs={"units": "degC", "long_name": "bulk sea surface temperature"},
            ),
            "already_prefixed": xr.DataArray(
                np.full(N, 1.0),
                dims=("sample",),
                attrs={"long_name": "custom l3 field"},
            ),
            "L3_existing_name": xr.DataArray(
                np.full(N, 2.0),
                dims=("sample",),
                attrs={"long_name": "already prefixed in source"},
            ),
            "station_name": xr.DataArray("WHOI", attrs={"long_name": "station identifier"}),
            "INST_Gill": xr.DataArray(1.0, attrs={"serial_number": "G-123"}),
        }
    )

    ds = coare36_to_netcdf(out_path, attrs_json, **inputs, source_dataset=src)

    np.testing.assert_allclose(ds["L3_wind_speed"].values, inputs["u"])
    np.testing.assert_allclose(ds["L3_sea_surface_temperature"].values, inputs["ts"])
    np.testing.assert_allclose(ds["L3_already_prefixed"].values, src["already_prefixed"].values)
    np.testing.assert_allclose(ds["L3_existing_name"].values, src["L3_existing_name"].values)
    assert ds["L3_wind_speed"].attrs == src["wind_speed"].attrs
    assert ds["L3_sea_surface_temperature"].attrs == src["sea_surface_temperature"].attrs
    assert ds["L3_already_prefixed"].attrs == src["already_prefixed"].attrs
    assert ds["L3_existing_name"].attrs == src["L3_existing_name"].attrs
    assert ds["L3_station_name"].item() == "WHOI"
    assert ds["L3_station_name"].attrs == src["station_name"].attrs
    assert "L3_INST_Gill" not in ds


def test_coare36_to_netcdf_uses_input_vars_used_for_builtin_l3_exports(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_l3_mapped.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text('{"project": "unit-test"}', encoding="utf-8")
    src = xr.Dataset(
        {
            "gill_wind_speed": xr.DataArray(
                inputs["u"],
                dims=("sample",),
                attrs={
                    "units": "m/s",
                    "long_name": "Gill wind speed",
                    "standard_name": "wind_speed",
                },
            ),
            "wxt_air_temperature": xr.DataArray(
                inputs["t"],
                dims=("sample",),
                attrs={"units": "degC", "long_name": "WXT air temperature"},
            ),
        }
    )

    ds = coare36_to_netcdf(
        out_path,
        attrs_json,
        **inputs,
        source_dataset=src,
        input_vars_used={
            "L3_wind_speed": "gill_wind_speed",
            "L3_air_temperature": "wxt_air_temperature",
        },
    )

    assert ds["L3_wind_speed"].attrs == src["gill_wind_speed"].attrs
    assert ds["L3_air_temperature"].attrs == src["wxt_air_temperature"].attrs
    np.testing.assert_allclose(ds["L3_wind_speed"].values, inputs["u"])
    np.testing.assert_allclose(ds["L3_air_temperature"].values, inputs["t"])


def test_coare36_to_netcdf_builtin_l3_exports_fall_back_to_wrapper_inputs_when_source_var_missing(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_l3_wrapper_fallback.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text('{"project": "unit-test"}', encoding="utf-8")
    src = xr.Dataset(
        {
            "gill_wind_speed": xr.DataArray(
                np.full(N, 99.0),
                dims=("sample",),
                attrs={"units": "m/s", "long_name": "Gill wind speed"},
            ),
        }
    )

    ds = coare36_to_netcdf(
        out_path,
        attrs_json,
        **inputs,
        source_dataset=src,
        input_vars_used={"L3_wind_speed": "gill_wind_speed"},
    )

    np.testing.assert_allclose(ds["L3_wind_speed"].values, inputs["u"])
    np.testing.assert_allclose(ds["L3_air_pressure"].values, inputs["P"])
    assert ds["L3_wind_speed"].attrs == src["gill_wind_speed"].attrs
    assert ds["L3_air_pressure"].attrs["units"] == "hPa"
    assert "description" in ds["L3_air_pressure"].attrs


def test_coare36_to_netcdf_builtin_l3_exports_keep_wrapper_values_and_source_metadata(tmp_path):
    N = 3
    inputs = _build_constant_inputs(N)
    out_path = tmp_path / "coare36_l3_wrapper_values.nc"
    attrs_json = tmp_path / "global_attrs.json"
    attrs_json.write_text('{"project": "unit-test"}', encoding="utf-8")
    src = xr.Dataset(
        {
            "wind_speed": xr.DataArray(
                np.full(N, 99.0),
                dims=("sample",),
                attrs={"units": "m/s", "long_name": "source wind speed"},
            ),
            "air_pressure": xr.DataArray(
                np.full(N, 900.0),
                dims=("sample",),
                attrs={"units": "mb", "long_name": "source air pressure"},
            ),
        }
    )

    ds = coare36_to_netcdf(out_path, attrs_json, **inputs, source_dataset=src)

    np.testing.assert_allclose(ds["L3_wind_speed"].values, inputs["u"])
    np.testing.assert_allclose(ds["L3_air_pressure"].values, inputs["P"])
    assert ds["L3_wind_speed"].attrs == src["wind_speed"].attrs
    assert ds["L3_air_pressure"].attrs == src["air_pressure"].attrs
