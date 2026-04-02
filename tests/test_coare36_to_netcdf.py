import numpy as np
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
            "INST_Gill": xr.DataArray(1.0, attrs={"long_name": "wind instrument"}),
            "INST_HC2A": xr.DataArray(2.0, attrs={"long_name": "temp/humidity instrument"}),
            "not_instrument": xr.DataArray(3.0),
            "u": xr.DataArray(inputs["u"], dims=("time",), attrs={"units": "m/s", "long_name": "streamwise wind speed"}),
            "t": xr.DataArray(inputs["t"], dims=("time",), attrs={"units": "degC", "long_name": "air temperature"}),
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
    assert "L3_wind_speed" in ds
    assert "L3_air_temperature" in ds
    np.testing.assert_allclose(ds["L3_wind_speed"].values, inputs["u"])
    np.testing.assert_allclose(ds["L3_air_temperature"].values, inputs["t"])
    assert ds["L3_wind_speed"].attrs["units"] == "m/s"
    assert ds["L3_wind_speed"].attrs["long_name"] == "L3 input: streamwise wind speed"
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
        assert "L3_wind_speed" in saved
        np.testing.assert_allclose(saved["L3_wind_speed"].values, inputs["u"])


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
