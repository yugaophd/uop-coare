import numpy as np
import pytest

from uop_coare.coare36_variables_config import get_variables_info
from uop_coare.coare36vn_zrf_et import albedo_vector, coare36vn_zrf_et


def _build_inputs(n_samples=6):
    return {
        "u": np.full(n_samples, 5.0),
        "zu": np.full(n_samples, 10.0),
        "t": np.full(n_samples, 20.0),
        "zt": np.full(n_samples, 10.0),
        "rh": np.full(n_samples, 80.0),
        "zq": np.full(n_samples, 10.0),
        "P": np.full(n_samples, 1013.0),
        "ts": np.full(n_samples, 21.0),
        "sw_dn": np.full(n_samples, 200.0),
        "lw_dn": np.full(n_samples, 400.0),
        "lat": np.full(n_samples, 45.0),
        "lon": np.full(n_samples, -30.0),
        "jd": np.linspace(100.0, 100.5, n_samples),
        "zi": np.full(n_samples, 600.0),
        "rain": np.full(n_samples, 0.0),
        "Ss": np.full(n_samples, 35.0),
    }


def _run(inputs, **kwargs):
    return coare36vn_zrf_et(
        **inputs,
        albedo=kwargs.pop("albedo", "constant"),
        return_albedo=True,
        auto_plot=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("input_name", "dependent_output"),
    [
        ("u", "friction_velocity"),
        ("zu", "friction_velocity"),
        ("t", "sensible_heat_flux"),
        ("zt", "sensible_heat_flux"),
        ("rh", "latent_heat_flux"),
        ("zq", "latent_heat_flux"),
        ("P", "sensible_heat_flux"),
        ("ts", "latent_heat_flux"),
        ("sw_dn", "net_solar_radiation"),
        ("lw_dn", "net_longwave_radiation"),
        ("rain", "rain_heat_flux"),
        ("Ss", "latent_heat_flux"),
    ],
)
def test_nan_input_is_local_and_marks_dependent_output(input_name, dependent_output):
    inputs = _build_inputs()
    baseline = _run(inputs)
    sample = 2
    inputs_with_nan = {name: values.copy() for name, values in inputs.items()}
    inputs_with_nan[input_name][sample] = np.nan

    output = _run(inputs_with_nan)
    unaffected_rows = np.arange(output.shape[0]) != sample
    np.testing.assert_array_equal(output[unaffected_rows], baseline[unaffected_rows])

    columns = list(get_variables_info(albedo_method="constant"))
    assert np.isnan(output[sample, columns.index(dependent_output)])
    assert np.all(
        ~np.isnan(output[:, columns.index("friction_velocity")])
        | np.isnan(output[:, columns.index("gustiness_velocity")])
    )


@pytest.mark.parametrize(
    ("input_name", "dependent_output"),
    [
        ("lat", "gustiness_velocity"),
        ("zi", "gustiness_velocity"),
        ("lon", None),
        ("jd", None),
    ],
)
def test_coordinate_and_boundary_layer_nan_inputs_are_local(input_name, dependent_output):
    inputs = _build_inputs()
    inputs["t"][:] = 10.0
    inputs["ts"][:] = 20.0
    baseline = _run(inputs)
    sample = 2
    inputs_with_nan = {name: values.copy() for name, values in inputs.items()}
    inputs_with_nan[input_name][sample] = np.nan

    output = _run(inputs_with_nan)
    unaffected_rows = np.arange(output.shape[0]) != sample
    np.testing.assert_array_equal(output[unaffected_rows], baseline[unaffected_rows])

    columns = list(get_variables_info(albedo_method="constant"))
    if dependent_output is None:
        np.testing.assert_array_equal(output[sample], baseline[sample])
    else:
        assert np.isnan(output[sample, columns.index(dependent_output)])


@pytest.mark.parametrize("input_name", ["sw_dn", "jd", "lat", "lon"])
def test_payne_albedo_returns_nan_for_non_finite_lookup_inputs(input_name):
    inputs = _build_inputs()
    inputs[input_name][2] = np.nan

    output = _run(inputs, albedo="Payne1972")
    columns = list(get_variables_info(albedo_method="Payne1972"))
    assert np.isnan(output[2, columns.index("albedo")])
    assert np.isnan(output[2, -3:]).all()


def test_payne_albedo_scalar_non_finite_input_returns_nan_diagnostics():
    albedo, transmission, solarmax, solar_altitude = albedo_vector(
        sw_dn=200.0,
        jd=np.nan,
        lon=-30.0,
        lat=45.0,
        eorw="E",
    )

    assert np.isnan([albedo, transmission, solarmax, solar_altitude]).all()


@pytest.mark.parametrize("albedo", ["constant", "Payne1972"])
def test_all_nan_inputs_do_not_raise_or_export_first_guess_values(albedo):
    inputs = _build_inputs()
    for values in inputs.values():
        values[:] = np.nan

    output = _run(inputs, albedo=albedo)
    columns = list(get_variables_info(albedo_method=albedo))
    expected_nan_columns = [idx for idx, name in enumerate(columns) if name != "albedo"]

    assert np.isnan(output[:, expected_nan_columns]).all()
    if albedo == "Payne1972":
        assert np.isnan(output[:, columns.index("albedo")]).all()
    else:
        np.testing.assert_array_equal(output[:, columns.index("albedo")], 0.055)


@pytest.mark.parametrize("coolskin", [False, True])
@pytest.mark.parametrize(("radiation_input", "affected_radiation", "independent_radiation"), [
    ("sw_dn", "net_solar_radiation", "net_longwave_radiation"),
    ("lw_dn", "net_longwave_radiation", "net_solar_radiation"),
])
def test_missing_radiation_keeps_turbulent_fluxes_missing(
    coolskin, radiation_input, affected_radiation, independent_radiation
):
    inputs = _build_inputs()
    baseline = _run(inputs, coolskin=coolskin)
    sample = 2
    inputs_with_nan = {name: values.copy() for name, values in inputs.items()}
    inputs_with_nan[radiation_input][sample] = np.nan

    output = _run(inputs_with_nan, coolskin=coolskin)
    columns = list(get_variables_info(albedo_method="constant"))
    for name in [
        "friction_velocity",
        "wind_stress",
        "sensible_heat_flux",
        "latent_heat_flux",
        affected_radiation,
    ]:
        assert np.isnan(output[sample, columns.index(name)])
    if independent_radiation == "net_longwave_radiation" and coolskin:
        assert np.isnan(output[sample, columns.index(independent_radiation)])
    else:
        assert np.isfinite(output[sample, columns.index(independent_radiation)])

    unaffected_rows = np.arange(output.shape[0]) != sample
    np.testing.assert_array_equal(output[unaffected_rows], baseline[unaffected_rows])
