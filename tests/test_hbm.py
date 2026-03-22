import numpy as np
import pytest

from kinamax.hbm import (
    FourierCoeffs,
    SampledSignal,
    coeffs_derivative,
    coeffs_pow,
    coeffs_to_complex,
    coeffs_to_table,
    coeffs_to_time_signal,
    complex_to_coeffs,
    time_grid,
    time_to_coeffs,
)


def test_complex_conversion_round_trip():
    coeffs = np.array([1.5, 2.0, -1.0, 0.5, -0.75])

    complex_coeffs = coeffs_to_complex(coeffs)

    np.testing.assert_allclose(
        np.asarray(complex_coeffs),
        np.array([1.5 + 0.0j, 2.0 - 0.5j, -1.0 + 0.75j]),
    )
    np.testing.assert_allclose(np.asarray(complex_to_coeffs(complex_coeffs)), coeffs)


def test_hbm_container_repr_handles_vectorized_fields():
    coeffs = FourierCoeffs(
        values=np.arange(12.0).reshape(3, 4),
        frequency=np.array([5.0, 7.5, 10.0]),
    )
    signal = SampledSignal(
        values=np.arange(16.0).reshape(2, 8),
        frequency=np.array([12.5, 6.25]),
    )

    coeffs_repr = repr(coeffs)
    signal_repr = repr(signal)

    assert "FourierCoeffs" in coeffs_repr
    assert "frequency" in coeffs_repr
    assert "(3, 4)" in coeffs_repr
    assert "SampledSignal" in signal_repr
    assert "frequency" in signal_repr
    assert "(2, 8)" in signal_repr


def test_sampled_signal_time_properties_are_derived_from_frequency():
    signal = SampledSignal(values=np.array([0.0, 1.0, 0.0, -1.0]), frequency=5.0)

    np.testing.assert_allclose(np.asarray(signal.time_step), 0.05)
    np.testing.assert_allclose(np.asarray(signal.time_grid), np.array([0.0, 0.05, 0.1, 0.15]))


def test_fourier_coeffs_container_preserves_frequency_through_operations():
    coeffs = FourierCoeffs(
        values=np.array([1.5, 2.0, -1.0, 0.5, -0.75]),
        frequency=3.0 / (2.0 * np.pi),
    )

    first = coeffs_derivative(coeffs, order=1)
    table = coeffs_to_table(coeffs)

    assert isinstance(first, FourierCoeffs)
    np.testing.assert_allclose(np.asarray(first.frequency), coeffs.frequency)
    np.testing.assert_allclose(
        np.asarray(first.values),
        np.array([0.0, 1.5, -4.5, -6.0, 6.0]),
    )
    np.testing.assert_allclose(
        np.asarray(table),
        np.array([[1.5, 2.0, -1.0], [0.0, -0.5, 0.75]]),
    )


def test_first_and_second_derivatives_follow_trigonometric_rules():
    coeffs = np.array([1.5, 2.0, -1.0, 0.5, -0.75])
    wd = 3.0

    first = coeffs_derivative(coeffs, wd=wd, order=1)
    second = coeffs_derivative(coeffs, wd=wd, order=2)

    np.testing.assert_allclose(
        np.asarray(first),
        np.array([0.0, 1.5, -4.5, -6.0, 6.0]),
    )
    np.testing.assert_allclose(
        np.asarray(second),
        np.array([0.0, -18.0, 36.0, -4.5, 27.0]),
    )


def test_time_signal_round_trip_with_oversampling():
    coeffs = np.array([0.25, 1.0, -0.5, 0.75, 0.125])

    signal = coeffs_to_time_signal(coeffs, oversample=3)
    recovered = time_to_coeffs(signal, downsample=3)

    np.testing.assert_allclose(np.asarray(recovered), coeffs, atol=1e-6)


def test_container_round_trip_through_sampled_signal_preserves_frequency():
    coeffs = FourierCoeffs(
        values=np.array([0.25, 1.0, -0.5, 0.75, 0.125]),
        frequency=5.0,
    )

    signal = coeffs_to_time_signal(coeffs, oversample=3)
    recovered = time_to_coeffs(signal, downsample=3)

    assert isinstance(signal, SampledSignal)
    assert isinstance(recovered, FourierCoeffs)
    np.testing.assert_allclose(np.asarray(signal.frequency), 5.0)
    np.testing.assert_allclose(np.asarray(signal.time_step), 1.0 / (12 * 5.0))
    np.testing.assert_allclose(np.asarray(recovered.frequency), 5.0)
    np.testing.assert_allclose(np.asarray(recovered.values), coeffs.values, atol=1e-6)


def test_coeffs_pow_reprojects_powered_signal():
    coeffs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    squared = coeffs_pow(coeffs, p=2, oversample=2)

    np.testing.assert_allclose(
        np.asarray(squared),
        np.array([0.5, 0.0, 0.5, 0.0, 0.0]),
        atol=1e-6,
    )


def test_coeffs_pow_preserves_fourier_coeffs_frequency():
    coeffs = FourierCoeffs(values=np.array([0.0, 1.0, 0.0, 0.0, 0.0]), frequency=7.0)

    squared = coeffs_pow(coeffs, p=2, oversample=2)

    assert isinstance(squared, FourierCoeffs)
    np.testing.assert_allclose(np.asarray(squared.frequency), 7.0)
    np.testing.assert_allclose(
        np.asarray(squared.values),
        np.array([0.5, 0.0, 0.5, 0.0, 0.0]),
        atol=1e-6,
    )


def test_coeffs_to_table_exposes_real_and_imaginary_parts():
    coeffs = np.array([1.5, 2.0, -1.0, 0.5, -0.75])

    table = coeffs_to_table(coeffs)

    np.testing.assert_allclose(
        np.asarray(table),
        np.array([[1.5, 2.0, -1.0], [0.0, -0.5, 0.75]]),
    )


def test_time_grid_spans_one_period_without_endpoint():
    wd = 2.0 * np.pi * 5.0

    grid = time_grid(wd, n=4, oversample=2)

    expected = np.arange(8) / 40.0
    np.testing.assert_allclose(np.asarray(grid), expected)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (coeffs_to_complex, {"X": np.array([1.0, 2.0])}),
        (coeffs_derivative, {"X": np.array([1.0]), "wd": 1.0, "order": -1}),
        (time_grid, {"wd": 1.0, "n": 0}),
        (coeffs_to_time_signal, {"X": np.array([1.0]), "oversample": 0}),
        (time_to_coeffs, {"x": np.ones(5), "downsample": 2}),
        (
            coeffs_derivative,
            {
                "X": FourierCoeffs(values=np.array([1.0]), frequency=1.0),
                "wd": 1.0,
            },
        ),
        (
            coeffs_to_time_signal,
            {"X": FourierCoeffs(values=np.array([1.0]), frequency=0.0)},
        ),
    ],
)
def test_invalid_inputs_raise_value_error(func, kwargs):
    with pytest.raises(ValueError):
        func(**kwargs)
