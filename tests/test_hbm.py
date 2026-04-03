import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kinamax.hbm import (
    SampledSignal,
    add_fourier_coeffs,
    add_sampled_signals,
    coeffs_derivative,
    coeffs_pow,
    coeffs_to_complex,
    coeffs_to_table,
    coeffs_to_time_signal,
    complex_to_coeffs,
    cosine_forcing,
    format_fourier_coeffs,
    fourier_zeros,
    phased_forcing,
    print_fourier_coeffs,
    scale_fourier_coeffs,
    scale_sampled_signal,
    sine_forcing,
    sub_fourier_coeffs,
    sub_sampled_signals,
    sum_fourier_coeffs,
    sum_sampled_signals,
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


def test_format_fourier_coeffs_renders_cosine_sine_table():
    coeffs = np.array([1.5, 2.0, -1.0, 0.5, -0.75])

    rendered = format_fourier_coeffs(coeffs)

    assert "FourierCoeffs" in rendered
    assert "basis" in rendered
    assert "cosine" in rendered
    assert "sine" in rendered
    assert "0" in rendered
    assert "1" in rendered
    assert "2" in rendered


def test_print_fourier_coeffs_writes_table_to_stdout(capsys: pytest.CaptureFixture[str]):
    coeffs = np.array([1.5, 2.0, -1.0, 0.5, -0.75])

    print_fourier_coeffs(coeffs)

    out = capsys.readouterr().out
    assert "FourierCoeffs" in out
    assert "cosine" in out
    assert "sine" in out


def test_sampled_signal_time_properties_are_derived_from_frequency():
    signal = SampledSignal(values=np.array([0.0, 1.0, 0.0, -1.0]), frequency=5.0)

    np.testing.assert_allclose(np.asarray(signal.time_step), 0.05)
    np.testing.assert_allclose(np.asarray(signal.time_grid), np.array([0.0, 0.05, 0.1, 0.15]))


def test_sampled_signal_linear_algebra_preserves_frequency():
    a = SampledSignal(values=np.array([0.0, 1.0, 0.0, -1.0]), frequency=5.0)
    b = SampledSignal(values=np.array([1.0, -1.0, 2.0, 0.5]), frequency=5.0)

    added = add_sampled_signals(a, b)
    subtracted = sub_sampled_signals(a, b)
    scaled = scale_sampled_signal(a, 2.0)
    total = sum_sampled_signals(a, b, -a)

    np.testing.assert_allclose(np.asarray(added.frequency), 5.0)
    np.testing.assert_allclose(np.asarray(added.values), np.array([1.0, 0.0, 2.0, -0.5]))
    np.testing.assert_allclose(np.asarray(subtracted.values), np.array([-1.0, 2.0, -2.0, -1.5]))
    np.testing.assert_allclose(np.asarray(scaled.values), np.array([0.0, 2.0, 0.0, -2.0]))
    np.testing.assert_allclose(np.asarray(total.values), np.array([1.0, -1.0, 2.0, 0.5]))


def test_sampled_signal_algebra_is_jittable():
    @jax.jit
    def combine(a: SampledSignal, b: SampledSignal) -> SampledSignal:
        return a + 2.0 * b - a / 2.0 + sum_sampled_signals(-b, a)

    a = SampledSignal(
        values=jnp.array([0.0, 1.0, 0.0, -1.0]),
        frequency=jnp.float32(5.0),
    )
    b = SampledSignal(
        values=jnp.array([1.0, -1.0, 2.0, 0.5]),
        frequency=jnp.float32(5.0),
    )

    out = combine(a, b)

    assert isinstance(out, SampledSignal)
    np.testing.assert_allclose(np.asarray(out.frequency), 5.0)
    np.testing.assert_allclose(np.asarray(out.values), np.array([1.0, 0.5, 2.0, -1.0]))


def test_fourier_coeffs_table_and_derivative_follow_real_cosine_sine_layout():
    coeffs = np.array([1.5, 2.0, -1.0, 0.5, -0.75])
    wd = 3.0

    first = coeffs_derivative(coeffs, wd=wd, order=1)
    table = coeffs_to_table(coeffs)

    np.testing.assert_allclose(
        np.asarray(first),
        np.array([0.0, 1.5, -4.5, -6.0, 6.0]),
    )
    np.testing.assert_allclose(
        np.asarray(table),
        np.array([[1.5, 2.0, -1.0], [0.0, 0.5, -0.75]]),
    )


def test_fourier_coeffs_linear_algebra_uses_arrays():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-1.0, 0.5, 4.0])

    added = add_fourier_coeffs(a, b)
    subtracted = sub_fourier_coeffs(a, b)
    scaled = scale_fourier_coeffs(a, 2.0)
    total = sum_fourier_coeffs(a, b, -a)

    np.testing.assert_allclose(np.asarray(added), np.array([0.0, 2.5, 7.0]))
    np.testing.assert_allclose(np.asarray(subtracted), np.array([2.0, 1.5, -1.0]))
    np.testing.assert_allclose(np.asarray(scaled), np.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(np.asarray(total), np.array([-1.0, 0.5, 4.0]))


def test_fourier_convenience_constructors_build_expected_harmonics():
    zeros = fourier_zeros(order=2)
    cosine = cosine_forcing(amplitude=3.0, harmonic=2, order=2)
    sine = sine_forcing(amplitude=-4.0, harmonic=1, order=2)
    phased = phased_forcing(
        amplitude=2.0,
        phase=jnp.pi / 2.0,
        harmonic=1,
        order=2,
    )

    np.testing.assert_allclose(np.asarray(zeros), np.zeros(5))
    np.testing.assert_allclose(np.asarray(cosine), np.array([0.0, 0.0, 3.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.asarray(sine), np.array([0.0, 0.0, 0.0, -4.0, 0.0]))
    np.testing.assert_allclose(np.asarray(phased), np.array([0.0, 0.0, 0.0, -2.0, 0.0]), atol=1e-7)


def test_fourier_coeffs_algebra_is_jittable():
    @jax.jit
    def build_residual(a: jax.Array, b: jax.Array) -> jax.Array:
        return 2.0 * a - b / 2.0 + sum_fourier_coeffs(a, -b)

    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([0.5, -1.0, 4.0])

    residual = build_residual(a, b)

    np.testing.assert_allclose(
        np.asarray(residual),
        np.array([2.25, 7.5, 3.0]),
    )


def test_fourier_convenience_constructors_are_jittable_with_static_order():
    @jax.jit
    def forcing(amplitude: jax.Array) -> jax.Array:
        return cosine_forcing(
            amplitude=amplitude,
            harmonic=1,
            order=2,
        ) + sine_forcing(
            amplitude=2.0 * amplitude,
            harmonic=2,
            order=2,
        )

    out = forcing(jnp.float32(3.0))

    np.testing.assert_allclose(np.asarray(out), np.array([0.0, 3.0, 0.0, 0.0, 6.0]))


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


def test_sampled_signal_round_trip_keeps_frequency_metadata():
    coeffs = np.array([0.25, 1.0, -0.5, 0.75, 0.125])

    signal = coeffs_to_time_signal(coeffs, frequency=5.0, oversample=3)
    recovered = time_to_coeffs(signal, downsample=3)

    assert isinstance(signal, SampledSignal)
    np.testing.assert_allclose(np.asarray(signal.frequency), 5.0)
    np.testing.assert_allclose(np.asarray(signal.time_step), 1.0 / (12 * 5.0))
    np.testing.assert_allclose(np.asarray(recovered), coeffs, atol=1e-6)


def test_coeffs_to_time_signal_accepts_traced_frequency():
    coeffs = jnp.array([0.0, 1.0, 0.0])

    @jax.jit
    def reconstruct(frequency: jax.Array) -> SampledSignal:
        return coeffs_to_time_signal(coeffs, frequency=frequency, oversample=2)

    signal = reconstruct(jnp.float32(5.0))

    assert isinstance(signal, SampledSignal)
    np.testing.assert_allclose(np.asarray(signal.frequency), 5.0)
    np.testing.assert_allclose(np.asarray(signal.values), np.array([1.0, 0.0, -1.0, 0.0]), atol=1e-6)


def test_coeffs_pow_reprojects_powered_signal():
    coeffs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    squared = coeffs_pow(coeffs, p=2, oversample=2)

    np.testing.assert_allclose(
        np.asarray(squared),
        np.array([0.5, 0.0, 0.5, 0.0, 0.0]),
        atol=1e-6,
    )


def test_coeffs_to_table_exposes_cosine_and_sine_rows():
    coeffs = np.array([1.5, 2.0, -1.0, 0.5, -0.75])

    table = coeffs_to_table(coeffs)

    np.testing.assert_allclose(
        np.asarray(table),
        np.array([[1.5, 2.0, -1.0], [0.0, 0.5, -0.75]]),
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
        (coeffs_to_time_signal, {"X": np.array([1.0]), "frequency": 0.0}),
        (time_to_coeffs, {"x": np.ones(5), "downsample": 2}),
        (fourier_zeros, {"order": -1}),
        (cosine_forcing, {"amplitude": 1.0, "harmonic": 2, "order": 1}),
        (sine_forcing, {"amplitude": 1.0, "harmonic": 0, "order": 1}),
    ],
)
def test_invalid_inputs_raise_value_error(func, kwargs):
    with pytest.raises(ValueError):
        func(**kwargs)
