"""Helpers for Harmonic Balance Method (HBM) coefficient manipulations.

The real Fourier coefficient layout used across this module is:

    X = [a0, a1, ..., aN, b1, ..., bN]

which represents the real periodic signal

    x(t) = a0 + sum_n (a_n cos(n * wd * t) + b_n sin(n * wd * t))
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .core import namedtuple_repr

__all__ = [
    "FourierCoeffs",
    "SampledSignal",
    "coeffs_to_complex",
    "complex_to_coeffs",
    "coeffs_derivative",
    "coeffs_to_table",
    "time_grid",
    "coeffs_to_time_signal",
    "time_to_coeffs",
    "coeffs_pow",
]

class FourierCoeffs(NamedTuple):
    """Real Fourier coefficients paired with the fundamental frequency in Hz.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = FourierCoeffs(
    ...     values=jnp.array([0.0, 1.0, 0.5]),
    ...     frequency=jnp.float32(5.0),
    ... )
    >>> print(coeffs)
    FourierCoeffs
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (3,)   ┆ float32 ┆ [0. , 1. , 0.5]...
    ...
    """

    values: jax.Array
    frequency: float | jax.Array

    def __repr__(self) -> str:
        return namedtuple_repr(
            "FourierCoeffs",
            {
                "frequency": self.frequency,
                "values": self.values,
            },
        )


class SampledSignal(NamedTuple):
    """Uniformly sampled signal over one period, tagged by its frequency in Hz.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> signal = SampledSignal(
    ...     values=jnp.array([0.0, 1.0, 0.0, -1.0]),
    ...     frequency=jnp.float32(5.0),
    ... )
    >>> print(signal)
    SampledSignal
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (4,)   ┆ float32 ┆ [ 0.,  1.,  0., -1.]...
    ...
    >>> print(signal.time_step)
    0.05
    >>> print(signal.time_grid)
    [0.   0.05 0.1  0.15]
    """

    values: jax.Array
    frequency: float | jax.Array

    @property
    def time_step(self) -> jax.Array:
        """Return the uniform time step over one period."""
        values = jnp.asarray(self.values)
        sample_count = values.shape[-1] if values.ndim else 1
        return 1.0 / (jnp.asarray(self.frequency) * sample_count)

    @property
    def time_grid(self) -> jax.Array:
        """Return the evenly spaced time grid over one period."""
        values = jnp.asarray(self.values)
        sample_count = values.shape[-1] if values.ndim else 1
        return jnp.arange(sample_count) * self.time_step[..., None]

    def __repr__(self) -> str:
        return namedtuple_repr(
            "SampledSignal",
            {
                "frequency": self.frequency,
                "values": self.values,
            },
        )


def _harmonic_order_count(coeffs: jax.Array) -> int:
    """Return the highest retained harmonic order for a stacked real vector."""
    size = coeffs.size
    if size < 1 or size % 2 == 0:
        raise ValueError("HBM coefficient vectors must have odd length 2*N + 1.")
    return (size - 1) // 2


def _require_positive_factor(name: str, value: int) -> None:
    """Validate integer sampling factors used for FFT domain conversions."""
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")


def _coeff_values(X: ArrayLike | FourierCoeffs) -> jax.Array:
    """Return the raw coefficient vector from an array or a FourierCoeffs object."""
    if isinstance(X, FourierCoeffs):
        return jnp.asarray(X.values)
    return jnp.asarray(X)


def _sample_values(x: ArrayLike | SampledSignal) -> jax.Array:
    """Return the raw sampled signal from an array or a SampledSignal object."""
    if isinstance(x, SampledSignal):
        return jnp.asarray(x.values)
    return jnp.asarray(x)


def _frequency_to_wd(frequency: ArrayLike) -> jax.Array:
    """Convert a frequency in Hz to an angular frequency in rad/s."""
    return 2.0 * jnp.pi * jnp.asarray(frequency)


def coeffs_to_complex(X: ArrayLike | FourierCoeffs) -> jax.Array:
    """Convert stacked real Fourier coefficients to positive-frequency phasors."""
    coeffs = _coeff_values(X)
    N = _harmonic_order_count(coeffs)
    a0 = coeffs[0]
    a = coeffs[1 : N + 1]
    b = coeffs[N + 1 :]
    return jnp.concatenate([a0[None], a - 1j * b])


def complex_to_coeffs(
    C: ArrayLike, frequency: ArrayLike | None = None
) -> jax.Array | FourierCoeffs:
    """Convert positive-frequency phasors back to stacked real coefficients."""
    complex_coeffs = jnp.asarray(C)
    if complex_coeffs.size < 1:
        raise ValueError("At least one complex coefficient is required.")
    a0 = jnp.real(complex_coeffs[0])
    a = jnp.real(complex_coeffs[1:])
    b = -jnp.imag(complex_coeffs[1:])
    real_dtype = jnp.real(complex_coeffs).dtype
    values = jnp.concatenate([a0[None], a, b]).astype(real_dtype)
    if frequency is None:
        return values
    return FourierCoeffs(values=values, frequency=jnp.asarray(frequency))


def coeffs_derivative(
    X: ArrayLike | FourierCoeffs, wd: ArrayLike | None = None, order: int = 1
) -> jax.Array | FourierCoeffs:
    """Differentiate a coefficient vector with respect to time.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = FourierCoeffs(values=jnp.array([0.0, 1.0, 0.0]), frequency=5.0)
    >>> velocity = coeffs_derivative(coeffs, order=1)
    >>> print(velocity)
    FourierCoeffs
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (3,)   ┆ float32 ┆ [  0.      ,   0.      , -31.4...
    ...
    """
    if order < 0:
        raise ValueError("order must be a non-negative integer.")
    if isinstance(X, FourierCoeffs):
        if wd is not None:
            raise ValueError("Pass either FourierCoeffs or raw coeffs with wd, not both.")
        wd = _frequency_to_wd(X.frequency)
    elif wd is None:
        raise ValueError("wd must be provided when differentiating raw coefficient arrays.")
    coeffs = coeffs_to_complex(X)
    harmonic_ids = jnp.arange(coeffs.size)
    dcoeffs = (1j * harmonic_ids * jnp.asarray(wd)) ** order * coeffs
    if isinstance(X, FourierCoeffs):
        return complex_to_coeffs(dcoeffs, frequency=X.frequency)
    return complex_to_coeffs(dcoeffs)


def coeffs_to_table(X: ArrayLike | FourierCoeffs) -> jax.Array:
    """Return a JAX table with stacked real and imaginary phasor parts."""
    complex_coeffs = coeffs_to_complex(X)
    return jnp.stack([jnp.real(complex_coeffs), jnp.imag(complex_coeffs)], axis=0)


def time_grid(wd: ArrayLike, n: int, oversample: int = 1) -> jax.Array:
    """Build an evenly spaced grid over one forcing period."""
    _require_positive_factor("n", n)
    _require_positive_factor("oversample", oversample)
    period = 2.0 * jnp.pi / jnp.asarray(wd)
    sample_count = n * oversample
    return jnp.linspace(0.0, period, sample_count, endpoint=False)


def coeffs_to_time_signal(
    X: ArrayLike | FourierCoeffs, oversample: int = 1
) -> jax.Array | SampledSignal:
    """Synthesize a real time signal from stacked real Fourier coefficients.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> coeffs = FourierCoeffs(values=jnp.array([0.0, 1.0, 0.0]), frequency=5.0)
    >>> signal = coeffs_to_time_signal(coeffs, oversample=4)
    >>> print(signal)
    SampledSignal
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (8,)   ┆ float32 ┆ [1.        , 0.70710677, 0.   ...
    ...
    """
    _require_positive_factor("oversample", oversample)
    complex_coeffs = coeffs_to_complex(X)
    harmonic_count = complex_coeffs.size - 1
    sample_count = max(2 * harmonic_count * oversample, 1)

    # irfft expects positive-frequency amplitudes scaled by the FFT convention.
    fft_coeffs = complex_coeffs * (sample_count / 2.0)
    fft_coeffs = fft_coeffs.at[0].set(complex_coeffs[0] * sample_count)
    values = jnp.fft.irfft(fft_coeffs, n=sample_count).real
    if isinstance(X, FourierCoeffs):
        frequency = jnp.asarray(X.frequency)
        if jnp.any(frequency == 0):
            raise ValueError(
                "FourierCoeffs.frequency must be non-zero to build a sampled signal."
            )
        return SampledSignal(values=values, frequency=frequency)
    return values


def time_to_coeffs(
    x: ArrayLike | SampledSignal, downsample: int = 1
) -> jax.Array | FourierCoeffs:
    """Project a sampled real time signal back onto the HBM basis.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> signal = SampledSignal(
    ...     values=jnp.cos(2.0 * jnp.pi * jnp.arange(16) / 16),
    ...     frequency=5.0,
    ... )
    >>> coeffs = time_to_coeffs(signal, downsample=4)
    >>> print(coeffs)
    FourierCoeffs
    ...
    │ frequency ┆ scalar ┆ float32 ┆ 5.0...
    │ values    ┆ (5,)   ┆ float32 ┆ [ 4.0854182e-08,  1.0000000e+0...
    ...
    """
    _require_positive_factor("downsample", downsample)
    signal = _sample_values(x)
    sample_count = signal.size
    if sample_count != 1 and sample_count % (2 * downsample) != 0:
        raise ValueError(
            "Signal length must be 1 or divisible by 2 * downsample."
        )

    fft_coeffs = jnp.fft.rfft(signal) * 2.0 / sample_count
    fft_coeffs = fft_coeffs.at[0].set(fft_coeffs[0] * 0.5)
    kept_coeffs = fft_coeffs[: sample_count // (2 * downsample) + 1]
    if isinstance(x, SampledSignal):
        return complex_to_coeffs(kept_coeffs, frequency=jnp.asarray(x.frequency))
    return complex_to_coeffs(kept_coeffs)


def coeffs_pow(
    X: ArrayLike | FourierCoeffs, p: ArrayLike, oversample: int = 1
) -> jax.Array | FourierCoeffs:
    """Raise a periodic signal to a power and re-project it onto the HBM basis."""
    signal = coeffs_to_time_signal(X, oversample=oversample)
    if isinstance(signal, SampledSignal):
        return time_to_coeffs(
            SampledSignal(values=signal.values ** p, frequency=signal.frequency),
            downsample=oversample,
        )
    return time_to_coeffs(signal ** p, downsample=oversample)
