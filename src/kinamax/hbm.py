"""Helpers for Harmonic Balance Method (HBM) coefficient manipulations.

The real Fourier coefficient layout used across this module is:

    X = [a0, a1, ..., aN, b1, ..., bN]

which represents the real periodic signal

    x(t) = a0 + sum_n (a_n cos(n * wd * t) + b_n sin(n * wd * t))
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "coeffs_to_complex",
    "complex_to_coeffs",
    "coeffs_derivative",
    "coeffs_to_table",
    "time_grid",
    "coeffs_to_time_signal",
    "time_to_coeffs",
    "coeffs_pow",
]


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


def coeffs_to_complex(X: ArrayLike) -> jax.Array:
    """Convert stacked real Fourier coefficients to positive-frequency phasors."""
    coeffs = jnp.asarray(X)
    N = _harmonic_order_count(coeffs)
    a0 = coeffs[0]
    a = coeffs[1 : N + 1]
    b = coeffs[N + 1 :]
    return jnp.concatenate([a0[None], a - 1j * b])


def complex_to_coeffs(C: ArrayLike) -> jax.Array:
    """Convert positive-frequency phasors back to stacked real coefficients."""
    complex_coeffs = jnp.asarray(C)
    if complex_coeffs.size < 1:
        raise ValueError("At least one complex coefficient is required.")
    a0 = jnp.real(complex_coeffs[0])
    a = jnp.real(complex_coeffs[1:])
    b = -jnp.imag(complex_coeffs[1:])
    real_dtype = jnp.real(complex_coeffs).dtype
    return jnp.concatenate([a0[None], a, b]).astype(real_dtype)


def coeffs_derivative(X: ArrayLike, wd: ArrayLike, order: int = 1) -> jax.Array:
    """Differentiate a coefficient vector with respect to time."""
    if order < 0:
        raise ValueError("order must be a non-negative integer.")
    coeffs = coeffs_to_complex(X)
    harmonic_ids = jnp.arange(coeffs.size)
    dcoeffs = (1j * harmonic_ids * jnp.asarray(wd)) ** order * coeffs
    return complex_to_coeffs(dcoeffs)


def coeffs_to_table(X: ArrayLike) -> jax.Array:
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


def coeffs_to_time_signal(X: ArrayLike, oversample: int = 1) -> jax.Array:
    """Synthesize a real time signal from stacked real Fourier coefficients."""
    _require_positive_factor("oversample", oversample)
    complex_coeffs = coeffs_to_complex(X)
    harmonic_count = complex_coeffs.size - 1
    sample_count = max(2 * harmonic_count * oversample, 1)

    # irfft expects positive-frequency amplitudes scaled by the FFT convention.
    fft_coeffs = complex_coeffs * (sample_count / 2.0)
    fft_coeffs = fft_coeffs.at[0].set(complex_coeffs[0] * sample_count)
    return jnp.fft.irfft(fft_coeffs, n=sample_count).real


def time_to_coeffs(x: ArrayLike, downsample: int = 1) -> jax.Array:
    """Project a sampled real time signal back onto the HBM basis."""
    _require_positive_factor("downsample", downsample)
    signal = jnp.asarray(x)
    sample_count = signal.size
    if sample_count != 1 and sample_count % (2 * downsample) != 0:
        raise ValueError(
            "Signal length must be 1 or divisible by 2 * downsample."
        )

    fft_coeffs = jnp.fft.rfft(signal) * 2.0 / sample_count
    fft_coeffs = fft_coeffs.at[0].set(fft_coeffs[0] * 0.5)
    kept_coeffs = fft_coeffs[: sample_count // (2 * downsample) + 1]
    return complex_to_coeffs(kept_coeffs)


def coeffs_pow(X: ArrayLike, p: ArrayLike, oversample: int = 1) -> jax.Array:
    """Raise a periodic signal to a power and re-project it onto the HBM basis."""
    signal = coeffs_to_time_signal(X, oversample=oversample)
    return time_to_coeffs(signal ** p, downsample=oversample)
